# losses/yolo_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
    from ultralytics.utils.ops import xyxy2xywhn, non_max_suppression
    from ultralytics.nn.tasks import v8DetectionLoss
    from ultralytics.utils.loss import BboxLoss
except ImportError:
    raise ImportError("Please install 'ultralytics': pip install ultralytics")

# 匯入我們自定義的 CB-Focal Loss
from losses.cb_focal_loss import ClassBalancedFocalLoss

class YOLOLongTailLoss(v8DetectionLoss):
    """
    繼承 YOLOv8 Loss，並客製化 Class Loss
    
    Milestone 1 (use_cb_loss=False):
      - 完全等同於 v8DetectionLoss
    
    Milestone 2 (use_cb_loss=True):
      - 將 self.bce (BCEWithLogitsLoss) 替換為 ClassBalancedFocalLoss
    """
    
    def __init__(self, 
                 model, # YOLOv8 model (DetectionModel)
                 use_cb_loss: bool = False,
                 samples_per_cls: Optional[List[int]] = None,
                 cb_beta: float = 0.9999,
                 focal_gamma: float = 2.0
                 ):
        
        # 呼叫父類 (v8DetectionLoss) 的 init
        # 它會自動設定 reg_max, dfl_loss, iou_loss (BboxLoss)
        # 並且會設定 self.bce = nn.BCEWithLogitsLoss(reduction='none')
        super().__init__(model)
        
        self.use_cb_loss = use_cb_loss
        
        # --- (Milestone 2+) 替換 Loss ---
        if self.use_cb_loss:
            if samples_per_cls is None:
                raise ValueError("use_cb_loss=True, but samples_per_cls is not provided.")
            
            print(f"[Loss] Using ClassBalancedFocalLoss (beta={cb_beta}, gamma={focal_gamma})")
            
            # 替換掉 v8DetectionLoss 預設的 BCE loss
            # 我們自定義的 CB-Focal Loss 需要是 BCE-style (logits, targets)
            self.bce = ClassBalancedFocalLoss(
                samples_per_cls=samples_per_cls,
                beta=cb_beta,
                gamma=focal_gamma
            )
        else:
            print("[Loss] Using standard BCEWithLogitsLoss for classification.")
            # 保持父類預設的 self.bce
    
    def decode_and_nms(self, preds, conf_thres, iou_thres):
        """
        Helper function: 將模型的 raw output 解碼並執行 NMS
        (此功能在 v8DetectionLoss 中沒有，我們自己加，方便 evaluate)
        
        Args:
            preds: (reg_preds, cls_preds) (來自 YOLODetector)
            conf_thres: (float)
            iou_thres: (float)
        
        Returns:
            List[Tensor[N, 6]] (x1, y1, x2, y2, conf, cls_id)
        """
        
        # reg_preds: List[Tensor[B, 64, H, W], ...]
        # cls_preds: List[Tensor[B, NC, H, W], ...]
        
        # (來自 v8DetectionLoss.preprocess)
        # 1. 將 (reg, cls) 合併
        x = [] # list of [B, 64+NC, H, W]
        for i in range(len(preds[0])):
            x.append(torch.cat((preds[0][i], preds[1][i]), 1))
        
        # 2. 解碼 (來自 v8DetectionLoss.__call__)
        box, cls = self.pred_decode(x)
        # box: [B, Num_Anchors, 4] (xyxy)
        # cls: [B, Num_Anchors, NC] (sigmoid)
        
        # 3. 執行 NMS (來自 ultralytics.utils.ops)
        # NMS 需要 (boxes, scores)
        # scores = cls.max(-1, keepdim=True)[0]
        # boxes = (boxes * scores) # C_i = P(obj) * P(cls_i) - YOLOv5/v8
        # ... 不對，v8 的 cls output 已經是 P(cls|obj) * P(obj)
        
        # ultralytics v8 NMS:
        # box (xyxy) [B, N, 4]
        # cls (sigmoid) [B, N, NC]
        
        B, N, NC = cls.shape
        
        # 建立 [B, N, 4 + NC]
        pred_nms = torch.cat((box, cls), dim=2)

        # 執行 NMS
        # conf_thres 是 class confidence
        # iou_thres 是 IoU
        # multi_label=True (因為一張圖可能有多個同類物體)
        results = non_max_suppression(
            pred_nms,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=True
        )
        # results: List[Tensor[Num_Dets, 6]] (x1, y1, x2, y2, conf, cls_id)
        return results
        
    def pred_decode(self, x):
        """
        Helper: 將 head output 解碼為 (box, cls)
        (來自 v8DetectionLoss.__call__)
        """
        device = x[0].device
        # 確保 anchors/strides 在正確的 device
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.anchors = self.anchors.to(device)
        self.strides = self.strides.to(device)
        
        x_cat = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        # x_cat: [B, 64(reg)+NC(cls), Num_Anchors_Total]
        
        # 解碼
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # 1. Box decode
        a, b = box.chunk(2, 1)
        a = self.project(a) # [B, 32, N]
        b = self.project(b) # [B, 32, N]
        # a/b shape: [B, 2 * reg_max, N]
        
        box = torch.cat((a, b), 1)
        box = box.transpose(1, 2).view(-1, 4, self.reg_max)
        box = F.softmax(box, dim=-1)
        # box: [B*N, 4, 16] (reg_max=16)
        
        # DFL
        box = box @ self.dfl
        box = dist2bbox(box, self.anchors.unsqueeze(0), xywh=False, dim=1)
        # box: [B, N, 4] (xyxy)
        
        # 2. Class decode
        cls = cls.transpose(1, 2) # [B, N, NC]
        cls = cls.sigmoid()
        
        return box, cls


    # __call__ 方法會自動繼承
    # 它會使用 self.bce (我們在 init 中替換的)
    # 所以 Milestone 2 會自動啟用 CB-Focal Loss