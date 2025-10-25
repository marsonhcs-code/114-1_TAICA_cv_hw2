# models/yolo_detector.py
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any

try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import ops
except ImportError:
    raise ImportError("Please install 'ultralytics': pip install ultralytics")

# 匯入我們自定義的 Loss
from losses.yolo_loss import YOLOLongTailLoss

class YOLODetector(nn.Module):
    """
    YOLO 偵測器 Wrapper
    
    職責：
    1. 載入 YOLOv8 'from scratch' (by .yaml)
    2. 綁定自定義的 Loss Function
    3. 整合 train forward (回傳 loss dict) 和 eval forward (回傳 NMS 預測)
    """
    
    def __init__(self, 
                 model_name: str = "yolov8n.yaml",
                 num_classes: int = 4,
                 # Loss 參數
                 use_cb_loss: bool = False,
                 samples_per_cls: Optional[List[int]] = None,
                 cb_beta: float = 0.9999,
                 focal_gamma: float = 2.0,
                 # Eval 參數
                 use_logit_adjustment: bool = False,
                 tau: float = 1.0,
                 conf_threshold: float = 0.001,
                 iou_threshold: float = 0.6,
                 nms_iou: float = 0.7
                 ):
        super().__init__()
        
        # --- 1. 載入模型架構 (From Scratch) ---
        # DetectionModel 是 ultralytics 中的實際 nn.Module
        self.model = DetectionModel(cfg=model_name, nc=num_classes)
        # 獲取 stride (用於 loss)
        self.stride = self.model.stride
        self.num_classes = num_classes
        
        # --- 2. 建立 Loss Function ---
        self.loss_fn = YOLOLongTailLoss(
            model=self.model, # Loss 需要 model (anchors, stride)
            use_cb_loss=use_cb_loss,
            samples_per_cls=samples_per_cls,
            cb_beta=cb_beta,
            focal_gamma=focal_gamma
        )
        
        # --- 3. Eval 參數 ---
        self.use_logit_adjustment = use_logit_adjustment
        self.tau = tau
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.nms_iou = nms_iou #
        
        if self.use_logit_adjustment:
            # (Milestone TBD) Logit Adjustment
            if samples_per_cls is None:
                print("Warning: use_logit_adjustment=True but samples_per_cls is None. Ignoring.")
                self.use_logit_adjustment = False
            else:
                print(f"[Model] Using Test-Time Logit Adjustment with tau={self.tau}")
                class_freq = np.array(samples_per_cls)
                class_prior = class_freq / class_freq.sum()
                self.register_buffer(
                    'logit_adjustment',
                    torch.log(torch.tensor(class_prior, dtype=torch.float) + 1e-12) * self.tau
                )

    def forward(self, 
                images: List[torch.Tensor], 
                targets: Optional[List[Dict]] = None):
        """
        根據 model.training 決定執行 訓練 或 驗證
        """
        # images 是 list of [C, H, W], stack 成 [B, C, H, W]
        batch_imgs = torch.stack(images)
        
        if self.training:
            return self.forward_train(batch_imgs, targets)
        else:
            return self.forward_eval(batch_imgs)

    def forward_train(self, 
                      batch_imgs: torch.Tensor, 
                      targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        訓練模式：回傳 loss dict
        """
        # --- 1. 準備 Targets 格式 (for ultralytics loss) ---
        # (x, y, w, h) normalized
        h, w = batch_imgs.shape[2:]
        batch_idx_boxes_labels = []
        for i, target in enumerate(targets):
            boxes_xyxy = target['boxes']
            labels = target['labels']
            
            if len(boxes_xyxy) == 0:
                continue
            
            # 轉換 xyxy (pixel) -> xywh (normalized)
            boxes_xywh_norm = ops.xyxy2xywhn(boxes_xyxy, w=w, h=h)
            
            # 建立 [batch_idx, class, x_c, y_c, w_n, h_n]
            batch_idx_col = torch.full((len(labels), 1), i, 
                                       device=labels.device, dtype=labels.dtype)
            
            # [N, 6]
            target_tensor = torch.cat([
                batch_idx_col,
                labels.unsqueeze(1),
                boxes_xywh_norm
            ], dim=1)
            
            batch_idx_boxes_labels.append(target_tensor)
        
        targets_tensor = torch.cat(batch_idx_boxes_labels, dim=0)
        
        # --- 2. 模型 Forward (取得 raw feature maps) ---
        # preds: Tuple[torch.Tensor, ...]
        preds = self.model(batch_imgs)
        
        # --- 3. 計算 Loss ---
        loss_dict = self.loss_fn(preds, targets_tensor)
        
        # train_one_epoch 需要 'loss' 
        return loss_dict

    def forward_eval(self, batch_imgs: torch.Tensor) -> List[List[Dict]]:
        """
        驗證/測試模式：回傳 NMS 後的預測結果
        
        回傳: List[List[Dict]]
        - [Batch_size, Num_Heads, Dict]
        - Dict: {'boxes': [N, 4], 'scores': [N], 'labels': [N]}
        """
        
        # --- 1. 模型 Forward (取得 raw feature maps) ---
        preds = self.model(batch_imgs)
        
        # --- 2. 將 Preds 解碼並執行 NMS ---
        # self.loss_fn.decode_and_nms 會處理
        # 它內部會呼叫 self.loss_fn.pre_process (decode)
        # 和 ops.non_max_suppression (NMS)
        
        # (Milestone TBD) Logit Adjustment
        if self.use_logit_adjustment:
            preds = self.apply_logit_adjustment(preds)

        results = self.loss_fn.decode_and_nms(
            preds,
            conf_thres=self.conf_threshold,
            iou_thres=self.nms_iou
        )
        
        # --- 3. 格式化輸出 (for evaluate hook) ---
        # results: [B, N, 6] (x1, y1, x2, y2, conf, cls)
        
        batch_outputs = []
        for res in results:
            boxes = res[:, :4]
            scores = res[:, 4]
            labels = res[:, 5].long()
            
            # 每個 head 一個 dict (雖然 v8 只有一個 head)
            head_output = {
                'boxes': boxes,  # xyxy
                'scores': scores,
                'labels': labels
            }
            batch_outputs.append([head_output]) # 包一層 list
            
        return batch_outputs
        
    def apply_logit_adjustment(self, preds):
        """ (Milestone TBD) Test-Time Logit Adjustment """
        # preds: List[Tensor[B, C, H, W], ...]
        # 我們只調整 class logits
        
        # 假設 preds[1] 是 class head (YOLOv8)
        # 實際: preds 是 List of 3 Tensors, [B, 84, 80, 80], [B, 84, 40, 40], [B, 84, 20, 20]
        # 84 = 64 (reg) + 20 (cls)
        # 不對，DetectionModel 回傳的是 [B, 4(reg)+1(obj)+NC(cls), H, W]
        # 也不對，v8DetectionLoss.preprocess 顯示
        # shape(p[0]) = [B, 64, 80, 80], shape(p[1]) = [B, 4, 80, 80], shape(p[2]) = [B, NC, 80, 80]
        # 好，YOLOv8 的 DetectionModel (P5) 回傳的是
        # (Tensor(reg_output), Tensor(cls_output))
        # reg_output: list of 3 [B, 64, H, W]
        # cls_output: list of 3 [B, NC, H, W]
        
        reg_preds, cls_preds = preds
        
        adjusted_cls_preds = []
        for cls_p in cls_preds:
            # cls_p: [B, NC, H, W]
            # self.logit_adjustment: [NC]
            # 廣播: [B, NC, H, W] + [1, NC, 1, 1]
            adj_cls_p = cls_p + self.logit_adjustment.view(1, -1, 1, 1)
            adjusted_cls_preds.append(adj_cls_p)
            
        return (reg_preds, adjusted_cls_preds)