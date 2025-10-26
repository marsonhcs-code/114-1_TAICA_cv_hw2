# models/yolo_detector.py
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
from types import SimpleNamespace

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
                 loss_hpy: Dict[str, Any] = {},
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
        
        h = SimpleNamespace()
        h.box = loss_hpy.get('box', 7.5)  # box loss gain
        h.cls = loss_hpy.get('cls', 0.5)  # cls loss gain
        h.dfl = loss_hpy.get('dfl', 1.5)  # dfl loss gain
        h.iou_type = loss_hpy.get('iou_type', 'ciou') # IoU 類型
        
        # (這些是 v8 內部預設值，不太會動，保持原樣)
        h.box_bce_pos_weight = 1.0
        h.dfl_bce_pos_weight = 1.0
        
        # 將這個 "args" 附加到 self.model 上
        self.model.args = h

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
        (【!!! 重寫此函數 !!!】)
        """
        
        # --- 1. 模型 Forward (取得 raw feature maps) ---
        preds = self.model(batch_imgs)
        
        # --- 2. 準備 Targets 格式 (for v8DetectionLoss) ---
        h, w = batch_imgs.shape[2:]
        device = batch_imgs.device

        batch_bboxes_list = []
        batch_cls_list = []
        batch_idx_list = []

        for i, target in enumerate(targets):
            boxes_xyxy = target['boxes'].to(device)
            labels = target['labels'].to(device)
            
            if len(boxes_xyxy) == 0:
                continue
                
            # 轉換 xyxy (pixel) -> xywh (normalized)
            boxes_xywh_norm = ops.xyxy2xywhn(boxes_xyxy, w=w, h=h)
            
            # 建立 batch_idx tensor
            batch_idx_tensor = torch.full((len(labels),), i, device=device, dtype=labels.dtype)
            
            batch_bboxes_list.append(boxes_xywh_norm.to(device))  # ✅ 強制放 GPU
            batch_cls_list.append(labels)
            batch_idx_list.append(batch_idx_tensor)

        # 處理 batch 中沒有任何標註的情況
        if not batch_idx_list:
            loss_batch = {
                'bboxes': torch.empty(0, 4, device=device),
                'cls': torch.empty(0, device=device, dtype=torch.long),
                'batch_idx': torch.empty(0, device=device, dtype=torch.long),
                'imgsz': torch.tensor([h, w], device=device)
            }
        else:
            loss_batch = {
                'bboxes': torch.cat(batch_bboxes_list, dim=0).to(device),  # ✅
                'cls': torch.cat(batch_cls_list, dim=0).to(device),        # ✅
                'batch_idx': torch.cat(batch_idx_list, dim=0).to(device),  # ✅
                'imgsz': torch.tensor([h, w], device=device)
            }
        print(f"[DEBUG] loss_batch device: bboxes={loss_batch['bboxes'].device}, imgsz={loss_batch['imgsz'].device}")

        # models/yolo_detector.py (forward_train 函數結尾)

        # models/yolo_detector.py (forward_train 函數結尾)

        # --- 3. 計算 loss ---
        loss_out = self.loss_fn(preds, loss_batch)

        # 統一回傳 dict，至少有 key "loss"
        result = {}

        # Case A: loss_out 是已經加總好的單一 tensor
        if torch.is_tensor(loss_out):
            result["loss"] = loss_out

        # Case B: loss_out 是 tuple 或 list
        elif isinstance(loss_out, (tuple, list)):
            if len(loss_out) == 2:
                # 典型 Ultralytics 形式: (total_loss, loss_items)
                total_loss, loss_items = loss_out
                # 確保 total_loss 是 Tensor
                if not torch.is_tensor(total_loss):
                     raise RuntimeError(f"Expected total_loss to be a tensor, but got {type(total_loss)} in loss_out tuple")
                result["loss"] = total_loss

                # 嘗試拆分 loss_items 做 logging
                if isinstance(loss_items, (list, tuple)):
                    if len(loss_items) > 0: result["loss_box"] = loss_items[0]
                    if len(loss_items) > 1: result["loss_cls"] = loss_items[1]
                    if len(loss_items) > 2: result["loss_dfl"] = loss_items[2]
                elif torch.is_tensor(loss_items):
                    result["loss_items"] = loss_items # 純記錄用

            elif len(loss_out) == 3:
                # 自訂/簡化版: (lbox, lcls, ldfl)
                lbox, lcls, ldfl = loss_out
                if not all(torch.is_tensor(t) for t in loss_out):
                     raise RuntimeError(f"Expected 3 tensors in loss_out tuple, but got types {[type(t) for t in loss_out]}")
                total_loss = lbox + lcls + ldfl
                result["loss"] = total_loss
                result["loss_box"] = lbox
                result["loss_cls"] = lcls
                result["loss_dfl"] = ldfl

            else:
                # 不認得的 tuple/list 形狀 -> 盡量拿第一個當主 loss
                main_loss = loss_out[0]
                if not torch.is_tensor(main_loss):
                    raise RuntimeError(
                        f"loss_fn returned {type(loss_out).__name__}(len={len(loss_out)}) "
                        f"but first element is {type(main_loss)}, not Tensor"
                    )
                result["loss"] = main_loss
                result["loss_extra"] = loss_out[1:]

        # Case C: loss_out 是 dict
        elif isinstance(loss_out, dict):
            if "loss" not in loss_out:
                raise RuntimeError(
                    f"loss_fn returned dict without 'loss' key: {loss_out.keys()}"
                )
            # Make sure the 'loss' value is a tensor
            if not torch.is_tensor(loss_out['loss']):
                raise RuntimeError(f"loss_fn returned dict, but 'loss' value is not a tensor ({type(loss_out['loss'])})")
            result = loss_out.copy() # Use the dict directly

        else:
            raise RuntimeError(
                f"Unsupported loss_out type from loss_fn: {type(loss_out)}"
            )

        # ---- 關鍵補強：確保 result['loss'] 是 0-dim scalar tensor ----
        if "loss" not in result or not torch.is_tensor(result["loss"]):
             # This should not happen if the logic above is correct, but added as a safeguard
             raise RuntimeError(f"Robust parsing failed to produce a tensor for key 'loss'. Result: {result}, loss_out type: {type(loss_out)}")

        main_loss = result["loss"]
        if main_loss.dim() > 0:
            # print(f"Warning: main_loss had dim > 0 ({main_loss.shape}), summing to scalar.") # Optional debug print
            main_loss = main_loss.sum()

        # 更新回 dict，確保 train loop 一定拿到 scalar tensor
        result["loss"] = main_loss

        # (可選) 確保其他 loss components 也是 scalar (如果存在)
        for key in ["loss_box", "loss_cls", "loss_dfl", "loss_items"]:
             if key in result and torch.is_tensor(result[key]) and result[key].dim() > 0:
                 # Be careful summing 'loss_items' if it's not meant to be summed
                 if key != 'loss_items':
                     result[key] = result[key].sum()

        return result

    def forward_eval(self, batch_imgs: List[torch.Tensor]):
        """
        推論 / 驗證模式：
        - 使用 Ultralytics 內建的推論 (model.eval() 時會自動做 decode + NMS)
        - 把結果轉成 hooks.evaluate() 期待的格式
        """

        # batch_imgs: List[Tensor[C,H,W]] from dataloader.
        # Ultralytics model 需要一個 4D tensor (B,C,H,W)，所以把它們疊起來
        imgs = torch.stack(batch_imgs, dim=0)  # [B, C, H, W]

        # （可選）把 conf / iou threshold 傳給底層 model，確保一致
        # Ultralytics DetectionModel 支援 .conf / .iou / .max_det 這些屬性
        if hasattr(self.model, "conf"):
            self.model.conf = getattr(self, "conf_threshold", 0.25)
        if hasattr(self.model, "iou"):
            self.model.iou = getattr(self, "nms_iou", 0.45)
        if hasattr(self.model, "max_det") and hasattr(self, "max_det"):
            self.model.max_det = self.max_det

        # 取得 Ultralytics 的推論結果
        # 這裡 self.model 已經是 eval() 狀態，所以會回 list[Results]
        ul_results = self.model(imgs)

        # 轉成 evaluate() 需要的格式
        batch_outputs = []
        for res in ul_results:
            # res.boxes is a Boxes object
            boxes_xyxy = res.boxes.xyxy          # Tensor [N,4] on device
            scores     = res.boxes.conf          # Tensor [N]
            labels     = res.boxes.cls.to(torch.long)  # Tensor [N]

            head_output = {
                'boxes': boxes_xyxy,
                'scores': scores,
                'labels': labels,
            }

            # evaluate() 期望的是 List[List[dict]] per image
            batch_outputs.append([head_output])

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