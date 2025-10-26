import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
    from ultralytics.utils import ops
    from ultralytics.nn.tasks import v8DetectionLoss
except ImportError:
    raise ImportError("Please install 'ultralytics': pip install ultralytics")

from losses.cb_focal_loss import ClassBalancedFocalLoss


class YOLOLongTailLoss(v8DetectionLoss):
    """
    繼承 YOLOv8 Loss，並客製化 Class Loss
    Milestone 1 (use_cb_loss=False): 同原版 YOLOv8
    Milestone 2 (use_cb_loss=True): 使用 ClassBalancedFocalLoss
    """

    def __init__(self,
                 model,
                 use_cb_loss: bool = False,
                 samples_per_cls: Optional[List[int]] = None,
                 cb_beta: float = 0.9999,
                 focal_gamma: float = 2.0):
        super().__init__(model)
        self.model = model  # ✅ 關鍵修正：保存 model
        self.use_cb_loss = use_cb_loss

        if self.use_cb_loss:
            if samples_per_cls is None:
                raise ValueError("use_cb_loss=True, but samples_per_cls is not provided.")
            print(f"[Loss] Using ClassBalancedFocalLoss (beta={cb_beta}, gamma={focal_gamma})")
            self.bce = ClassBalancedFocalLoss(samples_per_cls=samples_per_cls,
                                              beta=cb_beta, gamma=focal_gamma)
        else:
            print("[Loss] Using standard BCEWithLogitsLoss for classification.")

    def __call__(self, preds, batch):
        """
        最小且正確的補丁：
        1) 確保 batch 內所有 tensor 在同一張 GPU (避免 gt 還在 CPU)
        2) 把 self.stride 搬到同一張 GPU (避免 make_anchors() 產 CPU anchor_points)
        3) 把 self.proj / self.assigner 等等也對齊到同一張 GPU
        4) 然後呼叫父類的 super().__call__() 讓 Ultralytics 原本的 loss 流程跑完
        """

        # 用 preds 的裝置當作 ground truth，因為 preds 來自 model.forward() 後一定在正確的卡上
        device = preds[0].device

        # 1) 把 batch 中的資料 (gt_bboxes, gt_labels, imgsz...) 全部搬到同一張卡
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
            elif isinstance(v, (tuple, list)):
                # 假如有 list of tensors，也一起搬；沒有就不影響
                batch[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]

        # 2) 關鍵：把 stride 搬到 GPU
        #    v8DetectionLoss 在 __init__ 時會把 model.stride 複製到 self.stride
        #    但那一份是 CPU tensor；父類 __call__ 後面要用 self.stride 產 anchors
        if hasattr(self, "stride") and torch.is_tensor(self.stride):
            self.stride = self.stride.to(device)

        # 3) 保險：把 proj (DFL用)、assigner (task-aligned assigner) 也搬到同一張卡
        if hasattr(self, "proj") and torch.is_tensor(self.proj):
            self.proj = self.proj.to(device)

        # self.dfl 通常是 buffer/tensor用在 bbox decode，跟 proj 有關
        if hasattr(self, "dfl") and torch.is_tensor(self.dfl):
            self.dfl = self.dfl.to(device)

        # assigner 是一個 nn.Module；理論上 model.to(device) 會帶著走
        # 但如果 assigner 內有沒註冊成 buffer 的 tensor，我們手動 to() 一次最安全
        if hasattr(self, "assigner"):
            self.assigner = self.assigner.to(device)

            # 一些 ultralytics 版本會在 assigner 裡面 cache grid_points 之類的東西
            if hasattr(self.assigner, "grid_points"):
                gp = self.assigner.grid_points
                if torch.is_tensor(gp):
                    self.assigner.grid_points = gp.to(device)

        # 4) （可選除錯）你可以暫時印一下，之後確認沒問題就刪掉
        # print("[DEBUG Loss.__call__]",
        #       "stride:", self.stride.device if hasattr(self,"stride") and torch.is_tensor(self.stride) else "N/A",
        #       "proj:",   self.proj.device   if hasattr(self,"proj")   and torch.is_tensor(self.proj)   else "N/A")

        # 5) 交還給父類的 loss 計算邏輯
        return super().__call__(preds, batch)

    def decode_and_nms(self, preds, conf_thres, iou_thres):
        """輔助函式：解碼 + NMS"""
        x = [torch.cat((preds[0][i], preds[1][i]), 1) for i in range(len(preds[0]))]
        box, cls = self.pred_decode(x)
        B, N, NC = cls.shape
        pred_nms = torch.cat((box, cls), dim=2)
        results = ops.non_max_suppression(pred_nms, conf_thres=conf_thres,
                                          iou_thres=iou_thres, multi_label=True)
        return results

    def pred_decode(self, x):
        """輔助：從 head output 解碼"""
        device = x[0].device
        self.anchors, self.strides = (y.transpose(0, 1) for y in make_anchors(x, self.stride.to(device), 0.5))
        self.anchors = self.anchors.to(device)
        self.strides = self.strides.to(device)
        x_cat = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        box = torch.cat((self.project(box[:, :self.reg_max*2]), self.project(box[:, self.reg_max*2:])), 1)
        box = box.transpose(1, 2).view(-1, 4, self.reg_max)
        box = F.softmax(box, dim=-1)
        box = box @ self.dfl.to(device)
        box = dist2bbox(box, self.anchors.unsqueeze(0), xywh=False, dim=1)
        cls = cls.transpose(1, 2).sigmoid()
        return box, cls
