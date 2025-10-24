# hooks.py
# 支援兩種任務： cfg["task"] in {"detection","classification"}
# 你只需把 TODO 區塊換成你的模型/資料/評估；其餘框架已接好。

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 可選：IoU 工具（若沒有 torchvision 也 ok；mAP 真正評估請接 pycocotools）
try:
    from torchvision.ops import box_iou as _box_iou
except Exception:
    _box_iou = None


# --------------- 公用：Detection collate（list形式） ---------------
def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ===================== 1) 建立模型 =====================
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    TODO: 這裡換成你的實際模型
    - 若是 YOLO/Detector，建議包成 forward(images, targets)->{'loss':...} / forward(images)->preds
    - 若是分類模型，forward(x)->logits
    """
    task = cfg.get("task", "detection")

    if task == "classification":
        num_classes = int(cfg.get("num_classes", 1000))
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )
        return model

    elif task == "detection":
        num_classes = int(cfg.get("num_classes", 80))

        class TinyDet(nn.Module):
            """教學示例：把每張圖視作 1 個偵測；真實專案請替換。"""
            def __init__(self, num_classes: int):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1, stride=2),
                    nn.ReLU(True),
                    nn.Conv2d(32, 64, 3, padding=1, stride=2),
                    nn.ReLU(True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.cls = nn.Linear(64, num_classes)
                self.box = nn.Linear(64, 4)  # xywh（示意）
                self.crit = nn.CrossEntropyLoss()

            def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
                feats = []
                for img in images:
                    x = self.backbone(img.unsqueeze(0))   # [1,64,1,1]
                    feats.append(x.view(1, -1))
                F = torch.cat(feats, dim=0)  # [B,64]
                logits = self.cls(F)
                box = self.box(F)

                if targets is not None:
                    # 以第一個 GT label 當目標（示意）
                    labels = []
                    for t in targets:
                        if t.get("labels") is not None and t["labels"].numel() > 0:
                            labels.append(t["labels"][0].clamp(min=0, max=logits.shape[1]-1))
                        else:
                            labels.append(torch.zeros((), dtype=torch.long, device=logits.device))
                    labels = torch.stack(labels, 0)
                    loss = self.crit(logits, labels)
                    return {"loss": loss}
                else:
                    probs = logits.softmax(-1)
                    scores, classes = probs.max(-1)
                    # 假輸出：空框（示意）；真實專案請輸出 NMS 後多框（xyxy）
                    xyxy = torch.zeros((logits.size(0), 4), device=logits.device)
                    return [{"boxes": xyxy, "scores": scores, "labels": classes} for _ in range(logits.size(0))]

        return TinyDet(num_classes)

    else:
        raise ValueError(f"Unknown task: {task}")


# ===================== 2) DataLoaders =====================
def build_dataloaders(cfg: Dict[str, Any],
                      fold_split: Optional[Dict[str, List[str]]] = None,
                      dist: bool = False) -> Dict[str, Any]:
    """
    TODO: 把你的 Dataset 接上來；並確保 dataset.task = cfg["task"]
      - Detection: __getitem__ 回 (image, target_dict)；target 需含 boxes[xyxy], labels, image_id
      - Classification: __getitem__ 回 (image, label)
    """
    task = cfg.get("task", "detection")
    B = int(cfg.get("batch", 16))
    num_workers = int(cfg.get("workers", 4))

    # 示意資料集（請替換為你的資料集）
    class ClassificationDataset(Dataset):
        def __init__(self, ids: List[str], imgsz=224):
            self.ids = ids; self.imgsz = imgsz
            self.num_classes = int(cfg.get("num_classes", 10))
            self.task = "classification"
        def __len__(self): return len(self.ids)
        def __getitem__(self, idx):
            img = torch.rand(3, self.imgsz, self.imgsz)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return img, label

    class DetectionDataset(Dataset):
        def __init__(self, ids: List[str], imgsz=640):
            self.ids = ids; self.imgsz = imgsz
            self.num_classes = int(cfg.get("num_classes", 80))
            self.task = "detection"
        def __len__(self): return len(self.ids)
        def __getitem__(self, idx):
            img = torch.rand(3, self.imgsz, self.imgsz)
            boxes = torch.tensor([[50., 60., 200., 220.]], dtype=torch.float32)  # xyxy
            labels = torch.tensor([1], dtype=torch.long)
            target = {"boxes": boxes, "labels": labels, "image_id": self.ids[idx]}
            return img, target

    # 依 fold_split or cfg 切資料
    if fold_split is not None:
        train_ids = fold_split.get("train_ids", [])
        val_ids   = fold_split.get("val_ids", [])
    else:
        all_ids = [f"img_{i:05d}.jpg" for i in range(int(cfg.get("num_samples", 1000)))]
        split = int(len(all_ids) * 0.8)
        train_ids, val_ids = all_ids[:split], all_ids[split:]

    test_ids = cfg.get("test_ids", [])

    if task == "classification":
        ds_train = ClassificationDataset(train_ids, imgsz=int(cfg.get("imgsz", 224)))
        ds_val   = ClassificationDataset(val_ids,   imgsz=int(cfg.get("imgsz", 224)))
        ds_test  = ClassificationDataset(test_ids,  imgsz=int(cfg.get("imgsz", 224))) if test_ids else None
        collate_fn = None
    else:
        ds_train = DetectionDataset(train_ids, imgsz=int(cfg.get("imgsz", 640)))
        ds_val   = DetectionDataset(val_ids,   imgsz=int(cfg.get("imgsz", 640)))
        ds_test  = DetectionDataset(test_ids,  imgsz=int(cfg.get("imgsz", 640))) if test_ids else None
        collate_fn = detection_collate_fn

    # Distributed samplers
    sampler_train = DistributedSampler(ds_train, shuffle=True) if dist else None
    sampler_val   = DistributedSampler(ds_val, shuffle=False) if dist else None
    sampler_test  = DistributedSampler(ds_test, shuffle=False) if (dist and ds_test is not None) else None

    dl_train = DataLoader(
        ds_train, batch_size=B, shuffle=(sampler_train is None),
        sampler=sampler_train, num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False
    )
    dl_val = DataLoader(
        ds_val, batch_size=B, shuffle=False,
        sampler=sampler_val, num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False
    )
    dl_test = None
    if ds_test is not None:
        dl_test = DataLoader(
            ds_test, batch_size=B, shuffle=False,
            sampler=sampler_test, num_workers=num_workers, pin_memory=True,
            collate_fn=collate_fn, drop_last=False
        )

    meta = {
        "class_names": cfg.get("class_names"),  # 可選
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids
    }
    return {"train": dl_train, "val": dl_val, "test": dl_test, "meta": meta}


# ===================== 3) 評估 =====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    回傳：
      metrics:   Dict[str, float] -> {"map50":..., "map5095":..., "val_loss":... 或 "accuracy":...}
      per_class: List[Dict]       -> [{"class":..., "ap50":..., "ap5095":..., "ar":...}, ...]
      per_dets:  List[Dict]       -> [{"image_id":..., "class":..., "score":..., "x":..., "y":..., "w":..., "h":..., "format":"xywh"}, ...]
    * 這份為簡化版模板，真實專案建議接入你的正式 evaluator（COCO 等）。
    """
    model.eval()
    task = getattr(loader.dataset, "task", "detection")

    # ---- Classification ----
    if task == "classification":
        crit = nn.CrossEntropyLoss(reduction="sum")
        total = 0
        correct = 0
        val_loss = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            out = model(x)
            val_loss += crit(out, y).item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

        acc = correct / max(1, total)
        val_loss = val_loss / max(1, total)
        metrics = {"accuracy": acc, "val_loss": val_loss}
        return metrics, [], []  # per_class / per_dets 略

    # ---- Detection（簡化，可替換為正式 COCO 評估）----
    per_dets: List[Dict] = []
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)  # list[dict]: boxes(xyxy), scores, labels
        for i, out in enumerate(outputs):
            img_id = targets[i].get("image_id", f"img_{i}")
            boxes = out.get("boxes", torch.empty(0, 4, device=device)).detach().float().cpu()
            scores = out.get("scores", torch.empty(0, device=device)).detach().cpu()
            labels = out.get("labels", torch.empty(0, dtype=torch.long, device=device)).detach().cpu()

            if boxes.numel() > 0:
                x = boxes[:, 0]; y = boxes[:, 1]; w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
                for j in range(boxes.shape[0]):
                    per_dets.append({
                        "image_id": str(img_id),
                        "class": int(labels[j].item()) if labels.numel() else 0,
                        "score": float(scores[j].item()) if scores.numel() else 0.0,
                        "x": float(x[j].item()), "y": float(y[j].item()),
                        "w": float(w[j].item()), "h": float(h[j].item()),
                        "format": "xywh"
                    })

    # 模板不計算真正 mAP，避免誤導；請換成你的 evaluator 並回填 map50/map5095
    metrics = {"map50": None, "map5095": None, "val_loss": None}
    per_class: List[Dict] = []
    return metrics, per_class, per_dets
