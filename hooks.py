# hooks.py
# 職責：作為 train.py 與專案模組 (model, data, loss) 之間的橋樑。

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import json
import os

# 匯入我們自定義的模組
from data_process.dataset import DroneTrafficDataset
from data_process.sampler import RepeatFactorSampler, DistributedRepeatFactorSampler
from models.yolo_detector import YOLODetector
from tools.utils import get_rank, is_distributed
from pathlib import Path # 確保
import os # 確保
import numpy as np # 確保
from data_process.dataset import DroneTrafficDataset # 確保


def detection_collate_fn(batch):
    """
    (保持不變)
    """
    images, targets = zip(*batch)
    return list(images), list(targets)

def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    根據 config 建立模型 (YOLODetector)
    """
    model_name = cfg.get("model_name", "yolov8n.yaml") # 讀取 .yaml 代表 from scratch
    if model_name.endswith(".pt"):
        print(f"Warning: model_name '{model_name}' ends with .pt, but we are training from scratch.")
        print("Using yolov8n.yaml instead.")
        model_name = "yolov8n.yaml"

    loss_hpy = cfg.get('loss_hpy', {}) # 獲取 loss_hpy 字典

    model = YOLODetector(
        model_name=model_name,
        num_classes=cfg.get('num_classes', 4),
        loss_hpy=cfg.get('loss_hpy', {}),   # <-- 將 loss 超參數傳入

        # Long-Tail 相關參數 (Milestone 2+)
        use_cb_loss=cfg.get('use_cb_loss', False),
        samples_per_cls=cfg.get('samples_per_cls'),
        cb_beta=cfg.get('cb_beta', 0.9999),
        focal_gamma=cfg.get('focal_gamma', 2.0),
        
        # Test-time Logit Adjustment (Milestone TBD)
        use_logit_adjustment=cfg.get('use_logit_adjustment', False),
        tau=cfg.get('tau', 1.0)
    )
    
    return model


def build_dataloaders(cfg: Dict[str, Any],
                      fold_split: Optional[Dict[str, List[str]]] = None,
                      dist: bool = False) -> Dict[str, Any]:
    """
    建立 Dataloaders
    (修改：不再讀取 annotation_file，而是掃描 image_dir)
    """
    
    batch_size = int(cfg.get("batch", 16))
    num_workers = int(cfg.get("workers", 8))
    
    # --- 1. 取得 Train/Val Image IDs ---
    train_ids = []
    val_ids = []
    all_ids = [] # 用於 k-fold 或 ratio split
    
    # 獲取 image_dir (e.g., "data/CVPDL_hw2/train")
    image_dir = Path(cfg['image_dir'])
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # 優先 1: K-Fold
    if fold_split is not None:
        print(f"[Data] Using K-Fold split: Fold {fold_split.get('fold', '?')}")
        train_ids = fold_split.get("train_ids", [])
        val_ids = fold_split.get("val_ids", [])
    
    # 優先 2: 檔案列表
    elif cfg.get("split_method") == "file":
        print(f"[Data] Using split_method: 'file'")
        if "train_ids_file" in cfg:
            train_ids = Path(cfg["train_ids_file"]).read_text().splitlines()
        if "val_ids_file" in cfg:
            val_ids = Path(cfg["val_ids_file"]).read_text().splitlines()
        
    # 優先 3: 比例切分 (預設)
    else:
        # 【修改點】
        print(f"[Data] Using split_method: 'ratio'. Scanning {image_dir} for images...")
        # 掃描 image_dir 找出所有圖片 (e.g., img0001.png, img0002.png)
        all_ids = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
             all_ids.extend([p.name for p in image_dir.glob(f'*{ext}')])
        
        if not all_ids:
            raise FileNotFoundError(f"No images found in {image_dir}")
        
        # 打亂
        np.random.seed(cfg.get("seed", 42))
        np.random.shuffle(all_ids)
        
        train_r = cfg.get("train_ratio", 0.8)
        val_r = cfg.get("val_ratio", 0.2)
        
        if (train_r + val_r) > 1.0:
            print(f"Warning: train_ratio ({train_r}) + val_ratio ({val_r}) > 1.0. Adjusting val_ratio.")
            val_r = 1.0 - train_r
            
        train_idx = int(len(all_ids) * train_r)
        val_idx = int(len(all_ids) * (train_r + val_r))
        
        train_ids = all_ids[:train_idx]
        val_ids = all_ids[train_idx:val_idx]
        # test_ids = all_ids[val_idx:] # 剩下的當作 test
        
    print(f"[Data] Total images scanned: {len(all_ids)}")
    print(f"[Data] Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    
    # --- 2. 建立 Datasets ---
    # 【修改點】: 不再傳入 annotation_file
    ds_train = DroneTrafficDataset(
        image_dir=cfg['image_dir'],
        image_ids=train_ids,
        imgsz=cfg.get('imgsz', 640),
        augment=True,
        class_names=cfg.get('class_names'),
        strong_aug_for_tail=cfg.get('strong_aug_for_tail', False),
        tail_class_ids=cfg.get('tail_class_ids', []),
        augmentation_config=cfg.get('augmentation', {})
    )
    
    ds_val = DroneTrafficDataset(
        image_dir=cfg['image_dir'],
        image_ids=val_ids,
        imgsz=cfg.get('imgsz', 640),
        augment=False,
        class_names=cfg.get('class_names')
    )
    
    # (可選) 建立 Test Dataset (for predict.py)
    ds_test = None
    test_image_dir = cfg.get("test_image_dir") # e.g., "data/CVPDL_hw2/test"
    if test_image_dir and Path(test_image_dir).exists():
        test_ids_list = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            test_ids_list.extend([p.name for p in Path(test_image_dir).glob(f'*{ext}')])
            
        ds_test = DroneTrafficDataset(
            image_dir=test_image_dir,
            image_ids=test_ids_list,
            imgsz=cfg.get('imgsz', 640),
            augment=False,
            class_names=cfg.get('class_names')
        )
        print(f"[Data] Test images: {len(test_ids_list)}")
    
    # --- 3. 建立 Samplers (Milestone 1 vs 3) ---
    # ( ... 這部分保持不變 ...)
    sampler_train = None
    sampler_val = None
    sampler_test = None
    
    use_rfs = cfg.get('use_repeat_factor_sampling', False)
    
    if dist:
        # DDP
        if use_rfs:
            sampler_train = DistributedRepeatFactorSampler(
                ds_train, 
                repeat_thresh=cfg.get('repeat_thresh', 0.001),
                rank=get_rank(),
                num_replicas=int(os.environ["WORLD_SIZE"]),
                seed=cfg.get("seed", 42)
            )
        else:
            sampler_train = DistributedSampler(ds_train, shuffle=True)
            
        sampler_val = DistributedSampler(ds_val, shuffle=False)
        sampler_test = DistributedSampler(ds_test, shuffle=False) if ds_test else None
        
    elif use_rfs:
        sampler_train = RepeatFactorSampler(
            ds_train, 
            repeat_thresh=cfg.get('repeat_thresh', 0.001)
        )
    
    # --- 4. 建立 Dataloaders ---
    # ( ... 這部分保持不變 ...)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size * 2,
        shuffle=False,
        sampler=sampler_val,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False
    )
    
    dl_test = None
    if ds_test:
        dl_test = DataLoader(
            ds_test,
            batch_size=batch_size * 2,
            shuffle=False,
            sampler=sampler_test,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn,
            drop_last=False
        )
    
    meta = {
        "all_ids": all_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
    
    return {"train": dl_train, "val": dl_val, "test": dl_test, "meta": meta}

# ==================== Evaluation ====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    評估模型並回傳指標
    使用 torchmetrics 計算 mAP
    """
    model.eval()
    
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        use_torchmetrics = True
    except ImportError:
        print("Warning: torchmetrics not found. Install with: pip install torchmetrics")
        print("Falling back to dummy metrics.")
        use_torchmetrics = False
    
    # 收集所有預測和標註
    all_preds = []
    all_targets = []
    per_dets = [] # 用於儲存原始預測框
    
    for images, targets in loader:
        # 注意：images 是 list，但在 YOLODetector.forward_eval 中會處理
        images = [img.to(device, non_blocking=True) for img in images]
        
        # YOLODetector 在 eval 模式下會直接回傳 NMS 後的結果
        # outputs: List[List[Dict{'boxes', 'scores', 'labels'}]]
        outputs = model(images) 
        
        for i, (out_list, tgt) in enumerate(zip(outputs, targets)):
            # out_list 是 YOLODetector.forward_eval 的回傳
            # 我們取第一個 (也是唯一一個) head 的結果
            out = out_list[0] 
            
            pred_boxes = out['boxes'].cpu()
            pred_scores = out['scores'].cpu()
            pred_labels = out['labels'].cpu()
            
            gt_boxes = tgt['boxes']
            gt_labels = tgt['labels']
            img_id = tgt.get('image_id', f"batch_{i}")
            
            # 格式轉換 for torchmetrics
            all_preds.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
            
            all_targets.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })
            
            # 儲存 per-detection results (jsonl)
            for j in range(len(pred_boxes)):
                x1, y1, x2, y2 = pred_boxes[j].tolist()
                per_dets.append({
                    'image_id': str(img_id),
                    'class': int(pred_labels[j].item()),
                    'score': float(pred_scores[j].item()),
                    'x': float(x1),
                    'y': float(y1),
                    'w': float(x2 - x1),
                    'h': float(y2 - y1),
                    'format': 'xywh'
                })
    
    # --- 計算指標 ---
    metrics = {'map50': 0.0, 'map5095': 0.0, 'val_loss': None}
    per_class = []

    if use_torchmetrics and len(all_preds) > 0:
        # mAP @ IoU=0.5:0.95 (COCO primary)
        # mAP @ IoU=0.5 (PASCAL VOC)
        metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True, 
                                      iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        metric.update(all_preds, all_targets)
        results = metric.compute()
        
        metrics = {
            'map50': float(results.get('map_50', 0)),
            'map5095': float(results.get('map', 0)), # 'map' is mAP@50:95
            'map75': float(results.get('map_75', 0)),
            'val_loss': None # 'evaluate' 階段不計算 loss
        }
        
        # Per-class metrics
        class_names = loader.dataset.class_names
        
        # 確保 results key 存在
        map_per_class = results.get('map_per_class', torch.zeros(len(class_names)))
        map_50_per_class = results.get('map_50_per_class', torch.zeros(len(class_names)))
        mar_100_per_class = results.get('mar_100_per_class', torch.zeros(len(class_names)))

        for cls_id in range(len(class_names)):
            per_class.append({
                'class': class_names[cls_id],
                'class_id': cls_id,
                'ap50': float(map_50_per_class[cls_id]),
                'ap5095': float(map_per_class[cls_id]),
                'ar100': float(mar_100_per_class[cls_id]),
            })
    
    return metrics, per_class, per_dets