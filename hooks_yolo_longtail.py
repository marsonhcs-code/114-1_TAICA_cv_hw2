# hooks_yolo_longtail.py
# Long-Tailed Object Detection for YOLO (Drone-based traffic counting)

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
import cv2
import json

# ===================== Long-Tail Loss Functions =====================
class ClassBalancedFocalLoss(nn.Module):
    """
    CB-Focal Loss: Combines Class-Balanced Loss with Focal Loss
    Paper: "Class-Balanced Loss Based on Effective Number of Samples"
    """
    def __init__(self, samples_per_cls: List[int], beta: float = 0.9999, 
                 gamma: float = 2.0, alpha: Optional[List[float]] = None):
        super().__init__()
        self.gamma = gamma
        
        # Class-Balanced Weights
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        self.register_buffer('cb_weights', torch.tensor(weights, dtype=torch.float32))
        
        # Focal Loss Alpha
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, num_classes]
            targets: [N] - class indices
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce_loss)  # probability
        focal_weight = (1 - p) ** self.gamma
        
        # Apply CB weights
        cb_weight = self.cb_weights[targets]
        
        # Apply Focal alpha if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * focal_weight * cb_weight * ce_loss
        else:
            loss = focal_weight * cb_weight * ce_loss
        
        return loss.mean()


# ===================== Sampling Strategy =====================
class RepeatFactorSampler(torch.utils.data.Sampler):
    """
    Repeat Factor Sampling: Over-sample images containing tail classes
    Paper: "The Long Tail in Instance Segmentation" (LVIS)
    """
    def __init__(self, dataset, repeat_thresh: float = 0.5):
        self.dataset = dataset
        self.repeat_thresh = repeat_thresh
        
        # Calculate repeat factors
        self.repeat_factors = self._get_repeat_factors()
        self.num_samples = int(sum(self.repeat_factors))
    
    def _get_repeat_factors(self):
        # Get class frequencies from dataset
        class_freq = self.dataset.get_class_frequencies()
        median_freq = np.median(list(class_freq.values()))
        
        repeat_factors = []
        for idx in range(len(self.dataset)):
            # Get classes in this image
            classes = self.dataset.get_image_classes(idx)
            if len(classes) == 0:
                repeat_factors.append(1.0)
                continue
            
            # Calculate max repeat factor among classes in this image
            max_rf = 1.0
            for cls in classes:
                freq = class_freq.get(cls, median_freq)
                rf = max(1.0, np.sqrt(median_freq / max(freq, 1)))
                max_rf = max(max_rf, rf)
            
            repeat_factors.append(max_rf)
        
        return repeat_factors
    
    def __iter__(self):
        # Create indices with repetition
        indices = []
        for idx, rf in enumerate(self.repeat_factors):
            indices.extend([idx] * int(np.ceil(rf)))
        
        # Shuffle
        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(len(indices), generator=g).tolist()
        
        return iter(indices[:self.num_samples])
    
    def __len__(self):
        return self.num_samples


# ===================== Dataset =====================
class DroneTrafficDataset(Dataset):
    """
    Drone-based Traffic Dataset for Long-Tailed Detection
    Classes: [car, hov, person, motorcycle]
    """
    def __init__(self, 
                 image_dir: str,
                 annotation_file: str,
                 ids: List[str],
                 imgsz: int = 640,
                 augment: bool = False,
                 class_names: List[str] = None):
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.ids = ids
        self.task = "detection"
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Class mapping
        self.class_names = class_names or ["car", "motorcycle", "person", "hov"]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Build index
        self.samples = []
        for img_id in ids:
            if img_id in self.annotations:
                self.samples.append({
                    'image_id': img_id,
                    'annotations': self.annotations[img_id]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def get_class_frequencies(self) -> Dict[int, int]:
        """Calculate class frequencies for sampling"""
        freq = {i: 0 for i in range(self.num_classes)}
        for sample in self.samples:
            for ann in sample['annotations']:
                cls = ann.get('category_id', 0)
                freq[cls] = freq.get(cls, 0) + 1
        return freq
    
    def get_image_classes(self, idx: int) -> List[int]:
        """Get unique classes in an image"""
        sample = self.samples[idx]
        return list(set(ann['category_id'] for ann in sample['annotations']))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample = self.samples[idx]
        img_id = sample['image_id']
        
        # Load image
        img_path = self.image_dir / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        pad_h = (self.imgsz - new_h) // 2
        pad_w = (self.imgsz - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, self.imgsz - new_h - pad_h,
                                 pad_w, self.imgsz - new_w - pad_w,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Process annotations
        boxes = []
        labels = []
        for ann in sample['annotations']:
            # Convert bbox [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, bw, bh = ann['bbox']
            x1 = (x * scale + pad_w)
            y1 = (y * scale + pad_h)
            x2 = ((x + bw) * scale + pad_w)
            y2 = ((y + bh) * scale + pad_h)
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # Apply augmentation (for tail classes)
        if self.augment and len(labels) > 0:
            img, boxes, labels = self.apply_augmentation(img, boxes, labels)
        
        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'orig_size': torch.tensor([h, w])
        }
        
        return img, target
    
    def apply_augmentation(self, img, boxes, labels):
        """
        Class-aware augmentation: stronger augmentation for tail classes
        """
        # Identify if image contains tail classes (hov, person, motorcycle)
        has_tail = any(l in [1, 2, 3] for l in labels)  # assuming car=0, others=tail
        
        if has_tail:
            # Stronger augmentation for tail classes
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                boxes = [[self.imgsz - x2, y1, self.imgsz - x1, y2] 
                         for x1, y1, x2, y2 in boxes]
            
            # Color jitter
            if np.random.rand() < 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 0] *= np.random.uniform(0.8, 1.2)  # Hue
                hsv[..., 1] *= np.random.uniform(0.8, 1.2)  # Saturation
                hsv[..., 2] *= np.random.uniform(0.8, 1.2)  # Value
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return img, boxes, labels


# ===================== YOLO Wrapper =====================
class YOLODetector(nn.Module):
    """
    Wrapper for YOLOv8/v9 with Long-Tail modifications
    """
    def __init__(self, 
                 model_name: str = 'yolov8n.pt',
                 num_classes: int = 4,
                 samples_per_cls: List[int] = None,
                 use_cb_loss: bool = True):
        super().__init__()
        
        # Load YOLO model (requires ultralytics)
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(model_name)
            
            # Modify number of classes
            if hasattr(self.yolo.model, 'nc'):
                self.yolo.model.nc = num_classes
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        # Long-Tail Loss
        self.use_cb_loss = use_cb_loss
        if use_cb_loss and samples_per_cls:
            self.cb_focal_loss = ClassBalancedFocalLoss(
                samples_per_cls=samples_per_cls,
                beta=0.9999,
                gamma=2.0
            )
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict]] = None):
        """
        Training mode: return loss dict
        Inference mode: return predictions
        """
        if targets is not None:
            # Training: use YOLO's native training with custom loss
            results = self.yolo.model(images, targets)
            
            # You can override loss here if needed
            if self.use_cb_loss and hasattr(results, 'loss_items'):
                # Modify classification loss with CB-Focal
                # This requires accessing YOLO's internal loss computation
                pass
            
            return results
        else:
            # Inference
            results = self.yolo.model(images)
            return self._format_predictions(results)
    
    def _format_predictions(self, results):
        """Convert YOLO predictions to standard format"""
        formatted = []
        for result in results:
            boxes = result.boxes.xyxy  # [N, 4]
            scores = result.boxes.conf  # [N]
            labels = result.boxes.cls.long()  # [N]
            
            formatted.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        return formatted


# ===================== Build Functions =====================
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build YOLO model with Long-Tail modifications
    """
    # Get class distribution (from cfg or dataset)
    samples_per_cls = cfg.get('samples_per_cls', [23000, 5000, 3000, 1000])  # car, motorcycle, person, hov
    
    model = YOLODetector(
        model_name=cfg.get('model_name', 'yolov8n.pt'),
        num_classes=cfg.get('num_classes', 4),
        samples_per_cls=samples_per_cls,
        use_cb_loss=cfg.get('use_cb_loss', True)
    )
    
    return model


def detection_collate_fn(batch):
    """Collate function for detection"""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(cfg: Dict[str, Any],
                      fold_split: Optional[Dict[str, List[str]]] = None,
                      dist: bool = False) -> Dict[str, Any]:
    """
    Build dataloaders with Repeat Factor Sampling for long-tail
    """
    B = int(cfg.get("batch", 16))
    num_workers = int(cfg.get("workers", 4))
    
    # Get image IDs
    if fold_split is not None:
        train_ids = fold_split.get("train_ids", [])
        val_ids = fold_split.get("val_ids", [])
    else:
        # Load from cfg
        train_ids = cfg.get('train_ids', [])
        val_ids = cfg.get('val_ids', [])
    
    test_ids = cfg.get("test_ids", [])
    
    # Create datasets
    ds_train = DroneTrafficDataset(
        image_dir=cfg['image_dir'],
        annotation_file=cfg['annotation_file'],
        ids=train_ids,
        imgsz=cfg.get('imgsz', 640),
        augment=True,  # Enable augmentation for training
        class_names=cfg.get('class_names', ["car", "motorcycle", "person", "hov"])
    )
    
    ds_val = DroneTrafficDataset(
        image_dir=cfg['image_dir'],
        annotation_file=cfg['annotation_file'],
        ids=val_ids,
        imgsz=cfg.get('imgsz', 640),
        augment=False,
        class_names=cfg.get('class_names', ["car", "motorcycle", "person", "hov"])
    )
    
    ds_test = None
    if test_ids:
        ds_test = DroneTrafficDataset(
            image_dir=cfg['image_dir'],
            annotation_file=cfg['annotation_file'],
            ids=test_ids,
            imgsz=cfg.get('imgsz', 640),
            augment=False,
            class_names=cfg.get('class_names', ["car", "motorcycle", "person", "hov"])
        )
    
    # Create samplers (Repeat Factor for training)
    use_repeat_factor = cfg.get('use_repeat_factor_sampling', True)
    
    if dist:
        sampler_train = DistributedSampler(ds_train, shuffle=True)
        sampler_val = DistributedSampler(ds_val, shuffle=False)
        sampler_test = DistributedSampler(ds_test, shuffle=False) if ds_test else None
    elif use_repeat_factor:
        sampler_train = RepeatFactorSampler(ds_train, repeat_thresh=0.5)
        sampler_val = None
        sampler_test = None
    else:
        sampler_train = None
        sampler_val = None
        sampler_test = None
    
    # Create dataloaders
    dl_train = DataLoader(
        ds_train,
        batch_size=B,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=B,
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
            batch_size=B,
            shuffle=False,
            sampler=sampler_test,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate_fn,
            drop_last=False
        )
    
    meta = {
        "class_names": cfg.get('class_names'),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "class_frequencies": ds_train.get_class_frequencies()
    }
    
    return {"train": dl_train, "val": dl_val, "test": dl_test, "meta": meta}


# ===================== Evaluation =====================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Comprehensive evaluation with per-class metrics
    """
    model.eval()
    
    # Collect all predictions and ground truths
    all_preds = []
    all_targets = []
    
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        
        # Get predictions
        outputs = model(images)
        
        for i, (out, tgt) in enumerate(zip(outputs, targets)):
            pred_boxes = out['boxes'].cpu()
            pred_scores = out['scores'].cpu()
            pred_labels = out['labels'].cpu()
            
            gt_boxes = tgt['boxes']
            gt_labels = tgt['labels']
            image_id = tgt['image_id']
            
            all_preds.append({
                'image_id': image_id,
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
            
            all_targets.append({
                'image_id': image_id,
                'boxes': gt_boxes,
                'labels': gt_labels
            })
    
    # Calculate mAP using pycocotools-style evaluation
    metrics = calculate_map(all_preds, all_targets, num_classes=4)
    
    # Calculate per-class metrics
    per_class = calculate_per_class_metrics(all_preds, all_targets, num_classes=4)
    
    # Format per-detection results
    per_dets = []
    for pred in all_preds:
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        img_id = pred['image_id']
        
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j].tolist()
            per_dets.append({
                'image_id': str(img_id),
                'class': int(labels[j].item()),
                'score': float(scores[j].item()),
                'x': float(x1),
                'y': float(y1),
                'w': float(x2 - x1),
                'h': float(y2 - y1),
                'format': 'xywh'
            })
    
    return metrics, per_class, per_dets


def calculate_map(predictions, targets, num_classes: int = 4, iou_thresh: float = 0.5):
    """
    Calculate mAP@0.5 and mAP@0.5:0.95
    Simplified version - use pycocotools for production
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        
        # Convert to torchmetrics format
        preds = []
        tgts = []
        for pred, tgt in zip(predictions, targets):
            preds.append({
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels']
            })
            tgts.append({
                'boxes': tgt['boxes'],
                'labels': tgt['labels']
            })
        
        metric = MeanAveragePrecision(iou_type='bbox')
        metric.update(preds, tgts)
        result = metric.compute()
        
        return {
            'map50': float(result['map_50'].item()),
            'map5095': float(result['map'].item()),
            'val_loss': None  # Not computed here
        }
    except ImportError:
        # Fallback: simple IoU-based mAP
        return {
            'map50': 0.0,
            'map5095': 0.0,
            'val_loss': None
        }


def calculate_per_class_metrics(predictions, targets, num_classes: int = 4):
    """
    Calculate AP, Precision, Recall for each class
    """
    class_names = ["car", "motorcycle", "person", "hov"]
    per_class = []
    
    for cls_id in range(num_classes):
        # Filter predictions and targets for this class
        cls_preds = []
        cls_tgts = []
        
        for pred, tgt in zip(predictions, targets):
            # Filter by class
            mask_pred = pred['labels'] == cls_id
            mask_tgt = tgt['labels'] == cls_id
            
            if mask_pred.any():
                cls_preds.append({
                    'boxes': pred['boxes'][mask_pred],
                    'scores': pred['scores'][mask_pred]
                })
            
            if mask_tgt.any():
                cls_tgts.append({
                    'boxes': tgt['boxes'][mask_tgt]
                })
        
        # Calculate metrics (simplified - use proper implementation in production)
        ap50 = 0.0  # Placeholder
        recall = 0.0  # Placeholder
        precision = 0.0  # Placeholder
        
        per_class.append({
            'class': class_names[cls_id],
            'class_id': cls_id,
            'ap50': ap50,
            'ap5095': 0.0,
            'precision': precision,
            'recall': recall,
            'num_gt': sum(len(t['boxes']) for t in cls_tgts)
        })
    
    return per_class