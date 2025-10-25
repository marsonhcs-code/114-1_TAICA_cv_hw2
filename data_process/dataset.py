# data_process/dataset.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from collections import defaultdict

# 為了未來的 Milestone 4 (Augmentation) 先 import
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

class DroneTrafficDataset(Dataset):
    """
    讀取 COCO format 的無人機交通數據集
    
    支援功能：
    - COCO format 讀取
    - Resize & Letterbox padding
    - Class Frequencies 計算 (for RFS)
    - (Future) Class-aware augmentation
    """
    
    def __init__(self,
                 annotation_file: str,
                 image_dir: str,
                 image_ids: List[str],
                 imgsz: int = 640,
                 augment: bool = False,
                 class_names: Optional[List[str]] = None,
                 # Milestone 4+ params
                 strong_aug_for_tail: bool = False,
                 tail_class_ids: List[int] = [],
                 augmentation_config: Dict = {}
                 ):
        
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.image_ids = image_ids # 只載入傳入的 IDs
        
        self.strong_aug_for_tail = strong_aug_for_tail
        self.tail_class_ids = set(tail_class_ids)
        self.aug_cfg = augmentation_config
        
        # --- 載入 Annotations ---
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = {} # {image_id: [list of anns]}
        self.class_counts = defaultdict(int)
        img_id_map = {} # {coco_img_id: file_name}

        # 處理 COCO 格式
        if 'images' in data and 'annotations' in data:
            for img in data['images']:
                img_id_map[img['id']] = img['file_name']
            
            for ann in data['annotations']:
                img_file_name = img_id_map.get(ann['image_id'])
                if img_file_name not in self.annotations:
                    self.annotations[img_file_name] = []
                
                self.annotations[img_file_name].append({
                    'bbox': ann['bbox'],  # [x_min, y_min, w, h]
                    'category_id': ann['category_id']
                })
        
        # 處理 {image_id: [anns]} 格式
        else:
            self.annotations = data

        # --- 建立 Samples & 計算 Class Frequencies ---
        self.samples = [] # [ (img_id, img_path, list_of_anns), ... ]
        self.image_class_map = {} # {idx: set(class_ids)}
        
        for idx, img_id in enumerate(self.image_ids):
            if img_id not in self.annotations:
                # print(f"Warning: Image ID {img_id} not found in annotations. Skipping.")
                continue
                
            img_path = self.find_image_path(img_id)
            if img_path is None:
                # print(f"Warning: Image file for {img_id} not found in {self.image_dir}. Skipping.")
                continue

            anns = self.annotations[img_id]
            self.samples.append((img_id, img_path, anns))
            
            # 計算 Frequencies (僅限 training set)
            classes_in_image = set()
            if self.augment: # 假設 augment=True 代表是 training set
                for ann in anns:
                    cls_id = ann['category_id']
                    self.class_counts[cls_id] += 1
                    classes_in_image.add(cls_id)
            self.image_class_map[idx] = classes_in_image

        if self.augment:
             print(f"Training set class counts: {dict(self.class_counts)}")
        
        # --- Class Names ---
        if class_names:
            self.class_names = class_names
        elif 'categories' in data:
            self.class_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]
        else:
            self.class_names = [str(i) for i in range(max(self.class_counts.keys()) + 1)]
        
        self.num_classes = len(self.class_names)
        
    def find_image_path(self, img_id: str) -> Optional[Path]:
        """ 尋找圖片路徑，自動嘗試 .jpg, .png, .jpeg """
        img_path = self.image_dir / img_id
        if img_path.exists():
            return img_path
        
        stem = Path(img_id).stem
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = self.image_dir / f"{stem}{ext}"
            if img_path.exists():
                return img_path
        return None

    def __len__(self):
        return len(self.samples)

    def get_class_frequencies(self) -> Dict[int, int]:
        """ 供 RepeatFactorSampler 使用 """
        return self.class_counts

    def get_image_classes(self, idx: int) -> List[int]:
        """ 供 RepeatFactorSampler 獲取該 index 圖像中的所有 class IDs """
        return list(self.image_class_map.get(idx, []))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id, img_path, anns = self.samples[idx]
        
        # --- 1. 載入圖片 ---
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found (on getitem): {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # --- 2. Resize & Letterbox (Padding) ---
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            w1, h1 = int(w0 * r), int(h0 * r)
            img = cv2.resize(img, (w1, h1), interpolation=interp)
        
        # 計算 padding
        h1, w1 = img.shape[:2]
        pad_h = (self.imgsz - h1) // 2
        pad_w = (self.imgsz - w1) // 2
        
        img_padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + h1, pad_w:pad_w + w1] = img
        
        # --- 3. 處理 Bounding Boxes ---
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox'] # COCO [x, y, w, h]
            cls_id = ann['category_id']
            
            # 轉換為 [x_min, y_min, x_max, y_max]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            box = np.array([x1, y1, x2, y2])
            
            # 依據 resize 和 padding 調整座標
            box = box * r
            box[[0, 2]] += pad_w # x
            box[[1, 3]] += pad_h # y
            
            # 裁剪到圖像邊界
            box[[0, 2]] = box[[0, 2]].clip(0, self.imgsz)
            box[[1, 3]] = box[[1, 3]].clip(0, self.imgsz)
            
            # 忽略無效的 box
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
                
            boxes.append(box)
            labels.append(cls_id)

        # --- 4. Augmentation (Milestone 1: 簡易) ---
        # TODO (Milestone 4): 換成 Albumentations
        if self.augment:
            # 簡易左右翻轉
            if np.random.rand() < 0.5:
                img_padded = cv2.flip(img_padded, 1)
                if len(boxes) > 0:
                    boxes_np = np.array(boxes)
                    x1 = boxes_np[:, 0].copy()
                    x2 = boxes_np[:, 2].copy()
                    boxes_np[:, 0] = self.imgsz - x2
                    boxes_np[:, 2] = self.imgsz - x1
                    boxes = boxes_np.tolist()

        # --- 5. 轉換為 Tensors ---
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        
        target = {
            'boxes': boxes_tensor,  # [N, 4] (xyxy)
            'labels': labels_tensor, # [N]
            'image_id': img_id,
            'orig_size': torch.tensor([h0, w0]),
            'pad_info': torch.tensor([pad_w, pad_h, r])
        }
        
        return img_tensor, target