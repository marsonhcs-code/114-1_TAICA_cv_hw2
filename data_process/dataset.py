# data_process/dataset.py
# (重寫以支援 YOLO .txt 格式)

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

class DroneTrafficDataset(Dataset):
    """
    讀取 YOLO 格式的無人機交通數據集
    (每張 .png/.jpg 圖片對應一個 .txt 標註檔)
    
    支援功能：
    - YOLO format 讀取 (class,x_c,y_c,w,h) (像素座標, 逗號分隔)
    - Resize & Letterbox padding
    - Class Frequencies 計算 (for RFS)
    """
    
    def __init__(self,
                 image_dir: str,
                 image_ids: List[str], # <-- 這將是圖片檔名列表 (e.g., ["img0001.png", ...])
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
        
        self.strong_aug_for_tail = strong_aug_for_tail
        self.tail_class_ids = set(tail_class_ids)
        self.aug_cfg = augmentation_config
        
        self.class_names = class_names if class_names else []
        self.class_counts = defaultdict(int)
        
        self.samples = [] # [ (img_id, img_path, list_of_anns), ... ]
        self.image_class_map = {} # {idx: set(class_ids)}
        
        print(f"Loading YOLO-style data from: {self.image_dir}")
        
        for idx, img_id in enumerate(image_ids):
            img_path = self.image_dir / img_id
            # 標註檔的路徑 (e.g., img0001.png -> img0001.txt)
            label_path = img_path.with_suffix('.txt')

            if not img_path.exists():
                print(f"Warning: Image file not found {img_path}. Skipping.")
                continue
            if not label_path.exists():
                print(f"Warning: Label file not found {label_path}. Skipping.")
                continue

            # --- 載入標註 ---
            # anns = [ {'bbox': [x1,y1,x2,y2], 'category_id': id}, ... ]
            anns = self.parse_yolo_txt(label_path)
            
            self.samples.append((img_id, img_path, anns))
            
            # --- 計算 Class Frequencies (for RFS) ---
            classes_in_image = set()
            if self.augment: # 假設 augment=True 代表是 training set
                for ann in anns:
                    cls_id = ann['category_id']
                    self.class_counts[cls_id] += 1
                    classes_in_image.add(cls_id)
            self.image_class_map[idx] = classes_in_image

        if self.augment:
             print(f"Training set class counts (from .txt files): {dict(self.class_counts)}")
        
        if not self.class_names and self.class_counts:
             self.class_names = [str(i) for i in range(max(self.class_counts.keys()) + 1)]
        self.num_classes = len(self.class_names)

    def parse_yolo_txt(self, label_path: Path) -> List[Dict]:
        """
        解析您的 .txt 檔案: "class_id,x_center,y_center,width,height"  (像素)
        並轉換為 [x_min, y_min, x_max, y_max] (像素)
        """
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 5:
                        print(f"Warning: Skipping malformed line in {label_path}: {line}")
                        continue
                        
                    cls_id, x_c, y_c, w, h = map(float, parts)
                    
                    # 轉換 (xc, yc, w, h) -> (x1, y1, x2, y2)
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2
                    
                    annotations.append({
                        'bbox': [x1, y1, x2, y2], # 格式：xyxy (像素)
                        'category_id': int(cls_id)
                    })
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            
        return annotations

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
        
        h1, w1 = img.shape[:2]
        pad_h = (self.imgsz - h1) // 2
        pad_w = (self.imgsz - w1) // 2
        
        img_padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + h1, pad_w:pad_w + w1] = img
        
        # --- 3. 處理 Bounding Boxes ---
        boxes = []
        labels = []
        for ann in anns:
            # 'bbox' 已經是 [x1, y1, x2, y2] (像素) 格式
            box = np.array(ann['bbox'])
            cls_id = ann['category_id']
            
            # 依據 resize 和 padding 調整座標
            box = box * r
            box[[0, 2]] += pad_w # x
            box[[1, 3]] += pad_h # y
            
            # 裁剪到圖像邊界
            box[[0, 2]] = box[[0, 2]].clip(0, self.imgsz)
            box[[1, 3]] = box[[1, 3]].clip(0, self.imgsz)
            
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
                
            boxes.append(box)
            labels.append(cls_id)

        # --- 4. Augmentation (Milestone 1: 簡易) ---
        if self.augment:
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
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        
        # boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        # labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        
        # 【修改】: 先轉為單一 numpy array 再轉 tensor
        boxes_np = np.array(boxes) if boxes else np.zeros((0, 4))
        boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)

        labels_np = np.array(labels) if labels else np.zeros((0,))
        labels_tensor = torch.tensor(labels_np, dtype=torch.long) 
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': img_id,
            'orig_size': torch.tensor([h0, w0]),
            'pad_info': torch.tensor([pad_w, pad_h, r])
        }
        
        return img_tensor, target