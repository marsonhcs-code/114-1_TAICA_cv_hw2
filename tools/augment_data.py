import os
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

PROJ_ROOT  = Path(__file__).resolve().parents[1]
IMG_DIR = r"data/CVPDL_hw2_split/images/train"
LBL_DIR = r"data/CVPDL_hw2_split/labels/train"

# 你定義哪些是 tail classes (請改成你的少數類別ID)
TAIL_CLASSES = {2, 3}

# 每張 tail 圖要產生幾張增強版
DUP_FACTOR = 2  # 建議先 2~3，不要太爆

def has_tail_class(label_file):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in TAIL_CLASSES:
                return True
    return False

def random_brightness_contrast(img):
    # alpha = 對比 (1.0 ~ 1.4)
    alpha = 1.0 + 0.4 * (random.random() - 0.5) * 2  # roughly 0.6~1.4
    # beta  = 亮度 (-30 ~ +30)
    beta = random.uniform(-30, 30)
    img = img.astype(np.float32) * alpha + beta
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_color_jitter(img):
    # HSV 抖動：稍微改顏色，不旋轉到整個變奇怪
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift (-10~+10), Saturation scale (0.8~1.2), Value scale (0.8~1.2)
    h_shift = random.uniform(-10, 10)
    s_scale = random.uniform(0.8, 1.2)
    v_scale = random.uniform(0.8, 1.2)

    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)

    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def random_gaussian_noise(img):
    # 給整張圖加一點點高斯雜訊
    if random.random() < 0.5:
        return img
    h, w, c = img.shape
    noise = np.random.normal(0, 10, (h, w, c)).astype(np.float32)  # std=10
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def random_blur(img):
    # 有時候加一點模糊 (模擬運動模糊/對焦不準)
    if random.random() < 0.5:
        return img
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)

def random_cutout(img):
    # 用一個黑/灰色方塊遮一小塊區域，模擬遮擋
    if random.random() < 0.5:
        return img
    h, w, _ = img.shape
    box_w = random.randint(w // 20, w // 8)   # 小區塊
    box_h = random.randint(h // 20, h // 8)
    x1 = random.randint(0, w - box_w)
    y1 = random.randint(0, h - box_h)
    color = random.randint(0, 50)  # 接近黑 / 暗灰
    img[y1:y1+box_h, x1:x1+box_w] = color
    return img

def augment_once(img):
    # pipe: brightness/contrast -> color jitter -> noise -> blur -> cutout
    out = img.copy()
    out = random_brightness_contrast(out)
    out = random_color_jitter(out)
    out = random_gaussian_noise(out)
    out = random_blur(out)
    out = random_cutout(out)
    return out

# 掃描所有 label
for lbl_name in os.listdir(LBL_DIR):
    if not lbl_name.endswith(".txt"):
        continue

    lbl_path = os.path.join(LBL_DIR, lbl_name)
    img_base = os.path.splitext(lbl_name)[0]

    # 找對應影像 (支援 .jpg/.jpeg/.png)
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(IMG_DIR, img_base + ext)
        if os.path.exists(p):
            img_path = p
            img_ext = ext
            break
    if img_path is None:
        continue  # 沒找到就跳過

    if not has_tail_class(lbl_path):
        continue  # 只有含 tail class 的圖片才 oversample

    # 讀原圖
    img = cv2.imread(img_path)
    if img is None:
        continue

    # 產生多份增強副本
    for k in range(1, DUP_FACTOR + 1):
        aug_img = augment_once(img)

        new_img_name = f"{img_base}_dup{k}{img_ext}"
        new_lbl_name = f"{img_base}_dup{k}.txt"

        new_img_path = os.path.join(IMG_DIR, new_img_name)
        new_lbl_path = os.path.join(LBL_DIR, new_lbl_name)

        # 存增強後的影像
        cv2.imwrite(new_img_path, aug_img)

        # label 直接複製
        shutil.copyfile(lbl_path, new_lbl_path)

        print(f"[AUG DUP] {img_path} -> {new_img_path}")
