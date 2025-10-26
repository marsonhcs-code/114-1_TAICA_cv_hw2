import os
import shutil
import random
from pathlib import Path

# ========================= 使用者可調參 =========================
SOURCE_DIR = Path("data/CVPDL_hw2/train")  # 原本所有影像+標註都在這
OUT_ROOT   = Path("data/CVPDL_hw2_split")  # 新的 train/val 輸出根目錄

TRAIN_RATIO = 0.8       # 80% train / 20% val
SEED = 42               # 固定隨機種子，確保可重現切分
IMG_EXTS = [".jpg", ".jpeg", ".png"]  # 允許的影像副檔
YAML_PATH = Path("data/cv_hw2_data.yaml")  # 產生給 Ultralytics 用的 data.yaml

# 你的四個類別，順序要跟你訓練時一致
CLASS_NAMES = {
    0: "car",
    1: "hov",
    2: "person",
    3: "motorcycle",
}
# ===============================================================


def collect_images(src_dir):
    imgs = []
    for p in src_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return sorted(imgs)


def find_label_for_image(img_path, src_dir):
    """
    嘗試找到 YOLO 標註檔 (.txt)
    兩種常見結構：
    1. 跟影像在同一層: data/.../train/xxx.jpg + data/.../train/xxx.txt
    2. 分開兩層: data/.../train/images/xxx.jpg + data/.../train/labels/xxx.txt
       (我們做 fallback 嘗試)
    """
    stem = img_path.stem

    # Case A: same directory
    cand_same = img_path.with_suffix(".txt")
    if cand_same.exists():
        return cand_same

    # Case B: sibling "labels" directory
    # e.g. .../train/images/xxx.jpg -> .../train/labels/xxx.txt
    # 推斷路徑
    parent = img_path.parent
    if parent.name.lower() == "images":
        maybe_labels_dir = parent.parent / "labels"
        cand_labels = maybe_labels_dir / f"{stem}.txt"
        if cand_labels.exists():
            return cand_labels

    # Case C: global "labels" under SOURCE_DIR
    cand_global = src_dir / f"{stem}.txt"
    if cand_global.exists():
        return cand_global

    # 沒標註也允許回傳 None（可能該圖沒有 bbox）
    return None


def ensure_dirs(root):
    img_tr = root / "images" / "train"
    img_va = root / "images" / "val"
    lbl_tr = root / "labels" / "train"
    lbl_va = root / "labels" / "val"
    for d in [img_tr, img_va, lbl_tr, lbl_va]:
        d.mkdir(parents=True, exist_ok=True)
    return img_tr, img_va, lbl_tr, lbl_va


def write_yaml(yaml_path, out_root, class_names):
    # Ultralytics data.yaml 需要 nc, names, train, val (相對或絕對都可以)
    # 我們讓它使用相對於專案根目錄的路徑
    lines = []
    lines.append(f"path: {out_root.as_posix()}")
    lines.append(f"train: { (out_root/'images'/'train').as_posix() }")
    lines.append(f"val: { (out_root/'images'/'val').as_posix() }")
    lines.append("")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for k, v in class_names.items():
        lines.append(f"  {k}: {v}")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote data.yaml -> {yaml_path}")


def main():
    assert SOURCE_DIR.exists(), f"SOURCE_DIR not found: {SOURCE_DIR}"

    # 1. 收集全影像
    all_imgs = collect_images(SOURCE_DIR)
    if len(all_imgs) == 0:
        raise RuntimeError(f"No images found in {SOURCE_DIR}")
    print(f"[INFO] Found {len(all_imgs)} images in {SOURCE_DIR}")

    # 2. 打亂 + 切 train / val
    random.seed(SEED)
    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_train = int(n_total * TRAIN_RATIO)
    train_imgs = all_imgs[:n_train]
    val_imgs   = all_imgs[n_train:]

    print(f"[INFO] Split -> train: {len(train_imgs)}, val: {len(val_imgs)}")

    # 3. 建立輸出資料夾
    img_tr_dir, img_va_dir, lbl_tr_dir, lbl_va_dir = ensure_dirs(OUT_ROOT)

    # 4. 複製檔案
    def copy_pair(img_list, dst_img_dir, dst_lbl_dir):
        for img_path in img_list:
            stem = img_path.stem

            # 複製影像
            out_img_path = dst_img_dir / img_path.name
            shutil.copy2(img_path, out_img_path)

            # 尋找並複製標註
            label_path = find_label_for_image(img_path, SOURCE_DIR)
            out_lbl_path = dst_lbl_dir / f"{stem}.txt"

            if label_path is not None and label_path.exists():
                shutil.copy2(label_path, out_lbl_path)
            else:
                # 沒找到標註, 仍然要放一個空的 .txt
                out_lbl_path.write_text("", encoding="utf-8")

    print("[INFO] Copying train set ...")
    copy_pair(train_imgs, img_tr_dir, lbl_tr_dir)

    print("[INFO] Copying val set ...")
    copy_pair(val_imgs,   img_va_dir, lbl_va_dir)

    # 5. 產生 data.yaml
    write_yaml(YAML_PATH, OUT_ROOT, CLASS_NAMES)

    print("============================================")
    print("[DONE] YOLO train/val split complete.")
    print(f"Train images dir: {img_tr_dir}")
    print(f"Val images dir:   {img_va_dir}")
    print(f"Train labels dir: {lbl_tr_dir}")
    print(f"Val labels dir:   {lbl_va_dir}")
    print("Use data.yaml at:", YAML_PATH)
    print("============================================")


if __name__ == "__main__":
    main()
