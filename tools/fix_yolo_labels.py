from pathlib import Path
import cv2

# 路徑根據你現在的 data split
DATA_ROOT = Path("/home/n26141826/114-1_TAICA_cv_hw2/data/CVPDL_hw2_split")

IMG_TRAIN_DIR = DATA_ROOT / "images" / "train"
IMG_VAL_DIR   = DATA_ROOT / "images" / "val"
LBL_TRAIN_DIR = DATA_ROOT / "labels" / "train"
LBL_VAL_DIR   = DATA_ROOT / "labels" / "val"

def fix_split(img_dir: Path, lbl_dir: Path):
    """
    將某一個 split (train or val) 的所有 label txt 檔，
    從 'cls,x,y,w,h' (pixel,逗號分隔) 轉成
    'cls cx_norm cy_norm w_norm h_norm' (空白分隔,0~1)
    """
    assert img_dir.exists(), f"{img_dir} not found"
    assert lbl_dir.exists(), f"{lbl_dir} not found"

    txt_files = list(lbl_dir.glob("*.txt"))
    print(f"[INFO] Processing {len(txt_files)} label files under {lbl_dir}")

    for txt_path in txt_files:
        stem = txt_path.stem  # e.g. "img0579"
        # 找對應影像 (支援 .jpg/.png/.jpeg)
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break

        if img_path is None:
            print(f"[WARN] no image found for {stem}, skip")
            continue

        # 讀影像大小
        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] cannot read image {img_path}, skip")
            continue
        H, W = im.shape[:2]

        # 讀原始 label
        raw = txt_path.read_text().strip().splitlines()

        new_lines = []
        bad_line_found = False

        for line in raw:
            line = line.strip()
            if not line:
                continue

            # 支援兩種可能：
            # 1. 逗號分隔: "1,881,551,80,44"
            # 2. 空白分隔(也許你某些檔案其實已經是好的): "1 0.5 0.5 0.1 0.2"
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()

            if len(parts) != 5:
                print(f"[WARN] {txt_path.name}: unexpected line format -> {line}")
                bad_line_found = True
                continue

            cls_str, x_str, y_str, w_str, h_str = parts

            try:
                cls_id = int(cls_str)
                x = float(x_str)
                y = float(y_str)
                bw = float(w_str)
                bh = float(h_str)
            except ValueError:
                print(f"[WARN] {txt_path.name}: cannot parse numbers -> {line}")
                bad_line_found = True
                continue

            # 如果這行是 pixel 左上角座標 + w,h (像 '881,551,80,44')
            # 我們要轉成 YOLO 中心點 + normalize
            # 中心:
            cx = x + bw / 2.0
            cy = y + bh / 2.0

            # normalize 到 0~1
            cx_n = cx / W
            cy_n = cy / H
            bw_n = bw / W
            bh_n = bh / H

            # 邊界 clamp，避免浮出範圍
            cx_n = min(max(cx_n, 0.0), 1.0)
            cy_n = min(max(cy_n, 0.0), 1.0)
            bw_n = min(max(bw_n, 0.0), 1.0)
            bh_n = min(max(bh_n, 0.0), 1.0)

            # 丟掉完全沒面積的框
            if bw_n <= 0 or bh_n <= 0:
                continue

            new_lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

        # 如果整張圖沒有任何合法框，YOLO 仍然允許 label 檔是空的
        # 只要寫成空字串就好，不要刪檔
        new_txt = "\n".join(new_lines)
        txt_path.write_text(new_txt)

        if bad_line_found:
            print(f"[INFO] fixed {txt_path.name} with warnings, wrote {len(new_lines)} boxes")
        else:
            print(f"[INFO] fixed {txt_path.name}, wrote {len(new_lines)} boxes")

def main():
    fix_split(IMG_TRAIN_DIR, LBL_TRAIN_DIR)
    fix_split(IMG_VAL_DIR,   LBL_VAL_DIR)
    print("[DONE] all labels processed.")

if __name__ == "__main__":
    main()
