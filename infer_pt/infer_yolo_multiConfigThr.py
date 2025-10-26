from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ================== 推論設定 ==================

PROJ_ROOT = Path(__file__).resolve().parents[0]

TEST_IMG_DIR = PROJ_ROOT / "data" / "CVPDL_hw2" / "test"

WEIGHTS = PROJ_ROOT / "runs_yolo_baseline" / "yolov10m_from_scratch_baseline" / "weights" / "best.pt"

OUT_CSV = PROJ_ROOT / "results" / "submission_pureYOLO_2_1024_perClassThr.csv"

SAMPLE_SUB_PATH = PROJ_ROOT / "data" / "sample_submission2.csv"
USE_SAMPLE_ORDER = SAMPLE_SUB_PATH.exists()

# per-class confidence threshold
PER_CLASS_THR = {
    0: 0.40,
    1: 0.40,
    2: 0.20,
    3: 0.30,
}

IOU_THR  = 0.65
MAX_DETS = 3000

KEEP_LEADING_ZEROS_ID = False  # False => "img0002" -> "2"

CONF_DECIMALS = 6
POS_DECIMALS  = 2

SAVE_DEBUG_DIR = None  # or Path("debug_infer_vis")

INFER_IMGSZ_STRATEGY = 1024  # or "max_side"

# ======================================================

def _to_submission_id(stem: str) -> str:
    """
    將檔名轉成 leaderboard 要的 ID。
    規則：
      - 抓檔名最後一段數字，去前導零。
      - 若沒數字，就原樣回傳。
    """
    m = re.findall(r"\d+", stem)
    if not m:
        # 沒數字, 回傳原樣 (確保是字串)
        return str(stem)
    last_num = m[-1]
    try:
        return str(int(last_num))
    except ValueError:
        # 理論上不會發生，但保底
        return str(last_num)

def _imread_size(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None, None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    H, W = im.shape[:2]
    return im, H, W

def _decide_imgsz(H, W):
    if isinstance(INFER_IMGSZ_STRATEGY, int):
        return INFER_IMGSZ_STRATEGY
    return max(H, W)

def _draw_boxes_debug(img_bgr, det_xyxy, det_conf, out_path):
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), cf in zip(det_xyxy, det_conf):
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cf:.2f}",
            (x1i, max(y1i - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

def main():
    # --- 準備輸出資料夾 ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_DIR is not None:
        SAVE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading weights: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # --- 掃測試圖片 ---
    name2path = {
        p.stem: p
        for p in TEST_IMG_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    # --- 決定推論順序 ---
    if USE_SAMPLE_ORDER:
        samp = pd.read_csv(SAMPLE_SUB_PATH)
        # sample_submission["Image_ID"] 可能是 "0001", "0002"...
        # 我們仍舊要輸出純數字 (去前導零)，所以後續用 _to_submission_id
        raw_ids = [str(x) for x in samp["Image_ID"].tolist()]
        # 這邊的 stems_order 我們「先用 sample_submission 順序的 ID 字串當 key」
        # 但實際找圖時還是用 name2path，所以我們要 map 回原本實際檔名
        # 假設測試集的檔案名稱本身就長得像 sample 的 ID (或至少能還原)
        # 如果完全對不上，就 fallback 用字串比對 (通常競賽會對得上)
        stems_order = []
        for iid in raw_ids:
            sid = _to_submission_id(iid) if not KEEP_LEADING_ZEROS_ID else iid
            # sid 是像 "2"
            # 現在我們要反找原圖 stem。最穩的方式：嘗試在 name2path 裡找 match
            # 優先找完全一樣的 key
            if sid in name2path:
                stems_order.append(sid)
                continue
            # 否則找檔名中最後數字段也等於 sid 的
            hit = None
            for stem_candidate in name2path.keys():
                if _to_submission_id(stem_candidate) == sid:
                    hit = stem_candidate
                    break
            if hit is not None:
                stems_order.append(hit)
            else:
                # 沒找到對應圖，還是放一個 sid 佔位，後面會輸出空列
                stems_order.append(sid)
    else:
        # 沒 sample_submission：用檔名自然排序 (依轉成數字之後的大小)
        def _sort_key(stem):
            sid = _to_submission_id(stem)
            return (0, int(sid)) if sid.isdigit() else (1, sid)
        stems_order = sorted(name2path.keys(), key=_sort_key)

    print(f"[INFO] Total test images to infer: {len(stems_order)}")

    rows_ImageID = []
    rows_PredStr = []

    # --- 逐張推論 ---
    for stem in stems_order:
        # 轉成 leaderboard 要的 Image_ID (純數字 / 去前導零)
        if KEEP_LEADING_ZEROS_ID:
            image_id = str(stem)
        else:
            image_id = _to_submission_id(stem)
        # 保證 image_id 是字串
        image_id = str(image_id)

        img_path = name2path.get(stem, None)
        if img_path is None:
            # 沒有對應圖 -> PredictionString = "" (空字串，而不是 NaN)
            rows_ImageID.append(image_id)
            rows_PredStr.append("")
            continue

        img_bgr, H, W = _imread_size(img_path)
        if img_bgr is None:
            rows_ImageID.append(image_id)
            rows_PredStr.append("")
            continue

        infer_size = _decide_imgsz(H, W)

        # 預測：conf 設很低，之後手動用 per-class threshold 篩選
        res_list = model.predict(
            source=str(img_path),
            imgsz=infer_size,
            conf=0.001,
            iou=IOU_THR,
            max_det=MAX_DETS,
            agnostic_nms=True,
            verbose=False
        )

        parts = []
        kept_xyxy = []
        kept_conf = []

        if len(res_list) > 0:
            det = res_list[0].boxes
            if det is not None and det.xyxy is not None:
                xyxy = det.xyxy.cpu().numpy()
                conf = det.conf.cpu().numpy() if det.conf is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                cls_arr = det.cls.cpu().numpy() if det.cls is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)

                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, conf, cls_arr):
                    cls_id_int = int(cls_id)

                    # per-class threshold
                    thr = PER_CLASS_THR.get(cls_id_int, 0.5)
                    if c < thr:
                        continue

                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    if w <= 1 or h <= 1:
                        continue

                    x1 = float(np.clip(x1, 0, W - 1))
                    y1 = float(np.clip(y1, 0, H - 1))
                    w  = float(np.clip(w,  0, W - x1))
                    h  = float(np.clip(h,  0, H - y1))
                    if w <= 1 or h <= 1:
                        continue

                    out_cls = cls_id_int  # 如果只允許單一 class，可在此改成常數

                    parts.extend([
                        f"{float(c):.{CONF_DECIMALS}f}",
                        str(round(x1, POS_DECIMALS)),
                        str(round(y1, POS_DECIMALS)),
                        str(round(w,  POS_DECIMALS)),
                        str(round(h,  POS_DECIMALS)),
                        str(out_cls),
                    ])

                    kept_xyxy.append([x1, y1, x1 + w, y1 + h])
                    kept_conf.append(c)

        # PredictionString（可能是空字串）
        if len(parts) == 0:
            # 沒有偵測到東西，給一個安全佔位，避免空字串/NaN
            # 格式: "conf x y w h class"
            # 我們用 conf=0, bbox=0 0 0 0, class=0
            pred_str = "0 0 0 0 0 0"
        else:
            pred_str = " ".join(parts)

        # 存起來，**務必是字串，不要 None**
        rows_ImageID.append(image_id)
        rows_PredStr.append(pred_str)

        # optional debug可視化
        if SAVE_DEBUG_DIR is not None and len(kept_xyxy) > 0:
            _draw_boxes_debug(
                img_bgr,
                kept_xyxy,
                kept_conf,
                SAVE_DEBUG_DIR / f"{image_id}.jpg"
            )

    # --- 轉 DataFrame ---
    df = pd.DataFrame({
        "Image_ID": rows_ImageID,
        "PredictionString": rows_PredStr
    })

    # 最後排序：Image_ID=1,2,3,... (字串也行)
    def _final_sort_key(v):
        vs = str(v)
        if vs.isdigit():
            return (0, int(vs))
        return (1, vs)

    df = df.sort_values(
        by="Image_ID",
        key=lambda col: col.map(_final_sort_key)
    ).reset_index(drop=True)

    # **保證沒有 NaN**
    df = df.fillna("")

    # double check
    assert not df["Image_ID"].isna().any(), "Image_ID 有 NaN"
    assert not df["PredictionString"].isna().any(), "PredictionString 有 NaN"

    df.to_csv(OUT_CSV, index=False)
    print("[DONE] wrote:", OUT_CSV, "rows(images)=", len(df))
    print(df.head())


if __name__ == "__main__":
    main()
