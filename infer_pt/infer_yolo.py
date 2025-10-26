from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ================== 推論設定 (可調參) ==================

PROJ_ROOT = Path(__file__).resolve().parents[0]

TEST_IMG_DIR = PROJ_ROOT / "data" / "CVPDL_hw2" / "test"

WEIGHTS = PROJ_ROOT / "runs_yolo_baseline" / "yolov10m_12g_highres" / "weights" / "best.pt"

OUT_CSV = PROJ_ROOT / "results" / "submission_pureYOLO_limit_1536_0.03confThr.csv"

SAMPLE_SUB_PATH = PROJ_ROOT / "data" / "sample_submission2.csv"
USE_SAMPLE_ORDER = SAMPLE_SUB_PATH.exists()

CONF_THR = 0.03
IOU_THR  = 0.65
MAX_DETS = 3000

KEEP_LEADING_ZEROS_ID = False

CONF_DECIMALS = 6
POS_DECIMALS  = 2

CLASS_ID_OUT = 0

SAVE_DEBUG_DIR = None  # or Path("debug_infer_vis")

INFER_IMGSZ_STRATEGY = 2048  # "max_side"  # or int like 1024

# ======================================================

def _strip_num(s: str) -> str:
    """
    "000123" -> "123"
    "17"     -> "17"
    "img5"   -> "img5" (因為不是純數字，直接回傳原樣)
    """
    return str(int(str(s))) if re.fullmatch(r"\d+", str(s)) else str(s)


def _to_submission_id(stem: str) -> str:
    """
    目標：回傳 leaderboard 要的 Image_ID (純數字字串)

    規則：
    1. 找出檔名裡「最後一段連續數字」。
       例如 "img0002" -> "0002", "abc123def45" -> "45"
    2. 去掉前導零 -> "2", "45"
    3. 如果整個檔名本來就是數字 (e.g. "000123"), 也會變 "123".
    4. 如果完全找不到數字，就 fallback 回原 stem。

    這讓我們可以把 "img0002" 輸出成 "2"。
    """
    m = re.findall(r"\d+", stem)
    if not m:
        return stem  # 沒有數字，只能原樣
    last_num = m[-1]  # 最後一段數字
    # 去前導零
    try:
        return str(int(last_num))
    except ValueError:
        # 理論上不會進來，保險一下
        return last_num


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
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_DIR is not None:
        SAVE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading weights: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # 建立測試圖清單
    name2path = {
        p.stem: p
        for p in TEST_IMG_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    if USE_SAMPLE_ORDER:
        samp = pd.read_csv(SAMPLE_SUB_PATH)
        id_list = [str(x) for x in samp["Image_ID"].tolist()]

        # 如果 sample 裡 "Image_ID" 其實已經是數字（或帶前導零），
        # 這段邏輯會嘗試把它轉成 leaderboard 需要的對應樣式。
        stems_order = [
            (_to_submission_id(iid) if not KEEP_LEADING_ZEROS_ID else iid)
            for iid in id_list
        ]
    else:
        # 沒有 sample_submission，就用檔名排序。
        # 我們要的是「按數字大小」的排序，跟比賽通常一樣。
        def _sort_key(stem):
            sid = _to_submission_id(stem)
            # 如果 sid 是純數字，就用數字排序，否則用字串排序
            return (0, int(sid)) if sid.isdigit() else (1, sid)
        stems_order = sorted(name2path.keys(), key=_sort_key)

    print(f"[INFO] Total test images: {len(stems_order)}")

    output_rows = []

    for stem in stems_order:
        # 產生最後要寫進 CSV 的 Image_ID
        #   e.g. "img0002" -> "2"
        #   e.g. "000123" -> "123"
        if KEEP_LEADING_ZEROS_ID:
            image_id = stem
        else:
            image_id = _to_submission_id(stem)

        img_path = name2path.get(stem, None)
        if img_path is None:
            # sample_submission 裡有 ID 但資料夾沒有圖
            output_rows.append((image_id, ""))
            continue

        img_bgr, H, W = _imread_size(img_path)
        if img_bgr is None:
            output_rows.append((image_id, ""))
            continue

        infer_size = _decide_imgsz(H, W)

        res_list = model.predict(
            source=str(img_path),
            imgsz=infer_size,
            conf=CONF_THR,
            iou=IOU_THR,
            max_det=MAX_DETS,
            agnostic_nms=True,
            verbose=False
        )

        parts = []
        if len(res_list) > 0:
            det = res_list[0].boxes
            if det is not None and det.xyxy is not None:
                xyxy = det.xyxy.cpu().numpy()
                conf = (
                    det.conf.cpu().numpy()
                    if det.conf is not None
                    else np.zeros((xyxy.shape[0],), dtype=np.float32)
                )
                cls_arr = (
                    det.cls.cpu().numpy()
                    if det.cls is not None
                    else np.zeros((xyxy.shape[0],), dtype=np.float32)
                )

                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, conf, cls_arr):
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

                    # class 輸出
                    out_cls = int(cls_id)
                    # 如果 leaderboard 強制一個 class:
                    # out_cls = int(CLASS_ID_OUT)

                    parts.extend([
                        f"{float(c):.{CONF_DECIMALS}f}",
                        str(round(x1, POS_DECIMALS)),
                        str(round(y1, POS_DECIMALS)),
                        str(round(w,  POS_DECIMALS)),
                        str(round(h,  POS_DECIMALS)),
                        str(out_cls),
                    ])

                if SAVE_DEBUG_DIR is not None and len(xyxy) > 0:
                    _draw_boxes_debug(
                        img_bgr,
                        xyxy,
                        conf,
                        SAVE_DEBUG_DIR / f"{image_id}.jpg"
                    )

        if len(parts) == 0:
            # 沒有偵測到東西，給一個安全佔位，避免空字串/NaN
            # 格式: "conf x y w h class"
            # 我們用 conf=0, bbox=0 0 0 0, class=0
            pred_str = "0 0 0 0 0 0"
        else:
            pred_str = " ".join(parts)
        output_rows.append((image_id, pred_str))

    # 轉成 DataFrame
    df = pd.DataFrame(output_rows, columns=["Image_ID", "PredictionString"])

    # 重新排序，確保 "1,2,3,..."
    def _final_sort_key(v):
        # v 是 Image_ID，例如 "2"
        # 能轉 int 就用 int，不能就 fallback 字串
        if str(v).isdigit():
            return (0, int(v))
        return (1, str(v))

    df = (
        df.sort_values(by="Image_ID", key=lambda col: col.map(_final_sort_key))
          .reset_index(drop=True)
    )

    df.to_csv(OUT_CSV, index=False)
    print("[DONE] wrote:", OUT_CSV, "rows(images)=", len(df))

if __name__ == "__main__":
    main()
