from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import openpyxl

# ================== 基本設定 ==================
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJ_ROOT / "data" / "cv_hw2_data1.yaml"
DEVICE = "0,1"
WEIGHTS = PROJ_ROOT / "runs_yolo_exp" / "yolov10s_exp" / "weights" / "best.pt"
# 推論與驗證資料路徑
TEST_IMG_DIR = PROJ_ROOT / "data" / "CVPDL_hw2" / "test"
VAL_IMG_DIR  = PROJ_ROOT / "data" / "CVPDL_hw2_split" / "images" / "val"
VAL_LABEL_DIR = VAL_IMG_DIR.parent.parent / "labels" / "val"  # 預設對應的 labels/val

# 輸出結果
RESULTS_DIR = PROJ_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "submission_perClassThr.csv"
EVAL_OUT_CSV = RESULTS_DIR / "evaluation_metrics.csv"
SAVE_TEST_DIR = PROJ_ROOT / "debug_test_combined"
SAVE_VAL_DIR  = PROJ_ROOT / "debug_val_combined"
SAVE_TEST_DIR.mkdir(parents=True, exist_ok=True)
SAVE_VAL_DIR.mkdir(parents=True, exist_ok=True)

# ================== 推論參數 ==================

CLASS_NAMES = {0: "car", 1: "hov", 2: "p", 3: "m"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0)}

IOU_THR  = 0.001
MAX_DETS = 300
CONF_DECIMALS = 6
POS_DECIMALS  = 2
INFER_IMGSZ   = 1920  # "max_side"  # or int like 1024
KEEP_LEADING_ZEROS_ID = False
SAVE_EVERY = 13  # 每隔多少張輸出一張組合圖

# ========== threshold 最佳化設定 ==========
# 初始 per-class threshold（若 opt_mode='none' 就使用這個；若最佳化則會覆蓋）
# PER_CLASS_THR: Dict[int, float] = {0: 0.3, 1: 0.35, 2: 0.2, 3: 0.23}
# PER_CLASS_IOU: Dict[int, float] = {0: 0.00001, 1: 0.00001, 2: 0.00001, 3: 0.00001}
PER_CLASS_THR = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01,}

PER_CLASS_IOU = {0: 0.45,1: 0.45,2: 0.30,3: 0.30,}

CONF_OPT_MODE = "per_class"       # 可選: "none" | "global" | "per_class"   
IoU_OPT_MODE = "none"   # 可選: "none" | "global" | "per_class"       

GLOBAL_CONF_CANDIDATES = [0.01, 0.05, 0.1, 0.2, 0.3]
GLOBAL_IOU_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

PER_CLASS_CONF_CANDIDATES = [0.01, 0.13, 0.15, 0.17, 0.2, 0.3, 0.4]  # 每類別掃描的候選
PER_CLASS_IOU_CANDIDATES = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.4, 0.5]  # 每類別掃描的候選
# PER_CLASS_IOU_CANDIDATES = [0.0000001, 0.000001,0.00001, 0.0001,]  # 每類別掃描的候選

# ======================================================
# 工具函式
# ======================================================
def _to_submission_id(stem: str) -> str:
    m = re.findall(r"\d+", stem)
    if not m:
        return str(stem)
    try:
        return str(int(m[-1]))  # 去前導零
    except ValueError:
        return str(m[-1])

def _imread(p: Path) -> Optional[np.ndarray]:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return im

def _draw_boxes(img_bgr: np.ndarray,
                boxes_xyxy: np.ndarray,
                conf: Optional[np.ndarray],
                cls_ids: np.ndarray,
                title: str,
                colors: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        cid = int(cls_ids[i])
        color = colors.get(cid, (200, 200, 200))
        txt = CLASS_NAMES.get(cid, str(cid))
        if conf is not None:
            txt = f"{txt} {float(conf[i]):.2f}"
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, txt, (int(x1), max(int(y1)-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if title:
        cv2.putText(vis, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)
    return vis

def _concat_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    def _resize_h(im):
        if im.shape[0] == h:
            return im
        scale = h / im.shape[0]
        w = int(round(im.shape[1]*scale))
        return cv2.resize(im, (w, h))
    L = _resize_h(left)
    R = _resize_h(right)
    return np.hstack([L, R])

def _xywhn_to_xyxy(img_shape: Tuple[int,int], labels: np.ndarray) -> np.ndarray:
    """
    將 YOLO txt 的 (cls, x, y, w, h) (normalized) 轉為 xyxy (像素)
    labels: [N,5] -> cls, x, y, w, h (normalized)
    """
    H, W = img_shape[:2]
    xyxy = []
    for r in labels:
        _, x, y, w, h = r
        cx, cy = x*W, y*H
        bw, bh = w*W, h*H
        x1, y1 = cx - bw/2, cy - bh/2
        x2, y2 = cx + bw/2, cy + bh/2
        xyxy.append([x1, y1, x2, y2])
    return np.array(xyxy, dtype=np.float32)

def _load_yolo_txt(txt_path: Path) -> np.ndarray:
    """
    讀取 YOLO label txt -> ndarray [N, 5] (cls, cx, cy, w, h) normalized
    若檔案不存在或空，回傳 shape (0,5)
    """
    if not txt_path.exists():
        return np.zeros((0,5), dtype=np.float32)
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            ss = line.strip().split()
            if len(ss) != 5:
                continue
            c, x, y, w, h = map(float, ss)
            rows.append([c, x, y, w, h])
    if not rows:
        return np.zeros((0,5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

def _iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    a: [Na, 4], b: [Nb, 4] -> IoU [Na, Nb]
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)
    a = a_xyxy
    b = b_xyxy
    Na, Nb = a.shape[0], b.shape[0]
    iou = np.zeros((Na, Nb), dtype=np.float32)
    for i in range(Na):
        x11,y11,x12,y12 = a[i]
        a_area = max(0.0, x12-x11) * max(0.0, y12-y11)
        for j in range(Nb):
            x21,y21,x22,y22 = b[j]
            inter_x1 = max(x11, x21)
            inter_y1 = max(y11, y21)
            inter_x2 = min(x12, x22)
            inter_y2 = min(y12, y22)
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            b_area = max(0.0, x22-x21) * max(0.0, y22-y21)
            union = a_area + b_area - inter
            iou[i, j] = 0.0 if union <= 0 else inter / union
    return iou

def _match_tp_fp_fn(gt_xyxy: np.ndarray, gt_cls: np.ndarray,
                    pr_xyxy: np.ndarray, pr_cls: np.ndarray,
                    iou_thr: float = 0.5) -> Tuple[int,int,int]:
    """
    在單張影像上做 greedy matching（同類別之間配對），回傳 TP, FP, FN
    """
    TP = 0
    used_pred = set()
    used_gt   = set()
    # 依類別各自配對
    classes = np.unique(np.concatenate([gt_cls, pr_cls]).astype(int)) if (gt_cls.size or pr_cls.size) else []
    for c in classes:
        gi = np.where(gt_cls == c)[0]
        pi = np.where(pr_cls == c)[0]
        if gi.size == 0 and pi.size == 0:
            continue
        A = gt_xyxy[gi] if gi.size else np.zeros((0,4), dtype=np.float32)
        B = pr_xyxy[pi] if pi.size else np.zeros((0,4), dtype=np.float32)
        iou = _iou_matrix(A, B)
        # greedy: 從最大 IoU 開始配對
        while True:
            if iou.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou, axis=None), iou.shape)
            best = iou[idx]
            if best < iou_thr:
                break
            g_local, p_local = int(idx[0]), int(idx[1])
            g_idx, p_idx = int(gi[g_local]), int(pi[p_local])
            if (g_idx in used_gt) or (p_idx in used_pred):
                iou[g_local, p_local] = -1.0
                continue
            TP += 1
            used_gt.add(g_idx)
            used_pred.add(p_idx)
            iou[g_local, :] = -1.0
            iou[:, p_local] = -1.0
    FP = int(pr_cls.size) - len(used_pred)
    FN = int(gt_cls.size) - len(used_gt)
    return TP, FP, FN

# ======================================================
# (A) 推論 Test → 產生 CSV，並每 20 張存可視化
# ======================================================
def infer_test(model: YOLO, per_class_thr: Dict[int, float]):
    print(f"[INFO] Inference on TEST ({TEST_IMG_DIR})...")
    image_paths = [p for p in sorted(TEST_IMG_DIR.iterdir()) if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    rows = []

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        img = _imread(img_path)
        if img is None:
            rows.append((_to_submission_id(stem), ""))
            continue

        res_list = model.predict(
            source=str(img_path),
            imgsz=INFER_IMGSZ,
            conf=0.001,      # 超低，後面自己用 per-class thr 過濾
            iou=IOU_THR,
            max_det=MAX_DETS,
            agnostic_nms=True,
            verbose=False,
            device=DEVICE
        )

        parts = []
        kept_xyxy, kept_conf, kept_cls = [], [], []
        if len(res_list) > 0 and res_list[0].boxes is not None:
            det = res_list[0].boxes
            xyxy = det.xyxy.cpu().numpy()
            conf = det.conf.cpu().numpy()
            cls_arr = det.cls.cpu().numpy()
            for (x1,y1,x2,y2), c, cid in zip(xyxy, conf, cls_arr):
                thr = per_class_thr.get(int(cid), 0.3)
                if c < thr: 
                    continue
                w, h = x2-x1, y2-y1
                if w <= 1 or h <= 1:
                    continue
                parts.extend([
                    f"{c:.{CONF_DECIMALS}f}",
                    str(round(x1, POS_DECIMALS)),
                    str(round(y1, POS_DECIMALS)),
                    str(round(w,  POS_DECIMALS)),
                    str(round(h,  POS_DECIMALS)),
                    str(int(cid))
                ])
                kept_xyxy.append([x1,y1,x2,y2])
                kept_conf.append(c)
                kept_cls.append(cid)

        pred_str = " ".join(parts) if parts else "0 0 0 0 0 0"
        rows.append((_to_submission_id(stem), pred_str))

        # 每 SAVE_EVERY 張輸出 side-by-side（原圖 | Pred）
        if (idx % SAVE_EVERY == 0) and len(kept_xyxy) > 0:
            kept_xyxy = np.array(kept_xyxy, dtype=np.float32)
            kept_conf = np.array(kept_conf, dtype=np.float32)
            kept_cls  = np.array(kept_cls,  dtype=np.float32)
            pred_vis = _draw_boxes(img, kept_xyxy, kept_conf, kept_cls, "Pred", CLASS_COLORS)
            combined = _concat_side_by_side(img, pred_vis)
            cv2.imwrite(str(SAVE_TEST_DIR / f"{stem}_test_combined.jpg"), combined)

    df = pd.DataFrame(rows, columns=["Image_ID", "PredictionString"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] Test CSV -> {OUT_CSV}  (images={len(rows)})")
# ======== 新增小工具函式 ========
def _compute_iou(box1, box2):
    """計算兩個框的 IoU"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms_by_iou(boxes, scores, iou_thr):
    """根據 IoU 做 per-class NMS，返回保留索引"""
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]  # 按分數降序
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        ious = np.array([_compute_iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious <= iou_thr]  # 移除 IoU 過高的框
    return keep


# ======== 主函式 ========
def infer(model: YOLO,
          img_dir: Path,
          save_dir: Path,
          per_class_thr: dict[int, float],
          per_class_iou: dict[int, float] | float = 0.5,
          label_dir: Optional[Path] = None,
          save_csv: Optional[Path] = None):
    """
    ✅ 通用推論函式（可用於 test 或 val）
    - 回傳值修正為: (preds, gts) 或 rows
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Inference on {img_dir} ... (Labels: {'Yes' if label_dir else 'No'})")

    image_paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    rows = []
    preds = []  # ✅ 新增：儲存預測結果
    gts = []    # ✅ 新增：儲存 GT

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        img = _imread(img_path)
        if img is None:
            rows.append((_to_submission_id(stem), ""))
            if label_dir:
                preds.append((np.array([]), np.array([])))
                gts.append((np.array([]), np.array([])))
            continue

        # === YOLO 推論 ===
        res_list = model.predict(
            source=str(img_path),
            imgsz=INFER_IMGSZ,
            conf=0.0001,
            iou=IOU_THR,  # ⚠️ 這裡的 iou 是 YOLO 內建的 NMS，與後面的 per_class_iou 不同
            max_det=MAX_DETS,
            agnostic_nms=True,
            verbose=False,
            device=DEVICE
        )

        kept_xyxy, kept_conf, kept_cls = [], [], []
        if len(res_list) > 0 and res_list[0].boxes is not None:
            det = res_list[0].boxes
            xyxy = det.xyxy.cpu().numpy()
            conf = det.conf.cpu().numpy()
            cls_arr = det.cls.cpu().numpy()

            # === ✅ 修正：per-class 過濾 + NMS ===
            for cid in np.unique(cls_arr):
                cls_mask = cls_arr == cid
                cls_boxes = xyxy[cls_mask]
                cls_conf = conf[cls_mask]
                
                # 1. confidence 過濾
                thr = per_class_thr.get(int(cid), 0.3)
                valid_idx = cls_conf >= thr
                cls_boxes, cls_conf = cls_boxes[valid_idx], cls_conf[valid_idx]
                if len(cls_boxes) == 0:
                    continue

                # 2. per-class IoU NMS
                iou_thr = per_class_iou.get(int(cid), 0.5) if isinstance(per_class_iou, dict) else per_class_iou
                keep_idx = _nms_by_iou(cls_boxes, cls_conf, iou_thr)
                cls_boxes = cls_boxes[keep_idx]
                cls_conf = cls_conf[keep_idx]

                # 3. 過濾過小的框
                for b, c in zip(cls_boxes, cls_conf):
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    if w <= 1 or h <= 1:
                        continue
                    kept_xyxy.append([x1, y1, x2, y2])
                    kept_conf.append(c)
                    kept_cls.append(cid)

        # === 組合輸出字串（用於 CSV）===
        parts = []
        for (x1, y1, x2, y2), c, cid in zip(kept_xyxy, kept_conf, kept_cls):
            parts.extend([
                f"{c:.{CONF_DECIMALS}f}",
                str(round(x1, POS_DECIMALS)),
                str(round(y1, POS_DECIMALS)),
                str(round(x2 - x1, POS_DECIMALS)),
                str(round(y2 - y1, POS_DECIMALS)),
                str(int(cid))
            ])
        pred_str = " ".join(parts) if parts else "0 0 0 0 0 0"
        rows.append((_to_submission_id(stem), pred_str))

        # === ✅ 修正：儲存 pred 和 gt 用於評估 ===
        if label_dir:
            label_path = label_dir / f"{stem}.txt"
            gt_lbl = _load_yolo_txt(label_path)
            if gt_lbl.size:
                gt_cls = gt_lbl[:, 0].astype(int)
                gt_xyxy = _xywhn_to_xyxy(img.shape, gt_lbl)
            else:
                gt_cls = np.array([])
                gt_xyxy = np.array([]).reshape(0, 4)
            
            gts.append((gt_xyxy, gt_cls))
            preds.append((np.array(kept_xyxy), np.array(kept_cls)))

        # === 視覺化（每 SAVE_EVERY 張）===
        if (idx % SAVE_EVERY == 0):
            if label_dir:  # 有 GT：畫 GT|Pred
                label_path = label_dir / f"{stem}.txt"
                gt_lbl = _load_yolo_txt(label_path)
                if gt_lbl.size:
                    gt_cls = gt_lbl[:, 0].astype(int)
                    gt_xyxy = _xywhn_to_xyxy(img.shape, gt_lbl)
                    gt_vis = _draw_boxes(img, gt_xyxy, None, gt_cls, "GT", CLASS_COLORS)
                else:
                    gt_vis = img.copy()

                if kept_xyxy:
                    pred_vis = _draw_boxes(img, np.array(kept_xyxy), np.array(kept_conf), 
                                          np.array(kept_cls), "Pred", CLASS_COLORS)
                else:
                    pred_vis = img.copy()

                combined = _concat_side_by_side(gt_vis, pred_vis)
                cv2.imwrite(str(save_dir / f"{stem}_combined.jpg"), combined)
            
            elif kept_xyxy:  # 無 GT：只畫 Pred
                pred_vis = _draw_boxes(img, np.array(kept_xyxy), np.array(kept_conf),
                                      np.array(kept_cls), "Pred", CLASS_COLORS)
                cv2.imwrite(str(save_dir / f"{stem}_pred.jpg"), pred_vis)

    # === ✅ 修正：根據是否有 label_dir 回傳不同結果 ===
    if save_csv:
        df = pd.DataFrame(rows, columns=["Image_ID", "PredictionString"])
        df.to_csv(save_csv, index=False)
        print(f"[DONE] CSV saved -> {save_csv}")

    print(f"[DONE] Inference complete. ({len(rows)} images)")
    
    if label_dir:
        return preds, gts  # ✅ 用於評估
    else:
        return rows  # ✅ 用於產生 submission
    
# ======================================================
# (B) 推論 Val → 計算指標，並每 20 張存 GT|Pred 組合圖
# ======================================================
def evaluate_val(model: YOLO,
                 per_class_thr: Dict[int, float],
                 per_class_iou: Dict[int, float] | float = IOU_THR) -> Dict[str, float]:
    """
    ✅ 修正版：正確接收 infer() 的回傳值
    """
    print(f"\n[INFO] Step 1: Inference + GT alignment")
    
    # ✅ 修正：infer 現在回傳 (preds, gts)
    preds, gts = infer(
        model=model,
        img_dir=VAL_IMG_DIR,
        save_dir=SAVE_VAL_DIR,
        per_class_thr=per_class_thr,
        per_class_iou=per_class_iou,
        label_dir=VAL_LABEL_DIR,  # 有 label_dir 才會回傳 (preds, gts)
        save_csv=None
    )

    # === 計算 Precision/Recall/F1 ===
    total_TP = total_FP = total_FN = 0
    per_class_stats = {}

    for (pred_boxes, pred_cls), (gt_boxes, gt_cls) in zip(preds, gts):
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        
        # ✅ 確保是 numpy array
        pred_boxes = np.array(pred_boxes) if not isinstance(pred_boxes, np.ndarray) else pred_boxes
        pred_cls = np.array(pred_cls) if not isinstance(pred_cls, np.ndarray) else pred_cls
        
        all_cls = np.concatenate([gt_cls, pred_cls]) if (len(gt_cls) or len(pred_cls)) else np.array([])
        for c in np.unique(all_cls).astype(int):
            gi = gt_cls == c
            pi = pred_cls == c
            c_iou = per_class_iou.get(int(c), 0.5) if isinstance(per_class_iou, dict) else per_class_iou
            
            cTP, cFP, cFN = _match_tp_fp_fn(
                gt_boxes[gi], gt_cls[gi],
                pred_boxes[pi], pred_cls[pi],
                iou_thr=c_iou
            )
            
            total_TP += cTP
            total_FP += cFP
            total_FN += cFN
            per_class_stats.setdefault(c, [0, 0, 0])
            per_class_stats[c][0] += cTP
            per_class_stats[c][1] += cFP
            per_class_stats[c][2] += cFN

    precision = total_TP / (total_TP + total_FP + 1e-12)
    recall = total_TP / (total_TP + total_FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # === 取 YOLO 官方的 mAP ===
    print(f"\n[INFO] Step 2: Running model.val() ...")
    metrics = model.val(data=str(DATA_YAML), imgsz=INFER_IMGSZ, device=DEVICE)
    map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map5095 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))

    # === 整理 per-class 結果 ===
    rows = []
    for c, (cTP, cFP, cFN) in per_class_stats.items():
        p = cTP / (cTP + cFP + 1e-12)
        r = cTP / (cTP + cFN + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)
        rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES.get(c, str(c)),
            "TP": cTP, 
            "FP": cFP, 
            "FN": cFN,
            "Precision": p, 
            "Recall": r, 
            "F1": f,
            "conf_thr": per_class_thr.get(c, None),
            "iou_thr": per_class_iou.get(c, 0.5) if isinstance(per_class_iou, dict) else per_class_iou
        })

    per_class_df = pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)
    summary_df = pd.DataFrame([{
        "TP": total_TP, 
        "FP": total_FP, 
        "FN": total_FN,
        "Precision": precision, 
        "Recall": recall, 
        "F1": f1,
        "mAP50": map50, 
        "mAP50-95": map5095
    }])

    # === 儲存結果到 Excel ===
    excel_path = EVAL_OUT_CSV.with_suffix(".xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as w:
        summary_df.to_excel(w, index=False, sheet_name="summary")
        per_class_df.to_excel(w, index=False, sheet_name="per_class")

    # === 終端輸出結果 ===
    print("\n" + "="*80)
    print("[VAL SUMMARY]")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("[VAL PER-CLASS]")
    print("="*80)
    print(per_class_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"[DONE] Evaluation results saved to: {excel_path}")
    print("="*80 + "\n")
    
    return {"Precision": precision, "Recall": recall, "F1": f1,
            "mAP50": map50, "mAP50-95": map5095}
# ======================================================
# (C) confidence threshold 最佳化
# ======================================================
def optimize_thresholds(model: YOLO,
                        mode: str = "global",
                        base_thr: Dict[int, float] = None,
                        static_iou: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    static_iou = static_iou or PER_CLASS_IOU

    if mode == "none":
        return dict(base_thr or PER_CLASS_THR)

    print(f"\n[INFO] Optimizing confidence thresholds, mode = {mode}")
    base = dict(base_thr or PER_CLASS_THR)

    # ---------- 全域模式 ----------
    if mode == "global":
        best_f1, best_thr = -1.0, None
        for t in GLOBAL_CONF_CANDIDATES:
            thr = {c: t for c in CLASS_NAMES.keys()}
            print(f"  >> Testing global conf={t:.2f} (static IoU per class)")
            res = evaluate_val(model, thr, iou_thr=static_iou)
            f1 = res["F1"]
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        final = {c: best_thr for c in CLASS_NAMES.keys()}
        print(f"[OPT] Best global conf={best_thr:.2f} → F1={best_f1:.4f}")
        return final

    # ---------- 每類別模式 ----------
    if mode == "per_class":
        final = dict(base)
        for c in CLASS_NAMES.keys():
            best_f1, best_t = -1.0, None
            print(f"\n[CLASS {c}] {CLASS_NAMES[c]} conf sweep (fixed IoU={static_iou.get(c,0.5):.2f})...")
            for t in PER_CLASS_CONF_CANDIDATES:
                cand = dict(final)
                cand[c] = t
                res = evaluate_val(model, cand, static_iou)
                f1 = res["F1"]
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            final[c] = best_t
            print(f"[OPT] class {c} ({CLASS_NAMES[c]}): best conf={best_t:.2f}, F1={best_f1:.4f}")
        print(f"\n[OPT] Final per-class confidence thresholds: {final}")
        return final

    print("[WARN] Unknown confidence optimization mode, using base thresholds.")
    return dict(base)


# ======================================================
# (D) IoU threshold 優化控制函式
# ======================================================
def optimize_iou_thresholds(model: YOLO,
                            per_class_thr: Dict[int, float],
                            mode: str = "none",
                            static_iou: Optional[Dict[int, float]] = None,
                            iou_candidates: Optional[List[float]] = None) -> Dict[int, float]:
    if iou_candidates is None:
        iou_candidates = (GLOBAL_IOU_CANDIDATES if mode == "global" else PER_CLASS_IOU_CANDIDATES)


    # ---------- 靜態模式 ----------
    if mode == "none":
        print(f"[INFO] Using static per-class IoU thresholds: {static_iou}")
        return dict(static_iou or PER_CLASS_IOU)

    # ---------- 全域最佳化 ----------
    if mode == "global":
        best_iou, best_f1 = None, -1.0
        print(f"\n[INFO] Optimizing global IoU threshold ...")
        for t in iou_candidates:
            res = evaluate_val(model, per_class_thr, iou_thr=t)
            f1 = res["F1"]
            if f1 > best_f1:
                best_f1, best_iou = f1, t
        print(f"[OPT] Best global IoU={best_iou:.5f} → F1={best_f1:.4f}")
        return {c: best_iou for c in CLASS_NAMES.keys()}

    # ---------- per-class 最佳化 ----------
    if mode == "per_class":
        base_iou = dict(static_iou or PER_CLASS_IOU)
        best_iou_dict = dict(base_iou)
        print(f"\n[INFO] Optimizing IoU thresholds per class ...")

        for c in CLASS_NAMES.keys():
            best_iou, best_f1 = None, -1.0
            print(f"\n[CLASS {c}] {CLASS_NAMES[c]} IoU sweep ...")
            for t in iou_candidates:
                cand_iou = dict(best_iou_dict)
                cand_iou[c] = t  # 只改該類別的 IoU，其餘類別維持原 base_iou
                res = evaluate_val(model, per_class_thr, iou_thr=cand_iou)
                f1 = res["F1"]
                if f1 > best_f1:
                    best_f1, best_iou = f1, t
            best_iou_dict[c] = best_iou
            print(f"[OPT] class {c} ({CLASS_NAMES[c]}): best IoU={best_iou:.5f}, F1={best_f1:.4f}")

        print(f"\n[OPT] Final per-class IoU thresholds: {best_iou_dict}")
        return best_iou_dict

    print(f"[WARN] Unknown IoU optimization mode '{mode}', using static IoU.")
    return dict(static_iou or PER_CLASS_IOU)

# ======================================================
# 主程式
# ======================================================
if __name__ == "__main__":
    model = YOLO(str(WEIGHTS))

    # (1) IoU threshold 最佳化
    best_iou = optimize_iou_thresholds(
        model, 
        mode=IoU_OPT_MODE, 
        per_class_thr=PER_CLASS_THR, 
        static_iou=PER_CLASS_IOU
    )

    # (2) Confidence threshold 最佳化
    best_thr = optimize_thresholds(
        model, 
        mode=CONF_OPT_MODE, 
        base_thr=PER_CLASS_THR, 
        static_iou=best_iou
    )

    print("\n" + "="*60)
    print("[FINAL] Best IoU thresholds:", best_iou)
    print("[FINAL] Best Conf thresholds:", best_thr)
    print("="*60 + "\n")

    # (3) Test 推論 - ✅ 修正：移除不存在的函式
    # infer(
    #     model,
    #     img_dir=TEST_IMG_DIR,
    #     save_dir=SAVE_TEST_DIR,
    #     per_class_thr=best_thr,
    #     per_class_iou=best_iou,
    #     label_dir=None,  # 沒有 GT
    #     save_csv=OUT_CSV
    # )

    # (4) 最終驗證
    evaluate_val(model, per_class_thr=best_thr, per_class_iou=best_iou)


# [VAL PER-CLASS]
#    class_id class_name    TP    FP   FN  Precision    Recall        F1  thr_used
# 0         0        car  4110   566  570   0.878956  0.878205  0.878581      0.30
# 1         1        hov   178    33   50   0.843602  0.780702  0.810934      0.30
# 2         2          p   492  1145   91   0.300550  0.843911  0.443243      0.05
# 3         3          m   480    62  471   0.885609  0.504732  0.643001      0.30

# [VAL PER-CLASS]
#    class_id class_name    TP   FP   FN  Precision    Recall        F1  thr_used
# 0         0        car  4110  566  570   0.878956  0.878205  0.878581      0.30
# 1         1        hov   188   46   40   0.803419  0.824561  0.813853      0.27
# 2         2          p   365  251  218   0.592532  0.626072  0.608841      0.15
# 3         3          m   712  211  239   0.771398  0.748686  0.759872      0.23


#  origin_conf : 0.45   , conf->iou


#  origin_conf : 0.45  , iou->conf

