from pathlib import Path
from ultralytics import YOLO
import torch

# ----------------- 你要確認/修改的設定 -----------------

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_YAML = PROJ_ROOT / "data" / "cv_hw2_data.yaml"

WEIGHTS = PROJ_ROOT / "runs_yolo_baseline" / "yolov10m_from_scratch_baseline" / "weights" / "best.pt"
# WEIGHTS = PROJ_ROOT / "runs_yolo_baseline" / "yolov10l_from_scratch_baseline" / "weights" / "best.pt"

IMGSZ = 1024          # val/infer 時用的 imgsz
IOU_THR = 0.65        # 跟你推 Kaggle 的 iou 一致
DEVICE = 0            # 0 表示 cuda:0, 也可以設 "cpu"

# 這些是要 sweep 的 confidence threshold 候選
CANDIDATE_THRS = [0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25]


def main():
    assert WEIGHTS.exists(), f"weights not found: {WEIGHTS}"
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    if DEVICE != "cpu":
        assert torch.cuda.is_available(), "No CUDA available but DEVICE != cpu ?"

    print(f"[INFO] loading model: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    results_list = []

    for thr in CANDIDATE_THRS:
        print(f"\n[SWEEP] evaluating conf_thres={thr} ...")

        # 跑 validation
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=IMGSZ,
            conf=thr,
            iou=IOU_THR,
            device=DEVICE,
            verbose=False,
        )

        # 依照你這版 ultralytics:
        # metrics.box.map      -> mAP50-95
        # metrics.box.map50    -> mAP50
        # metrics.box.mp       -> mean precision across classes
        # metrics.box.mr       -> mean recall across classes
        map5095 = metrics.box.map
        map50   = metrics.box.map50
        mp      = metrics.box.mp
        mr      = metrics.box.mr

        print(
            f"[RESULT] conf={thr:.4f} | "
            f"mAP50-95={map5095:.5f} | "
            f"mAP50={map50:.5f} | "
            f"Precision(mp)={mp:.5f} | "
            f"Recall(mr)={mr:.5f}"
        )

        results_list.append({
            "conf": float(thr),
            "mAP50-95": float(map5095),
            "mAP50": float(map50),
            "mp": float(mp),
            "mr": float(mr),
        })

    # 找 mAP50-95 最高的 threshold
    best_item = max(results_list, key=lambda d: d["mAP50-95"])

    print("\n================ SWEEP SUMMARY ================")
    for r in results_list:
        print(
            f"conf={r['conf']:.4f} | "
            f"mAP50-95={r['mAP50-95']:.5f} | "
            f"mAP50={r['mAP50']:.5f} | "
            f"mp={r['mp']:.5f} | "
            f"mr={r['mr']:.5f}"
        )

    print("\n[BEST]")
    print(
        f"Best conf={best_item['conf']:.4f} "
        f"with mAP50-95={best_item['mAP50-95']:.5f}, "
        f"mp={best_item['mp']:.5f}, "
        f"mr={best_item['mr']:.5f}"
    )
    print("=> 這個 conf 就是你之後在產 submission CSV 時要用的 CONF_THR 😎")


if __name__ == "__main__":
    main()
