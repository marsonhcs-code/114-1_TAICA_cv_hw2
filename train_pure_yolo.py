from pathlib import Path
from ultralytics import YOLO
import torch

# ================== 12G VRAM-friendly 設定 ==================
PROJ_ROOT  = Path(__file__).resolve().parents[0]
DATA_YAML  = PROJ_ROOT / "data" / "cv_hw2_data_aug.yaml"
RUNS_DIR   = PROJ_ROOT / "runs_yolo_finallll"

MODEL_NAME = "yolov10s"     # 模型大小
IMGSZ      = 1920           # 高解析度，保留小物件細節
EPOCHS     = 120            # 讓學習曲線充分收斂
BATCH      = 2              # 12G GPU 的可行 batch
DEVICE     = 1              # GPU ID
SEED       = 42
# ============================================================

def main():
    RUNS_DIR.mkdir(exist_ok=True, parents=True)

    model_def = f"{MODEL_NAME}.yaml"
    print(f"[INFO] Initializing model from arch: {model_def}")
    model = YOLO(model_def)  # 不載入預訓練，從頭開始訓練

    print("[INFO] Start baseline YOLO training (12G-friendly)...")
    print(f"PROJ_ROOT: {PROJ_ROOT}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        seed=SEED,
        project=str(RUNS_DIR),
        name=f"{MODEL_NAME}_exp",
        optimizer="AdamW",       # 改為 AdamW 提升穩定性
        lr0=0.0015,
        weight_decay=0.05,
        cos_lr=True,             # Cosine learning rate schedule
        warmup_epochs=3,
        patience=30,             # early stop 容忍
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0005,
        scale=0.5,
        shear=0.0,
        mosaic=0.8,
        mixup=0.3,
        copy_paste=0.5,
        box=7.0,                 # 提升 box loss 權重 → 改善高 IoU 段表現
        cls=0.7,                 # 適度降低 cls loss，避免 overfit 多類
        dfl=1.0,
        pretrained=False,        # 不使用 coco 預訓練
        workers=2,               # 減少 dataloader 負擔
        val=True,                # 每 epoch 驗證
        plots=True,              # 儲存學習曲線圖
        save=True,
        save_period=10,          # 每隔幾 epoch 存一次
        exist_ok=True
    )

    print("============================================")
    print("[DONE] Training finished.")
    print("Run dir:", results.save_dir)
    print("best weights:", Path(results.save_dir) / "weights" / "best.pt")
    print("============================================")

if __name__ == "__main__":
    main()
