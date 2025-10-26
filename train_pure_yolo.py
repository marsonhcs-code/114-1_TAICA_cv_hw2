from pathlib import Path
from ultralytics import YOLO
import torch

# ================== 12G VRAM-friendly 設定 ==================
PROJ_ROOT  = Path(__file__).resolve().parents[0]
DATA_YAML  = PROJ_ROOT / "data" / "cv_hw2_data.yaml"
RUNS_DIR   = PROJ_ROOT / "runs_yolo_baseline"

MODEL_NAME = "yolov10m"     # 小一階的模型，節省 VRAM
IMGSZ      = 1536           # 高解析度，保留小物細節
EPOCHS     = 250            # 多訓一點，補模型變小、batch 變小的影響
BATCH      = 2              # 目標是在 12G 卡上撐住 batch=2
DEVICE     = 0              # 選你要用的 GPU ID
SEED       = 42
# ============================================================

def main():
    RUNS_DIR.mkdir(exist_ok=True, parents=True)

    model_def = f"{MODEL_NAME}.yaml"
    print(f"[INFO] Initializing model from arch: {model_def}")
    model = YOLO(model_def)  # 從架構 YAML 起訓，不載別人預訓練

    print("[INFO] Start baseline YOLO training (12G-friendly)...")
    print(f"PROJ_ROOT: {PROJ_ROOT}")

    results = model.train(
        # --- 資料 ---
        data=str(DATA_YAML),

        # --- 主要超參 ---
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,  # 單卡訓練
        project=str(RUNS_DIR),
        name=f"{MODEL_NAME}_12g_highres",

        # ---- optimizer / LR schedule ----
        optimizer="AdamW",
        lr0=1e-3,          # 起始學習率
        lrf=0.05,          # cosine 最終學習率比例
        weight_decay=1e-4, # 比你原本的 0.05 小很多，不會過度L2壓縮
        warmup_epochs=3,
        cos_lr=True,       # 餵 cosine decay

        # ---- augmentation ----
        close_mosaic=10,   # 最後幾個 epoch 關掉 mosaic 幫助收斂
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.1,
        mixup=0.0,

        # ---- Loss 權重 ----
        box=7.0,
        cls=0.5,
        dfl=1.5,

        # ---- 訓練/記憶體策略 ----
        amp=True,          # 半精度 => 關鍵，省 VRAM
        patience=50,       # 早停耐心
        cache=False,       # 如果 RAM 很夠也可以後面嘗試 True 來加速 IO
        workers=4,
        save=True,
        save_period=10,    # 每 10 epoch 存一次 checkpoint
        seed=SEED,

        # ---- val / NMS 推論設定 (指的是訓練過程中的驗證步) ---
        max_det=3000,
        iou=0.7,
        conf=0.001,
    )

    print("============================================")
    print("[DONE] Training finished.")
    print("Run dir:", results.save_dir)
    print("best weights:", Path(results.save_dir) / "weights" / "best.pt")
    print("============================================")

if __name__ == "__main__":
    main()
