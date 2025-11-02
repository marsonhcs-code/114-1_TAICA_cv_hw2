from ultralytics import YOLO
from pathlib import Path
import yaml
import torch   # ← 新增

PROJ_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJ_ROOT / "data" / "cv_hw2_data_aug.yaml"  # 使用你的 data.yaml
DEVICE = "0,1"       # 雙卡 GPU
SEED = 42
EPOCHS = 120
IMGSZ = 2048         # 從 1536 降到 1024 (節省顯存)
BATCH = 4           # 從 3 增到 4 5ok

# =====================================================
# 基礎訓練參數（Baseline Stability Check 用）
# =====================================================
BASE_CFG = {
    # --- Data ---
    "data": str(DATA_YAML),
    
    # --- 主要超參 ---
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "device": DEVICE,
    "seed": SEED,

    # ---- Optimizer / LR Schedule ----
    "optimizer": "AdamW",
    "lr0": 1e-3,          # 初始學習率
    "lrf": 0.05,          # 最終學習率比例
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "cos_lr": True,       # Cosine learning rate
    
    # ---- Data Augmentation ----
    "close_mosaic": 10,   # 最後 10 個 epoch 關閉 mosaic
    
    # HSV 色彩增強
    "hsv_h": 0.015,       # Hue
    "hsv_s": 0.7,         # Saturation
    "hsv_v": 0.4,         # Value
    
    # 幾何變換
    "degrees": 0.0,       # Rotation (關閉，車輛不旋轉)
    "translate": 0.1,     # Translation
    "scale": 0.25,        # Scale variation
    "shear": 0.0,         # Shear (關閉)
    "perspective": 0.0,   # Perspective (關閉)
    
    # Flip
    "flipud": 0.0,        # Vertical flip (關閉)
    "fliplr": 0.5,        # Horizontal flip
    
    # Advanced augmentation
    "mosaic": 0.3,        # Mosaic probability (YOLO 原生支援)
    "mixup": 0.0,         # MixUp probability
    "copy_paste": 0.4,    # Copy-paste probability

    # ---- Loss 權重 ----
    "box": 8.0,           # Box loss gain
    "cls": 1.0,           # Classification loss gain
    "dfl": 1.5,           # DFL loss gain
    # ---- 訓練策略 ----
    "amp": True,          # Automatic Mixed Precision
    "patience": 50,       # Early stopping patience
    "cache": False,       # Cache images (設 True 需要大量 RAM)
    "workers": 8,         # DataLoader workers
    "save": True,
    "save_period": 10,    # 每 10 個 epoch 存一次

    # ---- 驗證/推論設定 ----
    "max_det": 3000,      # 最大偵測數量
    "iou": 0.6,           # NMS IoU threshold
    "conf": 0.25,        # Confidence threshold
}


# =====================================================
# 實驗一：Data Augmentation Sweep
# 目的：找出最適合小物體和長尾類別的增強策略
# =====================================================
exp1_param_grid = [
    # 實驗 1-1：輕量增強（Baseline）
    {
        "desc": "AUG_LIGHT",
        "mosaic": 0.3,
        "mixup": 0.0,
        "copy_paste": 0.1,
        "scale": 0.2,
        "translate": 0.05,
        "fliplr": 0.5
    },
    
    # 實驗 1-2：中度增強
    {
        "desc": "AUG_MEDIUM",
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "scale": 0.3,
        "translate": 0.1,
        "fliplr": 0.5
    },
    
    # 實驗 1-3：強增強（適合長尾）
    {
        "desc": "AUG_STRONG",
        "mosaic": 0.7,
        "mixup": 0.25,
        "copy_paste": 0.5,
        "scale": 0.4,
        "translate": 0.15,
        "fliplr": 0.6,
        "hsv_h": 0.02,  # 增強色彩變化
        "hsv_s": 0.8,
        "hsv_v": 0.5
    },
]
#    Experiment    mAP50  mAP50-95  Precision   Recall  Box_loss  Cls_loss
# 0   AUG_LIGHT  0.61962   0.25022    0.68591  0.54466   3.16789   1.83186
# 1  AUG_MEDIUM  0.62495   0.26228    0.71893  0.52853   3.36152   1.99956
# 2  AUG_STRONG  0.61812   0.26500    0.73658  0.48926   3.46098   2.10213

# =====================================================
# 實驗二：Loss / 正則化 Sweep
# 目的：平衡 box 和 classification loss，改善長尾類別
# =====================================================
exp2_param_grid = [
    # A. Baseline-like（對照組）
    {
        "desc": "LOSS_A_baseline",
        "lr0": 1e-3,
        "lrf": 0.05,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "mosaic": 0.30,
        "mixup": 0.10,
        "copy_paste": 0.30,
        "cls": 0.5,   # baseline
        "box": 7.0,
        "dfl": 1.5,
    },

    # B. 強化分類（提升 tail class 辨識力）
    {
        "desc": "LOSS_B_stronger_cls",
        "lr0": 1e-3,
        "lrf": 0.05,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "mosaic": 0.30,
        "mixup": 0.10,
        "copy_paste": 0.30,
        "cls": 0.9,   # 更重視分類誤差
        "box": 7.0,
        "dfl": 1.5,
    },

    # C. 分類＋框同時強化（針對小物件框重疊問題）
    {
        "desc": "LOSS_C_cls_box",
        "lr0": 1e-3,
        "lrf": 0.05,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "mosaic": 0.30,
        "mixup": 0.10,
        "copy_paste": 0.30,
        "cls": 1.0,   # 分類更重
        "box": 8.0,   # 更貼框
        "dfl": 1.5,
    },

    {
    "desc": "LOSS_D_dfl_focus",
    "lr0": 1e-3,
    "lrf": 0.05,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,
    "mosaic": 0.30,
    "mixup": 0.10,
    "copy_paste": 0.30,
    "cls": 0.9,
    "box": 7.0,
    "dfl": 2.0,  # 加強邊界分布學習，提升 mAP50-95 精細度
}
]
#             Experiment    mAP50  mAP50-95  Precision   Recall  Box_loss  Cls_loss
# 0      LOSS_A_baseline  0.61737   0.25279    0.69903  0.52091   3.34198   1.99606
# 1  LOSS_B_stronger_cls  0.61543   0.25542    0.71834  0.50856   3.37724   3.59523
# 2       LOSS_C_cls_box  0.62656   0.26356    0.71026  0.53317   3.85041   3.97001
# 3     LOSS_D_dfl_focus  0.62204   0.25941    0.71136  0.53118   3.36126   3.59824


# =====================================================
# 實驗三：Learning Rate / Warmup Sweep
# 目的：找出最佳收斂速度
# =====================================================
exp3_param_grid = [
    # 實驗 3-1：基準 LR
    {
        "desc": "LR_BASE",
        "lr0": 0.001,
        "lrf": 0.1,
        "warmup_epochs": 5
    },
    
    # 實驗 3-2：快速學習
    {
        "desc": "LR_FAST",
        "lr0": 0.002,
        "lrf": 0.05,
        "warmup_epochs": 3
    },
    
    # 實驗 3-3：慢速學習（更穩定）
    {
        "desc": "LR_SLOW",
        "lr0": 0.0005,
        "lrf": 0.2,
        "warmup_epochs": 7
    },
]
#   Experiment    mAP50  mAP50-95  Precision   Recall  Box_loss  Cls_loss
# 0    LR_BASE  0.64280   0.26204    0.71903  0.56131   3.75093   3.72925
# 1    LR_FAST  0.64748   0.26205    0.72322  0.55870   3.79361   3.74196
# 2    LR_SLOW  0.61469   0.25506    0.70654  0.52095   3.81286   3.89519

# =====================================================
# 實驗四：Batch Size Sweep（針對你的雙卡設定）
# =====================================================
# 未做
exp4_param_grid = [
    {"desc": "LR_FAST_v2", "lr0": 0.002, "lrf": 0.05, "warmup_epochs": 5},   # 延長 warmup，提升穩定性
    {"desc": "LR_FAST_cosine", "lr0": 0.002, "lrf": 0.05, "cos_lr": True},   # 使用 cosine scheduler
    {"desc": "LR_FAST_highfinal", "lr0": 0.002, "lrf": 0.15, "warmup_epochs": 3},  # 收尾更高的最終 lr
]

# 觀察 person 類別是否可由 oversample 提升
exp5_param_grid = [
    {
        "desc": "PH0_no_oversample",
        "model": "yolov10s.yaml",
        "epochs": 70,
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "cls": 1.0,
        "box": 8.0,
        "dfl": 1.5,
        "lr0": 0.002,
        "lrf": 0.05,
        "warmup_epochs": 3
    },
    {
        "desc": "PH0_oversample_person",
        "model": "yolov10s.yaml",
        "epochs": 70,
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "cls": 1.0,
        "box": 8.0,
        "dfl": 1.5,
        "lr0": 0.002,
        "lrf": 0.05,
        "warmup_epochs": 3,
        "use_oversample": True     # ← 資料層實驗旗標
    }
]

# --- Exp6: YOLOv10m 探索 (快速驗證) ---
exp6_param_grid = [
    # 6A: 80 epochs (基準)
    {
        "desc": "M_80ep_baseline",
        "model": "yolov10m.yaml",
        "epochs": 80,
        
        # 最佳 Data Aug (from Exp1)
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "scale": 0.3,
        "translate": 0.1,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        # 最佳 Loss (from Exp2)
        "cls": 1.0,
        "box": 8.0,
        "dfl": 1.5,
        
        # 最佳 LR (from Exp3)
        "lr0": 0.002,
        "lrf": 0.05,
        "warmup_epochs": 3,
        "cos_lr": True,
        
        # 其他穩定設定
        "weight_decay": 1e-4,
        "close_mosaic": 10,
        "patience": 50,
    },
    
    # 6B: 90 epochs (延長訓練)
    {
        "desc": "M_90ep_extended",
        "model": "yolov10m.yaml",
        "epochs": 90,
        
        # 相同的增強/Loss/LR
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "scale": 0.3,
        "translate": 0.1,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        "cls": 1.0,
        "box": 8.0,
        "dfl": 1.5,
        
        "lr0": 0.002,
        "lrf": 0.05,
        "warmup_epochs": 3,
        "cos_lr": True,
        
        "weight_decay": 1e-4,
        "close_mosaic": 10,
        "patience": 50,
    },
    
    # 6C: 90 epochs + 更高尾段學習率 (避免過早停滯)
    {
        "desc": "M_90ep_highfinal",
        "model": "yolov10m.yaml",
        "epochs": 90,
        
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "scale": 0.3,
        "translate": 0.1,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        "cls": 1.0,
        "box": 8.0,
        "dfl": 1.5,
        
        "lr0": 0.002,
        "lrf": 0.10,  # ← 從 0.05 提高到 0.10 (尾段學習力)
        "warmup_epochs": 5,  # ← 延長 warmup 提升穩定性
        "cos_lr": True,
        
        "weight_decay": 1e-4,
        "close_mosaic": 10,
        "patience": 50,
    }
]

# --- Exp7: 最終版 (120 epochs 長訓練) ---
exp7_param_grid = [
    # ===================================================================
    # 7A: 保守版 (穩定基線)
    # 目標: mAP50 ~0.67, Recall ~0.62, 穩定收斂
    # ===================================================================
    {
        "desc": "M_120ep_conservative",
        "model": "yolov10m.yaml",
        "epochs": 120,
        # "batch": 10,
        
        # === Data Augmentation (中度) ===
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.35,  # ← 改進 1: 從 0.3 提高到 0.35 (補償 Recall)
        "scale": 0.25,
        "translate": 0.1,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        # === Loss Weights ===
        "cls": 0.9,   # ← 改進 2: 從 1.0 降到 0.9 (避免 Cls_loss 過高)
        "box": 8.0,
        "dfl": 1.5,
        
        # === Learning Rate ===
        "lr0": 0.002,
        "lrf": 0.08,
        "warmup_epochs": 5,
        "cos_lr": True,
        
        # === Regularization ===
        "weight_decay": 1.5e-4,
        "close_mosaic": 15,
        "patience": 60,
    },
    
    # ===================================================================
    # 7B: 激進版 (Long-tail 專攻)
    # 目標: mAP50-95 ~0.30+, 犧牲部分 Precision 換取 Recall
    # ===================================================================
    {
        "desc": "M_120ep_aggressive",
        "model": "yolov10m.yaml",
        "epochs": 120,
        # "batch": 10,
        
        # === 強增強 (針對小物件 & 長尾類別) ===
        "mosaic": 0.6,
        "mixup": 0.20,
        "copy_paste": 0.5,   # ← 改進 3: 從 0.4 提高到 0.5 (大幅增加小物件)
        "scale": 0.35,
        "translate": 0.12,
        "fliplr": 0.5,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        
        # === Loss Weights (降低分類壓力) ===
        "cls": 0.85,  # ← 改進 4: 從 1.0 降到 0.85 (更激進)
        "box": 8.0,
        "dfl": 1.8,   # ← 改進 5: 從 1.5 提高到 1.8 (更精細的邊界)
        
        # === Learning Rate (保持學習力) ===
        "lr0": 0.002,
        "lrf": 0.12,
        "warmup_epochs": 5,
        "cos_lr": True,
        
        # === Regularization (輕度,允許過擬合長尾) ===
        "weight_decay": 8e-5,  # ← 改進 6: 降低正則化 (from 1e-4)
        "close_mosaic": 10,
        "patience": 60,
    },
    
    # ===================================================================
    # 7C: 平衡版 (推薦 ⭐) - 結合 Exp6 highfinal 優點
    # 目標: mAP50 ~0.68, Recall ~0.64, F1 最高
    # ===================================================================
    {
        "desc": "M_120ep_balanced",
        "model": "yolov10m.yaml",
        "epochs": 120,
        # "batch": 10,
        
        # === 漸進增強 (吸收 Exp6_highfinal 經驗) ===
        "mosaic": 0.55,
        "mixup": 0.18,
        "copy_paste": 0.42,  # ← 改進 7: 從 0.35 提高 (重點!)
        "scale": 0.30,
        "translate": 0.11,
        "fliplr": 0.5,
        "hsv_h": 0.018,
        "hsv_s": 0.75,
        "hsv_v": 0.45,
        
        # === Loss Weights (平衡) ===
        "cls": 0.92,  # ← 改進 8: 從 1.0 降到 0.92 (介於保守與激進)
        "box": 8.0,
        "dfl": 1.6,   # ← 改進 9: 從 1.5 提高到 1.6
        
        # === Learning Rate (高尾段,from Exp6_highfinal) ===
        "lr0": 0.002,
        "lrf": 0.10,  # ← 保持,這在 Exp6 證明有效
        "warmup_epochs": 5,
        "cos_lr": True,
        
        # === Regularization (適度) ===
        "weight_decay": 1.2e-4,
        "close_mosaic": 12,
        "patience": 60,
    },
    
    # ===================================================================
    # 7D: 實驗版 (可選) - 測試極端 copy_paste
    # 目標: 驗證 copy_paste > 0.5 是否對 person 類別有幫助
    # ===================================================================
    {
        "desc": "M_120ep_copypaste_extreme",
        "model": "yolov10m.yaml",
        "epochs": 120,
        # "batch": 10,
        
        # === 極端 copy_paste 策略 ===
        "mosaic": 0.5,
        "mixup": 0.15,
        "copy_paste": 0.6,   # ← 改進 10: 極端值,專門針對小物件
        "scale": 0.28,       # ← 降低其他增強避免過度
        "translate": 0.09,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        # === Loss Weights ===
        "cls": 0.88,  # ← 更低,配合極端增強
        "box": 8.0,
        "dfl": 1.5,
        
        # === Learning Rate ===
        "lr0": 0.002,
        "lrf": 0.09,
        "warmup_epochs": 5,
        "cos_lr": True,
        
        # === Regularization ===
        "weight_decay": 1.0e-4,
        "close_mosaic": 12,
        "patience": 60,
    }
]

exp8_param_grid = [
    {
    "desc": "M_120ep_conservative_warmup8",
    "model": "yolov10m.yaml",
    "epochs": 120,
    "mosaic": 0.5,
    "mixup": 0.15,
    "copy_paste": 0.35,
    "scale": 0.25,
    "translate": 0.1,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cls": 0.9,
    "box": 8.0,
    "dfl": 1.5,
    "lr0": 0.002,
    "lrf": 0.08,
    "warmup_epochs": 8,     # ← 從 5 延長至 8
    "cos_lr": True,
    "weight_decay": 1.5e-4,
    "close_mosaic": 15,
    "patience": 70,         # ← 稍延長 early-stop
    },
    {
    "desc": "m_1920img_highres_large",
    "model": "yolov10m.yaml",
    "epochs": EPOCHS,       # 你全域控制即可
    "imgsz": 2048,          # ← 主變因
    "batch": 4,             # 若 GPU 24 GB 建議上限 6–8
    "mosaic": 0.55,
    "mixup": 0.15,
    "copy_paste": 0.35,
    "scale": 0.3,
    "translate": 0.1,
    "fliplr": 0.5,
    "cls": 1.0,
    "box": 8.0,
    "dfl": 1.5,
    "lr0": 0.002,
    "lrf": 0.10,
    "warmup_epochs": 5,
    "cos_lr": True,
    "weight_decay": 1e-4,
    "close_mosaic": 12,
    "patience": 60,
}


]

def load_class_weights(yaml_path):
    """從 data.yaml 讀取 class weights"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    weights = data.get("weights", None)
    if weights:
        print(f"[INFO] Loaded class weights from YAML: {weights}")
        return torch.tensor(weights)
    else:
        print("[INFO] No class weights specified, using uniform [1.0].")
        return None

# =====================================================
# 實驗執行函式
# =====================================================
def run_experiment(exp_name, param_grid, epochs=None):
    """
    執行一組實驗
    
    Args:
        exp_name: 實驗名稱
        param_grid: 參數網格
        epochs: 可選，覆蓋預設 epochs
    """
    print("\n" + "="*70)
    print(f"Running Experiment: {exp_name}")
    print("="*70)
    
    class_weights = load_class_weights(DATA_YAML)

    for i, params in enumerate(param_grid, 1):
        desc = params.get("desc", f"config{i}")
        print(f"\n[{i}/{len(param_grid)}] {exp_name} - {desc}")
        
        # 建立模型（從零開始）
        model = YOLO("yolov10s.yaml")  # 使用 nano (最快)
        # model = YOLO("yolov8s.yaml")  # 或用 small (更準)
        
        # 合併參數
        params_cleaned = {k: v for k, v in params.items() if k != "desc"}
        overrides = BASE_CFG.copy()
        overrides.update(params_cleaned)
        
        # 覆蓋 epochs（如果指定）
        if epochs is not None:
            overrides["epochs"] = epochs
        
        # 加入 class weights（若有設定）
        # if class_weights is not None:
        #     overrides["class_weights"] = class_weights

        
        # 開始訓練
        try:
            model.train(
                **overrides,
                project="runs_yolo_exp",
                name=f"{exp_name}_{desc}"
            )
            print(f"✅ {desc} completed")
        except Exception as e:
            print(f"❌ {desc} failed: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Experiment {exp_name} completed")
    print(f"{'='*70}\n")


# =====================================================
# Baseline Stability Check（快速驗證）
# =====================================================
def run_stability_check():
    """執行 30 epochs 的穩定性檢查"""
    print("\n" + "="*70)
    print("Running Baseline Stability Check (30 Epochs)")
    print("="*70)
    
    model = YOLO("yolov8n.yaml")
    
    stability_cfg = BASE_CFG.copy()
    stability_cfg.update({
        "epochs": 30,  # 短程訓練
        "patience": 15,
        "mosaic": 0.3,  # 輕量增強
        "mixup": 0.0,
        "copy_paste": 0.0,
    })
    
    model.train(
        **stability_cfg,
        project="runs_yolo_exp",
        name="baseline_stability_check"
    )
    
    print("\n✅ Stability check completed")
    print("Review: runs_yolo_exp/baseline_stability_check/")

# =====================================================
# 主程式入口：控制各階段實驗流程
# =====================================================
if __name__ == "__main__":
    # -------------------------------------------------
    # 實驗一：Data Augmentation Sweep
    # 目的：比較輕／中／強三種增強對長尾與小物件的影響
    # -------------------------------------------------
    # run_experiment("Exp1_DataAug", exp1_param_grid)

    # -------------------------------------------------
    # 實驗二：Loss / 正則化 Sweep
    # 目的：平衡 Box 與 Cls Loss，改善長尾類別表現
    # -------------------------------------------------
    # run_experiment("Exp2_LossReg", exp2_param_grid)

    # -------------------------------------------------
    # 實驗三：Learning Rate / Warmup Sweep
    # 目的：找出最佳收斂速度與穩定性
    # -------------------------------------------------
    # run_experiment("Exp3_LR_Schedule", exp3_param_grid)

    # -------------------------------------------------
    # 實驗四：Learning Rate 延伸測試（LR_FAST 衍生）
    # 目的：延長 warmup、測試 cosine 與 highfinal 收尾策略
    # -------------------------------------------------
    # run_experiment("Exp4_LR_FAST", exp4_param_grid)

    # -------------------------------------------------
    # 實驗五：Oversample 驗證（person 類別強化）
    # 目的：比較 oversample 與否對 Recall 與 Precision 的影響
    # -------------------------------------------------
    # run_experiment("Exp5_PersonOversample", exp5_param_grid)

    # 根據 Exp5 分析結果決定是否加入 oversample
    # run_experiment("Exp6_ModelScale_10m", exp6_param_grid)
    
    # Exp6 完成後，選擇 Exp7 配置
    # run_experiment("Exp7_Final", exp7_param_grid) 
    
    run_experiment("Exp8_ScaleWarm_ResolutionSweep", [exp8_param_grid[1]]) 


    print("\n✅ 所有實驗執行完畢。請查看 runs_yolo_exp/ 資料夾。")

