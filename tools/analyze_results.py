import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# === 主要路徑設定 ===
ROOT = Path("runs_yolo_exp")
SUMMARY_CSV = ROOT / "runs_yolo_exp_summary.csv"
SUMMARY_PNG = ROOT / "runs_yolo_exp_comparison.png"

# === 自動搜尋所有 results.csv ===
result_files = list(ROOT.rglob("results.csv"))
if not result_files:
    print("⚠️ 未找到任何 results.csv，請確認路徑是否正確。")
    exit()

summary = []

for result_path in result_files:
    exp_name = result_path.parent.name
    try:
        df = pd.read_csv(result_path)
        # --- 若有 epoch 欄位則確保排序正確 ---
        if "epoch" in df.columns:
            df = df.sort_values("epoch").reset_index(drop=True)

        # --- 取出最佳 (mAP50) 的那一行 ---
        if "metrics/mAP50-95(B)" in df.columns:
            best_idx = df["metrics/mAP50-95(B)"].idxmax()
            best = df.loc[best_idx]
        else:
            best = df.iloc[-1]  # fallback：若欄位不存在取最後一行

        summary.append({
            "best_mAP50-95_in_Experiment": exp_name,
            "best_mAP50-95": best.get("metrics/mAP50-95(B)", None),
            "mAP50": best.get("metrics/mAP50(B)", None),
            # "Precision": best.get("metrics/precision(B)", None),
            # "Recall": best.get("metrics/recall(B)", None),
            # "Box_loss": best.get("val/box_loss", None),
            # "Cls_loss": best.get("val/cls_loss", None),
            "Epoch": best.get("epoch", None),
            "ToTal Epoch": df["epoch"].max(),

            # "CSV_Path": str(result_path)
        })

    except Exception as e:
        print(f"❌ 讀取失敗: {result_path} ({e})")

# === 匯出總表 ===
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values("best_mAP50-95_in_Experiment", ascending=True).reset_index(drop=True)
# summary_df.to_csv(SUMMARY_CSV, index=False)
print("\n✅ Summary saved to:", SUMMARY_CSV)
print(summary_df)

# === 自動調整 y 軸以放大差異 ===
def auto_ylim(values, scale=0.2):
    vmin, vmax = np.min(values), np.max(values)
    diff = vmax - vmin
    return vmin - diff * 0.05, vmax + diff * scale

# === 建立兩個子圖（共享 x 軸） ===
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'hspace': 0.35})

# ---- 圖1：mAP50 ----
ymin, ymax = auto_ylim(summary_df["mAP50"], scale=0.25)
axes[0].bar(summary_df["best_mAP50-95_in_Experiment"], summary_df["mAP50"],
            color="#1f77b4", alpha=0.9, edgecolor="black", linewidth=1)
axes[0].set_ylim(ymin, ymax)
axes[0].set_title("YOLOv10 Experiment - mAP50", fontsize=14, fontweight="bold")
axes[0].set_ylabel("mAP50", fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(summary_df["mAP50"]):
    axes[0].text(i, v + 0.002, f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# ---- 圖2：mAP50-95 ----
ymin, ymax = auto_ylim(summary_df["best_mAP50-95"], scale=0.25)
axes[1].bar(summary_df["best_mAP50-95_in_Experiment"], summary_df["best_mAP50-95"],
            color="#ff7f0e", alpha=0.9, edgecolor="black", linewidth=1)
axes[1].set_ylim(ymin, ymax)
axes[1].set_title("YOLOv10 Experiment - mAP50-95", fontsize=14, fontweight="bold")
axes[1].set_ylabel("mAP50-95", fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

for i, v in enumerate(summary_df["best_mAP50-95"]):
    axes[1].text(i, v + 0.001, f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# ---- X 軸與整體設定 ----
plt.xticks(range(len(summary_df)), summary_df["best_mAP50-95_in_Experiment"], rotation=45, ha='right', fontsize=9)
plt.suptitle("YOLOv10 Experiment Performance Comparison", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# === 輸出合併圖檔 ===
plt.savefig("Exp_Performance_Comparison_Enhanced.png", dpi=400)
plt.close()

print("✅ Saved enhanced chart → Exp_Performance_Comparison_Enhanced.png")
