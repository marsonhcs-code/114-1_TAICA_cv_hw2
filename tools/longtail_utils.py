# tools/longtail_utils.py
# Utility functions for Long-Tail Object Detection analysis

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pandas as pd
from collections import defaultdict


# ===================== Dataset Analysis =====================
def analyze_class_distribution(annotation_file: str, class_names: List[str]):
    """
    分析數據集的類別分佈
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    class_counts = defaultdict(int)
    image_counts = defaultdict(int)
    
    for img_id, anns in annotations.items():
        classes_in_image = set()
        for ann in anns:
            cls = ann['category_id']
            class_counts[cls] += 1
            classes_in_image.add(cls)
        
        for cls in classes_in_image:
            image_counts[cls] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    total_instances = sum(class_counts.values())
    total_images = len(annotations)
    
    for cls_id, cls_name in enumerate(class_names):
        inst_count = class_counts.get(cls_id, 0)
        img_count = image_counts.get(cls_id, 0)
        inst_ratio = inst_count / total_instances * 100
        img_ratio = img_count / total_images * 100
        
        print(f"\n{cls_name:12s} (id={cls_id}):")
        print(f"  Instances: {inst_count:6d} ({inst_ratio:5.2f}%)")
        print(f"  Images:    {img_count:6d} ({img_ratio:5.2f}%)")
        print(f"  Avg/Image: {inst_count/max(img_count,1):5.2f}")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n{'Total instances:':20s} {total_instances}")
    print(f"{'Total images:':20s} {total_images}")
    print(f"{'Imbalance ratio:':20s} {imbalance_ratio:.2f}x")
    print(f"  ({'Head/Tail':^15s} = {class_names[np.argmax(list(class_counts.values()))]} / {class_names[np.argmin(list(class_counts.values()))]})")
    print("="*60 + "\n")
    
    return {
        'class_counts': dict(class_counts),
        'image_counts': dict(image_counts),
        'imbalance_ratio': imbalance_ratio
    }


def plot_class_distribution(class_counts: Dict[int, int], 
                           class_names: List[str],
                           save_path: str = None):
    """
    繪製類別分佈圖
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot
    classes = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    colors = ['#2ecc71' if c == max(counts) else '#e74c3c' if c == min(counts) else '#3498db' 
              for c in counts]
    
    ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution (Absolute)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (cls, cnt) in enumerate(zip(classes, counts)):
        ax1.text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontweight='bold')
    
    # Log-scale plot
    ax2.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Instances (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


# ===================== Training Analysis =====================
def analyze_training_results(results_csv: str, class_names: List[str]):
    """
    分析訓練結果，比較不同類別的 AP
    """
    df = pd.read_csv(results_csv)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    # Overall metrics
    final_epoch = df.iloc[-1]
    print(f"\nFinal Epoch: {int(final_epoch['epoch'])}")
    print(f"  Train Loss:  {final_epoch['train_loss']:.4f}")
    print(f"  Val Loss:    {final_epoch.get('val_loss', 'N/A')}")
    print(f"  mAP@50:      {final_epoch.get('map50', 'N/A')}")
    print(f"  mAP@50-95:   {final_epoch.get('map5095', 'N/A')}")
    print(f"  Learning Rate: {final_epoch['lr']:.6f}")
    
    # Best epoch
    if 'map5095' in df.columns:
        best_idx = df['map5095'].idxmax()
        best_epoch = df.iloc[best_idx]
        print(f"\nBest Epoch: {int(best_epoch['epoch'])}")
        print(f"  mAP@50-95:   {best_epoch['map5095']:.4f}")
    
    print("="*60 + "\n")
    
    return df


def plot_training_curves(results_csv: str, save_path: str = None):
    """
    繪製訓練曲線
    """
    df = pd.read_csv(results_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax = axes[0, 0]
    if 'train_loss' in df.columns:
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # mAP curves
    ax = axes[0, 1]
    if 'map50' in df.columns:
        ax.plot(df['epoch'], df['map50'], label='mAP@50', linewidth=2)
    if 'map5095' in df.columns:
        ax.plot(df['epoch'], df['map5095'], label='mAP@50-95', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    ax.plot(df['epoch'], df['lr'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(alpha=0.3)
    
    # Precision & Recall (if available)
    ax = axes[1, 1]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    
    plt.show()


def compare_experiments(exp_dirs: List[str], exp_names: List[str],
                       metric: str = 'map5095', save_path: str = None):
    """
    比較多個實驗的結果
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    results = []
    
    for exp_dir, exp_name in zip(exp_dirs, exp_names):
        results_csv = Path(exp_dir) / 'results.csv'
        if not results_csv.exists():
            print(f"Warning: {results_csv} not found, skipping...")
            continue
        
        df = pd.read_csv(results_csv)
        
        # Plot curves
        if metric in df.columns:
            axes[0].plot(df['epoch'], df[metric], label=exp_name, linewidth=2)
        
        # Collect final results
        final_value = df[metric].iloc[-1] if metric in df.columns else 0
        best_value = df[metric].max() if metric in df.columns else 0
        results.append({
            'experiment': exp_name,
            'final': final_value,
            'best': best_value
        })
    
    # Curve comparison
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel(metric.upper(), fontsize=12)
    axes[0].set_title(f'{metric.upper()} Comparison Across Experiments', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Bar comparison
    results_df = pd.DataFrame(results)
    x = np.arange(len(results_df))
    width = 0.35
    
    axes[1].bar(x - width/2, results_df['final'], width, label='Final', alpha=0.7)
    axes[1].bar(x + width/2, results_df['best'], width, label='Best', alpha=0.7)
    axes[1].set_xlabel('Experiment', fontsize=12)
    axes[1].set_ylabel(metric.upper(), fontsize=12)
    axes[1].set_title(f'Final vs Best {metric.upper()}', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['experiment'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()
    
    return results_df


# ===================== Per-Class Analysis =====================
def analyze_per_class_metrics(per_class_csv: str, class_names: List[str]):
    """
    分析每個類別的表現
    """
    df = pd.read_csv(per_class_csv)
    
    print("\n" + "="*60)
    print("PER-CLASS METRICS ANALYSIS")
    print("="*60)
    
    for _, row in df.iterrows():
        cls_name = row.get('class', 'Unknown')
        ap50 = row.get('ap50', 0)
        ap5095 = row.get('ap5095', 0)
        precision = row.get('precision', 0)
        recall = row.get('recall', 0)
        num_gt = row.get('num_gt', 0)
        
        print(f"\n{cls_name:12s}:")
        print(f"  AP@50:      {ap50:.4f}")
        print(f"  AP@50-95:   {ap5095:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  # GT:       {num_gt}")
    
    print("="*60 + "\n")
    
    return df


def plot_per_class_metrics(per_class_csv: str, class_names: List[str],
                           save_path: str = None):
    """
    繪製每個類別的 AP
    """
    df = pd.read_csv(per_class_csv)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    classes = df['class'].tolist()
    ap50 = df['ap50'].tolist()
    ap5095 = df['ap5095'].tolist()
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Color code: green for head, red for tail
    num_gt = df['num_gt'].tolist()
    colors_50 = ['#2ecc71' if gt == max(num_gt) else '#e74c3c' if gt == min(num_gt) else '#3498db' 
                 for gt in num_gt]
    colors_5095 = ['#27ae60' if gt == max(num_gt) else '#c0392b' if gt == min(num_gt) else '#2980b9' 
                   for gt in num_gt]
    
    ax.bar(x - width/2, ap50, width, label='AP@50', color=colors_50, alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, ap5095, width, label='AP@50-95', color=colors_5095, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Precision', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Average Precision', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (a50, a95) in enumerate(zip(ap50, ap5095)):
        ax.text(i - width/2, a50, f'{a50:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, a95, f'{a95:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics to: {save_path}")
    
    plt.show()


# ===================== Main Analysis Script =====================
if __name__ == "__main__":
    # Example usage
    
    # 1. Analyze dataset distribution
    class_names = ["car", "motorcycle", "person", "hov"]
    stats = analyze_class_distribution(
        annotation_file="data/annotations.json",
        class_names=class_names
    )
    
    plot_class_distribution(
        class_counts=stats['class_counts'],
        class_names=class_names,
        save_path="analysis/class_distribution.png"
    )
    
    # 2. Analyze training results
    analyze_training_results(
        results_csv="runs/exp4_full/results.csv",
        class_names=class_names
    )
    
    plot_training_curves(
        results_csv="runs/exp4_full/results.csv",
        save_path="analysis/training_curves.png"
    )
    
    # 3. Compare experiments
    compare_experiments(
        exp_dirs=[
            "runs/exp1_baseline",
            "runs/exp2_cb_loss",
            "runs/exp3_cb_rfs",
            "runs/exp4_full"
        ],
        exp_names=["Baseline", "CB Loss", "CB+RFS", "Full Pipeline"],
        metric="map5095",
        save_path="analysis/experiment_comparison.png"
    )
    
    # 4. Analyze per-class metrics
    analyze_per_class_metrics(
        per_class_csv="runs/exp4_full/metrics/per_class_metrics.csv",
        class_names=class_names
    )
    
    plot_per_class_metrics(
        per_class_csv="runs/exp4_full/metrics/per_class_metrics.csv",
        class_names=class_names,
        save_path="analysis/per_class_ap.png"
    )