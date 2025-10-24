#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, csv
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# project modules
from tools.utils import (
    set_seed, is_main, default_device, ensure_dir,
    capture_env, cleanup_ddp
)
from tools.io import write_json, append_csv_row, save_ckpt, load_ckpt, load_cfg
from tools.kfold import make_kfold_splits
from hooks_yolo_longtail import build_model, build_dataloaders, evaluate

# ======================= 直接在這裡改參數 =======================
# train.py - Modified CONFIG for Long-Tail Detection

CONFIG = dict(
    # ========== 基本設定 ==========
    cfg="configs/exp_longtail.yaml",
    out="runs/longtail_yolo_exp1",
    model_name="yolo_longtail",
    epochs=150,  # Long-tail 通常需要更多 epochs
    seed=42,
    note="YOLOv8 + CB-Focal Loss + Repeat Factor Sampling + Strong Aug for Tail",
    
    # ========== 訓練優化 ==========
    amp=True,
    compile=False,  # Set True if PyTorch 2.x
    accum=4,  # Gradient accumulation (effective batch = 16*4 = 64)
    grad_clip=10.0,  # Clip gradients to prevent explosion
    
    # ========== DDP (Multi-GPU) ==========
    dist=False,  # Set True when using torchrun
    find_unused_parameters=False,
    
    # ========== 最佳模型指標 ==========
    # Long-tail 建議用 tail classes 的平均 AP
    best_metric="map5095",  # or "tail_map" if implemented
    
    # ========== 測試評估 ==========
    eval_test_each_epoch=False,  # 節省時間，只在最後測試
    
    # ========== Resume Training ==========
    resume=None,  # "runs/longtail_yolo_exp1/weights/best.pt"
    
    # ========== Early Stopping ==========
    early_stop_patience=20,  # Long-tail 需要更多耐心
    
    # ========== Learning Rate Scheduler ==========
    scheduler="cos",  # Cosine Annealing 適合 long-tail
    # For step scheduler:
    # scheduler="step"
    # step_size=50
    # gamma=0.1
    
    # ========== K-Fold Cross Validation ==========
    kfold=0,  # 0 = disabled; 5 = 5-fold CV
    save_splits=True,
    
    # ========== Long-Tail Specific ==========
    # 以下是額外參數，會傳入 cfg
    longtail_config=dict(
        # Two-stage training
        enable_two_stage=False,
        stage1_epochs=100,
        stage2_epochs=50,
        
        # Logit adjustment (test-time)
        use_logit_adjustment=True,
        tau=1.0,
        
        # Monitoring
        monitor_tail_classes=True,
        tail_class_ids=[1, 2, 3],  # motorcycle, person, hov
        
        # Visualization
        save_confusion_matrix=True,
        save_pr_curves=True,
    )
)


# ========== 推薦的訓練流程 ==========
"""
實驗 1: Baseline (no long-tail handling)
---------------------------------------
CONFIG = {
    'note': 'Baseline YOLO without long-tail strategies',
    'cfg': 'configs/exp_baseline.yaml',  # use_cb_loss=False, use_repeat_factor_sampling=False
    'out': 'runs/exp1_baseline',
    'epochs': 100,
}

預期結果：
- Overall mAP: ~65%
- Car AP: ~80% (good)
- HOV AP: ~25% (poor) ← 這是我們要改善的


實驗 2: CB-Focal Loss
----------------------
CONFIG = {
    'note': 'Add Class-Balanced Focal Loss',
    'cfg': 'configs/exp_cb_loss.yaml',  # use_cb_loss=True
    'out': 'runs/exp2_cb_loss',
    'epochs': 100,
}

預期改善：
- HOV AP: 25% → 35% (+10%)
- Overall mAP: 65% → 67% (+2%)


實驗 3: CB-Focal + Repeat Factor Sampling
------------------------------------------
CONFIG = {
    'note': 'CB Loss + Repeat Factor Sampling',
    'cfg': 'configs/exp_cb_rfs.yaml',  # use_repeat_factor_sampling=True
    'out': 'runs/exp3_cb_rfs',
    'epochs': 120,  # RFS 需要更多 epochs
}

預期改善：
- HOV AP: 35% → 45% (+10%)
- Overall mAP: 67% → 70% (+3%)


實驗 4: Full Pipeline (CB + RFS + Strong Aug)
----------------------------------------------
CONFIG = {
    'note': 'Full long-tail pipeline',
    'cfg': 'configs/exp_longtail.yaml',  # All strategies enabled
    'out': 'runs/exp4_full',
    'epochs': 150,
    'accum': 4,
}

預期改善：
- HOV AP: 45% → 55% (+10%)
- Overall mAP: 70% → 73% (+3%)


實驗 5: Two-Stage Training (Optional)
--------------------------------------
CONFIG = {
    'note': 'Two-stage: pretrain → tail fine-tune',
    'cfg': 'configs/exp_two_stage.yaml',
    'out': 'runs/exp5_two_stage',
    'epochs': 150,  # 100 stage1 + 50 stage2
    'longtail_config': {
        'enable_two_stage': True,
        'stage1_epochs': 100,
        'stage2_epochs': 50,
    }
}

預期改善：
- HOV AP: 55% → 60% (+5%)
- Overall mAP: 73% → 74% (+1%)
"""


# ========== 執行命令範例 ==========
"""
# Single GPU
nohup python train.py > log/run_exp4_full.log 2>&1 & echo $! > log/run_exp4.pid

# Multi-GPU (2 cards)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone train.py > log/run_exp4_full.log 2>&1 & echo $! > log/run_exp4.pid

# Check training progress
tail -f log/run_exp4_full.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Kill training
kill $(cat log/run_exp4.pid) && rm log/run_exp4.pid
"""


# ========== Important Notes ==========
"""
1. Batch Size 調整：
   - 單 GPU (24GB): batch=16, accum=4 → effective_batch=64
   - 雙 GPU (24GB): batch=16, accum=2 → effective_batch=64
   - 若 OOM: 降低 batch 並增加 accum 保持 effective_batch

2. Learning Rate 調整：
   - effective_batch=64 → lr=0.001
   - effective_batch=128 → lr=0.002
   - Linear scaling rule

3. Epochs 調整：
   - Baseline: 100 epochs
   - With RFS: 120-150 epochs (因為每個 epoch 看到更多樣本)
   - Two-stage: 100 + 50 epochs

4. 監控指標：
   - 不要只看 overall mAP！
   - 重點看 tail classes (hov, person, motorcycle) 的 AP
   - 使用 per_class_metrics.csv 追蹤每個類別

5. 調試技巧：
   - 先用小資料集 (100 images) 快速驗證 pipeline
   - 確認 loss 有正常下降
   - 檢查 confusion matrix 找出混淆的類別
"""


def train_one_epoch(model, loader, optimizer, device, scaler=None,
                    accum_steps: int = 1, grad_clip: Optional[float] = None) -> Dict[str, float]:
    """
    通用單 epoch 訓練：
      - 支援 images/targets 批次；model 可回 {'loss': ...} 或 Tensor loss
      - 支援 AMP 與梯度累積與可選梯度裁切
    """
    model.train()
    total_loss = 0.0
    n = 0

    for step, batch in enumerate(loader, start=1):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        else:
            images, targets = batch, None

        def to_dev(x):
            if torch.is_tensor(x): return x.to(device, non_blocking=True)
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
            return x

        images = to_dev(images); targets = to_dev(targets)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                out = model(images, targets) if targets is not None else model(images)
                loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters() if not isinstance(model, DDP) else model.module.parameters(),
                        max_norm=grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
        else:
            out = model(images, targets) if targets is not None else model(images)
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
            (loss / accum_steps).backward()
            if step % accum_steps == 0:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters() if not isinstance(model, DDP) else model.module.parameters(),
                        max_norm=grad_clip
                    )
                optimizer.step()

        total_loss += float(loss.detach().item()); n += 1

    return {"train_loss": total_loss / max(1, n)}


def run_once(args, cfg, device, dist_enabled=False, fold_split=None) -> Dict[str, Any]:
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir / "weights")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "preds")
    ensure_dir(out_dir / "splits")

    # ----- 建模 / 優化器 / AMP / Scheduler / Resume -----
    model = build_model(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4)
    )

    # scheduler
    sched = None
    if args.scheduler == "cos":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 資料載入（於 hooks 實作；會根據 cfg["task"] 決定任務）
    dls = build_dataloaders(cfg, fold_split=fold_split, dist=dist_enabled)
    train_loader, val_loader, test_loader = dls.get("train"), dls.get("val"), dls.get("test")

    # DDP 包裝
    if dist_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters)

    # env & hparams（主進程記錄一次）
    start_epoch = 0
    if is_main():
        write_json(out_dir / "hparams.json",
                   {"CONFIG": {**vars(args), "start_epoch": start_epoch}, "cfg": cfg, "note": args.note})
        capture_env(out_dir)

    # resume（可選）
    if args.resume:
        try:
            ckpt_path = Path(args.resume)
            if ckpt_path.is_dir():
                # 若傳入資料夾，預設從 best.pt 續訓
                ckpt_path = ckpt_path / "weights" / "best.pt"
            start_epoch = load_ckpt(ckpt_path, model, optimizer, scaler, map_location="cpu")
            if is_main(): print(f"[resume] loaded from {ckpt_path}, next epoch = {start_epoch+1}")
        except Exception as e:
            if is_main(): print(f"[resume] skip ({e})")

    # ----- 主訓練迴圈 -----
    best_metric_name = args.best_metric
    best_metric_val = -1e9
    header = ["epoch", "train_loss", "val_loss", "map50", "map5095", "lr"]
    patience_left = args.early_stop_patience if args.early_stop_patience else None

    for epoch in range(start_epoch, args.epochs):
        if dist_enabled and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_stats = {}
        if train_loader is not None:
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, scaler,
                accum_steps=max(1, cfg.get("accum", args.accum)),
                grad_clip=args.grad_clip
            )

        val_metrics, val_per_class, val_per_dets = {}, [], []
        if val_loader is not None:
            val_metrics, val_per_class, val_per_dets = evaluate(
                model.module if isinstance(model, DDP) else model, val_loader, device
            )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats.get("train_loss"),
            "val_loss": val_metrics.get("val_loss"),
            "map50": val_metrics.get("map50"),
            "map5095": val_metrics.get("map5095"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if is_main():
            append_csv_row(out_dir / "results.csv", header, row)

        # 每 epoch 覆蓋最新的 per-class 與 val 偵測輸出
        if is_main() and val_per_class:
            with (out_dir / "metrics" / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                cols = sorted({k for d in val_per_class for k in d.keys()})
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for d in val_per_class: w.writerow(d)

        if is_main() and val_per_dets:
            with (out_dir / "preds" / f"per_det_val.jsonl").open("w", encoding="utf-8") as f:
                for d in val_per_dets: f.write(json.dumps(d) + "\n")

        # checkpoint：每 epoch + best.pt
        if is_main():
            save_ckpt(out_dir / "weights" / f"epoch_{epoch+1:03d}.pt", model, optimizer, epoch, scaler)
            metric_for_best = val_metrics.get(best_metric_name)
            if metric_for_best is not None and float(metric_for_best) >= best_metric_val:
                best_metric_val = float(metric_for_best)
                save_ckpt(out_dir / "weights" / "best.pt", model, optimizer, epoch, scaler)
                if args.early_stop_patience:
                    patience_left = args.early_stop_patience
            elif args.early_stop_patience:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early-stop] no improvement on {best_metric_name} for {args.early_stop_patience} epochs.")
                    break

        # scheduler 每 epoch step
        if sched is not None:
            sched.step()

        # optional: 每 epoch 做一次 test
        if args.eval_test_each_epoch and test_loader is not None and is_main():
            tst_metrics, _, tst_per_dets = evaluate(
                model.module if isinstance(model, DDP) else model, test_loader, device
            )
            write_json(out_dir / "metrics" / f"test_metrics_epoch_{epoch+1:03d}.json", tst_metrics)
            if tst_per_dets:
                with (out_dir / "preds" / f"per_det_test_epoch_{epoch+1:03d}.jsonl").open("w", encoding="utf-8") as f:
                    for d in tst_per_dets: f.write(json.dumps(d) + "\n")

    # final test（一次）
    if test_loader is not None and is_main():
        tst_metrics, tst_per_class, tst_per_dets = evaluate(
            model.module if isinstance(model, DDP) else model, test_loader, device
        )
        write_json(out_dir / "metrics" / "test_metrics.json", tst_metrics)
        if tst_per_class:
            with (out_dir / "metrics" / "per_class_metrics_test.csv").open("w", newline="", encoding="utf-8") as f:
                cols = sorted({k for d in tst_per_class for k in d.keys()})
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for d in tst_per_class: w.writerow(d)
        if tst_per_dets:
            with (out_dir / "preds" / f"per_det_test.jsonl").open("w", encoding="utf-8") as f:
                for d in tst_per_dets: f.write(json.dumps(d) + "\n")

    return {"best_metric": best_metric_val}

def run_kfold(args, cfg):
    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    # 取得資料 ID：優先從 cfg["all_ids"]；否則請讓 build_dataloaders 回傳 meta["all_ids"]/["train_ids"]
    if "all_ids" in cfg:
        ids = list(cfg["all_ids"])
    else:
        tmp = build_dataloaders(cfg, fold_split=None, dist=False)
        meta = tmp.get("meta") or {}
        ids = meta.get("all_ids") or meta.get("train_ids")
        if not ids:
            raise RuntimeError("K-fold 需要 IDs；請在 cfg['all_ids'] 提供，或讓 build_dataloaders(meta['all_ids']) 回傳。")

    splits = make_kfold_splits(ids, args.kfold, args.seed)
    if args.save_splits:
        for sp in splits:
            write_json(out_root / "splits" / f"fold_{sp['fold']}.json", sp)

    import csv as _csv
    summary_rows = []
    for sp in splits:
        fold_k = sp["fold"]
        fold_out = out_root / f"fold_{fold_k}"
        args_single = SimpleNamespace(**vars(args))
        args_single.out = str(fold_out)

        device, _ = default_device(dist_enabled=args.dist)
        set_seed(args.seed + fold_k)
        result = run_once(args_single, cfg, device, dist_enabled=args.dist, fold_split=sp)
        summary_rows.append({"fold": fold_k, args.best_metric: result["best_metric"]})

    kf_path = out_root / f'kfold_{args.model_name}_val.csv'
    with kf_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["fold", args.best_metric]
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary_rows: w.writerow(r)


def main():
    args = SimpleNamespace(**CONFIG)
    cfg = load_cfg(args.cfg)
    set_seed(args.seed)

    if args.kfold and args.kfold > 0:
        run_kfold(args, cfg)
    else:
        device, _ = default_device(dist_enabled=args.dist)
        run_once(args, cfg, device, dist_enabled=args.dist)

    cleanup_ddp()


if __name__ == "__main__":
    main()
