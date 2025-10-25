# ROQ Roadmap — 實驗總覽
---  
## 快速目錄
1. 實驗 1 — Baseline                               - exp_baseline.yaml
2. 實驗 2 — CB-Focal Loss                          - exp_cb_loss.yaml
3. 實驗 3 — CB-Focal + RFS                         - exp_cb_rfs.yaml
4. 實驗 4 — Full Pipeline (CB + RFS + Strong Aug)  - exp_longtail.yaml
5. 實驗 5 — Two-Stage Training (Optional)          - exp_two_stage.yaml
------

## 實驗 1: Baseline (no long-tail handling)
- Config: **exp_baseline.yaml**
  - use_cb_loss: False
  - use_repeat_factor_sampling: False- Epochs: 200（從零訓練需更多 epochs）  
- Purpose: 建立 baseline

```yaml
# CONFIG (Baseline)
note: "Baseline YOLO without long-tail strategies"
cfg: "configs/exp_baseline.yaml"   # use_cb_loss=False, use_repeat_factor_sampling=False
out: "runs/exp1_baseline"
epochs: 100
```

預期結果：
- Overall mAP: ~65%  
- Car AP: ~80%  
- HOV AP: ~25%  ← 目標改善重點

---

## 實驗 2: CB-Focal Loss
- Config: **exp_cb_loss.yaml**
  - use_cb_loss: True
  - use_repeat_factor_sampling: False- Purpose: 驗證 loss function 對 tail 的改善

```yaml
# CONFIG (CB-Focal)
note: "Add Class-Balanced Focal Loss"
cfg: "configs/exp_cb_loss.yaml"   # use_cb_loss=True
out: "runs/exp2_cb_loss"
epochs: 100
```

預期改善：
- Overall mAP: 65% → 67% (+2%)  
- HOV AP: 25% → 35% (+10%)

---

## 實驗 3: CB-Focal + Repeat Factor Sampling (RFS)
- Config: **exp_cb_rfs.yaml**
  - use_cb_loss: True
  - use_repeat_factor_sampling: True- Epochs: 250（RFS 建議更多 epochs）  
- Purpose: 驗證 sampling 策略

```yaml
# CONFIG (CB + RFS)
note: "CB Loss + Repeat Factor Sampling"
cfg: "configs/exp_cb_rfs.yaml"   # use_repeat_factor_sampling=True
out: "runs/exp3_cb_rfs"
epochs: 120   # 實際可調整為 250 視訓練策略
```

預期改善：
- Overall mAP: 67% → 70% (+3%)  
- HOV AP: 35% → 45% (+10%)

---

## 實驗 4: Full Pipeline (CB + RFS + Strong Aug)
- Config: **exp_longtail.yaml**
  - use_cb_loss: True
  - use_repeat_factor_sampling: True
  - strong_augmentation: True  
- Purpose: 驗證 augmentation 與整合效果

```yaml
# CONFIG (Full long-tail pipeline)
note: "Full long-tail pipeline"
cfg: "configs/exp_longtail.yaml"   # All strategies enabled
out: "runs/exp4_full"
epochs: 150
accum: 4
```

預期改善：
- Overall mAP: 70% → 73% (+3%)  
- HOV AP: 45% → 55% (+10%)

---

## 實驗 5: Two-Stage Training (Optional)
- Config: **exp_two_stage.yaml**
  - two_stage_training: True
  - stage1_epochs: 100
  - stage2_epochs: 50
- Epochs: 200 + 100（建議 stage1 pretrain → stage2 tail fine-tune）  
- Purpose: 進一步提升 tail classes

- Epochs: 200 + 100（建議 stage1 pretrain → stage2 tail fine-tune）  
- Purpose: 進一步提升 tail classes

```yaml
# CONFIG (Two-stage)
note: "Two-stage: pretrain → tail fine-tune"
cfg: "configs/exp_two_stage.yaml"
out: "runs/exp5_two_stage"
epochs: 150   # 100 stage1 + 50 stage2 (示意)
longtail_config:
  enable_two_stage: true
  stage1_epochs: 100
  stage2_epochs: 50
```

預期改善：
- Overall mAP: 73% → 74% (+1%)  
- HOV AP: 55% → 60% (+5%)
