your_project/
├─ train.py                          # ✅ 保持不變
├─ hooks.py                          # 🔧 需大幅修改
├─ configs/
│   ├─ exp_baseline.yaml             # 🆕 Baseline（無 long-tail）
│   ├─ exp_cb_loss.yaml              # 🆕 CB-Focal Loss
│   ├─ exp_cb_rfs.yaml               # 🆕 CB-Focal + Repeat Factor Sampling (RFS)
│   ├─ exp_longtail.yaml             # 🆕 Full pipeline
│   └─ exp_two_stage.yaml.yaml       # 🆕 Two-Stage Training (Optional)
├─ tools/
│   ├─ utils.py                      # ✅ 保持不變
│   ├─ io.py                         # ✅ 保持不變
│   ├─ kfold.py                      # ✅ 保持不變
│   └─ longtail_utils.py             # 🆕 分析工具
├─ models/
│   ├─ yolo_backbone.py              # 🆕 YOLO backbone（從零實作）
│   ├─ yolo_neck.py                  # 🆕 PAFPN
│   ├─ yolo_head.py                  # 🆕 Detection head
│   └─ yolo_detector.py              # 🆕 完整模型
├─ losses/
│   ├─ focal_loss.py                 # 🆕 Focal Loss
│   ├─ cb_loss.py                    # 🆕 Class-Balanced Loss
│   └─ yolo_loss.py                  # 🆕 YOLO Loss + Long-Tail
├─ data/
│   ├─ dataset.py                    # 🆕 DroneTrafficDataset
│   ├─ sampler.py                    # 🆕 RepeatFactorSampler
│   ├─  augmentation.py              # 🆕 Class-aware augmentation
│   ├─ images/
│   ├─   ├─ img_0001.jpg
│   ├─   ├─ img_0002.jpg
│   ├─   └─ ...
│   └─ annotations.json
└─ log/
    ├─ run_model**.log
    └─ run.pid

