# losses/cb_focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class ClassBalancedFocalLoss(nn.Module):
    """
    CB-Focal Loss (BCE-Style)
    
    取代 nn.BCEWithLogitsLoss
    
    適用於 YOLOv8 的 Class Head (multi-label)
    
    Loss = CB_Weight * Focal_Weight * BCE_Loss
    """
    
    def __init__(self, 
                 samples_per_cls: List[int],  # [23000, 5000, 3000, 1000]
                 beta: float = 0.9999,
                 gamma: float = 2.0,
                 reduction: str = 'none'): # 必須是 'none' 才能讓 v8DetectionLoss 正確 reduction
        super().__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        
        # --- Class-Balanced Weights ---
        # E_n = (1 - beta^n)
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        # W_c = (1 - beta) / E_n
        weights = (1.0 - beta) / np.array(effective_num)
        
        # 歸一化 (可選，但 v8 loss 會自己加權，這裡先保持 LVIS 論文的作法)
        # weights = weights / weights.sum() * len(weights) 
        
        # [NC]
        self.register_buffer('cb_weights', torch.tensor(weights, dtype=torch.float32))
        print(f"[CB-Loss] weights: {self.cb_weights}")

    def forward(self, 
                logits: torch.Tensor,   # [B * N, NC] (raw logits)
                targets: torch.Tensor   # [B * N, NC] (0.0 or 1.0)
                ) -> torch.Tensor:      # [B * N, NC] (loss per element)
        
        # --- 1. Standard BCE Loss (with logits) ---
        # pos_weight=None, F.binary_cross_entropy_with_logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # --- 2. Focal Weight ---
        p = torch.sigmoid(logits)
        # P_t: if target=1, p_t = p; if target=0, p_t = 1-p
        p_t = p * targets + (1 - p) * (1 - targets)
        # Focal: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # --- 3. Class-Balanced Weight ---
        # W_c: [NC]
        # targets: [B*N, NC]
        
        # if target=1, use W_c; if target=0, use 1.0
        # cb_weight: [B*N, NC]
        cb_weight = self.cb_weights.view(1, -1) * targets + (1 - targets)
        
        # --- 4. 組合 ---
        loss = cb_weight * focal_weight * bce_loss
        
        if self.reduction == 'none':
            return loss # [B*N, NC]
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")