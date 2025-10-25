# data_process/sampler.py
# 實作 LVIS 的 Repeat Factor Sampling
import torch
import numpy as np
from torch.utils.data import Sampler, DistributedSampler
from typing import Optional, Iterator, Sized
import torch.distributed as dist

class RepeatFactorSampler(Sampler):
    """
    Repeat Factor Sampling (RFS)
    
    1. 計算每個類別的 frequency `f(c)`
    2. 計算 repeat factor `r(c) = max(1, sqrt(t / f(c)))`, t 是 median frequency
    3. 對於一張圖片 `I`，其 repeat factor `r(I) = max(r(c) for c in I)`
    4. 依據 `r(I)` 對圖片進行 oversample
    """
    
    def __init__(self, dataset: Sized, repeat_thresh: float = 0.001):
        self.dataset = dataset
        self.repeat_thresh = repeat_thresh # LVIS 論文中的 t (threshold)
        
        # --- 1. 計算 class frequencies ---
        # 假設 dataset 有 .get_class_frequencies()
        class_freq = dataset.get_class_frequencies()
        if not class_freq:
            print("Warning: [RFS] class_freq is empty. RFS will be disabled.")
            self.repeat_factors = [1.0] * len(self.dataset)
        else:
            freqs = np.array(list(class_freq.values()))
            # LVIS 論文使用 median frequency 作為 threshold
            # t = np.median(freqs)
            
            # 另一個常見作法是用 repeat_thresh (e.g., 1e-3)
            # 這裡我們用 config 傳入的 repeat_thresh
            # Note: 您的 `sampler.py` 範例 似乎混用了 t (threshold) 和 median_freq
            # 我們採用 median_freq 的作法，因為它更 auto-adaptive
            
            median_freq = np.median(freqs)
            print(f"[RFS] Median frequency: {median_freq}")
            
            # --- 2. 計算 repeat factor for classes ---
            class_rf = {}
            for cls_id, freq in class_freq.items():
                rf = max(1.0, np.sqrt(median_freq / max(freq, 1)))
                class_rf[cls_id] = rf
            
            # --- 3. 計算 repeat factor for images ---
            self.repeat_factors = []
            for idx in range(len(dataset)):
                # 假設 dataset 有 .get_image_classes()
                classes_in_image = dataset.get_image_classes(idx)
                if not classes_in_image:
                    self.repeat_factors.append(1.0)
                    continue
                
                max_rf = 1.0
                for cls in classes_in_image:
                    max_rf = max(max_rf, class_rf.get(cls, 1.0))
                
                self.repeat_factors.append(max_rf)

        # --- 4. 建立 oversampled indices ---
        self.weights = torch.tensor(self.repeat_factors, dtype=torch.float)
        self.num_samples = int(torch.sum(self.weights).item())
        print(f"[RFS] Original dataset size: {len(dataset)}, RFS size: {self.num_samples}")

    def __iter__(self) -> Iterator[int]:
        # 使用 torch.multinomial 依據 repeat factors 進行採樣
        # 這比 LVIS 論文的 ceil(rf) 更平滑
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class DistributedRepeatFactorSampler(DistributedSampler):
    """
    DDP 版本的 RFS
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0,
                 repeat_thresh=0.001):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        
        # --- 建立 RFS 的權重 ---
        class_freq = dataset.get_class_frequencies()
        if not class_freq:
            self.weights = torch.ones(len(dataset), dtype=torch.float)
        else:
            freqs = np.array(list(class_freq.values()))
            median_freq = np.median(freqs)
            
            class_rf = {}
            for cls_id, freq in class_freq.items():
                rf = max(1.0, np.sqrt(median_freq / max(freq, 1)))
                class_rf[cls_id] = rf
            
            repeat_factors = []
            for idx in range(len(dataset)):
                classes_in_image = dataset.get_image_classes(idx)
                if not classes_in_image:
                    repeat_factors.append(1.0)
                    continue
                max_rf = 1.0
                for cls in classes_in_image:
                    max_rf = max(max_rf, class_rf.get(cls, 1.0))
                repeat_factors.append(max_rf)
                
            self.weights = torch.tensor(repeat_factors, dtype=torch.float)

        print(f"[DDP-RFS] Rank {self.rank} initialized.")

    def __iter__(self) -> Iterator[int]:
        # --- 1. 生成 RFS indices ---
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 依據權重採樣，產生一個 oversampled index list
        # 長度與 DDP rank 0 的 RFS 相同 (num_samples = sum(weights))
        num_samples_total = int(torch.sum(self.weights).item())
        
        # 確保 DDP 下每個 rank 採樣的總數一致
        # 這裡我們不
        indices = torch.multinomial(self.weights, num_samples_total, replacement=True, generator=g).tolist()

        # --- 2. DDP 切分 ---
        if not self.drop_last:
            # 補齊，確保所有 DDP process 拿到一樣多的
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * int(np.ceil(padding_size / len(indices))))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # 切分給當前的 rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        # self.num_samples 已由 DistributedSampler 父類計算
        return self.num_samples