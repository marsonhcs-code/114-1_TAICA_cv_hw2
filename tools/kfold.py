from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def make_kfold_splits(ids: List[Any], K: int, seed: int) -> List[Dict[str, List[Any]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    splits = []
    for k in range(K):
        val_idx = set(folds[k].tolist())
        train_ids = [ids[i] for i in range(len(ids)) if i not in val_idx]
        val_ids = [ids[i] for i in folds[k].tolist()]
        splits.append({"fold": k+1, "train_ids": train_ids, "val_ids": val_ids})
    return splits
