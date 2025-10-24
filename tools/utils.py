from __future__ import annotations
import os, sys, json, subprocess, random
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def default_device(dist_enabled: bool):
    if dist_enabled and "LOCAL_RANK" in os.environ:
        local_rank = setup_ddp()
        return torch.device(f"cuda:{local_rank}"), local_rank
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def capture_env(out_dir: Path):
    info = {}
    info["python"] = sys.version
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if hasattr(torch.version, "cuda") else None
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        info["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        info["git_commit"] = None
    out_dir.joinpath("env.txt").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

def set_torch_deterministic(flag: bool = True):
    import torch, os
    torch.backends.cudnn.deterministic = flag
    torch.backends.cudnn.benchmark = not flag
    if flag:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA >= 10.2
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def get_rank() -> int:
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else 0

# 
def is_distributed() -> bool:
    import torch.distributed as dist
    return dist.is_initialized()
