from __future__ import annotations
import json, csv
from pathlib import Path
from typing import Any, List, Dict
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def write_json(p: Path, obj: Any):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]):
    write_header = (not csv_path.exists())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k) for k in header})

def save_ckpt(path: Path, model, optimizer, epoch: int, scaler=None):
    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler is not None else None
    }
    torch.save(state, path)

def load_ckpt(path: Path, model, optimizer=None, scaler=None, map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"], strict=True)
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", -1)) + 1

def load_cfg(path: str|Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
        except Exception:
            raise RuntimeError("Please `pip install pyyaml` to parse YAML config.")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Config parse failed for {path}: {e}")

