"""Checkpoint save/load with full training state including RNG for exact resume."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def collect_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    # set_rng_state requires CPU ByteTensors; a checkpoint loaded with a CUDA
    # map_location would otherwise hand over GPU tensors
    torch.set_rng_state(states["torch"].cpu())
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])


def save_checkpoint(path: str | Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, scaler,
                    epoch: int, step: int, best_metric: float, patience_left: int,
                    config_dict: dict[str, Any]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "patience_left": patience_left,
        "config": config_dict,
        "rng": collect_rng_states(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
