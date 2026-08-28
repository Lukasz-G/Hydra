"""torch.distributed setup with a zero-config single-process fallback.

Launched via torchrun (RANK/WORLD_SIZE env set) -> process group with nccl
(Linux + CUDA) or gloo (Windows / CPU). Launched plainly -> single process,
no process group, cuda:0 or CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_main: bool
    is_distributed: bool


def init_distributed(backend: str = "auto") -> DistInfo:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistInfo(0, 1, 0, device, True, False)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # ranks beyond the local GPU count fall back to CPU (needs gloo)
    use_cuda = torch.cuda.is_available() and local_rank < torch.cuda.device_count()
    if backend == "auto":
        backend = "nccl" if use_cuda and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend=backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistInfo(rank, world_size, local_rank, device, rank == 0, True)


def barrier(info: DistInfo) -> None:
    if info.is_distributed:
        dist.barrier()


def broadcast_flag(info: DistInfo, flag: bool) -> bool:
    """Broadcast a boolean decision (e.g. early stop) from rank 0 to all ranks."""
    if not info.is_distributed:
        return flag
    # nccl needs a CUDA tensor; gloo needs CPU
    dev = info.device if dist.get_backend() == "nccl" else torch.device("cpu")
    t = torch.tensor([int(flag)], dtype=torch.int64, device=dev)
    dist.broadcast(t, src=0)
    return bool(t.item())


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def cleanup(info: DistInfo) -> None:
    if info.is_distributed:
        dist.destroy_process_group()
