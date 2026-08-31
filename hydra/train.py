"""Training entry: single-process or DDP (torchrun), AMP, early stopping, resume."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import load_checkpoint, restore_rng_states, save_checkpoint
from .config import Config
from .data import HydraDataset, collate, load_split_tokens, resolve_splits
from .distributed import DistInfo, barrier, broadcast_flag, cleanup, init_distributed, unwrap
from .evaluate import evaluate_dataset
from .losses import compute_loss
from .metrics import JsonlLogger
from .model import HydraModel
from .vocab import Vocabs

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int,
                    total_steps: int, lr: float, lr_min: float):
    floor = lr_min / lr

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        span = max(1, total_steps - warmup_steps)
        t = min(1.0, (step - warmup_steps) / span)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def prepare_data(cfg: Config, info: DistInfo, run_dir: Path, resuming: bool = False):
    """Resolve splits and vocab (rank 0 writes, others wait), load datasets."""
    split_path = run_dir / "split.json"
    vocab_path = run_dir / "vocab.json"
    if info.is_main:
        splits = resolve_splits(cfg.data, run_dir)
    barrier(info)
    splits = json.loads(split_path.read_text(encoding="utf-8"))

    train_docs = load_split_tokens(splits["train"], cfg.data.on_mismatch, cfg.model.n_slots)
    # a resumed run must keep its vocab (label ids are baked into the model);
    # a fresh run must never inherit a stale one from a previous run_dir
    if info.is_main and not (resuming and vocab_path.exists()):
        all_train_tokens = [t for doc in train_docs for t in doc]
        Vocabs.build(all_train_tokens, cfg.data.lemma_type_min_freq,
                     cfg.data.word_type_min_freq).save(vocab_path)
    barrier(info)
    vocabs = Vocabs.load(vocab_path)

    train_ds = HydraDataset(train_docs, vocabs, cfg.data, cfg.model.n_slots,
                            training=True, mask_tokens=cfg.model.masked_lm)
    dev_ds = None
    if info.is_main:
        dev_docs = load_split_tokens(splits["dev"], cfg.data.on_mismatch, cfg.model.n_slots)
        dev_ds = HydraDataset(dev_docs, vocabs, cfg.data, cfg.model.n_slots)
    return vocabs, train_ds, dev_ds


def train(cfg: Config, resume: str | None = None,
          init_weights: str | None = None) -> dict[str, float]:
    info = init_distributed(cfg.distributed.backend)
    seed_everything(cfg.run.seed)
    run_dir = Path(cfg.run.run_dir)
    if info.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(cfg.to_dict(), indent=1), encoding="utf-8")
    barrier(info)

    logger = JsonlLogger(run_dir / "metrics.jsonl" if info.is_main else None,
                         echo=info.is_main)
    vocabs, train_ds, dev_ds = prepare_data(cfg, info, run_dir, resuming=resume is not None)
    if info.is_main:
        logger.log(event="data", train_chunks=len(train_ds),
                   dev_chunks=len(dev_ds) if dev_ds else 0,
                   n_chars=len(vocabs.chars), n_pos=len(vocabs.pos),
                   n_morph=len(vocabs.morph))

    model = HydraModel(cfg.model, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       cfg.data.max_word_len, cfg.data.max_lemma_len,
                       cfg.data.chunk_len, cfg.data.halo,
                       n_lemma_types=len(vocabs.lemma_types),
                       n_word_types=len(vocabs.word_types),
                       n_joint_types=len(vocabs.joint_types)).to(info.device)
    if init_weights and not resume:
        # warm-start from a compatible checkpoint: matching keys only, fresh
        # optimizer/schedule (e.g. adding the lemma classifier to a trained model)
        payload = load_checkpoint(init_weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if info.is_main:
            logger.log(event="init_weights", source=str(init_weights),
                       missing=len(missing), unexpected=len(unexpected))
    if info.is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.log(event="model", params=n_params)
    if info.is_distributed:
        model = DDP(model, device_ids=[info.local_rank] if info.device.type == "cuda" else None)

    sampler = DistributedSampler(train_ds, shuffle=True) if info.is_distributed else None
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_chunks,
                        shuffle=sampler is None, sampler=sampler, collate_fn=collate,
                        num_workers=cfg.data.num_workers, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay, betas=(0.9, 0.98))
    total_steps = len(loader) * cfg.train.max_epochs
    scheduler = build_scheduler(optimizer, cfg.train.warmup_steps, total_steps,
                                cfg.train.lr, cfg.train.lr_min)
    use_amp = cfg.train.amp and info.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, step = 0, 0
    best_metric = -1.0
    patience_left = cfg.train.patience
    if resume:
        # load to CPU: state_dicts move to the model's device on load_state_dict,
        # and RNG states must stay on CPU
        payload = load_checkpoint(resume, map_location="cpu")
        unwrap(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = payload["epoch"] + 1
        step = payload["step"]
        best_metric = payload["best_metric"]
        patience_left = payload["patience_left"]
        restore_rng_states(payload["rng"])
        if info.is_main:
            logger.log(event="resume", from_epoch=start_epoch, best=best_metric)

    snapper = None
    if info.is_main and cfg.infer.snap_lemmas and vocabs.lemma_counts:
        from .snap import LemmaSnapper
        snapper = LemmaSnapper(vocabs.lemma_inventory)

    last_dev: dict[str, float] = {}
    for epoch in range(start_epoch, cfg.train.max_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        running: dict[str, float] = {}
        n_running = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            chars = batch["chars"].to(info.device, non_blocking=True)
            targets = {k: batch[k].to(info.device, non_blocking=True)
                       for k in ("pos", "morph", "lemma", "lemtype", "joint", "mlm")}
            with torch.autocast(info.device.type, dtype=torch.float16, enabled=use_amp):
                out = model(chars)
                loss, parts = compute_loss(out, targets, cfg.loss, len(vocabs.pos))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n_running += 1
            if info.is_main and step % cfg.train.log_every_steps == 0:
                avg = {k: v / n_running for k, v in running.items()}
                logger.log(event="train", epoch=epoch, step=step,
                           lr=scheduler.get_last_lr()[0], **avg)
                running, n_running = {}, 0

        # ---- end of epoch: dev eval on rank 0, checkpoint, early stop ----
        stop = False
        if info.is_main:
            metric = best_metric
            if dev_ds is not None and len(dev_ds) > 0 and not cfg.model.pretrain_mlm:
                last_dev = evaluate_dataset(unwrap(model), dev_ds, vocabs, info.device,
                                            cfg.infer.batch_chunks, snapper=snapper,
                                            cls_min_prob=cfg.infer.classifier_min_prob)
                logger.log(event="dev", epoch=epoch, step=step,
                           epoch_seconds=round(time.time() - t0, 1), **last_dev)
                metric = last_dev.get("acc_lemma_pos", 0.0)
            improved = metric > best_metric + cfg.train.min_delta
            if improved:
                best_metric = metric
                patience_left = cfg.train.patience
            else:
                patience_left -= 1
            ckpt_args = dict(model=unwrap(model), optimizer=optimizer, scheduler=scheduler,
                             scaler=scaler, epoch=epoch, step=step, best_metric=best_metric,
                             patience_left=patience_left, config_dict=cfg.to_dict())
            save_checkpoint(run_dir / "last.pt", **ckpt_args)
            if improved:
                save_checkpoint(run_dir / "best.pt", **ckpt_args)
                logger.log(event="best", epoch=epoch, metric=best_metric)
            stop = patience_left <= 0
        stop = broadcast_flag(info, stop)
        if stop:
            if info.is_main:
                logger.log(event="early_stop", epoch=epoch, best=best_metric)
            break

    barrier(info)
    cleanup(info)
    return last_dev
