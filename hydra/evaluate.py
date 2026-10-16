"""Evaluation of a model over a gold dataset.

Reports overall / multi-item / OOV metrics, everything twice: over all
supervised tokens and over the "clean" subset that excludes damaged tokens
(gold POS '--', lemma '[!]' — unannotatable text). Also reports lemma accuracy
with lexicon snapping applied when a snapper is given.
"""
from __future__ import annotations

import torch

from .data import HydraDataset, collate
from .metrics import EvalAccumulator, decode_batch
from .snap import LemmaSnapper
from .vocab import Vocabs

DAMAGED_POS = "--"


def bf16_in_hardware() -> bool:
    """True only where bfloat16 runs on the silicon, not under emulation."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:                       # torch < 2.6 has no such flag
        return torch.cuda.get_device_capability()[0] >= 8


def evaluate_dataset(model: torch.nn.Module, ds: HydraDataset, vocabs: Vocabs,
                     device: torch.device, batch_chunks: int,
                     snapper: LemmaSnapper | None = None,
                     cls_min_prob: float = 0.5,
                     tag_oracle: bool = False) -> dict[str, float]:
    model.eval()
    acc_all = EvalAccumulator()
    acc_clean = EvalAccumulator()
    snap_ok_all = snap_ok_clean = 0
    use_amp = device.type == "cuda"
    # bf16 where the GPU supports it IN HARDWARE: fp32-range exponent avoids
    # activation overflow on large models (and matches bf16-trained runs).
    #
    # is_bf16_supported() alone is not that test. Since torch 2.6 it answers
    # for emulation too, so a Turing card (sm_75, no bf16 silicon) says yes and
    # then runs the emulated path: measured on a Quadro RTX 5000, one 128-token
    # chunk took 15.4s in bf16 against 0.13s in fp16, a factor of 118. The eval
    # is correct either way, which is why this hid: it only ever looked slow.
    amp_dtype = torch.float16
    if use_amp and bf16_in_hardware():
        amp_dtype = torch.bfloat16
    if use_amp:
        torch.cuda.empty_cache()  # release training caches before the big eval tensors
    with torch.inference_mode():
        for lo in range(0, len(ds), batch_chunks):
            idxs = list(range(lo, min(lo + batch_chunks, len(ds))))
            batch = collate([ds[i] for i in idxs])
            # oracle diagnostic: feed GOLD tags into the tag_condition cascade,
            # separating "does conditioning help" from "does the tag head's own
            # error rate eat the gain". Not for reported numbers.
            tag_teacher = ((batch["pos"].to(device), batch["morph"].to(device))
                           if tag_oracle else None)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch["chars"].to(device), tag_teacher=tag_teacher)
            surfaces = [ds.chunk_surfaces(i) for i in idxs]
            preds = decode_batch(out, vocabs, surfaces, cls_min_prob, model=model)
            golds = [ds.chunk_gold(i) for i in idxs]
            n_items = batch["n_items"].numpy()
            for b in range(len(idxs)):
                for t, gold in enumerate(golds[b]):
                    if gold is None:
                        continue
                    p = preds[b][t]
                    n = int(n_items[b, t])
                    clean = gold[1] != DAMAGED_POS
                    acc_all.update(p, surfaces[b][t], gold[0], gold[1], gold[2],
                                   n, vocabs.train_surfaces)
                    if clean:
                        acc_clean.update(p, surfaces[b][t], gold[0], gold[1], gold[2],
                                         n, vocabs.train_surfaces)
                    if snapper is not None:
                        snapped = snapper.snap(p.lemma, len(p.pos.split("+")))
                        ok = snapped == gold[0]
                        snap_ok_all += ok
                        snap_ok_clean += ok and clean
    model.train()
    metrics = acc_all.as_dict()
    for k, v in acc_clean.as_dict().items():
        metrics[f"clean_{k}"] = v
    if snapper is not None and acc_all.overall.n:
        metrics["acc_lemma_snapped"] = snap_ok_all / acc_all.overall.n
        if acc_clean.overall.n:
            metrics["clean_acc_lemma_snapped"] = snap_ok_clean / acc_clean.overall.n
    return metrics
