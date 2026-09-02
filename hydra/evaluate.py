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


def evaluate_dataset(model: torch.nn.Module, ds: HydraDataset, vocabs: Vocabs,
                     device: torch.device, batch_chunks: int,
                     snapper: LemmaSnapper | None = None,
                     cls_min_prob: float = 0.5) -> dict[str, float]:
    model.eval()
    acc_all = EvalAccumulator()
    acc_clean = EvalAccumulator()
    snap_ok_all = snap_ok_clean = 0
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.empty_cache()  # release training caches before the big eval tensors
    with torch.inference_mode():
        for lo in range(0, len(ds), batch_chunks):
            idxs = list(range(lo, min(lo + batch_chunks, len(ds))))
            batch = collate([ds[i] for i in idxs])
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["chars"].to(device))
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
