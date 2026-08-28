"""Evaluation of a model over a gold dataset: overall / multi-item / OOV metrics."""
from __future__ import annotations

import torch

from .data import HydraDataset, collate
from .metrics import EvalAccumulator, decode_batch
from .vocab import Vocabs


def evaluate_dataset(model: torch.nn.Module, ds: HydraDataset, vocabs: Vocabs,
                     device: torch.device, batch_chunks: int) -> dict[str, float]:
    model.eval()
    acc = EvalAccumulator()
    with torch.inference_mode():
        for lo in range(0, len(ds), batch_chunks):
            idxs = list(range(lo, min(lo + batch_chunks, len(ds))))
            batch = collate([ds[i] for i in idxs])
            out = model(batch["chars"].to(device))
            surfaces = [ds.chunk_surfaces(i) for i in idxs]
            preds = decode_batch(out, vocabs, surfaces)
            golds = [ds.chunk_gold(i) for i in idxs]
            n_items = batch["n_items"].numpy()
            for b in range(len(idxs)):
                for t, gold in enumerate(golds[b]):
                    if gold is None:
                        continue
                    acc.update(preds[b][t], surfaces[b][t], gold[0], gold[1], gold[2],
                               int(n_items[b, t]), vocabs.train_surfaces)
    model.train()
    return acc.as_dict()
