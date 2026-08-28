"""Multi-head loss over slot targets with NULL down-weighting."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig
from .data import IGNORE
from .model import ModelOutput
from .vocab import NULL


def compute_loss(out: ModelOutput, batch: dict[str, torch.Tensor],
                 cfg: LossConfig, n_pos: int) -> tuple[torch.Tensor, dict[str, float]]:
    pos_t = batch["pos"]      # (B, T, K)
    morph_t = batch["morph"]  # (B, T, K)
    lemma_t = batch["lemma"]  # (B, T, K, L)

    weight = torch.ones(n_pos, device=out.pos_logits.device, dtype=out.pos_logits.dtype)
    weight[NULL] = cfg.null_weight

    l_pos = F.cross_entropy(out.pos_logits.reshape(-1, out.pos_logits.shape[-1]),
                            pos_t.reshape(-1), weight=weight, ignore_index=IGNORE)
    l_morph = F.cross_entropy(out.morph_logits.reshape(-1, out.morph_logits.shape[-1]),
                              morph_t.reshape(-1), ignore_index=IGNORE)
    l_lemma = F.cross_entropy(out.lemma_logits.reshape(-1, out.lemma_logits.shape[-1]),
                              lemma_t.reshape(-1), ignore_index=IGNORE)

    # a head with no supervised targets in the batch yields NaN -> treat as 0
    zero = out.pos_logits.sum() * 0.0
    l_pos = torch.where(torch.isnan(l_pos), zero, l_pos)
    l_morph = torch.where(torch.isnan(l_morph), zero, l_morph)
    l_lemma = torch.where(torch.isnan(l_lemma), zero, l_lemma)

    total = cfg.w_pos * l_pos + cfg.w_morph * l_morph + cfg.w_lemma * l_lemma
    parts = {"loss": float(total.detach()), "loss_pos": float(l_pos.detach()),
             "loss_morph": float(l_morph.detach()), "loss_lemma": float(l_lemma.detach())}
    return total, parts
