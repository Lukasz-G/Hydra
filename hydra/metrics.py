"""Slot decoding to strings, accuracy metrics, JSONL logging."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import ModelOutput
from .vocab import LABEL_UNK, NULL, Vocabs


@dataclass
class Prediction:
    lemma: str  # '+'-joined
    pos: str
    morph: str


def decode_batch(out: ModelOutput, vocabs: Vocabs, surfaces: list[list[str]],
                 cls_min_prob: float = 0.5, model=None) -> list[list[Prediction]]:
    """Greedy prefix decoding: read slots until the first NULL POS (slot 0 is
    forced non-NULL). Lemma chars up to the first EOW; empty lemma falls back
    to the surface form. With the classify-or-generate head, a slot's lemma is
    the classifier's choice only when it is not UNK ("not a known type") AND
    its softmax probability reaches cls_min_prob; otherwise the generated
    characters are used. Returns per (chunk, central position) predictions."""
    pos_ids = out.pos_logits.argmax(dim=-1)                       # (B, T, K)
    pos_ids0 = out.pos_logits[..., 0, :].clone()                  # slot 0: mask NULL
    pos_ids0[..., NULL] = torch.finfo(pos_ids0.dtype).min
    pos_ids[..., 0] = pos_ids0.argmax(dim=-1)
    # argmax over classes >= 1 skips NULL without materializing a masked copy
    morph_ids = out.morph_logits[..., 1:].argmax(dim=-1) + 1       # (B, T, K)
    if out.lemma_logits is not None:
        char_ids = out.lemma_logits.argmax(dim=-1)                 # (B, T, K, L)
    else:
        # autoregressive decoder: generate char ids step by step
        Bq, Tq, Kq = pos_ids.shape
        gen = model.lemma_decoder.generate(out.slot_states, out.char_states,
                                           out.char_pad_mask, model.max_lemma_len)
        char_ids = gen.view(Bq, Tq, Kq, -1)

    cls_ids = None
    if out.lemma_cls_logits is not None:
        # confidence without materializing a softmax over ~36k classes:
        # p(argmax) = exp(max_logit - logsumexp); only (B,T,K)-sized outputs
        cls = out.lemma_cls_logits[..., 1:]
        cls_max, cls_arg = cls.max(dim=-1)
        cls_prob = (cls_max.float() - torch.logsumexp(cls, dim=-1).float()).exp()
        cls_arg = cls_arg + 1                                      # undo NULL skip
        cls_arg[cls_prob < cls_min_prob] = LABEL_UNK               # low confidence -> generate
        cls_ids = cls_arg.cpu().numpy()                            # (B, T, K)

    pos_ids = pos_ids.cpu().numpy()
    morph_ids = morph_ids.cpu().numpy()
    char_ids = char_ids.cpu().numpy()

    B, T, K = pos_ids.shape
    results: list[list[Prediction]] = []
    for b in range(B):
        row: list[Prediction] = []
        for t in range(T):
            lemmas, poss, morphs = [], [], []
            for k in range(K):
                if k > 0 and pos_ids[b, t, k] == NULL:
                    break
                lemma = ""
                if cls_ids is not None and cls_ids[b, t, k] != LABEL_UNK:
                    lemma = vocabs.lemma_types.decode(cls_ids[b, t, k])
                if not lemma:
                    lemma = vocabs.chars.decode(char_ids[b, t, k].tolist())
                if not lemma:
                    lemma = surfaces[b][t]
                lemmas.append(lemma)
                poss.append(vocabs.pos.decode(pos_ids[b, t, k]))
                morphs.append(vocabs.morph.decode(morph_ids[b, t, k]))
            row.append(Prediction("+".join(lemmas), "+".join(poss), "+".join(morphs)))
        results.append(row)
    return results


@dataclass
class AccuracyCounts:
    n: int = 0
    lemma: int = 0
    pos: int = 0
    morph: int = 0
    joint: int = 0            # all three correct
    lemma_pos_joint: int = 0  # model-selection metric

    def update(self, pred: Prediction, gold_lemma: str, gold_pos: str, gold_morph: str) -> None:
        self.n += 1
        okl = pred.lemma == gold_lemma
        okp = pred.pos == gold_pos
        okm = pred.morph == gold_morph
        self.lemma += okl
        self.pos += okp
        self.morph += okm
        self.joint += okl and okp and okm
        self.lemma_pos_joint += okl and okp

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        if self.n == 0:
            return {f"{prefix}n": 0}
        return {
            f"{prefix}n": self.n,
            f"{prefix}acc_lemma": self.lemma / self.n,
            f"{prefix}acc_pos": self.pos / self.n,
            f"{prefix}acc_morph": self.morph / self.n,
            f"{prefix}acc_joint": self.joint / self.n,
            f"{prefix}acc_lemma_pos": self.lemma_pos_joint / self.n,
        }


@dataclass
class EvalAccumulator:
    """Overall + multi-item + OOV accuracy over a split."""
    overall: AccuracyCounts = field(default_factory=AccuracyCounts)
    multi: AccuracyCounts = field(default_factory=AccuracyCounts)
    oov: AccuracyCounts = field(default_factory=AccuracyCounts)
    lemma_dist: int = 0  # summed Levenshtein over evaluated tokens

    def update(self, pred: Prediction, surface: str, gold_lemma: str, gold_pos: str,
               gold_morph: str, n_items: int, train_surfaces: set[str]) -> None:
        self.overall.update(pred, gold_lemma, gold_pos, gold_morph)
        if n_items > 1:
            self.multi.update(pred, gold_lemma, gold_pos, gold_morph)
        if surface not in train_surfaces:
            self.oov.update(pred, gold_lemma, gold_pos, gold_morph)
        self.lemma_dist += levenshtein(pred.lemma, gold_lemma)

    def as_dict(self) -> dict[str, float]:
        d = self.overall.as_dict()
        d.update(self.multi.as_dict("multi_"))
        d.update(self.oov.as_dict("oov_"))
        if self.overall.n:
            d["lemma_levenshtein"] = self.lemma_dist / self.overall.n
        return d


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class JsonlLogger:
    """Rank-0 metrics log: one JSON object per line, mirrored to console."""

    def __init__(self, path: str | Path | None, echo: bool = True):
        self.path = Path(path) if path is not None else None
        self.echo = echo
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **event) -> None:
        event.setdefault("time", round(time.time(), 2))
        line = json.dumps(event, ensure_ascii=False)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if self.echo:
            shown = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in event.items() if k != "time"}
            print(" ".join(f"{k}={v}" for k, v in shown.items()), flush=True)
