"""Score a baseline system's output with Hydra's metric conventions.

Usage:
  python tools/score_baseline.py GOLD_TSV PRED_FILE TRAIN_TSV [--pred=tagtab|pie]

GOLD_TSV: Pie-format gold file (token/lemma/pos/morph, blank sentence lines).
PRED_FILE formats:
  tagtab (RNNTagger chain): token<TAB>tag<TAB>lemma, where tag is 'POS|morph'
          (the '|' split recovers the columns) or bare POS
  pie:    token/lemma/pos/morph like the gold (pie tag output reshaped)
TRAIN_TSV: the training file, for the OOV surface inventory.

Alignment is positional over non-blank lines; token mismatches are counted
and tolerated up to 0.5% (systems may copy tokens through unchanged only —
anything above that aborts as a misalignment). Reports the standard block:
overall / multi-item / OOV, each also on the clean subset (gold POS != '--'),
in the 'TEST: k=v' format the batch logs use.
"""
from __future__ import annotations

import sys
from pathlib import Path

DAMAGED_POS = "--"


def read_gold(path: Path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            token, lemma, pos, morph = line.split("\t")
            rows.append((token, lemma, pos, morph))
    return rows


def read_pred(path: Path, fmt: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if fmt == "pie":
                token, lemma, pos, morph = (cols + ["", "", "", ""])[:4]
            else:  # tagtab: token, tag, lemma
                token = cols[0]
                tag = cols[1] if len(cols) > 1 else ""
                lemma = cols[2] if len(cols) > 2 else ""
                pos, _, morph = tag.partition("|")
            rows.append((token, lemma, pos, morph))
    return rows


class Bucket:
    def __init__(self):
        self.n = self.lemma = self.pos = self.morph = self.joint = self.lemma_pos = 0

    def update(self, g, p):
        self.n += 1
        lem = p[1] == g[1]
        pos = p[2] == g[2]
        morph = p[3] == g[3]
        self.lemma += lem
        self.pos += pos
        self.morph += morph
        self.joint += lem and pos and morph
        self.lemma_pos += lem and pos

    def report(self, prefix=""):
        if self.n == 0:
            return {f"{prefix}n": 0}
        return {
            f"{prefix}acc_lemma": self.lemma / self.n,
            f"{prefix}acc_pos": self.pos / self.n,
            f"{prefix}acc_morph": self.morph / self.n,
            f"{prefix}acc_lemma_pos": self.lemma_pos / self.n,
            f"{prefix}acc_joint": self.joint / self.n,
            f"{prefix}n": self.n,
        }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gold_path, pred_path, train_path = map(Path, args[:3])
    fmt = "tagtab"
    for a in sys.argv[1:]:
        if a.startswith("--pred="):
            fmt = a.split("=", 1)[1]

    gold = read_gold(gold_path)
    pred = read_pred(pred_path, fmt)
    if len(gold) != len(pred):
        sys.exit(f"length mismatch: {len(gold)} gold vs {len(pred)} pred tokens")
    train_surfaces = {r[0] for r in read_gold(train_path)}

    mismatch = sum(1 for g, p in zip(gold, pred) if g[0] != p[0])
    if mismatch / len(gold) > 0.005:
        sys.exit(f"token misalignment: {mismatch}/{len(gold)} surface mismatches")

    buckets = {k: Bucket() for k in
               ("", "multi_", "oov_", "clean_", "clean_multi_", "clean_oov_")}
    for g, p in zip(gold, pred):
        multi = "+" in g[2]
        oov = g[0] not in train_surfaces
        clean = g[2] != DAMAGED_POS
        buckets[""].update(g, p)
        if multi:
            buckets["multi_"].update(g, p)
        if oov:
            buckets["oov_"].update(g, p)
        if clean:
            buckets["clean_"].update(g, p)
            if multi:
                buckets["clean_multi_"].update(g, p)
            if oov:
                buckets["clean_oov_"].update(g, p)

    out: dict = {}
    for prefix, b in buckets.items():
        out.update(b.report(prefix))
    if mismatch:
        out["token_mismatches"] = mismatch
    print("TEST: " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
