"""Score runs on MULTI-ITEM tokens, across architectures that disagree about
what an item is.

Why this exists
---------------
The combined-tag ablation (data.combined_tags, paper SS6.2) makes every token
a single item by construction, so its multi_acc_lemma is NA and the ordinary
sweep cannot answer the question the ablation was built for: does the K=8 slot
decoder actually beat a crude combined tag on multi-item tokens? That gap is
-4.01pp against RNNTagger, 4.7x the seed noise -- the one effect in this area
big enough to measure.

How it stays fair
-----------------
Both architectures emit a '+'-joined string per token, and gold is '+'-joined
either way, so predictions are directly comparable. The multi-item subset is
defined by the GOLD POS string containing '+', which is identical for both
arms -- not by each model's own idea of how many items it produced.

The arms are verified to be token-aligned before anything is scored: same
length, same surfaces, same gold. They are only aligned because
data.align_max_items pins the malformed-token limit to the baseline's slot
count; without it the ablation silently drops every multi-item token. The
assertion here is the backstop for that.

Usage:
  python tools/multi_item_eval.py RUN_DIR [RUN_DIR ...] [--split test]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.data import HydraDataset, collate, load_split_tokens  # noqa: E402
from hydra.metrics import decode_batch                            # noqa: E402
from hydra.snap import LemmaSnapper                               # noqa: E402
from hydra.tag import load_model_for_inference                    # noqa: E402


def flat_predictions(run_dir: Path, split: str, device: torch.device):
    """Every supervised token of the split, in file order, as
    (surface, gold_lemma, gold_pos, pred_lemma, pred_pos)."""
    model, vocabs, cfg = load_model_for_inference(run_dir / "model_only.pt", device)
    model.tag_cond_min_prob = cfg.infer.tag_cond_min_prob
    model.count_min_prob = cfg.infer.count_min_prob
    splits = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    docs = load_split_tokens(splits[split], cfg.data.on_mismatch, cfg.model.n_slots,
                             cfg.data.combined_tags, cfg.data.align_max_items)
    ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots, role=None)
    snapper = LemmaSnapper(vocabs.lemma_inventory) if vocabs.lemma_counts else None

    rows = []
    bs = cfg.infer.batch_chunks
    with torch.inference_mode():
        for lo in range(0, len(ds), bs):
            idxs = list(range(lo, min(lo + bs, len(ds))))
            batch = collate([ds[i] for i in idxs])
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch["chars"].to(device))
            surfaces = [ds.chunk_surfaces(i) for i in idxs]
            preds = decode_batch(out, vocabs, surfaces, cfg.infer.classifier_min_prob,
                                 model=model)
            golds = [ds.chunk_gold(i) for i in idxs]
            for b in range(len(idxs)):
                for t, gold in enumerate(golds[b]):
                    if gold is None:
                        continue
                    p = preds[b][t]
                    lemma = p.lemma
                    if snapper is not None:
                        lemma = snapper.snap(lemma, len(p.pos.split("+")))
                    rows.append((surfaces[b][t], gold[0], gold[1], lemma, p.pos))
    return rows, cfg


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = "test"
    for a in sys.argv[1:]:
        if a.startswith("--split="):
            split = a.split("=", 1)[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = {}
    ref = None
    for rd in args:
        rd = Path(rd)
        rows, cfg = flat_predictions(rd, split, device)
        key = (len(rows), tuple(r[0] for r in rows[:500]), tuple(r[2] for r in rows[:500]))
        if ref is None:
            ref = key
        elif key != ref:
            raise SystemExit(
                f"{rd.name}: token stream does NOT align with the first run "
                f"({len(rows)} rows). Refusing to compare -- the subsets would "
                f"not be the same tokens. Check data.align_max_items."
            )
        results[rd.name] = (rows, cfg)

    print(f"split={split}  runs={list(results)}\n")
    hdr = f"{'run':16s} {'slots':>5} {'comb':>5} | {'n_multi':>8} {'lemma':>8} {'pos':>8} " \
          f"{'lem+pos':>8} | {'n_single':>9} {'lemma':>8} | {'n_all':>7} {'lemma':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, (rows, cfg) in results.items():
        multi = [r for r in rows if "+" in r[2]]      # gold POS has >1 item
        single = [r for r in rows if "+" not in r[2]]
        def acc(rs, i, j):
            return sum(1 for r in rs if r[i] == r[j]) / len(rs) if rs else float("nan")
        print(f"{name:16s} {cfg.model.n_slots:5d} {str(cfg.data.combined_tags):>5} | "
              f"{len(multi):8d} {acc(multi,3,1):8.4f} {acc(multi,4,2):8.4f} "
              f"{sum(1 for r in multi if r[3]==r[1] and r[4]==r[2])/max(len(multi),1):8.4f} | "
              f"{len(single):9d} {acc(single,3,1):8.4f} | "
              f"{len(rows):7d} {acc(rows,3,1):8.4f}")


if __name__ == "__main__":
    main()
