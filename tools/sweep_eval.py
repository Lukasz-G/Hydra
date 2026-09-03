"""Dev tau sweep + test evaluation for a finished run (the batch-log format).

Usage:
  python tools/sweep_eval.py RUN_DIR [--taus=0.3,0.5,0.7,0.9] [--metric=acc_lemma_pos]

Loads RUN_DIR/best.pt, sweeps the classifier confidence gate tau on dev,
picks the best by --metric, evaluates test once at that tau. Prints the
'dev tau=..' / 'BEST_TAU' / 'TEST tau=..' lines that the batch results logs
(and journal/make_numbers.py) parse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.data import HydraDataset, load_split_tokens
from hydra.evaluate import evaluate_dataset
from hydra.snap import LemmaSnapper
from hydra.tag import load_model_for_inference

REPORT_KEYS = ("acc_lemma", "acc_lemma_snapped", "acc_pos", "acc_morph",
               "acc_lemma_pos", "acc_joint", "multi_acc_lemma", "multi_acc_joint",
               "oov_acc_lemma", "clean_acc_lemma", "clean_acc_lemma_snapped",
               "oov_n", "multi_n", "n")


def fmt(m: dict) -> str:
    parts = []
    for k in REPORT_KEYS:
        v = m.get(k)
        if v is None:
            parts.append(f"{k}=NA")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def main() -> None:
    run_dir = Path(sys.argv[1])
    taus = [0.3, 0.5, 0.7, 0.9]
    metric = "acc_lemma_pos"
    for a in sys.argv[2:]:
        if a.startswith("--taus="):
            taus = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--metric="):
            metric = a.split("=", 1)[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, vocabs, cfg = load_model_for_inference(run_dir / "best.pt", device)
    splits = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    snapper = LemmaSnapper(vocabs.lemma_inventory) if vocabs.lemma_counts else None
    chunk_mode = cfg.data.split_mode == "chunk" and cfg.data.train_dir is None

    def dataset(split: str) -> HydraDataset:
        docs = load_split_tokens(splits[split], cfg.data.on_mismatch, cfg.model.n_slots)
        return HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots,
                            role=split if chunk_mode else None)

    dev = dataset("dev")
    best_tau, best_val = taus[0], -1.0
    for tau in taus:
        m = evaluate_dataset(model, dev, vocabs, device, cfg.infer.batch_chunks,
                             snapper=snapper, cls_min_prob=tau)
        print(f"dev tau={tau}: {fmt(m)}", flush=True)
        if m.get(metric, 0.0) > best_val:
            best_tau, best_val = tau, m.get(metric, 0.0)
    print(f"BEST_TAU {best_tau}", flush=True)
    del dev

    test = dataset("test")
    m = evaluate_dataset(model, test, vocabs, device, cfg.infer.batch_chunks,
                         snapper=snapper, cls_min_prob=best_tau)
    print(f"TEST tau={best_tau}: {fmt(m)}", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
