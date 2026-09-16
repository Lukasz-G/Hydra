"""Break a pooled run's test numbers down by source corpus.

Usage:
  python tools/per_corpus_eval.py RUN_DIR [--tau=0.3] [--split=test]
                                          [--meta=meta/stage2_metadata.csv]
                                          [--field=corpus]

Why this exists. Stage 2 trains one model on ReM, ReF, ReN and LeA at once,
and a single pooled accuracy cannot answer the question the experiment is
actually for: does ReM -- the corpus every earlier number in this project is
measured on -- get BETTER from being trained alongside the others, or is the
pooled figure just ReF's 2.4M tokens carrying a weaker ReM? Those two worlds
produce the same aggregate. Only the breakdown separates them.

The split is by FILE (split_mode = 'stratified'), so a per-corpus subset of
the test split is exact: no chunk straddles two corpora and nothing leaks.
Everything else is held identical to tools/sweep_eval.py -- same weights, same
tau, same snapper -- so the per-corpus lines and the pooled TEST line are
directly comparable, and the pooled line is reproduced here as ALL to prove it.

OOV stays defined against the POOLED training surfaces, which is the honest
definition for a pooled model: a ReN word the model met in ReF training is not
unseen just because it is new to ReN.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.data import HydraDataset, load_split_tokens
from hydra.evaluate import evaluate_dataset
from hydra.snap import LemmaSnapper
from hydra.tag import load_model_for_inference

REPORT_KEYS = ("n", "acc_lemma", "acc_lemma_snapped", "acc_pos", "acc_morph",
               "acc_lemma_pos", "acc_joint", "multi_n", "multi_acc_lemma",
               "multi_acc_lemma_pos", "oov_n", "oov_acc_lemma",
               "clean_acc_lemma", "clean_acc_lemma_pos")


def fmt(m: dict) -> str:
    out = []
    for k in REPORT_KEYS:
        v = m.get(k)
        out.append(f"{k}=" + ("NA" if v is None else
                              f"{v:.4f}" if isinstance(v, float) else str(v)))
    return " ".join(out)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    tau, split = None, "test"
    meta_path, field = "meta/stage2_metadata.csv", "corpus"
    for a in sys.argv[2:]:
        if a.startswith("--tau="):
            tau = float(a.split("=", 1)[1])
        elif a.startswith("--split="):
            split = a.split("=", 1)[1]
        elif a.startswith("--meta="):
            meta_path = a.split("=", 1)[1]
        elif a.startswith("--field="):
            field = a.split("=", 1)[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_only.pt"
    model, vocabs, cfg = load_model_for_inference(ckpt, device)
    if tau is None:
        tau = cfg.infer.classifier_min_prob
    model.tag_cond_min_prob = cfg.infer.tag_cond_min_prob
    model.count_min_prob = cfg.infer.count_min_prob
    if hasattr(model, "lang_min_prob"):
        model.lang_min_prob = cfg.infer.lang_min_prob
    snapper = LemmaSnapper(vocabs.lemma_inventory) if vocabs.lemma_counts else None

    with open(meta_path, encoding="utf-8") as fh:
        group_of = {r["file"]: r[field] for r in csv.DictReader(fh) if r.get("file")}

    files = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))[split]
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        name = Path(f).name
        groups[group_of.get(name, group_of.get(Path(f).stem, "?"))].append(f)
    unknown = len(groups.get("?", []))
    if unknown:
        # silently bucketing these as '?' would quietly shrink a corpus's test
        # set and make its accuracy look like it came from more data than it did
        print(f"WARNING {unknown} {split} files not in {meta_path}", flush=True)

    def score(fs: list[str]) -> dict:
        docs = load_split_tokens(fs, cfg.data.on_mismatch, cfg.model.n_slots,
                                 cfg.data.combined_tags, cfg.data.align_max_items)
        ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots)
        return evaluate_dataset(model, ds, vocabs, device, cfg.infer.batch_chunks,
                                snapper=snapper, cls_min_prob=tau)

    print(f"PERCORPUS_START {run_dir.name} split={split} tau={tau} "
          f"groups={len(groups)}", flush=True)
    for name in sorted(groups):
        print(f"PERCORPUS {name} files={len(groups[name])} {fmt(score(groups[name]))}",
              flush=True)
    print(f"PERCORPUS ALL files={len(files)} {fmt(score(files))}", flush=True)
    print("PERCORPUS_DONE", flush=True)


if __name__ == "__main__":
    main()
