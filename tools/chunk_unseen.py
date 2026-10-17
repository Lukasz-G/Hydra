"""What the random-chunk protocol's unseen-surface rate really is.

Under split_mode='chunk' every file goes into all three split lists and the
roles are decided per chunk inside HydraDataset. The vocabularies, though,
are built once from splits['train'] -- every file -- so train_surfaces holds
every surface in the corpus and the measured unseen rate is zero whatever the
data looks like. That is an artefact of the order the two things happen in,
not a property of the protocol.

This recomputes the rate the way the stratified protocol computes it: from
the training CHUNKS alone. It replicates HydraDataset's own partition (same
chunk enumeration, same seeded shuffle, same fractions), so the subsets are
the ones the model actually trained and was scored on. CPU only; no
checkpoint is opened.

Usage:
  python tools/chunk_unseen.py --run=runs/c_joint --corpus=D:/Corpora/MHD
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hydra.config import config_from_dict                       # noqa: E402
from hydra.data import load_split_tokens                        # noqa: E402


def main() -> None:
    opts = {"run": "runs/c_joint", "corpus": "", "out": "runs/chunk_unseen.json"}
    for a in sys.argv[1:]:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            if k not in opts:
                raise SystemExit(f"unknown option --{k}")
            opts[k] = v

    repo = pathlib.Path(__file__).resolve().parents[1]
    run = pathlib.Path(opts["run"])
    run = run if run.is_absolute() else repo / run
    corpus = pathlib.Path(opts["corpus"]) if opts["corpus"] else None

    cfg = config_from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    if cfg.data.split_mode != "chunk":
        raise SystemExit(f"{run.name} is split_mode={cfg.data.split_mode!r}, not 'chunk'")
    splits = json.loads((run / "split.json").read_text(encoding="utf-8"))
    files = splits["train"]
    if corpus:
        files = [str(corpus / f.replace("\\", "/").rsplit("/", 1)[-1]) for f in files]

    docs = load_split_tokens(files, cfg.data.on_mismatch, cfg.model.n_slots,
                             cfg.data.combined_tags, cfg.data.align_max_items)
    T = cfg.data.chunk_len

    # HydraDataset builds its chunk list exactly this way, then shuffles the
    # INDEX list with split_seed and slices dev/test off the front.
    chunks = [(d, s) for d, doc in enumerate(docs) for s in range(0, len(doc), T)]
    order = list(range(len(chunks)))
    random.Random(cfg.data.split_seed).shuffle(order)
    n_dev = max(1, round(len(order) * cfg.data.dev_fraction))
    n_test = max(1, round(len(order) * cfg.data.test_fraction))
    role_of = {}
    for i in order[:n_dev]:
        role_of[i] = "dev"
    for i in order[n_dev:n_dev + n_test]:
        role_of[i] = "test"
    for i in order[n_dev + n_test:]:
        role_of[i] = "train"

    train_surf, train_lemmas = set(), set()
    test_tok = test_unseen = 0
    test_lemma_types, test_only_lemma_types = set(), set()

    for i, (d, s) in enumerate(chunks):
        span = docs[d][s:s + T]
        if role_of[i] == "train":
            for t in span:
                train_surf.add(t.surface)
                if t.lemmas is not None:
                    train_lemmas.update(t.lemmas)

    for i, (d, s) in enumerate(chunks):
        if role_of[i] != "test":
            continue
        for t in docs[d][s:s + T]:
            if t.lemmas is None:
                continue
            test_tok += 1
            if t.surface not in train_surf:
                test_unseen += 1
            test_lemma_types.update(t.lemmas)

    test_only_lemma_types = {l for l in test_lemma_types if l not in train_lemmas}

    # the whole-corpus inventory the run actually built
    corpus_surf = {t.surface for doc in docs for t in doc}

    res = {
        "run": run.name,
        "chunks": len(chunks),
        "test_tokens": test_tok,
        "test_unseen": test_unseen,
        "test_unseen_pct": 100.0 * test_unseen / test_tok if test_tok else 0.0,
        "train_surface_types": len(train_surf),
        "corpus_surface_types": len(corpus_surf),
        "surface_types_only_outside_train": len(corpus_surf) - len(train_surf),
        "test_lemma_types": len(test_lemma_types),
        "test_only_lemma_types": len(test_only_lemma_types),
    }
    for k, v in res.items():
        print(f"  {k:34s} {v:,}" if isinstance(v, int) else f"  {k:34s} {v}")
    dest = repo / opts["out"]
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", dest.relative_to(repo).as_posix())


if __name__ == "__main__":
    main()
