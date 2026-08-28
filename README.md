# Hydra

Neural tagger/lemmatizer for pre-modern languages (Middle High German, Middle
Dutch, …). Complete rewrite of the original mpi4py-based Hydra (the old code
is preserved untouched in `legacy/`).

## The task

Input corpora are 4-column TSV files, one token per line:

```
surface <TAB> lemma <TAB> POS <TAB> morphology
```

Medieval tokens often have **no 1-to-1 mapping** to lemma/tag: one surface
token can realize several items, joined by `+` with aligned counts:

```
inhandon	in+hant	APPR+NA	c.D+Dat.Pl
```

ReM-style `/` notation for discontinuous units is understood: in
`hièr/+inne` the internal `+` is part of ONE item's lemma (the POS column is
the authoritative item counter; see `hydra.data.split_lemma_items`).

Lines starting with `@` are comments. Each file is one document — context
windows never cross file boundaries.

## Model

Character-level, fully convolutional, open lemma vocabulary (~6–12M params):

- shared character embedding → **TCN char encoder** → masked max-pool = token vector
- **TCN context encoder** over the token axis (receptive field ≈ ±14 tokens)
- **K = 8 parallel decoder slots** (non-autoregressive). Each slot classifies
  atomic POS (class 0 = NULL marks an unused slot → variable item count) and
  atomic morph, and generates the lemma as a character grid: transposed-conv
  upsampling + cross-attention to the surface characters + refinement TCN.
  Unseen lemmas are generated character by character.

Training targets: slot k < n gets (lemma chars + EOW, POS, morph); slots
k ≥ n get only a NULL POS target. NULL is down-weighted in the loss
(`loss.null_weight`) and chunks containing multi-item tokens are upsampled
(`data.multi_item_upsample`) to counter the ~95% single-item imbalance.

## Install

```
pip install -e .[dev]        # Python >= 3.10, PyTorch >= 2.1
pytest -q                    # 27 tests, CPU, ~15 s
```

## Train

```
hydra-train --config configs/default.toml
hydra-train --config configs/default.toml --set train.lr=1e-4 --set run.run_dir=runs/x
hydra-train --config configs/default.toml --resume runs/mhd_base/last.pt
```

Everything lands in `run.run_dir`: `config.json`, `split.json`, `vocab.json`,
`metrics.jsonl` (one JSON per event), `best.pt` (best dev lemma+POS joint
accuracy), `last.pt` (resume bit-exactly, RNG state included). Early stopping
via `train.patience`.

Data can be one directory (`data.corpus_dir`, split by file with
`dev_fraction`/`test_fraction`/`split_seed`) or explicit
`train_dir`/`dev_dir`/`test_dir`.

### Multi-GPU / multi-node (torch.distributed DDP)

The same command scales; no code changes:

```
torchrun --nproc_per_node=4 -m hydra.cli train --config configs/default.toml
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
         --rdzv_endpoint=host0:29500 -m hydra.cli train --config configs/default.toml
```

Backend is auto-selected (NCCL on Linux+CUDA, gloo otherwise). Without
torchrun it runs as a plain single process on `cuda:0` or CPU — zero setup.
Rank 0 owns vocab building, logging, dev evaluation and checkpoints; data is
sharded per step by `DistributedSampler` over shuffled chunks.

Windows notes (dev box): only gloo works; set `USE_LIBUV=0`; torchrun's
rendezvous can be flaky (Docker Desktop hosts entries) — launching processes
manually with `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK` env vars
works, and non-main ranks may linger at interpreter exit (kill them; Linux is
the real multi-GPU target).

## Tag

```
hydra-tag --model runs/mhd_base/best.pt --input texts/ --output tagged/
```

`--format txt` (whitespace tokenization), `tsv` (retag column 1), or `auto`.
Output is the 4-column TSV with `+`-joined items; `@` lines pass through.

## Evaluate

```
hydra-eval --model runs/mhd_base/best.pt --split test
hydra-eval --model runs/mhd_base/best.pt --input D:/Corpora/other_gold/
```

Reports accuracy (lemma / POS / morph / joint) overall, on multi-item tokens,
and on tokens unseen in training (OOV), plus mean lemma Levenshtein distance.

## Layout

```
hydra/          the package: config, vocab, data, model, losses, metrics,
                distributed, checkpoint, train, tag, evaluate, cli
configs/        default.toml (full corpus), smoke.toml (6 files, 3 epochs)
tests/          pytest suite incl. an end-to-end overfit test
legacy/         the pre-2024 mpi4py implementation (reference only)
```
