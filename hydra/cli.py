"""Command-line entry points: hydra-train / hydra-tag / hydra-eval,
also runnable as `python -m hydra.cli {train,tag,eval} ...` (torchrun-friendly)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="path to TOML config")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config value")


def train_main(argv: list[str] | None = None) -> None:
    from .config import load_config
    from .train import train

    p = argparse.ArgumentParser(prog="hydra-train")
    _add_config_args(p)
    p.add_argument("--resume", default=None, help="checkpoint to resume from (e.g. last.pt)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    train(cfg, resume=args.resume)


def tag_main(argv: list[str] | None = None) -> None:
    from .tag import tag_files

    p = argparse.ArgumentParser(prog="hydra-tag")
    p.add_argument("--model", required=True, help="checkpoint, e.g. runs/x/best.pt")
    p.add_argument("--input", required=True, help="file or directory of .txt files")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--format", choices=["auto", "txt", "tsv"], default="auto")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = tag_files(args.model, args.input, args.output, args.format)
    print(f"wrote {len(written)} file(s) to {args.output}")


def eval_main(argv: list[str] | None = None) -> None:
    from .config import config_from_dict
    from .checkpoint import load_checkpoint
    from .data import HydraDataset, load_split_tokens
    from .evaluate import evaluate_dataset
    from .tag import load_model_for_inference

    p = argparse.ArgumentParser(prog="hydra-eval")
    p.add_argument("--model", required=True, help="checkpoint, e.g. runs/x/best.pt")
    p.add_argument("--split", choices=["dev", "test"], default="test",
                   help="evaluate this split from the run's split.json")
    p.add_argument("--input", default=None,
                   help="alternatively: a gold TSV file or directory to evaluate on")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, vocabs, cfg = load_model_for_inference(args.model, device)

    if args.input:
        path = Path(args.input)
        files = sorted(str(f) for f in (path.glob("*.txt") if path.is_dir() else [path]))
    else:
        split_path = Path(args.model).parent / "split.json"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        files = splits[args.split]
        if not files:
            sys.exit(f"split {args.split!r} is empty in {split_path}")
    docs = load_split_tokens(files, cfg.data.on_mismatch, cfg.model.n_slots)
    ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots)
    metrics = evaluate_dataset(model, ds, vocabs, device, cfg.infer.batch_chunks)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m hydra.cli")
    p.add_argument("command", choices=["train", "tag", "eval"])
    args, rest = p.parse_known_args()
    {"train": train_main, "tag": tag_main, "eval": eval_main}[args.command](rest)


if __name__ == "__main__":
    main()
