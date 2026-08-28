"""Tagging mode: annotate plain-text or TSV files with lemma/POS/morph columns."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from .config import Config, config_from_dict
from .data import HydraDataset, Token, collate
from .checkpoint import load_checkpoint
from .metrics import decode_batch
from .model import HydraModel
from .vocab import Vocabs

log = logging.getLogger(__name__)


def load_model_for_inference(model_path: str | Path,
                             device: torch.device) -> tuple[torch.nn.Module, Vocabs, Config]:
    """Build model + vocab from a checkpoint; vocab.json must sit next to it."""
    model_path = Path(model_path)
    payload = load_checkpoint(model_path, map_location=str(device))
    cfg = config_from_dict(payload["config"])
    vocab_path = model_path.parent / "vocab.json"
    vocabs = Vocabs.load(vocab_path)
    model = HydraModel(cfg.model, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       cfg.data.max_word_len, cfg.data.max_lemma_len,
                       cfg.data.chunk_len, cfg.data.halo).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, vocabs, cfg


def read_input_file(path: Path, fmt: str) -> list[tuple[str, str]]:
    """Returns a list of ('token', surface) / ('raw', line) entries in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if fmt == "auto":
        has_tabs = any("\t" in ln for ln in lines if ln.strip() and not ln.startswith("@"))
        fmt = "tsv" if has_tabs else "txt"
    entries: list[tuple[str, str]] = []
    for ln in lines:
        if not ln.strip() or ln.startswith("@"):
            entries.append(("raw", ln))
        elif fmt == "tsv":
            entries.append(("token", ln.split("\t")[0].strip()))
        else:
            entries.extend(("token", tok) for tok in ln.split())
    return entries


def tag_document(model: torch.nn.Module, vocabs: Vocabs, cfg: Config,
                 surfaces: list[str], device: torch.device) -> list[tuple[str, str, str]]:
    """Tag one document (list of surfaces); returns (lemma, pos, morph) per token."""
    if not surfaces:
        return []
    doc = [Token(s, None, None, None) for s in surfaces]
    ds = HydraDataset([doc], vocabs, cfg.data, cfg.model.n_slots)
    results: list[tuple[str, str, str]] = []
    bs = cfg.infer.batch_chunks
    with torch.inference_mode():
        for lo in range(0, len(ds), bs):
            idxs = list(range(lo, min(lo + bs, len(ds))))
            batch = collate([ds[i] for i in idxs])
            out = model(batch["chars"].to(device))
            chunk_surfaces = [ds.chunk_surfaces(i) for i in idxs]
            preds = decode_batch(out, vocabs, chunk_surfaces)
            for b, i in enumerate(idxs):
                _, start = ds.chunks[i]
                n_here = min(cfg.data.chunk_len, len(surfaces) - start)
                for t in range(n_here):
                    p = preds[b][t]
                    results.append((p.lemma, p.pos, p.morph))
    return results


def tag_files(model_path: str | Path, input_path: str | Path, output_dir: str | Path,
              fmt: str = "auto") -> list[Path]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, vocabs, cfg = load_model_for_inference(model_path, device)

    input_path = Path(input_path)
    files = sorted(input_path.glob("*.txt")) if input_path.is_dir() else [input_path]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for f in files:
        entries = read_input_file(f, fmt)
        surfaces = [text for kind, text in entries if kind == "token"]
        preds = tag_document(model, vocabs, cfg, surfaces, device)
        out_path = output_dir / f.name
        with open(out_path, "w", encoding="utf-8") as fh:
            it = iter(preds)
            for kind, text in entries:
                if kind == "raw":
                    fh.write(text + "\n")
                else:
                    lemma, pos, morph = next(it)
                    fh.write(f"{text}\t{lemma}\t{pos}\t{morph}\n")
        written.append(out_path)
        log.info("tagged %s -> %s (%d tokens)", f, out_path, len(surfaces))
    return written
