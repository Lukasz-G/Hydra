"""TSV parsing, corpus splitting, chunk-based Dataset and collate.

A sample is a chunk of `chunk_len` (T) consecutive tokens from one file plus a
halo of `halo` (H) context tokens on each side. Characters are encoded once per
token; the context TCN's receptive field provides the token window. Loss and
metrics apply only to the central T positions. Context never crosses files.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .vocab import EOW, PAD, UNK, Vocabs

log = logging.getLogger(__name__)

IGNORE = -100  # loss ignore_index; also the "unused" filler in target arrays


@dataclass
class Token:
    surface: str
    lemmas: list[str] | None  # None => context-only: untagged, skipped, or inference
    pos: list[str] | None
    morph: list[str] | None

    @property
    def n_items(self) -> int:
        return 0 if self.lemmas is None else len(self.lemmas)


def split_lemma_items(s: str, n_expected: int) -> list[str]:
    """Split a lemma string into items, honouring ReM '/' notation.

    In ReM-style annotation a lemma item may itself contain '+' as part of a
    discontinuous-unit reference: 'hièr/+inne' is ONE item (lemma hièr, the
    +inne part is realised by another token), and 'dâr/+zuo+zuo/dâr+' is TWO
    items ('dâr/+zuo', 'zuo/dâr+'). The POS column is the authoritative item
    counter (POS tags never contain '+'). Heuristic: a segment ending in '/'
    absorbs the following segment (its reference follows the '+'), and an
    empty segment re-attaches its '+' to the previous item (trailing '+' of a
    reference). If the count still disagrees and one item is expected, the
    whole string is that item.
    """
    segs = s.split("+")
    items: list[str] = []
    for seg in segs:
        if items and (items[-1].endswith("/") or seg == ""):
            items[-1] += "+" + seg
        else:
            items.append(seg)
    if len(items) != n_expected and n_expected == 1:
        return [s]
    return items


def parse_tsv_file(path: str | Path, on_mismatch: str = "skip",
                   n_slots: int = 8) -> tuple[list[Token], int]:
    """Parse one 4-column TSV file. Returns (tokens, n_skipped).

    Lines starting with '@' and blank lines are ignored. The POS column
    determines the item count; the lemma column is split with
    split_lemma_items. A token whose item counts cannot be aligned, or which
    has more than n_slots items, is kept as context-only (on_mismatch='skip')
    or raises (on_mismatch='error').
    """
    tokens: list[Token] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip() or line.startswith("@"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or not cols[0].strip():
                if on_mismatch == "error":
                    raise ValueError(f"{path}:{lineno}: expected 4 tab-separated columns: {line!r}")
                skipped += 1
                surface = cols[0].strip() if cols and cols[0].strip() else "?"
                tokens.append(Token(surface, None, None, None))
                continue
            surface = cols[0].strip()
            pos = cols[2].strip().split("+")
            n = len(pos)
            lemmas = split_lemma_items(cols[1].strip(), n)
            morph = cols[3].strip().split("+")
            if len(morph) != n and n == 1:
                morph = [cols[3].strip()]  # '+' inside a single morph annotation
            if not (len(lemmas) == n == len(morph)) or n > n_slots \
                    or any(not x for x in lemmas) or any(not x for x in pos) \
                    or any(not x for x in morph):
                if on_mismatch == "error":
                    raise ValueError(
                        f"{path}:{lineno}: item counts misaligned or > {n_slots}: {line!r}")
                skipped += 1
                tokens.append(Token(surface, None, None, None))
                continue
            tokens.append(Token(surface, lemmas, pos, morph))
    return tokens, skipped


def list_corpus_files(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files in {directory}")
    return files


def split_corpus_files(files: list[Path], dev_fraction: float, test_fraction: float,
                       seed: int) -> dict[str, list[str]]:
    """Deterministic by-file split. Returns {'train': [...], 'dev': [...], 'test': [...]}."""
    names = sorted(str(f) for f in files)
    rng = random.Random(seed)
    rng.shuffle(names)
    n = len(names)
    n_dev = max(1, round(n * dev_fraction))
    n_test = max(1, round(n * test_fraction))
    if n_dev + n_test >= n:
        raise ValueError(f"dev+test fractions leave no training files (n={n})")
    return {
        "dev": sorted(names[:n_dev]),
        "test": sorted(names[n_dev:n_dev + n_test]),
        "train": sorted(names[n_dev + n_test:]),
    }


def resolve_splits(cfg: DataConfig, run_dir: Path | None = None) -> dict[str, list[str]]:
    """Explicit train/dev/test dirs if given, else split corpus_dir by file."""
    if cfg.train_dir is not None:
        splits = {
            "train": [str(f) for f in list_corpus_files(cfg.train_dir)],
            "dev": [str(f) for f in list_corpus_files(cfg.dev_dir)],
            "test": [str(f) for f in list_corpus_files(cfg.test_dir)] if cfg.test_dir else [],
        }
    else:
        files = list_corpus_files(cfg.corpus_dir)
        splits = split_corpus_files(files, cfg.dev_fraction, cfg.test_fraction, cfg.split_seed)
    if cfg.limit_files > 0:
        splits = {k: v[:cfg.limit_files] for k, v in splits.items()}
    if run_dir is not None:
        (run_dir / "split.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    return splits


def load_split_tokens(files: list[str], on_mismatch: str, n_slots: int) -> list[list[Token]]:
    """Parse each file into its own document (token list)."""
    docs = []
    total_skipped = 0
    for f in files:
        tokens, skipped = parse_tsv_file(f, on_mismatch, n_slots)
        total_skipped += skipped
        if tokens:
            docs.append(tokens)
    if total_skipped:
        log.info("parsed %d files, %d malformed tokens kept as context-only",
                 len(files), total_skipped)
    return docs


@dataclass
class EncodedDoc:
    chars: np.ndarray    # (n, max_word_len) int16, PAD-filled
    pos: np.ndarray      # (n, n_slots) int16: label id | NULL | IGNORE
    morph: np.ndarray    # (n, n_slots) int16
    lemma: np.ndarray    # (n, n_slots, max_lemma_len) int16
    n_items: np.ndarray  # (n,) int8, 0 for context-only
    surfaces: list[str]
    gold: list[tuple[str, str, str] | None]  # '+'-joined (lemma, pos, morph) or None


def encode_document(tokens: list[Token], vocabs: Vocabs, max_word_len: int,
                    max_lemma_len: int, n_slots: int) -> EncodedDoc:
    n = len(tokens)
    chars = np.full((n, max_word_len), PAD, dtype=np.int16)
    pos = np.full((n, n_slots), IGNORE, dtype=np.int16)
    morph = np.full((n, n_slots), IGNORE, dtype=np.int16)
    lemma = np.full((n, n_slots, max_lemma_len), IGNORE, dtype=np.int16)
    n_items = np.zeros(n, dtype=np.int8)
    truncated = 0
    for i, tok in enumerate(tokens):
        ids = vocabs.chars.encode(tok.surface) or [UNK]
        if len(ids) > max_word_len:
            ids = ids[:max_word_len]
            truncated += 1
        chars[i, :len(ids)] = ids
        if tok.lemmas is None:
            continue
        k = len(tok.lemmas)
        n_items[i] = k
        for s in range(k):
            pos[i, s] = vocabs.pos.encode(tok.pos[s])
            morph[i, s] = vocabs.morph.encode(tok.morph[s])
            lids = vocabs.chars.encode(tok.lemmas[s])[:max_lemma_len - 1]
            lemma[i, s, :len(lids)] = lids
            lemma[i, s, len(lids)] = EOW
        # unused slots: NULL POS target, everything else stays IGNORE
        pos[i, k:] = 0  # vocab.NULL
    if truncated:
        log.warning("%d surfaces longer than %d chars were truncated", truncated, max_word_len)
    gold = [None if t.lemmas is None else
            ("+".join(t.lemmas), "+".join(t.pos), "+".join(t.morph)) for t in tokens]
    return EncodedDoc(chars, pos, morph, lemma, n_items, [t.surface for t in tokens], gold)


class HydraDataset(Dataset):
    """Chunk dataset over encoded documents.

    Item tensors (int64 unless noted):
      chars      (T+2H, max_word_len)
      pos        (T, K)      targets incl. NULL / IGNORE
      morph      (T, K)
      lemma      (T, K, L)
      token_mask (T,) bool   True where a token carries supervision
    """

    def __init__(self, docs: list[list[Token]], vocabs: Vocabs, cfg: DataConfig,
                 n_slots: int, training: bool = False):
        self.cfg = cfg
        self.n_slots = n_slots
        self.T = cfg.chunk_len
        self.H = cfg.halo
        self.docs = [encode_document(d, vocabs, cfg.max_word_len, cfg.max_lemma_len, n_slots)
                     for d in docs]
        self.chunks: list[tuple[int, int]] = []  # (doc_id, start)
        for d, doc in enumerate(self.docs):
            n = len(doc.n_items)
            for start in range(0, n, self.T):
                self.chunks.append((d, start))
        if training and cfg.multi_item_upsample > 1:
            extra = []
            for (d, start) in self.chunks:
                if (self.docs[d].n_items[start:start + self.T] > 1).any():
                    extra.extend([(d, start)] * (cfg.multi_item_upsample - 1))
            self.chunks.extend(extra)

    def __len__(self) -> int:
        return len(self.chunks)

    def _slice_padded(self, arr: np.ndarray, lo: int, hi: int, fill: int) -> np.ndarray:
        """arr[lo:hi] along axis 0 with out-of-range positions filled."""
        n = arr.shape[0]
        out = np.full((hi - lo,) + arr.shape[1:], fill, dtype=arr.dtype)
        a, b = max(lo, 0), min(hi, n)
        if a < b:
            out[a - lo:b - lo] = arr[a:b]
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d, start = self.chunks[idx]
        doc = self.docs[d]
        T, H = self.T, self.H
        chars = self._slice_padded(doc.chars, start - H, start + T + H, PAD)
        pos = self._slice_padded(doc.pos, start, start + T, IGNORE)
        morph = self._slice_padded(doc.morph, start, start + T, IGNORE)
        lemma = self._slice_padded(doc.lemma, start, start + T, IGNORE)
        n_items = self._slice_padded(doc.n_items, start, start + T, 0)
        return {
            "chars": torch.from_numpy(chars.astype(np.int64)),
            "pos": torch.from_numpy(pos.astype(np.int64)),
            "morph": torch.from_numpy(morph.astype(np.int64)),
            "lemma": torch.from_numpy(lemma.astype(np.int64)),
            "token_mask": torch.from_numpy(n_items > 0),
            "n_items": torch.from_numpy(n_items.astype(np.int64)),
        }

    def chunk_surfaces(self, idx: int) -> list[str]:
        """Surfaces of the central T positions of chunk idx ('' for padding)."""
        d, start = self.chunks[idx]
        doc = self.docs[d]
        out = doc.surfaces[start:start + self.T]
        return out + [""] * (self.T - len(out))

    def chunk_gold(self, idx: int) -> list[tuple[str, str, str] | None]:
        """Gold '+'-joined strings for the central T positions (None = no supervision)."""
        d, start = self.chunks[idx]
        doc = self.docs[d]
        out = doc.gold[start:start + self.T]
        return out + [None] * (self.T - len(out))


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}
