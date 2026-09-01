"""Convert raw running-text files (e.g. the MHDBDB plain-text collection) into
Hydra's one-token-per-line format for unannotated pretraining data.

Usage: python tools/prepare_raw_corpus.py SRC_DIR OUT_DIR
Whitespace-tokenizes each *.txt; each source file becomes one document.
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
n_files = n_tokens = 0
for f in sorted(src.glob("*.txt")):
    tokens = f.read_text(encoding="utf-8", errors="replace").split()
    if len(tokens) < 10:
        continue
    (out / f.name).write_text("\n".join(tokens) + "\n", encoding="utf-8")
    n_files += 1
    n_tokens += len(tokens)
print(f"{n_files} files, {n_tokens} tokens -> {out}")
