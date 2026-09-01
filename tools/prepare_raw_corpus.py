"""Convert raw running-text files (e.g. the MHDBDB plain-text collection) into
Hydra's one-token-per-line format for unannotated pretraining data.

Usage: python tools/prepare_raw_corpus.py SRC_DIR OUT_DIR [EXCLUDE_FILE]
Whitespace-tokenizes each *.txt; each source file becomes one document.
EXCLUDE_FILE lists file stems (one per line) to skip — used to keep works
overlapping held-out evaluation manuscripts out of pretraining data.
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
exclude = set()
if len(sys.argv) > 3:
    exclude = {ln.strip() for ln in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
               if ln.strip()}
out.mkdir(parents=True, exist_ok=True)
n_files = n_tokens = 0
for f in sorted(src.glob("*.txt")):
    if f.stem in exclude:
        continue
    tokens = f.read_text(encoding="utf-8", errors="replace").split()
    if len(tokens) < 10:
        continue
    (out / f.name).write_text("\n".join(tokens) + "\n", encoding="utf-8")
    n_files += 1
    n_tokens += len(tokens)
print(f"{n_files} files, {n_tokens} tokens -> {out}")
