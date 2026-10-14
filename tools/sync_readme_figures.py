"""Refresh the figures README.md embeds from the directory that draws them.

    python tools/sync_readme_figures.py ../journal/figures

The three images under docs/figures are copies. They were taken by hand once
and then drifted: the architecture figure in the README still labelled the
masked-token head "pretraining only" weeks after the drawing had been
corrected, because nothing connected the two directories. This script is that
connection, and it reads the list of what to copy out of README.md itself, so
a figure added to the README is picked up without editing anything here.

Exits non-zero if the README asks for a figure the source directory does not
have, which is the case worth catching: a README that embeds a file nobody
generates any more.
"""
from __future__ import annotations

import filecmp
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEST = ROOT / "docs" / "figures"


def wanted() -> list[str]:
    """Every docs/figures/*.png the README embeds, in the order it uses them."""
    text = README.read_text(encoding="utf-8")
    return list(dict.fromkeys(re.findall(r"\(docs/figures/([^)]+\.png)\)", text)))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    src = Path(argv[0]).expanduser().resolve()
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    missing, copied, same = [], [], []
    for name in wanted():
        source = src / name
        target = DEST / name
        if not source.exists():
            missing.append(name)
            continue
        if target.exists() and filecmp.cmp(source, target, shallow=False):
            same.append(name)
            continue
        shutil.copy2(source, target)
        copied.append(name)

    for name in copied:
        print(f"updated docs/figures/{name}")
    for name in same:
        print(f"unchanged docs/figures/{name}")
    for name in missing:
        print(f"MISSING in {src}: {name}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
