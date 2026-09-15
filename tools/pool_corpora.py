"""Pool ReM (MHG) and ReF (ENHG) into one corpus with a SHARED POS space and
DISJOINT morph spaces, plus the merged metadata the stratified split needs.

Usage:
  python tools/pool_corpora.py OUT_DIR OUT_META

Why this shape (user decision, 2026-09-15). Measured on the two corpora:
POS already shares 65 tag types covering 97.8% of ReF tokens and 94.6% of
ReM's, so the shared label space the cross-variety experiment needs exists
without any mapping. Morph does not: the schemes differ notationally --
ReM writes 'Masc.Nom.Sg' and 'Ind.Pres.Sg.3', ReF writes 'Mask.Nom.Sg' and
'3.Sg.Praes.Ind.Unr' (different abbreviations, reversed field order, an extra
verb-class field) -- and only 53% of ReF tokens land on a shared morph tag.

Rather than guess at a mapping, the first joint run tests ONE thing: whether a
shared POS space transfers across varieties. ReF's morph tags are prefixed so
the two morph spaces cannot collide, and the tagging heads simply learn both.
Morph harmonisation is then a separate later experiment with a clean baseline
to measure against. If morph were mapped now and the joint run underperformed,
there would be no way to tell whether the architecture or the mapping failed.

Two details that matter:

  * ONLY ReF is prefixed. ReM's labels stay exactly as every previous run saw
    them, so the joint model's ReM-side numbers remain comparable to s_crf and
    the rest of the single-corpus results.
  * '--' is NOT prefixed. It means "uninflected", which is the same fact in
    both schemes, and splitting it in two would repeat precisely the mistake
    the MND export made with '<none>' against '--'.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

REM_DIR = Path("D:/Corpora/MHD")
REF_DIR = Path("D:/Corpora/FNHD")
REM_META = Path("meta/rem_metadata.csv")
REF_META = Path("meta/ref_metadata.csv")
PREFIX = "ReF:"
NO_MORPH = "--"


def prefix_morph(col: str) -> str:
    """Prefix each '+'-joined morph item, leaving the uninflected marker alone."""
    return "+".join(p if p == NO_MORPH else PREFIX + p for p in col.split("+"))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, out_meta = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    n_rem = n_ref = t_rem = t_ref = 0
    for src in sorted(REM_DIR.glob("*.txt")):        # unchanged
        shutil.copyfile(src, out / src.name)
        n_rem += 1
        t_rem += sum(1 for ln in src.open(encoding="utf-8") if "\t" in ln)

    for src in sorted(REF_DIR.glob("*.txt")):        # morph prefixed
        lines = []
        for ln in src.read_text(encoding="utf-8").splitlines():
            cols = ln.split("\t")
            if len(cols) >= 4:
                cols[3] = prefix_morph(cols[3])
                t_ref += 1
            lines.append("\t".join(cols))
        (out / src.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ref += 1

    rows = []
    with REM_META.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"sigle": r["sigle"], "dialect": r.get("dialect") or "?",
                         "period": r.get("period") or "?", "corpus": "ReM"})
    with REF_META.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # ReF's header names the same facts differently
            rows.append({"sigle": r["sigle"],
                         "dialect": r.get("language-area") or "?",
                         "period": r.get("time") or "?", "corpus": "ReF"})
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sigle", "dialect", "period", "corpus"])
        w.writeheader()
        w.writerows(rows)

    print(f"ReM: {n_rem} files, {t_rem:,} annotated tokens (labels unchanged)")
    print(f"ReF: {n_ref} files, {t_ref:,} annotated tokens (morph prefixed '{PREFIX}')")
    print(f"pool: {n_rem + n_ref} files, {t_rem + t_ref:,} tokens -> {out}")
    print(f"metadata: {len(rows)} rows -> {out_meta}")


if __name__ == "__main__":
    main()
