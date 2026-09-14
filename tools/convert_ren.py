"""Convert the ReN (Referenzkorpus Mittelniederdeutsch/Niederrheinisch) v1.1
tab export into Hydra's 4-column format, and extract its transcription-only
texts as unannotated pretraining data.

Usage:
  python tools/convert_ren.py anno REN_TAB_DIR   OUT_DIR
  python tools/convert_ren.py trans REN_TRANS_XML_DIR OUT_DIR

Why this exists (measured 2026-09-11): the Middle Low German corpus actually
wired into training, D:/Corpora/MND, holds 57 files / 409,703 tokens — while
ReN-v1.1_tab holds 161 fully annotated files / 1,347,523 tokens. MND is also
NOT convention-compatible with the Middle High German corpus: it writes the
uninflected-morph class as '<none>' where both ReM (MHD) and ReN-v1.1_tab
write '--'. Pooling MND with MHD would therefore split one morph class in
two. ReN-v1.1_tab needs no such fixing, so this converter reads it directly.

The only transformation on the annotated side is stripping the ReN proper-name
marker '°' from lemma items: 31,345 of its 31,431 occurrences sit on a bare NE
tag and it is always item-final, so it is an annotation marker fully predictable
from the POS tag the model already predicts. Keeping it would force the lemma
decoder to generate a redundant character and would split the lemma-type
vocabulary into X / X° duplicates.

The transcription-only side (ReN_trans_*, 74 files / 828,147 tokens) carries
<token>/<dipl>/<anno> with utf attributes but checked="n" and no POS/lemma/
morph children — unusable for fine-tuning, but it is the Middle Low German
analogue of the MHDBDB plain text used to pretrain the Middle High German
encoder, which MLG otherwise lacks entirely.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NAME_MARK = "°"


def strip_name_mark(lemma: str) -> str:
    """Drop the ReN proper-name marker from every '+'-joined lemma item."""
    return "+".join(part[:-1] if part.endswith(NAME_MARK) else part
                    for part in lemma.split("+"))


def convert_annotated(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_files = n_tokens = n_stripped = 0
    for path in sorted(src.glob("*.txt")):
        lines_out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("@"):
                lines_out.append(line)
                continue
            cols = line.split("\t")
            if len(cols) >= 4:
                before = cols[1]
                cols[1] = strip_name_mark(cols[1])
                n_stripped += before != cols[1]
                n_tokens += 1
            lines_out.append("\t".join(cols))
        (out / path.name).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        n_files += 1
    print(f"annotated: {n_files} files, {n_tokens} tokens, "
          f"{n_stripped} lemmas had '{NAME_MARK}' stripped -> {out}")


def convert_transcribed(src: Path, out: Path) -> None:
    """One token per line (surface only) — Hydra parses a 1-column file as
    context-only tokens, which is exactly the masked-LM pretraining input."""
    out.mkdir(parents=True, exist_ok=True)
    n_files = n_tokens = n_empty = 0
    for path in sorted(src.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            print(f"  SKIP {path.name}: {exc}", file=sys.stderr)
            continue
        toks = []
        for tok in root.iter("token"):
            # prefer the <dipl> diplomatic reading; fall back to <anno>, which
            # on these files is an untagged shell carrying the same utf text
            parts = [d.get("utf", "") for d in tok.findall("dipl")]
            if not any(parts):
                parts = [a.get("utf", "") for a in tok.findall("anno")]
            word = "".join(parts).strip()
            if word:
                toks.append(word)
            else:
                n_empty += 1
        if not toks:
            continue
        (out / f"{path.stem}.txt").write_text("\n".join(toks) + "\n", encoding="utf-8")
        n_files += 1
        n_tokens += len(toks)
    print(f"transcribed: {n_files} files, {n_tokens} tokens "
          f"({n_empty} empty tokens dropped) -> {out}")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("anno", "trans"):
        sys.exit(__doc__)
    mode, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if not src.is_dir():
        sys.exit(f"not a directory: {src}")
    (convert_annotated if mode == "anno" else convert_transcribed)(src, out)


if __name__ == "__main__":
    main()
