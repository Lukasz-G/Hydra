"""Convert ReF (Referenzkorpus Fruehneuhochdeutsch, 1350-1650) CorA-XML into
Hydra's 4-column format, plus a metadata CSV for the stratified split.

Usage:
  python tools/convert_ref.py REF_ROOT OUT_DIR [--meta meta/ref_metadata.csv]

REF_ROOT is the unpacked ReF-v1.0.2 directory. Only ref-rub/ and ref-mlu/ are
read: they are CorA-XML with morphological annotation, the same format and the
same HiTS-derived tagset as ReM, so their tokens can join ReM's directly. The
third subcorpus, ref-up/, is TigerXML syntax trees and has no lemma layer.

Conventions, matched to ReM so the two corpora can be pooled without silently
splitting a class in two (the mistake MND made with '<none>' vs '--'):

  * one <token> with several <tok_anno> children is a multi-item token; its
    columns are '+'-joined, exactly as ReM writes them.
  * a <tok_anno> with no <morph> is uninflected and gets '--', which is what
    ReM writes and what hydra.data already treats as a real label.
  * surface is the DIPLOMATIC form (tok_dipl/@utf joined), not the normalised
    tok_anno/@utf, because that is what ReM's MHD export uses and what the
    model is trained to read.
  * a token whose tok_anno carries no lemma AND no pos is transcription-only;
    it is written surface-only, which hydra.data keeps as a context token.

Licence: ReF is CC BY-SA 4.0 (see the corpus README; the Zenodo landing page
says only CC BY). Attribution: Wegera, Solms, Demske, Dipper (2021).
"""
from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NO_MORPH = "--"          # ReM's uninflected marker
SUBCORPORA = ("ref-rub", "ref-mlu")
# header fields worth keeping for a stratified split
META_FIELDS = ("sigle", "name", "language", "language-area", "language-region",
               "language-type", "genre", "medium", "time", "date", "text-place",
               "text-type", "library")


def parse_header(text_el: ET.Element) -> dict:
    """The CorA <header> is a free-text block of 'key: value' lines."""
    out: dict[str, str] = {}
    h = text_el.find("header")
    if h is None or not h.text:
        return out
    for line in h.text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def convert_file(path: Path) -> tuple[list[str], dict, int, int]:
    """Return (tsv lines, metadata, n_tokens, n_context_only)."""
    tree = ET.parse(path)
    root = tree.getroot()
    meta = parse_header(root)
    ch = root.find("cora-header")
    if ch is not None:
        meta.setdefault("sigle", ch.get("sigle", ""))
        meta.setdefault("name", ch.get("name", ""))

    lines: list[str] = []
    n_tok = n_ctx = 0
    for token in root.iter("token"):
        dipl = [d.get("utf", "") for d in token.findall("tok_dipl")]
        surface = "".join(dipl).strip()
        if not surface:
            continue
        lemmas, poss, morphs = [], [], []
        for anno in token.findall("tok_anno"):
            lem = anno.find("lemma")
            pos = anno.find("pos")
            if lem is None and pos is None:
                continue
            lemmas.append((lem.get("tag") if lem is not None else "") or "")
            poss.append((pos.get("tag") if pos is not None else "") or "")
            m = anno.find("morph")
            morphs.append((m.get("tag") if m is not None else None) or NO_MORPH)
        if not poss or not all(lemmas) or not all(poss):
            # transcription-only or incompletely annotated: keep as context
            lines.append(surface)
            n_ctx += 1
            continue
        lines.append("\t".join((surface, "+".join(lemmas),
                                "+".join(poss), "+".join(morphs))))
        n_tok += 1
    return lines, meta, n_tok, n_ctx


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta_path = None
    for i, a in enumerate(sys.argv):
        if a == "--meta" and i + 1 < len(sys.argv):
            meta_path = Path(sys.argv[i + 1])
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for sub in SUBCORPORA:
        files += sorted((root / sub).glob("*.xml"))
    if not files:
        raise SystemExit(f"no CorA-XML found under {root}/{{{','.join(SUBCORPORA)}}}")

    rows, tot_tok, tot_ctx, n_multi = [], 0, 0, 0
    for path in files:
        try:
            lines, meta, n_tok, n_ctx = convert_file(path)
        except ET.ParseError as e:
            print(f"SKIP {path.name}: {e}", file=sys.stderr)
            continue
        sigle = meta.get("sigle") or path.stem
        (out / f"{sigle}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_multi += sum(1 for ln in lines if "\t" in ln and "+" in ln.split("\t")[2])
        tot_tok += n_tok
        tot_ctx += n_ctx
        rows.append({k: meta.get(k, "") for k in META_FIELDS} | {"sigle": sigle,
                                                                "file": f"{sigle}.txt",
                                                                "n_tokens": n_tok})
        print(f"{sigle:10s} {n_tok:8d} annotated  {n_ctx:6d} context-only  {path.name}")

    print(f"\n{len(rows)} files, {tot_tok:,} annotated tokens, "
          f"{tot_ctx:,} context-only, {n_multi:,} multi-item")
    if meta_path:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(META_FIELDS) + ["file", "n_tokens"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
