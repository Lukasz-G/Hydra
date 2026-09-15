"""Pool ReM, ReF, ReN and LeA into one corpus for stage-2 joint training.

Usage:
  python tools/pool_stage2.py OUT_DIR OUT_META

Shared POS, disjoint morph (user decision, 2026-09-15). POS already aligns
across the three HiTS corpora without any mapping -- 97.8% of ReF tokens and
81.1% of ReN's fall under tags ReM also uses -- and tools/map_lea_tags.py
brings 80.7% of Old German in as well. Morph does not align: the schemes
differ notationally rather than substantively, and only 53% (ReF), 64% (ReN)
and 28% (LeA) of tokens land on a tag ReM also writes. Rather than guess at a
mapping, each non-ReM corpus keeps its own morph space behind a prefix, and
the first joint run tests exactly one thing: whether a shared POS space
transfers across varieties.

Conventions, all of which exist to stop one semantic class splitting in two:

  * ReM is NEVER prefixed. Its labels stay as every previous run saw them, so
    the joint model's ReM-side numbers remain comparable to s_crf and the rest.
  * '--' is never prefixed. It means "uninflected" in all four schemes, and
    splitting it would repeat the MND '<none>' mistake exactly.
  * LeA's POS is already mapped or routed by map_lea_tags.py, so its POS
    column is passed through untouched here; only its morph is prefixed.

Metadata for the stratified split is assembled from whatever each corpus
carries: ReM and ReF from their CSVs, ReN from its CorA-XML headers
(language-area + time), LeA from its own per-token language layer, since the
reading corpus ships no manuscript metadata at all. Run the split with
data.group_by_manuscript = true.
"""
from __future__ import annotations

import csv
import glob
import io
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

NO_MORPH = "--"
REN_XML = "D:/Corpora/CorA-ReN-XML_1.1/CorA-ReN-XML_1.1/ReN_anno_*/*.xml"

# name, directory, morph prefix (None = leave alone)
CORPORA = [
    ("ReM", Path("D:/Corpora/MHD"), None),
    ("ReF", Path("D:/Corpora/FNHD"), "ReF:"),
    ("ReN", Path("D:/Corpora/MND_full"), "ReN:"),
    ("LeA", Path("D:/Corpora/AHD_hits"), "OHG:"),
]


def century(period: str) -> str:
    """Coarsen a period label to its century.

    The four corpora write periods differently -- ReM '13_2', ReF '16,1',
    ReN '15/2' -- and at half-century granularity the pooled corpus shatters:
    ReF gets 76 strata for 190 files (10 of them singletons) and ReN 62 for
    161 (27 singletons). A stratum of one file can never be held out, and a
    stratum of two or three LARGE files cannot either, which is how ReF --
    the biggest corpus here -- ended up with a single file in the test split.
    Century granularity gives every corpus strata of usable size while keeping
    the temporal balance the protocol is for.
    """
    m = re.match(r"\s*(\d{2})", period or "")
    return m.group(1) if m else (period or "?")


def prefix_morph(col: str, prefix: str) -> str:
    return "+".join(p if p == NO_MORPH else prefix + p for p in col.split("+"))


def ren_metadata() -> dict[str, tuple[str, str]]:
    """stem -> (language-area, time), from ReN's CorA-XML headers."""
    out = {}
    for p in glob.glob(REN_XML):
        head = re.search(r"<header>(.*?)</header>",
                         io.open(p, encoding="utf-8").read(8000), re.S)
        if not head:
            continue
        d = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
             for l in head.group(1).splitlines() if ":" in l}
        out[Path(p).stem] = (d.get("language-area") or "?", d.get("time") or "?")
    return out


def lea_metadata(src: Path) -> dict[str, tuple[str, str]]:
    """stem -> (dominant language, 'OldGerman'), from LeA's own language layer.

    LeA ships no manuscript metadata, and it is genuinely several languages --
    Old High German 64.5%, Old Saxon 16.5%, Old Low Franconian 1.5% of its
    annotated tokens. Treating it as one "OHG" block would put the Heliand and
    Otfrid in the same stratum, so the dominant per-token language of each work
    stands in for dialect. Period is a single bucket: these texts are all
    8th-11th century and the corpus does not date them individually.
    """
    cell = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
    row = re.compile(r'<tr data-layer="Sprache"[^>]*>(.*?)</tr>', re.S)
    langs: dict[str, Counter] = {}
    for page in Path("D:/Corpora/LEA_cache").glob("*.html"):
        if page.name.startswith("_"):
            continue
        work = re.split(r"[_.\-]", page.stem)[0]
        body = page.read_text(encoding="utf-8", errors="replace")
        c = langs.setdefault(work, Counter())
        for blk in row.findall(body):
            for x in cell.findall(blk):
                v = re.sub(r"<[^>]+>", "", x).strip().lower().rstrip(".")
                if v and not v.startswith(("lat", "grc")):
                    c[v.split("-")[0].split("/")[0]] += 1
    out = {}
    for f in sorted(src.glob("*.txt")):
        c = langs.get(f.stem)
        out[f.stem] = (c.most_common(1)[0][0] if c else "ahd", "OldGerman")
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, out_meta = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    rem = {r["sigle"]: (r.get("dialect") or "?", r.get("period") or "?")
           for r in csv.DictReader(open("meta/rem_metadata.csv", encoding="utf-8"))}
    ref = {r["sigle"]: (r.get("language-area") or "?", r.get("time") or "?")
           for r in csv.DictReader(open("meta/ref_metadata.csv", encoding="utf-8"))}
    ren = ren_metadata()
    lea = lea_metadata(Path("D:/Corpora/AHD_hits"))

    rows, stats = [], Counter()
    for name, src, prefix in CORPORA:
        files = sorted(src.glob("*.txt"))
        if not files:
            raise SystemExit(f"{name}: no .txt under {src}")
        for f in files:
            if prefix is None:
                shutil.copyfile(f, out / f.name)
                n = sum(1 for ln in f.open(encoding="utf-8") if "\t" in ln)
            else:
                lines, n = [], 0
                for ln in f.read_text(encoding="utf-8").splitlines():
                    c = ln.split("\t")
                    if len(c) >= 4:
                        c[3] = prefix_morph(c[3], prefix)
                        n += 1
                    lines.append("\t".join(c))
                (out / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[name] += n
            stem = f.stem
            if name == "ReM":
                m = re.match(r"(M\d{3}[A-Za-z]?)", f.name)
                key = m.group(1).upper() if m else stem
                dia, per = rem.get(key, rem.get(key.rstrip("Y"), ("?", "?")))
            elif name == "ReF":
                key, (dia, per) = stem, ref.get(stem, ("?", "?"))
            elif name == "ReN":
                key, (dia, per) = stem, ren.get(stem, ("?", "?"))
            else:
                key, (dia, per) = stem, lea.get(stem, ("ahd", "OldGerman"))
            rows.append({"sigle": key, "dialect": f"{name}/{dia}",
                         "period": century(per), "corpus": name, "file": f.name})

    # sigles must be unique or load_strata silently keeps only the last
    dup = [s for s, c in Counter(r["sigle"] for r in rows).items() if c > 1]
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sigle", "dialect", "period", "corpus", "file"])
        w.writeheader()
        w.writerows(rows)

    tot = sum(stats.values())
    for name, _, prefix in CORPORA:
        print(f"{name:5s} {stats[name]:>10,} tokens  morph prefix "
              f"{prefix or '(none - reference space)'}")
    print(f"pool  {tot:>10,} tokens, {len(rows)} files -> {out}")
    print(f"metadata {len(rows)} rows -> {out_meta}"
          + (f"   WARNING duplicate sigles: {dup[:5]}" if dup else ""))


if __name__ == "__main__":
    main()
