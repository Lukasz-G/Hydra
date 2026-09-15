"""Scrape LeA (Lesekorpus Altdeutsch) into Hydra's 4-column format.

LeA is the Referenzkorpus Altdeutsch (ReA) re-served as annotated reading
texts: Old High German, Old Saxon and Old Low Franconian. There is no bulk
download, so the annotation has to come off the HTML -- but the HTML is
machine-generated and regular, and carries the layers aligned by token id.

Usage:
  python tools/scrape_lea.py fetch CACHE_DIR        # crawl, with caching
  python tools/scrape_lea.py build CACHE_DIR OUT_DIR

Crawl completely, never sample. An earlier corpus estimate in this project was
out by 3x because it extrapolated from two pages; the rule since is to fetch
everything and count. Each work's page carries a dropdown listing every
chapter of that work, so one fetch per entry point yields the full inventory.

Layers used (of the ten present):
  thead <th scope="col">  diplomatic surface
  Lemma                   lemma
  M1a_DDDTS_Lemma         part of speech ("Wortart")
  M2c_Flexion_Beleg_2     inflection ("Flexion") -> morph column

Note the tagset is DDDTS, ReA's own, NOT the HiTS scheme shared by ReM and
ReF. Old German therefore sits outside their label space and needs either a
mapping or a routed, disjoint label space.

Also available and not used here: Standard_W (a normalised form, which is what
SS7.1 wants as an auxiliary target), Sprache (per-token language), and
Uebersetzung (a dictionary gloss).
"""
from __future__ import annotations

import html
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://titus.uni-frankfurt.de/lea/"
DELAY = 0.5           # be polite; the corpus is small and we cache
NO_MORPH = "--"       # ReM's uninflected marker, kept for consistency

LAYER_LEMMA = "Lemma"
LAYER_POS = "M1a_DDDTS_Lemma"
LAYER_MORPH = "M2c_Flexion_Beleg_2"

TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)
THEAD_RE = re.compile(r"<thead\b.*?</thead>", re.S)
COL_RE = re.compile(r'<th scope="col">(.*?)</th>', re.S)
ROW_RE = re.compile(r'<tr data-layer="([^"]+)"[^>]*>(.*?)</tr>', re.S)
CELL_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S)
HREF_RE = re.compile(r'(?:data-href|href)="([^"]+\.html)"')


def clean(fragment: str) -> str:
    """Cell text: drop nested markup, unescape entities, collapse space."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def fetch(url: str, dest: Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "hydra-corpus-prep/1.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        body = fh.read().decode("utf-8", errors="replace")
    dest.write_text(body, encoding="utf-8")
    time.sleep(DELAY)
    return body


def crawl(cache: Path) -> list[Path]:
    """Transitive closure over the chapter dropdowns, starting at the index."""
    cache.mkdir(parents=True, exist_ok=True)
    index = fetch(BASE, cache / "_index.html")
    todo = {h for h in HREF_RE.findall(index) if not h.startswith("http")}
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            body = fetch(BASE + name, cache / name)
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {name}: {e}", file=sys.stderr)
            continue
        # every page lists all chapters of its own work
        todo |= {h for h in HREF_RE.findall(body)
                 if not h.startswith("http") and h not in seen}
        if len(seen) % 25 == 0:
            print(f"  fetched {len(seen)}, queued {len(todo)}", flush=True)
    print(f"crawled {len(seen)} pages into {cache}")
    return sorted(cache.glob("*.html"))


def parse_page(body: str) -> tuple[list[str], int]:
    """Return (tsv lines, n_skipped_tables)."""
    lines, skipped = [], 0
    for table in TABLE_RE.findall(body):
        head = THEAD_RE.search(table)
        if not head:
            continue
        surfaces = [clean(c) for c in COL_RE.findall(head.group(0))]
        layers = {lay: [clean(c) for c in CELL_RE.findall(body_)]
                  for lay, body_ in ROW_RE.findall(table)}
        if LAYER_POS not in layers or LAYER_LEMMA not in layers:
            skipped += 1
            continue
        n = len(surfaces)
        if any(len(v) != n for v in (layers[LAYER_POS], layers[LAYER_LEMMA])):
            # a table whose layers do not line up cannot be aligned safely
            skipped += 1
            continue
        morph = layers.get(LAYER_MORPH, [""] * n)
        if len(morph) != n:
            morph = [""] * n
        for i in range(n):
            surf = surfaces[i].strip()
            if not surf:
                continue
            lem, pos, mor = layers[LAYER_LEMMA][i], layers[LAYER_POS][i], morph[i]
            if not lem or not pos:
                lines.append(surf)               # context-only token
            else:
                lines.append("\t".join((surf, lem, pos, mor.strip() or NO_MORPH)))
        lines.append("")                          # blank line between blocks
    return lines, skipped


def build(cache: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pages = sorted(p for p in cache.glob("*.html") if not p.name.startswith("_"))
    # group chapters of one work into one document: "B_07.html" -> "B"
    works: dict[str, list[Path]] = {}
    for p in pages:
        works.setdefault(re.split(r"[_.\-]", p.stem)[0], []).append(p)
    tot_tok = tot_ctx = tot_skip = 0
    for work, chapters in sorted(works.items()):
        lines: list[str] = []
        for p in sorted(chapters):
            ls, sk = parse_page(p.read_text(encoding="utf-8", errors="replace"))
            lines += ls
            tot_skip += sk
        n_tok = sum(1 for ln in lines if "\t" in ln)
        n_ctx = sum(1 for ln in lines if ln and "\t" not in ln)
        tot_tok += n_tok
        tot_ctx += n_ctx
        (out / f"{work}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{work:12s} {len(chapters):4d} pages  {n_tok:8d} annotated  {n_ctx:7d} context")
    print(f"\n{len(works)} works, {tot_tok:,} annotated tokens, "
          f"{tot_ctx:,} context-only, {tot_skip} tables skipped")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "fetch":
        crawl(Path(sys.argv[2]))
    elif mode == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
