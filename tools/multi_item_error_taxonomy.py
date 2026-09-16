"""Where multi-item POS errors actually come from, and what a legality
constraint could possibly buy.

Usage:
  python tools/multi_item_error_taxonomy.py ROWS.tsv [--train-split=RUN_DIR]
                                                     [--corpus-dir=DIR]
                                                     [--compare=OTHER.tsv]

ROWS.tsv is a dump with columns surface/gold_lemma/gold_pos/pred_lemma/pred_pos
(tools/multi_item_eval.py --dump writes these). --train-split names the run
whose split.json defines "attested"; basenames are re-resolved against
--corpus-dir so a dump made on a rented box can be analysed locally.

Why this exists. §6.2 found that COMBINED tags beat slots on multi-item POS by
4.43pp at 12 sigma, and the obvious explanation was legality: eight independent
slot classifiers can emit a tag sequence no annotator ever wrote, while a
combined tag can only ever emit an attested one. That explanation is testable
before writing any code for it, and it is wrong -- the ceiling on repairing
illegal sequences is far below the effect it was meant to explain. This script
is the measurement, kept so the same question can be re-asked of any later
model rather than re-argued.

--compare puts two token-aligned dumps side by side and splits the difference
between them into segmentation (item COUNT) and identification (components
given the right count), and between multi-item and single-item tokens. That
decomposition is what says whether an effect reported on the multi-item subset
is actually a multi-item effect at all.

It reports, on the multi-item subset:
  * how many predicted tag sequences are UNATTESTED in training (all a
    legality constraint can touch), how many of those are already wrong, and
    how many have an attested gold -- the ceiling for a perfect repair;
  * how many GOLD sequences are unattested, which a constraint makes
    unreachable: the cost side, which a constraint proposal usually forgets;
  * a taxonomy of the errors themselves -- wrong item COUNT, wrong components
    at the right count, or right multiset in the wrong ORDER.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def attested_combos(run_dir: Path, corpus_dir: Path | None) -> Counter:
    files = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))["train"]
    if corpus_dir:
        files = [corpus_dir / Path(f).name for f in files]
    combos: Counter = Counter()
    for f in files:
        f = Path(f)
        if not f.exists():
            raise SystemExit(f"{f} not found; pass --corpus-dir")
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln or ln.startswith("@"):
                continue
            c = ln.split("\t")
            if len(c) >= 3 and c[2]:
                combos[c[2]] += 1
    return combos


def legality(rs: list[dict], combos: Counter, label: str) -> None:
    n = len(rs)
    if not n:
        return
    wrong = [r for r in rs if r["pred_pos"] != r["gold_pos"]]
    pred_unatt = [r for r in rs if r["pred_pos"] not in combos]
    pu_wrong = [r for r in pred_unatt if r["pred_pos"] != r["gold_pos"]]
    fixable = [r for r in pu_wrong if r["gold_pos"] in combos]
    gold_unatt = [r for r in rs if r["gold_pos"] not in combos]
    gu_right = [r for r in gold_unatt if r["pred_pos"] == r["gold_pos"]]
    pct = lambda x: f"{len(x):>6,}  {100 * len(x) / n:6.2f}%"
    print(f"\n--- {label}: n={n:,} ---")
    print(f"  POS wrong now              {pct(wrong)}")
    print(f"  predicted seq UNATTESTED   {pct(pred_unatt)}   all a constraint can touch")
    print(f"    of those already wrong   {pct(pu_wrong)}")
    print(f"    ...with attested gold    {pct(fixable)}   CEILING, perfect repair")
    print(f"  GOLD seq unattested        {pct(gold_unatt)}   constraint makes these unreachable")
    print(f"    and right at the moment  {pct(gu_right)}   COST, guaranteed loss")


def taxonomy(multi: list[dict]) -> None:
    wrong = [r for r in multi if r["pred_pos"] != r["gold_pos"]]
    if not wrong:
        return
    cat: Counter = Counter()
    for r in wrong:
        g, p = r["gold_pos"].split("+"), r["pred_pos"].split("+")
        if len(g) != len(p):
            cat["wrong item COUNT"] += 1
        elif sorted(g) == sorted(p):
            cat["right multiset, wrong ORDER"] += 1
        else:
            bad = sum(1 for a, b in zip(g, p) if a != b)
            cat[f"right count, {bad}/{len(g)} components wrong"] += 1
    print(f"\n--- error taxonomy: {len(wrong):,} wrong of {len(multi):,} multi-item ---")
    for k, v in cat.most_common():
        print(f"  {k:<40s} {v:>5,}  {100 * v / len(wrong):5.1f}% of errors"
              f"  {100 * v / len(multi):5.2f}pp of multi-item")
    print("\n  item count confusion (gold n -> predicted n):")
    cm = Counter((len(r["gold_pos"].split("+")), len(r["pred_pos"].split("+")))
                 for r in multi)
    for (g, p), v in sorted(cm.items()):
        print(f"    {g} -> {p}: {v:>5,}{'   correct' if g == p else ''}")


def compare(a: list[dict], b: list[dict], na: str, nb: str) -> None:
    """Side-by-side of two token-aligned dumps, A against B."""
    if len(a) != len(b):
        raise SystemExit(f"{len(a)} vs {len(b)} rows -- dumps are not aligned")
    bad = sum(1 for x, y in zip(a, b)
              if x["surface"] != y["surface"] or x["gold_pos"] != y["gold_pos"])
    if bad:
        raise SystemExit(f"{bad} rows differ in surface or gold -- not the same tokens")
    items = lambda s: len(s.split("+"))

    def profile(rs):
        multi = [r for r in rs if "+" in r["gold_pos"]]
        cnt_ok = [r for r in multi if items(r["pred_pos"]) == items(r["gold_pos"])]
        single = [r for r in rs if "+" not in r["gold_pos"]]
        return {
            "multi POS": 100 * sum(1 for r in multi if r["pred_pos"] == r["gold_pos"]) / len(multi),
            "count ok": 100 * len(cnt_ok) / len(multi),
            "comps|count ok": 100 * sum(1 for r in cnt_ok if r["pred_pos"] == r["gold_pos"]) / max(len(cnt_ok), 1),
            "over-split": 100 * sum(1 for r in single if "+" in r["pred_pos"]) / len(single),
            "all POS": 100 * sum(1 for r in rs if r["pred_pos"] == r["gold_pos"]) / len(rs),
        }

    pa, pb = profile(a), profile(b)
    print()
    print(f"--- {na} vs {nb}, {len(a):,} aligned tokens ---")
    print(f"  {'':<16}{na[:18]:>19}{nb[:18]:>19}{'delta':>10}")
    for k in pa:
        print(f"  {k:<16}{pa[k]:18.2f}%{pb[k]:18.2f}%{pa[k] - pb[k]:+9.2f}pp")

    # the decomposition that matters: an effect measured on the multi-item
    # subset can still be mostly a single-item effect in absolute tokens
    ga = sum(1 for r in a if r["pred_pos"] == r["gold_pos"])
    gb = sum(1 for r in b if r["pred_pos"] == r["gold_pos"])
    mg = (sum(1 for r in a if "+" in r["gold_pos"] and r["pred_pos"] == r["gold_pos"])
          - sum(1 for r in b if "+" in r["gold_pos"] and r["pred_pos"] == r["gold_pos"]))
    print()
    print(f"  net POS tokens gained by {na}: {ga - gb:+,}")
    print(f"    from multi-item gold tokens : {mg:+,}")
    print(f"    from single-item gold tokens: {(ga - gb) - mg:+,}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows_path = Path(sys.argv[1])
    run_dir, corpus_dir, other = None, None, None
    for a in sys.argv[2:]:
        if a.startswith("--train-split="):
            run_dir = Path(a.split("=", 1)[1])
        elif a.startswith("--corpus-dir="):
            corpus_dir = Path(a.split("=", 1)[1])
        elif a.startswith("--compare="):
            other = Path(a.split("=", 1)[1])

    with rows_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    multi = [r for r in rows if "+" in r["gold_pos"]]
    single = [r for r in rows if "+" not in r["gold_pos"]]
    over = [r for r in single if "+" in r["pred_pos"]]
    print(f"{rows_path.name}: {len(rows):,} rows, {len(multi):,} multi-item gold")

    if run_dir:
        combos = attested_combos(run_dir, corpus_dir)
        multi_c = {k: v for k, v in combos.items() if "+" in k}
        print(f"attested: {len(combos):,} distinct sequences "
              f"({len(multi_c):,} multi-item), "
              f"{sum(1 for v in combos.values() if v == 1):,} hapax")
        legality(rows, combos, "all tokens")
        legality(multi, combos, "multi-item tokens (gold)")

    taxonomy(multi)
    if other:
        with other.open(encoding="utf-8") as fh:
            rows_b = list(csv.DictReader(fh, delimiter="	"))
        compare(rows, rows_b, rows_path.stem, other.stem)
    # the cost nobody counts: splitting a token that was never split
    print(f"\nsingle-item gold predicted as multi: {len(over):,} of {len(single):,}"
          f"  ({100 * len(over) / max(len(single), 1):.3f}%)")


if __name__ == "__main__":
    main()
