"""Confidence intervals resampled over manuscripts, and seed spread.

Tokens inside one manuscript are not independent observations: they share a
hand, a workshop's orthography and often a scribe's habitual spelling of the
same word. An interval computed over tokens therefore reports a precision the
data does not have. This resamples whole MANUSCRIPTS with replacement, which
is the unit the stratified protocol holds out.

Every system scored on the same split can be compared this way as long as its
per-token predictions were kept, and the difference between two systems is
resampled on the same draws, so the paired structure is preserved.

CPU only; the dumps are already on disk.

Usage:
  python tools/bootstrap_ci.py --run=runs/s_joint --corpus=D:/Corpora/MHD \
      --hydra=s_joint,s_joint_s1,s_joint_s2 --pie=runs/pie_strat_pred.tsv
"""
from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hydra.config import config_from_dict                                # noqa: E402
from hydra.data import file_sigle, load_manuscript_ids, load_split_tokens  # noqa: E402

B = 2000
ALPHA = 0.05


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def main() -> None:
    opts = {"run": "runs/s_joint", "corpus": "", "hydra": "s_joint",
            "pie": "", "meta": "meta/rem_metadata.csv", "boot": str(B),
            "seed": "20261017", "out": "runs/bootstrap_ci.json"}
    for a in sys.argv[1:]:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            if k not in opts:
                raise SystemExit(f"unknown option --{k}")
            opts[k] = v

    repo = pathlib.Path(__file__).resolve().parents[1]
    run = pathlib.Path(opts["run"])
    run = run if run.is_absolute() else repo / run
    corpus = pathlib.Path(opts["corpus"]) if opts["corpus"] else None
    nboot = int(opts["boot"])

    cfg = config_from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    splits = json.loads((run / "split.json").read_text(encoding="utf-8"))
    ms_of = {k.upper(): v for k, v in load_manuscript_ids(repo / opts["meta"]).items()}

    # token -> manuscript, by rebuilding the dump's file order
    groups, spans = [], []
    off = 0
    for f in splits["test"]:
        name = f.replace("\\", "/").rsplit("/", 1)[-1]
        p = (corpus / name) if corpus else pathlib.Path(f)
        docs = load_split_tokens([str(p)], cfg.data.on_mismatch, cfg.model.n_slots,
                                 cfg.data.combined_tags, cfg.data.align_max_items)
        n = sum(1 for doc in docs for t in doc if t.lemmas is not None)
        sig = file_sigle(pathlib.Path(name)).upper()
        spans.append((ms_of.get(sig, sig), off, off + n))
        off += n
    total = off

    by_ms: dict[str, list[tuple[int, int]]] = {}
    for ms, a, b in spans:
        by_ms.setdefault(ms, []).append((a, b))
    names = sorted(by_ms)
    print(f"{len(splits['test'])} files, {len(names)} manuscripts, {total:,} tokens")

    def correct_vector(rows, lemma_col, gold_col=1):
        return [r[lemma_col] == r[gold_col] for r in rows]

    systems = {}
    for tag in opts["hydra"].split(","):
        if not tag:
            continue
        rows = [l.split(chr(9)) for l in (run.parent / f"dump.{tag}.tsv")
                .read_text(encoding="utf-8").splitlines()[1:]]
        if len(rows) != total:
            raise SystemExit(f"dump.{tag}: {len(rows)} rows against {total}")
        systems[f"hydra:{tag}"] = correct_vector(rows, 3)

    gold = None
    if opts["pie"]:
        pie_path = pathlib.Path(opts["pie"])
        pie_path = pie_path if pie_path.is_absolute() else repo / pie_path
        prows = [l.split(chr(9)) for l in pie_path.read_text(encoding="utf-8").splitlines()]
        href = [l.split(chr(9)) for l in (run.parent / f"dump.{opts['hydra'].split(',')[0]}.tsv")
                .read_text(encoding="utf-8").splitlines()[1:]]
        if len(prows) != total or any(a[0] != b[0] for a, b in zip(prows, href)):
            raise SystemExit("pie predictions do not align with the hydra dump")
        gold = [r[1] for r in href]
        systems["pie"] = [p[1] == g for p, g in zip(prows, gold)]

    # --- point estimate, manuscript bootstrap, and (hydra) seed spread -----
    rng = random.Random(int(opts["seed"]))
    draws = [[rng.choice(names) for _ in names] for _ in range(nboot)]

    def acc(vec, idx_spans):
        n = ok = 0
        for a, b in idx_spans:
            n += b - a
            ok += sum(vec[a:b])
        return 100.0 * ok / n if n else 0.0

    def boot(vec):
        out = []
        for d in draws:
            sp = [s for ms in d for s in by_ms[ms]]
            out.append(acc(vec, sp))
        return out

    res = {"manuscripts": len(names), "tokens": total, "boot": nboot}
    allspans = [s for ms in names for s in by_ms[ms]]
    for name, vec in systems.items():
        bs = boot(vec)
        res[name] = {"acc": acc(vec, allspans),
                     "lo": percentile(bs, ALPHA / 2), "hi": percentile(bs, 1 - ALPHA / 2)}
        print(f"  {name:22s} {res[name]['acc']:6.2f}  "
              f"[{res[name]['lo']:.2f}, {res[name]['hi']:.2f}]")

    hyd = [k for k in systems if k.startswith("hydra:")]
    if len(hyd) > 1:
        accs = [res[k]["acc"] for k in hyd]
        res["hydra_seeds"] = {"n": len(accs), "mean": statistics.mean(accs),
                              "sd": statistics.stdev(accs),
                              "min": min(accs), "max": max(accs)}
        print(f"  seeds n={len(accs)} mean={statistics.mean(accs):.2f} "
              f"sd={statistics.stdev(accs):.2f} range={max(accs)-min(accs):.2f}")

    if "pie" in systems and hyd:
        a, b = systems[hyd[0]], systems["pie"]
        # paired: the SAME manuscript draw scores both systems, so the
        # interval is on the difference and not on two separate estimates
        pair = []
        for d in draws:
            sp = [s for ms in d for s in by_ms[ms]]
            pair.append(acc(a, sp) - acc(b, sp))
        res["hydra_minus_pie"] = {"diff": acc(a, allspans) - acc(b, allspans),
                                  "lo": percentile(pair, ALPHA / 2),
                                  "hi": percentile(pair, 1 - ALPHA / 2)}
        d = res["hydra_minus_pie"]
        print(f"  hydra - pie            {d['diff']:6.2f}  [{d['lo']:.2f}, {d['hi']:.2f}]"
              f"  {'excludes 0' if d['lo'] > 0 or d['hi'] < 0 else 'includes 0'}")

    dest = repo / opts["out"]
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", dest.relative_to(repo).as_posix())


if __name__ == "__main__":
    main()
