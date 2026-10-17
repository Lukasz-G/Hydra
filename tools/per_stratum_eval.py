"""Accuracy by dialect region and by period, from the prediction dumps.

tools/multi_item_eval.py --dump writes every supervised token of a split in
FILE ORDER, so the file boundaries can be rebuilt afterwards with the same
loader it used, and each token assigned the stratum of the file it came from.
That makes this a CPU job: no checkpoint is opened and no inference is run.

Several dumps of the same split (one per seed) give a range per stratum as
well as a mean, which matters here because the thin strata are small enough
that a single seed says little.

Usage:
  python tools/per_stratum_eval.py --run=runs/s_joint \
      --dumps=s_joint,s_joint_s1,s_joint_s2 --corpus=D:/Corpora/MHD \
      --meta=meta/rem_metadata.csv --out=runs/per_stratum.json
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hydra.config import config_from_dict                       # noqa: E402
from hydra.data import file_sigle, load_split_tokens            # noqa: E402


def parse_args(argv: list[str]) -> dict:
    out = {"run": "runs/s_joint", "dumps": "s_joint", "corpus": "", "split": "test",
           "meta": "meta/rem_metadata.csv", "out": "runs/per_stratum.json"}
    for a in argv:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            if k not in out:
                raise SystemExit(f"unknown option --{k}")
            out[k] = v
    return out


def supervised_per_file(files, cfg, corpus: pathlib.Path | None):
    """(split-order path, supervised token count) for each file.

    A split recorded on a rented box carries that box's paths; --corpus
    repoints them at the same corpus on this machine.
    """
    out = []
    for f in files:
        p = (corpus / f.replace("\\", "/").rsplit("/", 1)[-1]) if corpus else pathlib.Path(f)
        docs = load_split_tokens([str(p)], cfg.data.on_mismatch, cfg.model.n_slots,
                                 cfg.data.combined_tags, cfg.data.align_max_items)
        out.append((f, sum(1 for doc in docs for t in doc if t.lemmas is not None)))
    return out


def pearson(xs: list[tuple[float, float]]) -> float:
    n = len(xs)
    mx, my = sum(x for x, _ in xs) / n, sum(y for _, y in xs) / n
    cov = sum((x - mx) * (y - my) for x, y in xs)
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in xs))
    sy = math.sqrt(sum((y - my) ** 2 for _, y in xs))
    return cov / (sx * sy) if sx and sy else 0.0


def main() -> None:
    a = parse_args(sys.argv[1:])
    repo = pathlib.Path(__file__).resolve().parents[1]
    run = pathlib.Path(a["run"]) if pathlib.Path(a["run"]).is_absolute() else repo / a["run"]
    corpus = pathlib.Path(a["corpus"]) if a["corpus"] else None
    seeds = [s for s in a["dumps"].split(",") if s]

    cfg = config_from_dict(json.loads((run / "config.json").read_text(encoding="utf-8")))
    splits = json.loads((run / "split.json").read_text(encoding="utf-8"))

    meta = {}
    with (repo / a["meta"]).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("sigle"):
                meta[r["sigle"].upper()] = (r.get("dialect") or "?", r.get("period") or "?")

    def stratum(f: str) -> tuple[str, str]:
        name = f.replace("\\", "/").rsplit("/", 1)[-1]
        return meta.get(file_sigle(pathlib.Path(name)).upper(), ("unknown", "unknown"))

    per_file = supervised_per_file(splits[a["split"]], cfg, corpus)
    total = sum(n for _, n in per_file)

    dumps = {}
    for s in seeds:
        p = run.parent / f"dump.{s}.tsv"
        rows = [l.split("\t") for l in p.read_text(encoding="utf-8").splitlines()[1:]]
        if len(rows) != total:
            raise SystemExit(f"{p.name}: {len(rows)} rows against {total} supervised "
                             f"tokens -- the dump is not this split's")
        dumps[s] = rows
    print(f"{len(per_file)} files, {total:,} supervised tokens, {len(seeds)} seeds")

    # exposure: supervised training tokens per stratum
    train = supervised_per_file(splits["train"], cfg, corpus)
    exposure = {0: collections.Counter(), 1: collections.Counter()}
    for f, n in train:
        st = stratum(f)
        exposure[0][st[0]] += n
        exposure[1][st[1]] += n

    result = {}
    for axis, name in ((0, "dialect"), (1, "period")):
        acc = collections.defaultdict(lambda: collections.defaultdict(
            lambda: {"n": 0, "lemma": 0, "pos": 0}))
        for s in seeds:
            rows, off = dumps[s], 0
            for f, n in per_file:
                cell = acc[s][stratum(f)[axis]]
                for _surf, gl, gp, pl, pp in rows[off:off + n]:
                    cell["n"] += 1
                    cell["lemma"] += (pl == gl)
                    cell["pos"] += (pp == gp)
                off += n
        out = {}
        for k in acc[seeds[0]]:
            lem = [100 * acc[s][k]["lemma"] / acc[s][k]["n"] for s in seeds]
            pos = [100 * acc[s][k]["pos"] / acc[s][k]["n"] for s in seeds]
            out[k] = {"n": acc[seeds[0]][k]["n"], "train": exposure[axis].get(k, 0),
                      "lemma": sum(lem) / len(lem), "lemma_rng": max(lem) - min(lem),
                      "pos": sum(pos) / len(pos), "pos_rng": max(pos) - min(pos)}
        result[name] = dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))
        lo = min(v["lemma"] for v in out.values())
        hi = max(v["lemma"] for v in out.values())
        print(f"{name}: {len(out)} strata, lemma {lo:.1f}-{hi:.1f} "
              f"(spread {hi - lo:.1f}), corr(train, lemma) = "
              f"{pearson([(v['train'], v['lemma']) for v in out.values()]):+.2f}")
        result[name + "_corr"] = pearson([(v["train"], v["lemma"]) for v in out.values()])

    result["seeds"] = seeds
    result["split"] = a["split"]
    dest = repo / a["out"]
    dest.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", dest.relative_to(repo).as_posix())


if __name__ == "__main__":
    main()
