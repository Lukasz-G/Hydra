"""Where the protocol gap lives, once the random-chunk side is measured.

The paper's decomposition backed the gap out of tokens unseen under the
STRATIFIED protocol while treating the random-chunk side as having none.
It has some: tools/chunk_unseen.py puts the rate at about eight per cent
once the question is asked of the training chunks rather than of a
vocabulary built before the chunks were divided.

This scores the random-chunk test split and splits the result by whether the
surface occurs in a training CHUNK, so the two protocols can be compared on
the subset each actually saw. One inference pass over one existing
checkpoint; nothing is trained.

Usage:
  python tools/chunk_gap_decomp.py --run=runs/c_joint --corpus=D:/Corpora/MHD
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hydra.data import HydraDataset, collate, load_split_tokens   # noqa: E402
from hydra.metrics import decode_batch                            # noqa: E402
from hydra.snap import LemmaSnapper                               # noqa: E402
from hydra.tag import load_model_for_inference                    # noqa: E402


def train_chunk_surfaces(docs, cfg):
    """Surfaces occurring in a TRAINING chunk, by HydraDataset's own partition."""
    T = cfg.data.chunk_len
    chunks = [(d, s) for d, doc in enumerate(docs) for s in range(0, len(doc), T)]
    order = list(range(len(chunks)))
    random.Random(cfg.data.split_seed).shuffle(order)
    n_dev = max(1, round(len(order) * cfg.data.dev_fraction))
    n_test = max(1, round(len(order) * cfg.data.test_fraction))
    out = set()
    for i in order[n_dev + n_test:]:
        d, s = chunks[i]
        for t in docs[d][s:s + T]:
            out.add(t.surface)
    return out


def main() -> None:
    opts = {"run": "runs/c_joint", "corpus": "", "batch": "8",
            "strat_run": "runs/s_joint", "strat_dump": "dump.s_joint",
            "out": "runs/gap_decomposition.json"}
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, vocabs, cfg = load_model_for_inference(run / "model_only.pt", device)
    splits = json.loads((run / "split.json").read_text(encoding="utf-8"))
    files = splits["train"]
    if corpus:
        files = [str(corpus / f.replace("\\", "/").rsplit("/", 1)[-1]) for f in files]
    docs = load_split_tokens(files, cfg.data.on_mismatch, cfg.model.n_slots,
                             cfg.data.combined_tags, cfg.data.align_max_items)

    seen = train_chunk_surfaces(docs, cfg)
    print(f"{len(seen):,} surfaces occur in a training chunk")

    ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots, role="test")
    snapper = LemmaSnapper(vocabs.lemma_inventory) if vocabs.lemma_counts else None
    bs = int(opts["batch"])

    tot = {"n": 0, "ok": 0}
    sub = {"seen": {"n": 0, "ok": 0}, "unseen": {"n": 0, "ok": 0}}
    with torch.inference_mode():
        for lo in range(0, len(ds), bs):
            idxs = list(range(lo, min(lo + bs, len(ds))))
            batch = collate([ds[i] for i in idxs])
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(batch["chars"].to(device))
            surfaces = [ds.chunk_surfaces(i) for i in idxs]
            preds = decode_batch(out, vocabs, surfaces, cfg.infer.classifier_min_prob,
                                 model=model)
            golds = [ds.chunk_gold(i) for i in idxs]
            for b in range(len(idxs)):
                for t, gold in enumerate(golds[b]):
                    if gold is None:
                        continue
                    p = preds[b][t]
                    lemma = snapper.snap(p.lemma, len(p.pos.split("+"))) if snapper else p.lemma
                    ok = lemma == gold[0]
                    tot["n"] += 1
                    tot["ok"] += ok
                    k = "seen" if surfaces[b][t] in seen else "unseen"
                    sub[k]["n"] += 1
                    sub[k]["ok"] += ok
            if lo % (bs * 40) == 0:
                print(f"  {lo}/{len(ds)} chunks", flush=True)

    pct = lambda d: 100.0 * d["ok"] / d["n"] if d["n"] else 0.0
    res = {
        "run": run.name,
        "n": tot["n"], "lemma": pct(tot),
        "seen_n": sub["seen"]["n"], "seen_lemma": pct(sub["seen"]),
        "unseen_n": sub["unseen"]["n"], "unseen_lemma": pct(sub["unseen"]),
        "unseen_pct": 100.0 * sub["unseen"]["n"] / tot["n"] if tot["n"] else 0.0,
    }
    for k, v in res.items():
        print(f"  {k:14s} {v:,.2f}" if isinstance(v, float) else f"  {k:14s} {v}")

    # --- the stratified side, from its dump and its own recorded vocabulary.
    # Both sides are snapped lemma accuracy, the metric Table 3 reports; the
    # oov_acc_lemma quoted elsewhere in the paper is the unsnapped one and is
    # about a point lower, so the two must not be mixed in one subtraction.
    srun = pathlib.Path(opts["strat_run"])
    srun = srun if srun.is_absolute() else repo / srun
    train_surf = set(json.loads((srun / "vocab.json").read_text(encoding="utf-8"))
                     ["train_surfaces"])
    rows = [l.split(chr(9)) for l in (srun.parent / f'{opts["strat_dump"]}.tsv')
            .read_text(encoding="utf-8").splitlines()[1:]]
    st = {"n": 0, "ok": 0, "seen_n": 0, "seen_ok": 0, "unseen_n": 0, "unseen_ok": 0}
    for surf, gl, _gp, pl, _pp in rows:
        good = pl == gl
        st["n"] += 1
        st["ok"] += good
        k = "seen" if surf in train_surf else "unseen"
        st[k + "_n"] += 1
        st[k + "_ok"] += good
    s = {"n": st["n"], "lemma": pct({"n": st["n"], "ok": st["ok"]}),
         "seen_n": st["seen_n"], "seen_lemma": pct({"n": st["seen_n"], "ok": st["seen_ok"]}),
         "unseen_n": st["unseen_n"],
         "unseen_lemma": pct({"n": st["unseen_n"], "ok": st["unseen_ok"]}),
         "unseen_pct": 100.0 * st["unseen_n"] / st["n"]}

    # Gap = a composition term (the protocols hold different proportions of
    # unseen tokens) plus a performance term on each subset. The three sum to
    # the gap by construction, which is the check printed below.
    rc, rs = res["unseen_pct"] / 100, s["unseen_pct"] / 100
    gap = res["lemma"] - s["lemma"]
    comp = (rs - rc) * (s["seen_lemma"] - s["unseen_lemma"])
    pseen = (1 - rc) * (res["seen_lemma"] - s["seen_lemma"])
    punseen = rc * (res["unseen_lemma"] - s["unseen_lemma"])
    out = {"chunk": res, "stratified": s, "gap_pp": gap,
           "composition_pp": comp, "seen_perf_pp": pseen, "unseen_perf_pp": punseen,
           "composition_share_pct": 100 * comp / gap,
           "seen_share_pct": 100 * pseen / gap,
           "unseen_share_pct": 100 * punseen / gap,
           "residual_pp": gap - (comp + pseen + punseen)}
    print()
    print(f"  gap {gap:.2f} pts = composition {comp:.2f} ({100*comp/gap:.0f}%)"
          f" + seen {pseen:.2f} ({100*pseen/gap:.0f}%)"
          f" + unseen {punseen:.2f} ({100*punseen/gap:.0f}%)"
          f"   residual {out['residual_pp']:.4f}")
    dest = repo / opts["out"]
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest.relative_to(repo).as_posix())


if __name__ == "__main__":
    main()
