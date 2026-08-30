"""Error decomposition of runs/mhd_base/best.pt on the dev split."""
import collections
import json
from pathlib import Path

import torch

from hydra.data import HydraDataset, collate, load_split_tokens, split_lemma_items
from hydra.metrics import decode_batch, levenshtein
from hydra.tag import load_model_for_inference

RUN = Path("runs/mhd_base")
OUT = RUN / "error_analysis.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, vocabs, cfg = load_model_for_inference(RUN / "best.pt", device)
splits = json.loads((RUN / "split.json").read_text(encoding="utf-8"))

# ---- train-side inventories -------------------------------------------------
print("parsing train split for lemma inventory ...", flush=True)
train_docs = load_split_tokens(splits["train"], cfg.data.on_mismatch, cfg.model.n_slots)
inventory: set[str] = set()
surf2lem: dict[str, set[str]] = collections.defaultdict(set)
for doc in train_docs:
    for t in doc:
        if t.lemmas:
            inventory.update(t.lemmas)
            surf2lem[t.surface].add("+".join(t.lemmas))
del train_docs
print(f"inventory: {len(inventory)} atomic lemmas", flush=True)

# SymSpell-style deletion index for distance-1 lookup
del_index: dict[str, list[str]] = collections.defaultdict(list)
def deletions(s):
    yield s
    for i in range(len(s)):
        yield s[:i] + s[i + 1:]
for lem in inventory:
    for d in deletions(lem):
        del_index[d].append(lem)

def snap_candidates(item: str) -> set[str]:
    """Inventory lemmas within edit distance 1 of item."""
    cands = set()
    for d in deletions(item):
        for lem in del_index.get(d, ()):
            if levenshtein(item, lem) <= 1:
                cands.add(lem)
    return cands

# ---- run the model over dev -------------------------------------------------
print("running model over dev ...", flush=True)
dev_docs = load_split_tokens(splits["dev"], cfg.data.on_mismatch, cfg.model.n_slots)
ds = HydraDataset(dev_docs, vocabs, cfg.data, cfg.model.n_slots)

records = []  # (surface, gold_lemma, gold_pos, gold_morph, pred_lemma, pred_pos, pred_morph, gold_n)
with torch.inference_mode():
    for lo in range(0, len(ds), 32):
        idxs = list(range(lo, min(lo + 32, len(ds))))
        batch = collate([ds[i] for i in idxs])
        out = model(batch["chars"].to(device))
        surfaces = [ds.chunk_surfaces(i) for i in idxs]
        preds = decode_batch(out, vocabs, surfaces)
        n_items = batch["n_items"].numpy()
        for b, i in enumerate(idxs):
            for t, gold in enumerate(ds.chunk_gold(i)):
                if gold is None:
                    continue
                p = preds[b][t]
                records.append((surfaces[b][t], gold[0], gold[1], gold[2],
                                p.lemma, p.pos, p.morph, int(n_items[b, t])))
print(f"{len(records)} supervised dev tokens", flush=True)

# ---- analyses ---------------------------------------------------------------
rep = []
def emit(line=""):
    rep.append(line)
    print(line, flush=True)

N = len(records)
lem_err = [r for r in records if r[4] != r[1]]
pos_err = [r for r in records if r[5] != r[2]]
mor_err = [r for r in records if r[6] != r[3]]
emit(f"dev tokens: {N}")
emit(f"errors: lemma {len(lem_err)} ({len(lem_err)/N:.1%})  pos {len(pos_err)} "
     f"({len(pos_err)/N:.1%})  morph {len(mor_err)} ({len(mor_err)/N:.1%})")

# 1. lemma error decomposition
emit("\n== LEMMA ERRORS ==")
dist_hist = collections.Counter()
pred_nonword = pred_realword = 0
for r in lem_err:
    d = levenshtein(r[4], r[1])
    dist_hist[min(d, 3)] += 1
    items = split_lemma_items(r[4], len(r[5].split("+")))
    if all(it in inventory for it in items):
        pred_realword += 1
    else:
        pred_nonword += 1
for d in sorted(dist_hist):
    label = f"distance {d}" if d < 3 else "distance >=3"
    emit(f"  {label}: {dist_hist[d]} ({dist_hist[d]/len(lem_err):.1%} of errors)")
emit(f"  prediction is a known train lemma (wrong choice): {pred_realword} ({pred_realword/len(lem_err):.1%})")
emit(f"  prediction is a non-word (hallucinated spelling): {pred_nonword} ({pred_nonword/len(lem_err):.1%})")

# gold reachable?
gold_in_inv = sum(1 for r in lem_err
                  if all(it in inventory for it in split_lemma_items(r[1], len(r[2].split('+')))))
emit(f"  gold lemma fully in train inventory: {gold_in_inv} ({gold_in_inv/len(lem_err):.1%})")

# 2. lexicon-snap simulation (distance-1, unique candidate, only non-inventory items)
emit("\n== LEXICON SNAP SIMULATION (dist<=1, unique candidate, non-inventory items only) ==")
wins = losses = changed = 0
for r in records:
    n = len(r[5].split("+"))
    items = split_lemma_items(r[4], n)
    new_items = []
    any_change = False
    for it in items:
        if it and it not in inventory:
            cands = snap_candidates(it)
            if len(cands) == 1:
                new_items.append(next(iter(cands)))
                any_change = True
                continue
        new_items.append(it)
    if not any_change:
        continue
    changed += 1
    new = "+".join(new_items)
    was_ok, now_ok = r[4] == r[1], new == r[1]
    if now_ok and not was_ok:
        wins += 1
    elif was_ok and not now_ok:
        losses += 1
emit(f"  tokens changed by snapping: {changed}")
emit(f"  wins (wrong->right): {wins}   losses (right->wrong): {losses}")
emit(f"  net lemma accuracy delta: {(wins-losses)/N:+.2%}")

# 3. multi-item decomposition
emit("\n== MULTI-ITEM TOKENS (gold n>1) ==")
multi = [r for r in records if r[7] > 1]
count_ok = [r for r in multi if len(r[5].split('+')) == r[7]]
emit(f"  n={len(multi)}; predicted item COUNT correct: {len(count_ok)} ({len(count_ok)/len(multi):.1%})")
cnt_pred = collections.Counter(len(r[5].split('+')) for r in multi if len(r[5].split('+')) != r[7])
emit(f"  count-error distribution (predicted counts): {dict(sorted(cnt_pred.items()))}")
if count_ok:
    l_ok = sum(1 for r in count_ok if r[4] == r[1])
    p_ok = sum(1 for r in count_ok if r[5] == r[2])
    m_ok = sum(1 for r in count_ok if r[6] == r[3])
    emit(f"  given correct count: lemma {l_ok/len(count_ok):.1%}  pos {p_ok/len(count_ok):.1%}  morph {m_ok/len(count_ok):.1%}")
single = [r for r in records if r[7] == 1]
oversplit = sum(1 for r in single if len(r[5].split('+')) > 1)
emit(f"  over-split (gold n=1, predicted n>1): {oversplit} / {len(single)} ({oversplit/len(single):.2%})")

# 4. POS confusions
emit("\n== TOP POS CONFUSIONS (gold -> pred) ==")
conf = collections.Counter((r[2], r[5]) for r in pos_err)
for (g, p), c in conf.most_common(12):
    emit(f"  {g:>12} -> {p:<12} {c}")

# 5. morph cascade
emit("\n== MORPH vs POS ==")
pos_ok = [r for r in records if r[5] == r[2]]
pos_bad = [r for r in records if r[5] != r[2]]
m1 = sum(1 for r in pos_ok if r[6] == r[3]) / max(1, len(pos_ok))
m2 = sum(1 for r in pos_bad if r[6] == r[3]) / max(1, len(pos_bad))
emit(f"  morph accuracy when POS correct: {m1:.1%}   when POS wrong: {m2:.1%}")

# 6. homographs (seen surfaces only)
emit("\n== SEEN-SURFACE AMBIGUITY (lemma) ==")
seen = [r for r in records if r[0] in surf2lem]
unamb = [r for r in seen if len(surf2lem[r[0]]) == 1]
amb = [r for r in seen if len(surf2lem[r[0]]) > 1]
if unamb:
    emit(f"  unambiguous seen surfaces: n={len(unamb)}, lemma acc {sum(1 for r in unamb if r[4]==r[1])/len(unamb):.1%}")
if amb:
    emit(f"  ambiguous seen surfaces:   n={len(amb)}, lemma acc {sum(1 for r in amb if r[4]==r[1])/len(amb):.1%}")

OUT.write_text("\n".join(rep), encoding="utf-8")
print(f"\nreport saved to {OUT}", flush=True)
