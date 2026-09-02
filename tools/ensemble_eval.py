"""Logit-averaging ensemble evaluation.

Usage: python tools/ensemble_eval.py SPLIT CKPT1 CKPT2 [CKPT3 ...]

SPLIT is 'dev' or 'test' (resolved from the FIRST checkpoint's run dir:
split.json and vocab.json must sit next to it; its config drives the data
pipeline). Each CKPT is a checkpoint containing at least {'model', 'config'}.
Averages fp32 logits per head across models (lemma_cls averaged over the
models that have the head) and reports the standard metric block.
"""
import json
import sys
from pathlib import Path

import torch

from hydra.data import HydraDataset, collate, load_split_tokens
from hydra.evaluate import DAMAGED_POS
from hydra.metrics import EvalAccumulator, decode_batch
from hydra.model import ModelOutput
from hydra.snap import LemmaSnapper
from hydra.tag import load_model_for_inference

split, ckpts = sys.argv[1], sys.argv[2:]
assert split in ("dev", "test") and len(ckpts) >= 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = []
vocabs = cfg = None
for c in ckpts:
    m, v, cf = load_model_for_inference(c, device)
    models.append(m)
    if vocabs is None:
        vocabs, cfg = v, cf
        run = Path(c).parent

splits = json.loads((run / "split.json").read_text(encoding="utf-8"))
docs = load_split_tokens(splits[split], cfg.data.on_mismatch, cfg.model.n_slots)
role = split if cfg.data.split_mode == "chunk" else None
ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots, role=role)
snapper = LemmaSnapper(vocabs.lemma_inventory)

acc = EvalAccumulator()
snap_ok = 0
clean_acc = EvalAccumulator()
with torch.inference_mode():
    for lo in range(0, len(ds), 16):
        idxs = list(range(lo, min(lo + 16, len(ds))))
        batch = collate([ds[i] for i in idxs])
        chars = batch["chars"].to(device)
        pos = morph = lemma = None
        cls_sum, cls_n = None, 0
        for m in models:
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = m(chars)
            if out.lemma_logits is None:
                sys.exit("ensemble_eval supports grid-decoder models only (AR models "
                         "produce no parallel lemma logits to average)")
            pos = out.pos_logits.float() if pos is None else pos + out.pos_logits.float()
            morph = out.morph_logits.float() if morph is None else morph + out.morph_logits.float()
            lemma = out.lemma_logits.float() if lemma is None else lemma + out.lemma_logits.float()
            if out.lemma_cls_logits is not None:
                c = out.lemma_cls_logits.float()
                cls_sum = c if cls_sum is None else cls_sum + c
                cls_n += 1
        n = len(models)
        avg = ModelOutput(pos / n, morph / n, lemma / n,
                          cls_sum / cls_n if cls_sum is not None else None)
        surfaces = [ds.chunk_surfaces(i) for i in idxs]
        preds = decode_batch(avg, vocabs, surfaces, cfg.infer.classifier_min_prob)
        golds = [ds.chunk_gold(i) for i in idxs]
        n_items = batch["n_items"].numpy()
        for b in range(len(idxs)):
            for t, gold in enumerate(golds[b]):
                if gold is None:
                    continue
                p = preds[b][t]
                acc.update(p, surfaces[b][t], gold[0], gold[1], gold[2],
                           int(n_items[b, t]), vocabs.train_surfaces)
                if gold[1] != DAMAGED_POS:
                    clean_acc.update(p, surfaces[b][t], gold[0], gold[1], gold[2],
                                     int(n_items[b, t]), vocabs.train_surfaces)
                snap_ok += snapper.snap(p.lemma, len(p.pos.split("+"))) == gold[0]

m = acc.as_dict()
m["acc_lemma_snapped"] = snap_ok / max(1, acc.overall.n)
for k, v in clean_acc.as_dict().items():
    m[f"clean_{k}"] = v
print(f"ENSEMBLE({len(models)}) {split}: " + " ".join(
    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
