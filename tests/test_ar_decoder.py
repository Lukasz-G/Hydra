"""Autoregressive causal-TCN lemma decoder."""
import dataclasses

import torch

from hydra.config import LossConfig
from hydra.data import HydraDataset, collate, load_split_tokens
from hydra.evaluate import evaluate_dataset
from hydra.losses import compute_loss
from hydra.model import HydraModel, TCNBlock
from hydra.vocab import Vocabs


def test_causal_block_no_future_leak():
    torch.manual_seed(0)
    block = TCNBlock(16, 3, 2, dropout=0.0, causal=True).eval()
    x = torch.randn(1, 10, 16)
    y1 = block(x)
    x2 = x.clone()
    x2[0, 7] += 5.0  # perturb a later position
    y2 = block(x2)
    assert torch.allclose(y1[0, :7], y2[0, :7], atol=1e-6)   # past unchanged
    assert not torch.allclose(y1[0, 7:], y2[0, 7:], atol=1e-6)


def test_ar_teacher_forcing_and_generation(model_cfg):
    cfg = dataclasses.replace(model_cfg, lemma_decoder="ar_tcn")
    model = HydraModel(cfg, n_chars=30, n_pos=10, n_morph=12, max_word_len=12,
                       max_lemma_len=16, chunk_len=8, halo=4).eval()
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(3, 30, (2, 16, 12), generator=g)
    teacher = torch.randint(3, 30, (2, 8, 4, 16), generator=g)
    out = model(chars, lemma_teacher=teacher)
    assert out.lemma_logits.shape == (2, 8, 4, 16, 30)
    # causality through the full decoder: changing a later teacher char must
    # not affect earlier positions' logits
    teacher2 = teacher.clone()
    teacher2[0, 0, 0, 10] = 5
    out2 = model(chars, lemma_teacher=teacher2)
    assert torch.allclose(out.lemma_logits[0, 0, 0, :10], out2.lemma_logits[0, 0, 0, :10],
                          atol=1e-5)
    # no-teacher forward exposes generation state instead of logits
    out3 = model(chars)
    assert out3.lemma_logits is None and out3.slot_states is not None
    gen = model.lemma_decoder.generate(out3.slot_states, out3.char_states,
                                       out3.char_pad_mask, 16)
    assert gen.shape == (2 * 8 * 4, 16)


def test_ar_overfit(corpus_dir, data_cfg, model_cfg):
    torch.manual_seed(0)
    cfg = dataclasses.replace(model_cfg, lemma_decoder="ar_tcn", lemma_classifier=True)
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", cfg.n_slots)
    vocabs = Vocabs.build([t for d in docs for t in d])
    ds = HydraDataset(docs, vocabs, data_cfg, cfg.n_slots, training=True)
    batch = collate([ds[i] for i in range(len(ds))])
    model = HydraModel(cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       data_cfg.max_word_len, data_cfg.max_lemma_len,
                       data_cfg.chunk_len, data_cfg.halo,
                       n_lemma_types=len(vocabs.lemma_types))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    metrics = {}
    for it in range(1, 801):
        opt.zero_grad(set_to_none=True)
        out = model(batch["chars"], lemma_teacher=batch["lemma"])
        loss, _ = compute_loss(out, batch, LossConfig(), len(vocabs.pos))
        loss.backward()
        opt.step()
        if it % 200 == 0:
            metrics = evaluate_dataset(model, ds, vocabs, torch.device("cpu"), 4)
            if metrics.get("acc_joint", 0.0) == 1.0:
                break
    assert metrics.get("acc_joint", 0.0) == 1.0, f"failed to memorize: {metrics}"
