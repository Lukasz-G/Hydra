"""Channel gating, context self-attention, masked-token auxiliary."""
import dataclasses

import torch

from hydra.config import LossConfig
from hydra.data import IGNORE, HydraDataset, load_split_tokens
from hydra.losses import compute_loss
from hydra.model import HydraModel
from hydra.vocab import PAD, Vocabs


def test_forward_with_gate_attn_mlm(model_cfg):
    cfg = dataclasses.replace(model_cfg, tcn_channel_gate=True, ctx_self_attention=True,
                              masked_lm=True, lemma_classifier=True)
    model = HydraModel(cfg, n_chars=30, n_pos=10, n_morph=12, max_word_len=12,
                       max_lemma_len=16, chunk_len=8, halo=4,
                       n_lemma_types=50, n_word_types=40)
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(3, 30, (2, 16, 12), generator=g)
    out = model(chars)
    assert out.mlm_logits is not None and out.mlm_logits.shape == (2, 8, 40)
    for t in (out.pos_logits, out.lemma_logits, out.mlm_logits):
        assert not torch.isnan(t).any()
    # plain (no-extras) state dict warm-starts with only the new modules missing
    base = HydraModel(model_cfg, 30, 10, 12, 12, 16, 8, 4)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(any(s in k for s in ("gate", "ctx_attn", "mlm_head", "lemma_cls_head"))
               for k in missing)


def test_masking_transform(corpus_dir, data_cfg, model_cfg):
    import dataclasses as dc
    cfg = dc.replace(data_cfg, mask_prob=1.0)  # mask every real token
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", 4)
    vocabs = Vocabs.build([t for d in docs for t in d], word_type_min_freq=1)
    ds = HydraDataset(docs, vocabs, cfg, 4, training=True, mask_tokens=True)
    item = ds[0]
    real = (item["mlm"] != IGNORE)
    assert real.any()
    H = cfg.halo
    for t in torch.nonzero(real).flatten().tolist():
        assert (item["chars"][H + t] == PAD).all()      # input blanked
        assert (item["pos"][t] == IGNORE).all()          # tagging targets off
        assert item["mlm"][t] == vocabs.word_types.encode(ds.chunk_surfaces(0)[t])
    # eval dataset: never masked
    ds_eval = HydraDataset(docs, vocabs, cfg, 4)
    assert (ds_eval[0]["mlm"] == IGNORE).all()


def test_mlm_loss(model_cfg, corpus_dir, data_cfg):
    import dataclasses as dc
    cfg = dc.replace(model_cfg, masked_lm=True)
    model = HydraModel(cfg, n_chars=30, n_pos=10, n_morph=12, max_word_len=12,
                       max_lemma_len=16, chunk_len=8, halo=4, n_word_types=40)
    g = torch.Generator().manual_seed(1)
    chars = torch.randint(3, 30, (1, 16, 12), generator=g)
    out = model(chars)
    batch = {
        "pos": torch.full((1, 8, 4), IGNORE, dtype=torch.int64),
        "morph": torch.full((1, 8, 4), IGNORE, dtype=torch.int64),
        "lemma": torch.full((1, 8, 4, 16), IGNORE, dtype=torch.int64),
        "mlm": torch.full((1, 8), IGNORE, dtype=torch.int64),
    }
    batch["mlm"][0, 3] = 7
    loss, parts = compute_loss(out, batch, LossConfig(), n_pos=10)
    assert parts["loss_mlm"] > 0
    assert torch.isfinite(loss)
    loss.backward()