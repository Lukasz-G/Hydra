"""Explicit item-count head (model.count_head).

Item count was implicit: decoding read slots until the first NULL POS. The
error analysis localises the multi-item gap there -- count is right ~87% of
the time and the errors are UNDER-segmentation to n=1, while content given
the count is about as good as on single-item tokens. This head predicts the
count directly, so the test that matters is that decoding actually OBEYS it
even when the first-NULL rule would disagree.
"""
import dataclasses

import torch

from hydra.config import LossConfig
from hydra.data import IGNORE
from hydra.losses import compute_loss
from hydra.metrics import decode_batch
from hydra.model import HydraModel
from hydra.vocab import NULL, PAD

N_CHARS, N_POS, N_MORPH = 30, 10, 12
T, H, W, L = 8, 4, 12, 16


def build(model_cfg, **kw):
    cfg = dataclasses.replace(model_cfg, **kw)
    torch.manual_seed(0)
    return HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H)


def random_chars(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    chars = torch.randint(3, N_CHARS, (B, T + 2 * H, W), generator=g)
    chars[:, :, 6:] = PAD
    return chars


def test_off_by_default_and_output_absent(model_cfg):
    model = build(model_cfg)
    assert model.count_head is None
    with torch.no_grad():
        out = model(random_chars())
    assert out.count_logits is None


def test_shape(model_cfg):
    K = model_cfg.n_slots
    model = build(model_cfg, count_head=True)
    with torch.no_grad():
        out = model(random_chars())
    # classes 0..K; 0 marks context-only and is never a target
    assert out.count_logits.shape == (2, T, K + 1)


def test_decode_obeys_the_predicted_count(model_cfg):
    """The point of the head: emit n items because the COUNT head says n, even
    where the first-NULL rule would have stopped at 1."""
    from hydra.data import Token
    from hydra.vocab import Vocabs
    K = model_cfg.n_slots
    # build the MODEL from the vocab, not the other way round: a model whose
    # head is wider than the vocab emits ids that cannot be decoded, and the
    # resulting IndexError would mask what this test is actually checking
    toks = [Token(f"w{i}", [f"l{i}"], [f"P{i}"], [f"M{i}"]) for i in range(N_MORPH)]
    vocabs = Vocabs.build(toks)
    cfg = dataclasses.replace(model_cfg, count_head=True)
    torch.manual_seed(0)
    model = HydraModel(cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       W, L, T, H)
    model.eval()
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(3, len(vocabs.chars), (2, T + 2 * H, W), generator=g)
    chars[:, :, 6:] = PAD
    with torch.no_grad():
        out = model(chars)

    B, Tt = out.pos_logits.shape[0], out.pos_logits.shape[1]
    # make the first-NULL rule say "1 item" everywhere: slot 1 onward is NULL
    pos = out.pos_logits.clone()
    pos[:, :, 1:, :] = -1e4
    pos[:, :, 1:, NULL] = 1e4
    # ...but have the count head insist on 3
    cnt = torch.full((B, Tt, K + 1), -1e4)
    cnt[..., 3] = 1e4
    out = dataclasses.replace(out, pos_logits=pos, count_logits=cnt)

    preds = decode_batch(out, vocabs, [["x"] * Tt for _ in range(B)], 0.3, model=model)
    n_items = len(preds[0][0].pos.split("+"))
    assert n_items == 3, f"decode ignored the count head (emitted {n_items})"


def test_count_loss_ignores_context_only_tokens(model_cfg):
    """n_items == 0 marks a context-only token and doubles as the ignore
    index, so an all-context batch must contribute exactly zero count loss."""
    K = model_cfg.n_slots
    model = build(model_cfg, count_head=True)
    with torch.no_grad():
        out = model(random_chars())
    B, Tt = 2, T
    batch = {
        "pos": torch.full((B, Tt, K), IGNORE),
        "morph": torch.full((B, Tt, K), IGNORE),
        "lemma": torch.full((B, Tt, K, L), IGNORE),
        "n_items": torch.zeros(B, Tt, dtype=torch.long),   # all context-only
    }
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts.get("loss_count", 0.0) == 0.0

    batch["n_items"] = torch.full((B, Tt), 2, dtype=torch.long)
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts["loss_count"] > 0.0


def test_low_confidence_count_falls_back_to_first_null(model_cfg):
    """The warm-start guarantee.

    An untrained count head peaks near 1/(K+1), far below infer.count_min_prob,
    so decoding must ignore it and use the first-NULL rule -- otherwise adding
    the head to a trained checkpoint destroys it (observed: epoch-0 dev lemma
    0.108 before this gate existed). Contrast the tag-condition gate, which
    fell back to an UNTRAINED row and so only ever hurt; this one falls back to
    a trained, working mechanism.
    """
    from hydra.data import Token
    from hydra.vocab import Vocabs
    K = model_cfg.n_slots
    toks = [Token(f"w{i}", [f"l{i}"], [f"P{i}"], [f"M{i}"]) for i in range(N_MORPH)]
    vocabs = Vocabs.build(toks)
    cfg = dataclasses.replace(model_cfg, count_head=True)
    torch.manual_seed(0)
    model = HydraModel(cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph), W, L, T, H)
    model.eval()
    model.count_min_prob = 0.5
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(3, len(vocabs.chars), (2, T + 2 * H, W), generator=g)
    chars[:, :, 6:] = PAD
    with torch.no_grad():
        out = model(chars)

    B, Tt = out.pos_logits.shape[0], out.pos_logits.shape[1]
    pos = out.pos_logits.clone()
    pos[:, :, 1:, :] = -1e4
    pos[:, :, 1:, NULL] = 1e4              # first-NULL says 1 item
    flat = torch.zeros(B, Tt, K + 1)       # uniform -> max prob ~1/K, under the bar
    out = dataclasses.replace(out, pos_logits=pos, count_logits=flat)

    preds = decode_batch(out, vocabs, [["x"] * Tt for _ in range(B)], 0.3, model=model)
    assert len(preds[0][0].pos.split("+")) == 1, "unconfident count head was obeyed"
