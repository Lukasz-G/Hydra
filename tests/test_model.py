import dataclasses

import torch

from hydra.model import HydraModel
from hydra.vocab import PAD


def build_model(model_cfg, n_chars=30, n_pos=10, n_morph=12, T=8, H=4,
                max_word_len=12, max_lemma_len=16):
    return HydraModel(model_cfg, n_chars, n_pos, n_morph, max_word_len, max_lemma_len, T, H)


def random_chars(B, S, W, n_chars, seed=0):
    g = torch.Generator().manual_seed(seed)
    chars = torch.randint(3, n_chars, (B, S, W), generator=g)
    chars[:, :, 6:] = PAD  # words 6 chars long
    return chars


def test_forward_shapes(model_cfg):
    T, H, W, L = 8, 4, 12, 16
    model = build_model(model_cfg, T=T, H=H, max_word_len=W, max_lemma_len=L)
    chars = random_chars(2, T + 2 * H, W, 30)
    out = model(chars)
    K = model_cfg.n_slots
    assert out.pos_logits.shape == (2, T, K, 10)
    assert out.morph_logits.shape == (2, T, K, 12)
    assert out.lemma_logits.shape == (2, T, K, L, 30)
    for t in (out.pos_logits, out.morph_logits, out.lemma_logits):
        assert not torch.isnan(t).any()


def test_forward_no_cross_attention(model_cfg):
    cfg = dataclasses.replace(model_cfg, lemma_cross_attention=False)
    model = build_model(cfg)
    out = model(random_chars(1, 16, 12, 30))
    assert out.lemma_logits.shape == (1, 8, 4, 16, 30)


def test_all_pad_token_no_nan(model_cfg):
    model = build_model(model_cfg).eval()
    chars = random_chars(1, 16, 12, 30)
    chars[0, 3] = PAD  # a fully padded token inside the window
    out = model(chars)
    assert not torch.isnan(out.pos_logits).any()
    assert not torch.isnan(out.lemma_logits).any()


def test_context_receptive_field(model_cfg):
    # ctx dilations (1,) with 2 convs/block -> reach = 2 tokens per side
    T, H = 8, 4
    model = build_model(model_cfg, T=T, H=H).eval()
    base = random_chars(1, T + 2 * H, 12, 30)
    center_t = 4                       # central position, S index H+4 = 8
    with torch.no_grad():
        ref = model(base).pos_logits[0, center_t]

        far = base.clone()             # S index 8+3=11 -> beyond reach
        far[0, 11] = torch.randint(3, 30, (12,))
        far[0, 11, 6:] = PAD
        out_far = model(far).pos_logits[0, center_t]

        near = base.clone()            # S index 9 -> inside reach
        near[0, 9] = torch.randint(3, 30, (12,))
        near[0, 9, 6:] = PAD
        out_near = model(near).pos_logits[0, center_t]

    assert torch.allclose(ref, out_far, atol=1e-5)
    assert not torch.allclose(ref, out_near, atol=1e-5)
