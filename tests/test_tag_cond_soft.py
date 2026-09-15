"""Soft tag conditioning (model.tag_cond_soft): consume the tag DISTRIBUTION.

The confidence gate (infer.tag_cond_min_prob) was measured on 2026-09-14 and
only ever hurt -- monotonically, and worst on multi-item tokens (-4.26pp at
gate 0.9). It rejected the argmax in favour of the "unsure" row, which gets no
gradient in training. Soft conditioning is the repair: a blend over the TRAINED
rows only, which neither discards the argmax nor touches the untrained row.

The tests that matter here are the two guarantees the run design rests on:
lambda=0 is a bit-exact no-op, and the blend never reads the unsure row.
"""
import dataclasses

import torch

from hydra.model import HydraModel
from hydra.vocab import PAD

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


def tags(B, K, pos_val=3, morph_val=5):
    return (torch.full((B, T, K), pos_val), torch.full((B, T, K), morph_val))


def _pair(model_cfg, **kw):
    """A hard model and a soft model with identical weights.

    tag_to_lemma / tag_to_morph are ZERO-initialised by design, which makes
    conditioning completely inert in a freshly built model -- every test in
    this file would pass vacuously against it. Randomise them, which is the
    state any warm-started run is actually in.
    """
    hard = build(model_cfg, tag_condition="morph+lemma", tag_cond_soft=False, **kw)
    with torch.no_grad():
        for lin in (hard.tag_to_lemma, hard.tag_to_morph):
            torch.nn.init.normal_(lin.weight, std=0.2)
            torch.nn.init.normal_(lin.bias, std=0.2)
    soft = build(model_cfg, tag_condition="morph+lemma", tag_cond_soft=True, **kw)
    soft.load_state_dict(hard.state_dict())
    hard.eval()
    soft.eval()
    return hard, soft


def test_lambda_zero_is_bit_exact_noop(model_cfg):
    """THE guarantee the warm start rests on: at lambda=0 the soft model is
    bit-identical to the hard one, so a run warm-started from a hard-conditioned
    checkpoint changes nothing at step 0 and the blend is the isolated change."""
    hard, soft = _pair(model_cfg)
    soft.tag_cond_lambda = 0.0
    K = model_cfg.n_slots
    chars, tt = random_chars(), tags(2, K)
    with torch.no_grad():
        a = hard(chars, tag_teacher=tt)
        b = soft(chars, tag_teacher=tt)
    for x, y in ((a.pos_logits, b.pos_logits), (a.morph_logits, b.morph_logits),
                 (a.lemma_logits, b.lemma_logits)):
        assert torch.equal(x, y)


def test_lambda_one_actually_changes_something(model_cfg):
    """Guards against the blend being silently inert (a no-op test alone would
    pass just as well if tag_cond_soft were never read)."""
    hard, soft = _pair(model_cfg)
    soft.tag_cond_lambda = 1.0
    K = model_cfg.n_slots
    chars, tt = random_chars(), tags(2, K)
    with torch.no_grad():
        a = hard(chars, tag_teacher=tt)
        b = soft(chars, tag_teacher=tt)
    # POS is upstream of the conditioning: it must NOT move
    assert torch.equal(a.pos_logits, b.pos_logits)
    # morph and lemma are downstream: they must
    assert not torch.allclose(a.morph_logits, b.morph_logits)
    assert not torch.allclose(a.lemma_logits, b.lemma_logits)


def test_soft_never_reads_the_unsure_row(model_cfg):
    """The gate's failure mode, encoded as a test.

    Row n_classes is the "unsure" slot. No gradient reaches it in training, so
    it stays near its random init; feeding it is what made the confidence gate
    hurt. Poison it: soft conditioning must be completely unaffected, while the
    hard path (which routes IGNORE targets there) must not be.
    """
    hard, soft = _pair(model_cfg)
    soft.tag_cond_lambda = 1.0
    K = model_cfg.n_slots
    chars = random_chars()

    with torch.no_grad():
        before = soft(chars, tag_teacher=tags(2, K)).lemma_logits.clone()
        for m in (soft, hard):
            m.pos_cond_emb.weight[N_POS].fill_(1e3)      # the unsure rows
            m.morph_cond_emb.weight[N_MORPH].fill_(1e3)
        after = soft(chars, tag_teacher=tags(2, K)).lemma_logits
    assert torch.equal(before, after), "soft conditioning read the unsure row"

    # control: the hard path DOES read it when a target is IGNORE, so the
    # poisoning above is real and the test above is not vacuous
    from hydra.data import IGNORE
    ig = (torch.full((2, T, K), IGNORE), torch.full((2, T, K), IGNORE))
    with torch.no_grad():
        clean, _ = _pair(model_cfg)
        poisoned = clean(chars, tag_teacher=ig).lemma_logits.clone()
        clean.pos_cond_emb.weight[N_POS].fill_(1e3)
        clean.morph_cond_emb.weight[N_MORPH].fill_(1e3)
        assert not torch.equal(poisoned, clean(chars, tag_teacher=ig).lemma_logits)


def test_blend_is_convex_in_lambda(model_cfg):
    """lambda genuinely interpolates, so the ramp is a smooth path from the
    warm-started model to the soft one rather than a jump somewhere in between."""
    _, soft = _pair(model_cfg)
    K = model_cfg.n_slots
    chars, tt = random_chars(), tags(2, K)
    outs = {}
    for lam in (0.0, 0.5, 1.0):
        soft.tag_cond_lambda = lam
        with torch.no_grad():
            outs[lam] = soft(chars, tag_teacher=tt).morph_logits.clone()
    # the morph head is affine in its conditioning input, so a convex blend of
    # the conditioning vector gives a convex blend of the logits
    assert torch.allclose(outs[0.5], 0.5 * (outs[0.0] + outs[1.0]), atol=1e-4)


def test_distribution_is_detached(model_cfg):
    """This version changes what the lemma head CONSUMES, not what trains the
    tagger: no gradient may flow from the lemma path back into the POS head."""
    _, soft = _pair(model_cfg)
    soft.tag_cond_lambda = 1.0
    soft.train()
    K = model_cfg.n_slots
    out = soft(random_chars(), tag_teacher=tags(2, K))
    soft.zero_grad(set_to_none=True)
    out.lemma_logits.sum().backward()
    g = soft.pos_head.weight.grad
    assert g is None or torch.count_nonzero(g) == 0, "gradient leaked into pos_head"
