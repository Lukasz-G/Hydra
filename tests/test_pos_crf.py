"""Linear-chain CRF over the slot-0 POS sequence (model.pos_crf)."""
import dataclasses

import pytest
import torch

from hydra.config import LossConfig
from hydra.data import IGNORE
from hydra.losses import compute_loss
from hydra.model import HydraModel, PosCRF
from hydra.vocab import NULL, PAD

N_CHARS, N_POS, N_MORPH = 30, 10, 12
T, H, W, L = 8, 4, 12, 16


def build(model_cfg, **kw):
    cfg = dataclasses.replace(model_cfg, **kw)
    return HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H)


def random_chars(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    chars = torch.randint(3, N_CHARS, (B, T + 2 * H, W), generator=g)
    chars[:, :, 6:] = PAD
    return chars


# ---------------------------------------------------------------- CRF maths

def test_log_partition_matches_brute_force():
    """The forward algorithm must equal an explicit sum over every path.
    Checked on a short sequence where enumeration is feasible."""
    torch.manual_seed(0)
    n, t = 3, 4
    crf = PosCRF(n)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    em = torch.randn(1, t, n)
    mask = torch.ones(1, t)

    import itertools
    scores = []
    for path in itertools.product(range(n), repeat=t):
        s = crf.start[path[0]] + em[0, 0, path[0]]
        for i in range(1, t):
            s = s + crf.trans[path[i - 1], path[i]] + em[0, i, path[i]]
        scores.append(s + crf.end[path[-1]])
    brute = torch.logsumexp(torch.stack(scores), dim=0)
    assert torch.allclose(crf._log_partition(em, mask)[0], brute, atol=1e-4)


def test_viterbi_finds_the_true_best_path():
    torch.manual_seed(1)
    n, t = 3, 4
    crf = PosCRF(n)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    em = torch.randn(1, t, n)
    mask = torch.ones(1, t)

    import itertools
    best, best_path = None, None
    for path in itertools.product(range(n), repeat=t):
        s = crf.start[path[0]] + em[0, 0, path[0]]
        for i in range(1, t):
            s = s + crf.trans[path[i - 1], path[i]] + em[0, i, path[i]]
        s = s + crf.end[path[-1]]
        if best is None or s > best:
            best, best_path = s, path
    got = crf.viterbi(em, mask)[0].tolist()
    assert got == list(best_path)


def test_nll_is_non_negative_and_finite():
    """-log p(gold) cannot be negative: the gold path's score never exceeds
    the partition function."""
    torch.manual_seed(2)
    crf = PosCRF(N_POS)
    with torch.no_grad():
        crf.trans.normal_()
    em = torch.randn(4, T, N_POS)
    tags = torch.randint(0, N_POS, (4, T))
    mask = torch.ones(4, T)
    nll = crf.nll(em, tags, mask)
    assert torch.isfinite(nll) and nll.item() >= -1e-5


def test_nll_non_negative_with_SCATTERED_mask():
    """Regression: the masked-LM objective blanks ~15% of tokens and data.py
    sets their POS target to IGNORE, so the CRF's mask has holes in the MIDDLE
    and is NOT a contiguous prefix. An earlier version used mask.sum()-1 (the
    COUNT of valid positions) as the index of the last one, so the gold path
    collected an `end` transition belonging to some other position. The model
    could inflate that freely, driving the objective negative. Every mask in
    the original tests was contiguous, so none of them caught it.
    """
    crf = PosCRF(3)
    with torch.no_grad():
        crf.trans.zero_()
        crf.start.zero_()
        crf.end.copy_(torch.tensor([0., 0., 40.]))   # tag 2's end is hugely favoured
    # valid positions 0,1,3 -> count 3, so the buggy "last = count-1" points at
    # index 2, which is MASKED and happens to carry the favoured tag 2.
    mask = torch.tensor([[1., 1., 0., 1.]])
    tags = torch.tensor([[0, 0, 2, 0]])
    # emissions make tag 0 overwhelmingly likely, so no real path ends on tag 2
    em = torch.full((1, 4, 3), -30.0)
    em[:, :, 0] = 0.0
    nll = float(crf.nll(em, tags, mask))
    assert nll >= -1e-4, (
        f"NLL is {nll:.3f}: the gold path scored above the partition function. "
        "The mask has a hole, so the chain logic must compact it first.")


def test_end_transition_uses_the_last_VALID_index():
    """With holes in the mask, the end transition must attach to the last
    valid position, not to position (count-1)."""
    from hydra.model import _last_valid
    mask = torch.tensor([[1., 1., 0., 1., 0., 0.],     # valid 0,1,3 -> last = 3
                         [1., 0., 0., 0., 0., 0.],     # valid 0     -> last = 0
                         [1., 1., 1., 1., 1., 1.]])    # all valid   -> last = 5
    assert _last_valid(mask).tolist() == [3, 0, 5]
    # the buggy formula would have said count-1 = [2, 0, 5]
    assert (mask.sum(1).long() - 1).tolist() == [2, 0, 5]


def test_padding_is_ignored():
    """Trailing padded positions must not change the likelihood — otherwise
    short chunks would be scored differently from long ones."""
    torch.manual_seed(3)
    crf = PosCRF(N_POS)
    with torch.no_grad():
        crf.trans.normal_()
    em = torch.randn(1, 6, N_POS)
    tags = torch.randint(0, N_POS, (1, 6))
    full_mask = torch.tensor([[1.0, 1, 1, 0, 0, 0]])
    short = crf.nll(em[:, :3], tags[:, :3], torch.ones(1, 3))
    padded = crf.nll(em, tags, full_mask)
    # per-token normalisation divides by the same count in both cases
    assert torch.allclose(short * 3, padded * 3, atol=1e-4)


# ------------------------------------------------------- model integration

def test_zero_init_leaves_transitions_inert():
    """Transitions start at zero so a warm start is a no-op: with flat
    transitions the Viterbi path is just the per-position argmax."""
    model = build(model_cfg_default(), pos_crf=True).eval()
    chars = random_chars()
    with torch.no_grad():
        out = model(chars)
    em = out.pos_logits[:, :, 0, :].clone()
    em[..., NULL] = torch.finfo(em.dtype).min
    assert torch.equal(out.pos_path, em.argmax(dim=-1))


def test_trained_transitions_change_the_path():
    """Once transitions carry weight the sequence decision must diverge from
    the per-token argmax, or the CRF is decorative."""
    model = build(model_cfg_default(), pos_crf=True).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.pos_crf.trans, std=5.0)
    chars = random_chars()
    with torch.no_grad():
        out = model(chars)
    em = out.pos_logits[:, :, 0, :].clone()
    em[..., NULL] = torch.finfo(em.dtype).min
    assert not torch.equal(out.pos_path, em.argmax(dim=-1))


def test_viterbi_never_emits_null_for_slot0():
    """Decoding forces slot 0 non-NULL; the CRF must respect that invariant."""
    model = build(model_cfg_default(), pos_crf=True).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.pos_crf.trans, std=3.0)
        out = model(random_chars())
    assert (out.pos_path != NULL).all()


def test_viterbi_does_not_mutate_pos_logits():
    """Regression: masking NULL for the Viterbi emissions must work on a COPY.
    `.detach().float()` is a no-op view when the dtype already matches, so an
    earlier version wrote -inf straight back into the returned pos_logits."""
    model = build(model_cfg_default(), pos_crf=True).eval()
    chars = random_chars()
    with torch.no_grad():
        out = model(chars)
    null_logits = out.pos_logits[:, :, 0, NULL]
    assert torch.isfinite(null_logits).all(), "slot-0 NULL logits were clobbered"
    # and the whole tensor still matches a CRF-free forward on the same weights
    ref = build(model_cfg_default(), pos_crf=False).eval()
    ref.load_state_dict({k: v for k, v in model.state_dict().items()
                         if not k.startswith("pos_crf.")}, strict=False)
    with torch.no_grad():
        assert torch.equal(ref(chars).pos_logits, out.pos_logits)


def test_no_path_when_teacher_forced():
    """Training conditions on gold tags, so Viterbi is skipped."""
    model = build(model_cfg_default(), pos_crf=True)
    K = model.K
    tags = (torch.full((2, T, K), 3), torch.full((2, T, K), 5))
    out = model(random_chars(), tag_teacher=tags)
    assert out.pos_path is None


def test_off_by_default():
    model = build(model_cfg_default(), pos_crf=False).eval()
    with torch.no_grad():
        out = model(random_chars())
    assert model.pos_crf is None and out.pos_path is None


def test_loss_uses_crf_and_stays_finite():
    """compute_loss must accept the CRF, replace slot-0 CE with it, and
    survive a batch whose multi-item slots are all IGNORE."""
    model = build(model_cfg_default(), pos_crf=True)
    K = model.K
    chars = random_chars()
    pos = torch.full((2, T, K), IGNORE)
    pos[:, :, 0] = torch.randint(1, N_POS, (2, T))      # slot 0 always real
    batch = {
        "pos": pos,
        "morph": torch.full((2, T, K), IGNORE),
        "lemma": torch.full((2, T, K, L), IGNORE),
        "lemtype": torch.full((2, T, K), IGNORE),
        "joint": torch.full((2, T, K), IGNORE),
        "mlm": torch.full((2, T), IGNORE),
    }
    out = model(chars, tag_teacher=(batch["pos"], batch["morph"]))
    loss, parts = compute_loss(out, batch, LossConfig(), N_POS, crf=model.pos_crf)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.pos_crf.trans.grad is not None
    assert torch.isfinite(model.pos_crf.trans.grad).all()


def model_cfg_default():
    from hydra.config import ModelConfig
    return ModelConfig(n_slots=4)


@pytest.mark.parametrize("cond", ["off", "morph+lemma"])
def test_composes_with_tag_conditioning(cond):
    """The CRF and the tag-condition cascade must compose: with both on, the
    cascade should condition on the Viterbi path rather than the argmax."""
    model = build(model_cfg_default(), pos_crf=True, tag_condition=cond).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.pos_crf.trans, std=3.0)
        out = model(random_chars())
    assert out.pos_path is not None
    assert not torch.isnan(out.lemma_logits).any()
