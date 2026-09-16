"""Combined-tag head (model.combo_head).

The token's whole '+'-joined POS sequence predicted as ONE class, then fed
forward into the slot heads -- the same predict-then-condition discipline as
the language and tag cascades.

Why it is built the way it is. Three matched seed pairs (comb_s* against
s_joint_s*, 84,102 token-aligned predictions each) put the combined-tag
representation +0.64pp ahead on OVERALL POS at sd 0.04, three times the
0.21pp noise floor -- and 81% of the tokens it gains are SINGLE-item. So the
head is per-token rather than per-slot, and it is supervised on every tagged
token rather than only on '+' ones. A version restricted to multi-item tokens
would be trained out of the part of the effect that actually pays.
"""
import dataclasses

import torch

from hydra.config import LossConfig
from hydra.data import IGNORE, Token, encode_document
from hydra.losses import compute_loss
from hydra.model import HydraModel
from hydra.vocab import PAD, LabelVocab, Vocabs

N_CHARS, N_POS, N_MORPH, N_COMBOS = 30, 10, 12, 7
T, H, W, L = 8, 4, 12, 16


def build(model_cfg, **kw):
    cfg = dataclasses.replace(model_cfg, **kw)
    torch.manual_seed(0)
    return HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H, n_combos=N_COMBOS)


def chars(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(3, N_CHARS, (B, T + 2 * H, W), generator=g)
    c[:, :, 6:] = PAD
    return c


def test_off_by_default(model_cfg):
    m = build(model_cfg)
    assert m.combo_head is None
    with torch.no_grad():
        assert m(chars()).combo_logits is None


def test_predicts_one_combined_tag_per_token(model_cfg):
    """Per TOKEN, not per slot: the combined tag is a property of the whole
    token, which is the point of the representation."""
    m = build(model_cfg, combo_head=True)
    with torch.no_grad():
        out = m(chars())
    assert out.combo_logits.shape == (2, T, N_COMBOS)


def test_projection_is_zero_init_so_warm_start_is_a_noop(model_cfg):
    plain = build(model_cfg, combo_head=False)
    combo = build(model_cfg, combo_head=True)
    combo.load_state_dict(plain.state_dict(), strict=False)
    plain.eval()
    combo.eval()
    c = chars()
    with torch.no_grad():
        a, b = plain(c), combo(c)
    for x, y in ((a.pos_logits, b.pos_logits), (a.morph_logits, b.morph_logits),
                 (a.lemma_logits, b.lemma_logits)):
        assert torch.equal(x, y)


def test_combined_tag_conditions_the_slot_heads(model_cfg):
    m = build(model_cfg, combo_head=True)
    with torch.no_grad():
        torch.nn.init.normal_(m.combo_to_tag.weight, std=0.5)
        torch.nn.init.normal_(m.combo_to_tag.bias, std=0.5)
    m.eval()
    c = chars()
    outs = []
    for combo_id in (0, 4):
        t = torch.full((2, T), combo_id)
        with torch.no_grad():
            outs.append(m(c, combo_teacher=t))
    assert not torch.allclose(outs[0].pos_logits, outs[1].pos_logits)
    assert not torch.allclose(outs[0].lemma_logits, outs[1].lemma_logits)


def test_loss_is_applied(model_cfg):
    m = build(model_cfg, combo_head=True)
    with torch.no_grad():
        out = m(chars())
    batch = {"pos": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "morph": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "lemma": torch.full((2, T, model_cfg.n_slots, L), IGNORE),
             "combo": torch.randint(0, N_COMBOS, (2, T))}
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts["loss_combo"] > 0.0


def test_ignored_combined_tag_contributes_nothing(model_cfg):
    m = build(model_cfg, combo_head=True)
    with torch.no_grad():
        out = m(chars())
    batch = {"pos": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "morph": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "lemma": torch.full((2, T, model_cfg.n_slots, L), IGNORE),
             "combo": torch.full((2, T), IGNORE)}
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts.get("loss_combo", 0.0) == 0.0


def _vocabs():
    toks = [Token("inhandon", ["in", "hant"], ["APPR", "NA"], ["--", "Dat.Pl"]),
            Token("der", ["der"], ["DDART"], ["Nom.Sg.Masc"]),
            Token("man", ["man"], ["NA"], ["Nom.Sg"])]
    return toks, Vocabs.build(toks)


def test_whole_sequence_is_one_label():
    toks, v = _vocabs()
    real = [x for x in v.combo_types.itos if not x.startswith("<")]
    assert "APPR+NA" in real, real
    assert "DDART" in real and "NA" in real


def test_single_item_tokens_are_supervised_too():
    """The design point. 81% of this representation's measured advantage is on
    single-item tokens, so a head trained only where the gold POS contains '+'
    would miss almost all of it."""
    toks, v = _vocabs()
    doc = encode_document(toks, v, max_word_len=W, max_lemma_len=L, n_slots=4)
    assert doc.combo[0] == v.combo_types.encode("APPR+NA")
    assert doc.combo[1] == v.combo_types.encode("DDART")
    assert doc.combo[2] == v.combo_types.encode("NA")
    assert (doc.combo >= 0).all(), "no tagged token may be left unsupervised"


def test_context_only_token_is_ignored():
    toks, v = _vocabs()
    toks = toks + [Token("xyz", None, None, None)]
    doc = encode_document(toks, v, max_word_len=W, max_lemma_len=L, n_slots=4)
    assert doc.combo[-1] == IGNORE


def test_vocabulary_round_trips(tmp_path):
    _, v = _vocabs()
    p = tmp_path / "vocab.json"
    v.save(p)
    back = Vocabs.load(p)
    assert back.combo_types.itos == v.combo_types.itos
    assert back.combo_counts == v.combo_counts


def test_each_cascade_uses_its_own_confidence_gate(model_cfg):
    """Regression: every cascade shared tag_cond_min_prob, so infer.lang_min_prob
    (and any later knob) silently did nothing. A gate that reads as 'tried, no
    effect' when it was never wired is worse than no gate."""
    m = build(model_cfg, combo_head=True)
    m.eval()
    logits = torch.zeros(2, T, N_COMBOS)
    logits[..., 1] = 10.0                       # confident, p ~= 1
    m.tag_cond_min_prob = 0.0
    m.combo_min_prob = 0.99
    # a tight gate must still accept a near-certain prediction (p=0.9997)...
    assert (m._cond_ids(logits, None, N_COMBOS, m.combo_min_prob) == 1).all()
    # ...and reject a flat one, routing it to the learned "unsure" row
    flat = torch.zeros(2, T, N_COMBOS)
    assert (m._cond_ids(flat, None, N_COMBOS, m.combo_min_prob) == N_COMBOS).all()
    # while the shared knob, left at 0, does not silently override it
    assert (m._cond_ids(flat, None, N_COMBOS) == 0).all()


def test_stale_vocabulary_is_refused_not_ignored(model_cfg):
    """A vocab written before this head existed carries only the specials.
    Building the head on it would cost parameters and do nothing at all, so
    the model refuses rather than training a decoration."""
    import pytest
    cfg = dataclasses.replace(model_cfg, combo_head=True)
    with pytest.raises(ValueError, match="no combined tags"):
        HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H, n_combos=0)


def test_has_combos_distinguishes_them():
    _, v = _vocabs()
    assert v.has_combos
    empty = Vocabs.build([Token("x", None, None, None)])
    assert not empty.has_combos
