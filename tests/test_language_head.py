"""Language-ID cascade (model.language_head).

The variety a token's document is in is predicted FIRST and then fed forward
into the POS, morph and lemma heads -- the same predict-then-condition
discipline as tag_condition, one level up. It is active during masked-LM
pretraining as well as fine-tuning, so the encoder arrives at fine-tuning
already able to tell the varieties apart.
"""
import dataclasses

import torch

from hydra.config import LossConfig
from hydra.data import IGNORE, Token
from hydra.losses import compute_loss
from hydra.model import HydraModel
from hydra.vocab import PAD, Vocabs

N_CHARS, N_POS, N_MORPH, N_LANGS = 30, 10, 12, 5
T, H, W, L = 8, 4, 12, 16


def build(model_cfg, n_word_types=0, **kw):
    cfg = dataclasses.replace(model_cfg, **kw)
    torch.manual_seed(0)
    return HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H,
                      n_word_types=n_word_types, n_langs=N_LANGS)


def chars(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(3, N_CHARS, (B, T + 2 * H, W), generator=g)
    c[:, :, 6:] = PAD
    return c


def test_off_by_default(model_cfg):
    m = build(model_cfg)
    assert m.lang_head is None
    with torch.no_grad():
        assert m(chars()).lang_logits is None


def test_predicts_a_language_per_token(model_cfg):
    m = build(model_cfg, language_head=True)
    with torch.no_grad():
        out = m(chars())
    assert out.lang_logits.shape == (2, T, N_LANGS)


def test_projection_is_zero_init_so_warm_start_is_a_noop(model_cfg):
    """Adding the cascade to a trained checkpoint must change nothing at step
    0, or the language head and whatever else changed cannot be told apart."""
    plain = build(model_cfg, language_head=False)
    lang = build(model_cfg, language_head=True)
    lang.load_state_dict(plain.state_dict(), strict=False)
    plain.eval()
    lang.eval()
    c = chars()
    with torch.no_grad():
        a, b = plain(c), lang(c)
    for x, y in ((a.pos_logits, b.pos_logits), (a.morph_logits, b.morph_logits),
                 (a.lemma_logits, b.lemma_logits)):
        assert torch.equal(x, y)


def test_language_conditions_the_other_heads(model_cfg):
    """The point of the cascade: once the projection is trained, a different
    language must produce different POS/morph/lemma predictions."""
    m = build(model_cfg, language_head=True)
    with torch.no_grad():
        torch.nn.init.normal_(m.lang_to_tag.weight, std=0.5)
        torch.nn.init.normal_(m.lang_to_tag.bias, std=0.5)
    m.eval()
    c = chars()
    outs = []
    for lang_id in (0, 3):
        t = torch.full((2, T), lang_id)
        with torch.no_grad():
            outs.append(m(c, lang_teacher=t))
    assert not torch.allclose(outs[0].pos_logits, outs[1].pos_logits)
    assert not torch.allclose(outs[0].lemma_logits, outs[1].lemma_logits)


def test_head_is_alive_during_mlm_pretraining(model_cfg):
    """The user's requirement: the language head trains in BOTH phases. In
    pretrain_mlm the forward returns before slot decoding, so the head has to
    sit above that return or it would never be trained at all."""
    m = build(model_cfg, n_word_types=7, language_head=True,
              masked_lm=True, pretrain_mlm=True)
    with torch.no_grad():
        out = m(chars())
    assert out.pos_logits is None, "pretraining should skip the tagging heads"
    assert out.lang_logits is not None, "but NOT the language head"

    batch = {"mlm": torch.randint(0, 7, (2, T)),
             "lang": torch.randint(0, N_LANGS, (2, T))}
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts["loss_lang"] > 0.0, "language loss must be applied in pretraining"


def test_language_label_reaches_the_vocabulary():
    toks = [Token("a", ["a"], ["NA"], ["--"], None, lang)
            for lang in ("ReM", "ReF", "ReN", "LeA")]
    v = Vocabs.build(toks)
    assert [x for x in v.langs.itos if not x.startswith("<")] == ["LeA", "ReF", "ReM", "ReN"]


def test_ignored_language_does_not_contribute_loss(model_cfg):
    m = build(model_cfg, language_head=True)
    with torch.no_grad():
        out = m(chars())
    batch = {"pos": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "morph": torch.full((2, T, model_cfg.n_slots), IGNORE),
             "lemma": torch.full((2, T, model_cfg.n_slots, L), IGNORE),
             "lang": torch.full((2, T), IGNORE)}
    _, parts = compute_loss(out, batch, LossConfig(), N_POS)
    assert parts.get("loss_lang", 0.0) == 0.0
