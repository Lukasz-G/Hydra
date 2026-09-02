"""Label smoothing, EMA shadow, and the spelling-noise sampler."""
import json

import torch

from hydra.config import LossConfig
from hydra.noise import SpellingNoiser


def test_spelling_noiser_deterministic_and_case_safe(tmp_path):
    rules = {"rules": {"a": {"a": 0.1, "e": 0.9}, "s": {"s": 0.1, "ſ": 0.9}}}
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(rules), encoding="utf-8")
    n1 = SpellingNoiser(p, seed=7)
    n2 = SpellingNoiser(p, seed=7)
    words = ["hasan", "Sagen", "mas"]
    out1 = [n1.noise(w) for w in words]
    out2 = [n2.noise(w) for w in words]
    assert out1 == out2  # same seed, same stream
    # strength 0 keeps every token verbatim (identity mass only)
    n0 = SpellingNoiser(p, strength=0.0, seed=7)
    assert [n0.noise(w) for w in words] == words
    # uppercase S untouched; lowercase a/s rewritten with high probability
    assert out1[1][0] == "S"
    assert any(o != w for o, w in zip(out1, words))


def test_label_smoothing_changes_loss():
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets = torch.tensor([0, 1, 2, 3, 4, -100])
    plain = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
    smooth = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100,
                                               label_smoothing=0.1)
    assert plain != smooth
    assert LossConfig(label_smoothing=0.1).label_smoothing == 0.1


def test_ema_shadow_tracks_weights():
    from hydra.train import EMA
    model = torch.nn.Linear(4, 3)
    ema = EMA(model, decay=0.5)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(ema.shadow["weight"], before["weight"])
    # shadow is halfway between old and new at decay 0.5
    expected = before["weight"] * 0.5 + model.weight.float() * 0.5
    assert torch.allclose(ema.shadow["weight"], expected)
    raw = ema.apply_to(model)
    assert torch.allclose(model.weight.float(), ema.shadow["weight"])
    model.load_state_dict(raw)


def test_norm_lookup_folds_mlm_targets(tmp_path):
    from hydra.data import Token, encode_document, parse_tsv_file
    from hydra.vocab import Vocabs

    # 2-column unannotated line parses as surface + corpus-carried norm
    f = tmp_path / "noised.txt"
    f.write_text("vn̄\tunde\ndaz\n", encoding="utf-8")
    toks, skipped = parse_tsv_file(f)
    assert skipped == 0
    assert toks[0].surface == "vn̄" and toks[0].norm == "unde"
    assert toks[1].norm is None

    # spelling variants fold onto one word-type class via the lookup
    lookup = {"vnde": "unde", "vn̄": "unde"}
    train = [Token("unde", ["unde"], ["KON"], ["--"]),
             Token("vnde", ["unde"], ["KON"], ["--"]),
             Token("vn̄", ["unde"], ["KON"], ["--"])]
    v = Vocabs.build(train, word_type_min_freq=1, norm_of=lookup)
    assert "unde" in v.word_types.stoi
    assert "vnde" not in v.word_types.stoi  # folded away
    ids = {v.word_types.encode(lookup.get(t.surface, t.surface)) for t in train}
    assert len(ids) == 1

    # encode_document routes all three spellings to the same wtype id
    doc = encode_document(train, v, max_word_len=8, max_lemma_len=8, n_slots=2,
                          norm=lookup)
    assert doc.wtype[0] == doc.wtype[1] == doc.wtype[2]
    # and the corpus-carried norm wins over the lookup
    doc2 = encode_document([toks[0]], v, max_word_len=8, max_lemma_len=8,
                           n_slots=2, norm={})
    assert doc2.wtype[0] == doc.wtype[0]
