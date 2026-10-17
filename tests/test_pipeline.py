"""The library API: `import hydra; nlp = hydra.load(...); doc = nlp(text)`.

Loading a model needs a checkpoint, which the suite does not ship, so what is
checked here is everything around it: that the package exposes the names, that
a Doc behaves like a sequence, and above all that a fused token comes apart
into its items the way the corpora write it. That last one is the part a user
of this API will get wrong if we get it wrong.
"""
import pytest

import hydra
from hydra.pipeline import Doc, Token


def test_the_package_exposes_the_api_without_importing_torch():
    assert hydra.load and hydra.Doc and hydra.Token and hydra.Hydra
    assert "load" in dir(hydra)
    with pytest.raises(AttributeError):
        hydra.no_such_thing


def test_a_single_item_token_has_one_item():
    t = Token("mere", "mære", "NA", "Nom.Sg")
    assert t.items == [("mære", "NA", "Nom.Sg")]
    assert not t.is_multi


def test_a_fused_token_comes_apart_into_its_items():
    # zemerſten: preposition + article + adjective, one written word
    t = Token("zemerſten", "ze+dër+mèr(e)", "APPR+DDART+ADJA", "--+Dat.Sg+Sup.Dat.Sg")
    assert t.is_multi
    assert t.items == [("ze", "APPR", "--"),
                       ("dër", "DDART", "Dat.Sg"),
                       ("mèr(e)", "ADJA", "Sup.Dat.Sg")]


def test_items_survive_a_field_with_fewer_parts():
    # the morph column is '--' for the whole token in some corpora
    t = Token("bistu", "sîn+dû", "VAFIN+PPER", "--")
    assert [i[1] for i in t.items] == ["VAFIN", "PPER"]
    assert t.items[1][2] == ""


def test_doc_is_a_sequence_and_dumps_the_four_columns():
    toks = [Token("Was", "wësen", "VAFIN", "Ind.Past.Sg.3", 0),
            Token("daz", "dër", "DDART", "Neut.Nom.Sg", 1)]
    doc = Doc(toks, text="Was daz")
    assert len(doc) == 2
    assert doc[0].text == "Was"
    assert [t.pos_ for t in doc] == ["VAFIN", "DDART"]
    assert doc.lemmas == ["wësen", "dër"]
    assert doc.to_tsv().splitlines()[0].split("\t") == ["Was", "wësen", "VAFIN",
                                                        "Ind.Past.Sg.3"]
    assert doc.to_dicts()[1]["lemma"] == "dër"
    assert isinstance(doc[0:1], Doc)


def test_a_doc_built_from_tokens_reconstructs_its_text():
    doc = Doc([Token("ich", "ich", "PPER", "Nom.Sg.1"),
               Token("bin", "sîn", "VAFIN", "Ind.Pres.Sg.1")])
    assert doc.text == "ich bin"


def test_the_default_tokenizer_splits_on_whitespace_and_can_be_replaced():
    nlp = hydra.Hydra.__new__(hydra.Hydra)       # no checkpoint needed
    nlp.tokenizer = (lambda s: s.split())
    assert nlp.tokenizer("Was daz mere") == ["Was", "daz", "mere"]
    nlp.tokenizer = (lambda s: list(s.replace(" ", "")))
    assert nlp.tokenizer("ab c") == ["a", "b", "c"]


def test_a_plain_token_is_certain_by_default():
    # the defaults have to be the confident ones, or every caller that builds
    # a Token by hand would be told its own data is suspect
    t = Token("mere", "mære", "NA", "Nom.Sg")
    assert not t.is_uncertain
    assert t.pos_conf == 1.0 and t.lemma_conf == 1.0
    assert not t.lemma_generated and not t.is_unseen


def test_an_unseen_spelling_is_flagged_however_confident_the_tag():
    t = Token("ſchoeniu", "schœne", "ADJA", "Nom.Sg", pos_conf=0.99, is_unseen=True)
    assert t.is_uncertain


def test_a_generated_lemma_is_flagged():
    t = Token("x", "y", "NA", "--", lemma_generated=True)
    assert t.is_uncertain


def test_a_low_confidence_tag_is_flagged():
    assert Token("x", "y", "NA", "--", pos_conf=0.31).is_uncertain
    assert not Token("x", "y", "NA", "--", pos_conf=0.77).is_uncertain


def test_unseen_rate_is_the_share_of_unseen_tokens():
    doc = Doc([Token("a", "a", "NA", "--"),
               Token("b", "b", "NA", "--", is_unseen=True),
               Token("c", "c", "NA", "--", is_unseen=True),
               Token("d", "d", "NA", "--")])
    assert doc.unseen_rate == 0.5


def test_unseen_rate_of_an_empty_doc_is_zero_not_an_error():
    assert Doc([]).unseen_rate == 0.0


def test_uncertain_collects_the_rows_worth_checking():
    doc = Doc([Token("a", "a", "NA", "--"),
               Token("b", "b", "NA", "--", is_unseen=True),
               Token("c", "c", "NA", "--", pos_conf=0.2),
               Token("d", "d", "NA", "--", lemma_generated=True)])
    assert [t.text for t in doc.uncertain()] == ["b", "c", "d"]
    # a stricter threshold is the caller's to choose
    assert [t.text for t in doc.uncertain(pos_conf=0.1)] == ["b", "d"]


def test_to_dicts_carries_confidence_only_when_asked():
    doc = Doc([Token("a", "a", "NA", "--", pos_conf=0.5, is_unseen=True)])
    assert doc.to_dicts() == [{"text": "a", "lemma": "a", "pos": "NA", "morph": "--"}]
    d = doc.to_dicts(confidence=True)[0]
    assert d["unseen"] is True and d["pos_conf"] == 0.5
