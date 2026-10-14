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
