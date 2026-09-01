import pytest

from hydra.data import parse_tsv_file, split_lemma_items


def test_split_lemma_items_rem_notation():
    # single item whose lemma contains '+' (discontinuous unit reference)
    assert split_lemma_items("hièr/+inne", 1) == ["hièr/+inne"]
    assert split_lemma_items("inne/hièr+", 1) == ["inne/hièr+"]
    # two items, each with an internal reference
    assert split_lemma_items("dâr/+zuo+zuo/dâr+", 2) == ["dâr/+zuo", "zuo/dâr+"]
    # trailing '+' of a reference re-attaches, plain item follows
    assert split_lemma_items("hèben/ûf>++ër", 2) == ["hèben/ûf>+", "ër"]
    assert split_lemma_items("ne+schînen/ane+", 2) == ["ne", "schînen/ane+"]
    # three items with mixed references
    assert split_lemma_items("wâr/+umbe+umbe/wâr++ër", 3) == \
        ["wâr/+umbe", "umbe/wâr+", "ër"]
    # unsplittable but single item expected -> whole string
    assert split_lemma_items("haben/umbe+/ane+", 1) == ["haben/umbe+/ane+"]
    # plain multi-item unaffected
    assert split_lemma_items("in+hant", 2) == ["in", "hant"]


def test_basic_parsing(corpus_dir):
    tokens, skipped = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=4)
    assert skipped == 2  # mismatched + toolong
    assert len(tokens) == 10  # '@' comment dropped, skipped tokens kept as context
    assert tokens[0].surface == "Ad"
    assert tokens[0].lemmas == ["ad"] and tokens[0].pos == ["APPR"]
    assert tokens[0].morph == ["--"]


def test_plus_alignment(corpus_dir):
    tokens, _ = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=4)
    tok = tokens[2]
    assert tok.surface == "inhandon"
    assert tok.lemmas == ["in", "hant"]
    assert tok.pos == ["APPR", "NA"]
    assert tok.morph == ["c.D", "Dat.Pl"]
    assert tok.n_items == 2
    three = tokens[5]
    assert three.n_items == 3 and three.lemmas == ["ze", "wer", "ne"]


def test_mismatch_skip_keeps_context(corpus_dir):
    tokens, _ = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=4)
    bad = tokens[8]
    assert bad.surface == "mismatched"
    assert bad.lemmas is None and bad.pos is None and bad.morph is None
    assert bad.n_items == 0


def test_too_many_items_skipped(corpus_dir):
    tokens, _ = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=4)
    assert tokens[9].surface == "toolong" and tokens[9].lemmas is None
    # with enough slots the same line parses fine
    tokens16, skipped16 = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=16)
    assert tokens16[9].n_items == 9 and skipped16 == 1


def test_mismatch_error_raises(corpus_dir):
    with pytest.raises(ValueError):
        parse_tsv_file(corpus_dir / "doc1.txt", on_mismatch="error", n_slots=4)


def test_surface_only_lines_are_clean_context(tmp_path):
    f = tmp_path / "raw.txt"
    f.write_text("swer\nan\nrehte\n", encoding="utf-8")
    tokens, skipped = parse_tsv_file(f, n_slots=4)
    assert skipped == 0  # raw corpus lines are not "malformed"
    assert len(tokens) == 3 and all(t.lemmas is None for t in tokens)
    assert tokens[0].surface == "swer"


def test_empty_morph_item_is_context_only(tmp_path):
    # '--' is the corpus's "no morphology" value; an EMPTY item is malformed
    f = tmp_path / "d.txt"
    f.write_text("a\tb\tNA+NA\t+Dat.Pl\nc\td\tNA\t\n", encoding="utf-8")
    tokens, skipped = parse_tsv_file(f, n_slots=4)
    assert skipped == 2
    assert all(t.lemmas is None for t in tokens)
