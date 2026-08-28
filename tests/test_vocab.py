from hydra.data import parse_tsv_file
from hydra.vocab import EOW, LABEL_UNK, NULL, PAD, UNK, CharVocab, LabelVocab, Vocabs


def test_special_indices():
    cv = CharVocab.build(["ab"])
    assert cv.stoi["<PAD>"] == PAD == 0
    assert cv.stoi["<UNK>"] == UNK == 1
    assert cv.stoi["<EOW>"] == EOW == 2
    lv = LabelVocab.build(["NA", "APPR"])
    assert lv.stoi["<NULL>"] == NULL == 0
    assert lv.stoi["<UNK>"] == LABEL_UNK == 1


def test_deterministic_order():
    a = CharVocab.build(["zebra", "apfel"])
    b = CharVocab.build(["apfel", "zebra"])
    assert a.itos == b.itos
    la = LabelVocab.build(["NA", "APPR", "VVFIN"])
    lb = LabelVocab.build(["VVFIN", "NA", "APPR"])
    assert la.itos == lb.itos


def test_char_roundtrip_unicode():
    cv = CharVocab.build(["weſman", "ręhet", "zièhen"])
    for s in ("weſman", "ręhet", "zièhen"):
        assert cv.decode(cv.encode(s)) == s
    # unseen char -> UNK
    assert cv.encode("x")[0] == UNK


def test_label_unseen_is_unk():
    lv = LabelVocab.build(["NA"])
    assert lv.encode("NOPE") == LABEL_UNK
    assert lv.decode(lv.encode("NA")) == "NA"


def test_vocabs_build_save_load(corpus_dir, tmp_path):
    tokens, _ = parse_tsv_file(corpus_dir / "doc1.txt", n_slots=4)
    v = Vocabs.build(tokens)
    assert "APPR" in v.pos.stoi and "Dat.Pl" in v.morph.stoi
    assert "inhandon" in v.train_surfaces
    assert "mismatched" in v.train_surfaces  # context-only surfaces still count as seen
    path = tmp_path / "vocab.json"
    v.save(path)
    w = Vocabs.load(path)
    assert w.chars.itos == v.chars.itos
    assert w.pos.itos == v.pos.itos
    assert w.morph.itos == v.morph.itos
    assert w.train_surfaces == v.train_surfaces
