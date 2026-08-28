import numpy as np

from hydra.data import IGNORE, HydraDataset, collate, load_split_tokens, parse_tsv_file
from hydra.vocab import EOW, NULL, PAD, Vocabs


def make_ds(corpus_dir, data_cfg, n_slots=4):
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", n_slots)
    tokens = [t for d in docs for t in d]
    vocabs = Vocabs.build(tokens)
    return HydraDataset(docs, vocabs, data_cfg, n_slots), vocabs


def test_chunk_index_respects_files(corpus_dir, data_cfg):
    ds, _ = make_ds(corpus_dir, data_cfg)
    # doc1 has 10 tokens -> chunks (0,0),(0,8); doc2 has 5 -> (1,0)
    assert ds.chunks == [(0, 0), (0, 8), (1, 0)]


def test_item_shapes_and_halo(corpus_dir, data_cfg):
    ds, _ = make_ds(corpus_dir, data_cfg)
    item = ds[0]
    T, H, W = data_cfg.chunk_len, data_cfg.halo, data_cfg.max_word_len
    assert item["chars"].shape == (T + 2 * H, W)
    assert item["pos"].shape == (T, 4)
    assert item["lemma"].shape == (T, 4, data_cfg.max_lemma_len)
    # left halo of the first chunk is file padding -> all PAD chars
    assert (item["chars"][:H] == PAD).all()
    # real tokens have at least one non-PAD char
    assert (item["chars"][H] != PAD).any()


def test_targets_multi_item(corpus_dir, data_cfg):
    ds, vocabs = make_ds(corpus_dir, data_cfg)
    item = ds[0]
    # token index 2 in doc1 = 'inhandon' (2 items)
    pos = item["pos"][2].numpy()
    morph = item["morph"][2].numpy()
    lemma = item["lemma"][2].numpy()
    assert pos[0] == vocabs.pos.encode("APPR") and pos[1] == vocabs.pos.encode("NA")
    assert (pos[2:] == NULL).all()                       # unused slots -> NULL POS target
    assert (morph[2:] == IGNORE).all()                   # ... but morph ignored
    assert (lemma[2:] == IGNORE).all()                   # ... and lemma ignored
    # slot 0 lemma: 'in' + EOW then IGNORE
    decoded = vocabs.chars.decode([i for i in lemma[0] if i >= 0])
    assert decoded == "in"
    assert lemma[0][2] == EOW and (lemma[0][3:] == IGNORE).all()


def test_context_only_token_fully_ignored(corpus_dir, data_cfg):
    ds, _ = make_ds(corpus_dir, data_cfg)
    item = ds[1]  # doc1 tokens 8..9 = mismatched, toolong (both context-only)
    assert not item["token_mask"][:2].any()
    assert (item["pos"][0] == IGNORE).all()
    # file-end padding rows are ignored too
    assert (item["pos"][2:] == IGNORE).all()
    assert not item["token_mask"][2:].any()


def test_collate_shapes(corpus_dir, data_cfg):
    ds, _ = make_ds(corpus_dir, data_cfg)
    batch = collate([ds[i] for i in range(len(ds))])
    T, H = data_cfg.chunk_len, data_cfg.halo
    assert batch["chars"].shape == (3, T + 2 * H, data_cfg.max_word_len)
    assert batch["token_mask"].shape == (3, T)
    assert batch["token_mask"].dtype.is_floating_point is False


def test_multi_item_upsample(corpus_dir, data_cfg):
    import dataclasses
    cfg2 = dataclasses.replace(data_cfg, multi_item_upsample=3)
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", 4)
    vocabs = Vocabs.build([t for d in docs for t in d])
    ds = HydraDataset(docs, vocabs, cfg2, 4, training=True)
    # chunk (0,0) contains multi-item tokens -> duplicated twice more; others not
    assert ds.chunks.count((0, 0)) == 3
    assert ds.chunks.count((1, 0)) == 1


def test_gold_strings(corpus_dir, data_cfg):
    ds, _ = make_ds(corpus_dir, data_cfg)
    gold = ds.chunk_gold(0)
    assert gold[2] == ("in+hant", "APPR+NA", "c.D+Dat.Pl")
    surfaces = ds.chunk_surfaces(0)
    assert surfaces[2] == "inhandon"
    assert ds.chunk_gold(1)[0] is None  # mismatched token has no gold
