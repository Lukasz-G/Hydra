import dataclasses

from hydra.data import (HydraDataset, load_split_tokens, stratified_split_files,
                        file_sigle)
from hydra.vocab import Vocabs


def test_file_sigle():
    assert file_sigle("D:/x/M013B-N1.txt") == "M013B"
    assert file_sigle("M001-G1.txt") == "M001"


def test_stratified_split(tmp_path, corpus_dir):
    # 6 files across 2 strata (3 each) so dev/test can hold one file per stratum
    files = []
    for i, stratum in enumerate(["a", "a", "a", "b", "b", "b"]):
        f = tmp_path / f"M{i:03d}-N1.txt"
        f.write_text("tok\tlem\tNA\t--\n" * (50 + i), encoding="utf-8")
        files.append(f)
    meta = tmp_path / "meta.csv"
    meta.write_text("sigle,dialect,period\n" +
                    "\n".join(f"M{i:03d},{s},12" for i, s in
                              enumerate(["a", "a", "a", "b", "b", "b"])),
                    encoding="utf-8")
    splits = stratified_split_files(files, meta, dev_fraction=0.34, test_fraction=0.34, seed=1)
    all_assigned = splits["train"] + splits["dev"] + splits["test"]
    assert sorted(all_assigned) == sorted(str(f) for f in files)
    assert len(set(all_assigned)) == 6  # disjoint
    # each stratum keeps at least one file in train
    for s, sig_prefix in (("a", ["M000", "M001", "M002"]), ("b", ["M003", "M004", "M005"])):
        assert any(any(p in f for p in sig_prefix) for f in splits["train"])
    # deterministic
    assert splits == stratified_split_files(files, meta, 0.34, 0.34, 1)


def test_singleton_stratum_stays_in_train(tmp_path):
    f = tmp_path / "M111-N1.txt"
    f.write_text("tok\tlem\tNA\t--\n" * 20, encoding="utf-8")
    meta = tmp_path / "meta.csv"
    meta.write_text("sigle,dialect,period\nM111,x,12\n", encoding="utf-8")
    splits = stratified_split_files([f], meta, 0.3, 0.3, 0)
    assert splits["train"] == [str(f)] and not splits["dev"] and not splits["test"]


def test_chunk_role_split(corpus_dir, data_cfg):
    cfg = dataclasses.replace(data_cfg, chunk_len=4, halo=2,
                              dev_fraction=0.3, test_fraction=0.3)
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", 4)
    vocabs = Vocabs.build([t for d in docs for t in d])
    full = HydraDataset(docs, vocabs, cfg, 4)
    parts = {r: HydraDataset(docs, vocabs, cfg, 4, role=r) for r in ("train", "dev", "test")}
    union = sorted(c for p in parts.values() for c in p.chunks)
    assert union == sorted(full.chunks)                     # disjoint cover
    assert all(len(p.chunks) > 0 for p in parts.values())
    # deterministic per role
    again = HydraDataset(docs, vocabs, cfg, 4, role="dev")
    assert again.chunks == parts["dev"].chunks
