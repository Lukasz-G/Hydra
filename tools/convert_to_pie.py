"""Convert a Hydra run's exact split into Pie's tab-separated training format.

Usage:
  python tools/convert_to_pie.py RUN_DIR OUT_DIR [--sent-len=30]

RUN_DIR is a finished Hydra run directory: its config.json and split.json
define the corpus and the exact split (file, stratified, or chunk protocol).
For the chunk protocol the run's vocab.json must also be present, because the
chunk-role assignment is reproduced through HydraDataset itself (no reimplemented
logic that could drift).

Writes OUT_DIR/{train,dev,test}.tsv: columns token/lemma/pos/morph, no header,
one blank line between pseudo-sentences of --sent-len tokens; sentences never
cross a file (or, in chunk mode, a chunk) boundary. Multi-item tokens keep the
corpus's own '+'-joined convention as single combined strings — the combined-tag
treatment the Pie/Schmid papers use for fused tokens. Context-only tokens
(parse-skipped) are dropped and counted; damaged tokens keep their labels.

Pie settings to pair with the output: sep="\\t", tasks lemma/pos/morph in this
column order, header=False; blank lines already delimit sentences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra.config import config_from_dict
from hydra.data import HydraDataset, Token, load_split_tokens
from hydra.vocab import Vocabs

ROLES = ("train", "dev", "test")


def doc_rows(tokens: list[Token]):
    for t in tokens:
        if t.lemmas is None:
            yield t.surface, None
        else:
            yield t.surface, ("+".join(t.lemmas), "+".join(t.pos), "+".join(t.morph))


def chunk_rows(ds: HydraDataset, idx: int):
    for surface, gold in zip(ds.chunk_surfaces(idx), ds.chunk_gold(idx)):
        if surface:  # '' = padding past the document end
            yield surface, gold


def write_role(path: Path, blocks, sent_len: int) -> tuple[int, int]:
    written = dropped = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for block in blocks:
            in_sent = 0
            for surface, gold in block:
                if gold is None:
                    dropped += 1
                    continue
                lemma, pos, morph = gold
                fh.write(f"{surface}\t{lemma}\t{pos}\t{morph}\n")
                written += 1
                in_sent += 1
                if in_sent >= sent_len:
                    fh.write("\n")
                    in_sent = 0
            if in_sent:
                fh.write("\n")
    return written, dropped


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_dir, out_dir = Path(args[0]), Path(args[1])
    sent_len = 30
    corpus_dir = None  # remap split.json paths recorded on another machine
    for a in sys.argv[1:]:
        if a.startswith("--sent-len="):
            sent_len = int(a.split("=", 1)[1])
        elif a.startswith("--corpus-dir="):
            corpus_dir = Path(a.split("=", 1)[1])
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    splits = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    if corpus_dir is not None:
        splits = {r: [str(corpus_dir / Path(f).name) for f in fs] for r, fs in splits.items()}
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_mode = cfg.data.split_mode == "chunk" and cfg.data.train_dir is None
    stats = {}
    if chunk_mode:
        vocabs = Vocabs.load(run_dir / "vocab.json")
        docs = load_split_tokens(splits["train"], cfg.data.on_mismatch, cfg.model.n_slots)
        for role in ROLES:
            ds = HydraDataset(docs, vocabs, cfg.data, cfg.model.n_slots, role=role)
            blocks = (chunk_rows(ds, i) for i in range(len(ds)))
            stats[role] = write_role(out_dir / f"{role}.tsv", blocks, sent_len)
            del ds
    else:
        for role in ROLES:
            docs = load_split_tokens(splits[role], cfg.data.on_mismatch, cfg.model.n_slots)
            stats[role] = write_role(out_dir / f"{role}.tsv",
                                     (doc_rows(d) for d in docs), sent_len)

    for role, (written, dropped) in stats.items():
        print(f"{role}: {written} tokens written, {dropped} context-only dropped")
    (out_dir / "conversion.json").write_text(json.dumps({
        "run_dir": str(run_dir), "split_mode": cfg.data.split_mode,
        "sent_len": sent_len,
        "tokens": {r: s[0] for r, s in stats.items()},
        "dropped": {r: s[1] for r, s in stats.items()},
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
