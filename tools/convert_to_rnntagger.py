"""Derive RNNTagger training data from a Pie-format split directory.

Usage:
  python tools/convert_to_rnntagger.py PIE_DIR OUT_DIR [--tag=posmorph|pos]

PIE_DIR is a directory produced by convert_to_pie.py (train/dev/test.tsv,
columns token/lemma/pos/morph, blank lines between pseudo-sentences), so the
RNNTagger baseline trains on bit-identical data. Two products:

  OUT_DIR/tagger/{train,dev,test}.tsv   word<TAB>tag for PyRNN/rnn-train.py;
      tag is the '+'-joined POS string, with '|' + the '+'-joined morph
      appended under --tag=posmorph (the default, mirroring our joint metric)
  OUT_DIR/lemmatizer/{train,dev}.{src,tgt}   type-level parallel data for
      PyNMT/nmt-train.py in reformat.pl's encoding: characters
      space-separated, word and tag joined by ' ## '; one line per unique
      (word, tag) type from that split, target = lemma characters. A type
      with several gold lemmas keeps the most frequent (counted and printed —
      the type-level ceiling of this approach).

Evaluation later runs rnn-annotate.py + nmt-translate.py + lemma-lookup.pl on
test and scores end to end with OUR conventions (predicted tags, not gold).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_pie(path: Path):
    """Yield sentences as lists of (token, lemma, pos, morph)."""
    sent = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                    sent = []
                continue
            token, lemma, pos, morph = line.split("\t")
            sent.append((token, lemma, pos, morph))
    if sent:
        yield sent


def make_tag(pos: str, morph: str, mode: str) -> str:
    return f"{pos}|{morph}" if mode == "posmorph" else pos


def spell(s: str) -> str:
    """reformat.pl's encoding: every character space-separated, literal
    spaces rendered as <> (never occur in our tokens, kept for fidelity)."""
    return " ".join("<>" if c == " " else c for c in s)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pie_dir, out_dir = Path(args[0]), Path(args[1])
    mode = "posmorph"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            mode = a.split("=", 1)[1]
    assert mode in ("pos", "posmorph")

    tagger_dir = out_dir / "tagger"
    lem_dir = out_dir / "lemmatizer"
    tagger_dir.mkdir(parents=True, exist_ok=True)
    lem_dir.mkdir(parents=True, exist_ok=True)

    for role in ("train", "dev", "test"):
        sents = list(read_pie(pie_dir / f"{role}.tsv"))
        with open(tagger_dir / f"{role}.tsv", "w", encoding="utf-8",
                  newline="\n") as fh:
            for sent in sents:
                for token, _, pos, morph in sent:
                    fh.write(f"{token}\t{make_tag(pos, morph, mode)}\n")
                fh.write("\n")

        if role == "test":
            continue  # the lemmatiser sees test only through the live pipeline
        types: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for sent in sents:
            for token, lemma, pos, morph in sent:
                types[(token, make_tag(pos, morph, mode))][lemma] += 1
        ambiguous = sum(1 for c in types.values() if len(c) > 1)
        with open(lem_dir / f"{role}.src", "w", encoding="utf-8", newline="\n") as fs, \
                open(lem_dir / f"{role}.tgt", "w", encoding="utf-8", newline="\n") as ft:
            for (token, tag), lemmas in sorted(types.items()):
                fs.write(f"{spell(token)} ## {spell(tag)}\n")
                ft.write(f"{spell(lemmas.most_common(1)[0][0])}\n")
        print(f"{role}: {sum(len(s) for s in sents)} tokens, "
              f"{len(types)} word-tag types, {ambiguous} ambiguous types "
              f"(most-frequent lemma kept)")


if __name__ == "__main__":
    main()
