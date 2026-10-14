"""Tag a Pie-format gold file with a trained Pie model, one row per token.

  python tools/pie_predict.py MODEL.tar GOLD.tsv OUT.tsv [--device cuda]

Why not `pie tag`: its VRT reader yields a sentence on a blank line and then
tries to split that same blank line, so a blank-line-delimited file raises
ValueError before a single token is tagged. Its line reader, meanwhile,
flushes a sentence after every input line, which with one token per line
would hand the model no context at all. Both would also re-segment the text,
and the scorer aligns predictions to gold positionally, so the segmentation
has to be the gold's own.

This feeds the model the gold file's own sentences, in order, and writes
token/lemma/pos/morph: the format tools/score_baseline.py reads as --pred=pie.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def read_sentences(path: Path):
    """The gold file's own sentence segmentation, tokens only."""
    sents, sent = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if sent:
                    sents.append(sent)
                    sent = []
                continue
            sent.append(line.split("\t")[0])
    if sent:
        sents.append(sent)
    return sents


def main() -> None:
    model_path, gold_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    device = "cuda"
    batch_size = 100
    for i, a in enumerate(sys.argv):
        if a == "--device":
            device = sys.argv[i + 1]
        if a == "--batch_size":
            batch_size = int(sys.argv[i + 1])

    from pie.tagger import Tagger

    sents = read_sentences(gold_path)
    n_tok = sum(len(s) for s in sents)
    print("sentences %d, tokens %d" % (len(sents), n_tok), flush=True)

    tagger = Tagger(device=device, batch_size=batch_size)
    tagger.add_model(model_path)
    tasks = tagger.models[-1][0].label_encoder.tasks
    print("model tasks:", list(tasks), flush=True)

    rows = []
    t0 = time.time()
    done = 0
    for start in range(0, len(sents), batch_size):
        chunk = sents[start:start + batch_size]
        tagged, order = tagger.tag(chunk, [len(s) for s in chunk])
        pos_i = order.index("pos") if "pos" in order else None
        lem_i = order.index("lemma") if "lemma" in order else None
        mor_i = order.index("morph") if "morph" in order else None
        for sent in tagged:
            for token, vals in sent:
                rows.append((token,
                             vals[lem_i] if lem_i is not None else "_",
                             vals[pos_i] if pos_i is not None else "_",
                             vals[mor_i] if mor_i is not None else "_"))
        done += len(chunk)
        if start % (batch_size * 20) == 0:
            el = time.time() - t0
            print("  %d/%d sentences, %.0f s elapsed" % (done, len(sents), el),
                  flush=True)

    if len(rows) != n_tok:
        raise SystemExit("token count changed: %d written against %d read"
                         % (len(rows), n_tok))

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")
    el = time.time() - t0
    print("wrote %s: %d rows in %.0f s (%.0f tokens/s)"
          % (out_path, len(rows), el, n_tok / max(el, 1e-9)))
    print("PIE_INFERENCE_SECONDS %.1f" % el)
    print("PIE_INFERENCE_TOKENS_PER_SEC %.0f" % (n_tok / max(el, 1e-9)))


if __name__ == "__main__":
    main()
