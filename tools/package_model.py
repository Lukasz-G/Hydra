"""Bundle a run directory into a release asset that `hydra-tag` can fetch.

    python tools/package_model.py runs/stage2 de-4corpus

writes dist/hydra-de-4corpus.tar.gz holding model_only.pt, vocab.json and a
model card. hydra/hub.py expects those three names at the archive root; the
name on the command line must be a key of hub.MODELS.

Upload it as an asset of a GitHub release and `--model de-4corpus` works on
any machine:

    gh release create v2.0.0 dist/hydra-de-4corpus.tar.gz --generate-notes
    gh release upload v2.0.0 dist/hydra-mhg.tar.gz
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hydra.hub import MODELS  # noqa: E402

CARD = """# Hydra model: {name}

{summary}

| | |
|---|---|
| run | `{run}` |
| split protocol | {protocol} |
| parameters | {params} |
| training corpora | {corpora} |
| checkpoint | {ckpt_mb:.0f} MB |

Use it with:

    pip install "hydra-tagger @ git+https://github.com/Lukasz-G/Hydra"
    hydra-tag --model {name} --input texts/ --output tagged/

The first call downloads this archive and caches it; later calls read the
cache. Output is 4-column TSV: surface, lemma, part of speech, morphology,
with the items of a multi-item token joined by `+`.

Licence: Apache-2.0, as the code. The corpora the model was trained on are
not redistributed and carry their own terms; see NOTICE in the repository.
"""


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run, name = Path(sys.argv[1]), sys.argv[2]
    if name not in MODELS:
        raise SystemExit(f"{name!r} is not in hydra.hub.MODELS "
                         f"({', '.join(MODELS)}); add it there first")

    ckpt, vocab = run / "model_only.pt", run / "vocab.json"
    for f in (ckpt, vocab):
        if not f.exists():
            raise SystemExit(f"{f} is missing")

    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    import torch
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    params = sum(v.numel() for v in payload["model"].values())

    card = run / "MODEL_CARD.md"
    card.write_text(CARD.format(
        name=name,
        summary=MODELS[name]["summary"].capitalize() + ".",
        run=run.name,
        protocol=cfg.get("data", {}).get("split_mode", "unknown"),
        params=f"{params / 1e6:.1f}M",
        corpora=cfg.get("data", {}).get("corpus_dir", "see config.json"),
        ckpt_mb=ckpt.stat().st_size / 1e6,
    ), encoding="utf-8")

    out = Path("dist") / MODELS[name]["asset"]
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for f in (ckpt, vocab, card):
            tar.add(f, arcname=f.name)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB) "
          f"from {run} ({params / 1e6:.1f}M parameters)")


if __name__ == "__main__":
    main()
