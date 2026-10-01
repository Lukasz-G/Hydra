"""Resolve a released model name to a local checkpoint, downloading once.

`hydra-tag --model de-4corpus` should work on a machine that has never seen
this project. A checkpoint needs `vocab.json` beside it (see
tag.load_model_for_inference), so a release asset is an archive of the two,
and this module unpacks it into a cache directory and hands back the path.

A filesystem path is passed through untouched, so nothing here changes the
behaviour of `--model runs/x/best.pt`.
"""
from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

#: Published models, by the short name `--model` accepts. The URL points at
#: the GitHub release asset; `latest` follows whatever release is current, so
#: a retrained model reaches users without a code change.
REPO = "https://github.com/Lukasz-G/Hydra"
MODELS = {
    "de-4corpus": {
        "asset": "hydra-de-4corpus.tar.gz",
        "summary": "one model over Middle High German, Early New High German, "
                   "Middle Low German and Old German",
    },
    "mhg": {
        "asset": "hydra-mhg.tar.gz",
        "summary": "Middle High German only (ReM), manuscript-held-out split",
    },
}


def cache_dir() -> Path:
    """Where downloaded models live. HYDRA_CACHE overrides it."""
    env = os.environ.get("HYDRA_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    return Path(base or Path.home() / ".cache") / "hydra-tagger"


def asset_url(name: str) -> str:
    return f"{REPO}/releases/latest/download/{MODELS[name]['asset']}"


def _download(url: str, dest: Path) -> None:
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as r, dest.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    if total:
        print()


def resolve(model: str) -> str:
    """A path stays a path; a published name becomes one, downloading once."""
    if model not in MODELS:
        return model

    target = cache_dir() / model
    ckpt = target / "model_only.pt"
    if ckpt.exists() and (target / "vocab.json").exists():
        return str(ckpt)

    url = asset_url(model)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / MODELS[model]["asset"]
        try:
            _download(url, archive)
        except urllib.error.HTTPError as e:
            raise SystemExit(
                f"could not fetch the '{model}' model ({e.code} from {url}).\n"
                f"Published models are listed at {REPO}/releases. If none is "
                f"published yet, train one or point --model at a local "
                f"checkpoint."
            ) from e
        with tarfile.open(archive) as tar:
            # the archive holds model_only.pt and vocab.json at its root
            for member in tar.getmembers():
                if Path(member.name).name in ("model_only.pt", "vocab.json",
                                              "MODEL_CARD.md"):
                    member.name = Path(member.name).name
                    tar.extract(member, target)
    if not ckpt.exists():
        raise SystemExit(f"{MODELS[model]['asset']} held no model_only.pt")
    log.info("model cached in %s", target)
    return str(ckpt)


def describe() -> str:
    lines = ["published models (pass the name to --model):"]
    for name, meta in MODELS.items():
        lines.append(f"  {name:<12} {meta['summary']}")
    lines.append(f"  cache: {cache_dir()}")
    return "\n".join(lines)
