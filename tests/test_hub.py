"""The published-model resolver.

`--model` takes either a path or one of the names in hub.MODELS, and the
first of those must keep working exactly as before, so most of what is
checked here is that a path is left alone.
"""
import tarfile

import pytest

from hydra import hub


def test_a_path_is_passed_through_untouched(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not really a checkpoint")
    assert hub.resolve(str(ckpt)) == str(ckpt)
    # a path that does not exist is still not a model name, and resolving it
    # must not reach the network: tag.py reports the missing file instead
    assert hub.resolve("runs/nope/best.pt") == "runs/nope/best.pt"


def test_cache_dir_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CACHE", str(tmp_path / "somewhere"))
    assert hub.cache_dir() == tmp_path / "somewhere"


def test_every_published_model_has_an_asset_and_a_summary():
    assert hub.MODELS, "the table may not be empty: --help prints it"
    for name, meta in hub.MODELS.items():
        assert meta["asset"].endswith(".tar.gz")
        assert meta["summary"]
        assert name in hub.describe()
        assert hub.asset_url(name).endswith(meta["asset"])


def test_a_named_model_is_unpacked_into_the_cache(tmp_path, monkeypatch):
    name = next(iter(hub.MODELS))
    payload = tmp_path / "src"
    payload.mkdir()
    (payload / "model_only.pt").write_bytes(b"weights")
    (payload / "vocab.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / hub.MODELS[name]["asset"]
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(payload.iterdir()):
            tar.add(f, arcname=f.name)

    monkeypatch.setenv("HYDRA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(hub, "_download",
                        lambda url, dest: dest.write_bytes(archive.read_bytes()))

    resolved = hub.resolve(name)
    assert resolved.endswith("model_only.pt")
    # vocab.json has to land beside the checkpoint: load_model_for_inference
    # looks for it there and nowhere else
    assert (tmp_path / "cache" / name / "vocab.json").exists()

    # second call is served from the cache, so no download is attempted
    monkeypatch.setattr(hub, "_download", lambda url, dest: pytest.fail(
        "a cached model was downloaded again"))
    assert hub.resolve(name) == resolved
