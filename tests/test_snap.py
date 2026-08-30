import dataclasses

import torch

from hydra.model import HydraModel
from hydra.snap import LemmaSnapper


def test_snap_basics():
    snapper = LemmaSnapper({"hant", "hunt", "ross", "zièhen"})
    assert snapper.snap_item("ross") == "ross"          # already in inventory
    assert snapper.snap_item("zièhem") == "zièhen"      # unique distance-1
    assert snapper.snap_item("hont") == "hont"          # ambiguous (hant/hunt) -> keep
    assert snapper.snap_item("xyz") == "xyz"            # nothing close
    assert snapper.snap_item("") == ""


def test_snap_multi_item():
    snapper = LemmaSnapper({"in", "hant"})
    assert snapper.snap("in+hantt", 2) == "in+hant"
    # ReM '/'-notation items pass through split_lemma_items unharmed
    assert snapper.snap("hièr/+inne", 1) == "hièr/+inne"


def test_model_with_lemma_classifier(model_cfg):
    cfg = dataclasses.replace(model_cfg, lemma_classifier=True)
    model = HydraModel(cfg, n_chars=30, n_pos=10, n_morph=12, max_word_len=12,
                       max_lemma_len=16, chunk_len=8, halo=4, n_lemma_types=50)
    g = torch.Generator().manual_seed(0)
    chars = torch.randint(3, 30, (2, 16, 12), generator=g)
    out = model(chars)
    assert out.lemma_cls_logits is not None
    assert out.lemma_cls_logits.shape == (2, 8, 4, 50)
    # warm-start compatibility: a no-classifier state dict loads with the head missing
    base = HydraModel(model_cfg, 30, 10, 12, 12, 16, 8, 4)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("lemma_cls_head" in k for k in missing)
