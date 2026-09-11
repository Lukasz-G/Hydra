"""Tag-conditioning cascade (model.tag_condition): POS -> morph -> lemma."""
import dataclasses

import pytest
import torch

from hydra.config import ModelConfig
from hydra.data import IGNORE
from hydra.model import HydraModel
from hydra.vocab import PAD

N_CHARS, N_POS, N_MORPH = 30, 10, 12
T, H, W, L = 8, 4, 12, 16


def build(model_cfg, **kw):
    cfg = dataclasses.replace(model_cfg, **kw)
    return HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H)


def random_chars(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    chars = torch.randint(3, N_CHARS, (B, T + 2 * H, W), generator=g)
    chars[:, :, 6:] = PAD
    return chars


def tags(B, K, pos_val, morph_val):
    return (torch.full((B, T, K), pos_val), torch.full((B, T, K), morph_val))


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="tag_condition"):
        ModelConfig(tag_condition="sometimes")


@pytest.mark.parametrize("mode", ["lemma", "morph+lemma"])
def test_shapes_unchanged(model_cfg, mode):
    model = build(model_cfg, tag_condition=mode)
    K = model_cfg.n_slots
    out = model(random_chars(), tag_teacher=tags(2, K, 3, 5))
    assert out.pos_logits.shape == (2, T, K, N_POS)
    assert out.morph_logits.shape == (2, T, K, N_MORPH)
    assert out.lemma_logits.shape == (2, T, K, L, N_CHARS)
    for t in (out.pos_logits, out.morph_logits, out.lemma_logits):
        assert not torch.isnan(t).any()


@pytest.mark.parametrize("mode", ["lemma", "morph+lemma"])
def test_zero_init_makes_conditioning_a_noop(model_cfg, mode):
    """The conditioning projections start at zero, so at init the tag has no
    effect whatsoever — which is what makes warm-starting a mature checkpoint
    safe. Two different gold tags must give bit-identical outputs."""
    model = build(model_cfg, tag_condition=mode).eval()
    K = model_cfg.n_slots
    chars = random_chars()
    with torch.no_grad():
        a = model(chars, tag_teacher=tags(2, K, 1, 1))
        b = model(chars, tag_teacher=tags(2, K, 7, 9))
    assert torch.equal(a.lemma_logits, b.lemma_logits)
    assert torch.equal(a.morph_logits, b.morph_logits)


@pytest.mark.parametrize("mode", ["lemma", "morph+lemma"])
def test_conditioning_flows_once_trained(model_cfg, mode):
    """After the zero-init projections pick up weight, the tag must actually
    change the prediction — otherwise the cascade is decorative."""
    model = build(model_cfg, tag_condition=mode).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.tag_to_lemma.weight, std=0.5)
        if model.tag_to_morph is not None:
            torch.nn.init.normal_(model.tag_to_morph.weight, std=0.5)
    K = model_cfg.n_slots
    chars = random_chars()
    with torch.no_grad():
        a = model(chars, tag_teacher=tags(2, K, 1, 1))
        b = model(chars, tag_teacher=tags(2, K, 7, 9))
    assert not torch.equal(a.lemma_logits, b.lemma_logits)
    if mode == "morph+lemma":
        assert not torch.equal(a.morph_logits, b.morph_logits)


def test_ignore_targets_do_not_crash(model_cfg):
    """Unused slots carry morph target IGNORE (-100), which cannot index an
    embedding; it must be routed to the learned 'unsure' slot instead."""
    model = build(model_cfg, tag_condition="morph+lemma").eval()
    K = model_cfg.n_slots
    pos = torch.full((2, T, K), IGNORE)
    morph = torch.full((2, T, K), IGNORE)
    with torch.no_grad():
        out = model(random_chars(), tag_teacher=(pos, morph))
    assert not torch.isnan(out.lemma_logits).any()


def test_unsure_index_is_in_range(model_cfg):
    """IGNORE and gated-out predictions both map to index n_pos / n_morph, so
    the embedding tables must be one row larger than the label vocabularies."""
    model = build(model_cfg, tag_condition="morph+lemma")
    assert model.pos_cond_emb.num_embeddings == N_POS + 1
    assert model.morph_cond_emb.num_embeddings == N_MORPH + 1


def test_confidence_gate_routes_everything_to_unsure(model_cfg):
    """An unreachable threshold must send every prediction to the fallback
    embedding rather than crashing or silently trusting the argmax."""
    model = build(model_cfg, tag_condition="morph+lemma").eval()
    model.tag_cond_min_prob = 1.1  # no softmax value can clear this
    with torch.no_grad():
        out = model(random_chars())
    assert not torch.isnan(out.lemma_logits).any()

    ids = model._cond_ids(torch.randn(2, T, model_cfg.n_slots, N_POS), None, N_POS)
    assert (ids == N_POS).all()


def test_off_mode_builds_no_conditioning_params(model_cfg):
    model = build(model_cfg, tag_condition="off")
    assert model.pos_cond_emb is None
    assert model.tag_to_lemma is None
    # and gold tags passed anyway are simply ignored
    K = model_cfg.n_slots
    with torch.no_grad():
        a = model(random_chars())
        b = model(random_chars(), tag_teacher=tags(2, K, 3, 3))
    assert torch.equal(a.lemma_logits, b.lemma_logits)


def test_classifier_conditioning_is_separable(model_cfg):
    """tag_condition_classifier=False must leave the classify-or-generate head
    on the unconditioned representation while the generator still gets tags."""
    K = model_cfg.n_slots
    chars = random_chars()
    for flag, expect_equal in ((False, True), (True, False)):
        cfg = dataclasses.replace(model_cfg, tag_condition="lemma",
                                  lemma_classifier=True,
                                  tag_condition_classifier=flag)
        model = HydraModel(cfg, N_CHARS, N_POS, N_MORPH, W, L, T, H,
                           n_lemma_types=25).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.tag_to_lemma.weight, std=0.5)
            a = model(chars, tag_teacher=tags(2, K, 1, 1))
            b = model(chars, tag_teacher=tags(2, K, 7, 9))
        assert torch.equal(a.lemma_cls_logits, b.lemma_cls_logits) == expect_equal


def test_overfits_end_to_end_through_the_real_decode_path(corpus_dir, data_cfg, model_cfg):
    """The cascade must survive the full train->evaluate_dataset round trip:
    gold tags teacher-forced in training, PREDICTED tags at eval. If the two
    paths disagree the model cannot reach 100%, which makes this a direct
    check on the teacher-forcing/inference seam."""
    from hydra.config import LossConfig
    from hydra.data import HydraDataset, collate, load_split_tokens
    from hydra.evaluate import evaluate_dataset
    from hydra.losses import compute_loss
    from hydra.vocab import Vocabs

    torch.manual_seed(0)
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", model_cfg.n_slots)
    vocabs = Vocabs.build([t for d in docs for t in d])
    ds = HydraDataset(docs, vocabs, data_cfg, model_cfg.n_slots, training=True)
    batch = collate([ds[i] for i in range(len(ds))])

    cfg = dataclasses.replace(model_cfg, tag_condition="morph+lemma")
    model = HydraModel(cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       data_cfg.max_word_len, data_cfg.max_lemma_len,
                       data_cfg.chunk_len, data_cfg.halo)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    metrics = {}
    for it in range(1, 801):
        opt.zero_grad(set_to_none=True)
        out = model(batch["chars"], tag_teacher=(batch["pos"], batch["morph"]))
        loss, _ = compute_loss(out, batch, LossConfig(), len(vocabs.pos))
        loss.backward()
        opt.step()
        if it % 100 == 0:
            metrics = evaluate_dataset(model, ds, vocabs, torch.device("cpu"), 4)
            if metrics.get("acc_joint", 0.0) == 1.0:
                break
    assert metrics.get("acc_joint", 0.0) == 1.0, f"failed to memorize: {metrics}"


def test_ar_decoder_receives_conditioning(model_cfg):
    """The AR path returns slot_states for external generation; those must be
    the CONDITIONED vectors, or inference silently drops the cascade."""
    model = build(model_cfg, tag_condition="lemma", lemma_decoder="ar_tcn").eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.tag_to_lemma.weight, std=0.5)
    K = model_cfg.n_slots
    chars = random_chars()
    with torch.no_grad():
        a = model(chars, tag_teacher=tags(2, K, 1, 1))
        b = model(chars, tag_teacher=tags(2, K, 7, 9))
    assert a.slot_states is not None
    assert not torch.equal(a.slot_states, b.slot_states)
