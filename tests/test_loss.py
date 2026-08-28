import torch

from hydra.config import LossConfig
from hydra.data import IGNORE
from hydra.losses import compute_loss
from hydra.model import ModelOutput
from hydra.vocab import NULL


def make_output(B=1, T=2, K=2, P=5, M=6, L=4, C=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ModelOutput(
        pos_logits=torch.randn(B, T, K, P, generator=g, requires_grad=True),
        morph_logits=torch.randn(B, T, K, M, generator=g, requires_grad=True),
        lemma_logits=torch.randn(B, T, K, L, C, generator=g, requires_grad=True),
    )


def full_ignore_batch(B=1, T=2, K=2, L=4):
    return {
        "pos": torch.full((B, T, K), IGNORE, dtype=torch.int64),
        "morph": torch.full((B, T, K), IGNORE, dtype=torch.int64),
        "lemma": torch.full((B, T, K, L), IGNORE, dtype=torch.int64),
    }


def test_all_ignore_gives_zero():
    out = make_output()
    loss, parts = compute_loss(out, full_ignore_batch(), LossConfig(), n_pos=5)
    assert float(loss) == 0.0
    assert parts["loss_pos"] == 0.0 and parts["loss_lemma"] == 0.0
    loss.backward()  # must be differentiable even when empty


def test_finite_loss_and_parts():
    out = make_output()
    batch = full_ignore_batch()
    batch["pos"][0, 0] = torch.tensor([3, NULL])
    batch["morph"][0, 0, 0] = 2
    batch["lemma"][0, 0, 0] = torch.tensor([4, 5, 2, IGNORE])
    loss, parts = compute_loss(out, batch, LossConfig(), n_pos=5)
    assert torch.isfinite(loss)
    assert parts["loss_pos"] > 0 and parts["loss_morph"] > 0 and parts["loss_lemma"] > 0


def test_null_weight_scales_pos_loss():
    out = make_output()
    batch = full_ignore_batch()
    batch["pos"][0, 0] = torch.tensor([3, NULL])  # one real, one NULL target
    _, parts_lo = compute_loss(out, batch, LossConfig(null_weight=0.1), n_pos=5)
    _, parts_hi = compute_loss(out, batch, LossConfig(null_weight=1.0), n_pos=5)
    assert parts_lo["loss_pos"] != parts_hi["loss_pos"]
    # with weight 1.0 the weighted mean equals the plain mean
    plain = torch.nn.functional.cross_entropy(
        out.pos_logits.reshape(-1, 5), batch["pos"].reshape(-1), ignore_index=IGNORE)
    assert abs(parts_hi["loss_pos"] - float(plain)) < 1e-6
