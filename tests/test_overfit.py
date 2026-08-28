"""End-to-end: a tiny model must memorize the fixture corpus through the real
decode path — proves slot/NULL/EOW wiring (the failure class of the old code)."""
import torch

from hydra.config import LossConfig
from hydra.data import HydraDataset, collate, load_split_tokens
from hydra.evaluate import evaluate_dataset
from hydra.losses import compute_loss
from hydra.model import HydraModel
from hydra.vocab import Vocabs


def test_overfit_tiny_corpus(corpus_dir, data_cfg, model_cfg):
    torch.manual_seed(0)
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", model_cfg.n_slots)
    vocabs = Vocabs.build([t for d in docs for t in d])
    ds = HydraDataset(docs, vocabs, data_cfg, model_cfg.n_slots, training=True)
    batch = collate([ds[i] for i in range(len(ds))])

    model = HydraModel(model_cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       data_cfg.max_word_len, data_cfg.max_lemma_len,
                       data_cfg.chunk_len, data_cfg.halo)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_cfg = LossConfig()

    metrics = {}
    for it in range(1, 801):
        opt.zero_grad(set_to_none=True)
        out = model(batch["chars"])
        loss, _ = compute_loss(out, batch, loss_cfg, len(vocabs.pos))
        loss.backward()
        opt.step()
        if it % 100 == 0:
            metrics = evaluate_dataset(model, ds, vocabs, torch.device("cpu"), 4)
            if metrics.get("acc_joint", 0.0) == 1.0:
                break

    assert metrics.get("acc_joint", 0.0) == 1.0, f"failed to memorize: {metrics}"
    # multi-item tokens specifically must be perfect (2 in the fixture corpus)
    assert metrics["multi_n"] >= 2
    assert metrics["multi_acc_joint"] == 1.0
