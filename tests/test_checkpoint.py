"""Resume from checkpoint must reproduce training bit-exactly (incl. RNG)."""
import dataclasses

import torch

from hydra.checkpoint import load_checkpoint, restore_rng_states, save_checkpoint
from hydra.config import LossConfig
from hydra.data import HydraDataset, collate, load_split_tokens
from hydra.losses import compute_loss
from hydra.model import HydraModel
from hydra.train import build_scheduler
from hydra.vocab import Vocabs


def setup(corpus_dir, data_cfg, model_cfg):
    docs = load_split_tokens([str(corpus_dir / "doc1.txt"), str(corpus_dir / "doc2.txt")],
                             "skip", model_cfg.n_slots)
    vocabs = Vocabs.build([t for d in docs for t in d])
    ds = HydraDataset(docs, vocabs, data_cfg, model_cfg.n_slots)
    batch = collate([ds[i] for i in range(len(ds))])
    model = HydraModel(model_cfg, len(vocabs.chars), len(vocabs.pos), len(vocabs.morph),
                       data_cfg.max_word_len, data_cfg.max_lemma_len,
                       data_cfg.chunk_len, data_cfg.halo)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=3, total_steps=20, lr=1e-3, lr_min=1e-5)
    return vocabs, batch, model, opt, sched


def run_steps(model, opt, sched, batch, n_pos, n):
    model.train()
    losses = []
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        out = model(batch["chars"])
        loss, _ = compute_loss(out, batch, LossConfig(), n_pos)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
    return losses


def test_resume_is_bit_exact(corpus_dir, data_cfg, model_cfg, tmp_path):
    # dropout > 0 so RNG restoration is actually exercised
    model_cfg = dataclasses.replace(model_cfg, dropout=0.1)

    torch.manual_seed(123)
    vocabs, batch, model, opt, sched = setup(corpus_dir, data_cfg, model_cfg)
    run_steps(model, opt, sched, batch, len(vocabs.pos), 5)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sched, scaler=None,
                    epoch=0, step=5, best_metric=0.0, patience_left=3, config_dict={})
    losses_a = run_steps(model, opt, sched, batch, len(vocabs.pos), 5)
    state_a = {k: v.clone() for k, v in model.state_dict().items()}

    # fresh objects, different seed, then restore everything from the checkpoint
    torch.manual_seed(999)
    _, batch2, model2, opt2, sched2 = setup(corpus_dir, data_cfg, model_cfg)
    payload = load_checkpoint(ckpt)
    model2.load_state_dict(payload["model"])
    opt2.load_state_dict(payload["optimizer"])
    sched2.load_state_dict(payload["scheduler"])
    restore_rng_states(payload["rng"])
    losses_b = run_steps(model2, opt2, sched2, batch2, len(vocabs.pos), 5)
    state_b = model2.state_dict()

    assert losses_a == losses_b
    for k in state_a:
        assert torch.equal(state_a[k], state_b[k]), f"mismatch in {k}"
