"""HydraModel: character-level TCN encoder + K parallel slot decoders.

Encoder: shared char embedding -> char TCN -> masked max-pool (token vectors)
-> context TCN over the token axis -> fused per-token representation.
Decoder: K slot embeddings; each slot classifies POS (+NULL) and morph, and
emits the lemma as a character grid via transposed convolutions with optional
cross-attention to the surface characters (open vocabulary).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .vocab import NULL, PAD

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _math_sdpa():
        """The fused mem-efficient SDPA kernel returns NaN in backward for
        heavily key-padded rows (tiny documents inside a mostly-padded chunk)
        — even in fp32. The math backend computes the masked softmax
        correctly; these attentions are small, the cost is negligible."""
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:  # torch < 2.3
    import contextlib

    def _math_sdpa():
        return contextlib.nullcontext()


@dataclass
class ModelOutput:
    pos_logits: torch.Tensor | None    # (B, T, K, P); None in MLM pretraining mode
    morph_logits: torch.Tensor | None  # (B, T, K, M)
    lemma_logits: torch.Tensor | None  # (B, T, K, L, C)
    lemma_cls_logits: torch.Tensor | None = None  # (B, T, K, n_lemma_types)
    mlm_logits: torch.Tensor | None = None        # (B, T, n_word_types)
    joint_logits: torch.Tensor | None = None      # (B, T, K, n_joint_types)
    # AR decoding state (set when lemma_decoder='ar_tcn' runs without teacher):
    slot_states: torch.Tensor | None = None       # (B*T*K, d_model)
    char_states: torch.Tensor | None = None       # (B*T*K, W, d_tok)
    char_pad_mask: torch.Tensor | None = None     # (B*T*K, W)
    # slot-0 Viterbi path when model.pos_crf is on and no teacher was given;
    # decoding must prefer this over the per-token argmax
    pos_path: torch.Tensor | None = None          # (B, T)
    # model.count_head: item count per token, classes 0..K (0 never a
    # target -- it marks context-only tokens). Decoding prefers this over
    # the implicit first-NULL rule.
    count_logits: torch.Tensor | None = None      # (B, T, K+1)


class TCNBlock(nn.Module):
    """Pre-norm residual block: LN -> dilated conv -> GELU -> conv.

    Non-causal by default (symmetric padding: encoder use — right context must
    flow). causal=True pads left only, so position t sees positions <= t: the
    WaveNet-style variant for autoregressive decoding. Optional ECA-style
    channel gating (no token-to-token mixing)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float,
                 channel_gate: bool = False, causal: bool = False):
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1) if causal else 0
        pad = 0 if causal else dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False) if channel_gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, C)
        y = self.norm(x).transpose(1, 2)
        if self.left_pad:
            y = torch.nn.functional.pad(y, (self.left_pad, 0))
        y = self.act(self.conv1(y))
        if self.left_pad:
            y = torch.nn.functional.pad(y, (self.left_pad, 0))
        y = self.conv2(y)
        if self.gate is not None:
            g = torch.sigmoid(self.gate(y.mean(dim=2).unsqueeze(1)))  # (N, 1, C)
            y = y * g.transpose(1, 2)
        y = y.transpose(1, 2)
        return x + self.dropout(y)


def tcn_stack(channels: int, kernel_size: int, dilations: tuple[int, ...],
              dropout: float, channel_gate: bool = False,
              causal: bool = False) -> nn.Sequential:
    return nn.Sequential(*[TCNBlock(channels, kernel_size, d, dropout, channel_gate, causal)
                           for d in dilations])


class LemmaDecoder(nn.Module):
    """Per-slot lemma generator: seed -> 3x ConvTranspose (len x8) -> optional
    cross-attention to surface chars -> refinement TCN -> char logits."""

    def __init__(self, cfg: ModelConfig, max_lemma_len: int, n_chars: int):
        super().__init__()
        if max_lemma_len % 8 != 0:
            raise ValueError("max_lemma_len must be a multiple of 8 (3 stride-2 upsamplings)")
        self.seed_len = max_lemma_len // 8
        d = cfg.d_dec
        self.seed = nn.Linear(cfg.d_model, self.seed_len * d)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(d, d, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(d, d, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(d, d, 4, stride=2, padding=1), nn.GELU(),
        )
        self.pos_emb = nn.Embedding(max_lemma_len, d)
        self.use_attn = cfg.lemma_cross_attention
        if self.use_attn:
            self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True,
                                              kdim=cfg.d_tok, vdim=cfg.d_tok,
                                              dropout=cfg.dropout)
            self.attn_norm = nn.LayerNorm(d)
        self.refine = tcn_stack(d, cfg.kernel_size, (1, 2), cfg.dropout)
        self.out = nn.Linear(d, n_chars)

    def forward(self, slot_vec: torch.Tensor, char_states: torch.Tensor | None,
                char_pad_mask: torch.Tensor | None) -> torch.Tensor:
        """slot_vec (N, d_model); char_states (N, W, d_tok); char_pad_mask (N, W)
        True where the key is padding. Returns (N, L, n_chars)."""
        n = slot_vec.shape[0]
        x = self.seed(slot_vec).view(n, -1, self.seed_len)  # (N, d, seed_len)
        x = self.up(x).transpose(1, 2)                       # (N, L, d)
        x = x + self.pos_emb.weight.unsqueeze(0)
        if self.use_attn and char_states is not None:
            kpm = char_pad_mask.clone()
            kpm[kpm.all(dim=-1), 0] = False  # avoid NaN on fully-padded tokens
            with _math_sdpa():
                att, _ = self.attn(x, char_states, char_states, key_padding_mask=kpm,
                                   need_weights=False)
            x = self.attn_norm(x + att)
        x = self.refine(x)
        return self.out(x)


class LemmaDecoderAR(nn.Module):
    """Autoregressive fully-convolutional lemma decoder (WaveNet-style).

    Input position t holds the embedding of char t-1 (PAD acts as BOS) plus the
    slot conditioning vector and a positional embedding; a CAUSAL TCN ensures
    output position t depends only on chars < t. Training is one parallel
    teacher-forced pass; generation feeds predictions back step by step."""

    def __init__(self, cfg: ModelConfig, max_lemma_len: int, n_chars: int,
                 char_emb: nn.Embedding):
        super().__init__()
        d = cfg.d_dec
        self.char_emb = char_emb  # shared with the encoder
        self.in_proj = nn.Linear(char_emb.embedding_dim, d)
        self.cond = nn.Linear(cfg.d_model, d)
        self.pos_emb = nn.Embedding(max_lemma_len, d)
        self.tcn = tcn_stack(d, cfg.kernel_size, (1, 2, 4, 8), cfg.dropout, causal=True)
        self.use_attn = cfg.lemma_cross_attention
        if self.use_attn:
            self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True,
                                              kdim=cfg.d_tok, vdim=cfg.d_tok,
                                              dropout=cfg.dropout)
            self.attn_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_chars)

    def forward(self, slot_vec: torch.Tensor, prev_chars: torch.Tensor,
                char_states: torch.Tensor | None,
                char_pad_mask: torch.Tensor | None) -> torch.Tensor:
        """slot_vec (N, d_model); prev_chars (N, L) int64 (char t-1 at pos t,
        PAD as BOS/filler). Returns logits (N, L, n_chars)."""
        L = prev_chars.shape[1]
        x = self.in_proj(self.char_emb(prev_chars)) \
            + self.cond(slot_vec).unsqueeze(1) + self.pos_emb.weight[:L].unsqueeze(0)
        x = self.tcn(x)
        if self.use_attn and char_states is not None:
            kpm = char_pad_mask.clone()
            kpm[kpm.all(dim=-1), 0] = False
            with _math_sdpa():
                att, _ = self.attn(x, char_states, char_states, key_padding_mask=kpm,
                                   need_weights=False)
            x = self.attn_norm(x + att)
        return self.out(x)

    @torch.inference_mode()
    def generate(self, slot_vec: torch.Tensor, char_states: torch.Tensor | None,
                 char_pad_mask: torch.Tensor | None, max_len: int) -> torch.Tensor:
        """Greedy generation; returns char ids (N, max_len)."""
        n = slot_vec.shape[0]
        prev = torch.full((n, max_len), PAD, dtype=torch.long, device=slot_vec.device)
        outp = torch.full((n, max_len), PAD, dtype=torch.long, device=slot_vec.device)
        for t in range(max_len):
            logits = self.forward(slot_vec, prev, char_states, char_pad_mask)
            step = logits[:, t].argmax(dim=-1)
            outp[:, t] = step
            if t + 1 < max_len:
                prev[:, t + 1] = step
        return outp


def _last_valid(mask: torch.Tensor) -> torch.Tensor:
    """Index of the last valid position in each row of a 0/1 mask.

    NOT mask.sum()-1: that is the COUNT of valid positions, which only equals
    the last index when the mask is a contiguous prefix. It is not — the
    masked-LM objective blanks ~15% of tokens and data.py sets their POS
    target to IGNORE, punching scattered holes into the sequence. Using the
    count as an index made the gold path collect an `end` transition belonging
    to some other (often masked) position, which the model could then inflate
    freely: the training objective went NEGATIVE and dev accuracy fell below
    the baseline (run s_crf, 2026-09-14, discarded).
    """
    idx = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return (mask * idx).argmax(dim=1)


def _compact(em: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor):
    """Gather the valid positions of each row into a contiguous prefix.

    The masked-LM objective blanks ~15% of tokens and data.py sets their POS
    target to IGNORE, so the training mask has holes in the MIDDLE. A
    linear-chain CRF cannot simply skip them in place: the gold path would take
    its transition from tags[t-1] even when t-1 is blanked, whilst the forward
    recursion carries alpha from the last *valid* position — two different
    edges — and the end transition would attach to the wrong index. Both
    failures let the gold score exceed the partition function, i.e. a negative
    "likelihood" the model can inflate at will (run s_crf, 2026-09-14).

    Compacting first makes the mask a true prefix, so every position's
    predecessor is the previous ANNOTATED token and the simple indexing is
    correct. The CRF then models transitions between consecutive annotated
    tokens, skipping blanked ones — which is the intended semantics.
    """
    B, T, C = em.shape
    # stable sort puts valid (0) before invalid (1) whilst preserving order
    order = torch.argsort((~mask.bool()).to(torch.int8), dim=1, stable=True)
    em = em.gather(1, order.unsqueeze(-1).expand(B, T, C))
    tags = tags.gather(1, order)
    counts = mask.sum(1, keepdim=True)
    prefix = (torch.arange(T, device=mask.device).unsqueeze(0) < counts).to(mask.dtype)
    return em, tags, prefix


class PosCRF(nn.Module):
    """Linear-chain CRF over the slot-0 part-of-speech sequence.

    §5.4's oracle diagnostic showed the tag-conditioned lemma decoder's whole
    remaining headroom sits on tokens whose POS is predicted wrongly, so POS is
    the throttle on the lemmatiser, not merely one metric among several. The
    heads otherwise decide each token independently, whilst the data plainly
    does not: morph is 88.4% correct when POS is right and 35.8% when it is
    wrong. This models the missing dependency ACROSS tokens.

    Slot 0 only. It is the one slot guaranteed to carry a real tag (decoding
    forces it non-NULL), so the sequence is well defined at every position;
    slots 1..K-1 keep their per-token cross-entropy. A CRF over the combined
    POS|morph tag would be the tidier target -- it encodes a whole multi-item
    token in one label -- but that is 3,148 states, a 9.9M-parameter transition
    matrix that would be almost entirely unobserved. POS is 76 states, 5,776
    transitions.

    Transitions are ZERO-INITIALISED, so at step 0 the CRF contributes only the
    emission scores and warm-starting from a non-CRF checkpoint is a no-op --
    the same discipline the tag-conditioning projections use.

    Scope limit worth stating: the sequence is one chunk (chunk_len tokens).
    The halo feeds the encoder but is sliced off before the heads, so the
    transition spanning two adjacent chunks is not modelled -- one transition
    per chunk_len, which at the default 128 is negligible.
    """

    def __init__(self, n_tags: int):
        super().__init__()
        self.n_tags = n_tags
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))  # trans[i, j]: i -> j
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def _gold_score(self, em: torch.Tensor, tags: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """Score of the gold path. em (B,T,C), tags (B,T), mask (B,T) float."""
        B, T, _ = em.shape
        score = (self.start[tags[:, 0]]
                 + em[:, 0].gather(1, tags[:, :1]).squeeze(1) * mask[:, 0])
        for t in range(1, T):
            step = (self.trans[tags[:, t - 1], tags[:, t]]
                    + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1))
            score = score + step * mask[:, t]
        return score + self.end[tags.gather(1, _last_valid(mask).unsqueeze(1)).squeeze(1)]

    def _log_partition(self, em: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = em.shape
        # gate position 0's emission by the mask exactly as _gold_score does,
        # so the two remain comparable when position 0 is itself masked
        alpha = self.start.unsqueeze(0) + em[:, 0] * mask[:, :1]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0),
                                  dim=1) + em[:, t]
            keep = mask[:, t].unsqueeze(1).bool()
            alpha = torch.where(keep, nxt, alpha)   # padded steps carry alpha forward
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def nll(self, em: torch.Tensor, tags: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
        """Per-token negative log-likelihood, so loss.w_pos keeps the same
        meaning it has for the cross-entropy it replaces."""
        em = em.float()
        mask = mask.float()
        n = mask.sum().clamp(min=1.0)
        # holes must go before the chain logic can be trusted -- see _compact
        em, tags, mask = _compact(em, tags, mask)
        total = (self._log_partition(em, mask) - self._gold_score(em, tags, mask)).sum()
        return total / n

    @torch.no_grad()
    def viterbi(self, em: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Best path. Returns (B, T) tag ids; positions past the mask repeat
        the last valid tag, which decoding ignores."""
        em = em.float()
        B, T, C = em.shape
        score = self.start.unsqueeze(0) + em[:, 0]
        ident = torch.arange(C, device=em.device).unsqueeze(0).expand(B, C)
        backptr: list[torch.Tensor] = []
        for t in range(1, T):
            cand = score.unsqueeze(2) + self.trans.unsqueeze(0)      # (B, C, C)
            best, idx = cand.max(dim=1)
            nxt = best + em[:, t]
            keep = mask[:, t].unsqueeze(1).bool()
            score = torch.where(keep, nxt, score)
            # a padded step must not advance the path: point each tag at itself,
            # so backtracking through trailing padding is a no-op
            backptr.append(torch.where(keep, idx, ident))
        best_last = (score + self.end.unsqueeze(0)).argmax(dim=1)     # (B,)
        path = [best_last]
        for idx in reversed(backptr):
            best_last = idx.gather(1, best_last.unsqueeze(1)).squeeze(1)
            path.append(best_last)
        return torch.stack(list(reversed(path)), dim=1)


class HydraModel(nn.Module):
    def __init__(self, cfg: ModelConfig, n_chars: int, n_pos: int, n_morph: int,
                 max_word_len: int, max_lemma_len: int, chunk_len: int, halo: int,
                 n_lemma_types: int = 0, n_word_types: int = 0, n_joint_types: int = 0):
        super().__init__()
        self.cfg = cfg
        self.T = chunk_len
        self.H = halo
        self.K = cfg.n_slots

        self.char_emb = nn.Embedding(n_chars, cfg.d_char, padding_idx=PAD)
        self.char_in = nn.Linear(cfg.d_char, cfg.d_tok)
        self.char_pos_emb = nn.Embedding(max_word_len, cfg.d_tok)
        self.char_tcn = tcn_stack(cfg.d_tok, cfg.kernel_size, cfg.char_tcn_dilations,
                                  cfg.dropout, cfg.tcn_channel_gate)
        self.ctx_tcn = tcn_stack(cfg.d_tok, cfg.kernel_size, cfg.ctx_tcn_dilations,
                                 cfg.dropout, cfg.tcn_channel_gate)

        self.ctx_attn = self.ctx_attn_norm = None
        if cfg.ctx_self_attention:
            self.ctx_attn = nn.MultiheadAttention(cfg.d_tok, num_heads=8, batch_first=True,
                                                  dropout=cfg.dropout)
            self.ctx_attn_norm = nn.LayerNorm(cfg.d_tok)

        self.mlm_head = None
        if cfg.pretrain_mlm and not cfg.masked_lm:
            raise ValueError("pretrain_mlm=true requires masked_lm=true")
        if cfg.masked_lm:
            if n_word_types <= 0:
                raise ValueError("masked_lm=true requires n_word_types > 0")
            self.mlm_head = nn.Linear(cfg.d_tok, n_word_types)

        # multi-head attentive pooling over character states (suffix/edge-aware,
        # unlike max-pool); channels are split across 4 heads
        self.pool_scores = nn.Linear(cfg.d_tok, 4) if cfg.attention_pooling else None

        self.joint_head = None
        if cfg.joint_tag:
            if n_joint_types <= 0:
                raise ValueError("joint_tag=true requires n_joint_types > 0")
            self.joint_head = nn.Linear(cfg.d_model, n_joint_types)

        self.fuse = nn.Sequential(
            nn.Linear(2 * cfg.d_tok, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))

        self.slot_emb = nn.Parameter(torch.randn(cfg.n_slots, cfg.d_model) * 0.02)
        self.slot_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.slot_norm = nn.LayerNorm(cfg.d_model)

        self.count_head = nn.Linear(cfg.d_model, cfg.n_slots + 1) if cfg.count_head else None
        self.pos_head = nn.Linear(cfg.d_model, n_pos)
        self.morph_head = nn.Linear(cfg.d_model, n_morph)
        # zero-initialised transitions: warm-starting a non-CRF checkpoint is a
        # no-op at step 0, so a comparison isolates the CRF (same discipline as
        # the tag-conditioning projections below)
        self.pos_crf = PosCRF(n_pos) if cfg.pos_crf else None

        # predict-then-condition cascade. The conditioning projections are
        # ZERO-INITIALISED, so at step 0 the model is bit-identical to an
        # unconditioned one: warm-starting a mature checkpoint is safe by
        # construction (cf. the ungated lemma classifier, which clobbered a
        # trained generator and crashed epoch-0 dev lemma to 73%).
        # Index n_pos / n_morph is the learned "unsure" slot used when the
        # confidence gate rejects a prediction or a target is IGNORE.
        self.pos_cond_emb = self.morph_cond_emb = None
        self.tag_to_morph = self.tag_to_lemma = None
        self.n_pos, self.n_morph = n_pos, n_morph
        # inference-time gate, set from cfg.infer.tag_cond_min_prob by the
        # eval/tag entry points; 0.0 = always trust the argmax
        self.tag_cond_min_prob = 0.0
        # cfg.tag_cond_soft: 0 = pure hard lookup, 1 = pure distribution blend.
        # The training loop ramps it 0 -> 1 (train.tag_cond_ramp_steps); it
        # stays 1 for inference, so a converged model trains and infers on the
        # same signal -- the train/test match the confidence gate never had.
        self.tag_cond_lambda = 1.0
        if cfg.tag_condition != "off":
            dt = cfg.tag_cond_dim
            self.pos_cond_emb = nn.Embedding(n_pos + 1, dt)
            self.morph_cond_emb = nn.Embedding(n_morph + 1, dt)
            self.tag_to_lemma = nn.Linear(2 * dt, cfg.d_model)
            nn.init.zeros_(self.tag_to_lemma.weight)
            nn.init.zeros_(self.tag_to_lemma.bias)
            if cfg.tag_condition == "morph+lemma":
                self.tag_to_morph = nn.Linear(dt, cfg.d_model)
                nn.init.zeros_(self.tag_to_morph.weight)
                nn.init.zeros_(self.tag_to_morph.bias)

        self.max_lemma_len = max_lemma_len
        if cfg.lemma_decoder == "ar_tcn":
            self.lemma_decoder = LemmaDecoderAR(cfg, max_lemma_len, n_chars, self.char_emb)
        elif cfg.lemma_decoder == "grid":
            self.lemma_decoder = LemmaDecoder(cfg, max_lemma_len, n_chars)
        else:
            raise ValueError(f"unknown lemma_decoder {cfg.lemma_decoder!r}")
        # classify-or-generate: type-level lemma classifier, factored projection
        # to keep the ~37k-class output affordable; class UNK = "generate"
        self.lemma_cls_head = None
        if cfg.lemma_classifier:
            if n_lemma_types <= 0:
                raise ValueError("lemma_classifier=true requires n_lemma_types > 0")
            self.lemma_cls_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_dec), nn.GELU(),
                nn.Linear(cfg.d_dec, n_lemma_types))

    def _cond_ids(self, logits: torch.Tensor, teacher: torch.Tensor | None,
                  n_classes: int) -> torch.Tensor:
        """Resolve tag ids for the conditioning path: gold when teacher-forced,
        otherwise the (optionally confidence-gated) argmax. IGNORE targets and
        rejected low-confidence predictions both map to the learned "unsure"
        index n_classes. Discrete by design — no gradient flows back into the
        upstream head through this path."""
        if teacher is not None:
            return torch.where(teacher < 0, n_classes, teacher.long())
        if self.tag_cond_min_prob > 0:
            conf, ids = logits.float().softmax(dim=-1).max(dim=-1)
            return torch.where(conf >= self.tag_cond_min_prob, ids,
                               torch.full_like(ids, n_classes))
        return logits.argmax(dim=-1)

    def _cond_vec(self, logits: torch.Tensor, emb: nn.Embedding, n_classes: int,
                  ids: torch.Tensor) -> torch.Tensor:
        """The conditioning vector the downstream heads actually consume.

        Hard (default): a plain lookup of `ids` -- gold when teacher-forced,
        else the argmax. One discrete decision, exactly as before.

        Soft (cfg.tag_cond_soft): a softmax-weighted blend of the tag rows, so
        the head sees the whole distribution and can hedge where the tagger is
        unsure instead of inheriting one wrong decision. The blend spans rows
        [0, n_classes) ONLY -- never the "unsure" row at index n_classes, which
        receives no gradient in training (every path that indexes it has IGNORE
        targets downstream) and whose use is why the confidence gate only hurt.

        With the CRF on, `ids` carries the Viterbi tag for slot 0; the soft
        component uses the head's own per-token distribution instead, so
        sequence-level information reaches the output but not this path.
        """
        hard = emb(ids)
        if not self.cfg.tag_cond_soft or self.tag_cond_lambda <= 0.0:
            return hard
        # detached on purpose: this changes WHAT the lemma head consumes, not
        # what trains the tagger. Letting the lemma loss shape POS is a
        # separate ablation (it could trade POS accuracy, a reported metric).
        probs = logits[..., :n_classes].detach().float().softmax(dim=-1)
        soft = (probs.to(emb.weight.dtype) @ emb.weight[:n_classes]).to(hard.dtype)
        lam = self.tag_cond_lambda
        return soft if lam >= 1.0 else (1.0 - lam) * hard + lam * soft

    def forward(self, chars: torch.Tensor,
                lemma_teacher: torch.Tensor | None = None,
                tag_teacher: tuple[torch.Tensor, torch.Tensor] | None = None) -> ModelOutput:
        """chars: (B, S, W) int64 with S = T + 2H.
        lemma_teacher: gold lemma char grid (B, T, K, L) for teacher-forced
        training of the AR decoder (ignored by the grid decoder).
        tag_teacher: (pos, morph) gold ids (B, T, K) teacher-forcing the
        model.tag_condition cascade; None -> the heads' own predictions."""
        B, S, W = chars.shape
        T, H, K = self.T, self.H, self.K
        char_valid = chars != PAD                      # (B, S, W)
        token_valid = char_valid.any(dim=-1)           # (B, S)

        x = self.char_emb(chars.view(B * S, W))        # (B*S, W, d_char)
        x = self.char_in(x) + self.char_pos_emb.weight.unsqueeze(0)
        x = self.char_tcn(x)                           # (B*S, W, d_tok)

        # pool character positions -> token vectors
        neg = torch.finfo(x.dtype).min
        pool_mask = char_valid.view(B * S, W, 1)
        if self.pool_scores is not None:
            scores = self.pool_scores(x)                                # (B*S, W, 4)
            # under autocast scores may be fp16 while x is fp32: fill with
            # the minimum of the scores' own dtype
            scores = scores.masked_fill(~pool_mask, torch.finfo(scores.dtype).min)
            alpha = torch.nan_to_num(scores.softmax(dim=1))             # all-pad rows -> 0
            xh = x.view(B * S, W, 4, -1)
            tok = (alpha.unsqueeze(-1) * xh).sum(dim=1).reshape(B * S, -1)
        else:
            tok = x.masked_fill(~pool_mask, neg).max(dim=1).values
        tok = tok * token_valid.view(B * S, 1)         # zero all-pad tokens
        tok = tok.view(B, S, -1)                       # (B, S, d_tok)

        ctx = self.ctx_tcn(tok)                        # (B, S, d_tok)
        if self.ctx_attn is not None:
            kpm = ~token_valid                         # True = ignore
            kpm = kpm.clone()
            kpm[kpm.all(dim=-1), 0] = False            # avoid NaN on all-pad rows
            with _math_sdpa():
                att, _ = self.ctx_attn(ctx, ctx, ctx, key_padding_mask=kpm,
                                       need_weights=False)
            ctx = self.ctx_attn_norm(ctx + att)

        center = slice(H, H + T)

        if self.cfg.pretrain_mlm:
            # MLM-only pretraining: no slot decoding, no lemma decoder
            mlm_logits = self.mlm_head(ctx[:, center])
            return ModelOutput(None, None, None, None, mlm_logits, None)

        h = self.fuse(torch.cat([tok[:, center], ctx[:, center]], dim=-1))  # (B, T, d_model)

        hs = h.unsqueeze(2) + self.slot_emb.view(1, 1, K, -1)               # (B, T, K, d_model)
        hs = self.slot_norm(hs + self.slot_mlp(hs))

        count_logits = self.count_head(h) if self.count_head is not None else None
        pos_logits = self.pos_head(hs)

        cond_on = self.cfg.tag_condition != "off"
        pos_t = tag_teacher[0] if tag_teacher is not None else None
        morph_t = tag_teacher[1] if tag_teacher is not None else None

        # cascade stage 1: POS -> morph. Morph is 88.4% correct when POS is
        # right vs 35.1% when wrong, so the dependency is already there —
        # this makes it explicit rather than leaving it implicit in `hs`.
        # Viterbi replaces the slot-0 argmax at inference: the CRF's whole point
        # is that the best SEQUENCE differs from the best tag at each position.
        # Only when not teacher-forced -- training conditions on gold anyway.
        pos_path = None
        if self.pos_crf is not None and pos_t is None:
            # slot 0 is forced non-NULL by decoding, so NULL must be unreachable
            # for the Viterbi path too, or the CRF could break that invariant
            # .clone() is load-bearing: .detach().float() is a no-op view when
            # the dtype already matches, so masking NULL would write straight
            # back into the pos_logits this forward returns
            em0 = pos_logits[:, :, 0, :].detach().clone().float()
            em0[..., NULL] = torch.finfo(em0.dtype).min
            pos_path = self.pos_crf.viterbi(em0, token_valid[:, center])

        pos_ids = self._cond_ids(pos_logits, pos_t, self.n_pos) if cond_on else None
        if cond_on and pos_path is not None:
            # slots 1..K-1 keep their per-slot argmax; only slot 0 has a CRF
            pos_ids = pos_ids.clone()
            pos_ids[..., 0] = pos_path
        pos_vec = (self._cond_vec(pos_logits, self.pos_cond_emb, self.n_pos, pos_ids)
                   if cond_on else None)
        if self.tag_to_morph is not None:
            morph_logits = self.morph_head(hs + self.tag_to_morph(pos_vec))
        else:
            morph_logits = self.morph_head(hs)

        # cascade stage 2: POS+morph -> lemma
        hs_lem = hs
        if cond_on:
            morph_ids = self._cond_ids(morph_logits, morph_t, self.n_morph)
            morph_vec = self._cond_vec(morph_logits, self.morph_cond_emb,
                                       self.n_morph, morph_ids)
            cond = torch.cat([pos_vec, morph_vec], dim=-1)
            hs_lem = hs + self.tag_to_lemma(cond)

        cls_in = hs_lem if self.cfg.tag_condition_classifier else hs
        lemma_cls_logits = self.lemma_cls_head(cls_in) if self.lemma_cls_head is not None else None
        mlm_logits = self.mlm_head(ctx[:, center]) if self.mlm_head is not None else None
        # the joint head predicts POS|morph itself — conditioning it on POS
        # would be circular, so it keeps the unconditioned representation
        joint_logits = self.joint_head(hs) if self.joint_head is not None else None

        flat = hs_lem.reshape(B * T * K, -1)
        char_states = char_pad_mask = None
        if self.cfg.lemma_cross_attention:
            cs = x.view(B, S, W, -1)[:, center]        # (B, T, W, d_tok)
            char_states = (cs.unsqueeze(2).expand(B, T, K, W, cs.shape[-1])
                           .reshape(B * T * K, W, -1))
            cm = ~char_valid[:, center]                # True = padding
            char_pad_mask = cm.unsqueeze(2).expand(B, T, K, W).reshape(B * T * K, W)

        if isinstance(self.lemma_decoder, LemmaDecoderAR):
            if lemma_teacher is None:
                # generation happens outside (metrics.decode_batch drives it)
                return ModelOutput(pos_logits, morph_logits, None, lemma_cls_logits,
                                   mlm_logits, joint_logits, flat, char_states,
                                   char_pad_mask, pos_path, count_logits)
            gold = lemma_teacher.reshape(B * T * K, -1).clamp_min(PAD)  # IGNORE -> PAD
            prev = torch.cat([gold.new_full((gold.shape[0], 1), PAD), gold[:, :-1]], dim=1)
            lemma_logits = self.lemma_decoder(flat, prev, char_states, char_pad_mask)
        else:
            lemma_logits = self.lemma_decoder(flat, char_states, char_pad_mask)
        lemma_logits = lemma_logits.view(B, T, K, lemma_logits.shape[1], -1)

        return ModelOutput(pos_logits, morph_logits, lemma_logits, lemma_cls_logits,
                           mlm_logits, joint_logits, pos_path=pos_path,
                           count_logits=count_logits)
