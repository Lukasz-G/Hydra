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
from .vocab import PAD

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

        self.pos_head = nn.Linear(cfg.d_model, n_pos)
        self.morph_head = nn.Linear(cfg.d_model, n_morph)
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

    def forward(self, chars: torch.Tensor,
                lemma_teacher: torch.Tensor | None = None) -> ModelOutput:
        """chars: (B, S, W) int64 with S = T + 2H.
        lemma_teacher: gold lemma char grid (B, T, K, L) for teacher-forced
        training of the AR decoder (ignored by the grid decoder)."""
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

        pos_logits = self.pos_head(hs)
        morph_logits = self.morph_head(hs)
        lemma_cls_logits = self.lemma_cls_head(hs) if self.lemma_cls_head is not None else None
        mlm_logits = self.mlm_head(ctx[:, center]) if self.mlm_head is not None else None
        joint_logits = self.joint_head(hs) if self.joint_head is not None else None

        flat = hs.reshape(B * T * K, -1)
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
                                   char_pad_mask)
            gold = lemma_teacher.reshape(B * T * K, -1).clamp_min(PAD)  # IGNORE -> PAD
            prev = torch.cat([gold.new_full((gold.shape[0], 1), PAD), gold[:, :-1]], dim=1)
            lemma_logits = self.lemma_decoder(flat, prev, char_states, char_pad_mask)
        else:
            lemma_logits = self.lemma_decoder(flat, char_states, char_pad_mask)
        lemma_logits = lemma_logits.view(B, T, K, lemma_logits.shape[1], -1)

        return ModelOutput(pos_logits, morph_logits, lemma_logits, lemma_cls_logits,
                           mlm_logits, joint_logits)
