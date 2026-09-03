"""Configuration: TOML file -> nested frozen dataclasses, with dotted-key overrides."""
from __future__ import annotations

import ast
import dataclasses

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    run_dir: str = "runs/default"
    seed: int = 1337


@dataclass(frozen=True)
class DataConfig:
    corpus_dir: str | None = None
    train_dir: str | None = None
    dev_dir: str | None = None
    test_dir: str | None = None
    dev_fraction: float = 0.05
    test_fraction: float = 0.05
    split_seed: int = 42
    # "file": random whole manuscripts held out (hard protocol, historical default)
    # "stratified": whole manuscripts held out, balanced by dialect x period x type
    #               (requires metadata_csv), token-weighted
    # "chunk": random chunks within all files (protocol comparable to Pie/Schmid)
    split_mode: str = "file"
    metadata_csv: str | None = None
    # directory of additional UNANNOTATED text files (one token per line),
    # added to the training set as context-only tokens (masked-LM signal)
    extra_train_dir: str | None = None
    # use this existing vocab.json instead of building one — required when a
    # fine-tune must share embeddings with a pretraining run's vocabulary
    vocab_file: str | None = None
    limit_files: int = 0  # >0: cap files per split (smoke runs)
    max_word_len: int = 48
    max_lemma_len: int = 32
    chunk_len: int = 128
    halo: int = 16
    on_mismatch: str = "skip"  # "skip" | "error"
    # only useful with small chunk_len: at chunk_len=128 ~85% of chunks contain
    # a multi-item token, so upsampling them rebalances nothing
    multi_item_upsample: int = 1
    lemma_type_min_freq: int = 1  # train freq threshold for the lemma-classifier vocab
    word_type_min_freq: int = 2   # train freq threshold for the masked-LM word vocab
    mask_prob: float = 0.15       # fraction of tokens blanked for the masked-LM aux
    # train-time spelling augmentation: fraction of training tokens passed
    # through the learned diplomatic-noise sampler (input chars only; targets
    # untouched). Requires noise_rules. Within a chosen token the learned
    # identity mass still applies, so the changed-token rate is lower than this.
    spelling_noise: float = 0.0
    noise_rules: str | None = None  # path to meta/rem_layers/noise_rules.json
    # within a chosen token, scales the non-identity replacement mass (the
    # offline MHDBDB pipeline used 0.3). spelling_noise=1.0 with strength 0.3
    # reproduces that regime, resampled fresh every epoch: clean corpora fed
    # via extra_train_dir are then seen alternately clean and noised — the
    # normalised/diplomatic input swap
    spelling_noise_strength: float = 1.0
    # surface -> normalised-form lookup (tools/build_norm_lookup.py): the
    # masked-LM word-type targets are built over normalised forms, folding
    # spelling variants onto one class; a token's corpus-carried norm
    # (2-column unannotated files) takes precedence over the lookup
    norm_lookup: str | None = None
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.on_mismatch not in ("skip", "error"):
            raise ValueError(f"data.on_mismatch must be 'skip' or 'error', got {self.on_mismatch!r}")
        if self.split_mode not in ("file", "stratified", "chunk"):
            raise ValueError(f"data.split_mode must be file|stratified|chunk, got {self.split_mode!r}")
        if self.split_mode == "stratified" and not self.metadata_csv:
            raise ValueError("data.split_mode='stratified' requires data.metadata_csv")
        if self.spelling_noise > 0 and not self.noise_rules:
            raise ValueError("data.spelling_noise > 0 requires data.noise_rules")
        if self.corpus_dir is None and self.train_dir is None:
            raise ValueError("config must set data.corpus_dir or data.train_dir")
        if self.train_dir is not None and (self.dev_dir is None):
            raise ValueError("data.train_dir requires data.dev_dir")


@dataclass(frozen=True)
class ModelConfig:
    d_char: int = 64
    d_tok: int = 256
    d_model: int = 512
    n_slots: int = 8
    kernel_size: int = 3
    char_tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    ctx_tcn_dilations: tuple[int, ...] = (1, 2, 4)
    d_dec: int = 256
    lemma_cross_attention: bool = True
    lemma_classifier: bool = False   # classify-or-generate hybrid head
    lemma_decoder: str = "grid"      # "grid" (parallel) | "ar_tcn" (causal-TCN autoregressive)
    tcn_channel_gate: bool = False   # ECA-style channel gating in encoder TCN blocks
    ctx_self_attention: bool = False # one self-attention layer after the context TCN
    masked_lm: bool = False          # masked-token auxiliary head on the context encoder
    attention_pooling: bool = False  # learned-query pooling over char states (vs max-pool)
    joint_tag: bool = False          # auxiliary head over combined POS|morph tags
    pretrain_mlm: bool = False       # MLM-only mode: skip tagging heads in forward
    dropout: float = 0.15


@dataclass(frozen=True)
class LossConfig:
    w_pos: float = 1.0
    w_morph: float = 1.0
    w_lemma: float = 1.5
    w_lemma_cls: float = 1.0
    w_mlm: float = 0.5
    w_joint_tag: float = 0.5
    null_weight: float = 0.2
    # label smoothing for the tagging/classification heads (pos, morph, lemma
    # chars, lemma classifier, joint tag); the masked-LM aux stays unsmoothed —
    # its long-tail word-type distribution is diffuse enough already
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    batch_chunks: int = 16
    max_epochs: int = 60
    lr: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    amp: bool = True
    # "fp16" (GradScaler) or "bf16" (fp32-range exponent, no scaler — cures
    # activation overflows on large models; needs Ampere+)
    amp_dtype: str = "fp16"
    patience: int = 8
    min_delta: float = 0.0005
    log_every_steps: int = 50
    # LR multiplier for the large classification heads (lemma_cls/mlm/joint),
    # which train slower than the pretrained/warm encoder at big d_model
    cls_head_lr_mult: float = 1.0
    # exponential moving average of weights (0 = off; typical 0.999). Dev eval
    # and best.pt use the EMA weights; last.pt keeps raw weights + shadow.
    ema_decay: float = 0.0


@dataclass(frozen=True)
class DistributedConfig:
    backend: str = "auto"  # "auto" | "gloo" | "nccl"


@dataclass(frozen=True)
class InferConfig:
    batch_chunks: int = 32
    snap_lemmas: bool = True  # lexicon-constrained snapping of generated lemmas
    # classify-or-generate: use the classifier's lemma only when its softmax
    # probability clears this bar; otherwise trust the character generator.
    # 0.3 won the dev sweep {0.3, 0.5, 0.7, 0.9} (nearly flat to 0.5)
    classifier_min_prob: float = 0.3


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {f.name: f.type for f in fields(Config)}
_SECTION_CLASSES = {
    "run": RunConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "distributed": DistributedConfig,
    "infer": InferConfig,
}


def _build_section(cls: type, values: dict[str, Any], section: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, val in values.items():
        if key not in known:
            raise KeyError(f"unknown config key [{section}] {key}")
        # TOML arrays arrive as lists; tuple-typed fields want tuples
        if isinstance(val, list):
            val = tuple(val)
        kwargs[key] = val
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for section, values in raw.items():
        if section not in _SECTION_CLASSES:
            raise KeyError(f"unknown config section [{section}]")
        if not isinstance(values, dict):
            raise TypeError(f"config section [{section}] must be a table")
        kwargs[section] = _build_section(_SECTION_CLASSES[section], values, section)
    return Config(**kwargs)


def _parse_override_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # bare string


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply --set section.key=value overrides onto the raw TOML dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        section, key = parts
        raw.setdefault(section, {})[key] = _parse_override_value(value.strip())
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw)
