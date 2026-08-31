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
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.on_mismatch not in ("skip", "error"):
            raise ValueError(f"data.on_mismatch must be 'skip' or 'error', got {self.on_mismatch!r}")
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
    patience: int = 8
    min_delta: float = 0.0005
    log_every_steps: int = 50


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
