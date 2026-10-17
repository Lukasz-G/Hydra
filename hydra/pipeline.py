"""Hydra as a library: load a model once, call it on text.

    import hydra

    nlp = hydra.load("de-4corpus")          # or a path to a checkpoint
    doc = nlp("Was daz mere vil lip")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.morph_)

The shape is spaCy's, because that is the shape people already know: a
callable that returns a `Doc`, a `Doc` that iterates over `Token`s, and a
`.pipe()` for many texts at once. What it is not is a drop-in replacement --
there is no dependency parse, no named entities, no `Vocab` object -- and the
fields that do exist mean what this project means by them.

Two differences from spaCy are worth knowing before you write a loop:

* Hydra does not tokenise for you in any clever way. The default splits on
  whitespace, because a medieval corpus has its own conventions and guessing
  at them silently is worse than leaving the decision with you. Pass your own
  callable as `tokenizer=`, or hand `nlp.pipe_tokens()` a list of tokens you
  have already split.

* One written token can carry several annotated items -- a preposition fused
  with its article, a verb with an enclitic pronoun (see the paper, §2.2). The
  string fields join those with `+`, exactly as the corpora write them, and
  `token.items` gives them back as a list of `(lemma, pos, morph)` triples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Token:
    """One written token and the annotation Hydra gives it."""

    text: str
    lemma_: str
    pos_: str
    morph_: str
    i: int = 0
    # How much to trust this row. `pos_conf` and `lemma_conf` are softmax
    # probabilities; `lemma_generated` says the lemma was spelled out by the
    # character decoder because the classifier was not confident enough, and
    # `is_unseen` that this written form occurs nowhere in the training data.
    # The paper (§6.3, §8.1) finds the last of these the strongest single
    # predictor of an error.
    pos_conf: float = 1.0
    lemma_conf: float = 1.0
    lemma_generated: bool = False
    is_unseen: bool = False

    @property
    def is_uncertain(self) -> bool:
        """A row worth a human glance: unseen spelling, or a low-confidence tag."""
        return self.is_unseen or self.pos_conf < 0.5 or self.lemma_generated

    @property
    def items(self) -> list[tuple[str, str, str]]:
        """The annotated units inside this token, usually one of them.

        `zemerſten` is written once and analysed as three: a preposition, an
        article and an adjective. Each list entry is (lemma, pos, morph).
        """
        lemmas = self.lemma_.split("+")
        tags = self.pos_.split("+")
        morphs = self.morph_.split("+")
        n = max(len(lemmas), len(tags), len(morphs))

        def at(seq: list[str], k: int) -> str:
            return seq[k] if k < len(seq) else ""

        return [(at(lemmas, k), at(tags, k), at(morphs, k)) for k in range(n)]

    @property
    def is_multi(self) -> bool:
        return "+" in self.pos_

    def __repr__(self) -> str:
        return f"Token({self.text!r}, {self.lemma_!r}, {self.pos_!r}, {self.morph_!r})"


class Doc:
    """A tagged text: iterate it for tokens, or dump it as TSV."""

    def __init__(self, tokens: Sequence[Token], text: str | None = None):
        self.tokens = list(tokens)
        self.text = text if text is not None else " ".join(t.text for t in self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Doc(self.tokens[i])
        return self.tokens[i]

    def __repr__(self) -> str:
        head = " ".join(t.text for t in self.tokens[:8])
        return f"Doc({len(self.tokens)} tokens: {head}{' …' if len(self.tokens) > 8 else ''})"

    @property
    def lemmas(self) -> list[str]:
        return [t.lemma_ for t in self.tokens]

    @property
    def tags(self) -> list[str]:
        return [t.pos_ for t in self.tokens]

    def to_tsv(self) -> str:
        """The four columns the corpora and `hydra-tag` use."""
        return "".join(f"{t.text}\t{t.lemma_}\t{t.pos_}\t{t.morph_}\n" for t in self.tokens)

    def to_dicts(self, confidence: bool = False) -> list[dict]:
        out = [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_, "morph": t.morph_}
               for t in self.tokens]
        if confidence:
            for d, t in zip(out, self.tokens):
                d.update(pos_conf=round(t.pos_conf, 4),
                         lemma_conf=round(t.lemma_conf, 4),
                         lemma_generated=t.lemma_generated, unseen=t.is_unseen)
        return out

    @property
    def unseen_rate(self) -> float:
        """Share of tokens whose written form occurs nowhere in training.

        This is the number to look at before trusting the rest. Across the
        corpora of the paper it is the strongest single predictor of how a
        text will be tagged, and it can be read off a new manuscript without
        any gold annotation: see §4.2 and §8.1.
        """
        if not self.tokens:
            return 0.0
        return sum(t.is_unseen for t in self.tokens) / len(self.tokens)

    def uncertain(self, pos_conf: float = 0.5) -> "list[Token]":
        """Rows worth a human glance, for a scholar checking a concordance."""
        return [t for t in self.tokens
                if t.is_unseen or t.lemma_generated or t.pos_conf < pos_conf]


class Hydra:
    """A loaded model. Build it with `hydra.load()` rather than directly."""

    def __init__(self, model, vocabs, cfg, device, snapper=None,
                 tokenizer: Callable[[str], list[str]] | None = None,
                 name: str = ""):
        self._model = model
        self._vocabs = vocabs
        self._cfg = cfg
        self._device = device
        self._snapper = snapper
        self.tokenizer = tokenizer or (lambda s: s.split())
        self.name = name

    # -- the two ways in ---------------------------------------------------

    def __call__(self, text: str) -> Doc:
        """Tag one string. The text is split by `self.tokenizer`."""
        surfaces = self.tokenizer(text)
        return Doc(self._tag(surfaces), text=text)

    def pipe(self, texts: Iterable[str]) -> Iterator[Doc]:
        """Tag many strings, yielding a Doc each, in the order given.

        There is no `batch_size` here, unlike spaCy's `pipe`. Batching happens
        inside a document, over its 128-token chunks (`cfg.infer.batch_chunks`),
        and documents are not batched together because a chunk reads context
        from its neighbours -- running two texts through as one batch would let
        the end of one supply context to the start of the next.
        """
        for text in texts:
            yield self(text)

    def pipe_tokens(self, docs: Iterable[Sequence[str]]) -> Iterator[Doc]:
        """Tag documents you have tokenised yourself."""
        for surfaces in docs:
            yield Doc(self._tag(list(surfaces)))

    def tag_file(self, path: str | Path, fmt: str = "auto") -> Doc:
        """Tag a file on disk: one token per line, or running text."""
        from .tag import read_input_file

        entries = read_input_file(Path(path), fmt)
        surfaces = [text for kind, text in entries if kind == "token"]
        return Doc(self._tag(surfaces))

    # -- the work ----------------------------------------------------------

    def _tag(self, surfaces: list[str]) -> list[Token]:
        from .tag import tag_document

        if not surfaces:
            return []
        preds = tag_document(self._model, self._vocabs, self._cfg, surfaces,
                             self._device, with_confidence=True)
        known = getattr(self._vocabs, "train_surfaces", None) or set()
        out = []
        for i, (surface, p) in enumerate(zip(surfaces, preds)):
            lemma = p.lemma
            if self._snapper is not None:
                lemma = self._snapper.snap(lemma, len(p.pos.split("+")))
            out.append(Token(surface, lemma, p.pos, p.morph, i,
                             pos_conf=p.pos_conf, lemma_conf=p.lemma_conf,
                             lemma_generated=p.lemma_generated,
                             is_unseen=surface not in known))
        return out

    @property
    def meta(self) -> dict:
        return {"name": self.name, "device": str(self._device),
                "slots": self._cfg.model.n_slots,
                "pos_tags": len(self._vocabs.pos),
                "morph_tags": len(self._vocabs.morph),
                "lemma_inventory": len(self._vocabs.lemma_types),
                "known_surfaces": len(getattr(self._vocabs, "train_surfaces", ()) or ()),
                "classifier_min_prob": self._cfg.infer.classifier_min_prob}

    def __repr__(self) -> str:
        return f"Hydra({self.name or 'model'}, device={self._device})"


def load(model: str | Path = "de-4corpus", device: str | None = None,
         tokenizer: Callable[[str], list[str]] | None = None) -> Hydra:
    """Load a published model by name, or a checkpoint by path.

    `device` takes anything torch accepts ("cpu", "cuda:0"); the default picks
    a GPU when one is visible. A published name is downloaded once and cached
    (see `hydra.hub`), so the first call may take a minute.
    """
    import torch

    from .hub import resolve
    from .tag import load_model_for_inference

    dev = torch.device(device) if device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    path = resolve(str(model))
    net, vocabs, cfg = load_model_for_inference(path, dev)

    snapper = None
    if cfg.infer.snap_lemmas and vocabs.lemma_counts:
        from .snap import LemmaSnapper
        snapper = LemmaSnapper(vocabs.lemma_inventory)

    return Hydra(net, vocabs, cfg, dev, snapper, tokenizer, name=str(model))
