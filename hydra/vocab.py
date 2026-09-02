"""Character and label vocabularies with deterministic ordering and JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path

PAD, UNK, EOW = 0, 1, 2  # CharVocab specials
NULL = 0                  # LabelVocab: class 0 marks an unused slot
LABEL_UNK = 1


class CharVocab:
    specials = ("<PAD>", "<UNK>", "<EOW>")

    def __init__(self, chars: list[str]):
        self.itos = list(self.specials) + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, strings: "list[str] | set[str]") -> "CharVocab":
        charset: set[str] = set()
        for s in strings:
            charset.update(s)
        charset -= set(cls.specials)
        return cls(sorted(charset))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        unk = UNK
        return [self.stoi.get(c, unk) for c in s]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == EOW:
                break
            if i == PAD:
                continue
            out.append(self.itos[i] if i != UNK else "�")
        return "".join(out)


class LabelVocab:
    specials = ("<NULL>", "<UNK>")

    def __init__(self, labels: list[str]):
        self.itos = list(self.specials) + labels
        self.stoi = {l: i for i, l in enumerate(self.itos)}

    @classmethod
    def build(cls, labels: "list[str] | set[str]") -> "LabelVocab":
        return cls(sorted(set(labels) - set(cls.specials)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, label: str) -> int:
        return self.stoi.get(label, LABEL_UNK)

    def decode(self, idx: int) -> str:
        return self.itos[idx]


class Vocabs:
    """Bundle of the vocabularies plus train-side inventories.

    train_surfaces: surface forms seen in train (OOV evaluation).
    lemma_counts: atomic lemma -> train frequency (lexicon snapping and the
    classify-or-generate head). lemma_types: LabelVocab over atomic lemmas
    with count >= lemma_type_min_freq (class UNK = generate instead).
    """

    def __init__(self, chars: CharVocab, pos: LabelVocab, morph: LabelVocab,
                 train_surfaces: set[str], lemma_counts: dict[str, int] | None = None,
                 lemma_type_min_freq: int = 1,
                 surface_counts: dict[str, int] | None = None,
                 word_type_min_freq: int = 2,
                 joint_counts: dict[str, int] | None = None):
        self.chars = chars
        self.pos = pos
        self.morph = morph
        self.train_surfaces = train_surfaces
        self.lemma_counts = lemma_counts or {}
        self.lemma_type_min_freq = lemma_type_min_freq
        self.lemma_types = LabelVocab.build(
            [l for l, c in self.lemma_counts.items() if c >= lemma_type_min_freq])
        # word types for the masked-token auxiliary objective (UNK = rare)
        self.surface_counts = surface_counts or {}
        self.word_type_min_freq = word_type_min_freq
        self.word_types = LabelVocab.build(
            [w for w, c in self.surface_counts.items() if c >= word_type_min_freq])
        # combined POS|morph tags for the joint-tag auxiliary head
        self.joint_counts = joint_counts or {}
        self.joint_types = LabelVocab.build(list(self.joint_counts))

    @property
    def lemma_inventory(self) -> set[str]:
        return set(self.lemma_counts)

    @classmethod
    def build(cls, tokens: "list", lemma_type_min_freq: int = 1,
              word_type_min_freq: int = 2,
              norm_of: "dict[str, str] | None" = None) -> "Vocabs":
        """Build from a list of data.Token from the train split (tagged tokens only).

        norm_of: surface -> normalised form; when given, the masked-LM word-type
        vocabulary is built over normalised forms (a token's own .norm wins),
        folding spelling variants onto one class. Everything else — chars,
        train_surfaces, OOV bookkeeping — stays surface-level."""
        strings: set[str] = set()
        pos_labels: set[str] = set()
        morph_labels: set[str] = set()
        surfaces: set[str] = set()
        lemma_counts: dict[str, int] = {}
        surface_counts: dict[str, int] = {}
        joint_counts: dict[str, int] = {}
        for tok in tokens:
            surfaces.add(tok.surface)
            strings.add(tok.surface)
            wkey = getattr(tok, "norm", None)
            if wkey is None:
                wkey = norm_of.get(tok.surface, tok.surface) if norm_of else tok.surface
            surface_counts[wkey] = surface_counts.get(wkey, 0) + 1
            if tok.lemmas is None:
                continue
            strings.update(tok.lemmas)
            pos_labels.update(tok.pos)
            morph_labels.update(tok.morph)
            for l in tok.lemmas:
                lemma_counts[l] = lemma_counts.get(l, 0) + 1
            for p, m in zip(tok.pos, tok.morph):
                j = f"{p}|{m}"
                joint_counts[j] = joint_counts.get(j, 0) + 1
        return cls(CharVocab.build(strings), LabelVocab.build(pos_labels),
                   LabelVocab.build(morph_labels), surfaces, lemma_counts,
                   lemma_type_min_freq, surface_counts, word_type_min_freq,
                   joint_counts)

    def save(self, path: str | Path) -> None:
        payload = {
            "chars": self.chars.itos[len(CharVocab.specials):],
            "pos": self.pos.itos[len(LabelVocab.specials):],
            "morph": self.morph.itos[len(LabelVocab.specials):],
            "train_surfaces": sorted(self.train_surfaces),
            "lemma_counts": dict(sorted(self.lemma_counts.items())),
            "lemma_type_min_freq": self.lemma_type_min_freq,
            "surface_counts": dict(sorted(self.surface_counts.items())),
            "word_type_min_freq": self.word_type_min_freq,
            "joint_counts": dict(sorted(self.joint_counts.items())),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabs":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CharVocab(payload["chars"]), LabelVocab(payload["pos"]),
                   LabelVocab(payload["morph"]), set(payload["train_surfaces"]),
                   payload.get("lemma_counts"), payload.get("lemma_type_min_freq", 1),
                   payload.get("surface_counts"), payload.get("word_type_min_freq", 2),
                   payload.get("joint_counts"))
