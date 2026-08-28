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
    """Bundle of the three vocabularies plus the train surface set (for OOV eval)."""

    def __init__(self, chars: CharVocab, pos: LabelVocab, morph: LabelVocab,
                 train_surfaces: set[str]):
        self.chars = chars
        self.pos = pos
        self.morph = morph
        self.train_surfaces = train_surfaces

    @classmethod
    def build(cls, tokens: "list") -> "Vocabs":
        """Build from a list of data.Token from the train split (tagged tokens only)."""
        strings: set[str] = set()
        pos_labels: set[str] = set()
        morph_labels: set[str] = set()
        surfaces: set[str] = set()
        for tok in tokens:
            surfaces.add(tok.surface)
            strings.add(tok.surface)
            if tok.lemmas is None:
                continue
            strings.update(tok.lemmas)
            pos_labels.update(tok.pos)
            morph_labels.update(tok.morph)
        return cls(CharVocab.build(strings), LabelVocab.build(pos_labels),
                   LabelVocab.build(morph_labels), surfaces)

    def save(self, path: str | Path) -> None:
        payload = {
            "chars": self.chars.itos[len(CharVocab.specials):],
            "pos": self.pos.itos[len(LabelVocab.specials):],
            "morph": self.morph.itos[len(LabelVocab.specials):],
            "train_surfaces": sorted(self.train_surfaces),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabs":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CharVocab(payload["chars"]), LabelVocab(payload["pos"]),
                   LabelVocab(payload["morph"]), set(payload["train_surfaces"]))
