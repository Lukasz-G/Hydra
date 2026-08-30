"""Lexicon-constrained lemma snapping.

Generated lemma items that are not in the train inventory get replaced by the
unique inventory lemma within edit distance 1, when one exists (SymSpell-style
deletion index). Measured on dev: ~24x more wins than losses.
"""
from __future__ import annotations

from collections import defaultdict

from .metrics import levenshtein


def _deletions(s: str):
    yield s
    for i in range(len(s)):
        yield s[:i] + s[i + 1:]


class LemmaSnapper:
    def __init__(self, inventory: set[str]):
        self.inventory = inventory
        self.index: dict[str, list[str]] = defaultdict(list)
        for lemma in inventory:
            for d in _deletions(lemma):
                self.index[d].append(lemma)

    def snap_item(self, item: str) -> str:
        if not item or item in self.inventory:
            return item
        candidates = set()
        for d in _deletions(item):
            for lemma in self.index.get(d, ()):
                if levenshtein(item, lemma) <= 1:
                    candidates.add(lemma)
                    if len(candidates) > 1:
                        return item  # ambiguous -> keep generation
        return next(iter(candidates)) if len(candidates) == 1 else item

    def snap(self, lemma: str, n_items: int) -> str:
        """Snap a '+'-joined lemma string with n_items items."""
        from .data import split_lemma_items
        items = split_lemma_items(lemma, n_items)
        return "+".join(self.snap_item(it) for it in items)
