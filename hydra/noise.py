"""Learned diplomatic-spelling noise, applied to surface strings.

The rule table (meta/rem_layers/noise_rules.json) holds per-context character
replacement distributions learned from the aligned diplomatic/normalised
layers of ReM. `noise()` rewrites a token character by character: multi-char
contexts are tried longest-first, the replacement is sampled from the learned
distribution with the non-identity mass scaled by `strength`. Only lowercase
segments are eligible, so casing information survives.

Shared by the offline corpus tool (tools/apply_diplomatic_noise.py) and the
train-time augmentation in HydraDataset (data.spelling_noise).
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class SpellingNoiser:
    def __init__(self, rules_path: str | Path, strength: float = 1.0,
                 seed: int = 20260901):
        self.rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))["rules"]
        self.max_ctx = max((len(k) for k in self.rules if not k.startswith("+")), default=1)
        self.strength = strength
        self.rng = random.Random(seed)

    def noise(self, tok: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(tok):
            applied = False
            for width in range(min(self.max_ctx, len(tok) - i), 0, -1):
                seg = tok[i:i + width]
                if seg != seg.lower():
                    continue  # never rewrite segments carrying case information
                dist = self.rules.get(seg)
                if dist is None:
                    continue
                r = self.rng.random()
                acc = 0.0
                choice = seg
                for cand, p in dist.items():
                    acc += p if cand == seg else p * self.strength
                    if r <= acc:
                        choice = cand
                        break
                out.append(choice)
                i += width
                applied = True
                break
            if not applied:
                out.append(tok[i])
                i += 1
        return "".join(out)
