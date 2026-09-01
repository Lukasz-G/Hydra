"""Apply the learned diplomatic-noise model to one-token-per-line corpora.

Usage:
  python tools/apply_diplomatic_noise.py NOISE_RULES_JSON SRC_DIR OUT_DIR [strength]

Each token is rewritten character by character: multi-char rule contexts are
tried first (longest match), then single characters; the replacement is
sampled from the learned distribution. `strength` in [0,1] (default 1.0)
scales the probability of applying a non-identity replacement.
"""
import json
import random
import sys
from pathlib import Path

rules_path, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
rules = json.loads(rules_path.read_text(encoding="utf-8"))["rules"]
max_ctx = max((len(k) for k in rules if not k.startswith("+")), default=1)
rng = random.Random(20260901)


def noise_token(tok: str) -> str:
    s = tok.lower()
    outp = []
    i = 0
    while i < len(s):
        applied = False
        for width in range(min(max_ctx, len(s) - i), 0, -1):
            seg = s[i:i + width]
            dist = rules.get(seg)
            if dist is None:
                continue
            # sample from the distribution, damping non-identity mass by strength
            r = rng.random()
            acc = 0.0
            choice = seg
            for cand, p in dist.items():
                acc += p if cand == seg else p * strength
                if r <= acc:
                    choice = cand
                    break
            outp.append(choice)
            i += width
            applied = True
            break
        if not applied:
            outp.append(s[i])
            i += 1
    return "".join(outp)


out.mkdir(parents=True, exist_ok=True)
n_files = n_tok = n_changed = 0
for f in sorted(src.glob("*.txt")):
    toks = f.read_text(encoding="utf-8", errors="replace").split()
    noised = [noise_token(t) for t in toks]
    n_changed += sum(1 for a, b in zip(toks, noised) if a.lower() != b)
    n_tok += len(toks)
    (out / f.name).write_text("\n".join(noised) + "\n", encoding="utf-8")
    n_files += 1
print(f"{n_files} files, {n_tok} tokens, {n_changed} changed ({n_changed / max(1, n_tok):.1%})")
