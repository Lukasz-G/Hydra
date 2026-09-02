"""Build the surface -> normalised-form lookup from the aligned ReM layers.

Usage:
  python tools/build_norm_lookup.py meta/rem_layers/pairs.tsv meta/norm_lookup.tsv

pairs.tsv rows are `norm<TAB>surface<TAB>count` aggregated over the corpus.
For every surface form the most frequent normalised form wins; identity
mappings are omitted (the runtime default is the surface itself), so the
lookup holds exactly the folding information: which spellings collapse onto
which normalised type for the masked-LM target vocabulary.
"""
import sys
from collections import defaultdict
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
by_surface: dict[str, dict[str, int]] = defaultdict(dict)
with open(src, encoding="utf-8") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        norm, surface, count = parts[0], parts[1], int(parts[2])
        by_surface[surface][norm] = by_surface[surface].get(norm, 0) + count

n_identity = 0
rows = []
for surface in sorted(by_surface):
    best = max(by_surface[surface].items(), key=lambda kv: (kv[1], kv[0] == surface))
    if best[0] == surface:
        n_identity += 1
    else:
        rows.append(f"{surface}\t{best[0]}")
dst.write_text("\n".join(rows) + "\n", encoding="utf-8")

n_surfaces = len(by_surface)
n_norms = len({max(d.items(), key=lambda kv: kv[1])[0] for d in by_surface.values()})
print(f"{n_surfaces} surface types -> {n_norms} normalised types; "
      f"{len(rows)} non-identity mappings written ({n_identity} identities omitted)")
