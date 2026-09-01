"""Extract aligned (normalized, annotated-diplomatic) token pairs from the
ReM CorA-XML export and learn a character-level 'diplomatic noise' model.

Usage:
  python tools/extract_rem_layers.py CORA_XML_DIR OUT_DIR

Writes to OUT_DIR:
  pairs.tsv        norm<TAB>anno_utf<TAB>count   (aggregated over all texts)
  noise_rules.json {"rules": {norm_seg: {anno_seg: prob, ...}}, "stats": ...}

The rules capture how normalized MHG orthography surfaces in (annotated)
diplomatic spelling; tools/apply_diplomatic_noise.py uses them to make
normalized corpora (e.g. MHDBDB) look diplomatic for pretraining.
"""
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import xml.etree.ElementTree as ET

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

pair_counts: Counter = Counter()
for xml_file in sorted(src.glob("*.xml")):
    for event, elem in ET.iterparse(xml_file, events=("end",)):
        if elem.tag == "token":
            anno = elem.find("tok_anno")
            if anno is not None:
                norm = anno.find("norm")
                if norm is not None and norm.get("tag") and anno.get("utf"):
                    pair_counts[(norm.get("tag"), anno.get("utf"))] += 1
            elem.clear()

with open(out / "pairs.tsv", "w", encoding="utf-8") as fh:
    for (n, a), c in pair_counts.most_common():
        fh.write(f"{n}\t{a}\t{c}\n")
print(f"{sum(pair_counts.values())} tokens, {len(pair_counts)} distinct pairs")

# character-level edit rules norm -> anno, weighted by pair frequency
rule_counts: dict = defaultdict(Counter)
context_totals: Counter = Counter()
for (n, a), c in pair_counts.items():
    n_l, a_l = n.lower(), a.lower()
    sm = SequenceMatcher(None, n_l, a_l, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg_n, seg_a = n_l[i1:i2], a_l[j1:j2]
        if len(seg_n) > 3 or len(seg_a) > 4:
            continue  # skip long unalignable stretches
        if op == "equal":
            for ch in seg_n:
                rule_counts[ch][ch] += c
        else:
            key = seg_n if seg_n else f"+{n_l[max(0, i1 - 1):i1]}"  # insertion after char
            rule_counts[key][seg_a] += c

rules = {}
for seg, outs in rule_counts.items():
    total = sum(outs.values())
    if total < 20:
        continue
    dist = {o: cnt / total for o, cnt in outs.most_common(8) if cnt / total >= 0.002}
    if dist:
        rules[seg] = dist

(out / "noise_rules.json").write_text(
    json.dumps({"rules": rules}, ensure_ascii=False, indent=1), encoding="utf-8")
n_stochastic = sum(1 for s, d in rules.items() if d.get(s, 0) < 0.995)
print(f"{len(rules)} rule contexts, {n_stochastic} with real variation -> noise_rules.json")
