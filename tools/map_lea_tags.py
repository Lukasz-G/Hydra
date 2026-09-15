"""Map LeA's DDDTS-as-German part-of-speech labels onto ReM's HiTS tagset.

Usage:
  python tools/map_lea_tags.py IN_DIR OUT_DIR [--audit]

The three tiers, and why the split falls where it does. Measured shares are of
LeA's annotated (Germanic-vernacular) tokens.

  TIER 1  direct 1:1                          ~53%
      LeA's label and a HiTS tag denote the same category. Most of these are
      attested inside LeA itself: about 0.9% of the corpus is annotated in
      HiTS-ish tags rather than German labels (a second annotation campaign
      inside ReA), and lemmas tagged both ways give 'Subst.'->NA at weight
      709, 'Pers.-Pron.'->PPER at 203, and so on.

  TIER 2  recoverable from LeA's own morph    ~20%
      HiTS splits verbs by finiteness; LeA does not, but its morph column
      does: '3. Sg. Prät. Ind.' is finite, 'Inf.' an infinitive, 'Part. Perf.'
      a participle, 'Imp.' an imperative. Resolves 97.8% of 'Verb'.

  TIER 3  routed, not mapped                  ~26%
      ADJA/ADJD, DDART/DDS/DDA, DPOSA/DPOSS, DIA/DIART/PI are distinctions of
      SYNTACTIC FUNCTION that the Old German annotation simply does not make.
      Nothing recovers them, so these keep an 'OHG:' prefix and live in their
      own label space. Forcing them onto the most frequent HiTS variant would
      buy full sharing with systematic, silent error.

Conjunctions are tier 1 via a lemma lookup rather than a label rule: ReM tags
conjunctions almost deterministically by lemma ('unte'->KON 100%, 'dazz'->
KOUS 100%), but its lemmas are Middle High German and LeA's are Old High
German and Old Saxon, so the list below is a cognate mapping and NOT derived
from ReM. It is the author's philological judgement and wants an expert eye;
everything it does not cover falls back to the routed 'OHG:Konj.'.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROUTE = "OHG:"

# --- tier 1: LeA label -> HiTS tag ------------------------------------------
DIRECT = {
    "Subst.": "NA",
    "EN": "NE",
    "Pers.-Pron.": "PPER",
    "Refl.-Pron.": "PRF",
    "Adv.": "AVD",
    "Präp.": "APPR",
    "Partikel": "PTK",
    "Itj.": "ITJ",
    "Int.-Pron.": "PW",
    "Kard.": "CARDA",
    "Ord.": "ORDA",
    "PTKVZ": "PTKVZ",
    "PTKNEG": "PTKNEG",
}

# --- tier 1b: LeA's OWN HiTS-tagged portion --------------------------------
# About 0.9% of LeA carries HiTS-ish tags instead of German labels (a second
# annotation campaign inside ReA). They are not ReM's variant, though: LeA
# writes AP/KO/ADV where ReM writes APPR/KON/AVD, so they need a rename, not a
# pass-through. Tags whose HiTS equivalent needs a distinction Old German does
# not make (ADJ->ADJA/ADJD, DD->DDART/DDS/DDA, DI, DPOS) stay routed.
HITS_VARIANT = {
    "NA": "NA", "NE": "NE", "PPER": "PPER", "PRF": "PRF", "PTK": "PTK",
    "PTKNEG": "PTKNEG", "PTKVZ": "PTKVZ", "PW": "PW", "PWG": "PWG",
    "VVPP": "VVPP", "ITJ": "ITJ",
    "AP": "APPR", "ADV": "AVD", "CARD": "CARDA", "ORD": "ORDA",
}
# verb-like HiTS tags still need the finiteness split from morph
HITS_VERBAL = {"VV": "VV", "VA": "VA", "VM": "VM"}

# --- tier 2: verb-like labels split by the morph column ----------------------
VERBAL = {"Verb": "VV", "Hilfsverb": "VA", "Modalverb": "VM"}


def verb_suffix(morph: str) -> str | None:
    """HiTS verb subcategory from LeA's inflection string, or None."""
    if "Inf" in morph:
        return "INF"
    if "Part" in morph:
        return "PP"
    if "Imp" in morph:
        return "IMP"
    if re.search(r"\b[123]\.", morph) or "Ind" in morph or "Konj" in morph:
        return "FIN"
    return None


# --- tier 1 (lemma-based): conjunctions -------------------------------------
# KON  coordinating | KOUS subordinating | KOKOM comparative
CONJ = {
    # coordinating
    "inti": "KON", "endi": "KON", "joh": "KON", "noh": "KON", "ëdo": "KON",
    "alde": "KON", "afar": "KON", "ouh": "KON", "ak": "KON", "suntar": "KON",
    "odo": "KON", "oda": "KON", "nibu": "KOUS", "ni": "KON",
    # subordinating
    "daʒ": "KOUS", "that": "KOUS", "daz": "KOUS", "ibu": "KOUS", "oba": "KOUS",
    "wanta": "KOUS", "hwand": "KOUS", "hwanda": "KOUS", "bithiu": "KOUS",
    "dō": "KOUS", "mit diu": "KOUS", "unz": "KOUS", "unzi": "KOUS",
    "ēr": "KOUS", "after thiu": "KOUS", "sīd": "KOUS", "sīdor": "KOUS",
    # comparative / manner
    "sō": "KOUS", "sōsō": "KOUS", "alsō": "KOUS", "danne": "KOKOM",
    "thanne": "KOKOM", "wio": "KOUS", "hwō": "KOUS",
    # second pass, from the lemmas the first list left routed
    "doh": "KON", "thō̆h": "KON", "oh": "KON", "nalles": "KON",
    "enti": "KON", "eftha": "KON", "ānu": "KON", "ūʒan": "KON", "ne": "KON",
    "nī̆": "KON", "min": "KON",
    "ef": "KOUS", "than": "KOUS", "thō": "KOUS", "antthat": "KOUS",
    "bīdiu": "KOUS", "bīdiu wanta": "KOUS", "danta": "KOUS", "dār": "KOUS",
    "sama sō": "KOUS", "all sō": "KOUS", "ēr danne": "KOKOM",
}


def map_tag(pos: str, morph: str, lemma: str) -> str:
    if pos in DIRECT:
        return DIRECT[pos]
    if pos in HITS_VARIANT:
        return HITS_VARIANT[pos]
    if pos in HITS_VERBAL:
        suf = verb_suffix(morph)
        return HITS_VERBAL[pos] + suf if suf else ROUTE + pos
    if pos in VERBAL:
        suf = verb_suffix(morph)
        return VERBAL[pos] + suf if suf else ROUTE + pos
    if pos == "Konj.":
        return CONJ.get(lemma.strip(), ROUTE + pos)
    return ROUTE + pos          # tier 3, and anything unforeseen


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    audit = "--audit" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    tier = Counter()
    routed = Counter()
    for f in sorted(src.glob("*.txt")):
        lines = []
        for ln in f.read_text(encoding="utf-8").splitlines():
            c = ln.split("\t")
            if len(c) >= 4:
                mapped = "+".join(map_tag(p, m, l) for p, m, l in
                                  zip(c[2].split("+"), c[3].split("+") * 9, c[1].split("+") * 9))
                for t in mapped.split("+"):
                    tier["routed" if t.startswith(ROUTE) else "shared"] += 1
                    if t.startswith(ROUTE):
                        routed[t] += 1
                c[2] = mapped
            lines.append("\t".join(c))
        (out / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    tot = sum(tier.values())
    print(f"{tot:,} tag items: {tier['shared']:,} mapped to HiTS "
          f"({100 * tier['shared'] / tot:.1f}%), {tier['routed']:,} routed "
          f"({100 * tier['routed'] / tot:.1f}%)")
    if audit:
        print("\nrouted labels, by frequency:")
        for t, n in routed.most_common(20):
            print(f"  {t:28s} {n:7,} {100 * n / tot:5.2f}%")


if __name__ == "__main__":
    main()
