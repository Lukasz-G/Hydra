"""data.combined_tags: the paper's SS6.2 ablation.

Treat each token as ONE item with a combined tag ("APPR+NA") and a joined
lemma ("in+hant") instead of splitting on '+'. That is RNNTagger's convention,
and it beats the K=8 slot decoder by ~4pp on multi-item tokens -- the baseline
the slot architecture has never actually been measured against.
"""
from hydra.data import parse_tsv_file


def write(tmp_path, lines):
    p = tmp_path / "doc.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


LINES = [
    "@ comment",
    "inhandon\tin+hant\tAPPR+NA\tc.D+Dat.Pl",
    "der\tder\tDDART\tNom.Sg.Masc",
]


def test_split_mode_is_the_default(tmp_path):
    toks, _ = parse_tsv_file(write(tmp_path, LINES), "skip", 8)
    assert [t.surface for t in toks] == ["inhandon", "der"]
    assert toks[0].pos == ["APPR", "NA"]
    assert toks[0].lemmas == ["in", "hant"]
    assert toks[0].morph == ["c.D", "Dat.Pl"]
    assert toks[0].n_items == 2


def test_combined_tags_keeps_one_item(tmp_path):
    toks, _ = parse_tsv_file(write(tmp_path, LINES), "skip", 8, combined_tags=True)
    assert [t.surface for t in toks] == ["inhandon", "der"]
    # the whole column becomes a single item, '+' and all
    assert toks[0].pos == ["APPR+NA"]
    assert toks[0].lemmas == ["in+hant"]
    assert toks[0].morph == ["c.D+Dat.Pl"]
    assert toks[0].n_items == 1
    # single-item tokens are untouched by the flag
    assert toks[1].pos == ["DDART"] and toks[1].n_items == 1


def test_both_arms_see_identical_token_sets(tmp_path):
    """The ablation must skip exactly what the slot path skips.

    Joining is always well-formed, so a naive implementation keeps the
    malformed tokens the slot path drops -- handing the ablation extra
    supervision AND a different-sized test set, which would make any accuracy
    difference uninterpretable.
    """
    lines = LINES + [
        "weird	a+b+c	X+Y	M",          # misaligned counts
        "big	" + "+".join("l%d" % i for i in range(9)) + "	"
                + "+".join("P%d" % i for i in range(9)) + "	"
                + "+".join("M%d" % i for i in range(9)),   # 9 items > n_slots
    ]
    split_toks, split_skipped = parse_tsv_file(write(tmp_path, lines), "skip", 8)
    comb_toks, comb_skipped = parse_tsv_file(write(tmp_path, lines), "skip", 8,
                                             combined_tags=True)
    assert split_skipped == 2, "fixture no longer exercises the skip path"
    assert comb_skipped == split_skipped
    assert len(split_toks) == len(comb_toks)
    # identical supervision mask, token for token
    assert ([t.pos is None for t in split_toks]
            == [t.pos is None for t in comb_toks])


def test_ablation_with_one_slot_keeps_multi_item_tokens(tmp_path):
    """The bug that invalidated the first ablation run.

    The ablation sets model.n_slots=1 (that IS the ablation: no slots). But the
    malformed-token check rejects n > n_slots, so with n_slots=1 every
    multi-item token was skipped -- silently removing exactly the tokens the
    ablation exists to measure. It surfaced only as a dev-set size: n=91,726
    against the baseline's 94,259, short by precisely the 2,533 multi-item
    tokens. Nothing else complained.

    data.align_max_items pins the limit to the BASELINE's slot count, so both
    arms skip the same tokens regardless of how many slots the model has.
    """
    lines = LINES + ["dreistueck\ta+b+c\tX+Y+Z\tM1+M2+M3"]   # a 3-item token

    # what the slot baseline (n_slots=8) sees
    base, base_skipped = parse_tsv_file(write(tmp_path, lines), "skip", 8)

    # the ablation as configured: one slot, but the limit pinned to the baseline
    abl, abl_skipped = parse_tsv_file(write(tmp_path, lines), "skip", 1,
                                      combined_tags=True, max_items=8)
    assert abl_skipped == base_skipped == 0
    assert len(abl) == len(base)
    assert all(t.pos is not None for t in abl)
    assert all(t.n_items == 1 for t in abl), "combined tags must yield one item"

    # ...and the regression itself: without the pin, n_slots=1 eats them
    bad, bad_skipped = parse_tsv_file(write(tmp_path, lines), "skip", 1,
                                      combined_tags=True)
    assert bad_skipped == 2, "fixture no longer reproduces the bug"
    assert sum(t.pos is not None for t in bad) == len(base) - 2
