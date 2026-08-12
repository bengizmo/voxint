"""Name-accuracy core: matching, verdicts, aggregates, statistics."""

import math

import pytest

from voxint.harness import name_accuracy as na


# --------------------------------------------------------------------------- #
# is_named / normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value", [None, "", "   ", 42, "speaker_00", "SPEAKER_03", "auto_17", "unknown", "Unknown 2"]
)
def test_is_named_rejects_blanks_and_placeholders(value: object) -> None:
    assert not na.is_named(value)


@pytest.mark.parametrize("value", ["Dana Fox", "josé garcía", "Ling Wei"])
def test_is_named_accepts_real_names(value: str) -> None:
    assert na.is_named(value)


# --------------------------------------------------------------------------- #
# person_name_match
# --------------------------------------------------------------------------- #
def test_id_equality_wins_even_with_different_names() -> None:
    m = na.person_name_match("Dana Fox", "D. Fox Jr.", id_a="spk-1", id_b="spk-1")
    assert m.matched and m.strength == na.STRENGTH_ID


def test_differing_ids_fall_through_to_name_comparison() -> None:
    m = na.person_name_match("Dana Fox", "Dana Fox", id_a="spk-1", id_b="spk-2")
    assert m.matched and m.strength == na.STRENGTH_EXACT


def test_alias_table_is_symmetric_and_includes_canonical() -> None:
    aliases = {"Daniela Fox": ["Dana Fox", "D. Fox"]}
    assert na.person_name_match("Dana Fox", "D. Fox", aliases=aliases).strength == na.STRENGTH_ALIAS
    assert (
        na.person_name_match("Daniela Fox", "Dana Fox", aliases=aliases).strength
        == na.STRENGTH_ALIAS
    )


def test_exact_match_normalizes_case_whitespace_and_unicode() -> None:
    # NFD-decomposed vs composed accent + case + internal whitespace.
    decomposed = "José  García"
    m = na.person_name_match(decomposed, "josé garcía")
    assert m.matched and m.strength == na.STRENGTH_EXACT


def test_surname_given_tolerates_middle_names_and_initials() -> None:
    m = na.person_name_match("Dana Fox", "Dana Marie Fox")
    assert m.matched and m.strength == na.STRENGTH_SURNAME_GIVEN
    m = na.person_name_match("D. Fox", "Dana Fox")
    assert m.matched and m.strength == na.STRENGTH_SURNAME_GIVEN


def test_multi_token_different_surname_is_none_not_weak() -> None:
    m = na.person_name_match("Dana Fox", "Dana Wolfe")
    assert not m.matched and m.strength == na.STRENGTH_NONE


def test_bare_first_name_is_weak_and_never_matches() -> None:
    m = na.person_name_match("Dana", "Dana Fox")
    assert not m.matched and m.strength == na.STRENGTH_WEAK
    m = na.person_name_match("Fox", "Dana Fox")
    assert not m.matched and m.strength == na.STRENGTH_WEAK


def test_unrelated_single_tokens_are_none() -> None:
    m = na.person_name_match("Dana", "Wolfe")
    assert not m.matched and m.strength == na.STRENGTH_NONE


@pytest.mark.parametrize("a,b", [(None, "Dana Fox"), ("", ""), ("Dana Fox", None)])
def test_blank_sides_are_none(a: str | None, b: str | None) -> None:
    assert na.person_name_match(a, b).strength == na.STRENGTH_NONE


# --------------------------------------------------------------------------- #
# Channels / hosts
# --------------------------------------------------------------------------- #
def test_norm_channel_collapses_punctuation_variants() -> None:
    assert na.norm_channel("Acme Audio") == na.norm_channel("Acme-Audio")
    assert na.norm_channel(None) == ""
    assert na.norm_channel("  ") == ""


def test_curated_hits_uses_strict_matching() -> None:
    hosts = ["Dana Fox", "Ling Wei"]
    # A bare first name in the item does NOT flag the curated host.
    assert na.curated_hits(["Dana"], hosts) == []
    assert na.curated_hits(["Dana Fox", "guest"], hosts) == ["Dana Fox"]


def test_is_overnaming_blank_channel_never_flags() -> None:
    assert not na.is_overnaming("Dana Fox", None, {"Dana Fox": ["Acme Audio"]})
    assert not na.is_overnaming("Dana Fox", "", {"Dana Fox": ["Acme Audio"]})


def test_is_overnaming_home_vs_off_channel() -> None:
    homes = {"Dana Fox": ["Acme Audio"]}
    assert not na.is_overnaming("Dana Fox", "acme-audio", homes)
    assert na.is_overnaming("Dana Fox", "Other Show", homes)
    # A host with no recorded home channel is off-channel on any real channel.
    assert na.is_overnaming("Ling Wei", "Acme Audio", homes)


# --------------------------------------------------------------------------- #
# slot_verdict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "assigned,truth,expected",
    [
        ("Dana Fox", "Dana Fox", na.TP),
        ("Dana Wolfe", "Dana Fox", na.FP_WRONG),
        ("Dana Fox", na.ABSTAIN, na.FP_OVERNAME),
        (None, "Dana Fox", na.FN),
        ("speaker_02", "Dana Fox", na.FN),
        (None, na.ABSTAIN, na.TN),
        ("Dana Fox", na.NEITHER_DETERMINABLE, na.EXCLUDED),
        ("Dana Fox", None, na.EXCLUDED),
    ],
)
def test_slot_verdict_matrix(assigned: str | None, truth: str | None, expected: str) -> None:
    assert na.slot_verdict(assigned, truth) == expected


def test_slot_verdict_honors_aliases() -> None:
    aliases = {"Daniela Fox": ["Dana Fox"]}
    assert na.slot_verdict("Dana Fox", "Daniela Fox", aliases=aliases) == na.TP


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_counts_and_prf1() -> None:
    agg = na.aggregate([na.TP, na.TP, na.FP_WRONG, na.FP_OVERNAME, na.FN, na.TN, na.EXCLUDED])
    assert (agg.tp, agg.fp_wrong, agg.fp_overname, agg.fn, agg.tn, agg.excluded) == (
        2, 1, 1, 1, 1, 1,
    )
    assert agg.precision == pytest.approx(2 / 4)
    assert agg.recall == pytest.approx(2 / 3)
    assert agg.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert agg.confusion["true_name"]["named_correct"] == 2
    assert agg.confusion["true_abstain"]["named"] == 1


def test_aggregate_duration_weighting() -> None:
    agg = na.aggregate([(na.TP, 90.0), (na.FN, 10.0), na.TN])
    assert agg.weighted_recall == pytest.approx(0.9)
    assert agg.recall == pytest.approx(0.5)


def test_aggregate_empty_is_zeroed_not_raising() -> None:
    agg = na.aggregate([])
    assert agg.precision == 0.0 and agg.recall == 0.0 and agg.f1 == 0.0


def test_aggregate_rejects_unknown_verdicts_and_bad_weights() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        na.aggregate(["BOGUS"])
    with pytest.raises(ValueError, match="weight"):
        na.aggregate([(na.TP, -1.0)])
    with pytest.raises(ValueError, match="weight"):
        na.aggregate([(na.TP, float("nan"))])


# --------------------------------------------------------------------------- #
# wilson_ci
# --------------------------------------------------------------------------- #
def test_wilson_no_information() -> None:
    assert na.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_known_value() -> None:
    lo, hi = na.wilson_ci(90, 100)
    assert 0.82 < lo < 0.87
    assert 0.93 < hi < 0.95
    assert lo < 0.9 < hi


def test_wilson_bounds_clamped() -> None:
    lo, hi = na.wilson_ci(0, 5)
    assert lo == 0.0
    lo, hi = na.wilson_ci(5, 5)
    assert hi == 1.0


def test_wilson_one_sided_z_is_tighter() -> None:
    _, hi_two = na.wilson_ci(2, 60)
    _, hi_one = na.wilson_ci(2, 60, z=1.645)
    assert hi_one < hi_two


def test_wilson_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="z must be"):
        na.wilson_ci(1, 2, z=0.0)
    with pytest.raises(ValueError, match="successes"):
        na.wilson_ci(3, 2)


# --------------------------------------------------------------------------- #
# mcnemar / bootstrap
# --------------------------------------------------------------------------- #
def test_mcnemar_no_discordant_pairs() -> None:
    res = na.mcnemar([True, False], [True, False])
    assert res.n_discordant == 0 and res.p_value == 1.0 and res.net == 0


def test_mcnemar_counts_regressions_and_fixes() -> None:
    baseline = [True, True, False, False, True]
    candidate = [False, True, True, True, True]
    res = na.mcnemar(baseline, candidate)
    assert res.baseline_correct_candidate_wrong == 1
    assert res.baseline_wrong_candidate_correct == 2
    assert res.net == 1
    assert 0.0 < res.p_value <= 1.0


def test_mcnemar_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        na.mcnemar([True], [True, False])


def test_bootstrap_empty_is_zeroed() -> None:
    res = na.clustered_bootstrap_delta([])
    assert (res.point, res.lo, res.hi) == (0.0, 0.0, 0.0)


def test_bootstrap_deterministic_and_contains_point() -> None:
    items = [
        [(True, True), (False, True)],
        [(True, False)],
        [(False, True), (True, True), (False, False)],
    ]
    a = na.clustered_bootstrap_delta(items, seed=7)
    b = na.clustered_bootstrap_delta(items, seed=7)
    assert a == b
    flat = [int(post) - int(pre) for cluster in items for pre, post in cluster]
    assert a.point == pytest.approx(sum(flat) / len(flat))
    assert a.lo <= a.hi


def test_bootstrap_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="ci"):
        na.clustered_bootstrap_delta([[(True, True)]], ci=1.0)
    with pytest.raises(ValueError, match="n_boot"):
        na.clustered_bootstrap_delta([[(True, True)]], n_boot=0)


# --------------------------------------------------------------------------- #
# confidence / risk-coverage
# --------------------------------------------------------------------------- #
def test_combine_confidences_filters_and_collapses() -> None:
    values: list[object] = [0.4, True, "x", float("nan"), 0.9, None]
    assert na.combine_confidences(values) == pytest.approx(0.9)
    assert na.combine_confidences(values, method="mean") == pytest.approx(0.65)
    assert na.combine_confidences([True, None]) is None
    with pytest.raises(ValueError, match="method"):
        na.combine_confidences([0.5], method="median")


def test_risk_coverage_empty() -> None:
    rc = na.risk_coverage([])
    assert rc.points == [] and rc.chow_coverage is None
    assert rc.descriptive and not rc.calibrated


def test_risk_coverage_orders_none_last_and_finds_chow() -> None:
    items: list[tuple[float | None, bool]] = [
        (0.9, True),
        (0.8, True),
        (0.5, False),
        (None, True),
    ]
    rc = na.risk_coverage(items, target_accuracy=0.75)
    coverages = [p[0] for p in rc.points]
    assert coverages == pytest.approx([0.25, 0.5, 0.75, 1.0])
    # None-confidence sorts last regardless of correctness.
    assert rc.points[-1][2] is None
    # Full coverage has accuracy 3/4 >= 0.75, so Chow point is 1.0.
    assert rc.chow_coverage == pytest.approx(1.0)
    assert math.isclose(rc.points[2][1], 2 / 3)
