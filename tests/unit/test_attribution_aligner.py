import pytest

from voxint.harness.attribution_aligner import (
    AlignmentReport,
    AttributionTrial,
    Interval,
    SlotAlignment,
    SlotClassification,
    aggregate_trials,
    align_slots,
    build_overlap_matrix,
    build_trials,
    overlap_duration,
    total_duration,
)
from voxint.harness.calibration import Trial, TrialKind
from voxint.harness.name_accuracy import wilson_ci
from voxint.speakers.matching import MatchingGates


def _alignment(
    label: str = "slot-1",
    classification: SlotClassification = SlotClassification.GENUINE,
    gold: str | None = "Alice",
) -> SlotAlignment:
    return SlotAlignment(
        slot_label=label,
        classification=classification,
        dominant_gold_speaker=gold,
        purity=0.95,
        coverage=0.75,
        margin=0.90,
        slot_duration=12.0,
    )


def _report(*alignments: SlotAlignment) -> AlignmentReport:
    return AlignmentReport(
        alignments=list(alignments),
        n_total_slots=len(alignments),
        n_genuine=sum(a.classification == SlotClassification.GENUINE for a in alignments),
        n_impostor=sum(a.classification == SlotClassification.IMPOSTOR for a in alignments),
        n_mixed=sum(a.classification == SlotClassification.MIXED for a in alignments),
        n_unscoreable=sum(a.classification.value.startswith("unscoreable") for a in alignments),
        n_no_gold_overlap=sum(
            a.classification == SlotClassification.NO_GOLD_OVERLAP for a in alignments
        ),
    )


def _evidence(
    top_speaker_id: str = "speaker-a",
    *,
    similarity: float = 0.85,
    margin: float = 0.15,
    vote_agreement: float = 0.80,
    eligible_turns: int = 4,
    eligible_seconds: float = 12.0,
) -> dict:
    return {
        "run_id": "run-1",
        "top_speaker_id": top_speaker_id,
        "similarity": similarity,
        "margin": margin,
        "vote_agreement": vote_agreement,
        "eligible_turns": eligible_turns,
        "eligible_seconds": eligible_seconds,
        "roster_size": 2,
    }


def _attribution_trial(
    label: str,
    kind: TrialKind,
    classification: SlotClassification,
    *,
    similarity: float,
    margin: float,
    cluster: str,
) -> AttributionTrial:
    alignment = _alignment(label, classification, cluster)
    trial = Trial(
        run_id="run-1",
        label=label,
        similarity=similarity,
        margin=margin,
        vote_agreement=0.80,
        eligible_turns=4,
        eligible_seconds=12.0,
        roster_size=2,
        top_speaker_id="speaker-a",
        kind=kind,
        truth_anchoring="corpus_gold",
        cluster_id=cluster,
    )
    return AttributionTrial(trial=trial, slot_label=label, alignment=alignment)


def test_overlap_duration_zero_overlap() -> None:
    assert overlap_duration(Interval(0, 1), Interval(1, 2)) == 0.0


def test_overlap_duration_partial_overlap() -> None:
    assert overlap_duration(Interval(0, 3), Interval(2, 5)) == 1.0


def test_overlap_duration_full_containment() -> None:
    assert overlap_duration(Interval(0, 10), Interval(2, 4)) == 2.0


def test_overlap_duration_zero_length_interval() -> None:
    assert overlap_duration(Interval(2, 2), Interval(0, 4)) == 0.0


def test_total_duration_does_not_union_intervals() -> None:
    assert total_duration([Interval(0, 3), Interval(2, 5)]) == 6.0


def test_build_overlap_matrix_is_complete() -> None:
    cells = build_overlap_matrix(
        {"Alice": [Interval(0, 4)], "Bob": [Interval(4, 8)]},
        {
            "slot-a": [Interval(0, 3)],
            "slot-b": [Interval(3, 6)],
            "slot-c": [Interval(7, 9)],
        },
        collar=0,
    )

    values = {(c.gold_speaker, c.slot_label): c.overlap_seconds for c in cells}
    assert len(cells) == 6
    assert values == {
        ("Alice", "slot-a"): 3,
        ("Alice", "slot-b"): 1,
        ("Alice", "slot-c"): 0,
        ("Bob", "slot-a"): 0,
        ("Bob", "slot-b"): 2,
        ("Bob", "slot-c"): 1,
    }


def test_build_overlap_matrix_applies_collar() -> None:
    cells = build_overlap_matrix(
        {"Alice": [Interval(0, 4)]},
        {"slot": [Interval(0, 4)]},
        collar=0.5,
    )
    assert cells[0].overlap_seconds == 3.0


def test_build_overlap_matrix_skips_interval_shorter_than_two_collars() -> None:
    cells = build_overlap_matrix(
        {"Alice": [Interval(0, 0.9), Interval(2, 4)]},
        {"slot": [Interval(0, 4)]},
        collar=0.5,
    )
    assert cells[0].overlap_seconds == 1.0


def test_build_overlap_matrix_rejects_negative_collar() -> None:
    with pytest.raises(ValueError, match="collar"):
        build_overlap_matrix({}, {}, collar=-0.1)


def test_align_slots_perfect_alignment() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )

    aligned = report.alignments[0]
    assert aligned.classification == SlotClassification.GENUINE
    assert aligned.dominant_gold_speaker == "Alice"
    assert aligned.purity == 1.0
    assert aligned.coverage == 1.0
    assert aligned.margin == 0.0


def test_align_slots_partial_overlap_passes_lower_purity_floor() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 5)]},
        {"slot": [Interval(0, 10)]},
        purity_floor=0.5,
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.GENUINE
    assert report.alignments[0].purity == 0.5


def test_align_slots_partial_overlap_fails_purity() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 5)]},
        {"slot": [Interval(0, 10)]},
        purity_floor=0.6,
        margin_floor=0.4,
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_PURITY


def test_align_slots_tie_fails_margin_before_purity() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 5)], "Bob": [Interval(5, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_MARGIN
    assert report.alignments[0].dominant_gold_speaker == "Alice"
    assert report.alignments[0].margin == 0.0


def test_align_slots_near_tie_fails_margin() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 5.5)], "Bob": [Interval(5.5, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_MARGIN
    assert report.alignments[0].margin == pytest.approx(0.1)


def test_align_slots_below_coverage() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 100)]},
        {"slot": [Interval(0, 10)]},
        coverage_floor=0.2,
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_COVERAGE
    assert report.alignments[0].coverage == 0.1


def test_align_slots_below_duration_eligibility() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 5)]},
        {"slot": [Interval(0, 5)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_ELIGIBILITY


def test_align_slots_below_turn_eligibility_when_counts_supplied() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 10)]},
        {"slot": [Interval(0, 10)]},
        slot_turn_counts={"slot": 1},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.UNSCOREABLE_ELIGIBILITY


def test_align_slots_does_not_infer_turn_count_from_intervals() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.GENUINE


def test_align_slots_no_gold_overlap() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 10)]},
        {"slot": [Interval(20, 30)]},
        collar=0,
    )
    aligned = report.alignments[0]
    assert aligned.classification == SlotClassification.NO_GOLD_OVERLAP
    assert aligned.dominant_gold_speaker is None
    assert aligned.purity == 0.0


def test_align_slots_mixed_when_margin_passes_but_purity_fails() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 6)], "Bob": [Interval(6, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.MIXED
    assert report.alignments[0].purity == 0.6
    assert report.alignments[0].margin == pytest.approx(0.2)


def test_align_slots_fragmentation_can_produce_two_scoreable_slots() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 20)]},
        {"slot-a": [Interval(0, 10)], "slot-b": [Interval(10, 20)]},
        collar=0,
    )
    assert [a.classification for a in report.alignments] == [
        SlotClassification.GENUINE,
        SlotClassification.GENUINE,
    ]
    assert [a.coverage for a in report.alignments] == [0.5, 0.5]


def test_align_slots_allows_small_non_dominant_gold_overlap() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 9)], "Bob": [Interval(9, 10)]},
        {"slot": [Interval(0, 10)]},
        collar=0,
    )
    assert report.alignments[0].classification == SlotClassification.GENUINE
    assert report.alignments[0].dominant_gold_speaker == "Alice"
    assert report.alignments[0].purity == 0.9


def test_alignment_report_counts_match_alignments() -> None:
    report = align_slots(
        {"Alice": [Interval(0, 20)], "Bob": [Interval(20, 30)]},
        {
            "genuine": [Interval(0, 10)],
            "mixed": [Interval(14, 24)],
            "short": [Interval(24, 29)],
            "none": [Interval(40, 50)],
        },
        collar=0,
    )
    assert report.n_total_slots == len(report.alignments) == 4
    assert report.n_genuine == 1
    assert report.n_impostor == 0
    assert report.n_mixed == 1
    assert report.n_unscoreable == 1
    assert report.n_no_gold_overlap == 1


def test_build_trials_creates_genuine_trial_for_matching_identity() -> None:
    trials = build_trials(
        _report(_alignment()),
        {"slot-1": _evidence("speaker-a")},
        {"Alice": "speaker-a"},
    )
    assert trials[0].trial.kind == TrialKind.GENUINE
    assert trials[0].trial.truth_anchoring == "corpus_gold"


def test_build_trials_creates_impostor_trial_for_different_identity() -> None:
    trials = build_trials(
        _report(_alignment()),
        {"slot-1": _evidence("speaker-b")},
        {"Alice": "speaker-a"},
    )
    assert trials[0].trial.kind == TrialKind.IMPOSTOR


def test_build_trials_is_unscoreable_without_match_evidence() -> None:
    trials = build_trials(_report(_alignment()), {}, {"Alice": "speaker-a"})
    assert trials[0].trial.kind == TrialKind.UNSCOREABLE
    assert trials[0].trial.top_speaker_id is None


def test_build_trials_is_unscoreable_for_mixed_alignment() -> None:
    alignment = _alignment(classification=SlotClassification.MIXED)
    trials = build_trials(
        _report(alignment),
        {"slot-1": _evidence()},
        {"Alice": "speaker-a"},
    )
    assert trials[0].trial.kind == TrialKind.UNSCOREABLE


def test_build_trials_is_unscoreable_when_gold_is_not_enrolled() -> None:
    trials = build_trials(
        _report(_alignment()), {"slot-1": _evidence()}, {}
    )
    assert trials[0].trial.kind == TrialKind.UNSCOREABLE


def test_build_trials_uses_dominant_gold_as_cluster_id() -> None:
    trials = build_trials(
        _report(_alignment()),
        {"slot-1": _evidence()},
        {"Alice": "speaker-a"},
    )
    assert trials[0].trial.cluster_id == "Alice"
    assert trials[0].slot_label == "slot-1"
    assert trials[0].alignment.dominant_gold_speaker == "Alice"


def test_aggregate_trials_computes_far_frr_and_coverage() -> None:
    trials = [
        _attribution_trial(
            "g-auto", TrialKind.GENUINE, SlotClassification.GENUINE,
            similarity=0.85, margin=0.15, cluster="Alice",
        ),
        _attribution_trial(
            "g-review", TrialKind.GENUINE, SlotClassification.GENUINE,
            similarity=0.65, margin=0.06, cluster="Bob",
        ),
        _attribution_trial(
            "i-auto", TrialKind.IMPOSTOR, SlotClassification.GENUINE,
            similarity=0.85, margin=0.15, cluster="Carol",
        ),
        _attribution_trial(
            "i-abstain", TrialKind.IMPOSTOR, SlotClassification.GENUINE,
            similarity=0.40, margin=0.01, cluster="Dan",
        ),
    ]

    summary = aggregate_trials(trials)

    assert summary.n_genuine_trials == 2
    assert summary.n_impostor_trials == 2
    assert summary.n_auto_correct == 1
    assert summary.n_auto_wrong == 1
    assert summary.n_review == 1
    assert summary.n_abstain == 1
    assert summary.far == 0.5
    assert summary.frr == 0.5
    assert summary.coverage == 0.5
    assert summary.far_ci_upper == pytest.approx(wilson_ci(1, 2)[1])
    assert summary.frr_ci_upper == pytest.approx(wilson_ci(1, 2)[1])
    assert summary.n_speaker_clusters == 4


def test_aggregate_trials_zero_false_accepts() -> None:
    trials = [
        _attribution_trial(
            "i-review", TrialKind.IMPOSTOR, SlotClassification.GENUINE,
            similarity=0.65, margin=0.06, cluster="Alice",
        )
    ]
    summary = aggregate_trials(trials)
    assert summary.far == 0.0
    assert summary.n_auto_wrong == 0
    assert summary.far_ci_upper == pytest.approx(wilson_ci(0, 1)[1])


def test_aggregate_trials_excludes_unscoreable_from_band_counts() -> None:
    trials = [
        _attribution_trial(
            "mixed", TrialKind.UNSCOREABLE, SlotClassification.MIXED,
            similarity=0.90, margin=0.20, cluster="Alice",
        )
    ]
    summary = aggregate_trials(trials)
    assert summary.n_unscoreable == 1
    assert summary.n_auto_correct == 0
    assert summary.n_auto_wrong == 0
    assert summary.n_review == 0
    assert summary.n_abstain == 0
    assert summary.coverage == 0.0


def test_aggregate_trials_reports_alignment_attrition() -> None:
    trials = [
        _attribution_trial(
            "valid", TrialKind.GENUINE, SlotClassification.GENUINE,
            similarity=0.85, margin=0.15, cluster="Alice",
        ),
        _attribution_trial(
            "mixed", TrialKind.UNSCOREABLE, SlotClassification.MIXED,
            similarity=0.85, margin=0.15, cluster="Bob",
        ),
        _attribution_trial(
            "short", TrialKind.UNSCOREABLE,
            SlotClassification.UNSCOREABLE_ELIGIBILITY,
            similarity=0.85, margin=0.15, cluster="Carol",
        ),
    ]
    summary = aggregate_trials(trials)
    assert summary.alignment_attrition["genuine"] == 1
    assert summary.alignment_attrition["mixed"] == 1
    assert summary.alignment_attrition["unscoreable_eligibility"] == 1
    assert summary.alignment_attrition["no_gold_overlap"] == 0


def test_aggregate_trials_honors_custom_gates() -> None:
    trial = _attribution_trial(
        "candidate", TrialKind.GENUINE, SlotClassification.GENUINE,
        similarity=0.65, margin=0.06, cluster="Alice",
    )
    gates = MatchingGates(grounded_min_cosine=0.60, grounded_min_margin=0.05)
    summary = aggregate_trials([trial], gates)
    assert summary.n_auto_correct == 1
    assert summary.coverage == 1.0
