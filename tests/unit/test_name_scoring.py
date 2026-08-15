"""Tests for per-target name aggregation and explainable scoring (#38)."""

from voxint.enrichment.producers.name_patterns import (
    Attribution,
    MetadataRef,
    RawMention,
    SegmentRef,
)
from voxint.enrichment.producers.name_scoring import (
    SCORE_CAP,
    CandidateLevel,
    NameCandidate,
    aggregate,
)


def _metadata_mention(
    name: str,
    *,
    field: str = "title",
    pattern_id: str = "title_interview_with",
    reliability: float = 0.85,
    ambiguous: bool = False,
) -> RawMention:
    return RawMention(
        name=name,
        raw_span=name,
        pattern_id=pattern_id,
        reliability=reliability,
        attribution=Attribution.METADATA,
        source=MetadataRef(field=field),
        snippet=f"... {name} ...",
        ambiguous=ambiguous,
    )


def _transcript_mention(
    name: str,
    *,
    attribution: Attribution = Attribution.SELF,
    pattern_id: str = "self_my_name_is",
    reliability: float = 0.9,
    label: str | None = "SPEAKER_00",
    segment_index: int = 0,
    suspect: bool = False,
    ambiguous: bool = False,
) -> RawMention:
    return RawMention(
        name=name,
        raw_span=name,
        pattern_id=pattern_id,
        reliability=reliability,
        attribution=attribution,
        source=SegmentRef(
            segment_index=segment_index,
            diarization_label=label,
            start_seconds=float(segment_index) * 10.0,
            suspect=suspect,
        ),
        snippet=f"... {name} ...",
        ambiguous=ambiguous,
    )


# ---------------------------------------------------------------------------
# Target routing
# ---------------------------------------------------------------------------


def test_metadata_and_other_mentions_stay_run_level() -> None:
    candidates = aggregate(
        [
            _metadata_mention("Jane Doe"),
            _transcript_mention(
                "Jane Doe",
                attribution=Attribution.OTHER,
                pattern_id="other_joined_by",
                reliability=0.75,
                segment_index=3,
            ),
        ]
    )
    assert [(c.level, c.diarization_label, c.name) for c in candidates] == [
        (CandidateLevel.RUN, None, "Jane Doe")
    ]


def test_self_mention_targets_its_segment_label() -> None:
    (candidate,) = aggregate([_transcript_mention("Jane Doe", label="SPEAKER_01")])
    assert candidate.level is CandidateLevel.RUN_LABEL
    assert candidate.diarization_label == "SPEAKER_01"


def test_self_mention_without_label_degrades_to_run_level() -> None:
    (candidate,) = aggregate([_transcript_mention("Jane Doe", label=None)])
    assert candidate.level is CandidateLevel.RUN
    assert candidate.diarization_label is None


def test_no_cross_target_leakage() -> None:
    """A title mention must not create or inflate a cluster claim."""
    candidates = aggregate(
        [
            _metadata_mention("Jane Doe"),
            _transcript_mention("Jane Doe", label="SPEAKER_00"),
        ]
    )
    by_level = {c.level: c for c in candidates}
    assert set(by_level) == {CandidateLevel.RUN, CandidateLevel.RUN_LABEL}
    run_label = by_level[CandidateLevel.RUN_LABEL]
    # The cluster claim is scored from the self-intro alone: one mention,
    # no source-diversity or corroboration bonus from the title.
    assert run_label.score_components["mention_count"] == 1.0
    assert run_label.score_components["bonus_sources"] == 0.0
    assert run_label.score_components["bonus_patterns"] == 0.0


def test_same_name_two_labels_yields_two_cluster_candidates() -> None:
    candidates = aggregate(
        [
            _transcript_mention("Jane Doe", label="SPEAKER_00", segment_index=0),
            _transcript_mention("Jane Doe", label="SPEAKER_02", segment_index=9),
        ]
    )
    assert [(c.level, c.diarization_label) for c in candidates] == [
        (CandidateLevel.RUN_LABEL, "SPEAKER_00"),
        (CandidateLevel.RUN_LABEL, "SPEAKER_02"),
    ]


def test_no_given_name_prefix_merging() -> None:
    candidates = aggregate(
        [
            _metadata_mention("Jane"),
            _metadata_mention("Jane Doe", field="description", pattern_id="desc_guest"),
        ]
    )
    assert sorted(c.name for c in candidates) == ["Jane", "Jane Doe"]


def test_case_variants_group_together() -> None:
    (candidate,) = aggregate(
        [
            _metadata_mention("Jane Doe"),
            _metadata_mention("JANE DOE", field="description", pattern_id="desc_guest"),
        ]
    )
    assert candidate.name == "Jane Doe"  # first-seen display form
    assert candidate.score_components["mention_count"] == 2.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_base_is_max_adjusted_reliability_no_frequency_reward() -> None:
    once = aggregate([_metadata_mention("Jane Doe", reliability=0.85)])
    thrice = aggregate(
        [
            _metadata_mention("Jane Doe", reliability=0.85),
            _metadata_mention("Jane Doe", reliability=0.85),
            _metadata_mention("Jane Doe", reliability=0.85),
        ]
    )
    assert once[0].score == thrice[0].score  # same pattern, same source: no bonus
    assert thrice[0].score_components["mention_count"] == 3.0


def test_corroboration_and_diversity_bonuses() -> None:
    (candidate,) = aggregate(
        [
            _metadata_mention("Jane Doe", reliability=0.85),
            _metadata_mention(
                "Jane Doe", field="description", pattern_id="desc_guest", reliability=0.75
            ),
        ]
    )
    assert candidate.score_components["bonus_patterns"] == 0.05
    assert candidate.score_components["bonus_sources"] == 0.05
    assert candidate.score == round(0.85 + 0.05 + 0.05, 4)


def test_score_capped_below_certainty() -> None:
    (candidate,) = aggregate(
        [
            _transcript_mention("Jane Doe", reliability=0.9, label=None),
            _metadata_mention("Jane Doe"),
        ],
        name_seeds=["jane doe"],
    )
    assert candidate.score == SCORE_CAP


def test_suspect_segment_halves_reliability() -> None:
    (clean,) = aggregate([_transcript_mention("Jane Doe")])
    (suspect,) = aggregate([_transcript_mention("Jane Doe", suspect=True)])
    assert clean.score_components["base"] == 0.9
    assert suspect.score_components["base"] == 0.45
    assert suspect.score_components["suspect_penalty_applied"] == 1.0
    assert clean.score_components["suspect_penalty_applied"] == 0.0


def test_seed_match_bonus_exact_casefolded() -> None:
    (boosted,) = aggregate([_metadata_mention("Jane Doe")], name_seeds=["JANE DOE"])
    (plain,) = aggregate([_metadata_mention("Jane Doe")], name_seeds=["someone else"])
    assert boosted.score_components["bonus_seed"] == 0.05
    assert plain.score_components["bonus_seed"] == 0.0


# ---------------------------------------------------------------------------
# Ambiguity gate
# ---------------------------------------------------------------------------


def test_ambiguous_weak_mention_dropped() -> None:
    assert (
        aggregate(
            [_metadata_mention("Will", pattern_id="title_with", reliability=0.65, ambiguous=True)]
        )
        == []
    )


def test_ambiguous_kept_with_strong_self_intro() -> None:
    (candidate,) = aggregate([_transcript_mention("Will", reliability=0.9, ambiguous=True)])
    assert candidate.name == "Will"


def test_ambiguous_kept_with_seed_match() -> None:
    (candidate,) = aggregate(
        [_metadata_mention("Will", pattern_id="title_with", reliability=0.65, ambiguous=True)],
        name_seeds=["will"],
    )
    assert candidate.name == "Will"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _fixture_mentions() -> list[RawMention]:
    return [
        _transcript_mention("Jane Doe", label="SPEAKER_01", segment_index=4),
        _metadata_mention("Bob Smith", field="description", pattern_id="desc_hosted_by"),
        _metadata_mention("Jane Doe"),
        _transcript_mention(
            "Bob Smith",
            attribution=Attribution.OTHER,
            pattern_id="other_welcome",
            reliability=0.7,
            segment_index=1,
        ),
    ]


def test_output_fully_deterministic() -> None:
    first = aggregate(_fixture_mentions())
    second = aggregate(list(reversed(_fixture_mentions())))
    assert [(c.name, c.level, c.diarization_label, c.score) for c in first] == [
        (c.name, c.level, c.diarization_label, c.score) for c in second
    ]
    assert [c.score_components for c in first] == [c.score_components for c in second]


def test_evidence_ordered_strongest_first() -> None:
    (candidate,) = aggregate(
        [
            _metadata_mention("Jane Doe", field="tags", pattern_id="tag_person", reliability=0.4),
            _metadata_mention("Jane Doe", reliability=0.85),
        ]
    )
    assert [m.reliability for m in candidate.mentions] == [0.85, 0.4]


def test_run_candidates_sort_before_run_label() -> None:
    candidates = aggregate(_fixture_mentions())
    levels = [c.level for c in candidates]
    assert levels == sorted(levels, key=lambda level: level.value)


def test_empty_input() -> None:
    assert aggregate([]) == []


def test_candidate_is_frozen_value_object() -> None:
    (candidate,) = aggregate([_metadata_mention("Jane Doe")])
    assert isinstance(candidate, NameCandidate)


def test_suspect_mention_cannot_vouch_for_ambiguous_name() -> None:
    # The ambiguity gate uses suspect-adjusted reliability: a strong self-intro
    # pattern inside a hallucination-flagged segment (0.9 -> 0.45) must not
    # admit an ambiguous single-token name.
    assert (
        aggregate([_transcript_mention("Will", reliability=0.9, suspect=True, ambiguous=True)])
        == []
    )
    (clean,) = aggregate([_transcript_mention("Will", reliability=0.9, ambiguous=True)])
    assert clean.name == "Will"
