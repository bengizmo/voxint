"""Pure gold-to-diarization alignment and attribution trial construction.

This module deliberately operates only on caller-supplied values.  It is the
bridge between corpus timing truth and the generic calibration trial model;
it does not read application settings or persistence models.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import Any

from voxint.harness.calibration import Trial, TrialKind
from voxint.harness.name_accuracy import wilson_ci
from voxint.speakers.matching import MatchingGates
from voxint.speakers.tiers import passes_accept, passes_grounded


@dataclass(frozen=True)
class Interval:
    """A half-open interval measured in seconds."""

    start: float
    end: float


def overlap_duration(a: Interval, b: Interval) -> float:
    """Return the non-negative duration of the intersection of two intervals."""
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def total_duration(intervals: list[Interval]) -> float:
    """Sum interval durations without taking their union."""
    return sum(max(0.0, interval.end - interval.start) for interval in intervals)


@dataclass(frozen=True)
class OverlapCell:
    """Total overlap for one gold-speaker/predicted-slot pair."""

    gold_speaker: str
    slot_label: str
    overlap_seconds: float


def _collared(intervals: list[Interval], collar: float) -> list[Interval]:
    return [
        Interval(interval.start + collar, interval.end - collar)
        for interval in intervals
        if interval.end - interval.start >= 2 * collar
    ]


def build_overlap_matrix(
    gold_intervals: dict[str, list[Interval]],
    slot_intervals: dict[str, list[Interval]],
    collar: float = 0.25,
) -> list[OverlapCell]:
    """Build the complete gold x slot duration-overlap matrix.

    Gold boundaries are moved inward by ``collar`` on each side.  Pairwise
    interval intersections are summed without unioning either input list.
    """
    if collar < 0:
        raise ValueError("collar must be non-negative")

    cells: list[OverlapCell] = []
    for gold_speaker, gold_speech in gold_intervals.items():
        scoreable_gold = _collared(gold_speech, collar)
        for slot_label, slot_speech in slot_intervals.items():
            overlap = sum(
                overlap_duration(gold, slot)
                for gold in scoreable_gold
                for slot in slot_speech
            )
            cells.append(OverlapCell(gold_speaker, slot_label, overlap))
    return cells


class SlotClassification(enum.StrEnum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"
    MIXED = "mixed"
    UNSCOREABLE_PURITY = "unscoreable_purity"
    UNSCOREABLE_COVERAGE = "unscoreable_coverage"
    UNSCOREABLE_MARGIN = "unscoreable_margin"
    UNSCOREABLE_ELIGIBILITY = "unscoreable_eligibility"
    NO_GOLD_OVERLAP = "no_gold_overlap"


@dataclass(frozen=True)
class SlotAlignment:
    slot_label: str
    classification: SlotClassification
    dominant_gold_speaker: str | None
    purity: float
    coverage: float
    margin: float
    slot_duration: float


@dataclass(frozen=True)
class AlignmentReport:
    alignments: list[SlotAlignment]
    n_total_slots: int
    n_genuine: int
    n_impostor: int
    n_mixed: int
    n_unscoreable: int
    n_no_gold_overlap: int


def align_slots(
    gold_intervals: dict[str, list[Interval]],
    slot_intervals: dict[str, list[Interval]],
    *,
    purity_floor: float = 0.85,
    coverage_floor: float = 0.20,
    margin_floor: float = 0.15,
    collar: float = 0.25,
    min_slot_seconds: float = 6.0,
    min_slot_turns: int = 2,
    slot_turn_counts: dict[str, int] | None = None,
) -> AlignmentReport:
    """Classify each predicted slot against the collared gold partition.

    A valid timing alignment is marked ``GENUINE`` because it supplies genuine
    gold identity truth.  Whether the machine match is a genuine or impostor
    trial is determined later by :func:`build_trials`.

    Turn eligibility is applied when ``slot_turn_counts`` is supplied.  Without
    it, intervals are not assumed to correspond one-to-one with matcher turns.
    """
    for name, value in (
        ("purity_floor", purity_floor),
        ("coverage_floor", coverage_floor),
        ("margin_floor", margin_floor),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if min_slot_seconds < 0 or min_slot_turns < 0:
        raise ValueError("eligibility floors must be non-negative")

    matrix = build_overlap_matrix(gold_intervals, slot_intervals, collar)
    overlap_by_slot: dict[str, dict[str, float]] = {
        label: {} for label in slot_intervals
    }
    for cell in matrix:
        overlap_by_slot[cell.slot_label][cell.gold_speaker] = cell.overlap_seconds
    gold_durations = {
        speaker: total_duration(_collared(intervals, collar))
        for speaker, intervals in gold_intervals.items()
    }

    alignments: list[SlotAlignment] = []
    for slot_label, intervals in slot_intervals.items():
        slot_duration = total_duration(intervals)
        overlaps = overlap_by_slot[slot_label]
        ranked = sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))
        positive = [(speaker, seconds) for speaker, seconds in ranked if seconds > 0]

        if not positive:
            alignments.append(
                SlotAlignment(
                    slot_label=slot_label,
                    classification=SlotClassification.NO_GOLD_OVERLAP,
                    dominant_gold_speaker=None,
                    purity=0.0,
                    coverage=0.0,
                    margin=0.0,
                    slot_duration=slot_duration,
                )
            )
            continue

        dominant, top_overlap = positive[0]
        purity = min(1.0, top_overlap / slot_duration) if slot_duration > 0 else 0.0
        second_purity = (
            positive[1][1] / slot_duration
            if len(positive) > 1 and slot_duration > 0
            else 0.0
        )
        # A runner-up margin is undefined when only one gold speaker overlaps.
        # Record the protocol-specified zero, but do not reject an otherwise
        # pure single-speaker slot merely because no runner-up exists.
        margin = purity - second_purity if len(positive) > 1 else 0.0
        dominant_duration = gold_durations[dominant]
        coverage = min(1.0, top_overlap / dominant_duration) if dominant_duration > 0 else 0.0

        too_few_turns = (
            slot_turn_counts is not None
            and slot_turn_counts.get(slot_label, 0) < min_slot_turns
        )
        if slot_duration < min_slot_seconds or too_few_turns:
            classification = SlotClassification.UNSCOREABLE_ELIGIBILITY
        elif len(positive) > 1 and margin < margin_floor:
            classification = SlotClassification.UNSCOREABLE_MARGIN
        elif purity < purity_floor:
            classification = (
                SlotClassification.MIXED
                if len(positive) > 1
                else SlotClassification.UNSCOREABLE_PURITY
            )
        elif coverage < coverage_floor:
            classification = SlotClassification.UNSCOREABLE_COVERAGE
        else:
            classification = SlotClassification.GENUINE

        alignments.append(
            SlotAlignment(
                slot_label=slot_label,
                classification=classification,
                dominant_gold_speaker=dominant,
                purity=purity,
                coverage=coverage,
                margin=margin,
                slot_duration=slot_duration,
            )
        )

    counts = Counter(item.classification for item in alignments)
    unscoreable_classes = {
        SlotClassification.UNSCOREABLE_PURITY,
        SlotClassification.UNSCOREABLE_COVERAGE,
        SlotClassification.UNSCOREABLE_MARGIN,
        SlotClassification.UNSCOREABLE_ELIGIBILITY,
    }
    return AlignmentReport(
        alignments=alignments,
        n_total_slots=len(alignments),
        n_genuine=counts[SlotClassification.GENUINE],
        n_impostor=counts[SlotClassification.IMPOSTOR],
        n_mixed=counts[SlotClassification.MIXED],
        n_unscoreable=sum(counts[kind] for kind in unscoreable_classes),
        n_no_gold_overlap=counts[SlotClassification.NO_GOLD_OVERLAP],
    )


@dataclass(frozen=True)
class AttributionTrial:
    """A calibration trial paired with its timing-alignment provenance."""

    trial: Trial
    slot_label: str
    alignment: SlotAlignment


_SCOREABLE_ALIGNMENTS = {
    SlotClassification.GENUINE,
    SlotClassification.IMPOSTOR,
}


def _evidence_value(evidence: dict[str, Any], key: str, default: Any) -> Any:
    value = evidence.get(key, default)
    return default if value is None and default is not None else value


def build_trials(
    alignment: AlignmentReport,
    match_evidence: dict[str, dict[str, Any]],
    enrolled_speaker_map: dict[str, str],
    *,
    truth_source: str = "corpus_gold",
) -> list[AttributionTrial]:
    """Build calibration trials from gold alignments and matcher evidence."""
    results: list[AttributionTrial] = []
    for item in alignment.alignments:
        evidence = match_evidence.get(item.slot_label)
        gold_speaker_id = (
            enrolled_speaker_map.get(item.dominant_gold_speaker)
            if item.dominant_gold_speaker is not None
            else None
        )
        top_speaker_id = evidence.get("top_speaker_id") if evidence else None
        scoreable = (
            item.classification in _SCOREABLE_ALIGNMENTS
            and evidence is not None
            and gold_speaker_id is not None
            and top_speaker_id is not None
        )
        if scoreable:
            kind = (
                TrialKind.GENUINE
                if top_speaker_id == gold_speaker_id
                else TrialKind.IMPOSTOR
            )
            cluster_id = item.dominant_gold_speaker
        else:
            kind = TrialKind.UNSCOREABLE
            cluster_id = f"__unscoreable__:{item.slot_label}"

        evidence = evidence or {}
        trial = Trial(
            run_id=str(
                evidence.get("run_id", evidence.get("pipeline_run_id", ""))
            ),
            label=item.slot_label,
            similarity=evidence.get("similarity"),
            margin=evidence.get("margin"),
            vote_agreement=evidence.get("vote_agreement"),
            eligible_turns=int(_evidence_value(evidence, "eligible_turns", 0)),
            eligible_seconds=float(
                _evidence_value(evidence, "eligible_seconds", 0.0)
            ),
            roster_size=evidence.get("roster_size"),
            top_speaker_id=top_speaker_id,
            kind=kind,
            truth_anchoring=truth_source,
            cluster_id=str(cluster_id),
        )
        results.append(AttributionTrial(trial, item.slot_label, item))
    return results


@dataclass(frozen=True)
class AttributionSummary:
    n_genuine_trials: int
    n_impostor_trials: int
    n_unscoreable: int
    n_auto_correct: int
    n_auto_wrong: int
    n_review: int
    n_abstain: int
    far: float
    far_ci_upper: float
    frr: float
    frr_ci_upper: float
    coverage: float
    n_speaker_clusters: int
    alignment_attrition: dict[str, int]


def aggregate_trials(
    trials: list[AttributionTrial],
    gates: MatchingGates | None = None,
) -> AttributionSummary:
    """Aggregate attribution trials into FAR, FRR, and coverage metrics."""
    active_gates = gates or MatchingGates()
    scoreable = [item.trial for item in trials if item.trial.kind != TrialKind.UNSCOREABLE]
    n_genuine = sum(trial.kind == TrialKind.GENUINE for trial in scoreable)
    n_impostor = sum(trial.kind == TrialKind.IMPOSTOR for trial in scoreable)
    n_unscoreable = len(trials) - len(scoreable)
    n_auto_correct = 0
    n_auto_wrong = 0
    n_review = 0
    n_abstain = 0

    for trial in scoreable:
        fields = (
            trial.similarity,
            trial.margin,
            trial.vote_agreement,
            trial.eligible_turns,
            trial.eligible_seconds,
            trial.roster_size,
        )
        if passes_grounded(*fields, active_gates):
            if trial.kind == TrialKind.GENUINE:
                n_auto_correct += 1
            else:
                n_auto_wrong += 1
        elif passes_accept(*fields, active_gates):
            n_review += 1
        else:
            n_abstain += 1

    false_rejects = n_genuine - n_auto_correct
    far = n_auto_wrong / n_impostor if n_impostor else 0.0
    frr = false_rejects / n_genuine if n_genuine else 0.0
    auto_count = n_auto_correct + n_auto_wrong
    coverage = auto_count / len(scoreable) if scoreable else 0.0
    attrition_counts = Counter(item.alignment.classification for item in trials)
    attrition = {
        classification.value: attrition_counts[classification]
        for classification in SlotClassification
    }

    return AttributionSummary(
        n_genuine_trials=n_genuine,
        n_impostor_trials=n_impostor,
        n_unscoreable=n_unscoreable,
        n_auto_correct=n_auto_correct,
        n_auto_wrong=n_auto_wrong,
        n_review=n_review,
        n_abstain=n_abstain,
        far=far,
        far_ci_upper=wilson_ci(n_auto_wrong, n_impostor)[1],
        frr=frr,
        frr_ci_upper=wilson_ci(false_rejects, n_genuine)[1],
        coverage=coverage,
        n_speaker_clusters=len({trial.cluster_id for trial in scoreable}),
        alignment_attrition=dict(attrition),
    )
