"""Human-confirmed, single-cluster voice samples from other recordings."""

import uuid
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.attribution import AttributedInterval, attributed_intervals
from voxint.db.models import AdjudicationDecision, Decision, DiarizationTurn, MediaItem, PipelineRun
from voxint.media.reclaim import run_intermediate_reclaimed_at
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path
from voxint.speakers.aggregate import canonical_runs
from voxint.speakers.roster import active_speakers, canonicalize, merge_map

VOICE_SAMPLE_MIN_SECONDS = 2.0
VOICE_SAMPLE_MAX_SECONDS = 10.0
Span = tuple[float, float]
Availability = Literal["available", "gone"]


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Union half-open spans, including pieces that touch."""
    result: list[Span] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


@dataclass(frozen=True)
class SpanIndex:
    """Disjoint sorted spans with binary-searchable ends."""

    spans: tuple[Span, ...]
    ends: tuple[float, ...]

    @classmethod
    def build(cls, spans: Iterable[Span]) -> "SpanIndex":
        merged = tuple(merge_spans(spans))
        return cls(merged, tuple(end for _, end in merged))

    def intersect(self, start: float, end: float) -> Iterator[Span]:
        if end <= start:
            return
        index = bisect_right(self.ends, start)
        while index < len(self.spans):
            lo, hi = self.spans[index]
            if lo >= end:
                break
            yield max(start, lo), min(end, hi)
            index += 1


def index_turns(turns: Iterable[DiarizationTurn]) -> dict[str, tuple[SpanIndex, SpanIndex]]:
    """Build same-label and other-label unions once per run."""
    labels: dict[str, list[Span]] = {}
    for turn in turns:
        labels.setdefault(turn.label, []).append((turn.start_seconds, turn.end_seconds))
    merged = {label: merge_spans(spans) for label, spans in labels.items()}
    return {
        label: (
            SpanIndex.build(spans),
            SpanIndex.build(
                piece for other, pieces in merged.items() if other != label for piece in pieces
            ),
        )
        for label, spans in merged.items()
    }


def clean_spans(
    row: AttributedInterval, indexes: dict[str, tuple[SpanIndex, SpanIndex]]
) -> list[Span]:
    """Intersect with the row's label and subtract other labels at exact boundaries.

    Binary searches skip unrelated turns; only intersecting pieces are visited.
    Whole-turn overlap flags cannot locate the clean head/tail boundaries.
    """
    if row.diarization_label not in indexes:
        return []
    same, other = indexes[row.diarization_label]
    pieces: list[Span] = []
    for start, end in same.intersect(row.start_seconds, row.end_seconds):
        cursor = start
        for lo, hi in other.intersect(start, end):
            if cursor < lo:
                pieces.append((cursor, lo))
            cursor = hi
        if cursor < end:
            pieces.append((cursor, end))
    return pieces


def span_rank(span: Span) -> tuple[Decimal, float]:
    """Exact decimal duration, then earliest start."""
    return -(Decimal(str(span[1])) - Decimal(str(span[0]))), span[0]


def best_span(spans: Iterable[Span]) -> Span | None:
    """Union within one label, apply the floor, then rank eligible pieces."""
    eligible = [
        span
        for span in merge_spans(spans)
        if -span_rank(span)[0] >= Decimal(str(VOICE_SAMPLE_MIN_SECONDS))
    ]
    return min(eligible, key=span_rank, default=None)


def choose_run_spans(
    rows: Iterable[AttributedInterval], turns: Iterable[DiarizationTurn], wanted: set[uuid.UUID]
) -> dict[uuid.UUID, Span]:
    """Choose each wanted speaker's best span without stitching different labels."""
    indexes = index_turns(turns)
    spans: dict[tuple[uuid.UUID, str], list[Span]] = {}
    for row in rows:
        speaker = row.speaker_id
        if speaker is None or speaker not in wanted:
            continue
        if row.is_human_assign and row.diarization_label is not None:
            spans.setdefault((speaker, row.diarization_label), []).extend(clean_spans(row, indexes))
    result: dict[uuid.UUID, Span] = {}
    for (speaker, _), pieces in spans.items():
        chosen = best_span(pieces)
        if chosen is not None and (
            speaker not in result or span_rank(chosen) < span_rank(result[speaker])
        ):
            result[speaker] = chosen
    return result


def run_clean_spans(
    session: Session, run_id: uuid.UUID, wanted: set[uuid.UUID]
) -> dict[uuid.UUID, Span]:
    """One attribution walk and turn index per prefiltered run."""
    turns = session.scalars(
        select(DiarizationTurn).where(DiarizationTurn.pipeline_run_id == run_id)
    )
    return choose_run_spans(attributed_intervals(session, run_id), turns, wanted)


@dataclass(frozen=True)
class VoiceSample:
    speaker_id: uuid.UUID
    run_id: uuid.UUID
    span: Span
    availability: Availability

    @property
    def window(self) -> Span:
        start, end = self.span
        return start, min(end, start + VOICE_SAMPLE_MAX_SECONDS)


def voice_sample_candidates(
    session: Session,
    *,
    exclude_media_id: uuid.UUID,
    speaker_id: uuid.UUID | None = None,
    skip_resolved: set[uuid.UUID] | None = None,
) -> Iterator[VoiceSample]:
    """Shared newest-first walk; gone entries retain confirmed-audio evidence.

    No file access occurs here. The consumer can stop after its first servable
    candidate, or fold the entire walk for the availability list. The list
    consumer adds available speakers to skip_resolved to avoid redundant walks.
    """
    mapping = merge_map(session)
    active = {speaker.id for speaker in active_speakers(session)}
    if speaker_id is not None:
        canonical = canonicalize(speaker_id, mapping)
        active &= {canonical}
    if not active:
        return
    # Select canonical runs BEFORE the necessary-condition ledger prefilter.
    runs = canonical_runs(session)
    statement = (
        select(AdjudicationDecision.pipeline_run_id, AdjudicationDecision.speaker_id)
        .join(PipelineRun, PipelineRun.id == AdjudicationDecision.pipeline_run_id)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .where(
            AdjudicationDecision.pipeline_run_id.in_([run_id for run_id, _, _ in runs]),
            AdjudicationDecision.decision == Decision.ASSIGN.value,
            AdjudicationDecision.speaker_id.is_not(None),
            MediaItem.id != exclude_media_id,
            MediaItem.trashed_at.is_(None),
            MediaItem.purged_at.is_(None),
        )
        .distinct()
    )
    if speaker_id is not None:
        aliases = {canonical} | {
            source for source in mapping if canonicalize(source, mapping) == canonical
        }
        statement = statement.where(AdjudicationDecision.speaker_id.in_(aliases))
    run_speakers: dict[uuid.UUID, set[uuid.UUID]] = {}
    for run_id, assigned_speaker in session.execute(statement):
        if assigned_speaker is not None:
            canonical_speaker = canonicalize(assigned_speaker, mapping)
            if canonical_speaker in active:
                run_speakers.setdefault(run_id, set()).add(canonical_speaker)
    for run_id, _, _ in runs:
        wanted = run_speakers.get(run_id, set())
        if skip_resolved is not None:
            wanted = wanted - skip_resolved
        if not wanted:
            continue
        choices = run_clean_spans(session, run_id, wanted)
        if not choices:
            continue
        availability: Availability = "available"
        if run_intermediate_reclaimed_at(session, run_id) is not None:
            availability = "gone"
        else:
            try:
                normalized_audio_path(session, run_id, Path())
            except StageDataError:
                availability = "gone"
        for speaker, span in choices.items():
            if speaker in active:
                yield VoiceSample(speaker, run_id, span, availability)


def voice_sample_availability(
    session: Session, *, exclude_media_id: uuid.UUID
) -> dict[uuid.UUID, Availability]:
    """Return DB-only availability; the list route includes both values.

    Gate/PCM failures are discovered only by the clip route, which answers 410
    when no candidate can be served, even if this walk reports available.
    """
    result: dict[uuid.UUID, Availability] = {}
    resolved: set[uuid.UUID] = set()
    for candidate in voice_sample_candidates(
        session, exclude_media_id=exclude_media_id, skip_resolved=resolved
    ):
        result[candidate.speaker_id] = candidate.availability
        if candidate.availability == "available":
            resolved.add(candidate.speaker_id)
    return result
