"""Human-confirmed, single-cluster voice samples from other recordings."""

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.attribution import AttributedInterval, attributed_intervals
from voxint.db.models import DiarizationTurn, MediaItem
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


def clean_spans(row: AttributedInterval, turns: Iterable[DiarizationTurn]) -> list[Span]:
    """Intersect the attribution with its label, then subtract every other label.

    Whole-turn overlap flags cannot locate overlap boundaries. Actual turn
    intervals preserve the clean head/tail of a partially overlapped turn.
    The duration floor is applied by the per-speaker fold after unioning rows.
    """
    same: list[Span] = []
    other: list[Span] = []
    for turn in turns:
        span = (max(row.start_seconds, turn.start_seconds), min(row.end_seconds, turn.end_seconds))
        (same if turn.label == row.diarization_label else other).append(span)
    pieces = merge_spans(same)
    for lo, hi in merge_spans(other):
        remaining: list[Span] = []
        for start, end in pieces:
            if hi <= start or lo >= end:
                remaining.append((start, end))
            else:
                if start < lo:
                    remaining.append((start, lo))
                if hi < end:
                    remaining.append((hi, end))
        pieces = remaining
    return pieces


def best_span(spans: Iterable[Span]) -> Span | None:
    """Apply the floor, then choose longest with earliest-start tie breaking."""
    eligible = [
        span for span in merge_spans(spans) if span[1] - span[0] >= VOICE_SAMPLE_MIN_SECONDS
    ]
    return min(eligible, key=lambda span: (-(span[1] - span[0]), span[0]), default=None)


def run_clean_spans(session: Session, run_id: uuid.UUID) -> dict[uuid.UUID, Span]:
    """One attribution walk and one turn load for every speaker in a run."""
    turns = list(
        session.scalars(select(DiarizationTurn).where(DiarizationTurn.pipeline_run_id == run_id))
    )
    spans: dict[uuid.UUID, list[Span]] = {}
    for row in attributed_intervals(session, run_id):
        if row.is_human_assign and row.speaker_id is not None and row.diarization_label is not None:
            spans.setdefault(row.speaker_id, []).extend(clean_spans(row, turns))
    return {
        speaker: chosen
        for speaker, pieces in spans.items()
        if (chosen := best_span(pieces)) is not None
    }


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
    session: Session, *, exclude_media_id: uuid.UUID, speaker_id: uuid.UUID | None = None
) -> Iterator[VoiceSample]:
    """Shared newest-first walk; gone entries retain confirmed-audio evidence.

    No file access occurs here. The consumer can stop after its first servable
    candidate, or fold the entire walk for the availability list.
    """
    active = {speaker.id for speaker in active_speakers(session)}
    if speaker_id is not None:
        canonical = canonicalize(speaker_id, merge_map(session))
        active &= {canonical}
    if not active:
        return
    for run_id, media_id, _ in canonical_runs(session):
        if media_id == exclude_media_id:
            continue
        media = session.get(MediaItem, media_id)
        if media is None or media.trashed_at is not None or media.purged_at is not None:
            continue
        choices = run_clean_spans(session, run_id)
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
    """DB-only availability; a gate failure at serving time also means gone."""
    result: dict[uuid.UUID, Availability] = {}
    for candidate in voice_sample_candidates(session, exclude_media_id=exclude_media_id):
        if result.get(candidate.speaker_id) != "available":
            result[candidate.speaker_id] = candidate.availability
    return result
