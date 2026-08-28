"""Per-speaker aggregates from effective resolution (issue #159).

The numbers on the speakers overview and profile pages — files, spoken
minutes, first/last seen, verified — computed by folding the structured
attribution seam (``adjudication.attribution.attributed_intervals``), never by
joining raw ``speaker_assignments`` (machine proposals are not attribution,
and ledger rows keep pre-merge ids).

Semantics:

- **Canonical run per media**: the newest completed, non-archived run per
  media item (``created_at DESC, id DESC``). Reprocessing a file never
  inflates its minutes; every aggregate counts a media item exactly once.
- **Coverage**: ALL canonical runs, exact lifetime stats — no sampling cap.
  Measured on maintainer hardware (commit-1 spike): ~12 statements and ~4.4 ms
  per run, 0.87 s for a 200-file library — acceptable for the single-operator
  scale this serves. If a much larger library ever misses budget, the fix is a
  request-scoped batch loader behind ``attributed_intervals``, not a cap (a
  newest-first cap cannot give a truthful ``first_seen``).
- **Attribution**: an interval counts for the speaker iff its WINNING ruling
  attributes them (human assign at any scope, or grounded cosine). ``verified``
  = at least one positive-duration interval whose winning ruling is a human
  assign — a label assignment fully displaced by narrower overrides verifies
  nobody, and a later exclude flips the badge off.
- **first/last seen** = min/max of ``media_items.created_at`` over the
  speaker's attributed files: when the material entered the archive (label it
  "file added" in UI copy — it is not when the speaker was recognized).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.adjudication.attribution import attributed_intervals
from voxint.adjudication.resolver import Resolution
from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.speakers.roster import alias_ids, canonicalize, merge_map


@dataclass(frozen=True)
class SpeakerAppearance:
    """One media item a speaker is effectively attributed in (via its
    canonical run) — the profile page's associated-media row."""

    media_id: uuid.UUID
    run_id: uuid.UUID
    media_created_at: datetime
    seconds: float
    segments: int
    human_assigned: bool  # ≥1 surviving human-assign interval in this media
    auto_enrolled: bool = False


@dataclass(frozen=True)
class SpeakerAggregate:
    """Lifetime effective-resolution stats for one canonical speaker."""

    speaker_id: uuid.UUID
    files: int
    seconds: float
    segments: int
    first_seen: datetime | None
    last_seen: datetime | None
    verified: bool
    appearances: tuple[SpeakerAppearance, ...]  # newest media first
    # The (run, raw label) pairs where a grounded-cosine attribution SURVIVED
    # (won ≥1 positive-duration interval) — the tier module's evidence keys
    # (issue #159): a fully overridden label contributes no voice evidence.
    grounded_keys: tuple[tuple[uuid.UUID, str], ...] = ()
    auto_enrolled: bool = False


@dataclass(frozen=True)
class AggregateResult:
    """All speakers' aggregates plus explicit coverage facts."""

    by_speaker: dict[uuid.UUID, SpeakerAggregate]
    runs_scanned: int


def _canonical_runs(session: Session) -> list[tuple[uuid.UUID, uuid.UUID, datetime]]:
    """(run_id, media_id, media_created_at) for the newest completed,
    non-archived run of every media item — one bounded window-function SELECT
    (the ``media_query`` idiom), newest media first."""
    ranked = (
        select(
            PipelineRun.id.label("run_id"),
            PipelineRun.media_item_id.label("media_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rank"),
        )
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
        )
        .subquery()
    )
    rows = session.execute(
        select(ranked.c.run_id, ranked.c.media_id, MediaItem.created_at)
        .join(MediaItem, MediaItem.id == ranked.c.media_id)
        .where(ranked.c.rank == 1)
        .order_by(MediaItem.created_at.desc(), MediaItem.id.desc())
    ).all()
    return [(r.run_id, r.media_id, r.created_at) for r in rows]


@dataclass
class _Tally:
    seconds: float = 0.0
    segments: int = 0
    human: bool = False
    auto_enrolled: bool = False


def aggregate_speakers(session: Session) -> AggregateResult:
    """Fold every canonical run's attributed intervals into per-speaker stats.

    Interval speaker ids come out of the resolver already canonical, so the
    fold groups by them directly; a merge performed after a run was resolved
    is still honored because canonicalization happens at read time.
    """
    per_speaker: dict[uuid.UUID, list[SpeakerAppearance]] = {}
    grounded: dict[uuid.UUID, set[tuple[uuid.UUID, str]]] = {}
    runs = _canonical_runs(session)
    for run_id, media_id, media_created_at in runs:
        tallies: dict[uuid.UUID, _Tally] = {}
        for interval in attributed_intervals(session, run_id):
            if interval.speaker_id is None:
                continue
            duration = interval.end_seconds - interval.start_seconds
            if duration <= 0:
                # A degenerate (empty or malformed) interval attributes
                # nothing and must not inflate the segment count or drag the
                # seconds total (#159 review).
                continue
            tally = tallies.setdefault(interval.speaker_id, _Tally())
            tally.seconds += duration
            tally.segments += 1
            if interval.resolution is Resolution.HUMAN_ASSIGN:
                tally.human = True
            elif interval.resolution is Resolution.AUTO_ENROLL:
                tally.auto_enrolled = True
            elif (
                interval.resolution is Resolution.GROUNDED_COSINE
                and interval.diarization_label is not None
            ):
                grounded.setdefault(interval.speaker_id, set()).add(
                    (run_id, interval.diarization_label)
                )
        for speaker_id, tally in tallies.items():
            per_speaker.setdefault(speaker_id, []).append(
                SpeakerAppearance(
                    media_id=media_id,
                    run_id=run_id,
                    media_created_at=media_created_at,
                    seconds=tally.seconds,
                    segments=tally.segments,
                    human_assigned=tally.human,
                    auto_enrolled=tally.auto_enrolled,
                )
            )
    by_speaker = {
        speaker_id: SpeakerAggregate(
            speaker_id=speaker_id,
            files=len(appearances),
            seconds=sum(a.seconds for a in appearances),
            segments=sum(a.segments for a in appearances),
            first_seen=min(a.media_created_at for a in appearances),
            last_seen=max(a.media_created_at for a in appearances),
            verified=any(a.human_assigned for a in appearances),
            appearances=tuple(appearances),
            grounded_keys=tuple(sorted(grounded.get(speaker_id, set()), key=str)),
            auto_enrolled=any(a.auto_enrolled for a in appearances),
        )
        for speaker_id, appearances in per_speaker.items()
    }
    return AggregateResult(by_speaker=by_speaker, runs_scanned=len(runs))


def empty_aggregate(speaker_id: uuid.UUID) -> SpeakerAggregate:
    """The zero row for a roster speaker with no attributed intervals — the
    overview renders every active speaker, never silently drops one."""
    return SpeakerAggregate(
        speaker_id=speaker_id,
        files=0,
        seconds=0.0,
        segments=0,
        first_seen=None,
        last_seen=None,
        verified=False,
        appearances=(),
    )


def aggregate_for_speaker(
    session: Session, speaker_id: uuid.UUID
) -> SpeakerAggregate:
    """One speaker's aggregate (profile page), canonicalized first.

    Runs the same full fold: the per-run resolver output is shared across all
    speakers, so a filtered walk would do the same work anyway — and the
    overview and profile can never disagree on a number.
    """
    canonical = canonicalize(speaker_id, merge_map(session))
    result = aggregate_speakers(session)
    return result.by_speaker.get(canonical) or empty_aggregate(canonical)


def enrollment_count(session: Session, speaker_id: uuid.UUID) -> int:
    """The speaker's enrolled voiceprints, across merge aliases."""
    from voxint.db.models import SpeakerEmbedding

    aliases = alias_ids(session, speaker_id)
    return int(
        session.execute(
            select(func.count())
            .select_from(SpeakerEmbedding)
            .where(SpeakerEmbedding.speaker_id.in_(aliases))
        ).scalar_one()
    )
