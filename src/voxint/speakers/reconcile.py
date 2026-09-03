"""Post-batch reconcile: re-derive proposals for cold-start-affected runs.

When a batch of media files is processed, early runs see a small speaker
roster. Later runs benefit from auto-enrollment growing the roster.  A
reconcile pass re-evaluates earlier runs against the now-mature roster.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import and_, distinct, func, or_, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    MatchCandidate,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
)
from voxint.speakers.matching import MatchingGates
from voxint.speakers.reembed import refresh_run_matches
from voxint.speakers.roster import active_speaker_clause


@dataclass(frozen=True, slots=True)
class ReconcilePlan:
    current_roster_sizes: dict[str, int]
    affected_run_ids: tuple[uuid.UUID, ...]
    max_deficit: int


@dataclass(frozen=True, slots=True)
class RunReconcileResult:
    run_id: uuid.UUID
    added: int
    removed: int
    changed: int

    @property
    def unchanged(self) -> bool:
        return self.added == 0 and self.removed == 0 and self.changed == 0


def _current_roster_sizes(session: Session) -> dict[str, int]:
    """Count distinct active enrolled speakers per embedding space."""
    rows = session.execute(
        select(
            SpeakerEmbedding.embedding_space,
            func.count(distinct(SpeakerEmbedding.speaker_id)),
        )
        .join(Speaker, SpeakerEmbedding.speaker_id == Speaker.id)
        .where(active_speaker_clause())
        .group_by(SpeakerEmbedding.embedding_space)
    ).all()
    return {space: count for space, count in rows}


def cold_start_affected_runs(
    session: Session,
    *,
    run_id: uuid.UUID | None = None,
) -> ReconcilePlan:
    """Identify completed runs matched against a smaller roster than today.

    Only considers runs with non-NULL ``MatchCandidate.roster_size`` rows.
    Comparison is per embedding space: roster growth in space A does not
    flag runs matched in space B.

    When *run_id* is given it acts as a discovery filter: only that run is
    returned, and only if it qualifies.  Archived runs are included
    (consistent with auto-enroll-backfill).
    """
    sizes = _current_roster_sizes(session)
    if not sizes:
        return ReconcilePlan({}, (), 0)

    space_predicates = [
        and_(
            MatchCandidate.embedding_space == space,
            MatchCandidate.roster_size < count,
        )
        for space, count in sizes.items()
    ]

    query = (
        select(distinct(MatchCandidate.pipeline_run_id))
        .join(PipelineRun, MatchCandidate.pipeline_run_id == PipelineRun.id)
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            MatchCandidate.roster_size.isnot(None),
            or_(*space_predicates),
        )
        .order_by(MatchCandidate.pipeline_run_id)
    )
    if run_id is not None:
        query = query.where(MatchCandidate.pipeline_run_id == run_id)

    affected = tuple(session.scalars(query))

    max_deficit = 0
    if affected:
        for space, current in sizes.items():
            min_observed = session.scalar(
                select(func.min(MatchCandidate.roster_size)).where(
                    MatchCandidate.pipeline_run_id.in_(affected),
                    MatchCandidate.embedding_space == space,
                    MatchCandidate.roster_size.isnot(None),
                )
            )
            if min_observed is not None:
                max_deficit = max(max_deficit, current - min_observed)

    return ReconcilePlan(sizes, affected, max_deficit)


def _snapshot_assignments(
    session: Session, run_id: uuid.UUID
) -> set[tuple[str, uuid.UUID | None, str, str | None, float | None, bool]]:
    """Semantic snapshot of current speaker assignments for a run.

    Returns a set of tuples (label, speaker_id, method, proposed_name,
    confidence, grounded) -- enough to detect meaningful changes while
    ignoring row IDs and timestamps.
    """
    rows = session.execute(
        select(SpeakerAssignment).where(
            SpeakerAssignment.pipeline_run_id == run_id,
        )
    ).scalars()
    return {
        (
            row.diarization_label,
            row.speaker_id,
            row.method,
            row.proposed_name,
            row.confidence,
            row.grounded,
        )
        for row in rows
    }


def reconcile_run(
    session: Session,
    run_id: uuid.UUID,
    gates: MatchingGates,
) -> RunReconcileResult:
    """Re-derive proposals for one run and report what changed."""
    before = _snapshot_assignments(session, run_id)
    refresh_run_matches(session, run_id, gates)
    session.flush()
    after = _snapshot_assignments(session, run_id)

    added = len(after - before)
    removed = len(before - after)
    changed_labels = {a[0] for a in (before - after)} & {a[0] for a in (after - before)}

    return RunReconcileResult(
        run_id=run_id,
        added=added - len(changed_labels),
        removed=removed - len(changed_labels),
        changed=len(changed_labels),
    )
