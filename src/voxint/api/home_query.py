"""Read-only queries behind the Home page (Console 2.0 P1, issue #152).

Same shape as :mod:`voxint.api.runs_query` and :mod:`voxint.api.stats_query`:
frozen dataclasses plus functions that take a :class:`~sqlalchemy.orm.Session`
and issue bounded ``SELECT``s, no HTTP and no side effects. Home's stat counts
live in ``stats_query`` (shared with ``voxint stats``); this module carries only
what is Home-specific — the recent-activity feed.

The feed is derived entirely from existing tables (no event/outbox table; that
is a P7 concern if toasts ship): three bounded newest-first slices — runs
started, runs that reached a terminal outcome, speakers enrolled — merged and
trimmed in Python. Each slice is capped at the feed limit, so the merge sees at
most ``3 * limit`` rows regardless of table sizes.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import unresolved_label_count
from voxint.api.presentation import title_from_snapshot
from voxint.db.models import (
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    Speaker,
    StageRun,
)


@dataclass(frozen=True)
class ActivityItem:
    """One recent-activity row: what happened, when, and where to click.

    ``kind`` is one of ``run_started`` / ``run_completed`` / ``run_failed`` /
    ``speaker_enrolled``. ``title`` follows the run-listing display precedence
    (sidecar title over acquisition-metadata title, ``None`` otherwise — the
    template falls back to a cleaned filename via ``friendly_media_label``);
    for a speaker it is the display name and ``source_path`` is empty.
    """

    at: datetime
    kind: str
    title: str | None
    source_path: str
    run_id: uuid.UUID | None = None
    speaker_id: uuid.UUID | None = None
    unresolved_count: int = 0


# One terminal-outcome entry per run, not per stage attempt: the feed reports
# that a recording finished or failed, not the retry history (that lives on the
# run detail's stage ledger).
_TERMINAL_STATUSES = (RunStatus.COMPLETED.value, RunStatus.FAILED.value)


def _run_rows(session: Session, *, limit: int, terminal: bool) -> list[ActivityItem]:
    """Newest runs by start (``terminal=False``) or by last stage finish."""
    last_finished = (
        sa_select(func.max(StageRun.finished_at))
        .where(StageRun.pipeline_run_id == PipelineRun.id)
        .correlate(PipelineRun)
        .scalar_subquery()
    )
    # A terminal run's meaningful timestamp is when its last stage attempt
    # finished (the same convention as runs_query.latest_completed_run),
    # coalesced to updated_at for seeded/legacy runs with no stage rows.
    at = (
        func.coalesce(last_finished, PipelineRun.updated_at)
        if terminal
        else (PipelineRun.created_at)
    )
    columns = [
        PipelineRun.id,
        PipelineRun.status,
        PipelineRun.sidecar,
        MediaItem.source_path,
        MediaSourceMetadata.title.label("source_title"),
        at.label("at"),
    ]
    if terminal:
        columns.append(
            func.coalesce(unresolved_label_count(PipelineRun.id), 0).label("unresolved_count")
        )
    stmt = (
        sa_select(*columns)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        # Outer: most media has no metadata snapshot (uploads, pre-#36 runs).
        .outerjoin(
            MediaSourceMetadata,
            MediaSourceMetadata.media_item_id == MediaItem.id,
        )
        .where(PipelineRun.archived_at.is_(None))
        .order_by(at.desc(), PipelineRun.created_at.desc(), PipelineRun.id.desc())
        .limit(limit)
    )
    if terminal:
        stmt = stmt.where(PipelineRun.status.in_(_TERMINAL_STATUSES))
    items: list[ActivityItem] = []
    for row in session.execute(stmt):
        if terminal:
            kind = "run_completed" if row.status == RunStatus.COMPLETED.value else "run_failed"
        else:
            kind = "run_started"
        items.append(
            ActivityItem(
                # TIMESTAMPTZ comes back in the session timezone; normalize so
                # the template's "... UTC" title label is always true.
                at=row.at.astimezone(UTC),
                kind=kind,
                title=title_from_snapshot(row.sidecar) or row.source_title,
                source_path=row.source_path,
                run_id=row.id,
                unresolved_count=row.unresolved_count if terminal else 0,
            )
        )
    return items


def _speaker_rows(session: Session, *, limit: int) -> list[ActivityItem]:
    rows = session.execute(
        sa_select(Speaker.id, Speaker.display_name, Speaker.created_at)
        .order_by(Speaker.created_at.desc(), Speaker.id.desc())
        .limit(limit)
    )
    return [
        ActivityItem(
            at=row.created_at.astimezone(UTC),
            kind="speaker_enrolled",
            title=row.display_name,
            source_path="",
            speaker_id=row.id,
        )
        for row in rows
    ]


def recent_activity(session: Session, *, limit: int = 10) -> list[ActivityItem]:
    """The newest ``limit`` activity items across the three source families.

    Deterministic under equal timestamps: ties order by kind then by the row's
    own id, so two page loads over unchanged data render identically.
    """
    merged = [
        *_run_rows(session, limit=limit, terminal=False),
        *_run_rows(session, limit=limit, terminal=True),
        *_speaker_rows(session, limit=limit),
    ]
    merged.sort(key=lambda i: (i.at, i.kind, str(i.run_id or i.speaker_id)), reverse=True)
    return merged[:limit]
