"""The `/runs` execution-history browser: filters, keyset pagination, listing.

A read-only view over ``pipeline_runs`` for the operator. Two orthogonal
filters — ``status`` (the raw run status) and ``review`` (a post-hoc
adjudication classification) — combine with AND. The review classification is
derived in SQL from the same resolver definitions the workbench uses, so a run
is classified here exactly as it would be in the ``/review`` queue.

Pagination is keyset (seek) on ``(created_at, id)`` descending — newest first,
stable across concurrent inserts, and bounded to one page per request. The
cursor carries the full-precision sort key of the last row shown; a strict
tuple comparison walks strictly older rows, so identical ``created_at`` values
never drop or duplicate a row (``id`` breaks the tie).
"""

import base64
import enum
import uuid
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode

from sqlalchemy import and_, func, or_
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import (
    label_count,
    unresolved_label_count,
    unresolved_label_exists,
)
from voxint.db.models import MediaItem, PipelineRun, RunStatus


class ReviewFilter(enum.StrEnum):
    """Post-hoc adjudication classification, orthogonal to run status."""

    NEEDED = "needed"  # COMPLETED with >=1 label still needing a human ruling
    RESOLVED = "resolved"  # COMPLETED with nothing left to rule on
    CLAIMED = "claimed"  # a reviewer currently holds a live claim


class InvalidCursorError(ValueError):
    """The pagination cursor is malformed or corrupt."""


@dataclass(frozen=True)
class Cursor:
    """The keyset position: the sort key of the last row already shown."""

    created_at: datetime
    run_id: uuid.UUID

    def encode(self) -> str:
        raw = f"{self.created_at.isoformat()}|{self.run_id}"
        return base64.urlsafe_b64encode(raw.encode()).decode()

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        try:
            raw = base64.urlsafe_b64decode(token.encode()).decode()
            timestamp, run_id = raw.rsplit("|", 1)
            created_at = datetime.fromisoformat(timestamp)
            parsed_id = uuid.UUID(run_id)
        except ValueError as exc:  # bad base64, bad split, bad datetime, bad uuid
            raise InvalidCursorError(f"unparseable cursor {token!r}") from exc
        if created_at.tzinfo is None:
            # Every cursor we mint carries the TIMESTAMPTZ offset. A naive value
            # is a forged token: compared against the tz-aware column it would be
            # cast using the session timezone, silently shifting the seek
            # boundary under any non-UTC session. Reject it rather than paginate
            # off a moved fence.
            raise InvalidCursorError(f"cursor timestamp is not tz-aware: {token!r}")
        return cls(created_at=created_at, run_id=parsed_id)


@dataclass(frozen=True)
class RunListItem:
    """One row of the browser — everything the template renders per run."""

    run_id: uuid.UUID
    status: str
    source_path: str
    created_at: datetime
    unresolved_count: int
    label_count: int
    claim_live: bool
    claimed_by: str | None


@dataclass(frozen=True)
class RunsPage:
    """One bounded page plus the cursor that fetches the next (older) one."""

    items: list[RunListItem]
    next_cursor: Cursor | None


def parse_status_filter(raw: str | None) -> RunStatus | None:
    """A blank/absent value means 'all'; anything else must be a real status."""
    if raw in (None, ""):
        return None
    try:
        return RunStatus(raw)
    except ValueError as exc:
        raise ValueError(f"unknown status {raw!r}") from exc


def parse_review_filter(raw: str | None) -> ReviewFilter | None:
    if raw in (None, ""):
        return None
    try:
        return ReviewFilter(raw)
    except ValueError as exc:
        raise ValueError(f"unknown review filter {raw!r}") from exc


def runs_url(
    *,
    status: RunStatus | None = None,
    review: ReviewFilter | None = None,
    cursor: Cursor | None = None,
) -> str:
    """Build a ``/runs`` URL preserving the active filters (+ optional cursor)."""
    params: list[tuple[str, str]] = []
    if status is not None:
        params.append(("status", status.value))
    if review is not None:
        params.append(("review", review.value))
    if cursor is not None:
        params.append(("cursor", cursor.encode()))
    return "/runs" + (f"?{urlencode(params)}" if params else "")


def list_runs(
    session: Session,
    *,
    status: RunStatus | None,
    review: ReviewFilter | None,
    cursor: Cursor | None,
    page_size: int,
) -> RunsPage:
    """One bounded, newest-first keyset page of runs matching the filters."""
    claim_live = and_(
        PipelineRun.review_claim_expires_at.isnot(None),
        PipelineRun.review_claim_expires_at > func.now(),
    )
    stmt = (
        sa_select(
            PipelineRun.id,
            PipelineRun.status,
            PipelineRun.created_at,
            PipelineRun.review_claimed_by,
            MediaItem.source_path,
            unresolved_label_count(PipelineRun.id).label("unresolved_count"),
            label_count(PipelineRun.id).label("label_count"),
            claim_live.label("claim_live"),
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        .limit(page_size + 1)
    )

    if status is not None:
        stmt = stmt.where(PipelineRun.status == status.value)

    # needed/resolved are the COMPLETED complement of each other; claimed is a
    # live-lease predicate independent of status. Combined with an incompatible
    # status= (e.g. status=failed&review=needed) they yield an empty page, by
    # design — the filters are composable, not status-independent.
    if review is ReviewFilter.NEEDED:
        stmt = stmt.where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            unresolved_label_exists(PipelineRun.id),
        )
    elif review is ReviewFilter.RESOLVED:
        stmt = stmt.where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            ~unresolved_label_exists(PipelineRun.id),
        )
    elif review is ReviewFilter.CLAIMED:
        stmt = stmt.where(claim_live)

    if cursor is not None:
        stmt = stmt.where(
            or_(
                PipelineRun.created_at < cursor.created_at,
                and_(
                    PipelineRun.created_at == cursor.created_at,
                    PipelineRun.id < cursor.run_id,
                ),
            )
        )

    rows = session.execute(stmt).all()
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    items = [
        RunListItem(
            run_id=row.id,
            status=row.status,
            source_path=row.source_path,
            created_at=row.created_at,
            unresolved_count=row.unresolved_count,
            label_count=row.label_count,
            claim_live=row.claim_live,
            claimed_by=row.review_claimed_by if row.claim_live else None,
        )
        for row in rows
    ]
    next_cursor = (
        Cursor(created_at=rows[-1].created_at, run_id=rows[-1].id)
        if has_more and rows
        else None
    )
    return RunsPage(items=items, next_cursor=next_cursor)
