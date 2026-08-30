"""CRUD operations for saved KWIC quotes (issue #338, Phase 6)."""

from __future__ import annotations

import csv
import io
import re
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import (
    MediaFolder,
    MediaItem,
    PipelineRun,
    SavedQuote,
    TranscriptSegment,
)


class QuoteDuplicateError(Exception):
    """Raised when the (project, segment, query) triple already exists."""


class QuoteSegmentMismatchError(Exception):
    """Raised when segment_id does not belong to run_id."""


def resolve_project_id(session: Session, run_id: uuid.UUID) -> uuid.UUID | None:
    """Resolve the project that owns a pipeline run's recording.

    Follows pipeline_runs -> media_items -> media_folders -> projects.
    Returns None if any link in the chain is NULL (the recording is not in
    a project).
    """
    row = session.execute(
        select(MediaFolder.project_id)
        .join(MediaItem, MediaItem.media_folder_id == MediaFolder.id)
        .join(PipelineRun, PipelineRun.media_item_id == MediaItem.id)
        .where(PipelineRun.id == run_id)
    ).first()
    if row is None:
        return None
    project_id: uuid.UUID | None = row[0]
    return project_id


def _verify_segment_belongs_to_run(
    session: Session,
    segment_id: uuid.UUID,
    run_id: uuid.UUID,
) -> None:
    """Verify that segment_id belongs to run_id; raise on mismatch or missing."""
    seg = session.get(TranscriptSegment, segment_id)
    if seg is None:
        msg = "Segment not found. The transcript may have been re-processed; re-run the search."
        raise QuoteSegmentMismatchError(msg)
    if seg.pipeline_run_id != run_id:
        msg = "Segment does not belong to this run."
        raise QuoteSegmentMismatchError(msg)


def save_quote(
    session: Session,
    *,
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    search_query: str,
    left_context: str,
    hit: str,
    right_context: str,
    speaker_name: str | None,
    media_title: str,
    start_seconds: float,
    operator: str,
    note: str | None = None,
) -> SavedQuote:
    """Save a KWIC concordance row as project-scoped evidence."""
    project_id = resolve_project_id(session, run_id)
    if project_id is None:
        msg = "This recording is not assigned to a project."
        raise ValueError(msg)
    _verify_segment_belongs_to_run(session, segment_id, run_id)
    normalized_note = note.strip() or None if note else None
    query_trimmed = search_query.strip()
    quote = SavedQuote(
        project_id=project_id,
        segment_id=segment_id,
        run_id=run_id,
        search_query=query_trimmed,
        left_context=left_context,
        hit=hit,
        right_context=right_context,
        speaker_name=speaker_name,
        media_title=media_title,
        start_seconds=start_seconds,
        note=normalized_note,
        operator=operator,
    )
    try:
        with session.begin_nested():
            session.add(quote)
            session.flush()
    except IntegrityError as exc:
        if "saved_quotes_project_segment_query_key" in str(exc):
            raise QuoteDuplicateError from exc
        raise
    return quote


def list_quotes(
    session: Session,
    project_id: uuid.UUID,
    *,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[SavedQuote], int]:
    """Return saved quotes for a project, newest first."""
    base = select(SavedQuote).where(SavedQuote.project_id == project_id)
    total = session.scalar(
        select(func.count()).select_from(base.subquery()),
    ) or 0
    rows = list(
        session.scalars(
            base.order_by(SavedQuote.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    )
    return rows, total


def delete_quote(session: Session, quote_id: uuid.UUID) -> bool:
    """Hard-delete a saved quote. Returns True if the row existed."""
    quote = session.get(SavedQuote, quote_id)
    if quote is None:
        return False
    session.delete(quote)
    session.flush()
    return True


def update_quote_note(
    session: Session,
    quote_id: uuid.UUID,
    note: str | None,
) -> SavedQuote | None:
    """Update the operator note on a saved quote."""
    quote = session.get(SavedQuote, quote_id)
    if quote is None:
        return None
    quote.note = note.strip() or None if note else None
    session.flush()
    return quote


def _csv_safe(value: str) -> str:
    stripped = value.lstrip("\t\r\n ")
    if stripped and stripped[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def export_quotes_csv(
    session: Session,
    project_id: uuid.UUID,
    project_name: str,
) -> tuple[str, str]:
    """Export saved quotes for a project as CSV.

    Returns (csv_content, filename). Capped at 10,000 rows with a
    truncation notice when the project has more.
    """
    cap = 10_000
    quotes, total = list_quotes(session, project_id, limit=cap)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow((
        "search_query",
        "left_context",
        "hit",
        "right_context",
        "speaker_name",
        "media_title",
        "start_seconds",
        "note",
        "saved_at",
    ))
    for q in quotes:
        writer.writerow((
            _csv_safe(q.search_query),
            _csv_safe(q.left_context),
            _csv_safe(q.hit),
            _csv_safe(q.right_context),
            _csv_safe(q.speaker_name or ""),
            _csv_safe(q.media_title),
            q.start_seconds,
            _csv_safe(q.note or ""),
            q.created_at.isoformat() if q.created_at else "",
        ))
    if total > cap:
        writer.writerow((f"# {total - cap} additional quotes not exported (cap: {cap})",))
    safe_name = re.sub(r"[^a-z0-9-]", "-", project_name.lower())[:40].strip("-")
    filename = f"voxint-quotes-{safe_name or 'export'}.csv"
    return output.getvalue(), filename
