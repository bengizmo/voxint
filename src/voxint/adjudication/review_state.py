"""The one writer for per-segment operator review state (issues #53/#58).

Mutable, latest-wins UPSERT keyed on ``transcript_segment_id`` — no nonce, no
append-only ledger (verified/corrected is operator workflow state, orthogonal to
speaker attribution; see the provenance design note). Callers hold the run claim
lock (``verify_claim(..., for_update=True)``), which serializes writes per run,
so a plain get-or-mutate is race-free without ``ON CONFLICT`` machinery.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import effective_text
from voxint.db.models import SegmentReviewState, TranscriptSegment


def _get_or_create(session: Session, segment: TranscriptSegment) -> SegmentReviewState:
    row = session.get(SegmentReviewState, segment.id)
    if row is None:
        row = SegmentReviewState(
            transcript_segment_id=segment.id,
            pipeline_run_id=segment.pipeline_run_id,
        )
        session.add(row)
    return row


def set_verified(
    session: Session, *, segment: TranscriptSegment, verified: bool
) -> SegmentReviewState:
    """Mark (or unmark) a segment verified. Idempotent."""
    row = _get_or_create(session, segment)
    row.verified_at = datetime.now(UTC) if verified else None
    session.flush()
    return row


def set_correction(
    session: Session, *, segment: TranscriptSegment, text: str | None
) -> SegmentReviewState | None:
    """Set or clear the operator correction for a segment.

    Whitespace-only text, or text identical to what the pipeline already renders
    (``enhanced`` or ``raw``), means *no correction* (NULL) — a revert, never a
    badge on unchanged text. Otherwise the submission is stored **verbatim**
    (Whisper emits leading spaces; stripping meaningful input would both lose that
    whitespace and badge an unchanged segment).

    Only a genuine *change* to the effective correction clears the verified mark
    and rewrites ``corrected_at`` — replaying the same text (a retry, or a second
    tab saving identical text) is a true no-op, so it never silently unverifies a
    segment. When it does change, verification is cleared in the same transaction,
    so "verified" never sits on since-changed text.
    """
    submitted = text or ""
    # A revert if empty/whitespace-only, or verbatim-equal to the pipeline text.
    is_revert = not submitted.strip() or submitted == effective_text(segment, None)
    desired = None if is_revert else submitted
    row = session.get(SegmentReviewState, segment.id)
    current = row.corrected_text if row is not None else None
    if desired == current:
        return row  # no change → leave verification (and corrected_at) untouched
    row = _get_or_create(session, segment)
    now = datetime.now(UTC)
    row.corrected_text = desired
    row.corrected_at = now if desired is not None else None
    row.verified_at = None  # the text changed → it must be re-verified
    session.flush()
    return row


def verified_progress(session: Session, run_id: uuid.UUID) -> tuple[int, int]:
    """``(verified, total)`` segment counts for a run — the "N of M" counter."""
    total = session.execute(
        select(func.count())
        .select_from(TranscriptSegment)
        .where(TranscriptSegment.pipeline_run_id == run_id)
    ).scalar_one()
    verified = session.execute(
        select(func.count())
        .select_from(SegmentReviewState)
        .where(
            SegmentReviewState.pipeline_run_id == run_id,
            SegmentReviewState.verified_at.is_not(None),
        )
    ).scalar_one()
    return verified, total
