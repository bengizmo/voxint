"""Corpus exploration queries — KWIC concordance and term statistics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Date as SQLDate
from sqlalchemy import and_, case, cast, func, or_, select, true
from sqlalchemy.orm import Session

from voxint.db import search
from voxint.db.models import (
    AdjudicationDecision,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    Speaker,
    TranscriptSegment,
)

_HEADLINE_OPTIONS = "StartSel=<mark>, StopSel=</mark>, MaxWords=15, MinWords=5, MaxFragments=1"


@dataclass
class KWICRow:
    """One concordance hit."""

    left_context: str
    hit: str
    right_context: str
    speaker_name: str | None
    speaker_id: uuid.UUID | None
    run_id: uuid.UUID
    media_title: str
    segment_id: uuid.UUID
    start_seconds: float
    confidence: float | None
    suspect: bool


@dataclass
class KWICFilters:
    query: str = ""
    project_id: uuid.UUID | None = None
    speaker_id: uuid.UUID | None = None
    date_from: date | None = None
    date_to: date | None = None
    low_confidence_only: bool = False
    suspect_only: bool = False


@dataclass
class KWICResult:
    rows: list[KWICRow] = field(default_factory=list)
    total: int = 0
    query: str = ""
    filters: KWICFilters | None = None


def _strip_marks(text: str) -> str:
    return text.replace("<mark>", "").replace("</mark>", "")


def _split_headline(headline: str) -> tuple[str, str, str]:
    left, start, remainder = headline.partition("<mark>")
    if not start:
        return _strip_marks(headline), "", ""
    hit, stop, right = remainder.partition("</mark>")
    if not stop:
        return _strip_marks(left), _strip_marks(remainder), ""
    return _strip_marks(left), _strip_marks(hit), _strip_marks(right)


def kwic_search(
    session: Session,
    filters: KWICFilters,
    *,
    limit: int = 50,
    offset: int = 0,
) -> KWICResult:
    """Full-text KWIC concordance search across the corpus.

    Uses the effective transcript text: corrected > enhanced > raw.
    Resolves speaker identity through adjudication decisions.
    """
    query_text = filters.query.strip()
    if not query_text:
        return KWICResult(query=filters.query, filters=filters)

    tsquery = search.ts_query(query_text)
    corrected_vector = search.ts_vector(SegmentReviewState.corrected_text)
    enhanced_vector = search.ts_vector(TranscriptSegment.enhanced_text)
    raw_vector = search.ts_vector(TranscriptSegment.raw_text)
    matches = or_(
        and_(
            SegmentReviewState.corrected_text.isnot(None),
            corrected_vector.op("@@")(tsquery),
        ),
        and_(
            SegmentReviewState.corrected_text.is_(None),
            TranscriptSegment.enhanced_text.isnot(None),
            enhanced_vector.op("@@")(tsquery),
        ),
        and_(
            SegmentReviewState.corrected_text.is_(None),
            TranscriptSegment.enhanced_text.is_(None),
            raw_vector.op("@@")(tsquery),
        ),
    )
    relevance = case(
        (SegmentReviewState.corrected_text.isnot(None), func.ts_rank(corrected_vector, tsquery)),
        (TranscriptSegment.enhanced_text.isnot(None), func.ts_rank(enhanced_vector, tsquery)),
        else_=func.ts_rank(raw_vector, tsquery),
    )
    effective_text = func.coalesce(
        SegmentReviewState.corrected_text,
        TranscriptSegment.enhanced_text,
        TranscriptSegment.raw_text,
    )

    # The ledger is append-only. Resolve the newest whole-segment ruling first;
    # an explicit inherit falls back to the newest label-level ruling.
    segment_decision = (
        select(
            AdjudicationDecision.decision.label("decision"),
            AdjudicationDecision.speaker_id.label("speaker_id"),
        )
        .where(
            AdjudicationDecision.pipeline_run_id == TranscriptSegment.pipeline_run_id,
            AdjudicationDecision.transcript_segment_id == TranscriptSegment.id,
            AdjudicationDecision.start_word_index.is_(None),
        )
        .order_by(
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
        .limit(1)
        .correlate(TranscriptSegment)
        .lateral("segment_decision")
    )
    label_decision = (
        select(
            AdjudicationDecision.decision.label("decision"),
            AdjudicationDecision.speaker_id.label("speaker_id"),
        )
        .where(
            AdjudicationDecision.pipeline_run_id == TranscriptSegment.pipeline_run_id,
            AdjudicationDecision.diarization_label == TranscriptSegment.diarization_label,
            AdjudicationDecision.transcript_segment_id.is_(None),
        )
        .order_by(
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
        .limit(1)
        .correlate(TranscriptSegment)
        .lateral("label_decision")
    )
    effective_speaker_id = case(
        (
            segment_decision.c.decision == "assign",
            segment_decision.c.speaker_id,
        ),
        (
            segment_decision.c.decision == "inherit",
            label_decision.c.speaker_id,
        ),
        else_=label_decision.c.speaker_id,
    )
    effective_date = func.coalesce(
        MediaSourceMetadata.upload_date,
        cast(MediaItem.created_at, SQLDate),
    )

    stmt = (
        select(
            search.ts_headline(effective_text, tsquery, _HEADLINE_OPTIONS).label("headline"),
            Speaker.display_name.label("speaker_name"),
            Speaker.id.label("speaker_id"),
            PipelineRun.id.label("run_id"),
            func.coalesce(MediaSourceMetadata.title, MediaItem.source_path).label("media_title"),
            TranscriptSegment.id.label("segment_id"),
            TranscriptSegment.start_seconds,
            TranscriptSegment.confidence,
            TranscriptSegment.suspect,
            relevance.label("relevance"),
        )
        .select_from(TranscriptSegment)
        .outerjoin(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .join(PipelineRun, PipelineRun.id == TranscriptSegment.pipeline_run_id)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .outerjoin(
            MediaSourceMetadata,
            MediaSourceMetadata.media_item_id == MediaItem.id,
        )
        .outerjoin(segment_decision, true())
        .outerjoin(label_decision, true())
        .outerjoin(Speaker, Speaker.id == effective_speaker_id)
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
            matches,
        )
    )
    if filters.project_id is not None:
        stmt = stmt.join_from(
            MediaItem,
            MediaFolder,
            MediaFolder.id == MediaItem.media_folder_id,
        ).where(MediaFolder.project_id == filters.project_id)
    if filters.speaker_id is not None:
        stmt = stmt.where(effective_speaker_id == filters.speaker_id)
    if filters.date_from is not None:
        stmt = stmt.where(effective_date >= filters.date_from)
    if filters.date_to is not None:
        stmt = stmt.where(effective_date <= filters.date_to)
    if filters.low_confidence_only:
        stmt = stmt.where(TranscriptSegment.confidence < 0.7)
    if filters.suspect_only:
        stmt = stmt.where(TranscriptSegment.suspect.is_(True))

    total_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
    total = int(session.execute(total_stmt).scalar_one())
    result_rows = session.execute(
        stmt.order_by(relevance.desc(), TranscriptSegment.id).limit(limit).offset(offset)
    )
    rows: list[KWICRow] = []
    for row in result_rows:
        left, hit, right = _split_headline(row.headline)
        rows.append(
            KWICRow(
                left_context=left,
                hit=hit,
                right_context=right,
                speaker_name=row.speaker_name,
                speaker_id=row.speaker_id,
                run_id=row.run_id,
                media_title=row.media_title,
                segment_id=row.segment_id,
                start_seconds=row.start_seconds,
                confidence=row.confidence,
                suspect=row.suspect,
            )
        )
    return KWICResult(rows=rows, total=total, query=filters.query, filters=filters)


@dataclass
class CorpusStats:
    total_segments: int
    total_runs: int
    total_speakers: int
    total_hours: float


def corpus_stats(session: Session, project_id: uuid.UUID | None = None) -> CorpusStats:
    """Return headline counts for completed, non-archived corpus material."""
    run_ids = (
        select(PipelineRun.id.label("run_id"), PipelineRun.media_item_id)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
        )
    )
    if project_id is not None:
        run_ids = run_ids.join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id).where(
            MediaFolder.project_id == project_id
        )
    scoped_runs = run_ids.subquery()

    total_runs = int(session.execute(select(func.count()).select_from(scoped_runs)).scalar_one())
    total_segments = int(
        session.execute(
            select(func.count())
            .select_from(TranscriptSegment)
            .join(scoped_runs, scoped_runs.c.run_id == TranscriptSegment.pipeline_run_id)
        ).scalar_one()
    )
    total_speakers = int(
        session.execute(
            select(func.count(func.distinct(AdjudicationDecision.speaker_id)))
            .select_from(AdjudicationDecision)
            .join(
                scoped_runs,
                scoped_runs.c.run_id == AdjudicationDecision.pipeline_run_id,
            )
            .where(
                AdjudicationDecision.decision == "assign",
                AdjudicationDecision.speaker_id.isnot(None),
            )
        ).scalar_one()
    )
    scoped_media = select(scoped_runs.c.media_item_id).distinct().subquery()
    duration = session.execute(
        select(func.coalesce(func.sum(MediaItem.duration_seconds), 0.0))
        .select_from(MediaItem)
        .join(scoped_media, scoped_media.c.media_item_id == MediaItem.id)
    ).scalar_one()
    return CorpusStats(
        total_segments=total_segments,
        total_runs=total_runs,
        total_speakers=total_speakers,
        total_hours=float(duration or 0.0) / 3600.0,
    )
