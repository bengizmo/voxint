"""Corpus exploration queries — KWIC concordance and term statistics."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import Date as SQLDate
from sqlalchemy import and_, case, cast, delete, func, or_, select, true
from sqlalchemy.orm import Session

from voxint.api.term_stats import TermStat, compute_tfidf, source_hash
from voxint.db import search
from voxint.db.models import (
    AdjudicationDecision,
    AnnotationTag,
    AnnotationTagLink,
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    Speaker,
    TranscriptAnnotation,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

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


@dataclass
class TagStat:
    """Corpus-wide (or project-scoped) count of live annotations per tag."""

    tag_id: str
    name: str
    color: int
    count: int


def tag_stats(session: Session, project_id: uuid.UUID | None = None) -> list[TagStat]:
    """Count live annotations per active tag, optionally scoped to a project.

    Synchronous SQL by design (issue #331 Phase 7): one indexed join over at
    most 8 tags per annotation, nothing like the TF-IDF cost that justified
    caching term_stats. Archived tags, soft-deleted annotations, and archived
    runs are excluded (the run filter keeps the rollup consistent with every
    other Explore scope query); annotations on runs still under review DO
    count (a highlight is evidence the moment it exists, not once the run
    completes), so deliberately no run STATUS filter.
    """
    stmt = (
        select(
            AnnotationTag.id,
            AnnotationTag.name,
            AnnotationTag.color,
            func.count(AnnotationTagLink.annotation_id).label("annotation_count"),
        )
        .join(AnnotationTagLink, AnnotationTagLink.tag_id == AnnotationTag.id)
        .join(
            TranscriptAnnotation,
            TranscriptAnnotation.id == AnnotationTagLink.annotation_id,
        )
        .join(PipelineRun, PipelineRun.id == TranscriptAnnotation.pipeline_run_id)
        .where(
            AnnotationTag.archived_at.is_(None),
            TranscriptAnnotation.deleted_at.is_(None),
            PipelineRun.archived_at.is_(None),
        )
        .group_by(AnnotationTag.id, AnnotationTag.name, AnnotationTag.color)
        .order_by(func.count(AnnotationTagLink.annotation_id).desc(), AnnotationTag.name.asc())
    )
    if project_id is not None:
        stmt = (
            stmt.join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(MediaFolder.project_id == project_id)
        )
    return [
        TagStat(
            tag_id=str(row.id), name=row.name, color=row.color, count=int(row.annotation_count)
        )
        for row in session.execute(stmt).all()
    ]


# ---------------------------------------------------------------------------
# Term statistics (issue #334): TF-IDF over the corpus, cached in
# corpus_analysis_artifacts.
# ---------------------------------------------------------------------------


@dataclass
class TermStatsResult:
    terms: list[dict[str, Any]]
    stale: bool


def _completed_runs_base(project_id: uuid.UUID | None = None) -> Any:
    """Base select of completed, non-archived run IDs (+ project filter)."""
    stmt = select(PipelineRun.id, PipelineRun.updated_at).where(
        PipelineRun.status == RunStatus.COMPLETED.value,
        PipelineRun.archived_at.is_(None),
    )
    if project_id is not None:
        stmt = (
            stmt.join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(MediaFolder.project_id == project_id)
        )
    return stmt


def _corpus_fingerprint(session: Session, project_id: uuid.UUID | None = None) -> str:
    """Fast staleness check covering both run changes AND text corrections.

    Hashes (run_id, run.updated_at, max_corrected_at) so operator corrections
    in SegmentReviewState invalidate the cache, not just new/archived runs.
    """
    rows = session.execute(
        _completed_runs_base(project_id).order_by(PipelineRun.id)
    ).all()
    if not rows:
        return source_hash([])

    run_ids = [r.id for r in rows]
    corr_stmt = (
        select(
            TranscriptSegment.pipeline_run_id,
            func.max(SegmentReviewState.corrected_at),
        )
        .join(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .where(
            TranscriptSegment.pipeline_run_id.in_(run_ids),
            SegmentReviewState.corrected_text.isnot(None),
        )
        .group_by(TranscriptSegment.pipeline_run_id)
    )
    corr_by_run: dict[uuid.UUID, str] = {
        rid: str(ts) for rid, ts in session.execute(corr_stmt).all()
    }
    return source_hash([
        (str(r.id), f"{r.updated_at}:{corr_by_run.get(r.id, '')}")
        for r in rows
    ])


def _corpus_documents(
    session: Session, project_id: uuid.UUID | None = None
) -> list[tuple[uuid.UUID, str]]:
    """Return (run_id, concatenated effective text) for each completed run."""
    run_ids_stmt = _completed_runs_base(project_id)
    run_ids = [r.id for r in session.execute(run_ids_stmt).all()]
    if not run_ids:
        return []

    effective_text = func.coalesce(
        SegmentReviewState.corrected_text,
        TranscriptSegment.enhanced_text,
        TranscriptSegment.raw_text,
    )
    stmt = (
        select(
            TranscriptSegment.pipeline_run_id.label("run_id"),
            func.string_agg(effective_text, " ").label("text"),
        )
        .outerjoin(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .where(TranscriptSegment.pipeline_run_id.in_(run_ids))
        .group_by(TranscriptSegment.pipeline_run_id)
    )
    return [(row.run_id, row.text or "") for row in session.execute(stmt)]


def _get_cached_artifact(
    session: Session, project_id: uuid.UUID | None = None
) -> CorpusAnalysisArtifact | None:
    """Latest cached term_stats artifact for this scope."""
    stmt = select(CorpusAnalysisArtifact).where(
        CorpusAnalysisArtifact.artifact_kind == CorpusAnalysisArtifactKind.TERM_STATS.value,
    )
    if project_id is not None:
        stmt = stmt.where(
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
    else:
        stmt = stmt.where(
            CorpusAnalysisArtifact.scope_kind == "corpus",
            CorpusAnalysisArtifact.scope_id.is_(None),
        )
    stmt = stmt.order_by(CorpusAnalysisArtifact.generation.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def _artifact_lock_key(scope_kind: str, project_id: uuid.UUID | None) -> int:
    """Deterministic advisory-lock key for a (scope_kind, scope_id) pair."""
    raw = f"term_stats:{scope_kind}:{project_id or ''}"
    # Hash before truncating: the first 8 raw bytes are the constant prefix
    # "term_sta", which collapsed every scope onto one lock (over-serializing,
    # never under-locking). Same fix as semantic_layout._lock_key.
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & 0x7FFFFFFF


def _write_artifact(
    session: Session,
    project_id: uuid.UUID | None,
    stats: list[TermStat],
    fingerprint: str,
) -> None:
    """Serialized delete-and-insert under an advisory lock.

    The advisory lock prevents concurrent requests from both deleting + both
    inserting (creating duplicate rows when scope_id IS NULL, where the
    unique constraint cannot help).
    """
    scope_kind = "project" if project_id is not None else "corpus"
    lock_key = _artifact_lock_key(scope_kind, project_id)
    session.execute(select(func.pg_advisory_xact_lock(lock_key)))

    # Recheck after lock: the winner may have already written a fresh artifact.
    existing = _get_cached_artifact(session, project_id)
    if existing is not None and existing.source_hash == fingerprint:
        return

    del_stmt = delete(CorpusAnalysisArtifact).where(
        CorpusAnalysisArtifact.artifact_kind == CorpusAnalysisArtifactKind.TERM_STATS.value,
        CorpusAnalysisArtifact.scope_kind == scope_kind,
    )
    if project_id is not None:
        del_stmt = del_stmt.where(CorpusAnalysisArtifact.scope_id == project_id)
    else:
        del_stmt = del_stmt.where(CorpusAnalysisArtifact.scope_id.is_(None))
    session.execute(del_stmt)

    session.add(
        CorpusAnalysisArtifact(
            scope_kind=scope_kind,
            scope_id=project_id,
            artifact_kind=CorpusAnalysisArtifactKind.TERM_STATS.value,
            generation=1,
            source_hash=fingerprint,
            payload={
                "terms": [
                    {
                        "term": s.term,
                        "count": s.count,
                        "doc_count": s.doc_count,
                        "tfidf": s.tfidf,
                    }
                    for s in stats
                ],
            },
        )
    )
    session.flush()


def term_stats(
    session: Session, project_id: uuid.UUID | None = None
) -> TermStatsResult:
    """Return term stats, computing and caching if stale or missing."""
    fingerprint = _corpus_fingerprint(session, project_id)
    cached = _get_cached_artifact(session, project_id)
    if cached is not None and cached.source_hash == fingerprint:
        return TermStatsResult(terms=cached.payload.get("terms", []), stale=False)

    docs = _corpus_documents(session, project_id)
    if not docs:
        return TermStatsResult(terms=[], stale=False)

    stats = compute_tfidf(docs)
    _write_artifact(session, project_id, stats, fingerprint)
    return TermStatsResult(
        terms=[
            {
                "term": s.term,
                "count": s.count,
                "doc_count": s.doc_count,
                "tfidf": s.tfidf,
            }
            for s in stats
        ],
        stale=False,
    )
