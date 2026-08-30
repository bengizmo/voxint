""""More like this passage": nearest-neighbour search from a KWIC row (#357).

Given one transcript segment, find the corpus passages closest to it in the
embedding space the spine already maintains. The segment's operator-effective
text (corrected > enhanced > raw, the same precedence every read surface uses)
is re-embedded at query time with the in-process MiniLM ONNX embedder — the
same path Meaning search uses for typed queries. Re-embedding, not a covering-
chunk lookup, because split pieces of a long paragraph share identical
timestamps, so an interval lookup can silently pick the wrong piece; the
embedded text is resolved server-side from the segment id, never trusted from
the browser.

The one exception is a very short segment: a MiniLM vector of a few words is
noisy, so below a small token floor the covering chunk's stored vector stands
in (the paragraph context is a better description of the moment than the
fragment itself). If the run has no index yet, the short text is embedded
anyway — a weak query against the corpus still beats refusing.

Shaping guards against the near-duplicate trap chunk overlap creates: every
source-run chunk whose interval overlaps the segment's is excluded, which
removes the originating paragraph AND its overlapping split pieces. Other
passages from the same recording are kept (recurring themes inside one long
interview are legitimate hits) but capped per run. Raw cosine scores are not
returned: MiniLM distance is not a calibrated relevance scale, and presenting
it as one would mislead the audience this tool serves.

The pure shaping (:func:`shape_similar`) is DB-free so it unit-tests without
Postgres, mirroring :func:`voxint.api.meaning_query.rank_candidates`.
"""

import enum
import uuid
from dataclasses import dataclass, field, replace

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.meaning_query import (
    Candidate,
    base_columns,
    jump_url,
    make_candidate,
    with_run_joins,
)
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_semantic_index_enabled,
)
from voxint.config import Settings
from voxint.db.models import (
    PipelineRun,
    SegmentEmbedding,
    SegmentReviewState,
    TranscriptSegment,
)
from voxint.embeddings.onnx_embedder import (
    EMBEDDING_SPACE,
    TextEmbedder,
    get_text_embedder,
    minilm_artifacts_available,
)

# The exact-cosine candidate fetch is wider than the shaped page so the
# overlap/dedupe/per-run shaping still has material to work with after cuts.
SIMILAR_CANDIDATE_LIMIT = 100
SIMILAR_TOP_K = 10
SIMILAR_PER_RUN_CAP = 2
# Below this many whitespace tokens the segment's own vector is too noisy and
# the covering chunk's stored vector stands in when one exists.
MIN_QUERY_TOKENS = 15
# Plain-text preview budget per result (the island renders it as a text node).
_PREVIEW_CHARS = 200


class SimilarSearchState(enum.StrEnum):
    """Why the panel rendered what it did — drives an honest UI state."""

    OK = "ok"  # ran; ``items`` may still be empty (nothing else is close)
    OFF = "off"  # the semantic-index feature is disabled
    UNAVAILABLE = "unavailable"  # enabled, but the model weights are absent
    INDEXING = "indexing"  # enabled, but no passages are indexed yet
    NOT_FOUND = "not_found"  # unknown segment id
    EMPTY_TEXT = "empty_text"  # the segment has no effective text to match


@dataclass(frozen=True)
class SimilarPassage:
    """One similar passage, shaped for the JSON response."""

    run_id: uuid.UUID
    title: str | None
    speaker_label: str | None
    start_seconds: float
    end_seconds: float
    preview: str
    jump_url: str


@dataclass(frozen=True)
class SimilarResultsPage:
    state: SimilarSearchState
    items: list[SimilarPassage] = field(default_factory=list)


def shape_similar(
    candidates: list[Candidate],
    *,
    source_run_id: uuid.UUID,
    source_start: float,
    source_end: float,
    per_run_cap: int = SIMILAR_PER_RUN_CAP,
    top_k: int = SIMILAR_TOP_K,
) -> list[Candidate]:
    """Exclude the source span, dedupe, cap per run — the pure shaping, DB-free.

    Exclusion is interval overlap against the SOURCE RUN only: the originating
    paragraph's chunks (identical timestamps across split pieces) and any
    overlap-sharing neighbours disappear; non-overlapping passages from the
    same recording stay. Candidates sharing an exact (run, start, end) span are
    collapsed to the closest — split pieces of one paragraph are one moment,
    not several results. Order is deterministic: cosine distance, then a stable
    ``(run_id, chunk_index)``.
    """
    best_by_span: dict[tuple[uuid.UUID, float, float], Candidate] = {}
    for c in candidates:
        if (
            c.run_id == source_run_id
            and c.start_seconds < source_end
            and c.end_seconds > source_start
        ):
            continue
        span = (c.run_id, c.start_seconds, c.end_seconds)
        held = best_by_span.get(span)
        if held is None or _distance(c) < _distance(held):
            best_by_span[span] = c

    ordered = sorted(
        best_by_span.values(),
        key=lambda c: (_distance(c), str(c.run_id), c.chunk_index),
    )
    per_run: dict[uuid.UUID, int] = {}
    kept: list[Candidate] = []
    for candidate in ordered:
        if per_run.get(candidate.run_id, 0) >= per_run_cap:
            continue
        per_run[candidate.run_id] = per_run.get(candidate.run_id, 0) + 1
        kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept


def _distance(c: Candidate) -> float:
    return c.distance if c.distance is not None else float("inf")


def _preview(chunk_text: str) -> str:
    text = " ".join(chunk_text.split())
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS].rstrip() + " …"


@dataclass(frozen=True)
class _SourceSegment:
    """What the query needs from the clicked segment, read in one short session."""

    run_id: uuid.UUID
    start_seconds: float
    end_seconds: float
    effective_text: str
    covering_chunk_vector: list[float] | None


def _load_source(session: Session, segment_id: uuid.UUID) -> _SourceSegment | None:
    seg = session.get(TranscriptSegment, segment_id)
    if seg is None:
        return None
    review = session.get(SegmentReviewState, segment_id)
    corrected = review.corrected_text if review is not None else None
    effective = corrected or seg.enhanced_text or seg.raw_text or ""

    covering: list[float] | None = None
    if len(effective.split()) < MIN_QUERY_TOKENS:
        row = session.execute(
            select(SegmentEmbedding.embedding)
            .where(
                SegmentEmbedding.pipeline_run_id == seg.pipeline_run_id,
                SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
                SegmentEmbedding.start_seconds <= seg.start_seconds,
                SegmentEmbedding.end_seconds >= seg.start_seconds,
            )
            .order_by(SegmentEmbedding.chunk_index.asc())
            .limit(1)
        ).first()
        if row is not None:
            covering = list(row.embedding)

    return _SourceSegment(
        run_id=seg.pipeline_run_id,
        start_seconds=seg.start_seconds,
        end_seconds=seg.end_seconds,
        effective_text=effective.strip(),
        covering_chunk_vector=covering,
    )


def similar_passages(
    session_factory: sessionmaker[Session],
    *,
    settings: Settings,
    segment_id: uuid.UUID,
    embedder: TextEmbedder | None = None,
    top_k: int = SIMILAR_TOP_K,
    per_run_cap: int = SIMILAR_PER_RUN_CAP,
    candidate_limit: int = SIMILAR_CANDIDATE_LIMIT,
) -> SimilarResultsPage:
    """Find the corpus passages nearest this segment, or an honest empty state.

    Mirrors :func:`voxint.api.meaning_query.search_passages`: gates first, the
    embed happens on pure CPU between the two short read transactions (never
    holding a DB snapshot), and the scan reads one REPEATABLE READ snapshot so
    a concurrent embedding publish cannot straddle it.
    """
    with session_factory() as gate_session:
        if not resolve_effective_semantic_index_enabled(get_app_settings(gate_session), settings):
            return SimilarResultsPage(SimilarSearchState.OFF)
        source = _load_source(gate_session, segment_id)

    if source is None:
        return SimilarResultsPage(SimilarSearchState.NOT_FOUND)

    query_vector: list[float]
    if source.covering_chunk_vector is not None:
        query_vector = source.covering_chunk_vector
    else:
        if not source.effective_text:
            return SimilarResultsPage(SimilarSearchState.EMPTY_TEXT)
        resolved_embedder = embedder
        if resolved_embedder is None:
            if not minilm_artifacts_available():
                return SimilarResultsPage(SimilarSearchState.UNAVAILABLE)
            resolved_embedder = get_text_embedder()
        query_vector = resolved_embedder.embed_texts([source.effective_text])[0].tolist()

    with session_factory() as session:
        session.connection(
            execution_options={
                "isolation_level": "REPEATABLE READ",
                "postgresql_readonly": True,
            }
        )
        indexed = session.execute(
            select(SegmentEmbedding.id)
            .join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
            .where(
                SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
                PipelineRun.archived_at.is_(None),
            )
            .limit(1)
        ).first()
        if indexed is None:
            return SimilarResultsPage(SimilarSearchState.INDEXING)

        distance = SegmentEmbedding.embedding.cosine_distance(query_vector)
        stmt = (
            with_run_joins(select(*base_columns(), distance.label("distance")))
            .order_by(distance.asc(), SegmentEmbedding.id.asc())
            .limit(candidate_limit)
        )
        candidates = [
            replace(make_candidate(row), vector_rank=rank, distance=row.distance)
            for rank, row in enumerate(session.execute(stmt), start=1)
        ]

    shaped = shape_similar(
        candidates,
        source_run_id=source.run_id,
        source_start=source.start_seconds,
        source_end=source.end_seconds,
        per_run_cap=per_run_cap,
        top_k=top_k,
    )
    items = [
        SimilarPassage(
            run_id=c.run_id,
            title=c.title,
            speaker_label=c.speaker_label,
            start_seconds=c.start_seconds,
            end_seconds=c.end_seconds,
            preview=_preview(c.chunk_text),
            jump_url=jump_url(c.run_id, c.start_seconds),
        )
        for c in shaped
    ]
    return SimilarResultsPage(SimilarSearchState.OK, items)
