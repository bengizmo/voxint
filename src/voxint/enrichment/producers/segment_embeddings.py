"""Resolve → chunk → embed a run's transcript into vectors (issue #121).

The producer for the semantic-search spine. It reads the finished, operator-
effective transcript (the same attributed view the review console and exports
show, via :func:`voxint.adjudication.transcript.attributed_transcript` →
:func:`paragraphize_transcript`), splits it into token-bounded chunks
(:func:`voxint.embeddings.chunking.chunk_transcript`), and embeds each chunk in
process with the vendored MiniLM ONNX graph. No ``HttpLLMClient``, no
``llm_enabled`` gate, no egress — this is a local, additive artifact that never
touches ASR / diarization / TitaNet.

Two seams the job lane relies on:

- :func:`load_embedding_source` materializes the resolved transcript into a pure,
  session-free :class:`EmbeddingSource`. The job lane calls it inside a short
  ``REPEATABLE READ`` snapshot and commits that read transaction BEFORE the
  CPU-bound embed, so a correction committing mid-load can never produce a hybrid
  (torn-snapshot) vector index — a stronger guarantee than the run-asset lane
  needs, because these vectors are persisted, not transient.
- :func:`embedding_source_hash` is the staleness detector: a sha256 over the
  resolved grouping (effective text + timing + attributed speaker + dominant
  rendering, per paragraph). A correction, split, rename, or re-adjudication
  changes the resolved transcript and therefore the hash, marking the run's
  index stale. Chunking is a deterministic function of this grouping plus the
  fixed tokenizer/max-len (both encoded in ``EMBEDDING_SPACE``), so hashing the
  grouping is sufficient — a chunking or model change is a new space, a visible
  re-index, never silent drift.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import review_states
from voxint.adjudication.transcript import (
    TranscriptLine,
    TranscriptText,
    attributed_transcript,
    paragraphize_transcript,
)
from voxint.db.models import PipelineRun, TranscriptSegment
from voxint.embeddings.chunking import ParagraphInput, chunk_transcript
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE, TextEmbedder

# Bump only for a change that must invalidate every run's index regardless of
# content (e.g. a change to what the grouping serialization covers). A model /
# pooling / max-len / chunking change is a new EMBEDDING_SPACE instead, so those
# do not touch this constant.
SOURCE_SCHEMA_VERSION = 1


class EmbeddingError(Exception):
    """The transcript cannot be resolved into an embeddable source."""


@dataclass(frozen=True)
class EmbeddingSource:
    """The resolved, session-free transcript grouping for one run.

    Everything the producer reads, materialized in memory so the DB read
    transaction can close before the CPU embed. ``paragraphs`` are the reading
    paragraphs (one speaker's consecutive lines merged) the read mode and
    Markdown export share, each carrying its dominant rendering.
    """

    pipeline_run_id: uuid.UUID
    paragraphs: tuple[ParagraphInput, ...]


@dataclass(frozen=True)
class ChunkEmbedding:
    """One embedded chunk, ready to become a ``SegmentEmbedding`` row."""

    chunk_index: int
    start_seconds: float
    end_seconds: float
    speaker_label: str | None
    text_rendering: str
    chunk_text: str
    content_hash: str
    embedding: list[float]


def _paragraph_renderings(
    lines: list[TranscriptLine], rendering_by_segment: dict[uuid.UUID, str]
) -> list[str]:
    """The dominant rendering per reading paragraph, aligned to
    :func:`paragraphize_transcript`.

    Walks ``lines`` with the identical adjacent-same-speaker grouping rule, so
    the returned list is 1:1 index-aligned with the paragraphs that function
    returns. "Dominant" is the rendering covering the most characters within the
    paragraph (deterministic ties broken by the corrected → enhanced → raw
    precedence), so a paragraph mixing renderings records the one that supplies
    most of its text.
    """
    precedence = {
        TranscriptText.CORRECTED.value: 3,
        TranscriptText.ENHANCED.value: 2,
        TranscriptText.RAW.value: 1,
    }
    renderings: list[str] = []
    group: list[TranscriptLine] = []

    def flush() -> None:
        if not group:
            return
        weight: dict[str, int] = {}
        for ln in group:
            key = ln.source_segment_id or ln.segment_id
            rendering = (
                rendering_by_segment.get(key, TranscriptText.RAW.value)
                if key is not None
                else TranscriptText.RAW.value
            )
            weight[rendering] = weight.get(rendering, 0) + len(ln.text)
        # Most characters wins; a tie falls to the higher-precedence rendering.
        dominant = max(weight, key=lambda r: (weight[r], precedence.get(r, 0)))
        renderings.append(dominant)

    for line in lines:
        if group and line.speaker != group[0].speaker:
            flush()
            group = []
        group.append(line)
    flush()
    return renderings


def load_embedding_source(session: Session, pipeline_run_id: uuid.UUID) -> EmbeddingSource:
    """Materialize the run's resolved transcript into a pure embedding source.

    Resolves speakers and effective text exactly as the console/export do
    (:func:`attributed_transcript` with the default CORRECTED view), groups into
    reading paragraphs, and derives each paragraph's dominant rendering. Raises
    :class:`EmbeddingError` for an unknown run or one with no transcript yet
    (there is nothing to embed).

    Call inside a short consistent-snapshot read transaction: the resolution
    spans several statements, and a persisted hybrid index would be worse than a
    one-off stale marking.
    """
    run = session.get(PipelineRun, pipeline_run_id)
    if run is None:
        raise EmbeddingError(f"unknown pipeline run: {pipeline_run_id}")
    lines = attributed_transcript(session, pipeline_run_id, text=TranscriptText.CORRECTED)
    if not lines:
        raise EmbeddingError(
            "run has no transcript segments yet — the embedding index is built"
            " from the transcript, so the run must finish transcription first"
        )
    # Per-segment rendering: enhanced-text presence + whether a correction exists.
    enhanced_present: dict[uuid.UUID, bool] = {
        seg_id: bool(has_enhanced)
        for seg_id, has_enhanced in session.execute(
            select(TranscriptSegment.id, TranscriptSegment.enhanced_text.isnot(None)).where(
                TranscriptSegment.pipeline_run_id == pipeline_run_id
            )
        ).all()
    }
    corrected_ids = set(review_states(session, pipeline_run_id))
    rendering_by_segment = {
        seg_id: (
            TranscriptText.CORRECTED.value
            if seg_id in corrected_ids
            else (
                TranscriptText.ENHANCED.value
                if enhanced_present.get(seg_id)
                else TranscriptText.RAW.value
            )
        )
        for seg_id in enhanced_present
    }

    paragraphs = paragraphize_transcript(lines)
    renderings = _paragraph_renderings(lines, rendering_by_segment)
    # The two walks share paragraphize_transcript's grouping rule, so they align
    # 1:1; assert it so a future change to that rule can never silently mislabel
    # renderings.
    if len(paragraphs) != len(renderings):  # pragma: no cover - alignment guard
        raise EmbeddingError(
            "paragraph/rendering alignment broke — the grouping rule changed"
        )
    inputs = tuple(
        ParagraphInput(
            speaker=para.speaker,
            start_seconds=para.start_seconds,
            end_seconds=para.end_seconds,
            text=para.text,
            text_rendering=rendering,
        )
        for para, rendering in zip(paragraphs, renderings, strict=True)
    )
    return EmbeddingSource(pipeline_run_id=pipeline_run_id, paragraphs=inputs)


def embedding_source_hash(source: EmbeddingSource) -> str:
    """sha256 over the canonical serialization of the resolved grouping.

    Content-only and deterministic (stable key order, compact separators): a
    correction, split, rename, or re-adjudication flips it, a code upgrade does
    not. The chunking algorithm and model are NOT folded in — they are encoded in
    ``EMBEDDING_SPACE``, so a change there is a new space (a visible re-index),
    not a hash change within the current one.
    """
    payload = {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "pipeline_run_id": str(source.pipeline_run_id),
        "paragraphs": [
            [
                para.speaker,
                para.start_seconds,
                para.end_seconds,
                para.text,
                para.text_rendering,
            ]
            for para in source.paragraphs
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def produce_segment_embeddings(
    source: EmbeddingSource, embedder: TextEmbedder
) -> list[ChunkEmbedding]:
    """Chunk the resolved paragraphs and embed each chunk into a unit vector.

    Pure CPU: no DB, no session — call it OUTSIDE the read snapshot and the
    publish transaction. Returns one :class:`ChunkEmbedding` per chunk, in
    run-wide ``chunk_index`` order (empty when the transcript is all blank).
    """
    chunks = chunk_transcript(list(source.paragraphs), embedder.count_tokens)
    if not chunks:
        return []
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    if vectors.shape[0] != len(chunks):  # pragma: no cover - embedder contract
        raise EmbeddingError(
            f"embedder returned {vectors.shape[0]} vectors for {len(chunks)} chunks"
        )
    results: list[ChunkEmbedding] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        results.append(
            ChunkEmbedding(
                chunk_index=chunk.chunk_index,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                speaker_label=chunk.speaker_label,
                text_rendering=chunk.text_rendering,
                chunk_text=chunk.text,
                content_hash=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                embedding=[float(v) for v in vector],
            )
        )
    return results


# The text embedding space these vectors live in — cosine is only valid within
# it. Re-exported so the job lane and the PR2 query path share one constant.
__all__ = [
    "EMBEDDING_SPACE",
    "SOURCE_SCHEMA_VERSION",
    "ChunkEmbedding",
    "EmbeddingError",
    "EmbeddingSource",
    "embedding_source_hash",
    "load_embedding_source",
    "produce_segment_embeddings",
]
