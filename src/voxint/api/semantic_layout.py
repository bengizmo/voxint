"""Semantic meaning map: a 2D layout of the corpus chunk embeddings (#357).

An orientation view for theme discovery: every indexed passage becomes one
point, placed by projecting its 384-dim MiniLM vector onto the corpus's top
two principal components. PCA, deliberately not UMAP: at single-operator
scale (hundreds to a few thousand points) PCA is deterministic, instant, and
dependency-free — numpy is already in the closure — where umap-learn would
drag numba + llvmlite in for a stochastic layout that rearranges between
visits. The payload records the method so the algorithm-neutral artifact kind
(``semantic_layout``) never lies about what produced it.

Caching follows :func:`voxint.api.explore_query.term_stats` exactly:
compute-on-read into ``corpus_analysis_artifacts``, a fingerprint over the
indexed corpus for staleness, and a per-scope advisory lock (recheck after
acquiring) so concurrent first requests cannot double-write. The fingerprint
covers run membership, per-run embedding generation and row count, the
embedding space id (a model swap is a new space), and the algorithm tag —
so a re-index, an archive, or an algorithm change each invalidate it.

The layout is capped: runs are sampled round-robin (stable hash order within
each run) so one long recording cannot consume the whole map. The payload is
self-contained — an embedding regeneration deletes the rows it was built
from, so every point carries what the canvas needs and never joins back.

Coordinates are not durable identifiers: adding or correcting material
changes the principal components and every point may move. Per-component
sign fixing keeps recomputes from mirroring, nothing more.

Accepted staleness: the fingerprint covers the indexed corpus (run
membership, generation, row counts, embedding space, algorithm), not the
denormalized display metadata inside cached points. An edited media title
keeps showing its old text in tooltips until the next corpus change
recomputes the layout.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from voxint.api.meaning_query import jump_url
from voxint.api.presentation import title_from_snapshot
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_semantic_index_enabled,
)
from voxint.config import Settings
from voxint.db.models import (
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    SegmentEmbedding,
)
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE

# Identifies the layout algorithm + parameters in the fingerprint and payload;
# bump when either changes so cached layouts recompute.
LAYOUT_METHOD = "pca"
LAYOUT_VERSION = 1
# Hard point cap: keeps the JSONB artifact, the JSON response, and the canvas
# draw loop bounded. Sampling is run-stratified, not truncation.
MAX_POINTS = 3000
# A layout of fewer points than this is noise dressed up as structure.
MIN_POINTS = 5
_PREVIEW_CHARS = 160

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticLayoutResult:
    """The map payload plus an honest state for the UI."""

    state: str  # "ok" | "off" | "insufficient"
    payload: dict[str, Any] = field(default_factory=dict)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows onto the top two principal components, deterministically.

    Pure numpy: mean-center, eigendecompose the 384x384 covariance (`eigh`,
    exact and stable at this width), take the two largest-eigenvalue
    components. Sign is fixed per component — the largest-magnitude loading
    is made positive — so a recompute over similar data cannot mirror the map.
    Raises ``ValueError`` for degenerate input (too few rows, NaN, or a
    rank-deficient corpus with no second component).
    """
    if matrix.ndim != 2 or matrix.shape[0] < MIN_POINTS:
        raise ValueError("too few points for a meaningful layout")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite values in embedding matrix")
    centered = matrix - matrix.mean(axis=0)
    cov = (centered.T @ centered) / (matrix.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1][:2]
    components = eigenvectors[:, order]  # (dim, 2)
    # Relative threshold: a rank-deficient corpus yields a second eigenvalue
    # that is float noise (~1e-17 of the first), not an exact zero.
    lead = float(eigenvalues[order[0]])
    if lead <= 0 or float(eigenvalues[order[1]]) <= lead * 1e-9:
        raise ValueError("corpus has no second principal component")
    for j in range(2):
        pivot = np.argmax(np.abs(components[:, j]))
        if components[pivot, j] < 0:
            components[:, j] = -components[:, j]
    return centered @ components


def stratified_sample(
    ids_by_run: dict[uuid.UUID, list[uuid.UUID]], cap: int = MAX_POINTS
) -> set[uuid.UUID]:
    """Round-robin across runs, stable-hash order within each — deterministic.

    Fills one point per run per round until the cap, so a ten-hour recording
    cannot crowd short ones off the map. Within a run the order is a sha256 of
    the row id (spread across the timeline, stable across recomputes), and
    the run rotation is ordered by run id.
    """

    def row_key(row_id: uuid.UUID) -> str:
        return hashlib.sha256(str(row_id).encode()).hexdigest()

    queues = {
        run_id: sorted(ids, key=row_key)
        for run_id, ids in ids_by_run.items()
    }
    chosen: set[uuid.UUID] = set()
    run_order = sorted(queues, key=str)
    cursor = dict.fromkeys(run_order, 0)
    while len(chosen) < cap:
        progressed = False
        for run_id in run_order:
            queue = queues[run_id]
            i = cursor[run_id]
            if i >= len(queue):
                continue
            chosen.add(queue[i])
            cursor[run_id] = i + 1
            progressed = True
            if len(chosen) >= cap:
                break
        if not progressed:
            break
    return chosen


def _layout_fingerprint(session: Session, project_id: uuid.UUID | None) -> str:
    """Staleness signature over the indexed, visible corpus for this scope."""
    stmt = (
        select(
            SegmentEmbedding.pipeline_run_id,
            SegmentEmbedding.generation,
            func.count(SegmentEmbedding.id),
        )
        .join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
        .where(
            SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
            PipelineRun.archived_at.is_(None),
        )
        .group_by(SegmentEmbedding.pipeline_run_id, SegmentEmbedding.generation)
        .order_by(SegmentEmbedding.pipeline_run_id)
    )
    if project_id is not None:
        stmt = (
            stmt.join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(MediaFolder.project_id == project_id)
        )
    rows = session.execute(stmt).all()
    basis = "|".join(
        [f"{LAYOUT_METHOD}/{LAYOUT_VERSION}", EMBEDDING_SPACE, str(MAX_POINTS)]
        + [f"{run_id}:{generation}:{count}" for run_id, generation, count in rows]
    )
    return hashlib.sha256(basis.encode()).hexdigest()


def _get_cached(
    session: Session, project_id: uuid.UUID | None
) -> CorpusAnalysisArtifact | None:
    stmt = select(CorpusAnalysisArtifact).where(
        CorpusAnalysisArtifact.artifact_kind
        == CorpusAnalysisArtifactKind.SEMANTIC_LAYOUT.value,
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


def _lock_key(scope_kind: str, project_id: uuid.UUID | None) -> int:
    # Hash the WHOLE scope string: truncating the raw bytes would key every
    # scope off the identical "semantic" prefix, collapsing per-scope locking
    # into one global lock.
    raw = f"semantic_layout:{scope_kind}:{project_id or ''}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & 0x7FFFFFFF


def _write_artifact(
    session: Session,
    project_id: uuid.UUID | None,
    payload: dict[str, Any],
    fingerprint: str,
) -> None:
    """Serialized delete-and-insert under an advisory lock, recheck after."""
    scope_kind = "project" if project_id is not None else "corpus"
    session.execute(select(func.pg_advisory_xact_lock(_lock_key(scope_kind, project_id))))

    existing = _get_cached(session, project_id)
    if existing is not None and existing.source_hash == fingerprint:
        return

    del_stmt = delete(CorpusAnalysisArtifact).where(
        CorpusAnalysisArtifact.artifact_kind
        == CorpusAnalysisArtifactKind.SEMANTIC_LAYOUT.value,
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
            artifact_kind=CorpusAnalysisArtifactKind.SEMANTIC_LAYOUT.value,
            generation=1,
            source_hash=fingerprint,
            payload=payload,
        )
    )
    session.flush()


def _preview(chunk_text: str) -> str:
    text = " ".join(chunk_text.split())
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS].rstrip() + " …"


def _load_points(
    session: Session, project_id: uuid.UUID | None
) -> tuple[list[dict[str, Any]], int]:
    """Load, sample, and project the indexed corpus; returns (points, total)."""
    id_stmt = (
        select(SegmentEmbedding.id, SegmentEmbedding.pipeline_run_id)
        .join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
        .where(
            SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
            PipelineRun.archived_at.is_(None),
        )
    )
    if project_id is not None:
        id_stmt = (
            id_stmt.join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(MediaFolder.project_id == project_id)
        )
    ids_by_run: dict[uuid.UUID, list[uuid.UUID]] = {}
    total = 0
    for row_id, run_id in session.execute(id_stmt):
        ids_by_run.setdefault(run_id, []).append(row_id)
        total += 1
    if total == 0:
        return [], 0

    chosen = stratified_sample(ids_by_run)
    row_stmt = (
        select(
            SegmentEmbedding.id,
            SegmentEmbedding.pipeline_run_id,
            SegmentEmbedding.speaker_label,
            SegmentEmbedding.start_seconds,
            SegmentEmbedding.end_seconds,
            SegmentEmbedding.chunk_text,
            SegmentEmbedding.embedding,
            MediaItem.source_path,
            PipelineRun.sidecar,
            MediaSourceMetadata.title.label("source_title"),
        )
        .join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .where(SegmentEmbedding.id.in_(chosen))
        .order_by(SegmentEmbedding.id)
    )
    rows = session.execute(row_stmt).all()
    matrix = np.asarray([list(r.embedding) for r in rows], dtype=np.float64)
    coords = pca_2d(matrix)
    points = [
        {
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "run_id": str(r.pipeline_run_id),
            "media_title": title_from_snapshot(r.sidecar) or r.source_title or r.source_path,
            "speaker_label": r.speaker_label,
            "start_seconds": r.start_seconds,
            "end_seconds": r.end_seconds,
            "preview": _preview(r.chunk_text),
            "jump_url": jump_url(r.pipeline_run_id, r.start_seconds),
        }
        for r, (x, y) in zip(rows, coords, strict=True)
    ]
    return points, total


def semantic_layout(
    session: Session,
    settings: Settings,
    project_id: uuid.UUID | None = None,
) -> SemanticLayoutResult:
    """Return the map for this scope, computing and caching if stale or missing.

    The fingerprint and the point load are separate READ COMMITTED statements,
    so a concurrent embedding publish could straddle them and key a torn
    payload under the wrong hash. The load is therefore bracketed: the
    fingerprint is recomputed after loading and the pass retries (bounded)
    until the two agree; only a verified pairing is cached. A layout computed
    from a corpus that will not hold still is never written.
    """
    if not resolve_effective_semantic_index_enabled(get_app_settings(session), settings):
        return SemanticLayoutResult(state="off")

    fingerprint = _layout_fingerprint(session, project_id)
    cached = _get_cached(session, project_id)
    if cached is not None and cached.source_hash == fingerprint:
        return SemanticLayoutResult(state="ok", payload=cached.payload)

    points: list[dict[str, Any]] = []
    total = 0
    verified = False
    for _ in range(3):
        try:
            points, total = _load_points(session, project_id)
        except ValueError as exc:
            # pca_2d's documented degenerate inputs (too few points, rank
            # deficiency, non-finite values) all land here; log the reason so
            # genuine data corruption is visible in ops, not silently dressed
            # up as a small corpus.
            logger.warning("semantic layout not computable: %s", exc)
            return SemanticLayoutResult(state="insufficient")
        recheck = _layout_fingerprint(session, project_id)
        if recheck == fingerprint:
            verified = True
            break
        fingerprint = recheck

    if len(points) < MIN_POINTS:
        return SemanticLayoutResult(state="insufficient")

    payload = {
        "method": LAYOUT_METHOD,
        "version": LAYOUT_VERSION,
        "total_n": total,
        "shown_n": len(points),
        "sampled": total > len(points),
        "points": points,
    }
    if verified:
        _write_artifact(session, project_id, payload, fingerprint)
    else:
        logger.warning(
            "semantic layout fingerprint kept moving; serving uncached result"
        )
    return SemanticLayoutResult(state="ok", payload=payload)
