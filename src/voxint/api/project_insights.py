"""Orchestration and caching for project-level insights (issue #336).

Aggregates RunEnrichmentAsset data (topics, entity_mentions) across a
project's canonical runs, builds a speakers x recordings coverage matrix,
and caches the result in a CorpusAnalysisArtifact.

Unlike speaker insights (Celery-only refresh), project insights use
compute-on-read: the project detail page calls get_project_insights(),
which checks the fingerprint and recomputes if stale. This avoids the
timing gap where enrichment assets are generated after run completion.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import label_states
from voxint.api.term_stats import source_hash
from voxint.db.models import (
    AdjudicationDecision,
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    Speaker,
    SpeakerAssignment,
)

logger = logging.getLogger(__name__)

_MAX_ENTITIES = 50
_MAX_TOPICS = 30
_MAX_QUOTES_PER_ENTITY = 5
_MAX_COVERAGE_SPEAKERS = 100
_MAX_COVERAGE_RECORDINGS = 200

_ALGORITHM_VERSION = "1"


def _advisory_lock_key(project_id: uuid.UUID) -> int:
    """Per-project advisory lock to prevent concurrent insight writes."""
    h = hashlib.sha256(f"voxint:project_insights:{project_id}".encode())
    return int(h.hexdigest()[:8], 16) & 0x7FFFFFFF


def _normalize(text: str) -> str:
    """NFKC + casefold + whitespace collapse for entity/topic normalization."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


# ---------------------------------------------------------------------------
# Canonical runs scoped to a project
# ---------------------------------------------------------------------------


def _canonical_project_runs(
    session: Session, project_id: uuid.UUID
) -> list[tuple[uuid.UUID, uuid.UUID, str]]:
    """Newest completed non-archived run per media item in the project.

    Returns (run_id, media_item_id, media_item_title) tuples.
    """
    ranked = (
        select(
            PipelineRun.id.label("run_id"),
            PipelineRun.media_item_id.label("media_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rank"),
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
        .where(
            MediaFolder.project_id == project_id,
            PipelineRun.status == RunStatus.COMPLETED.value,
            PipelineRun.archived_at.is_(None),
        )
        .subquery()
    )
    media_title = func.coalesce(
        MediaSourceMetadata.title, MediaItem.source_path
    ).label("media_title")
    rows = session.execute(
        select(ranked.c.run_id, ranked.c.media_id, media_title)
        .join(MediaItem, MediaItem.id == ranked.c.media_id)
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .where(ranked.c.rank == 1)
        .order_by(media_title, MediaItem.id)
    ).all()
    return [(row.run_id, row.media_id, row.media_title) for row in rows]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _project_fingerprint(
    session: Session,
    project_id: uuid.UUID,
    canonical: list[tuple[uuid.UUID, uuid.UUID, str]],
) -> str:
    """Composite fingerprint covering all inputs to project insights.

    Scoped to the project's canonical runs to avoid invalidation from
    unrelated corpus changes. Covers: canonical run membership, enrichment
    asset state, speaker assignments (cosine proposals used by
    label_states), adjudication decisions, speaker roster, and algorithm
    version.
    """
    run_ids = [run_id for run_id, _, _ in canonical]

    # 1. Canonical runs: IDs + titles (title change = coverage matrix change)
    runs_fp = source_hash([
        (str(run_id), f"{media_id}:{title}")
        for run_id, media_id, title in canonical
    ])

    # 2. Enrichment assets: IDs + completed_at of current (non-superseded)
    #    TOPICS and ENTITY_MENTIONS assets for canonical runs.
    asset_fp = ""
    if run_ids:
        asset_rows = session.execute(
            select(RunEnrichmentAsset.id, RunEnrichmentAsset.completed_at)
            .where(
                RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
                RunEnrichmentAsset.asset_kind.in_([
                    RunAssetKind.TOPICS.value,
                    RunAssetKind.ENTITY_MENTIONS.value,
                ]),
                RunEnrichmentAsset.superseded_by_asset_id.is_(None),
            )
            .order_by(RunEnrichmentAsset.id)
        ).all()
        asset_fp = source_hash([
            (str(row.id), str(row.completed_at)) for row in asset_rows
        ])

    # 3. Speaker assignments for canonical runs (cosine proposals used by
    #    label_states for coverage matrix resolution)
    assignment_fp = ""
    if run_ids:
        assignment_rows = session.execute(
            select(
                SpeakerAssignment.id,
                SpeakerAssignment.speaker_id,
                SpeakerAssignment.diarization_label,
                SpeakerAssignment.grounded,
            )
            .where(SpeakerAssignment.pipeline_run_id.in_(run_ids))
            .order_by(SpeakerAssignment.id)
        ).all()
        assignment_fp = source_hash([
            (
                str(row.id),
                f"{row.speaker_id or ''}:{row.diarization_label}:{row.grounded}",
            )
            for row in assignment_rows
        ])

    # 4. Adjudication decisions scoped to canonical runs
    decision_fp = ""
    if run_ids:
        latest_decision = session.execute(
            select(func.max(AdjudicationDecision.created_at))
            .where(AdjudicationDecision.pipeline_run_id.in_(run_ids))
        ).scalar_one()
        decision_fp = str(latest_decision or "")

    # 5. Speaker roster (only speakers referenced by canonical runs, plus
    #    their merge targets)
    referenced_speaker_ids: set[uuid.UUID] = set()
    if run_ids:
        ref_rows = session.execute(
            select(SpeakerAssignment.speaker_id)
            .where(
                SpeakerAssignment.pipeline_run_id.in_(run_ids),
                SpeakerAssignment.speaker_id.is_not(None),
            )
        ).scalars().all()
        referenced_speaker_ids = {
            sid for sid in ref_rows if sid is not None
        }
        # Include merge targets for canonical resolution
        if referenced_speaker_ids:
            merge_rows = session.execute(
                select(Speaker.merged_into_id)
                .where(
                    Speaker.id.in_(referenced_speaker_ids),
                    Speaker.merged_into_id.is_not(None),
                )
            ).scalars().all()
            referenced_speaker_ids.update(
                mid for mid in merge_rows if mid is not None
            )

    roster_fp = ""
    if referenced_speaker_ids:
        roster_rows = session.execute(
            select(
                Speaker.id, Speaker.display_name,
                Speaker.merged_into_id, Speaker.deleted_at,
            )
            .where(Speaker.id.in_(referenced_speaker_ids))
            .order_by(Speaker.id)
        ).all()
        roster_fp = source_hash([
            (
                str(row.id),
                f"{row.display_name}:{row.merged_into_id or ''}:{row.deleted_at or ''}",
            )
            for row in roster_rows
        ])

    return source_hash([
        ("version", _ALGORITHM_VERSION),
        ("runs", runs_fp),
        ("assets", asset_fp),
        ("assignments", assignment_fp),
        ("decisions", decision_fp),
        ("roster", roster_fp),
    ])


# ---------------------------------------------------------------------------
# Pure aggregation functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedEntity:
    surface: str
    display_surface: str
    kind: str | None
    total_occurrences: int
    runs_count: int
    sample_quotes: list[str]


@dataclass(frozen=True)
class AggregatedTopic:
    label: str
    display_label: str
    description: str | None
    avg_confidence: float | None
    confidence_count: int
    runs_count: int


def aggregate_entities(
    assets: list[dict[str, Any]],
) -> list[AggregatedEntity]:
    """Merge ENTITY_MENTIONS payloads across runs.

    Normalizes surface forms via NFKC + casefold. Keeps the first-seen
    display form. Resolves kind conflicts by majority vote (None excluded).
    Caps quotes per entity.
    """
    # key = normalized surface
    display: dict[str, str] = {}
    kinds: dict[str, list[str]] = defaultdict(list)
    occurrences: dict[str, int] = defaultdict(int)
    runs_seen: dict[str, set[int]] = defaultdict(set)
    quotes: dict[str, list[str]] = defaultdict(list)

    for asset_idx, payload in enumerate(assets):
        mentions = payload.get("mentions", [])
        for mention in mentions:
            raw_surface = mention.get("surface", "")
            if not raw_surface or not raw_surface.strip():
                continue
            key = _normalize(raw_surface)
            if not key:
                continue
            if key not in display:
                display[key] = raw_surface.strip()
            kind = mention.get("kind")
            if kind and isinstance(kind, str):
                kinds[key].append(kind.lower().strip())
            occ = mention.get("occurrences", [])
            occurrences[key] += max(len(occ), 1)
            runs_seen[key].add(asset_idx)
            for o in occ:
                quote = o.get("quote", "")
                if quote and len(quotes[key]) < _MAX_QUOTES_PER_ENTITY:
                    trimmed = quote.strip()
                    if trimmed and trimmed not in quotes[key]:
                        quotes[key].append(trimmed)

    entities = []
    for key in display:
        kind_list = kinds.get(key, [])
        if kind_list:
            from collections import Counter
            kind_counts = Counter(kind_list)
            resolved_kind = kind_counts.most_common(1)[0][0]
        else:
            resolved_kind = None
        entities.append(AggregatedEntity(
            surface=key,
            display_surface=display[key],
            kind=resolved_kind,
            total_occurrences=occurrences[key],
            runs_count=len(runs_seen[key]),
            sample_quotes=quotes[key],
        ))

    entities.sort(key=lambda e: (-e.total_occurrences, -e.runs_count, e.surface))
    return entities[:_MAX_ENTITIES]


def aggregate_topics(
    assets: list[dict[str, Any]],
) -> list[AggregatedTopic]:
    """Merge TOPICS payloads across runs.

    Normalizes labels via NFKC + casefold. Averages confidence (ignoring
    nulls). Keeps the first non-null description.
    """
    # key = normalized label
    display: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    confidences: dict[str, list[float]] = defaultdict(list)
    runs_seen: dict[str, set[int]] = defaultdict(set)

    for asset_idx, payload in enumerate(assets):
        topics = payload.get("topics", [])
        for topic in topics:
            raw_label = topic.get("label", "")
            if not raw_label or not raw_label.strip():
                continue
            key = _normalize(raw_label)
            if not key:
                continue
            if key not in display:
                display[key] = raw_label.strip()
            desc = topic.get("description")
            if desc and isinstance(desc, str) and key not in descriptions:
                descriptions[key] = desc.strip()
            conf = topic.get("confidence")
            if conf is not None and isinstance(conf, (int, float)):
                confidences[key].append(float(conf))
            runs_seen[key].add(asset_idx)

    topics_out = []
    for key in display:
        conf_list = confidences.get(key, [])
        avg_conf = sum(conf_list) / len(conf_list) if conf_list else None
        topics_out.append(AggregatedTopic(
            label=key,
            display_label=display[key],
            description=descriptions.get(key),
            avg_confidence=avg_conf,
            confidence_count=len(conf_list),
            runs_count=len(runs_seen[key]),
        ))

    topics_out.sort(key=lambda t: (-t.runs_count, -(t.avg_confidence or 0), t.label))
    return topics_out[:_MAX_TOPICS]


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageRow:
    speaker_id: uuid.UUID
    speaker_name: str | None


@dataclass(frozen=True)
class CoverageColumn:
    media_item_id: uuid.UUID
    title: str


@dataclass(frozen=True)
class CoverageMatrix:
    speakers: list[CoverageRow]
    recordings: list[CoverageColumn]
    cells: set[tuple[int, int]]
    truncated: bool


def build_coverage_matrix(
    session: Session,
    canonical: list[tuple[uuid.UUID, uuid.UUID, str]],
) -> CoverageMatrix:
    """Speakers x recordings presence matrix using canonical speaker resolution.

    Uses label_states per run to resolve to canonical speakers (follows
    merges, prefers human decisions). Raw diarization labels that don't
    resolve to a named speaker are excluded.
    """
    # Column index: media_item_id -> (col_idx, title)
    col_index: dict[uuid.UUID, int] = {}
    columns: list[CoverageColumn] = []
    for _, media_id, title in canonical[:_MAX_COVERAGE_RECORDINGS]:
        if media_id not in col_index:
            col_index[media_id] = len(columns)
            columns.append(CoverageColumn(media_item_id=media_id, title=title))

    # Walk runs, resolve speakers
    speaker_index: dict[uuid.UUID, int] = {}
    speakers: list[CoverageRow] = []
    cells: set[tuple[int, int]] = set()

    for run_id, media_id, _ in canonical[:_MAX_COVERAGE_RECORDINGS]:
        col_idx = col_index.get(media_id)
        if col_idx is None:
            continue
        for state in label_states(session, run_id):
            sid = state.speaker_id
            if sid is None:
                continue
            if sid not in speaker_index:
                if len(speakers) >= _MAX_COVERAGE_SPEAKERS:
                    continue
                speaker_index[sid] = len(speakers)
                speakers.append(CoverageRow(
                    speaker_id=sid,
                    speaker_name=state.speaker_name,
                ))
            row_idx = speaker_index[sid]
            cells.add((row_idx, col_idx))

    truncated = (
        len(canonical) > _MAX_COVERAGE_RECORDINGS
        or len(speaker_index) >= _MAX_COVERAGE_SPEAKERS
    )
    # Sort speakers: named first (alphabetical), then unnamed, stable by index
    named = [(i, s) for i, s in enumerate(speakers) if s.speaker_name]
    unnamed = [(i, s) for i, s in enumerate(speakers) if not s.speaker_name]
    named.sort(key=lambda pair: pair[1].speaker_name.lower())  # type: ignore[union-attr]

    old_to_new: dict[int, int] = {}
    sorted_speakers: list[CoverageRow] = []
    for old_idx, spk in named + unnamed:
        old_to_new[old_idx] = len(sorted_speakers)
        sorted_speakers.append(spk)
    remapped_cells = {(old_to_new[r], c) for r, c in cells}

    return CoverageMatrix(
        speakers=sorted_speakers,
        recordings=columns,
        cells=remapped_cells,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Compute + cache
# ---------------------------------------------------------------------------


def _fetch_enrichment_assets(
    session: Session, run_ids: list[uuid.UUID], kind: RunAssetKind
) -> list[dict[str, Any]]:
    """Fetch latest-generation non-superseded assets of the given kind."""
    if not run_ids:
        return []
    rows = session.execute(
        select(RunEnrichmentAsset.payload)
        .where(
            RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
            RunEnrichmentAsset.asset_kind == kind.value,
            RunEnrichmentAsset.superseded_by_asset_id.is_(None),
        )
        .order_by(RunEnrichmentAsset.pipeline_run_id)
    ).scalars().all()
    return list(rows)


def _build_payload(
    entities: list[AggregatedEntity],
    topics: list[AggregatedTopic],
    coverage: CoverageMatrix,
    enrichment_coverage: dict[str, int],
) -> dict[str, Any]:
    """Assemble the cached artifact payload."""
    return {
        "schema_version": 1,
        "entities": [
            {
                "surface": e.display_surface,
                "kind": e.kind,
                "total_occurrences": e.total_occurrences,
                "runs_count": e.runs_count,
                "sample_quotes": e.sample_quotes,
            }
            for e in entities
        ],
        "topics": [
            {
                "label": t.display_label,
                "description": t.description,
                "avg_confidence": (
                    round(t.avg_confidence, 3) if t.avg_confidence is not None else None
                ),
                "confidence_count": t.confidence_count,
                "runs_count": t.runs_count,
            }
            for t in topics
        ],
        "coverage": {
            "speakers": [
                {"id": str(s.speaker_id), "name": s.speaker_name}
                for s in coverage.speakers
            ],
            "recordings": [
                {"id": str(r.media_item_id), "title": r.title}
                for r in coverage.recordings
            ],
            "cells": sorted([r, c] for r, c in coverage.cells),
            "truncated": coverage.truncated,
        },
        "enrichment_coverage": enrichment_coverage,
    }


def compute_project_insights(
    session: Session, project_id: uuid.UUID
) -> dict[str, Any] | None:
    """Compute-on-read: return fresh project insights, recomputing if stale.

    Advisory-locked per project. Returns None for empty projects (no runs).
    """
    session.execute(
        select(func.pg_advisory_xact_lock(_advisory_lock_key(project_id)))
    )

    canonical = _canonical_project_runs(session, project_id)
    if not canonical:
        return None

    fingerprint = _project_fingerprint(session, project_id, canonical)

    # Check cache freshness
    cached = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
            CorpusAnalysisArtifact.source_hash == fingerprint,
        )
        .limit(1)
    ).scalar_one_or_none()
    if cached is not None:
        return cached.payload

    # Recompute
    run_ids = [run_id for run_id, _, _ in canonical]

    entity_assets = _fetch_enrichment_assets(session, run_ids, RunAssetKind.ENTITY_MENTIONS)
    topic_assets = _fetch_enrichment_assets(session, run_ids, RunAssetKind.TOPICS)

    entities = aggregate_entities(entity_assets)
    topics = aggregate_topics(topic_assets)
    coverage = build_coverage_matrix(session, canonical)

    enrichment_coverage = {
        "total_runs": len(canonical),
        "entity_runs": len(entity_assets),
        "topic_runs": len(topic_assets),
    }

    payload = _build_payload(entities, topics, coverage, enrichment_coverage)

    # Replace cached artifact
    session.execute(
        delete(CorpusAnalysisArtifact).where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
    )
    session.add(CorpusAnalysisArtifact(
        scope_kind="project",
        scope_id=project_id,
        artifact_kind=CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
        generation=1,
        source_hash=fingerprint,
        payload=payload,
    ))
    session.flush()
    logger.info(
        "Computed project insights for %s: %d entities, %d topics, %dx%d coverage",
        project_id,
        len(entities),
        len(topics),
        len(coverage.speakers),
        len(coverage.recordings),
    )
    return payload


def get_project_insights(
    session: Session, project_id: uuid.UUID
) -> dict[str, Any] | None:
    """Read path for the project detail page.

    Compute-on-read: checks fingerprint and recomputes if stale. This
    self-heals after enrichment generation, adjudication changes, speaker
    merges, project folder reassignment, and archiving.
    """
    return compute_project_insights(session, project_id)
