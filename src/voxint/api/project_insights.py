"""Orchestration and caching for project insights (issue #336)."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import label_states
from voxint.api.presentation import friendly_media_label
from voxint.api.project_stats import (
    CoverageItem,
    SpeakerPresence,
    build_project_insights_payload,
    compute_coverage_matrix,
    compute_entity_stats,
    compute_topic_stats,
)
from voxint.api.term_stats import source_hash
from voxint.db.models import (
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
)

logger = logging.getLogger(__name__)


def _advisory_lock_key(project_id: uuid.UUID) -> int:
    return (
        int(
            hashlib.sha256(
                f"voxint:project_insights:{project_id}".encode()
            ).hexdigest()[:8],
            16,
        )
        & 0x7FFFFFFF
    )


def _project_run_ids(session: Session, project_id: uuid.UUID) -> list[uuid.UUID]:
    return list(
        session.execute(
            select(PipelineRun.id)
            .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .join(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
            .where(
                MediaFolder.project_id == project_id,
                PipelineRun.status == RunStatus.COMPLETED.value,
                PipelineRun.archived_at.is_(None),
            )
            .order_by(PipelineRun.created_at.asc(), PipelineRun.id)
        ).scalars().all()
    )


def _project_fingerprint(
    session: Session,
    project_id: uuid.UUID,
    run_ids: list[uuid.UUID],
) -> str:
    if not run_ids:
        return source_hash([])

    run_rows = session.execute(
        select(PipelineRun.id, PipelineRun.updated_at)
        .where(PipelineRun.id.in_(run_ids))
        .order_by(PipelineRun.id)
    ).all()

    asset_rows = session.execute(
        select(
            RunEnrichmentAsset.pipeline_run_id,
            func.max(RunEnrichmentAsset.completed_at),
        )
        .where(RunEnrichmentAsset.pipeline_run_id.in_(run_ids))
        .group_by(RunEnrichmentAsset.pipeline_run_id)
    ).all()
    asset_max = {rid: str(completed) for rid, completed in asset_rows}

    return source_hash([
        (str(r.id), f"{r.updated_at}:{asset_max.get(r.id, '')}")
        for r in run_rows
    ])


def get_project_insights(
    session: Session, project_id: uuid.UUID
) -> dict[str, Any] | None:
    artifact = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
        .order_by(CorpusAnalysisArtifact.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
    if artifact is None:
        return None
    return artifact.payload


def _gather_enrichment_data(
    session: Session,
    run_ids: list[uuid.UUID],
    project_id: uuid.UUID,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    list[tuple[str, dict[str, Any]]],
    list[CoverageItem],
    int,
    int,
    int,
]:
    entity_enrichments: list[tuple[str, dict[str, Any]]] = []
    topic_enrichments: list[tuple[str, dict[str, Any]]] = []
    runs_with_entities: set[uuid.UUID] = set()
    runs_with_topics: set[uuid.UUID] = set()

    if run_ids:
        asset_rows = session.execute(
            select(RunEnrichmentAsset)
            .where(
                RunEnrichmentAsset.pipeline_run_id.in_(run_ids),
                RunEnrichmentAsset.superseded_by_asset_id.is_(None),
            )
        ).scalars().all()

        for asset in asset_rows:
            if asset.asset_kind == RunAssetKind.ENTITY_MENTIONS.value:
                entity_enrichments.append((str(asset.pipeline_run_id), asset.payload))
                runs_with_entities.add(asset.pipeline_run_id)
            elif asset.asset_kind == RunAssetKind.TOPICS.value:
                topic_enrichments.append((str(asset.pipeline_run_id), asset.payload))
                runs_with_topics.add(asset.pipeline_run_id)

    media_info: dict[uuid.UUID, tuple[uuid.UUID, str, int | None]] = {}
    if run_ids:
        rows = session.execute(
            select(
                PipelineRun.id,
                MediaItem.id,
                MediaSourceMetadata.title,
                MediaItem.source_path,
                MediaItem.duration_seconds,
            )
            .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
            .outerjoin(
                MediaSourceMetadata,
                MediaSourceMetadata.media_item_id == MediaItem.id,
            )
            .where(PipelineRun.id.in_(run_ids))
        ).all()
        for run_id, mi_id, meta_title, source_path, dur in rows:
            media_info[run_id] = (
                mi_id,
                friendly_media_label(meta_title, source_path),
                int(dur) if dur else None,
            )

    coverage_items: list[CoverageItem] = []
    for run_id in run_ids:
        info = media_info.get(run_id)
        if info is None:
            continue
        mi_id, mi_title, mi_dur = info

        states = label_states(session, run_id)
        by_speaker: dict[str, SpeakerPresence] = {}
        for state in states:
            if state.speaker_id is None:
                continue
            sid = str(state.speaker_id)
            existing = by_speaker.get(sid)
            if existing is None:
                by_speaker[sid] = SpeakerPresence(
                    speaker_id=sid,
                    label=state.speaker_name or state.label,
                    segment_count=state.turn_count,
                )
            else:
                existing.segment_count += state.turn_count
        speakers = list(by_speaker.values())

        coverage_items.append(CoverageItem(
            media_item_id=str(mi_id),
            run_id=str(run_id),
            title=mi_title,
            duration_s=mi_dur,
            speakers=speakers,
        ))

    return (
        entity_enrichments,
        topic_enrichments,
        coverage_items,
        len(run_ids),
        len(runs_with_entities),
        len(runs_with_topics),
    )


def _write_artifact(
    session: Session,
    project_id: uuid.UUID,
    fingerprint: str,
    payload: dict[str, Any],
) -> None:
    session.execute(
        delete(CorpusAnalysisArtifact).where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
    )
    session.add(
        CorpusAnalysisArtifact(
            scope_kind="project",
            scope_id=project_id,
            artifact_kind=CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            generation=1,
            source_hash=fingerprint,
            payload=payload,
        )
    )
    session.flush()


def project_insights(
    session: Session, project_id: uuid.UUID
) -> dict[str, Any]:
    run_ids = _project_run_ids(session, project_id)
    fingerprint = _project_fingerprint(session, project_id, run_ids)

    cached = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
        .order_by(CorpusAnalysisArtifact.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
    if cached is not None and cached.source_hash == fingerprint:
        return cached.payload

    session.execute(
        select(func.pg_advisory_xact_lock(_advisory_lock_key(project_id)))
    )

    rechecked = session.execute(
        select(CorpusAnalysisArtifact)
        .where(
            CorpusAnalysisArtifact.artifact_kind
            == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            CorpusAnalysisArtifact.scope_kind == "project",
            CorpusAnalysisArtifact.scope_id == project_id,
        )
        .order_by(CorpusAnalysisArtifact.generation.desc())
        .limit(1)
    ).scalar_one_or_none()
    if rechecked is not None and rechecked.source_hash == fingerprint:
        return rechecked.payload

    (
        entity_enrichments,
        topic_enrichments,
        coverage_items,
        run_count,
        runs_with_entities,
        runs_with_topics,
    ) = _gather_enrichment_data(session, run_ids, project_id)

    entities = compute_entity_stats(entity_enrichments)
    topics = compute_topic_stats(topic_enrichments)
    coverage = compute_coverage_matrix(coverage_items)

    payload = build_project_insights_payload(
        entities,
        topics,
        coverage,
        run_count=run_count,
        runs_with_entities=runs_with_entities,
        runs_with_topics=runs_with_topics,
    )

    _write_artifact(session, project_id, fingerprint, payload)
    return payload
