"""Project temporal-trend caching over real PostgreSQL (issue #337)."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import set_correction
from voxint.api.temporal_trends import (
    _temporal_lock_key,
    compute_temporal_trends,
)
from voxint.db.models import (
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    Project,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    SegmentReviewState,
    TranscriptSegment,
)

NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


def _seed_project(
    session: Session, *, text_value: str = "alpha Acme"
) -> tuple[uuid.UUID, uuid.UUID]:
    project = Project(name=f"Temporal {uuid.uuid4()}")
    session.add(project)
    session.flush()
    folder = MediaFolder(path=f"temporal/{uuid.uuid4()}", project_id=project.id)
    session.add(folder)
    session.flush()
    media = MediaItem(
        source_path=f"incoming/{uuid.uuid4()}.wav",
        media_folder_id=folder.id,
        created_at=NOW,
    )
    session.add(media)
    session.flush()
    session.add(
        MediaSourceMetadata(
            media_item_id=media.id,
            source_kind="ytdlp",
            title="Temporal recording",
            upload_date=date(2026, 8, 1),
            tags=[],
            raw_schema_version=1,
            acquired_at=NOW,
        )
    )
    run = PipelineRun(
        media_item_id=media.id,
        status=RunStatus.COMPLETED.value,
        created_at=NOW,
        updated_at=NOW,
    )
    session.add(run)
    session.flush()
    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0,
            end_seconds=2,
            raw_text=text_value,
        )
    )
    session.commit()
    return project.id, run.id


def _artifacts(session: Session, project_id: uuid.UUID) -> list[CorpusAnalysisArtifact]:
    return list(
        session.execute(
            select(CorpusAnalysisArtifact).where(
                CorpusAnalysisArtifact.scope_kind == "project",
                CorpusAnalysisArtifact.scope_id == project_id,
                CorpusAnalysisArtifact.artifact_kind
                == CorpusAnalysisArtifactKind.TEMPORAL_TRENDS.value,
            )
        ).scalars()
    )


def _record_entity_asset(
    session: Session,
    run_id: uuid.UUID,
    *,
    generation: int,
    surface: str,
    previous: RunEnrichmentAsset | None = None,
) -> RunEnrichmentAsset:
    completed_at = NOW + timedelta(minutes=generation)
    asset = RunEnrichmentAsset(
        pipeline_run_id=run_id,
        asset_kind=RunAssetKind.ENTITY_MENTIONS.value,
        generation=generation,
        payload={"mentions": [{"surface": surface, "kind": "organization", "occurrences": [{}]}]},
        payload_schema_version=1,
        producer="test",
        producer_version="1",
        model="test-model",
        source_content_hash="a" * 64,
        idempotency_key=f"temporal:{run_id}:{generation}",
        started_at=completed_at - timedelta(seconds=1),
        completed_at=completed_at,
    )
    session.add(asset)
    session.flush()
    if previous is not None:
        previous.superseded_by_asset_id = asset.id
        session.flush()
    session.commit()
    return asset


def test_cache_hit_returns_stored_payload(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project_id, _ = _seed_project(session)
        first = compute_temporal_trends(session, project_id)
        session.commit()
        artifact = _artifacts(session, project_id)[0]
        artifact_id = artifact.id

        second = compute_temporal_trends(session, project_id)
        session.commit()

        assert second == first
        assert [row.id for row in _artifacts(session, project_id)] == [artifact_id]


def test_correction_invalidates_cache(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project_id, run_id = _seed_project(session)
        before = compute_temporal_trends(session, project_id)
        session.commit()
        old_artifact_id = _artifacts(session, project_id)[0].id
        segment = session.execute(
            select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
        ).scalar_one()
        set_correction(session, segment=segment, text="beta Acme")
        session.commit()

        after = compute_temporal_trends(session, project_id)
        session.commit()

        assert [term["key"] for term in before["terms"]] == ["acme", "alpha"]
        assert [term["key"] for term in after["terms"]] == ["acme", "beta"]
        assert _artifacts(session, project_id)[0].id != old_artifact_id
        assert (
            session.execute(select(func.count()).select_from(SegmentReviewState)).scalar_one() == 1
        )


def test_entity_regeneration_invalidates_cache(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project_id, run_id = _seed_project(session)
        first_asset = _record_entity_asset(session, run_id, generation=1, surface="Acme")
        before = compute_temporal_trends(session, project_id)
        session.commit()
        old_artifact_id = _artifacts(session, project_id)[0].id
        _record_entity_asset(
            session,
            run_id,
            generation=2,
            surface="Globex",
            previous=first_asset,
        )

        after = compute_temporal_trends(session, project_id)
        session.commit()

        assert [entity["key"] for entity in before["entities"]] == ["acme"]
        assert [entity["key"] for entity in after["entities"]] == ["globex"]
        assert _artifacts(session, project_id)[0].id != old_artifact_id


def test_empty_project_returns_and_caches_empty_payload(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project = Project(name=f"Empty {uuid.uuid4()}")
        session.add(project)
        session.commit()

        payload = compute_temporal_trends(session, project.id)
        session.commit()

        assert payload["buckets"] == []
        assert payload["terms"] == []
        assert payload["entities"] == []
        assert payload["range"]["bucket_unit"] is None
        assert len(_artifacts(session, project.id)) == 1


def _advisory_waiters(session_factory: sessionmaker[Session], key: int) -> int:
    with session_factory() as session:
        return int(
            session.execute(
                text(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND NOT granted "
                    "AND objid = :key"
                ),
                {"key": key},
            ).scalar_one()
        )


def test_concurrent_reads_produce_one_artifact(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as setup:
        project_id, _ = _seed_project(setup)

    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def _compute_b() -> None:
        try:
            with session_factory() as session_b:
                result["b"] = compute_temporal_trends(session_b, project_id)
                session_b.commit()
        except BaseException as exc:
            errors.append(exc)

    with session_factory() as session_a:
        result["a"] = compute_temporal_trends(session_a, project_id)
        thread = threading.Thread(target=_compute_b)
        thread.start()
        key = _temporal_lock_key(project_id)
        for _ in range(500):
            if _advisory_waiters(session_factory, key) >= 1:
                break
            time.sleep(0.01)
        else:
            thread.join(timeout=5)
            pytest.fail("concurrent temporal read never blocked on the advisory lock")
        session_a.commit()

    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    assert result["a"] == result["b"]
    with session_factory() as verify:
        assert len(_artifacts(verify, project_id)) == 1
