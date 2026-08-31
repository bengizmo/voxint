"""Annotation tag corpus rollup on the Explore surface (#331 Phase 7).

``tag_stats`` is deliberately synchronous SQL (no ``corpus_analysis_artifacts``
cache): one indexed join at a max of 8 tags per annotation. These tests pin the
exclusion invariants (archived tags and soft-deleted annotations never count),
the project scoping join, the count-desc/name-asc ordering, and the endpoint
wiring into ``explore_props``.
"""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.explore_query import tag_stats
from voxint.config import Settings
from voxint.db.models import (
    AnnotationTag,
    AnnotationTagLink,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunStatus,
    TranscriptAnnotation,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")


def _seed_run(
    session: Session,
    *,
    project_name: str | None = None,
    status: RunStatus = RunStatus.COMPLETED,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID | None]:
    """One media item + run (+ optional project/folder). Returns (run, seg, project)."""
    project_id: uuid.UUID | None = None
    folder_id: uuid.UUID | None = None
    if project_name is not None:
        project = Project(name=project_name)
        session.add(project)
        session.flush()
        project_id = project.id
        folder = MediaFolder(path=f"folder-{uuid.uuid4().hex[:8]}", project_id=project.id)
        session.add(folder)
        session.flush()
        folder_id = folder.id
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav", media_folder_id=folder_id)
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=status.value)
    session.add(run)
    session.flush()
    seg = TranscriptSegment(
        pipeline_run_id=run.id,
        segment_index=0,
        start_seconds=0.0,
        end_seconds=3.0,
        raw_text="Hello world there",
        diarization_label="S0",
    )
    session.add(seg)
    session.flush()
    return run.id, seg.id, project_id


def _add_tag(session: Session, name: str, *, color: int = 0, archived: bool = False) -> uuid.UUID:
    from datetime import UTC, datetime

    tag = AnnotationTag(
        name=name,
        name_normalized=name.strip().casefold(),
        color=color,
        archived_at=datetime.now(UTC) if archived else None,
    )
    session.add(tag)
    session.flush()
    return tag.id


def _add_annotation(
    session: Session,
    run_id: uuid.UUID,
    seg_id: uuid.UUID,
    tag_ids: list[uuid.UUID],
    *,
    deleted: bool = False,
) -> uuid.UUID:
    from datetime import UTC, datetime

    row = TranscriptAnnotation(
        pipeline_run_id=run_id,
        anchor_schema_version=1,
        anchor_kind="segment_range",
        start_segment_id=seg_id,
        end_segment_id=seg_id,
        start_segment_index=0,
        end_segment_index=0,
        source_text_hash="0" * 64,
        quote_text="Hello world there",
        color_index=0,
        operator="op",
        deleted_at=datetime.now(UTC) if deleted else None,
    )
    session.add(row)
    session.flush()
    for tag_id in tag_ids:
        session.add(AnnotationTagLink(annotation_id=row.id, tag_id=tag_id))
    session.flush()
    return row.id


def test_tag_stats_counts_and_ordering(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id, seg_id, _ = _seed_run(session)
        alpha = _add_tag(session, "Alpha")
        beta = _add_tag(session, "Beta")
        zeta = _add_tag(session, "Zeta")
        _add_annotation(session, run_id, seg_id, [alpha, beta])
        _add_annotation(session, run_id, seg_id, [beta])
        _add_annotation(session, run_id, seg_id, [zeta])
        session.commit()

        stats = tag_stats(session)
        assert [(s.name, s.count) for s in stats] == [("Beta", 2), ("Alpha", 1), ("Zeta", 1)]
        assert stats[0].tag_id == str(beta)
        assert all(isinstance(s.color, int) for s in stats)


def test_tag_stats_excludes_archived_tags_and_deleted_annotations(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id, seg_id, _ = _seed_run(session)
        live = _add_tag(session, "Live")
        archived = _add_tag(session, "Archived", archived=True)
        _add_annotation(session, run_id, seg_id, [live, archived])
        _add_annotation(session, run_id, seg_id, [live], deleted=True)
        session.commit()

        stats = tag_stats(session)
        assert [(s.name, s.count) for s in stats] == [("Live", 1)]


def test_tag_stats_untagged_and_unused(session_factory: sessionmaker[Session]) -> None:
    """A tag with no live annotations does not appear; empty corpus is []."""
    with session_factory() as session:
        assert tag_stats(session) == []
        run_id, seg_id, _ = _seed_run(session)
        _add_tag(session, "Unused")
        _add_annotation(session, run_id, seg_id, [])
        session.commit()
        assert tag_stats(session) == []


def test_tag_stats_counts_non_completed_runs(session_factory: sessionmaker[Session]) -> None:
    """A highlight is evidence the moment it exists — run status is irrelevant."""
    with session_factory() as session:
        run_id, seg_id, _ = _seed_run(session, status=RunStatus.AWAITING_ADJUDICATION)
        tag = _add_tag(session, "Early")
        _add_annotation(session, run_id, seg_id, [tag])
        session.commit()
        assert [(s.name, s.count) for s in tag_stats(session)] == [("Early", 1)]


def test_tag_stats_project_scoping(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_a, seg_a, project_a = _seed_run(session, project_name="Project A")
        run_b, seg_b, _ = _seed_run(session, project_name="Project B")
        run_c, seg_c, _ = _seed_run(session)  # no project
        shared = _add_tag(session, "Shared")
        only_b = _add_tag(session, "OnlyB")
        _add_annotation(session, run_a, seg_a, [shared])
        _add_annotation(session, run_b, seg_b, [shared, only_b])
        _add_annotation(session, run_c, seg_c, [shared])
        session.commit()

        assert project_a is not None
        scoped = tag_stats(session, project_a)
        assert [(s.name, s.count) for s in scoped] == [("Shared", 1)]
        unscoped = tag_stats(session)
        assert [(s.name, s.count) for s in unscoped] == [("Shared", 3), ("OnlyB", 1)]


def test_explore_page_hydrates_tag_stats(
    session_factory: sessionmaker[Session], tmp_path: object
) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,  # type: ignore[arg-type]
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    with session_factory() as session:
        run_id, seg_id, _ = _seed_run(session)
        tag = _add_tag(session, "Key point", color=3)
        _add_annotation(session, run_id, seg_id, [tag])
        session.commit()
        tag_id = str(tag)

    response = client.get("/explore")
    assert response.status_code == 200
    assert "tagStats" in response.text
    assert "Key point" in response.text
    assert tag_id in response.text
