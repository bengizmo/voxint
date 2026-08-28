"""Run-selection read model for the media editor (issue #156).

Tests the pure query layer in editor_query.py against a real Postgres database.
Skipped without VOXINT_TEST_DATABASE_URL.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.api.editor_query import media_detail
from voxint.db.models import MediaItem, PipelineRun, RunStatus


def _media(session: Session, *, path: str = "/audio/test.wav") -> MediaItem:
    m = MediaItem(source_path=path, current_path=path)
    session.add(m)
    session.flush()
    return m


def _run(
    session: Session,
    media: MediaItem,
    *,
    status: str = RunStatus.COMPLETED.value,
    created_offset_hours: int = 0,
) -> PipelineRun:
    r = PipelineRun(
        media_item_id=media.id,
        status=status,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=created_offset_hours),
    )
    session.add(r)
    session.flush()
    return r


def test_nonexistent_media_returns_none(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        result = media_detail(session, uuid.uuid4())
    assert result is None


def test_media_with_no_runs(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    assert result.media.id == m.id
    assert result.selected_run is None
    assert result.chooser == []


def test_selects_latest_completed_run(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        _run(session, m, created_offset_hours=0)
        newer = _run(session, m, created_offset_hours=1)
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    assert result.selected_run is not None
    assert result.selected_run.id == newer.id


def test_skips_non_completed_for_default(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        completed = _run(session, m, created_offset_hours=0)
        _run(session, m, status=RunStatus.QUEUED.value, created_offset_hours=1)
        _run(session, m, status=RunStatus.FAILED.value, created_offset_hours=2)
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    assert result.selected_run is not None
    assert result.selected_run.id == completed.id


def test_run_override_selects_exact_run(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        older = _run(session, m, created_offset_hours=0)
        _run(session, m, created_offset_hours=1)
        session.commit()
        result = media_detail(session, m.id, run_override=older.id)
    assert result is not None
    assert result.selected_run is not None
    assert result.selected_run.id == older.id


def test_run_override_allows_non_completed(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        queued = _run(session, m, status=RunStatus.QUEUED.value)
        session.commit()
        result = media_detail(session, m.id, run_override=queued.id)
    assert result is not None
    assert result.selected_run is not None
    assert result.selected_run.id == queued.id
    assert result.selected_run.status == RunStatus.QUEUED.value


def test_run_override_foreign_media_returns_none(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        m1 = _media(session, path="/audio/a.wav")
        m2 = _media(session, path="/audio/b.wav")
        foreign_run = _run(session, m2)
        session.commit()
        result = media_detail(session, m1.id, run_override=foreign_run.id)
    assert result is None


def test_run_override_unknown_returns_none(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        m = _media(session)
        _run(session, m)
        session.commit()
        result = media_detail(session, m.id, run_override=uuid.uuid4())
    assert result is None


def test_chooser_lists_all_runs_newest_first(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        m = _media(session)
        r1 = _run(session, m, created_offset_hours=0)
        r2 = _run(session, m, status=RunStatus.FAILED.value, created_offset_hours=1)
        r3 = _run(session, m, created_offset_hours=2)
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    ids = [c.id for c in result.chooser]
    assert ids == [r3.id, r2.id, r1.id]
    assert result.chooser[1].status == RunStatus.FAILED.value


def test_chooser_marks_archived(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        r = _run(session, m)
        r.archived_at = datetime.now(tz=UTC)
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    assert result.chooser[0].archived is True


def test_media_identity_fields(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        m = _media(session)
        m.duration_seconds = 120.5
        m.size_bytes = 1024000
        session.commit()
        result = media_detail(session, m.id)
    assert result is not None
    assert result.media.source_path == "/audio/test.wav"
    assert result.media.duration_seconds == 120.5
    assert result.media.size_bytes == 1024000
