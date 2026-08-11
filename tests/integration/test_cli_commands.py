"""CLI commands against real Postgres; the broker is stubbed at the task seam."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.cli import main
from voxint.db.models import MediaItem, PipelineRun, RunStatus, Stage
from voxint.pipeline.engine import submit


@pytest.fixture()
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture run ids handed to the broker instead of touching Redis."""
    calls: list[str] = []
    from voxint.worker import tasks

    monkeypatch.setattr(tasks.run_pipeline, "delay", lambda rid: calls.append(rid))
    return calls


@pytest.fixture()
def media_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    (tmp_path / "incoming").mkdir()
    (tmp_path / "incoming" / "a.wav").write_bytes(b"RIFF")
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    return tmp_path


def test_submit_creates_run_and_enqueues(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    media_env: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["submit", "incoming/a.wav"]) == 0
    printed = capsys.readouterr().out.strip()
    assert enqueued == [printed]
    with session_factory() as session:
        run = session.get(PipelineRun, uuid.UUID(printed))
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_path == "incoming/a.wav"


def test_submit_reuses_existing_media_item(
    session_factory: sessionmaker[Session], enqueued: list[str], media_env: Path
) -> None:
    assert main(["submit", "incoming/a.wav"]) == 0
    assert main(["submit", "incoming/a.wav"]) == 0
    with session_factory() as session:
        assert len(session.execute(select(MediaItem)).scalars().all()) == 1
        assert len(session.execute(select(PipelineRun)).scalars().all()) == 2


def test_submit_rejects_escape_and_missing(
    enqueued: list[str], media_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["submit", "../outside.wav"]) == 2
    assert main(["submit", "incoming/absent.wav"]) == 2
    assert enqueued == []


def test_status_shows_run_and_ledger(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as session:
        media = MediaItem(source_path="incoming/s.wav")
        session.add(media)
        session.flush()
        run_id = submit(session, media.id).id
        session.commit()
    assert main(["status", str(run_id)]) == 0
    out = capsys.readouterr().out
    assert "status: queued" in out
    assert main(["status", str(uuid.uuid4())]) == 2


def test_requeue_only_failed_runs(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as session:
        media = MediaItem(source_path="incoming/r.wav")
        session.add(media)
        session.flush()
        run = submit(session, media.id)
        run_id = run.id
        session.commit()

    assert main(["requeue", str(run_id)]) == 2  # queued, not failed
    assert enqueued == []

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.FAILED.value
        run.current_stage = Stage.TRANSCRIBE.value
        session.commit()

    assert main(["requeue", str(run_id)]) == 0
    assert enqueued == [str(run_id)]
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        assert run.current_stage == Stage.TRANSCRIBE.value
