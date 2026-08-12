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


def test_fetch_creates_source_url_run_and_enqueues(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = "https://www.youtube.com/watch?v=abc"
    assert main(["fetch", url]) == 0
    printed = capsys.readouterr().out.strip()
    assert enqueued == [printed]  # commit-before-publish: enqueued after commit
    with session_factory() as session:
        run = session.get(PipelineRun, uuid.UUID(printed))
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_url == url
        assert media.source_path.endswith("/source")  # pre-assigned, no file yet


def test_fetch_bad_url_errors_without_enqueue(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A rejected URL exits 2 with a URL-free message and writes no rows; the
    # rollback in session_scope leaves nothing behind, and nothing is enqueued.
    assert main(["fetch", "ftp://example.com/f.mp3"]) == 2
    assert "error:" in capsys.readouterr().out
    assert enqueued == []
    with session_factory() as session:
        assert session.execute(select(PipelineRun)).first() is None
        assert session.execute(select(MediaItem)).first() is None


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
    assert capsys.readouterr().out.strip() == (
        "error: run is queued, only failed runs can be requeued"
    )
    assert enqueued == []

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.FAILED.value
        run.current_stage = Stage.TRANSCRIBE.value
        session.commit()

    assert main(["requeue", str(run_id)]) == 0
    assert capsys.readouterr().out.strip() == f"requeued {run_id}"
    assert enqueued == [str(run_id)]
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED.value
        assert run.current_stage == Stage.TRANSCRIBE.value


def test_requeue_unknown_run_errors_without_enqueue(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = uuid.uuid4()
    assert main(["requeue", str(missing)]) == 2
    assert capsys.readouterr().out.strip() == f"error: no run {missing}"
    assert enqueued == []


def test_requeue_failed_without_stage_refuses_to_guess(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with session_factory() as session:
        media = MediaItem(source_path="incoming/corrupt.wav")
        session.add(media)
        session.flush()
        run_id = submit(session, media.id).id
        session.commit()
    with session_factory() as session:  # fabricate the impossible FAILED-with-no-stage state
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.FAILED.value
        run.current_stage = None
        session.commit()

    assert main(["requeue", str(run_id)]) == 2
    assert capsys.readouterr().out.strip() == (
        f"error: run {run_id} is FAILED with no current_stage; refusing to guess"
    )
    assert enqueued == []


def test_submit_publishes_only_after_commit(
    session_factory: sessionmaker[Session],
    media_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the broker publish runs, the run must already be durably committed —
    an independent session opened from inside delay() sees it as QUEUED."""
    from voxint.worker import tasks

    seen: dict[str, str | None] = {}

    def fake_delay(run_id: str) -> None:
        with session_factory() as session:
            run = session.get(PipelineRun, uuid.UUID(run_id))
            seen["status"] = run.status if run is not None else None

    monkeypatch.setattr(tasks.run_pipeline, "delay", fake_delay)
    assert main(["submit", "incoming/a.wav"]) == 0
    assert seen["status"] == RunStatus.QUEUED.value


def test_submit_keeps_committed_run_when_publish_fails(
    session_factory: sessionmaker[Session],
    media_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker outage at publish time must not roll back the durable run: the
    QUEUED row survives so the recovery sweep can re-enqueue it."""
    from voxint.worker import tasks

    def boom(run_id: str) -> None:
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(tasks.run_pipeline, "delay", boom)
    with pytest.raises(RuntimeError):
        main(["submit", "incoming/a.wav"])

    with session_factory() as session:
        run = session.execute(select(PipelineRun)).scalar_one()
        assert run.status == RunStatus.QUEUED.value
