"""CLI commands against real Postgres; the broker is stubbed at the task seam."""

import io
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.cli import main
from voxint.db.models import (
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
    TranscriptSegment,
)
from voxint.pipeline.engine import submit


def _seed_completed_run(session: Session) -> uuid.UUID:
    """A completed run with two transcript segments and two diarization turns."""
    media = MediaItem(source_path="incoming/talk.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    for index, (label, text) in enumerate([("SPEAKER_00", "hello"), ("SPEAKER_01", "hi")]):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index),
                end_seconds=float(index) + 1.0,
                raw_text=text,
                diarization_label=label,
            )
        )
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index),
                end_seconds=float(index) + 1.0,
                label=label,
                skip_reason="too_short",
            )
        )
    session.commit()
    return run.id


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


def test_fetch_reads_url_from_stdin_when_omitted(
    session_factory: sessionmaker[Session],
    enqueued: list[str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Piping the URL (no positional arg) keeps a signed URL off argv and shell
    # history; the stdin path creates the same source_url run as the positional.
    url = "https://podcast.example.com/ep/42.mp3"
    monkeypatch.setattr("sys.stdin", io.StringIO(url + "\n"))
    assert main(["fetch"]) == 0
    printed = capsys.readouterr().out.strip()
    assert enqueued == [printed]
    with session_factory() as session:
        run = session.get(PipelineRun, uuid.UUID(printed))
        assert run is not None
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.source_url == url


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


def test_list_shows_runs_table_and_json(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as session:
        run_id = _seed_completed_run(session)

    assert main(["list"]) == 0
    table = capsys.readouterr().out
    assert str(run_id) in table
    assert "completed" in table
    assert "incoming/talk.wav" in table

    import json

    assert main(["list", "--json", "--status", "completed"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in rows] == [str(run_id)]
    assert rows[0]["source_path"] == "incoming/talk.wav"


def test_list_empty_reports_no_runs(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["list"]) == 0
    assert "(no runs)" in capsys.readouterr().out


def test_export_formats_and_file_output(
    session_factory: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as session:
        run_id = _seed_completed_run(session)

    assert main(["export", str(run_id), "--format", "srt"]) == 0
    srt = capsys.readouterr().out
    assert "1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00:\nhello\n" in srt

    # RTTM reads the diarization turns (raw labels, run-uuid file id).
    assert main(["export", str(run_id), "--format", "rttm"]) == 0
    rttm = capsys.readouterr().out
    assert rttm.splitlines()[0] == f"SPEAKER {run_id} 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>"

    # -o writes the file and reports the path (no stdout payload).
    out = tmp_path / "t.json"
    assert main(["export", str(run_id), "--format", "json", "-o", str(out)]) == 0
    assert f"wrote {out}" in capsys.readouterr().out
    import json

    assert [r["text"] for r in json.loads(out.read_text())] == ["hello", "hi"]


def test_export_unknown_run_errors(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    missing = uuid.uuid4()
    assert main(["export", str(missing), "--format", "json"]) == 2
    assert f"no run {missing}" in capsys.readouterr().out


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


def _seed_run_with_stages(session: Session) -> uuid.UUID:
    """A completed run with finished stage attempts, incl. one transcribe failure."""
    from datetime import UTC, datetime, timedelta

    media = MediaItem(source_path="incoming/stats.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()

    base = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)
    # A failed transcribe attempt (attempt 1) then a completed one (attempt 2):
    # proves failures and durations count attempts, not distinct runs.
    session.add(
        StageRun(
            pipeline_run_id=run.id,
            stage=Stage.TRANSCRIBE.value,
            status=StageStatus.FAILED.value,
            attempt=1,
            started_at=base,
            finished_at=base + timedelta(seconds=5),
            error="boom",
        )
    )
    session.add(
        StageRun(
            pipeline_run_id=run.id,
            stage=Stage.TRANSCRIBE.value,
            status=StageStatus.COMPLETED.value,
            attempt=2,
            started_at=base + timedelta(seconds=10),
            finished_at=base + timedelta(seconds=25),
        )
    )
    session.commit()
    return run.id


def test_stats_text_and_json(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    with session_factory() as session:
        _seed_run_with_stages(session)

    assert main(["stats"]) == 0
    text = capsys.readouterr().out
    assert "completed" in text
    assert "transcribe" in text  # both a failure line and a duration line

    import json

    assert main(["stats", "--json", "--since", "7d"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_counts"]["completed"] == 1
    assert payload["stage_failure_counts"]["transcribe"] == 1  # one failed attempt
    durations = {d["stage"]: d for d in payload["stage_durations"]}
    # two finished transcribe attempts (5s + 15s) → avg 10s
    assert durations["transcribe"]["attempt_count"] == 2
    assert durations["transcribe"]["avg_seconds"] == 10.0


def test_watch_sees_committed_status_flip(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    import threading
    import time

    media = MediaItem(source_path="incoming/watch.wav")
    with session_factory() as session:
        session.add(media)
        session.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.RUNNING.value,
            current_stage=Stage.TRANSCRIBE.value,
        )
        session.add(run)
        session.commit()
        run_id = run.id

    errors: list[BaseException] = []

    def flip() -> None:
        # Capture any failure so it surfaces as a clear assertion below instead
        # of an opaque 124 timeout in the main thread.
        try:
            time.sleep(0.4)
            with session_factory() as s:
                row = s.get(PipelineRun, run_id)
                assert row is not None
                row.status = RunStatus.COMPLETED.value
                row.current_stage = None
                s.commit()
        except BaseException as exc:  # re-raised in the main thread
            errors.append(exc)

    worker = threading.Thread(target=flip)
    worker.start()
    try:
        # A fresh session per poll must observe the other transaction's commit.
        code = main(["watch", str(run_id), "--interval", "0.1", "--timeout", "10"])
    finally:
        worker.join()
    if errors:
        raise errors[0]
    assert code == 0
    assert "completed" in capsys.readouterr().err


def test_watch_missing_run_exits_two(
    session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    del session_factory
    code = main(["watch", str(uuid.uuid4()), "--interval", "0.1", "--timeout", "1"])
    assert code == 2
    assert "no run" in capsys.readouterr().err


def test_collect_stats_query_semantics(session_factory: sessionmaker[Session]) -> None:
    """Direct query checks the pure renderers can't catch: tz-aware cutoff
    inclusivity, negative-duration exclusion, and empty-stage behavior."""
    from datetime import UTC, datetime, timedelta

    from voxint.api.stats_query import collect_stats

    now = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)
    cutoff = now - timedelta(hours=24)  # since = 2026-08-13T12:00:00Z

    with session_factory() as session:
        # Three runs straddling the cutoff: strictly before, exactly on, after.
        for created, tag in [
            (cutoff - timedelta(seconds=1), "before"),
            (cutoff, "on"),
            (cutoff + timedelta(seconds=1), "after"),
        ]:
            media = MediaItem(source_path=f"incoming/{tag}.wav")
            session.add(media)
            session.flush()
            run = PipelineRun(
                media_item_id=media.id, status=RunStatus.COMPLETED.value, created_at=created
            )
            session.add(run)
            session.flush()
            # A negative-duration attempt (clock skew) must be excluded from the
            # average; a normal 4s attempt must be included.
            base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=UTC)
            session.add(
                StageRun(
                    pipeline_run_id=run.id,
                    stage=Stage.PREPARE.value,
                    status=StageStatus.COMPLETED.value,
                    attempt=1,
                    started_at=base,
                    finished_at=base + timedelta(seconds=4),
                )
            )
            session.add(
                StageRun(
                    pipeline_run_id=run.id,
                    stage=Stage.PREPARE.value,
                    status=StageStatus.FAILED.value,
                    attempt=2,
                    started_at=base + timedelta(seconds=10),
                    finished_at=base,  # finished_at < started_at → excluded
                )
            )
        session.commit()

        stats = collect_stats(session, since=cutoff, now=now)

    # created_at >= since is inclusive: the "on" and "after" runs count, "before" does not.
    assert stats.runs_created_count == 2
    # Negative-duration attempts excluded → only the 4s attempts averaged.
    prepare = {d.stage: d for d in stats.stage_durations}["prepare"]
    assert prepare.attempt_count == 3  # one valid attempt per run, skew rows dropped
    assert prepare.avg_seconds == 4.0
    # Stages with no finished attempts don't appear in the list at all.
    assert "finalize" not in {d.stage for d in stats.stage_durations}


def test_collect_stats_empty_database(session_factory: sessionmaker[Session]) -> None:
    from datetime import UTC, datetime, timedelta

    from voxint.api.stats_query import collect_stats

    now = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)
    with session_factory() as session:
        stats = collect_stats(session, since=now - timedelta(hours=24), now=now)
    assert stats.status_counts == {}
    assert stats.stage_durations == ()
    assert stats.stage_failure_counts == ()
    assert stats.roster_size == 0
    assert stats.runs_created_count == 0
