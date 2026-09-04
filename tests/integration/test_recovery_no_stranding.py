"""Cross-cutting no-stranding test: cap-deferred QUEUED runs are picked up
by the recovery sweep once their ``updated_at`` ages past the stale grace.

Drives ``recovery_sweep()`` against real Postgres with monkeypatched broker
dispatch, verifying that runs left behind by a batch cap (scan-confirm,
rerun-confirm, or watch-folder) drain through recovery rather than stranding.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus


def _settings(**over: object) -> Settings:
    over.setdefault("queued_run_stale_seconds", 3600)
    over.setdefault("recovery_publish_batch_size", 50)
    return Settings(_env_file=None, **over)  # type: ignore[call-arg]


def _seed_queued_run(
    factory: sessionmaker[Session],
    *,
    age_seconds: int = 0,
) -> uuid.UUID:
    """Create a QUEUED PipelineRun with ``updated_at`` backdated by *age_seconds*."""
    with factory() as s:
        media = MediaItem(source_path=f"test/{uuid.uuid4()}.wav")
        s.add(media)
        s.flush()
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.QUEUED.value,
        )
        s.add(run)
        s.flush()
        if age_seconds:
            backdated = datetime.now(tz=UTC) - timedelta(seconds=age_seconds)
            run.updated_at = backdated
        s.commit()
        return run.id


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


class _DispatchRecorder:
    """Captures apply_async calls; returns a MagicMock per task name."""

    def __init__(self) -> None:
        self.dispatched: list[uuid.UUID] = []
        self._tasks: dict[str, MagicMock] = {}

    def task_for(self, name: str) -> MagicMock:
        if name not in self._tasks:
            task = MagicMock()

            def _record(args: tuple[str, ...], **kw: object) -> None:
                self.dispatched.append(uuid.UUID(args[0]))

            task.apply_async = MagicMock(side_effect=_record)
            self._tasks[name] = task
        return self._tasks[name]


def test_stale_capped_runs_are_recovered(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs left behind by a batch cap become eligible after the stale grace
    and are dispatched by the recovery sweep."""
    import voxint.worker.tasks as tasks_mod

    recorder = _DispatchRecorder()
    settings = _settings(queued_run_stale_seconds=3600, recovery_publish_batch_size=50)

    # 3 stale runs (older than the 1h grace) -- these should be dispatched
    stale_ids = [_seed_queued_run(session_factory, age_seconds=7200) for _ in range(3)]

    # 2 fresh runs (just submitted, within the grace) -- these should NOT be dispatched
    fresh_ids = [_seed_queued_run(session_factory, age_seconds=0) for _ in range(2)]

    monkeypatch.setattr(tasks_mod, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        tasks_mod, "pipeline_task_for_stage", lambda stage: recorder.task_for("pipeline")
    )
    # Stub out functions recovery_sweep calls that we don't need for this test
    monkeypatch.setattr(
        tasks_mod, "recover_interrupted_runs", lambda session, **kw: []
    )
    monkeypatch.setattr(
        tasks_mod, "close_cancelled_run_claims", lambda session: []
    )
    monkeypatch.setattr(tasks_mod, "close_paused_run_claims", lambda session: None)
    # Stub out the embedding/asset/translation/plugin job recovery
    monkeypatch.setattr(tasks_mod.embedding_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod.asset_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod.translation_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod, "get_plugins", lambda: MagicMock(job_lanes=lambda: []))

    result = tasks_mod.recovery_sweep()

    assert set(recorder.dispatched) == set(stale_ids)
    for fresh_id in fresh_ids:
        assert fresh_id not in recorder.dispatched
    assert result["stale_queued"] == 3
    assert result["dispatched"] == 3


def test_recovery_batch_cap_drains_gradually(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When more stale runs exist than recovery_publish_batch_size, the sweep
    dispatches exactly the cap and leaves the rest for the next sweep."""
    import voxint.worker.tasks as tasks_mod

    recorder = _DispatchRecorder()
    settings = _settings(queued_run_stale_seconds=3600, recovery_publish_batch_size=3)

    stale_ids = [_seed_queued_run(session_factory, age_seconds=7200) for _ in range(5)]

    monkeypatch.setattr(tasks_mod, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        tasks_mod, "pipeline_task_for_stage", lambda stage: recorder.task_for("pipeline")
    )
    monkeypatch.setattr(
        tasks_mod, "recover_interrupted_runs", lambda session, **kw: []
    )
    monkeypatch.setattr(
        tasks_mod, "close_cancelled_run_claims", lambda session: []
    )
    monkeypatch.setattr(tasks_mod, "close_paused_run_claims", lambda session: None)
    monkeypatch.setattr(tasks_mod.embedding_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod.asset_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod.translation_jobs, "stale_queued_job_ids", lambda *a, **kw: [])
    monkeypatch.setattr(tasks_mod, "get_plugins", lambda: MagicMock(job_lanes=lambda: []))

    result = tasks_mod.recovery_sweep()

    assert result["dispatched"] == 3
    assert len(recorder.dispatched) == 3
    # The dispatched set is a subset of all stale ids
    assert set(recorder.dispatched).issubset(set(stale_ids))

    # Second sweep picks up the remaining 2
    recorder2 = _DispatchRecorder()
    monkeypatch.setattr(
        tasks_mod, "pipeline_task_for_stage", lambda stage: recorder2.task_for("pipeline")
    )

    result2 = tasks_mod.recovery_sweep()

    assert result2["dispatched"] == 2
    assert len(recorder2.dispatched) == 2
    # Together, both sweeps dispatched all 5
    assert set(recorder.dispatched) | set(recorder2.dispatched) == set(stale_ids)
