"""Real Postgres state, constraints, task gates and bounded lane dispatch."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tests.unit.test_gpu_phase import phase_settings
from voxint.app_settings import set_queue_paused
from voxint.db.models import (
    GPU_SEGMENT,
    POST_SEGMENT,
    MediaItem,
    PipelineRun,
    ResearchJob,
    RunAssetJob,
    Speaker,
    Stage,
    TranscriptSegment,
    TranslationJob,
)
from voxint.gpu_phase.dispatch import open_lanes, redispatch_queued_runs
from voxint.gpu_phase.state import (
    GpuPhase,
    OperatorRequest,
    gpu_lane_demand,
    gpu_lane_in_flight,
    post_lane_in_flight,
    read_phase,
    read_phase_if_enabled,
    set_phase,
    set_request,
)
from voxint.worker import tasks

NOW = datetime.now(UTC)


def seed_run(
    session: Session, stage: Stage | None, *, status: str = "queued", age: int = 7200
) -> PipelineRun:
    media = MediaItem(source_path=f"phase/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        current_stage=stage,
        status=status,
        updated_at=NOW - timedelta(seconds=age),
    )
    session.add(run)
    session.flush()
    return run


def test_state_round_trip_and_counts(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        session.execute(text("DELETE FROM gpu_phase"))
        assert read_phase(session) is None
        set_phase(
            session,
            GpuPhase.AUDIO,
            now=NOW,
            lease_id="lease",
            failures=2,
            lease_expires_at=NOW + timedelta(hours=1),
            last_error="bounded error",
            retry_after=NOW + timedelta(minutes=1),
        )
        session.commit()
        snapshot = read_phase(session)
        assert snapshot is not None and snapshot.phase == GpuPhase.AUDIO
        assert snapshot.phase_since == snapshot.updated_at == NOW
        assert snapshot.lease_id == "lease" and snapshot.failures == 2
        assert snapshot.lease_expires_at == NOW + timedelta(hours=1)
        assert snapshot.last_error == "bounded error"
        assert snapshot.retry_after == NOW + timedelta(minutes=1)
        set_phase(session, GpuPhase.AUDIO, now=NOW + timedelta(seconds=1), last_error=None)
        set_request(session, OperatorRequest.RELEASE)
        session.commit()
        snapshot = read_phase(session)
        assert snapshot is not None and snapshot.phase_since == NOW
        assert snapshot.operator_request == OperatorRequest.RELEASE
        assert snapshot.last_error is None
        set_request(session, None)
        set_phase(session, GpuPhase.LLM, now=NOW + timedelta(seconds=2))
        session.commit()
        snapshot = read_phase(session)
        assert snapshot is not None and snapshot.operator_request is None
        assert snapshot.phase_since == NOW + timedelta(seconds=2)
        for stage in [None, *Stage]:
            seed_run(session, stage)
            seed_run(session, stage, status="running")
            seed_run(session, stage, status="completed")
        assert gpu_lane_demand(session) == 5
        assert gpu_lane_in_flight(session) == 4
        assert post_lane_in_flight(session) == 2


@pytest.mark.parametrize(
    "assignment", ["phase = 'invalid'", "id = 2", "failures = -1", "operator_request = 'invalid'"]
)
def test_constraints(session_factory: sessionmaker[Session], assignment: str) -> None:
    with session_factory() as session:
        set_phase(session, GpuPhase.LLM, now=NOW)
        session.commit()
        with pytest.raises(IntegrityError):
            session.execute(text(f"UPDATE gpu_phase SET {assignment}"))
        session.rollback()


@pytest.mark.parametrize("segment", [GPU_SEGMENT, POST_SEGMENT])
@pytest.mark.parametrize("skip_pause", [False, True])
def test_drive_closed_leaves_run_untouched(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    segment: frozenset[Stage],
    skip_pause: bool,
) -> None:
    with session_factory() as session:
        run = seed_run(session, next(iter(segment)))
        set_phase(session, GpuPhase.DRAINING, now=NOW)
        if skip_pause:
            set_queue_paused(session, True, llm_enabled_default=False)
        session.commit()
        original = (run.status, run.revision, run.current_stage, run.updated_at)
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", phase_settings)
    assert (
        tasks._drive_segment(object(), str(run.id), segment, skip_queue_pause=skip_pause)
        == "gpu-phase-wait"
    )
    with session_factory() as session:
        after = session.get(PipelineRun, run.id)
        assert after is not None
        assert (after.status, after.revision, after.current_stage, after.updated_at) == original


def test_disabled_has_no_phase_queries(
    session_factory: sessionmaker[Session],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = phase_settings(gpu_phase_enabled=False)
    with session_factory() as session:
        run = seed_run(session, Stage.ACQUIRE)
        set_phase(session, GpuPhase.ERROR, now=NOW)
        session.commit()
    statements: list[str] = []

    def record(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        statements.append(statement)

    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    monkeypatch.setattr(tasks, "apply_run_preferences", lambda *a, **kw: SimpleNamespace(llm=None))
    monkeypatch.setattr(tasks, "build_stage_fns", lambda ctx: {})
    execute = MagicMock(
        return_value=SimpleNamespace(status=tasks.RunStatus.QUEUED, current_stage=Stage.ACQUIRE)
    )
    monkeypatch.setattr(tasks, "execute_run", execute)
    event.listen(engine, "before_cursor_execute", record)
    try:
        with session_factory() as session:
            assert read_phase_if_enabled(session, settings) is None
            assert open_lanes(session, settings) is None
        assert tasks._drive_segment(object(), str(run.id), GPU_SEGMENT) == "queued"
        for task, module, name in [
            (tasks.generate_run_asset, tasks.asset_jobs, "execute_job"),
            (tasks.translate_run, tasks.translation_jobs, "execute_job"),
            (tasks.research_speaker, tasks, "execute_job"),
        ]:
            execute_job = MagicMock()
            monkeypatch.setattr(module, name, execute_job)
            task(str(uuid.uuid4()))
            execute_job.assert_called_once()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    execute.assert_called_once()
    assert not any("gpu_phase" in sql.lower() for sql in statements)


def seed_jobs(session: Session) -> tuple[RunAssetJob, TranslationJob, ResearchJob]:
    run = seed_run(session, Stage.FINALIZE, status="completed")
    speaker = Speaker(display_name="Phase test")
    session.add(speaker)
    session.flush()
    jobs = (
        RunAssetJob(pipeline_run_id=run.id, asset_kind="summary", config={}),
        TranslationJob(
            pipeline_run_id=run.id, target_language="fr", config={}, source_content_hash="a" * 64
        ),
        ResearchJob(speaker_id=speaker.id, budget={}),
    )
    for job in jobs:
        job.created_at = NOW - timedelta(hours=2)
        session.add(job)
    session.flush()
    return jobs


@pytest.mark.parametrize("phase", [GpuPhase.AUDIO, GpuPhase.DRAINING_POST, GpuPhase.ERROR])
def test_post_jobs_stay_queued(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    phase: GpuPhase,
) -> None:
    with session_factory() as session:
        jobs = seed_jobs(session)
        set_phase(session, phase, now=NOW)
        session.commit()
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", phase_settings)
    for job, task in zip(
        jobs, [tasks.generate_run_asset, tasks.translate_run, tasks.research_speaker], strict=True
    ):
        task(str(job.id))
    with session_factory() as session:
        for job in jobs:
            after = session.get(type(job), job.id)
            assert after is not None and after.status == "queued"
            assert after.started_at is None


@pytest.mark.parametrize("phase", [None, GpuPhase.AUDIO, GpuPhase.LLM, GpuPhase.ERROR])
def test_dispatch_filter_before_limit(
    session_factory: sessionmaker[Session],
    phase: GpuPhase | None,
) -> None:
    settings = phase_settings(gpu_phase_enabled=phase is not None)
    with session_factory() as session:
        if phase is not None:
            set_phase(session, phase, now=NOW)
        gpu = seed_run(session, None, age=10000)
        post = seed_run(session, Stage.ENHANCE_MATCH, age=9000)
        session.commit()
        sent: list[uuid.UUID] = []

        def publish(rid: uuid.UUID, stage: Stage | None) -> bool:
            sent.append(rid)
            return True

        result = redispatch_queued_runs(
            session, lanes=open_lanes(session, settings), limit=1, publish=publish
        )
        expected = [] if phase == GpuPhase.ERROR else [post.id if phase == GpuPhase.LLM else gpu.id]
        assert sent == expected
        assert result.selected == result.dispatched == len(expected)


@pytest.mark.parametrize("phase", [GpuPhase.AUDIO, GpuPhase.LLM, GpuPhase.ERROR])
@pytest.mark.parametrize("paused", [False, True])
def test_recovery_lanes_and_research(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    phase: GpuPhase,
    paused: bool,
) -> None:
    settings = phase_settings(recovery_publish_batch_size=1)
    with session_factory() as session:
        set_phase(session, phase, now=NOW)
        gpu = seed_run(session, None, age=10000)
        post = seed_run(session, Stage.ENHANCE_MATCH, age=9000)
        jobs = seed_jobs(session)
        set_queue_paused(session, paused, llm_enabled_default=False)
        session.commit()
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    # Both recovered IDs also enter the resumable re-read, exercising its pre-LIMIT filter.
    monkeypatch.setattr(tasks, "recover_interrupted_runs", lambda *a, **kw: [gpu.id, post.id])
    pipeline = MagicMock()
    monkeypatch.setattr(tasks, "pipeline_task_for_stage", lambda stage: pipeline)
    publishers = [MagicMock() for _ in jobs]
    for task, publisher in zip(
        [tasks.generate_run_asset, tasks.translate_run, tasks.research_speaker],
        publishers,
        strict=True,
    ):
        monkeypatch.setattr(task, "apply_async", publisher)
    result = tasks.recovery_sweep()
    expected = (
        []
        if paused or phase == GpuPhase.ERROR
        else [str(post.id if phase == GpuPhase.LLM else gpu.id)]
    )
    assert [call.args[0][0] for call in pipeline.apply_async.call_args_list] == expected
    assert result["dispatched"] == len(expected)
    assert result["stale_queued"] == len(expected)
    for job, publisher, key in zip(
        jobs,
        publishers,
        ["stale_asset_jobs", "stale_translation_jobs", "stale_research_jobs"],
        strict=True,
    ):
        assert result[key] == (1 if phase == GpuPhase.LLM else 0)
        if phase == GpuPhase.LLM:
            publisher.assert_called_once_with((str(job.id),), ignore_result=True)
        else:
            publisher.assert_not_called()
    with session_factory() as session:
        assert tasks.research_jobs.stale_queued_job_ids(session, cutoff=NOW, limit=1) == [
            jobs[2].id
        ]
        assert tasks.research_jobs.stale_queued_job_ids(session, cutoff=NOW) == [jobs[2].id]


def test_resume_callers_keep_publish_failure_policy(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    from celery.exceptions import OperationalError

    from voxint import cli
    from voxint.api.routers import legacy_runs

    settings = phase_settings(recovery_publish_batch_size=2)
    with session_factory() as session:
        set_phase(session, GpuPhase.LLM, now=NOW)
        seed_run(session, None, age=10000)
        first = seed_run(session, Stage.ENHANCE_MATCH, age=9000)
        second = seed_run(session, Stage.FINALIZE, age=8000)
        session.commit()
    monkeypatch.setattr("voxint.config.get_settings", lambda: settings)
    monkeypatch.setattr("voxint.db.session.build_engine", lambda: None)
    monkeypatch.setattr("voxint.db.session.build_session_factory", lambda _: session_factory)
    publish = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(cli, "_publish_or_defer", publish)
    assert cli._queue_resume(argparse.Namespace()) == 0
    assert [call.args[0] for call in publish.call_args_list] == [first.id, second.id]
    assert capsys.readouterr().out == (
        "queue resumed; 1 queued runs dispatched; "
        "remaining queued runs drain via the recovery sweep\n"
    )
    task = MagicMock()
    task.apply_async.side_effect = OperationalError("broker unavailable")
    monkeypatch.setattr(tasks, "pipeline_task_for_stage", lambda stage: task)
    monkeypatch.setattr(legacy_runs, "_require_csrf", lambda *a: None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    with session_factory() as session:
        response = legacy_runs.queue_resume_route(request, None, session)
    assert response.status_code == 303 and response.headers["location"] == "/runs"
    task.apply_async.assert_called_once_with((str(first.id),), ignore_result=True)


def test_research_recovery_broker_failure_keeps_queued(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.exceptions import OperationalError

    with session_factory() as session:
        jobs = seed_jobs(session)
        session.commit()
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", lambda: phase_settings(gpu_phase_enabled=False))
    for task in [tasks.generate_run_asset, tasks.translate_run]:
        monkeypatch.setattr(task, "apply_async", MagicMock())
    publisher = MagicMock(side_effect=OperationalError("broker unavailable"))
    monkeypatch.setattr(tasks.research_speaker, "apply_async", publisher)
    result = tasks.recovery_sweep()
    assert result["stale_research_jobs"] == 1
    publisher.assert_called_once_with((str(jobs[2].id),), ignore_result=True)
    with session_factory() as session:
        job = session.get(ResearchJob, jobs[2].id)
        assert job is not None and job.status == "queued" and job.started_at is None


def test_missing_row_closes_post_jobs_and_embeddings_ignore_phase(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", phase_settings)
    with session_factory() as session:
        jobs = seed_jobs(session)
        session.execute(text("DELETE FROM gpu_phase"))
        session.commit()
    for job, task in zip(
        jobs, [tasks.generate_run_asset, tasks.translate_run, tasks.research_speaker], strict=True
    ):
        task(str(job.id))
    with session_factory() as session:
        for job in jobs:
            after = session.get(type(job), job.id)
            assert after is not None and after.status == "queued"
    with session_factory() as session:
        set_phase(session, GpuPhase.ERROR, now=NOW)
        session.commit()
    execute = MagicMock()
    monkeypatch.setattr(tasks.embedding_jobs, "execute_job", execute)
    tasks.generate_segment_embeddings(str(uuid.uuid4()))
    execute.assert_called_once()


@pytest.mark.parametrize("command", ["research", "assets"])
def test_inline_llm_cli_refuses_while_post_lane_closed(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    import argparse

    from sqlalchemy import func, select

    from voxint import cli

    with session_factory() as session:
        set_phase(session, GpuPhase.AUDIO, now=NOW)
        run = seed_run(session, Stage.FINALIZE, status="completed")
        speaker = Speaker(display_name="Inline phase test")
        session.add(speaker)
        session.commit()
        run_id, speaker_id = run.id, speaker.id
    monkeypatch.setattr("voxint.config.get_settings", phase_settings)
    monkeypatch.setattr("voxint.db.session.build_engine", lambda *a, **kw: None)
    monkeypatch.setattr("voxint.db.session.build_session_factory", lambda _: session_factory)
    if command == "research":
        code = cli._research_speaker(argparse.Namespace(speaker_id=str(speaker_id), note=None))
        model: type[ResearchJob] | type[RunAssetJob] = ResearchJob
    else:
        code = cli._enrich_assets(argparse.Namespace(run_id=run_id, kind=["summary"]))
        model = RunAssetJob
    assert code == 2
    assert "audio phase" in capsys.readouterr().out
    with session_factory() as session:
        assert session.execute(select(func.count()).select_from(model)).scalar_one() == 0


@pytest.mark.parametrize("segment", [GPU_SEGMENT, POST_SEGMENT])
def test_admission_rechecks_phase_after_early_gate(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    segment: frozenset[Stage],
) -> None:
    """A task passes the early gate, then the phase flips before the entry CAS."""
    open_phase = GpuPhase.AUDIO if segment == GPU_SEGMENT else GpuPhase.LLM
    closing_phase = GpuPhase.DRAINING if segment == GPU_SEGMENT else GpuPhase.DRAINING_POST
    stage = Stage.ACQUIRE if segment == GPU_SEGMENT else Stage.ENHANCE_MATCH
    with session_factory() as session:
        run = seed_run(session, stage)
        set_phase(session, open_phase, now=NOW)
        session.commit()
        original = (run.status, run.revision, run.current_stage, run.updated_at)

    def flip_phase(ctx: object) -> dict[Stage, object]:
        with session_factory() as session:
            set_phase(session, closing_phase, now=NOW + timedelta(seconds=1))
            session.commit()
        return {}

    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(tasks, "get_settings", phase_settings)
    monkeypatch.setattr(tasks, "apply_run_preferences", lambda *a, **kw: SimpleNamespace(llm=None))
    monkeypatch.setattr(tasks, "build_stage_fns", flip_phase)
    assert tasks._drive_segment(object(), str(run.id), segment) == "gpu-phase-wait"
    with session_factory() as session:
        after = session.get(PipelineRun, run.id)
        assert after is not None
        assert (after.status, after.revision, after.current_stage, after.updated_at) == original


def test_admission_lock_blocks_phase_change_until_commit(
    session_factory: sessionmaker[Session],
) -> None:
    """set_phase cannot slip between an admission read and its commit."""
    from sqlalchemy.exc import OperationalError

    from voxint.gpu_phase.state import admit_lane

    settings = phase_settings()
    with session_factory() as session:
        set_phase(session, GpuPhase.AUDIO, now=NOW)
        session.commit()
    with session_factory() as admitting:
        assert admit_lane(admitting, settings, GPU_SEGMENT) is True
        with session_factory() as orchestrator:
            orchestrator.execute(text("SET LOCAL lock_timeout = '200ms'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                set_phase(orchestrator, GpuPhase.DRAINING, now=NOW)
            orchestrator.rollback()
        admitting.commit()
    with session_factory() as orchestrator:
        set_phase(orchestrator, GpuPhase.DRAINING, now=NOW)
        orchestrator.commit()
        assert admit_lane(orchestrator, settings, GPU_SEGMENT) is False
        assert admit_lane(orchestrator, phase_settings(gpu_phase_enabled=False), GPU_SEGMENT)


@pytest.mark.parametrize("command", ["research", "assets"])
def test_inline_llm_cli_reports_a_phase_flip_after_the_precheck(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """The precheck passes, then the phase leaves llm before the claim."""
    import argparse

    from voxint import cli

    with session_factory() as session:
        set_phase(session, GpuPhase.LLM, now=NOW)
        run = seed_run(session, Stage.FINALIZE, status="completed")
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=2.0,
                raw_text="Hello there",
            )
        )
        speaker = Speaker(display_name="Inline flip test")
        session.add(speaker)
        session.commit()
        run_id, speaker_id = run.id, speaker.id
    settings = phase_settings()
    monkeypatch.setattr("voxint.config.get_settings", lambda: settings)
    monkeypatch.setattr("voxint.db.session.build_engine", lambda *a, **kw: None)
    monkeypatch.setattr("voxint.db.session.build_session_factory", lambda _: session_factory)
    real_blocked = cli._llm_phase_blocked

    def blocked_then_flip(factory: sessionmaker[Session], s: object) -> str | None:
        result = real_blocked(factory, settings)
        with session_factory() as session:
            set_phase(session, GpuPhase.DRAINING_POST, now=NOW + timedelta(seconds=1))
            session.commit()
        return result

    monkeypatch.setattr(cli, "_llm_phase_blocked", blocked_then_flip)
    # Gates other than the phase must pass so the job row is created.
    monkeypatch.setattr("voxint.enrichment.research_jobs.research_gates_open", lambda *a: True)
    monkeypatch.setattr("voxint.enrichment.asset_jobs.run_asset_gates_open", lambda *a: True)
    if command == "research":
        code = cli._research_speaker(argparse.Namespace(speaker_id=str(speaker_id), note=None))
        model: type[ResearchJob] | type[RunAssetJob] = ResearchJob
    else:
        code = cli._enrich_assets(argparse.Namespace(run_id=run_id, kind=["summary"]))
        model = RunAssetJob
    out = capsys.readouterr().out
    assert code == 1, out
    assert "deferred: GPU sharing switched the GPU" in out
    with session_factory() as session:
        jobs = session.query(model).all()
        assert len(jobs) == 1 and jobs[0].status == "queued"


def test_llm_unavailable_message_without_a_row(session_factory: sessionmaker[Session]) -> None:
    from voxint.gpu_phase.state import llm_unavailable_message

    with session_factory() as session:
        session.execute(text("DELETE FROM gpu_phase"))
        message = llm_unavailable_message(session, phase_settings())
        assert message is not None and "no phase record" in message
        assert llm_unavailable_message(session, phase_settings(gpu_phase_enabled=False)) is None
        session.rollback()
