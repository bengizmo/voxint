"""Real transaction/lock tests with a deterministic external GPU broker."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from unittest.mock import MagicMock

import pytest
from celery.exceptions import OperationalError
from sqlalchemy import delete

from tests.gpu_phase_fakes import FakeClock, FakeController, FakeGpuBroker, FakeProbe
from tests.integration.test_gpu_phase import seed_jobs, seed_run
from tests.unit.test_gpu_phase import phase_settings
from voxint.api.service_control import SERVICE_KEYS, ControlOutcome, ServiceState
from voxint.db.models import GpuPhaseState, Stage
from voxint.gpu_phase.client import LeaseClient
from voxint.gpu_phase.orchestrator import tick
from voxint.gpu_phase.state import (
    GpuPhase as P,
)
from voxint.gpu_phase.state import (
    OperatorRequest,
    gpu_lane_open,
    post_lane_open,
    read_phase,
    running_llm_jobs,
    set_phase,
    set_request,
)
from voxint.worker import tasks


@pytest.fixture
def rig(session_factory, monkeypatch):
    clock = FakeClock()
    broker = FakeGpuBroker(clock)
    controller = FakeController()
    probe = FakeProbe()
    settings = phase_settings(gpu_phase_min_dwell_seconds=0)
    client = LeaseClient(settings, transport=broker.transport)
    publishers = {}
    for name in (
        "run_pipeline",
        "finish_pipeline",
        "generate_run_asset",
        "translate_run",
        "research_speaker",
    ):
        publisher = MagicMock()
        monkeypatch.setattr(getattr(tasks, name), "apply_async", publisher)
        publishers[name] = publisher

    class Rig:
        def step(self):
            return tick(
                session_factory,
                settings,
                client=client,
                controller=controller,
                probe=probe,
                clock=clock,
            )

        def seed(self, phase, **fields):
            with session_factory.begin() as session:
                set_phase(session, phase, now=clock(), **fields)

        def read(self):
            with session_factory() as session:
                return read_phase(session)

    r = Rig()
    r.clock, r.broker, r.controller, r.probe = clock, broker, controller, probe
    r.settings, r.client, r.publishers = settings, client, publishers
    return r


def test_full_cycle_and_committed_publication(session_factory, rig):
    with session_factory.begin() as session:
        gpu = seed_run(session, None)
        post = seed_run(session, Stage.ENHANCE_MATCH)
        jobs = seed_jobs(session)

    def assert_committed(*args, **kwargs):
        assert rig.read().phase == P.AUDIO

    rig.publishers["run_pipeline"].side_effect = assert_committed
    phases = [rig.step().phase for _ in range(5)]
    assert phases == [
        P.DRAINING_POST,
        P.ACQUIRING,
        P.STARTING_SERVICES,
        P.STARTING_SERVICES,
        P.AUDIO,
    ]
    rig.publishers["run_pipeline"].assert_called_once_with((str(gpu.id),), ignore_result=True)
    with session_factory.begin() as session:
        set_request(session, OperatorRequest.RELEASE)
    assert [rig.step().phase for _ in range(4)] == [
        P.DRAINING,
        P.STOPPING_SERVICES,
        P.RELEASING,
        P.LLM,
    ]
    rig.publishers["finish_pipeline"].assert_called_once_with((str(post.id),), ignore_result=True)
    for name, job in zip(
        ("generate_run_asset", "translate_run", "research_speaker"), jobs, strict=True
    ):
        rig.publishers[name].assert_called_once_with((str(job.id),), ignore_result=True)
    assert rig.read().lease_id is None
    assert rig.read().failures == 0
    assert rig.broker.lease_id is None
    assert all(x == ServiceState.STOPPED for x in rig.controller.states.values())


@pytest.mark.parametrize("mode,elapsed", [("busy", 0), ("down", 0), ("pending", 600)])
def test_acquire_failure_backoff(rig, mode, elapsed):
    rig.seed(P.ACQUIRING)
    rig.broker.mode = mode
    rig.clock.advance(elapsed)
    assert rig.step().phase == P.LLM
    s = rig.read()
    assert s.failures == 1
    assert s.last_error
    assert s.retry_after == rig.clock() + timedelta(seconds=90 if mode == "busy" else 30)


def prepare_audio(rig, phase=P.AUDIO):
    held = rig.client.acquire()
    rig.seed(phase, lease_id=held.lease_id, lease_expires_at=held.expires_at)
    rig.controller.states = dict.fromkeys(SERVICE_KEYS, ServiceState.RUNNING)


@pytest.mark.parametrize("failure", ["start", "ready", "lost", "stop", "release"])
def test_failure_teardown_and_recovery(rig, failure):
    prepare_audio(rig)
    if failure == "start":
        rig.seed(P.STARTING_SERVICES)
        rig.controller.states["transcription"] = ServiceState.STOPPED
        rig.controller.failures[("start", "transcription")] = ControlOutcome.ERROR
    elif failure == "ready":
        rig.seed(P.STARTING_SERVICES)
        rig.probe.ready = False
        # Keep the lease held while the service readiness deadline elapses.
        rig.clock.advance(600)
        rig.broker.expiry = rig.clock() + timedelta(seconds=120)
    elif failure == "lost":
        rig.broker.lease_id = None
    elif failure == "stop":
        rig.seed(P.STOPPING_SERVICES)
        rig.controller.failures[("stop", "transcription")] = ControlOutcome.TIMEOUT
    else:
        rig.seed(P.RELEASING)
        rig.controller.states = dict.fromkeys(SERVICE_KEYS, ServiceState.STOPPED)
        rig.broker.release_mode = "down"
    expected = P.ERROR if failure in ("stop", "release") else P.STOPPING_SERVICES
    assert rig.step().phase == expected
    failed = rig.read()
    assert failed.last_error and failed.failures == 1
    if expected == P.ERROR:
        assert "could not return the GPU" in failed.last_error
        assert rig.step().phase == P.ERROR
    rig.controller.failures.clear()
    rig.broker.release_mode = "held"
    rig.clock.advance(30)
    for _ in range(4):
        if rig.read().phase == P.LLM:
            break
        rig.step()
    assert rig.read().phase == P.LLM
    assert rig.read().last_error == failed.last_error
    assert rig.read().failures == 1
    assert rig.read().lease_id is None


@pytest.mark.parametrize("phase", list(P))
def test_restart_each_phase_reaches_open_lane(rig, phase):
    if phase in (
        P.STARTING_SERVICES,
        P.AUDIO,
        P.DRAINING,
        P.STOPPING_SERVICES,
        P.RELEASING,
        P.ERROR,
    ):
        prepare_audio(rig, phase)
    else:
        rig.seed(phase)
    for _ in range(12):
        result = rig.step()
        if result.phase in (P.AUDIO, P.LLM):
            break
    s = rig.read()
    assert s.phase in (P.AUDIO, P.LLM)
    assert gpu_lane_open(s, rig.settings) == (s.phase == P.AUDIO)
    assert post_lane_open(s, rig.settings) == (s.phase == P.LLM)
    assert all(
        value == (ServiceState.RUNNING if s.phase == P.AUDIO else ServiceState.STOPPED)
        for value in rig.controller.states.values()
    )


@pytest.mark.parametrize("running", [False, True])
def test_llm_leak_reconcile(rig, running):
    held = rig.client.acquire()
    rig.seed(P.LLM)
    if running:
        rig.controller.states = dict.fromkeys(SERVICE_KEYS, ServiceState.RUNNING)
    assert rig.step().phase == P.STOPPING_SERVICES
    assert rig.read().lease_id == held.lease_id
    assert [rig.step().phase for _ in range(2)] == [P.RELEASING, P.LLM]


def test_uncontrollable_disabled_and_missing_row(session_factory, rig):
    with session_factory.begin() as session:
        session.execute(delete(GpuPhaseState))
    rig.settings.gpu_phase_enabled = False
    assert rig.step().kind == "disabled"
    assert rig.read() is None
    assert not rig.broker.calls and not rig.controller.calls
    rig.settings.gpu_phase_enabled = True
    rig.controller.controllable = False
    assert rig.step().phase == P.LLM
    assert rig.read().last_error == "service control unavailable"
    assert rig.read().phase_since == rig.clock()
    rig.seed(P.ACQUIRING)
    assert rig.step().phase == P.ERROR


def test_concurrent_ticks_try_lock_without_touching_row(session_factory, rig, monkeypatch):
    rig.seed(P.LLM)
    before = rig.read()
    entered, finish = Event(), Event()
    original = rig.client.status

    def hold_status():
        entered.set()
        assert finish.wait(10)
        return original()

    monkeypatch.setattr(rig.client, "status", hold_status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rig.step)
        try:
            assert entered.wait(10)
            second = pool.submit(rig.step)
            assert second.result(timeout=5).kind == "busy"
            assert rig.read() == before
        finally:
            finish.set()
        assert first.result(timeout=5).kind == "waiting"


def test_running_llm_counts_block_drain(session_factory, rig):
    with session_factory.begin() as session:
        jobs = seed_jobs(session)
        for job in jobs:
            job.status = "running"
        session.flush()
        assert running_llm_jobs(session) == 3
    rig.seed(P.DRAINING_POST)
    assert rig.step().phase == P.DRAINING_POST


def test_publication_failure_is_deferred(session_factory, rig):
    with session_factory.begin() as session:
        seed_run(session, Stage.ENHANCE_MATCH)
        seed_jobs(session)
    for publisher in rig.publishers.values():
        publisher.side_effect = OperationalError("offline")
    rig.seed(P.RELEASING)
    assert rig.step().phase == P.LLM
    assert rig.read().phase == P.LLM
    for name in ("finish_pipeline", "generate_run_asset", "translate_run", "research_speaker"):
        assert rig.publishers[name].call_count == 1


@pytest.mark.parametrize("mode", ["held", "down"])
def test_release_recovers_unknown_lease(rig, mode):
    rig.client.acquire()
    rig.seed(P.RELEASING, lease_id=None)
    rig.broker.mode = mode
    assert rig.step().phase == (P.LLM if mode == "held" else P.ERROR)
    if mode == "held":
        assert rig.broker.lease_id is None
    else:
        assert "broker status unavailable" in rig.read().last_error


@pytest.mark.parametrize("expiry", [None, -1, 60])
def test_audio_renew_failure_respects_optional_expiry(rig, expiry):
    prepare_audio(rig)
    rig.settings.gpu_phase_min_dwell_seconds = 600
    rig.seed(
        P.AUDIO,
        lease_expires_at=None if expiry is None else rig.clock() + timedelta(seconds=expiry),
    )
    rig.broker.mode = "down"
    assert rig.step().phase == (P.STOPPING_SERVICES if expiry == -1 else P.AUDIO)


def test_bounded_lane_publication(session_factory, rig):
    rig.settings.recovery_publish_batch_size = 1
    with session_factory.begin() as session:
        oldest = seed_run(session, None, age=10000)
        seed_run(session, None, age=9000)
        seed_run(session, Stage.ENHANCE_MATCH, age=11000)
    prepare_audio(rig, P.STARTING_SERVICES)
    assert rig.step().phase == P.AUDIO
    rig.publishers["run_pipeline"].assert_called_once_with((str(oldest.id),), ignore_result=True)
    rig.publishers["finish_pipeline"].assert_not_called()


def test_worker_task_constructs_dependencies(session_factory, rig, monkeypatch):
    monkeypatch.setattr(tasks, "get_settings", lambda: rig.settings)
    monkeypatch.setattr(tasks, "_runtime", lambda: (session_factory, None))
    monkeypatch.setattr(
        "voxint.api.service_control.get_controller", lambda settings: rig.controller
    )
    monkeypatch.setattr("voxint.api.health_probe.probe_services", rig.probe)
    monkeypatch.setattr("voxint.gpu_phase.client.LeaseClient", lambda settings: rig.client)
    assert tasks.gpu_phase_tick() == "waiting"
    rig.settings.gpu_phase_enabled = False
    monkeypatch.setattr(tasks, "_runtime", lambda: pytest.fail("disabled task accessed DB"))
    assert tasks.gpu_phase_tick() == "disabled"
