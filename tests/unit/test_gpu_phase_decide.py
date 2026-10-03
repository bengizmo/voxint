"""Pure phase transition matrix, including dwell, draining, expiry and backoff."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Unpack

import pytest

from tests.unit.test_gpu_phase import phase_settings
from voxint.api.service_control import ServiceState
from voxint.gpu_phase.client import AcquireResult, ReleaseResult, StatusResult
from voxint.gpu_phase.orchestrator import Counts, Observed, decide
from voxint.gpu_phase.state import GpuPhase as P
from voxint.gpu_phase.state import GpuPhaseSnapshot, OperatorRequest, PhaseFields

NOW = datetime.now(UTC)


def snapshot(phase: P, age: int = 0, **kw: Unpack[PhaseFields]) -> GpuPhaseSnapshot:
    return replace(
        GpuPhaseSnapshot(
            1, phase, NOW - timedelta(seconds=age), None, None, None, 0, None, None, NOW
        ),
        **kw,
    )


@pytest.mark.parametrize(
    "phase,age,counts,observed,fields,expected",
    [
        (P.LLM, 600, Counts(1), Observed(), {}, P.DRAINING_POST),
        (P.LLM, 599, Counts(1), Observed(), {}, P.LLM),
        (P.LLM, 600, Counts(), Observed(), {}, P.LLM),
        (P.LLM, 600, Counts(1), Observed(), {"retry_after": NOW + timedelta(seconds=1)}, P.LLM),
        (
            P.LLM,
            0,
            Counts(),
            Observed(),
            {"operator_request": OperatorRequest.AUDIO, "retry_after": NOW + timedelta(seconds=1)},
            P.DRAINING_POST,
        ),
        (P.LLM, 0, Counts(), Observed(services=(ServiceState.RUNNING,)), {}, P.STOPPING_SERVICES),
        (
            P.LLM,
            0,
            Counts(),
            Observed(status=StatusResult("held", "voxint", "id")),
            {},
            P.STOPPING_SERVICES,
        ),
        (P.LLM, 0, Counts(), Observed(status=StatusResult("failed")), {}, P.LLM),
        (P.DRAINING_POST, 9999, Counts(post_in_flight=1), Observed(), {}, P.DRAINING_POST),
        (P.DRAINING_POST, 9999, Counts(running_llm_jobs=1), Observed(), {}, P.DRAINING_POST),
        (P.DRAINING_POST, 0, Counts(), Observed(), {}, P.ACQUIRING),
        (
            P.DRAINING_POST,
            0,
            Counts(post_in_flight=1),
            Observed(),
            {"operator_request": OperatorRequest.RELEASE},
            P.LLM,
        ),
        (
            P.ACQUIRING,
            0,
            Counts(),
            Observed(acquire=AcquireResult("held", "id")),
            {},
            P.STARTING_SERVICES,
        ),
        (P.ACQUIRING, 599, Counts(), Observed(acquire=AcquireResult("pending")), {}, P.ACQUIRING),
        (P.ACQUIRING, 600, Counts(), Observed(acquire=AcquireResult("pending")), {}, P.LLM),
        (P.ACQUIRING, 0, Counts(), Observed(acquire=AcquireResult("busy")), {}, P.LLM),
        (P.ACQUIRING, 0, Counts(), Observed(acquire=AcquireResult("failed")), {}, P.LLM),
        (P.STARTING_SERVICES, 0, Counts(), Observed(started=True), {}, P.STARTING_SERVICES),
        (P.STARTING_SERVICES, 0, Counts(), Observed(ready=True), {}, P.AUDIO),
        (P.STARTING_SERVICES, 600, Counts(), Observed(), {}, P.STOPPING_SERVICES),
        (
            P.STARTING_SERVICES,
            0,
            Counts(),
            Observed(control_error="start failed"),
            {},
            P.STOPPING_SERVICES,
        ),
        (P.AUDIO, 599, Counts(), Observed(), {}, P.AUDIO),
        (P.AUDIO, 600, Counts(), Observed(), {}, P.DRAINING),
        (P.AUDIO, 600, Counts(1), Observed(), {}, P.AUDIO),
        (P.AUDIO, 600, Counts(gpu_in_flight=1), Observed(), {}, P.AUDIO),
        (P.AUDIO, 7200, Counts(1, 1), Observed(), {}, P.DRAINING),
        (
            P.AUDIO,
            0,
            Counts(1, 1),
            Observed(),
            {"operator_request": OperatorRequest.RELEASE},
            P.DRAINING,
        ),
        (P.AUDIO, 0, Counts(), Observed(acquire=AcquireResult("lost")), {}, P.STOPPING_SERVICES),
        (P.AUDIO, 0, Counts(), Observed(acquire=AcquireResult("failed")), {}, P.AUDIO),
        (
            P.AUDIO,
            0,
            Counts(),
            Observed(acquire=AcquireResult("failed")),
            {"lease_expires_at": NOW},
            P.STOPPING_SERVICES,
        ),
        (P.AUDIO, 0, Counts(), Observed(services=(ServiceState.STOPPED,)), {}, P.STOPPING_SERVICES),
        (P.DRAINING, 0, Counts(gpu_in_flight=1), Observed(), {}, P.DRAINING),
        (P.DRAINING, 0, Counts(), Observed(), {}, P.STOPPING_SERVICES),
        (P.STOPPING_SERVICES, 0, Counts(), Observed(stopped=True), {}, P.RELEASING),
        (P.STOPPING_SERVICES, 0, Counts(), Observed(control_error="stop failed"), {}, P.ERROR),
        (P.RELEASING, 0, Counts(), Observed(release=ReleaseResult("released")), {}, P.LLM),
        (
            P.RELEASING,
            0,
            Counts(),
            Observed(release=ReleaseResult("failed", "unreachable")),
            {},
            P.ERROR,
        ),
        (P.ERROR, 0, Counts(), Observed(), {}, P.STOPPING_SERVICES),
        (P.ERROR, 0, Counts(), Observed(), {"retry_after": NOW + timedelta(seconds=1)}, P.ERROR),
        (
            P.ERROR,
            0,
            Counts(),
            Observed(),
            {
                "retry_after": NOW + timedelta(seconds=1),
                "operator_request": OperatorRequest.RELEASE,
            },
            P.STOPPING_SERVICES,
        ),
    ],
)
def test_matrix(
    phase: P, age: int, counts: Counts, observed: Observed, fields: PhaseFields, expected: P
) -> None:
    assert (
        decide(snapshot(phase, age, **fields), counts, observed, NOW, phase_settings()).phase
        == expected
    )


@pytest.mark.parametrize("phase", list(P))
def test_uncontrollable(phase: P) -> None:
    result = decide(snapshot(phase), Counts(), Observed(controllable=False), NOW, phase_settings())
    assert result.phase == (P.LLM if phase == P.LLM else P.ERROR)
    assert result.fields["last_error"] == "service control unavailable"


def test_backoff_and_failure_preservation() -> None:
    result = decide(
        snapshot(P.ACQUIRING, failures=2),
        Counts(),
        Observed(acquire=AcquireResult("busy", retry_after_seconds=300)),
        NOW,
        phase_settings(),
    )
    assert result.fields["failures"] == 3
    assert result.fields["retry_after"] == NOW + timedelta(seconds=300)
    s = snapshot(P.RELEASING, failures=3, last_error="start failed", retry_after=NOW)
    result = decide(s, Counts(), Observed(release=ReleaseResult("released")), NOW, phase_settings())
    assert result.fields["failures"] == 3
    assert result.fields["retry_after"] == NOW


def test_busy_backoff_honors_long_broker_delay() -> None:
    result = decide(
        snapshot(P.ACQUIRING),
        Counts(),
        Observed(acquire=AcquireResult("busy", retry_after_seconds=100000)),
        NOW,
        phase_settings(),
    )
    assert result.fields["retry_after"] == NOW + timedelta(seconds=100000)


def test_backoff_caps_repeated_failures() -> None:
    result = decide(
        snapshot(P.STOPPING_SERVICES, failures=100),
        Counts(),
        Observed(control_error="stop failed"),
        NOW,
        phase_settings(),
    )
    assert result.fields["failures"] == 101
    assert result.fields["retry_after"] == NOW + timedelta(seconds=900)
