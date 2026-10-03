"""Plain-language GPU-sharing state for the console, the CLI and ``voxint doctor``.

Read-only. Every surface that explains GPU sharing (the Runs banner, the
progress strip, the Status page, doctor) takes its wording from here so the
copy cannot drift between them.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from voxint.config import Settings
from voxint.gpu_phase.state import (
    NEVER_TICKED,
    GpuPhase,
    GpuPhaseSnapshot,
    llm_lane_ready,
    post_lane_open,
    read_phase,
)

# One short phrase per phase, for status rows and the CLI.
PHASE_SUMMARY: dict[GpuPhase, str] = {
    GpuPhase.LLM: "the GPU is serving the language model",
    GpuPhase.DRAINING_POST: "switching the GPU to audio work (waiting for language-model work)",
    GpuPhase.ACQUIRING: "switching the GPU to audio work (asking for the GPU)",
    GpuPhase.STARTING_SERVICES: "switching the GPU to audio work (starting the model services)",
    GpuPhase.AUDIO: "the GPU is doing audio work",
    GpuPhase.DRAINING: "switching the GPU back (waiting for audio work to finish)",
    GpuPhase.STOPPING_SERVICES: "switching the GPU back (stopping the model services)",
    GpuPhase.RELEASING: "switching the GPU back (handing it to the language model)",
    GpuPhase.ERROR: "could not return the GPU to the language model",
}

_TO_AUDIO = frozenset({GpuPhase.DRAINING_POST, GpuPhase.ACQUIRING, GpuPhase.STARTING_SERVICES})
_TO_LLM = frozenset({GpuPhase.DRAINING, GpuPhase.STOPPING_SERVICES, GpuPhase.RELEASING})

# Phases where the model services are meant to be up: audio, and draining
# (in-flight GPU runs still need them until they finish).
_SERVICES_UP = frozenset({GpuPhase.AUDIO, GpuPhase.DRAINING})

# Phases where the other service (the language model) is meant to be stopped
# because Voxint holds, or is taking or returning, the GPU. ``error`` is not
# here: the language model being down then is the fault the GPU sharing check
# reports, not an expected stop.
_LLM_DOWN = frozenset(
    {
        GpuPhase.ACQUIRING,
        GpuPhase.STARTING_SERVICES,
        GpuPhase.AUDIO,
        GpuPhase.DRAINING,
        GpuPhase.STOPPING_SERVICES,
        GpuPhase.RELEASING,
    }
)

LLM_EXPECTED_DOWN_DETAIL = "not answering; expected while Voxint holds the GPU for audio work"


# A missing row closes both lanes while the feature is on; the orchestrator
# repairs it on its next write.
NO_ROW_SUMMARY = "no phase recorded yet; both lanes wait until the GPU sharing task records one"

_LAST_ERROR_MAX = 300
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")


def snapshot_phase(snapshot: GpuPhaseSnapshot | None) -> GpuPhase | None:
    return snapshot.phase if snapshot is not None else None


def phase_summary(phase: GpuPhase | None) -> str:
    return PHASE_SUMMARY[phase] if phase is not None else NO_ROW_SUMMARY


def readiness_summary(
    snapshot: GpuPhaseSnapshot | None, *, tick_seconds: int, now: datetime | None = None
) -> str:
    if (
        snapshot is not None
        and snapshot.phase == GpuPhase.LLM
        and not llm_lane_ready(
            snapshot.phase,
            snapshot.llm_ready,
            snapshot.llm_checked_at,
            tick_seconds=tick_seconds,
            now=now,
        )
    ):
        if not is_fresh(snapshot, now=now or datetime.now(UTC), tick_seconds=tick_seconds):
            # A stale check means the phase task stopped, not that the model is silent.
            return "language-model work is paused because the GPU sharing task stopped updating"
        return "waiting for the language model to answer"
    return phase_summary(snapshot_phase(snapshot))


def utc_minute(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def display_error(text: str) -> str:
    """``last_error`` made safe to print: control characters become spaces and
    the length is bounded."""
    flat = " ".join(_CONTROL_CHARS.sub(" ", text).split())
    if len(flat) <= _LAST_ERROR_MAX:
        return flat
    return flat[: _LAST_ERROR_MAX - 3].rstrip() + "..."


def not_run_text(updated_at: datetime) -> str:
    """How long the phase task has been silent, without printing the sentinel."""
    if updated_at == NEVER_TICKED:
        return "the GPU sharing task has not run yet"
    return f"the GPU sharing task has not run since {utc_minute(updated_at)}"


def stale_after_seconds(tick_seconds: int) -> int:
    """The phase task writes the row every tick; this long without a write means
    it is not running."""
    return max(3 * tick_seconds, 120)


def is_fresh(snapshot: GpuPhaseSnapshot, *, now: datetime, tick_seconds: int) -> bool:
    return (now - snapshot.updated_at).total_seconds() <= stale_after_seconds(tick_seconds)


def model_service_stop_detail(phase: GpuPhase) -> str | None:
    """Why a not-running model service is expected in ``phase``, or None when it
    should be up."""
    if phase in _SERVICES_UP:
        return None
    if phase in {GpuPhase.LLM, GpuPhase.DRAINING_POST}:
        return "stopped while the GPU serves the language model"
    if phase in {GpuPhase.ACQUIRING, GpuPhase.STARTING_SERVICES}:
        return "starting; GPU sharing is switching the GPU to audio work"
    if phase == GpuPhase.ERROR:
        return "stopped; GPU sharing could not return the GPU (see the GPU sharing check)"
    return "stopped while GPU sharing hands the GPU back"


def llm_expected_down(phase: GpuPhase) -> bool:
    return phase in _LLM_DOWN


def stage_pause_reason(phase: GpuPhase) -> str | None:
    """Progress-strip reason for a GPU stage whose model service is stopped on purpose."""
    if phase in _SERVICES_UP:
        return None
    if phase in {GpuPhase.LLM, GpuPhase.DRAINING_POST}:
        return "the GPU is serving the language model"
    if phase == GpuPhase.ERROR:
        return "GPU sharing needs attention"
    return "the GPU is switching"


def _recordings(count: int) -> str:
    return "1 recording is queued" if count == 1 else f"{count} recordings are queued"


def _llm_jobs(count: int) -> str:
    return (
        "1 language-model job is queued."
        if count == 1
        else f"{count} language-model jobs are queued."
    )


@dataclass(frozen=True)
class GpuSharingState:
    """The stored phase as the console reads it, without any lane counts."""

    phase: GpuPhase | None  # None: no row recorded yet
    stale_since: datetime | None  # set when the phase task stopped writing the row
    post_ready: bool = True

    @property
    def trusted_phase(self) -> GpuPhase | None:
        """The phase when the row is present and fresh: the only case in which a
        stopped service may be called expected."""
        return self.phase if self.stale_since is None else None

    @property
    def stage_reason(self) -> str | None:
        phase = self.trusted_phase
        if phase == GpuPhase.LLM and not self.post_ready:
            return "waiting for the language model to answer"
        return stage_pause_reason(phase) if phase is not None else None


@dataclass(frozen=True)
class GpuSharingView:
    """What the Runs page says about GPU sharing right now."""

    phase: GpuPhase | None
    waiting: int
    # Banner headline and detail; None when no banner is due (audio, or the
    # language-model phase with nothing waiting).
    headline: str | None
    detail: str | None
    # Short footer text for the progress strip and the page summary.
    note: str | None
    is_error: bool  # shown as an alert
    release_hint: bool  # the template adds the release command


def build_view(state: GpuSharingState, waiting: int, post_waiting: int = 0) -> GpuSharingView:
    phase = state.phase
    headline: str | None = None
    detail: str | None = None
    note: str | None = None
    if state.stale_since is not None:
        headline = "GPU sharing is not running."
        silent = not_run_text(state.stale_since)
        detail = (
            f"{silent[0].upper()}{silent[1:]}, "
            "so the GPU is not switching between audio and language-model work. "
            "Check that the gpu-phase service is running; see the Status page."
        )
        note = "GPU sharing is not running"
    elif phase is None:
        headline = "Waiting for GPU sharing to start."
        detail = (f"{_recordings(waiting)}. " if waiting else "") + (
            "No GPU phase is recorded yet, so audio and language-model work wait "
            "until the GPU sharing task records one."
        )
        note = "waiting for GPU sharing to start"
    elif phase == GpuPhase.ERROR:
        headline = "Could not return the GPU to the language model."
        # The Runs template adds the release command and the Status page link.
        detail = "LLM work is paused."
        note = "GPU sharing needs attention"
    elif phase == GpuPhase.LLM:
        if not state.post_ready:
            headline = "Waiting for the language model to answer."
            parts = [
                _llm_jobs(post_waiting)
                if post_waiting
                else "Language-model work starts once it answers."
            ]
            if waiting:
                parts.append(f"{_recordings(waiting)} for the next audio window.")
            detail = " ".join(parts)
            note = "waiting for the language model to answer"
        elif waiting:
            headline = "Waiting for the GPU."
            detail = (
                f"{_recordings(waiting)}; the GPU is serving the language model "
                "until the next audio window."
            )
            note = "waiting for the GPU"
    elif phase in _TO_AUDIO:
        headline = "Switching the GPU to audio work."
        detail = (
            f"{_recordings(waiting)} and will start once the model services are up."
            if waiting
            else "Audio work starts once the model services are up."
        )
        note = "switching the GPU to audio work"
    elif phase in _TO_LLM:
        headline = "Switching the GPU back."
        detail = (
            "Language-model work resumes once the GPU is handed back "
            "and the language model answers."
        ) + (
            f" {_recordings(waiting)} for the next audio window." if waiting else ""
        )
        note = "switching the GPU back"
    stale = state.stale_since is not None
    return GpuSharingView(
        phase=phase,
        waiting=waiting,
        headline=headline,
        detail=detail,
        note=note,
        is_error=stale or phase == GpuPhase.ERROR,
        release_hint=not stale and phase == GpuPhase.ERROR,
    )


def read_state(
    session: Session, settings: Settings, *, now: datetime | None = None
) -> GpuSharingState | None:
    """None when GPU sharing is off: no ``gpu_phase`` query is made. One
    primary-key read otherwise; lane counts are the caller's choice."""
    if not settings.gpu_phase_enabled:
        return None
    snapshot = read_phase(session)
    if snapshot is None:
        return GpuSharingState(phase=None, stale_since=None)
    current = now or datetime.now(UTC)
    fresh = is_fresh(snapshot, now=current, tick_seconds=settings.gpu_phase_tick_seconds)
    return GpuSharingState(
        phase=snapshot.phase,
        stale_since=None if fresh else snapshot.updated_at,
        post_ready=post_lane_open(snapshot, settings, now=current),
    )
