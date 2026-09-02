"""Read the per-attempt model identity off a run's stage attempts, for display.

The pipeline stamps which model actually answered each stage onto that attempt's
``StageRun.metrics`` under :data:`voxint.pipeline.model_identity.METRICS_KEY`
(the write side and the on-disk shape live there). This module is the read side
the run-detail page uses: it selects the right attempt to trust and turns the
raw JSON into display-ready values, with no database query, HTTP call, or clock
of its own, so it unit-tests without a database like
:mod:`voxint.api.presentation`.

Selection rule: **the latest completed attempt per stage.** A stage can be
retried, and a failed or lease-expired attempt may have stamped a different (or
no) identity; only a completed attempt reflects the model that produced the
result the operator is looking at. The identity is read only from that latest
completed attempt; when it carries none the stage renders "Not recorded" (a
legacy run from before this provenance existed, a stage with no completed
attempt, or a latest completed attempt that did not record it) rather than
borrowing an older attempt's stamp.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from voxint.db.models import Stage, StageStatus
from voxint.pipeline.model_identity import METRICS_KEY

# The stages that call a model service, in run order, each with the roles it
# exercises as ``(identity key, operator-facing label)``. This mirrors
# ``model_identity._STAGE_MODEL_SERVICES`` (the probe side) as the display side;
# a role the probe stops recording simply renders "Not observed" here.
_MODEL_STAGES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (Stage.TRANSCRIBE.value, "Transcription", (("asr", "Transcription model"),)),
    (
        Stage.DIARIZE_EMBED.value,
        "Voice separation and embedding",
        (
            ("diarizer", "Voice-separation model"),
            ("embedder", "Speaker embedding model"),
        ),
    ),
)


class _StageAttempt(Protocol):
    """The StageRun fields this module reads. StageRun satisfies it; tests may
    pass any object with these attributes, keeping the module database-free."""

    stage: str
    status: str
    attempt: int
    metrics: dict[str, Any] | None


@dataclass(frozen=True)
class ModelRole:
    """One model service's identity as observed for a stage, ready to render.

    ``reachable`` False means the probe could not read a trustworthy identity
    just before the attempt (service down, still loading, or the role was never
    recorded); ``detail`` carries the plain-language reason and the model fields
    are ``None``.
    """

    role: str
    label: str
    reachable: bool
    model: str | None
    revision: str | None
    engine: str | None
    decode_config_hash: str | None
    detail: str | None


@dataclass(frozen=True)
class StageModels:
    """The model identity to show for one stage, from its latest completed attempt.

    ``recorded`` False means the latest completed attempt carried no identity,
    or no attempt completed at all (a pre-provenance run, or the probe machinery
    itself failed): the template shows "Not recorded" and ``roles`` is empty.
    """

    stage: str
    label: str
    recorded: bool
    attempt: int | None
    roles: tuple[ModelRole, ...]


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _role_from_payload(role: str, label: str, payload: object) -> ModelRole:
    """Normalize one role's identity payload into a display record.

    A missing or malformed payload, or one the probe marked unreachable, becomes
    a ``reachable=False`` record; ``detail`` carries the probe's plain-language
    reason when it recorded one and stays ``None`` otherwise (the template then
    renders a bare "Not observed" instead of parroting a placeholder). The model
    fields are only trusted on an explicitly reachable payload.
    """
    if not isinstance(payload, dict):
        return ModelRole(
            role=role,
            label=label,
            reachable=False,
            model=None,
            revision=None,
            engine=None,
            decode_config_hash=None,
            detail=None,
        )
    if payload.get("reachable") is not True:
        detail = _str_or_none(payload.get("detail"))
        return ModelRole(
            role=role,
            label=label,
            reachable=False,
            model=None,
            revision=None,
            engine=None,
            decode_config_hash=None,
            detail=detail,
        )
    return ModelRole(
        role=role,
        label=label,
        reachable=True,
        model=_str_or_none(payload.get("model")),
        revision=_str_or_none(payload.get("revision")),
        engine=_str_or_none(payload.get("engine")),
        decode_config_hash=_str_or_none(payload.get("decode_config_hash")),
        detail=None,
    )


def _latest_completed_identity(
    stage_runs: Iterable[_StageAttempt], stage: str
) -> tuple[int, dict[str, Any]] | None:
    """The identity object from the latest completed attempt of ``stage``.

    Selects the highest-numbered completed attempt first, stamped or not, then
    reads its ``model_identity``: ``(attempt, identity)`` when that attempt is
    stamped, ``None`` otherwise. Attempt number breaks the tie (it is unique per
    stage and monotonic with retries), so neither a failed/lease-expired attempt
    nor an older *stamped* attempt can mask the completion that actually
    produced the result — an unstamped latest completion renders "Not recorded"
    rather than borrowing a stale identity.
    """
    best: _StageAttempt | None = None
    for run in stage_runs:
        if run.stage != stage or run.status != StageStatus.COMPLETED.value:
            continue
        if best is None or run.attempt > best.attempt:
            best = run
    if best is None:
        return None
    metrics = best.metrics
    if not isinstance(metrics, dict):
        return None
    identity = metrics.get(METRICS_KEY)
    if not isinstance(identity, dict):
        return None
    return (best.attempt, identity)


def select_run_model_identity(
    stage_runs: Iterable[_StageAttempt],
) -> list[StageModels]:
    """The model provenance to render on the run-detail page, one row per stage.

    One :class:`StageModels` per model-bearing stage, in run order. Each reflects
    that stage's latest completed attempt; a stage with no completed, stamped
    attempt is returned with ``recorded=False`` so the template can say "Not
    recorded" rather than imply the models are unknown for a subtle reason.
    """
    runs = list(stage_runs)
    result: list[StageModels] = []
    for stage, label, roles in _MODEL_STAGES:
        selected = _latest_completed_identity(runs, stage)
        if selected is None:
            result.append(
                StageModels(
                    stage=stage, label=label, recorded=False, attempt=None, roles=()
                )
            )
            continue
        attempt, identity = selected
        result.append(
            StageModels(
                stage=stage,
                label=label,
                recorded=True,
                attempt=attempt,
                roles=tuple(
                    _role_from_payload(role, role_label, identity.get(role))
                    for role, role_label in roles
                ),
            )
        )
    return result
