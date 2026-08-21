"""Run-detail model-provenance selection and normalization (B1) — no DB.

Reads the ``StageRun.metrics["model_identity"]`` shape the pipeline stamps and
turns it into the display records the run-detail page renders. The selection
rule under test is the load-bearing one: the *latest completed* attempt per
stage, so a failed or lease-expired retry can never mask (or overwrite the
displayed identity of) a completed attempt.
"""

from dataclasses import dataclass
from typing import Any

from voxint.api.model_provenance import (
    ModelRole,
    StageModels,
    select_run_model_identity,
)
from voxint.db.models import Stage, StageStatus
from voxint.pipeline.model_identity import METRICS_KEY


@dataclass
class _Attempt:
    """A StageRun stand-in carrying only the fields the reader touches."""

    stage: str
    status: str
    attempt: int
    metrics: dict[str, Any] | None


def _identity(**roles: object) -> dict[str, Any]:
    """A ``model_identity`` metrics object with the given role payloads."""
    return {"v": 1, "observed_before_attempt": True, **roles}


def _asr(**fields: object) -> dict[str, Any]:
    return {"reachable": True, "model": "large-v2", "engine": "ct2-legacy", **fields}


def _stage(result: list[StageModels], stage: Stage) -> StageModels:
    return next(s for s in result if s.stage == stage.value)


def _role(stage: StageModels, role: str) -> ModelRole:
    return next(r for r in stage.roles if r.role == role)


# ---- shape and ordering -----------------------------------------------------


def test_returns_both_model_stages_in_run_order() -> None:
    result = select_run_model_identity([])
    assert [s.stage for s in result] == [
        Stage.TRANSCRIBE.value,
        Stage.DIARIZE_EMBED.value,
    ]


def test_empty_ledger_records_nothing() -> None:
    result = select_run_model_identity([])
    assert all(s.recorded is False and s.roles == () for s in result)
    assert all(s.attempt is None for s in result)


# ---- selection rule: latest completed attempt -------------------------------


def test_selects_latest_completed_attempt() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity(asr=_asr(model="large-v2"))},
        ),
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            2,
            {METRICS_KEY: _identity(asr=_asr(model="large-v3"))},
        ),
    ]
    transcribe = _stage(select_run_model_identity(runs), Stage.TRANSCRIBE)
    assert transcribe.recorded is True
    assert transcribe.attempt == 2
    assert _role(transcribe, "asr").model == "large-v3"


def test_later_failed_attempt_never_masks_a_completed_one() -> None:
    # The failure mode this whole design exists to prevent: a failed attempt 2
    # (which stamped nothing, or stamped an unreachable probe) must not replace
    # the completed attempt 1's displayed identity.
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity(asr=_asr(model="large-v2"))},
        ),
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.FAILED.value,
            2,
            {METRICS_KEY: _identity(asr={"reachable": False, "detail": "timeout"})},
        ),
    ]
    transcribe = _stage(select_run_model_identity(runs), Stage.TRANSCRIBE)
    assert transcribe.attempt == 1
    assert _role(transcribe, "asr").model == "large-v2"


def test_running_and_failed_attempts_are_ignored_when_no_completed_exists() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.RUNNING.value,
            1,
            {METRICS_KEY: _identity(asr=_asr())},
        ),
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.FAILED.value,
            2,
            {METRICS_KEY: _identity(asr=_asr())},
        ),
    ]
    transcribe = _stage(select_run_model_identity(runs), Stage.TRANSCRIBE)
    assert transcribe.recorded is False


# ---- "Not recorded" cases ---------------------------------------------------


def test_completed_attempt_without_metrics_is_not_recorded() -> None:
    runs = [_Attempt(Stage.TRANSCRIBE.value, StageStatus.COMPLETED.value, 1, None)]
    assert _stage(select_run_model_identity(runs), Stage.TRANSCRIBE).recorded is False


def test_completed_attempt_missing_the_identity_key_is_not_recorded() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {"some_other_metric": 1},
        )
    ]
    assert _stage(select_run_model_identity(runs), Stage.TRANSCRIBE).recorded is False


def test_identity_key_holding_a_non_dict_is_not_recorded() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: "corrupt"},
        )
    ]
    assert _stage(select_run_model_identity(runs), Stage.TRANSCRIBE).recorded is False


# ---- role normalization -----------------------------------------------------


def test_reachable_role_surfaces_all_identity_fields() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {
                METRICS_KEY: _identity(
                    asr=_asr(
                        model="large-v2",
                        revision="a" * 40,
                        engine="ct2-legacy",
                        decode_config_hash="deadbeefcafe0000",
                    )
                )
            },
        )
    ]
    asr = _role(_stage(select_run_model_identity(runs), Stage.TRANSCRIBE), "asr")
    assert asr.reachable is True
    assert asr.model == "large-v2"
    assert asr.revision == "a" * 40
    assert asr.engine == "ct2-legacy"
    assert asr.decode_config_hash == "deadbeefcafe0000"
    assert asr.detail is None


def test_unreachable_role_passes_the_detail_through() -> None:
    runs = [
        _Attempt(
            Stage.DIARIZE_EMBED.value,
            StageStatus.COMPLETED.value,
            1,
            {
                METRICS_KEY: _identity(
                    diarizer={"reachable": False, "detail": "degraded (model not loaded)"},
                    embedder=_asr(model="titanet-large-v1", engine=None),
                )
            },
        )
    ]
    stage = _stage(select_run_model_identity(runs), Stage.DIARIZE_EMBED)
    diarizer = _role(stage, "diarizer")
    assert diarizer.reachable is False
    assert diarizer.detail == "degraded (model not loaded)"
    assert diarizer.model is None
    # The embedder role is recorded for completeness even though it is fixed.
    embedder = _role(stage, "embedder")
    assert embedder.reachable is True
    assert embedder.model == "titanet-large-v1"
    assert embedder.engine is None


def test_missing_role_in_a_recorded_identity_reads_not_observed() -> None:
    # The defensive-empty identity object (probe machinery itself failed) carries
    # v/observed_before_attempt but no role keys. The stage is recorded, but each
    # role honestly reads "not observed".
    runs = [
        _Attempt(
            Stage.DIARIZE_EMBED.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity()},
        )
    ]
    stage = _stage(select_run_model_identity(runs), Stage.DIARIZE_EMBED)
    assert stage.recorded is True
    for role in stage.roles:
        assert role.reachable is False
        assert role.detail == "not observed"


def test_malformed_role_payload_reads_not_observed() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity(asr="not-a-dict")},
        )
    ]
    asr = _role(_stage(select_run_model_identity(runs), Stage.TRANSCRIBE), "asr")
    assert asr.reachable is False
    assert asr.detail == "not observed"


def test_non_string_identity_fields_coerce_to_none() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {
                METRICS_KEY: _identity(
                    asr={"reachable": True, "model": 123, "revision": None, "engine": ["x"]}
                )
            },
        )
    ]
    asr = _role(_stage(select_run_model_identity(runs), Stage.TRANSCRIBE), "asr")
    assert asr.reachable is True
    assert asr.model is None
    assert asr.revision is None
    assert asr.engine is None


def test_reachable_field_must_be_exactly_true() -> None:
    # A truthy-but-not-True value is not a trustworthy identity.
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity(asr={"reachable": "yes", "model": "large-v2"})},
        )
    ]
    asr = _role(_stage(select_run_model_identity(runs), Stage.TRANSCRIBE), "asr")
    assert asr.reachable is False
    assert asr.model is None


def test_both_stages_render_independently() -> None:
    runs = [
        _Attempt(
            Stage.TRANSCRIBE.value,
            StageStatus.COMPLETED.value,
            1,
            {METRICS_KEY: _identity(asr=_asr(model="large-v2"))},
        ),
        _Attempt(
            Stage.DIARIZE_EMBED.value,
            StageStatus.COMPLETED.value,
            1,
            {
                METRICS_KEY: _identity(
                    diarizer=_asr(model="pyannote/speaker-diarization-3.1", engine=None),
                    embedder=_asr(model="titanet-large-v1", engine=None),
                )
            },
        ),
    ]
    result = select_run_model_identity(runs)
    assert _stage(result, Stage.TRANSCRIBE).recorded is True
    diarize = _stage(result, Stage.DIARIZE_EMBED)
    assert diarize.recorded is True
    assert {r.role for r in diarize.roles} == {"diarizer", "embedder"}
