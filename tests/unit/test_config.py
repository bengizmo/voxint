from pathlib import Path

import pytest
from pydantic import ValidationError

from voxint.config import Settings


def test_defaults_are_localhost_and_llm_disabled() -> None:
    s = Settings(_env_file=None)
    assert s.api_host == "127.0.0.1"
    assert s.llm_enabled is False
    assert s.domain_pack_path is None
    assert s.media_root == Path("/data/media")


def test_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("API_PORT", "9090")
    s = Settings(_env_file=None)
    assert s.llm_enabled is True
    assert s.api_port == 9090


def test_llm_budget_must_fit_stage_lease() -> None:
    with pytest.raises(ValidationError, match="stage_lease_seconds"):
        Settings(
            _env_file=None,
            llm_enabled=True,
            llm_run_budget_seconds=30000.0,
            stage_lease_seconds=21600,
        )
    # Irrelevant while the LLM is disabled — don't block unrelated overrides.
    Settings(_env_file=None, llm_run_budget_seconds=30000.0)


def test_gate_settings_reject_nan_and_out_of_range() -> None:
    # NaN comparisons are always False — a NaN threshold silently disables a gate.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, match_min_cosine=float("nan"))
    with pytest.raises(ValidationError):
        Settings(_env_file=None, match_min_cosine=1.5)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, match_min_vote_agreement=-0.1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, match_turn_weight_cap_seconds=0.0)


def test_grounding_gates_must_be_at_least_as_strict() -> None:
    with pytest.raises(ValidationError, match="at least as strict"):
        Settings(_env_file=None, grounded_min_cosine=0.5)  # below match_min_cosine 0.6
