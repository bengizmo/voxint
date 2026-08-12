import os
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


def test_ytdlp_and_acquire_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.ytdlp_enabled is True
    assert s.ytdlp_proxy == ""
    assert s.ytdlp_cookies_file is None
    assert s.ytdlp_max_bytes == 5 * 1024**3
    assert s.acquire_timeout_seconds == 7200.0
    assert s.acquire_lease_seconds == 10800
    # Sized for six back-to-back stage leases (~39 h with default leases).
    assert s.celery_visibility_timeout_seconds == 172800


def test_acquire_timeout_must_fit_lease() -> None:
    # timeout + cleanup margin must stay strictly below the lease, or a stale
    # ACQUIRE attempt could overrun into a live one.
    with pytest.raises(ValidationError, match="acquire_lease_seconds"):
        Settings(_env_file=None, acquire_timeout_seconds=10800.0, acquire_lease_seconds=10800)
    # Boundary: lease exactly equal to timeout + margin is still rejected (>=).
    with pytest.raises(ValidationError, match="acquire_lease_seconds"):
        Settings(_env_file=None, acquire_timeout_seconds=700.0, acquire_lease_seconds=1000)
    # A comfortably separated pair validates.
    s = Settings(_env_file=None, acquire_timeout_seconds=600.0, acquire_lease_seconds=1200)
    assert s.acquire_lease_seconds == 1200


def test_ytdlp_cookies_file_must_be_readable_regular_file(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    # Happy path: an existing readable regular file is accepted.
    s = Settings(_env_file=None, ytdlp_cookies_file=cookies)
    assert s.ytdlp_cookies_file == cookies

    # A directory is not a regular file.
    with pytest.raises(ValidationError, match="readable regular file"):
        Settings(_env_file=None, ytdlp_cookies_file=tmp_path)

    # A missing file fails fast at startup, not mid-download.
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, ytdlp_cookies_file=missing)
    # The path is a credential locator — it must never appear in the error.
    assert str(missing) not in str(exc.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission bits")
def test_ytdlp_cookies_file_unreadable_is_rejected(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    cookies.chmod(0o000)
    try:
        with pytest.raises(ValidationError, match="readable regular file"):
            Settings(_env_file=None, ytdlp_cookies_file=cookies)
    finally:
        cookies.chmod(0o600)  # let tmp_path cleanup unlink it


def test_celery_visibility_must_cover_all_leases() -> None:
    # Below the six-stage lease sum: acks-late redelivery could fire mid-run.
    with pytest.raises(ValidationError, match="stage leases"):
        Settings(_env_file=None, celery_visibility_timeout_seconds=1000)
    # The floor tracks lease edits: a longer diarize lease can outgrow the
    # default visibility horizon and must be caught.
    with pytest.raises(ValidationError, match="stage leases"):
        Settings(_env_file=None, diarize_embed_lease_seconds=200000)
    # Default settings clear the floor.
    Settings(_env_file=None)
