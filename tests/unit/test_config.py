import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from voxint.config import Settings, SettingsError, get_settings


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


def test_csrf_secret_empty_is_allowed() -> None:
    # Empty is the "mint an auto per-process secret" signal — not a validation error.
    assert Settings(_env_file=None, csrf_secret="").csrf_secret == ""


def test_csrf_secret_too_short_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 16 characters"):
        Settings(_env_file=None, csrf_secret="short")


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
    with pytest.raises(ValidationError, match="readable regular file"):
        Settings(_env_file=None, ytdlp_cookies_file=missing)


def test_get_settings_redacts_credential_paths_from_errors(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    # get_settings() is the single production construction point and its
    # ValidationError -> SettingsError sanitization is the real "never expose the
    # cookies path" guarantee. Probe it structurally: pydantic attaches the raw
    # value to .errors()['input']/.json() (hide_input_in_errors only cleans
    # str()), and this basename is short enough to survive str() truncation, so a
    # weak assertion could false-pass. Assert on str + repr + args together.
    secret_basename = "topsecret-cookies.txt"
    missing = tmp_path / secret_basename
    monkeypatch.chdir(tmp_path)  # no ambient .env leaks in
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(missing))

    with pytest.raises(SettingsError) as exc:
        get_settings()

    surfaced = "".join((str(exc.value), repr(exc.value), repr(exc.value.args)))
    assert secret_basename not in surfaced
    assert str(missing) not in surfaced
    assert str(tmp_path) not in surfaced
    # It still tells the operator which setting is at fault, just not its value.
    assert "ytdlp_cookies_file" in surfaced


def test_get_settings_returns_settings_when_valid(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    # A readable cookies file loads cleanly through the sanitizing wrapper.
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookies))
    settings = get_settings()
    assert settings.ytdlp_cookies_file == cookies


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


def test_compute_tier_defaults_to_gpu_baseline() -> None:
    s = Settings(_env_file=None)
    assert s.compute_tier == "gpu"
    assert s.gpu_http_timeout_seconds == 14400.0
    assert s.stage_lease_seconds == 21600
    assert s.diarize_embed_lease_seconds == 43200
    assert s.celery_visibility_timeout_seconds == 172800


def test_cpu_tier_scales_default_timing_chain() -> None:
    s = Settings(_env_file=None, compute_tier="cpu")
    assert s.gpu_http_timeout_seconds == 14400.0 * 4
    assert s.stage_lease_seconds == 21600 * 4
    assert s.diarize_embed_lease_seconds == 43200 * 4
    assert s.celery_visibility_timeout_seconds == 172800 * 4
    # Scaled ints stay ints (Celery/engine consume these as ints).
    assert isinstance(s.stage_lease_seconds, int)
    # Download-bound budgets are tier-independent.
    assert s.acquire_timeout_seconds == 7200.0
    assert s.acquire_lease_seconds == 10800


@pytest.mark.parametrize("tier", ["rocm", "metal"])
def test_accelerated_tiers_keep_gpu_timing(tier: str) -> None:
    s = Settings(_env_file=None, compute_tier=tier)
    assert s.gpu_http_timeout_seconds == 14400.0
    assert s.stage_lease_seconds == 21600


def test_cpu_tier_never_overrides_explicit_values(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Explicit env values win over the profile — and if the explicit value
    # breaks the chain against the other (scaled) values, startup must fail
    # loudly instead of silently un-scaling anything.
    monkeypatch.setenv("COMPUTE_TIER", "cpu")
    monkeypatch.setenv("STAGE_LEASE_SECONDS", "30000")
    with pytest.raises(SettingsError, match="stage_lease_seconds"):
        get_settings()
    # A consistent explicit set is accepted unscaled.
    monkeypatch.setenv("GPU_HTTP_TIMEOUT_SECONDS", "14400")
    s = get_settings()
    assert s.stage_lease_seconds == 30000
    assert s.gpu_http_timeout_seconds == 14400.0
    # Untouched fields still get the profile.
    assert s.diarize_embed_lease_seconds == 43200 * 4


def test_gpu_timeout_must_fit_stage_leases() -> None:
    # Timeout reaching the lease minus margin opens a duplicate-execution
    # window (recovery reclaims a stage still persisting a slow call).
    with pytest.raises(ValidationError, match="persistence margin"):
        Settings(_env_file=None, gpu_http_timeout_seconds=21600.0)
    with pytest.raises(ValidationError, match="diarize_embed_lease_seconds"):
        Settings(_env_file=None, diarize_embed_lease_seconds=14500)
    Settings(_env_file=None)
