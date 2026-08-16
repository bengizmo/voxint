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


def test_notify_defaults_off_and_present() -> None:
    # Off by default with no endpoint — the zero-config path never notifies.
    s = Settings(_env_file=None)
    assert s.notify_enabled is False
    assert s.notify_webhook_url == ""
    assert s.notify_webhook_secret == ""
    assert s.notify_max_attempts == 8
    assert s.notify_lease_seconds == 60


def test_notify_disabled_ignores_incomplete_config() -> None:
    # A stray URL/secret while disabled is not an error — no ceremony on the
    # default path, and enabling later is what triggers the completeness gate.
    s = Settings(_env_file=None, notify_webhook_url="not-a-url", notify_webhook_secret="x")
    assert s.notify_enabled is False


def test_notify_enabled_requires_url() -> None:
    with pytest.raises(ValidationError, match="notify_webhook_url is required"):
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_secret="a-sufficiently-long-secret",
        )


def test_notify_enabled_rejects_non_public_url() -> None:
    # Reuses the URL-ingestion string gate: localhost / private / credentialed
    # endpoints are refused, and the error never echoes the URL.
    for bad in (
        "http://localhost/hook",
        "http://127.0.0.1/hook",
        "http://10.0.0.5/hook",
        "https://user:pass@example.com/hook",
        "ftp://example.com/hook",
    ):
        with pytest.raises(ValidationError, match="notify_webhook_url is not permitted"):
            Settings(
                _env_file=None,
                notify_enabled=True,
                notify_webhook_url=bad,
                notify_webhook_secret="a-sufficiently-long-secret",
            )


def test_notify_enabled_requires_strong_secret() -> None:
    with pytest.raises(ValidationError, match="notify_webhook_secret is required"):
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_url="https://example.com/hook",
            notify_webhook_secret="short",
        )


def test_notify_enabled_valid_config_passes() -> None:
    s = Settings(
        _env_file=None,
        notify_enabled=True,
        notify_webhook_url="https://hooks.example.com/voxint",
        notify_webhook_secret="a-sufficiently-long-secret",
    )
    assert s.notify_enabled is True
    assert s.notify_webhook_url == "https://hooks.example.com/voxint"


def test_notify_secret_never_leaks_in_sanitized_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # get_settings() sanitizes to SettingsError; the secret value must never
    # appear in the message even when it is the reason for the failure.
    monkeypatch.setenv("NOTIFY_ENABLED", "true")
    monkeypatch.setenv("NOTIFY_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("NOTIFY_WEBHOOK_SECRET", "sup3rs3cr3t")  # 11 chars -> too short
    with pytest.raises(SettingsError) as excinfo:
        get_settings()
    assert "sup3rs3cr3t" not in str(excinfo.value)
    assert "notify_webhook_secret" in str(excinfo.value)


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


def test_names_llm_pass_requires_llm_enabled() -> None:
    # The LLM name pass rides the enhancement endpoint; enabling it with no
    # configured LLM would silently do nothing, so the combination is refused.
    with pytest.raises(ValidationError, match="enrichment_names_llm_enabled"):
        Settings(_env_file=None, enrichment_names_llm_enabled=True)
    s = Settings(_env_file=None, llm_enabled=True, enrichment_names_llm_enabled=True)
    assert s.enrichment_names_llm_enabled is True
    # Offline producer defaults on; the LLM pass defaults off.
    defaults = Settings(_env_file=None)
    assert defaults.enrichment_names_enabled is True
    assert defaults.enrichment_names_llm_enabled is False


def test_names_llm_pass_requires_offline_producer_enabled() -> None:
    # LLM pass on with the offline producer off would be unusable — no CLI
    # invocation could ever run it (the --llm flag rides `enrich names`).
    with pytest.raises(ValidationError, match="enrichment_names_enabled"):
        Settings(
            _env_file=None,
            llm_enabled=True,
            enrichment_names_enabled=False,
            enrichment_names_llm_enabled=True,
        )


def test_web_research_defaults_off_and_independent_of_llm() -> None:
    s = Settings(_env_file=None)
    assert s.voxint_web_research is False
    assert s.web_search_provider == "searxng"
    # Independence both ways: enabling the LLM does not enable retrieval, and
    # enabling retrieval (with a provider URL) does not need the LLM.
    s2 = Settings(_env_file=None, llm_enabled=True)
    assert s2.voxint_web_research is False
    s3 = Settings(
        _env_file=None,
        voxint_web_research=True,
        web_search_base_url="http://searx.lan:8888",
    )
    assert s3.llm_enabled is False


def test_web_research_env_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    s = Settings(_env_file=None)
    assert s.voxint_web_research is True
    assert s.web_search_base_url == "http://searx.lan:8888"


def test_web_research_enabled_requires_base_url() -> None:
    with pytest.raises(ValidationError, match="web_search_base_url"):
        Settings(_env_file=None, voxint_web_research=True)
    # Disabled: an empty base URL is fine — no dead-config refusal.
    Settings(_env_file=None, voxint_web_research=False)


@pytest.mark.parametrize(
    "base_url",
    [
        "searx.lan:8888",  # no scheme
        "ftp://searx.lan",  # wrong scheme
        "http://",  # no host
        "http://user:pass@searx.lan",  # embedded credentials
    ],
)
def test_web_research_base_url_shape_is_validated(base_url: str) -> None:
    with pytest.raises(ValidationError, match="web_search_base_url"):
        Settings(
            _env_file=None, voxint_web_research=True, web_search_base_url=base_url
        )


def test_web_research_api_key_never_in_settings_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A bad enabled-config must not leak the provider credential into the
    # sanitized SettingsError message.
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "")  # invalid while enabled
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "SUPERSECRETPROVIDERKEY")
    with pytest.raises(SettingsError) as exc:
        get_settings()
    assert "SUPERSECRETPROVIDERKEY" not in str(exc.value)


def test_web_search_base_url_bad_port_and_decorations_fail_at_startup() -> None:
    # A lazily-parsed bad port must fail at startup, not as an opaque
    # provider_error on the first search (review finding); query/fragment/
    # whitespace have no place in a bare endpoint.
    for bad in [
        "http://searxng.example:abc",
        "http://searxng.example:99999999",
        "http://searxng.example/search?q=x",
        "http://searxng.example/#frag",
        " http://searxng.example ",
        "http://searxng.example\\path",
    ]:
        with pytest.raises(ValidationError, match="web_search_base_url"):
            Settings(_env_file=None, voxint_web_research=True, web_search_base_url=bad)


# --- Media retention / GC (issue #15) ---


def test_media_retention_defaults_off() -> None:
    s = Settings(_env_file=None)
    assert s.media_retention_enabled is False
    assert s.media_retention_seconds == 2592000  # 30 d
    assert s.gc_sweep_seconds == 3600
    assert s.gc_batch_limit == 500


def test_media_retention_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MEDIA_RETENTION_ENABLED", "true")
    monkeypatch.setenv("MEDIA_RETENTION_SECONDS", "86400")
    monkeypatch.setenv("GC_SWEEP_SECONDS", "600")
    monkeypatch.setenv("GC_BATCH_LIMIT", "50")
    s = Settings(_env_file=None)
    assert s.media_retention_enabled is True
    assert s.media_retention_seconds == 86400
    assert s.gc_sweep_seconds == 600
    assert s.gc_batch_limit == 50


def test_media_retention_seconds_floor() -> None:
    # A sub-hour retention window is rejected — too aggressive to be a config
    # typo we should silently honor.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, media_retention_seconds=3599)
    # The floor itself is accepted.
    assert Settings(_env_file=None, media_retention_seconds=3600).media_retention_seconds == 3600


def test_gc_batch_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gc_batch_limit=0)


def test_gc_sweep_seconds_has_floor() -> None:
    # A zero/negative beat interval would make celery-beat tight-loop the sweep.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, gc_sweep_seconds=0)
    assert Settings(_env_file=None, gc_sweep_seconds=60).gc_sweep_seconds == 60


def test_media_retention_seconds_not_tier_scaled() -> None:
    # Retention is wall-clock policy, not a compute-tier timing budget — a cpu
    # tier must NOT quietly quadruple the operator's retention window.
    from voxint.config import TIER_SCALED_TIMING_FIELDS

    assert "media_retention_seconds" not in TIER_SCALED_TIMING_FIELDS
    assert (
        Settings(_env_file=None, compute_tier="cpu").media_retention_seconds == 2592000
    )
