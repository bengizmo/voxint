"""Typed application settings.

Every endpoint, path, credential, and tunable enters the system here — nothing is
hardcoded elsewhere. Values come from the environment (or an ``.env`` file in dev).
"""

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# NaN silently disables a threshold (every comparison is False), so all gate
# floats are finite and range-constrained at the settings boundary.
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Cosine = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]

# Headroom the ACQUIRE lease must keep over acquire_timeout_seconds. After the
# download timeout fires, the yt-dlp process group still needs its SIGTERM->
# SIGKILL grace (media.ytdlp._KILL_GRACE_SECONDS, 5 s) and — on a successful
# download — the produced file must be hashed and atomically published before
# the stage commits. If that tail could outlast the lease, recovery might
# reclaim ACQUIRE mid-publish. 300 s comfortably covers hashing a multi-GiB
# file plus the kill grace and the commit.
ACQUIRE_CLEANUP_MARGIN_SECONDS = 300.0


class Settings(BaseSettings):
    # hide_input_in_errors keeps offending values out of a ValidationError's
    # printed form: settings inputs include credentials (llm_api_key,
    # voxint_password, database_url, ytdlp_cookies_file), and a bad one must not
    # land in a startup traceback. This only sanitizes str(err); the structured
    # .errors()/.json() still carry the raw input, so get_settings() re-raises a
    # sanitized SettingsError as the real production guarantee.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    # Core services
    database_url: str = "postgresql+psycopg://voxint:voxint@localhost:5432/voxint"
    redis_url: str = "redis://localhost:6379/0"

    # Media
    media_root: Path = Path("/data/media")

    # API / review UI
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    voxint_user: str = "admin"
    voxint_password: str = "change-me"
    # Secret keying the stateless CSRF token on the mutation forms (POST /submit,
    # /fetch, /runs/{id}/requeue). Independent of voxint_password ON PURPOSE: a
    # human-memorable password would turn every rendered token into a fast offline
    # password-verification oracle. Empty ⇒ create_app() mints a random per-process
    # secret (fine for a single-operator localhost console, but open forms break on
    # restart / across workers); set a persistent random value
    # (`python -c "import secrets; print(secrets.token_urlsafe(32))"`) to make
    # tokens stable across restarts and multiple workers.
    csrf_secret: str = ""
    # How long a reviewer holds a run before the slot self-releases. Long
    # enough for one careful listen-through; short enough that a closed tab
    # doesn't dam the queue.
    review_claim_ttl_seconds: int = Field(default=1800, ge=60)
    # Hard bound on the ffprobe gate in the media-serving request path.
    media_probe_timeout_seconds: PositiveSeconds = 30.0
    # Transcript preview length per label in the workbench.
    review_preview_segments: int = Field(default=8, ge=1)
    # Bounded page size for the /runs execution-history browser (keyset paged).
    runs_page_size: int = Field(default=50, ge=1, le=500)
    # Hard ceiling on a browser upload (POST /submit), enforced authoritatively
    # while streaming — an oversized Content-Length is rejected early, but the
    # stream copy stops and unlinks its temp the moment it crosses this bound, so
    # a lying header can never write past it. 5 GB fits long-form podcast media.
    upload_max_bytes: int = Field(default=5 * 1024**3, gt=0)

    # yt-dlp URL acquisition (the ACQUIRE stage; only the worker downloads).
    # URL ingestion is an authenticated admin egress capability, on by default.
    # This flag is enforced at the *submission* surface (the fetch route/CLI
    # refuse when it is off, wired in a later slice); the worker's ACQUIRE stage
    # never consults it, so a run that already exists stays processable.
    ytdlp_enabled: bool = True
    # Authoritative ceiling on a downloaded file: passed to yt-dlp as an early
    # --max-filesize hint AND re-checked on the produced file before it is
    # published, so a hint yt-dlp fails to honour cannot write an oversized source.
    ytdlp_max_bytes: int = Field(default=5 * 1024**3, gt=0)
    # Hard wall-clock bound on the whole yt-dlp subprocess. socket_timeout alone
    # cannot cap a download that keeps trickling just under the socket deadline,
    # so the process group is killed on expiry. 2 h: the ceiling is 5 GiB and a
    # slow home uplink (~1 MB/s) needs ~90 min for a full-ceiling download, so a
    # 1 h bound would spuriously fail large but legitimate sources. Must sit below
    # acquire_lease_seconds by ACQUIRE_CLEANUP_MARGIN_SECONDS (validated below).
    acquire_timeout_seconds: PositiveSeconds = 7200.0  # 2 h
    # How long a worker may hold the ACQUIRE stage before recovery reclaims it.
    # ACQUIRE's analogue of stage_lease_seconds; it must outlast the download
    # timeout plus the kill/hash/publish tail (ACQUIRE_CLEANUP_MARGIN_SECONDS).
    acquire_lease_seconds: int = Field(default=10800, gt=0)  # 3 h
    # Per-socket connect/read timeout handed to yt-dlp (--socket-timeout).
    ytdlp_socket_timeout_seconds: PositiveSeconds = 30.0
    # Optional outbound proxy for yt-dlp (slice 6g). --proxy is passed ALWAYS: with
    # this value, or an empty string that yt-dlp reads as explicit DIRECT — so an
    # ambient HTTP(S)_PROXY/ALL_PROXY in the worker env can never silently reroute
    # egress (set this to use a proxy on purpose). A non-empty value is a credential:
    # scrubbed verbatim from errors (redact(extra_secrets=...)) and by --proxy flag
    # redaction.
    ytdlp_proxy: str = ""
    # Optional cookies file (Netscape format) for yt-dlp, wired to --cookies when
    # set (slice 6g). Validated below as a readable regular file so a typo fails at
    # startup, not mid-download. Treat its path and contents as a credential: the
    # path is scrubbed verbatim from errors (redact(extra_secrets=...)) and never
    # surfaced by the config validators.
    ytdlp_cookies_file: Path | None = None

    # GPU model services
    asr_url: str = "http://localhost:8022"
    diarizer_url: str = "http://localhost:8024"
    embedder_url: str = "http://localhost:8021"
    # One request = one synchronous inference run over media that can be hours
    # long. Must stay comfortably BELOW stage_lease_seconds: the lease covers a
    # whole stage (possibly several sequential calls) plus persistence margin.
    gpu_http_timeout_seconds: float = 14400.0  # 4 h
    # The first-run wizard's readiness probe of the three services' /healthz
    # (voxint.api.health_probe). Deliberately short and separate from the
    # inference timeout above: reusing the 4-hour budget would make the wizard's
    # "check services" step appear to hang. Advisory only — a probe never blocks
    # a run.
    health_probe_timeout_seconds: PositiveSeconds = 3.0

    # First-run wizard "scan for existing media" (step 2). Bounds the optional
    # walk over the registered media folders: at most setup_scan_max_files net-new
    # candidates are surfaced, and the walk stops after inspecting
    # setup_scan_max_entries directory entries so a deep/wide tree can never make
    # the step hang or auto-queue an unbounded number of runs. Advisory
    # convenience — the operator can always submit media individually.
    setup_scan_max_files: int = Field(default=500, ge=1)
    setup_scan_max_entries: int = Field(default=20000, ge=1)

    # Media normalization (prepare stage)
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    # LLM transcript enhancement (optional; any OpenAI-compatible endpoint).
    # Enhancement is best-effort: failures degrade to NULL enhanced_text and
    # never fail the run, so these budgets bound wasted time, not correctness.
    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_timeout_seconds: PositiveSeconds = 90.0  # per attempt (read/write)
    llm_attempts_per_batch: int = Field(default=2, ge=1)
    llm_batch_max_segments: int = Field(default=32, ge=1)
    llm_batch_max_chars: int = Field(default=12000, ge=1)
    # Hard ceiling on total LLM wall-time per run; must sit well below the
    # enhance_match stage lease (stage_lease_seconds) so matching + persistence
    # always fit inside the lease even when the endpoint is slow.
    llm_run_budget_seconds: PositiveSeconds = 14400.0  # 4 h
    # Consecutive failed batches before the stage stops calling the LLM for
    # this run (remaining segments stay NULL; matching still runs).
    llm_consecutive_failure_limit: int = Field(default=3, ge=1)

    # Speaker matching gates (see docs/quality-gates.md). Similarities are raw
    # cosine in [-1, 1]; stored confidence is (cosine + 1) / 2 — a transformed
    # similarity, NOT a calibrated probability. Defaults are conservative
    # titanet-large starting values; calibrate against adjudicated pairs.
    match_max_overlap_ratio: Ratio = 0.20  # turn eligibility
    match_turn_weight_cap_seconds: PositiveSeconds = 10.0
    match_min_turns: int = Field(default=2, ge=1)
    match_min_seconds: PositiveSeconds = 6.0
    match_min_cosine: Cosine = 0.60
    match_min_margin: Ratio = 0.05  # top-1 vs top-2 separation
    match_min_vote_agreement: Ratio = 0.60
    grounded_min_turns: int = Field(default=3, ge=1)
    grounded_min_seconds: PositiveSeconds = 10.0
    grounded_min_cosine: Cosine = 0.70
    grounded_min_margin: Ratio = 0.08
    grounded_min_vote_agreement: Ratio = 0.67

    # Pipeline
    # How long a worker may hold a stage before recovery may reclaim it.
    # Must exceed the longest realistic single-stage duration — for
    # diarize_embed that is a diarization call PLUS the embedding batches, so
    # keep it well above gpu_http_timeout_seconds.
    stage_lease_seconds: int = 21600  # 6 h
    # diarize_embed makes one diarization call plus N sequential embedding
    # batches; its lease must cover all of them, not one HTTP timeout.
    diarize_embed_lease_seconds: int = 43200  # 12 h
    # Per-stage transient-failure budget (attempts counted from the persisted
    # stage_runs ledger, so restarts cannot reset it).
    stage_max_attempts: int = 5
    # Base for exponential retry backoff: base * 2^(attempt-1), capped.
    retry_backoff_base_seconds: float = 30.0
    retry_backoff_max_seconds: float = 1800.0
    # How often beat sweeps expired leases and stranded QUEUED runs.
    recovery_sweep_seconds: int = 300
    # A QUEUED run untouched this long has no live task on the broker
    # (covers pending retry countdowns; keep it above retry_backoff_max_seconds).
    queued_run_stale_seconds: int = 3600
    # Redis redelivery horizon for acks-late tasks; must exceed the longest
    # possible run_pipeline execution — one task runs all SIX stages back to
    # back, so the horizon has to clear the sum of every stage lease. With
    # default leases that sum is ~39 h (acquire 3 h + prepare/transcribe/
    # enhance_match/finalize 4x6 h + diarize_embed 12 h). 48 h keeps headroom;
    # _celery_visibility_covers_all_leases enforces the floor if leases change.
    celery_visibility_timeout_seconds: int = 172800  # 48 h

    # Domain pack (defaults to the bundled generic pack when unset)
    domain_pack_path: Path | None = None

    @model_validator(mode="after")
    def _llm_budget_fits_stage_lease(self) -> "Settings":
        # The budget is checked before each attempt, so one in-flight batch can
        # overrun by up to attempts x timeout. Budget plus that worst case must
        # stay below the lease or recovery could reclaim enhance_match
        # mid-flight — catch the misconfiguration at startup, not in production.
        # The comparison itself lives in the module-level
        # ``llm_budget_fits_stage_lease`` so the first-run wizard's runtime enable
        # step (and the worker's fail-closed guard) share ONE invariant with this
        # env-time check and can never drift. This validator is still gated on
        # ``llm_enabled`` because the wizard can enable the LLM at runtime with the
        # env flag off — that path is covered by the wizard/worker guards, not here.
        if self.llm_enabled and not llm_budget_fits_stage_lease(self):
            worst_case = self.llm_run_budget_seconds + (
                self.llm_attempts_per_batch * self.llm_timeout_seconds
            )
            raise ValueError(
                f"llm_run_budget_seconds + attempts x timeout ({worst_case})"
                f" must be below stage_lease_seconds ({self.stage_lease_seconds})"
            )
        return self

    @model_validator(mode="after")
    def _no_default_credentials_off_loopback(self) -> "Settings":
        # Loopback + default credentials is a fine dev setup; exposing the
        # review UI beyond loopback with the shipped (or an empty) password is
        # not — refuse at startup rather than serve an effectively
        # unauthenticated console. `voxint serve` and the container entrypoint
        # bind from these settings, so this check sees the real bind address.
        if self.api_host not in ("127.0.0.1", "::1", "localhost") and (
            self.voxint_password in ("change-me", "")
        ):
            raise ValueError(
                "voxint_password must be set to a real value before binding"
                f" the API to non-loopback host {self.api_host!r}"
            )
        return self

    @model_validator(mode="after")
    def _grounding_gates_at_least_as_strict(self) -> "Settings":
        lax = [
            name
            for name, proposal, grounded in (
                ("min_turns", self.match_min_turns, self.grounded_min_turns),
                ("min_seconds", self.match_min_seconds, self.grounded_min_seconds),
                ("min_cosine", self.match_min_cosine, self.grounded_min_cosine),
                ("min_margin", self.match_min_margin, self.grounded_min_margin),
                (
                    "min_vote_agreement",
                    self.match_min_vote_agreement,
                    self.grounded_min_vote_agreement,
                ),
            )
            if grounded < proposal
        ]
        if lax:
            raise ValueError(
                f"grounding gates must be at least as strict as proposal gates: {lax}"
            )
        return self

    @model_validator(mode="after")
    def _acquire_timeout_fits_lease(self) -> "Settings":
        # A stale ("zombie") ACQUIRE attempt whose lease expired while a fresh
        # attempt runs is the failure the slice-6c os.link publish closes on the
        # data side; this closes it on the timing side. The download is bounded
        # by acquire_timeout_seconds, after which the process group is killed and
        # — on success — the file is hashed and atomically published before the
        # stage commits. If timeout plus that tail could reach the lease,
        # recovery might reclaim ACQUIRE mid-publish. Refuse the overlap at
        # startup so it can't be opened by misconfiguration.
        worst_case = self.acquire_timeout_seconds + ACQUIRE_CLEANUP_MARGIN_SECONDS
        if worst_case >= self.acquire_lease_seconds:
            raise ValueError(
                f"acquire_timeout_seconds + cleanup margin ({worst_case}) must be"
                f" below acquire_lease_seconds ({self.acquire_lease_seconds})"
            )
        return self

    @model_validator(mode="after")
    def _ytdlp_cookies_file_readable(self) -> "Settings":
        # If a cookies file is configured it must exist as a readable regular
        # file now, so a typo fails fast at startup instead of mid-download.
        # Never put the path (or its contents) in the error — a cookies file is
        # a credential.
        cookies = self.ytdlp_cookies_file
        if cookies is not None and not (cookies.is_file() and os.access(cookies, os.R_OK)):
            raise ValueError("ytdlp_cookies_file must point to a readable regular file")
        return self

    @model_validator(mode="after")
    def _csrf_secret_strong_enough(self) -> "Settings":
        # An empty csrf_secret is the "mint a random per-process secret" signal
        # (create_app handles it); a SET secret keys an HMAC, so a 1-char
        # CSRF_SECRET would be trivially brute-forceable. Guard the footgun without
        # forcing config on the zero-config localhost path.
        if self.csrf_secret and len(self.csrf_secret) < 16:
            raise ValueError(
                "csrf_secret must be empty (auto per-process) or at least 16 characters"
            )
        return self

    @model_validator(mode="after")
    def _celery_visibility_covers_all_leases(self) -> "Settings":
        # acks-late redelivery must not fire while a run is still legitimately
        # working. run_pipeline is a single task that advances through every
        # stage, so the visibility horizon has to outlast all stage leases held
        # back to back. Mirror default_stage_leases() (engine.py) inline —
        # importing it would cycle config <- engine <- config. ACQUIRE and
        # DIARIZE_EMBED carry dedicated leases; PREPARE, TRANSCRIBE,
        # ENHANCE_MATCH and FINALIZE (four stages) each use stage_lease_seconds.
        all_stage_leases = (
            self.acquire_lease_seconds
            + self.diarize_embed_lease_seconds
            + 4 * self.stage_lease_seconds
        )
        if self.celery_visibility_timeout_seconds < all_stage_leases:
            raise ValueError(
                "celery_visibility_timeout_seconds"
                f" ({self.celery_visibility_timeout_seconds}) must be at least the sum"
                f" of all six stage leases ({all_stage_leases})"
            )
        return self


def llm_budget_fits_stage_lease(settings: Settings) -> bool:
    """True iff the LLM run budget plus one worst-case in-flight overrun stays
    below the enhance_match stage lease.

    The budget is checked before each batch, so one in-flight attempt can overrun
    by up to ``attempts_per_batch x timeout``; that worst case plus the budget must
    stay strictly below ``stage_lease_seconds`` or the recovery sweep could reclaim
    enhance_match mid-flight. This is the SINGLE source of that comparison, shared
    by the env-time ``Settings`` validator, the first-run wizard's LLM-enable step
    (which refuses to persist ``llm_enabled=True`` when it returns False), and the
    worker's per-run fail-closed guard (``apply_run_preferences``) — so a runtime
    enable can never bypass an invariant the env path enforces.
    """
    worst_case = settings.llm_run_budget_seconds + (
        settings.llm_attempts_per_batch * settings.llm_timeout_seconds
    )
    return worst_case < settings.stage_lease_seconds


class SettingsError(Exception):
    """Raised when settings fail to load or validate.

    Its message carries only each failing field's location and constraint
    message — never the offending value. That matters because a Settings input
    can be a credential (llm_api_key, voxint_password, database_url, and the
    ytdlp_cookies_file path), and pydantic otherwise attaches the whole input to
    every error via ``.errors()[*]['input']`` and ``.json()`` (which
    ``hide_input_in_errors`` does not strip).
    """


def _sanitized_settings_error(exc: ValidationError) -> str:
    # Build the message from locations + messages only. Never read 'input' — for
    # a model-level validator that field is the entire settings dict, credentials
    # included. pydantic's own constraint messages describe the rule, not the
    # value, so they are safe to surface.
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(x) for x in err["loc"]) or "settings"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid settings — " + "; ".join(parts)


def get_settings() -> Settings:
    # The single production construction point (CLI, worker boot, API, engine all
    # route through here), so sanitizing the failure here covers every entrypoint.
    # `from None` drops the original ValidationError so its input can't ride along
    # in the traceback.
    try:
        return Settings()
    except ValidationError as exc:
        raise SettingsError(_sanitized_settings_error(exc)) from None
