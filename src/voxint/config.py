"""Typed application settings.

Every endpoint, path, credential, and tunable enters the system here — nothing is
hardcoded elsewhere. Values come from the environment (or an ``.env`` file in dev).
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# NaN silently disables a threshold (every comparison is False), so all gate
# floats are finite and range-constrained at the settings boundary.
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Cosine = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    # yt-dlp URL acquisition (the ACQUIRE stage; only the worker downloads). These
    # are the minimal knobs the slice-6c downloader consumes; the ACQUIRE lease,
    # the "acquire_timeout + margin < lease" validator, the six-stage visibility
    # recompute, and proxy/cookies/enabled land in slice 6d.
    # Authoritative ceiling on a downloaded file: passed to yt-dlp as an early
    # --max-filesize hint AND re-checked on the produced file before it is
    # published, so a hint yt-dlp fails to honour cannot write an oversized source.
    ytdlp_max_bytes: int = Field(default=5 * 1024**3, gt=0)
    # Hard wall-clock bound on the whole yt-dlp subprocess. socket_timeout alone
    # cannot cap a download that keeps trickling just under the socket deadline,
    # so the process group is killed on expiry. Must sit below the ACQUIRE stage
    # lease (that ordering validator is wired in 6d with acquire_lease_seconds).
    acquire_timeout_seconds: PositiveSeconds = 3600.0  # 1 h
    # Per-socket connect/read timeout handed to yt-dlp (--socket-timeout).
    ytdlp_socket_timeout_seconds: PositiveSeconds = 30.0

    # GPU model services
    asr_url: str = "http://localhost:8022"
    diarizer_url: str = "http://localhost:8024"
    embedder_url: str = "http://localhost:8021"
    # One request = one synchronous inference run over media that can be hours
    # long. Must stay comfortably BELOW stage_lease_seconds: the lease covers a
    # whole stage (possibly several sequential calls) plus persistence margin.
    gpu_http_timeout_seconds: float = 14400.0  # 4 h

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
    # possible run_pipeline execution (all stages back to back).
    celery_visibility_timeout_seconds: int = 108000  # 30 h ~= 5 stages x 6 h lease

    # Domain pack (defaults to the bundled generic pack when unset)
    domain_pack_path: Path | None = None

    @model_validator(mode="after")
    def _llm_budget_fits_stage_lease(self) -> "Settings":
        # The budget is checked before each attempt, so one in-flight batch can
        # overrun by up to attempts x timeout. Budget plus that worst case must
        # stay below the lease or recovery could reclaim enhance_match
        # mid-flight — catch the misconfiguration at startup, not in production.
        if self.llm_enabled:
            worst_case = self.llm_run_budget_seconds + (
                self.llm_attempts_per_batch * self.llm_timeout_seconds
            )
            if worst_case >= self.stage_lease_seconds:
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


def get_settings() -> Settings:
    return Settings()
