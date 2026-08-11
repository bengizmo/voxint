"""Typed application settings.

Every endpoint, path, credential, and tunable enters the system here — nothing is
hardcoded elsewhere. Values come from the environment (or an ``.env`` file in dev).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # LLM transcript enhancement (optional; any OpenAI-compatible endpoint)
    llm_enabled: bool = False
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""

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


def get_settings() -> Settings:
    return Settings()
