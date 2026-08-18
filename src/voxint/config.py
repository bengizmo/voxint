"""Typed application settings.

Every endpoint, path, credential, and tunable enters the system here — nothing is
hardcoded elsewhere. Values come from the environment (or an ``.env`` file in dev).
"""

import os
from pathlib import Path
from typing import Annotated, Literal

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

# Headroom every stage lease must keep over the GPU-call timeout it covers:
# after an inference call returns (or times out), the stage still persists its
# results and commits before the lease may expire, or recovery reclaims the
# stage mid-persist and a second worker re-executes it (docs/timeouts-and-leases.md).
GPU_CALL_PERSISTENCE_MARGIN_SECONDS = 600.0

# Per-attempt LLM timeout default, shared with the enrichment job modules'
# legacy-snapshot fallbacks (research_jobs / asset_jobs) so a job snapshot
# missing the key can never drift from the runtime default. 300 s reflects
# measured need: entity-mention extraction on a local ~35B model routinely
# takes 180-300 s per call, and 90 s made the default configuration fail for
# exactly the self-hosted deployments this project targets. Cloud endpoints
# answer in seconds regardless; connection establishment stays on its own
# short cap, so unreachable endpoints still fail fast.
DEFAULT_LLM_TIMEOUT_SECONDS = 300.0

# The CPU tier's scaling factor over the GPU-tier timing defaults. CPU
# inference for these models is roughly 5-20x slower than GPU depending on
# stage and cores; 4x on top of the already-generous GPU defaults (which carry
# their own multi-hour slack) keeps a healthy slow run distinguishable from a
# hung one without waiting a week to reclaim a genuinely dead lease. Explicit
# env values always win over the scaled defaults.
CPU_TIER_TIMEOUT_FACTOR = 4.0

# Timing fields the compute-tier profile scales when they are left at their
# defaults. acquire_* are download-bound, not inference-bound, so the tier
# never touches them. celery_visibility scales with the leases it must cover.
TIER_SCALED_TIMING_FIELDS = (
    "gpu_http_timeout_seconds",
    "stage_lease_seconds",
    "diarize_embed_lease_seconds",
    "celery_visibility_timeout_seconds",
)


class Settings(BaseSettings):
    # hide_input_in_errors keeps offending values out of a ValidationError's
    # printed form: settings inputs include credentials (llm_api_key,
    # voxint_password, database_url, ytdlp_cookies_file), and a bad one must not
    # land in a startup traceback. This only sanitizes str(err); the structured
    # .errors()/.json() still carry the raw input, so get_settings() re-raises a
    # sanitized SettingsError as the real production guarantee.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

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
    # Triage threshold for the review console's low-confidence highlight (#53):
    # a segment whose ASR confidence (exp(avg_logprob), a transformed likelihood
    # — NOT a calibrated probability; see docs/quality-gates.md) is BELOW this is
    # flagged "uncertain, not necessarily wrong". Segments with no stored
    # confidence (NULL) are never flagged. Configurable, but deliberately NOT a
    # UI slider: a non-technical operator mis-setting it would distrust the whole
    # signal. 0.6 is a conservative starting default on the exp(avg_logprob)
    # scale (whisper's own logprob_threshold=-1.0 ⇒ exp≈0.37 marks "failed"; 0.6
    # surfaces the genuinely shaky, not just the discarded); refine against a
    # real-corpus histogram in a follow-up before exposing any tuning UI.
    review_low_confidence_threshold: Ratio = Field(default=0.6)
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
    # Which compute tier the model services run on. A named timing PROFILE,
    # not a hardware switch: "cpu" multiplies the default inference timeouts /
    # stage leases / visibility horizon by CPU_TIER_TIMEOUT_FACTOR so a
    # healthy-but-slow CPU run is never reclaimed as hung mid-stage
    # (docs/timeouts-and-leases.md). Explicitly-set values are never scaled.
    # "rocm", "metal" (bare-metal Apple Silicon services, compose.metal.yaml)
    # and future accelerated tiers keep GPU-class timing. The metal tier's v1
    # CPU whisper measured 0.38-0.45x RT under these budgets (Gate M), so
    # metal intentionally has no factor of its own (docs/gpu-contracts.md).
    compute_tier: Literal["gpu", "cpu", "rocm", "metal"] = "gpu"
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
    llm_timeout_seconds: PositiveSeconds = DEFAULT_LLM_TIMEOUT_SECONDS  # per attempt (read/write)
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

    # Media retention / garbage collection (issue #15). File reclamation only:
    # the GC sweep unlinks the large normalized-audio intermediate
    # (artifacts/{run_id}/normalized.wav) for OLD TERMINAL runs and stamps the
    # audio_artifacts row (reclaimed_at/reclaimed_bytes). Source media, the
    # transcript, diarization, and the adjudication ledger are ALWAYS kept, so a
    # reclaimed run stays re-processable from source. OFF by default — no bytes
    # are ever reclaimed until an operator opts in.
    media_retention_enabled: bool = False
    # Age (since a run was last modified — reaching a terminal state, an
    # operator note edit, OR a review claim/release, all of which bump
    # updated_at; enrichment and adjudication decisions write to separate tables
    # and do NOT) after which a terminal run's normalized-audio intermediate
    # becomes eligible for reclamation. Keying on updated_at is intentionally
    # conservative: a run under active review keeps its clock reset, so the
    # intermediate is only ever reclaimed too LATE, never too early. Only
    # consulted when media_retention_enabled. Floor 1 h; the 30-day default is a
    # conservative starting point, tune via env.
    # NOTE: deliberately NOT tier-scaled (absent from TIER_SCALED_TIMING_FIELDS)
    # — retention is wall-clock policy, not a compute-tier timing budget.
    media_retention_seconds: int = Field(default=2592000, ge=3600)  # 30 d
    # How often the GC sweep runs (only registered on beat when retention is on).
    gc_sweep_seconds: int = Field(default=3600, ge=60)
    # Rows reclaimed per sweep, oldest-first — bounds one sweep's work and the
    # per-sweep IO burst. A backlog drains at gc_batch_limit per gc_sweep_seconds.
    gc_batch_limit: int = Field(default=500, ge=1)

    # Run notifications / webhooks (issue #12). A single signed webhook POST on
    # notifiable transitions (awaiting_adjudication / completed / failed),
    # delivered at-least-once via a transactional outbox + beat sweep. OFF by
    # default — nothing is emitted or delivered until an operator opts in, and
    # enabling later never back-fills runs that finished while it was off.
    notify_enabled: bool = False
    # Admin-configured outbound endpoint. Validated as an absolute, credential-free
    # public http/https URL (parse_http_url) only when notify_enabled — the same
    # egress posture as URL ingestion. A signed POST is sent here per arrival.
    notify_webhook_url: str = ""
    # Keys the HMAC-SHA256 signature. Wire format: X-Voxint-Signature =
    # "sha256=" + hex(hmac(secret, timestamp + "." + body)). Treated as a secret
    # everywhere: redacted from errors/logs. Required, and >= 16 chars, when
    # notify_enabled.
    notify_webhook_secret: str = ""
    # How often the delivery sweep runs (only registered on beat when enabled).
    notify_sweep_seconds: int = Field(default=30, ge=5)
    # Attempts before a delivery row is marked dead (capped exponential backoff
    # with jitter between attempts).
    notify_max_attempts: int = Field(default=8, ge=1)
    # Rows claimed per sweep, oldest-first — bounds one sweep's work.
    notify_batch_limit: int = Field(default=50, ge=1)
    # How long a claimed (in_flight) row is leased to a sweep before another may
    # reclaim it — must exceed one POST's worst-case wall-time (timeout below).
    notify_lease_seconds: int = Field(default=60, gt=0)
    # Per-POST connect+read timeout. Kept well under the lease.
    notify_timeout_seconds: float = Field(default=10.0, gt=0)
    # Initial hold on a FAILED arrival before its first delivery attempt, so a
    # synchronous requeue (recovery/retry) can settle and the row be suppressed
    # rather than sent as a misleading "failed". Applies to FAILED only.
    notify_failed_initial_delay_seconds: int = Field(default=15, ge=0)
    # Retry backoff between failed delivery attempts: base * 2^(attempt-1),
    # capped, plus jitter (mirrors the stage-retry idiom, but a distinct knob so
    # webhook cadence is not coupled to pipeline-stage retry cadence).
    notify_backoff_base_seconds: float = Field(default=10.0, gt=0)
    notify_backoff_max_seconds: float = Field(default=600.0, gt=0)
    # Age (seconds) after which a settled delivery row (delivered/suppressed) is
    # purged by the sweep, bounding table growth. dead rows are kept until an
    # operator acts on them; only the fully-resolved, no-longer-actionable rows
    # are reaped. Floor 3600 (1 h).
    notify_retention_seconds: int = Field(default=604800, ge=3600)

    # Watch-folder ingest (issue #60). A beat sweep auto-submits new media that
    # appears in the operator's registered folders (app_settings.media_folders),
    # skipping files already known (a MediaItem already claims the source_path).
    # OFF by default: this env value is the INSTALLATION default, overridable at
    # runtime per-installation by the nullable app_settings.watch_folder_enabled
    # column (NULL there = inherit this). The beat entry is registered
    # unconditionally (like the recovery sweep) and the task re-checks the
    # EFFECTIVE gate each run, so a UI toggle applies with no restart and a
    # disabled installation only pays one DB read per sweep.
    watch_folder_enabled: bool = False
    # How often the watch sweep re-walks the registered folders. Wall-clock
    # pickup latency for a dropped file, not a compute-tier budget — deliberately
    # NOT tier-scaled (absent from TIER_SCALED_TIMING_FIELDS). Floor 30 s.
    watch_folder_sweep_seconds: int = Field(default=300, ge=30)
    # Quiescence age a candidate file must reach (max of its mtime/ctime) before
    # the sweep submits it, so a file still being copied into the folder is not
    # ingested mid-write. A too-fresh file is left for a later sweep. mtime alone
    # is only a heuristic (some copy tools preserve timestamps); the reliable
    # producer workflow is an atomic rename into the watched folder. 0 disables
    # the wait (accept files immediately).
    watch_folder_settle_seconds: int = Field(default=60, ge=0)

    # Redis redelivery horizon for acks-late tasks; must exceed the longest
    # possible run_pipeline execution — one task runs all SIX stages back to
    # back, so the horizon has to clear the sum of every stage lease. With
    # default leases that sum is ~39 h (acquire 3 h + prepare/transcribe/
    # enhance_match/finalize 4x6 h + diarize_embed 12 h). 48 h keeps headroom;
    # _celery_visibility_covers_all_leases enforces the floor if leases change.
    celery_visibility_timeout_seconds: int = 172800  # 48 h

    # Domain pack (defaults to the bundled generic pack when unset). This is the
    # DEFAULT pack: used for a run whose watched folder has no per-folder mapping
    # (issue #11) and for legacy runs created before per-run selection existed.
    domain_pack_path: Path | None = None
    # Optional directory whose direct child folders are each a named domain pack
    # (a folder with a manifest.yaml). Enables selecting a pack BY NAME per
    # watched folder (app_settings.folder_domain_packs). Unset ⇒ only the bundled
    # generic pack and the configured default (domain_pack_path) are resolvable.
    domain_packs_dir: Path | None = None

    # Name-candidate enrichment (#38): the offline producer mines speaker-name
    # suggestions from stored metadata + transcript text. Dependency-free
    # (regex over rows already in the DB), so it defaults on; gates the CLI
    # command and the console trigger. The optional LLM pass is strictly
    # additive and requires the enhancement LLM to be configured — the offline
    # path must stay useful fully offline.
    enrichment_names_enabled: bool = True
    enrichment_names_llm_enabled: bool = False

    # Web research retrieval (#39): pluggable web_search + hardened read_url.
    # OFF by default, and deliberately INDEPENDENT of llm_enabled — configuring
    # an LLM must never imply outbound egress, and enabling retrieval must not
    # require an LLM (no cross-validator, on purpose). When off, the research
    # entrypoints return structured "disabled" outcomes without touching DNS or
    # the network. The provider base URL is operator-configured egress (a LAN
    # SearxNG is the expected setup) — the same trust class as llm_base_url,
    # NOT subject to the public-address policy; every result URL a provider
    # returns IS, via the shared netcheck string gate and read_url's per-hop
    # resolved-address revalidation.
    voxint_web_research: bool = False
    web_search_provider: Literal["searxng"] = "searxng"
    web_search_base_url: str = ""
    # Credential for future key-bearing providers; unused by searxng. Treated
    # as a secret everywhere (SettingsError sanitization, log redaction).
    web_search_api_key: str = ""
    web_search_max_results: int = Field(default=8, ge=1, le=20)
    web_search_timeout_seconds: PositiveSeconds = 20.0
    # read_url caps. Byte cap counts response body bytes as received (identity
    # encoding is REQUIRED — compressed responses are refused outright, which
    # removes the decompression-bomb class instead of bounding it).
    web_read_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    web_read_max_redirects: int = Field(default=5, ge=0)
    # Per-hop httpx timeout; the whole call additionally runs under the total
    # wall-clock budget below (DNS resolution is the one step a deadline cannot
    # hard-interrupt — see docs/architecture.md).
    web_read_timeout_seconds: PositiveSeconds = 30.0
    web_read_total_seconds: PositiveSeconds = 60.0
    web_read_max_text_chars: int = Field(default=60_000, ge=1)

    # Web-research speaker profile enrichment (#40): the operator-initiated
    # `web_researcher` producer — an LLM tool loop over web_search/read_url plus
    # a read-only roster lookup, writing reviewable drafts. OFF by default and
    # requires BOTH prerequisites (validated below): retrieval provides the
    # egress capability, the enhancement LLM provides the model. The budgets are
    # rendered in the console's start preview and are deliberately NOT
    # operator-raisable per job.
    enrichment_web_research_enabled: bool = False
    # Retrieval quotas handed to ResearchBudget for one job.
    research_max_searches: int = Field(default=3, ge=1, le=10)
    research_max_reads: int = Field(default=5, ge=1, le=20)
    # Tool-loop bounds enforced by the orchestrator (a "round" is one model
    # reply; each reply may carry a few actions). After the final round the
    # model gets exactly one tools-disabled conclude request.
    research_max_rounds: int = Field(default=5, ge=1, le=10)
    research_max_actions_per_round: int = Field(default=3, ge=1, le=5)
    # Wall-clock deadline for the whole job (LLM rounds + retrieval), enforced
    # via ResearchBudget's monotonic deadline and checked between rounds. The
    # floor keeps a fat-fingered near-zero value from producing a zero-work
    # job the moment it starts.
    research_deadline_seconds: float = Field(default=300.0, ge=30.0, allow_inf_nan=False)

    # Draft triage (#42): registrable domains the operator counts as
    # authoritative sources, feeding the PROFILE drafts' `source_authority`
    # triage component (fraction of a draft's distinct evidence domains on this
    # list). Comma/space-separated bare domains (`example.org, gov.uk`); a
    # subdomain matches its registrable domain. Deployments differ on what is
    # authoritative, so there is deliberately no built-in list — empty (the
    # default) means `source_authority` reads 0.0 for every draft, leaving
    # triage ordering to the other signals. A plain string (not a list): the
    # app parses it into a domain set; pydantic-settings would JSON-parse a
    # collection-typed env var. This is the fallback: Settings → Sources & research
    # (issue #76) edits it live via resolve_effective_source_authority_domains
    # (DB-row-wins-over-env, no restart); this env value applies when the row is blank.
    source_authority_domains: str = ""

    # Run-level enrichment assets (#41): on-demand LLM-generated summary,
    # topics, and entity mentions per run — three independently versioned,
    # independently failing assets. OFF by default; requires the enhancement
    # LLM (validated below) and nothing else — generation reads only rows
    # already in the DB, no egress beyond the configured LLM endpoint.
    enrichment_run_assets_enabled: bool = False
    # Opt-in post-finalize step: enqueue the three asset jobs when a pipeline
    # run completes (kinds whose current asset already matches the source are
    # skipped). Best-effort — a broker outage defers, never fails the run.
    enrichment_run_assets_autogenerate: bool = False
    # Character bound on the rendered source document handed to the model.
    # Longer transcripts are head+tail truncated (recorded in the asset's
    # config snapshot); the staleness hash always covers the full source.
    run_assets_max_input_chars: int = Field(default=48_000, ge=1_000)

    @model_validator(mode="after")
    def _apply_compute_tier_profile(self) -> "Settings":
        # Defined FIRST so the scaled values are what every later invariant
        # validator (LLM budget, visibility floor, GPU-timeout margin) checks.
        # Only fields still at their class default are scaled — an explicit env
        # value is an operator decision the profile must not override
        # (model_fields_set tracks exactly which fields were provided).
        if self.compute_tier == "cpu":
            for name in TIER_SCALED_TIMING_FIELDS:
                if name not in self.model_fields_set:
                    scaled = type(getattr(self, name))(
                        getattr(self, name) * CPU_TIER_TIMEOUT_FACTOR
                    )
                    object.__setattr__(self, name, scaled)
        return self

    @model_validator(mode="after")
    def _gpu_timeout_fits_stage_leases(self) -> "Settings":
        # Every stage lease must outlast the longest single inference call it
        # covers PLUS the persistence tail, or recovery reclaims a healthy
        # stage mid-persist and a second worker re-executes it (duplicate
        # execution). transcribe/prepare-class stages sit under
        # stage_lease_seconds; diarize_embed makes one diarization call plus N
        # embedding batches, so its (dedicated, larger) lease gets the same
        # floor against a single call — the batch-sum sizing is operational
        # guidance in docs/timeouts-and-leases.md, not statically checkable.
        floor = self.gpu_http_timeout_seconds + GPU_CALL_PERSISTENCE_MARGIN_SECONDS
        for lease_name in ("stage_lease_seconds", "diarize_embed_lease_seconds"):
            lease = getattr(self, lease_name)
            if lease <= floor:
                raise ValueError(
                    f"{lease_name} ({lease}) must exceed gpu_http_timeout_seconds"
                    f" plus the persistence margin ({floor})"
                )
        return self

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
    def _effective_flag_invariants(self) -> "Settings":
        # The five cross-flag invariants (names.llm ⇒ names+llm; web-research
        # producer ⇒ retrieval+llm; run-assets autogenerate ⇒ run-assets ⇒ llm;
        # retrieval ⇒ valid provider URL) live in ONE place —
        # app_settings.validate_effective_flags — so this env-time check and every
        # runtime settings form (which validates the effective, row-over-env
        # combination — issue #74) can never drift. A DB override that enables a
        # dependent flag while its prerequisite resolves false is caught there with
        # the same message. Imported locally because app_settings imports Settings,
        # so a module-level import would cycle. Raising the first message preserves
        # the prior boot-strict behavior and the exact operator-facing copy.
        from voxint.app_settings import EffectiveFlags, validate_effective_flags

        errors = validate_effective_flags(
            EffectiveFlags(
                llm_enabled=self.llm_enabled,
                enrichment_names_enabled=self.enrichment_names_enabled,
                enrichment_names_llm_enabled=self.enrichment_names_llm_enabled,
                enrichment_run_assets_enabled=self.enrichment_run_assets_enabled,
                enrichment_run_assets_autogenerate=self.enrichment_run_assets_autogenerate,
                voxint_web_research=self.voxint_web_research,
                enrichment_web_research_enabled=self.enrichment_web_research_enabled,
                web_search_base_url=self.web_search_base_url,
            )
        )
        if errors:
            raise ValueError(errors[0])
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
            raise ValueError(f"grounding gates must be at least as strict as proposal gates: {lax}")
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
    def _notify_config_complete_when_enabled(self) -> "Settings":
        # Enabling webhooks without a usable endpoint would silently drop every
        # notification (or, worse, accumulate undeliverable outbox rows), so fail
        # fast at startup instead. The URL is validated by the same string-level
        # gate as URL ingestion (absolute, credential-free, public http/https);
        # parse_http_url's errors never echo the URL, and the secret is never
        # named in any message — neither is a value we may leak. Only enforced
        # when enabled, keeping the zero-config default path free of ceremony.
        if not self.notify_enabled:
            return self
        from voxint.media.netcheck import UrlPolicyError, parse_http_url

        if not self.notify_webhook_url:
            raise ValueError("notify_webhook_url is required when notify_enabled")
        try:
            parse_http_url(self.notify_webhook_url)
        except UrlPolicyError as exc:
            # exc carries no URL, only the policy reason — safe to surface.
            raise ValueError(f"notify_webhook_url is not permitted: {exc}") from exc
        if len(self.notify_webhook_secret) < 16:
            raise ValueError(
                "notify_webhook_secret is required and must be at least 16 "
                "characters when notify_enabled"
            )
        # The claim lease must outlast one delivery attempt, or a sweep would keep
        # reclaiming rows whose first POST is still in flight — duplicate sends.
        # (Lease-guarded outcome writes keep that SAFE, just wasteful, so this is
        # a sanity floor rather than an exact worst-case bound.)
        if self.notify_lease_seconds < self.notify_timeout_seconds:
            raise ValueError(
                "notify_lease_seconds must be >= notify_timeout_seconds"
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
