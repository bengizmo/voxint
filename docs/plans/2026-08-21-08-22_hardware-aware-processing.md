# Plan: hardware-aware processing, worker lanes, and resource visibility

## Context

Voxint is a single-operator, self-hosted audio-intelligence pipeline for
non-technical users running on their own hardware. Today the backend is hardware
*blind* and hard to reason about at runtime:

- Device selection is availability-only. There is **no** telemetry of GPU
  utilization, VRAM, temperature, or thermal state anywhere (`psutil`/`pynvml`
  absent; `torch` is not a core-app dependency, it lives only inside the three
  model-service images).
- Safe sizing for a modest single GPU is manual and unfinished (issue **#96**:
  worker `--concurrency`, whisper `BATCH_SIZE`, `MAX_PENDING_REQUESTS`).
- The operator has no live view of resource state. `health_probe.probe_services`
  reports per-service `device` + latency, but only during first-run setup.
- A discovered latent bug: a local deployment compose override's comments
  describe a two-lane worker split via Celery `task_routes`, but **no
  `task_routes` exists in code**
  (`src/voxint/worker/app.py`), and `run_pipeline` is a *single monolithic task*
  running all six stages inside one `execute_run` call — so `task_routes` alone
  cannot move post stages to another lane. The `-Q post` lane receives nothing.

This work makes the backend hardware-*aware* (know the hardware, pick safe
defaults), splits processing into real GPU/post lanes so the machine manages
contention honestly, and gives the operator both a compact live status strip and
a dedicated resource-management/analytics page, all within the "no unnecessary
bloat, honest UX, numerics-stability-is-load-bearing" doctrine.

**Scope decisions (confirmed with the operator):**
1. Land the safe-defaults spine first, then visibility (both matter).
2. Ship **both** a compact dashboard resource strip **and** a dedicated resource
   management + analytics page.
3. **Auto-apply** a conservative, parity-gated tested-profile baseline at install;
   advisory tuning thereafter.
4. Do the **full stage-lane split** (GPU lane / post lane) in this feature.

> On approval, copy this plan to `docs/plans/{YYYYMMDD-HHMM}_hardware-aware-processing.md`
> (project convention) as the first implementation step; this path is the
> plan-mode working copy.

## Design decisions locked by the 3-model consult (codex + deepseek-v4-pro + kimi-k3)

All three models converged on these; they are non-negotiable guardrails:

- **Telemetry originates in the services** (only they hold the GPU + a CUDA
  context; the app/worker containers get no GPU reservation). Transport is an
  **additive, nested, optional `resources` block** on each service `/healthz`
  (the contract already sanctions additive v1 fields; `health_probe.py` already
  tolerates unknown keys).
- **Never read NVML per request.** A background sampler thread in each service
  refreshes cached values every N seconds; `/healthz` serves the cache. NVML
  failure must **never** change readiness (a healthy model stays `200`).
- **Resolve the GPU by UUID, not index.** NVML physical indices differ from
  torch's `CUDA_VISIBLE_DEVICES`-remapped indices; a naive "GPU 0" read reports
  the *wrong* GPU. Use `torch.cuda.get_device_uuid` -> `nvmlDeviceGetHandleByUUID`.
- **Deduplicate the shared GPU.** All three services commonly report the *same*
  physical card; the UI aggregates by UUID into one device, and NVML memory is
  device-*global*, never labeled as one service's usage.
- **Parse telemetry from `503`/degraded bodies too** (`health_probe._probe_one`
  currently returns before reading the body on 503 — telemetry is most useful
  exactly when a service is degraded).
- **Integer bytes on the wire**, converted for display (MB/MiB is ambiguous and
  desyncs the trio); reject NaN/inf, bound utilization 0-100, enforce
  `used <= total`, tolerate unknown future throttle-reason bits.
- **Batch-size is a numerics knob, not ops tuning.** `BATCH_SIZE` feeds whisper's
  `decode_config_hash` and can move outputs, so any auto-applied value must come
  from a *tested profile* behind the parity gates + an OOM soak, never a formula.
- **The stage-lane split is a state-machine change, not a config edit.** It must
  preserve the CAS/claim/lease/retry/recovery/cancellation contract and make
  recovery lane-aware.

## Workstreams (ordered; each independently reviewable)

### W0 - Full stage-lane split (foundation, lands first)
Corrects worker topology before any sizing work tunes against it.

- Introduce a single **stage -> lane** map (`GPU` = ACQUIRE..DIARIZE_EMBED,
  `POST` = ENHANCE_MATCH, FINALIZE) as the one source of truth, consumed by the
  handoff, recovery, and `task_routes`.
- Split `run_pipeline` at the DIARIZE_EMBED -> ENHANCE_MATCH boundary. `execute_run`
  is already per-stage transactional and resumable; add a lane-boundary stop so
  the GPU-lane task runs GPU stages, then **commits, then publishes** a
  continuation task to `-Q post` (reuse the commit-before-publish +
  `OperationalError`-swallow pattern from `_publish_watch_run`,
  `worker/tasks.py:297`). The post task rebuilds per-run preferences + the
  `HttpLLMClient` (as `run_pipeline` does at `tasks.py:145-171`) and calls
  `execute_run` to resume from ENHANCE_MATCH. Claims/CAS make a duplicate delivery
  a no-op.
- Make **`recover_interrupted_runs` lane-aware**: re-publish a stalled RUNNING run
  to the queue its `current_stage` maps to, not always the default queue
  (`worker/tasks.py:238` today re-publishes `run_pipeline.delay`). Same for the
  stale-QUEUED re-publish in `recovery_sweep`.
- Add `task_routes` in `worker/app.py` for the genuinely standalone tasks
  (`generate_run_asset`, `research_speaker`) -> `post`.
- Update `compose.yaml` to run the two lanes (base gains a `post` worker; today
  the split lives only in a local deployment compose override), and **correct
  the misleading comments there**.
- **Critical files:** `src/voxint/pipeline/engine.py` (`execute_run`,
  `recover_interrupted_runs`, `default_stage_leases`), `src/voxint/worker/tasks.py`
  (`run_pipeline`, `recovery_sweep`, new post-continuation task),
  `src/voxint/worker/app.py` (`task_routes`), `compose.yaml`, and the local
  deployment compose override.

### W1 - Telemetry at the source (services)
- New torch-free `resource_probe` helper (one shared copy vendored identically
  into all three service images; a contract test asserts they stay identical).
  Background sampler thread; NVML via **`nvidia-ml-py`** (pure-ctypes, cheap;
  earns its place *specifically* for temperature + throttle reasons, which
  torch-only cannot provide). CPU tier reports host-visible cores/load via stdlib,
  labeled advisory (cgroup quota caveat documented).
- Extend each service `HealthResponse` schema (`services/*/app/schemas.py`) with an
  optional nested `resources` object: `gpu_uuid`, `utilization_percent`,
  `vram_used_bytes`, `vram_total_bytes`, `temperature_celsius`, `throttle_active`,
  normalized `throttle_reasons` (decoded labels: `thermal_sw`/`thermal_hw`/
  `power`/`clock`/`idle`, never a raw bitmask), `sample_age_seconds`,
  `availability` (tri-state: `disabled`/`unsupported`/`ok`), plus a small
  `admission` block sourcing contention honestly: `pending`, `max_pending`,
  `rejected_since_start`, `process_started_at`.
- Update `docs/gpu-contracts.md` `/healthz` section + `tests/contracts/test_schemas.py`.
- Service-side telemetry config is **service env**, not `voxint.config.Settings`
  (services don't consume it); compose must forward it, with a compose-render test.

### W2 - App consumption
- Extend `health_probe.ServiceHealth` to carry the optional telemetry (tolerate
  absence exactly as `device` is today), and **parse it on 503 bodies**.
- New `resource_status` module beside `health_probe.py`: **bounded-concurrent**
  fan-out (not 3 sequential probes) + a short-TTL single-flight cache with
  `sample_age`, so a 15s browser poll across multiple tabs never blocks on live
  probes. Aggregate/dedup by `gpu_uuid`.
- Surface telemetry in `voxint doctor` **and** `voxint stats` (the trio must not
  silently split).

### W3 - Visibility: strip + dedicated page + trio
- One shared `ResourceSnapshot` type; `/dashboard`, `/metrics`, and `voxint stats`
  all render from it via pure renderers (extend `src/voxint/api/stats_query.py`;
  keep `/metrics` hand-rolled, add `voxint_gpu_*` gauges + `promtool`-style test).
  `/metrics` and `/dashboard` read the **cached** snapshot, never probe services
  synchronously at render time.
- **Dashboard strip:** compact, curated signals only (device state idle/working/
  saturated, thermal warning, VRAM-near-limit, queue backlog) with plain-language
  remedy copy - not raw utilization %, to avoid alarm fatigue (100% during
  transcription is healthy). htmx fragment, reuse `.stat-cards`/`.minibar`/`.pill`.
- **Dedicated resource page:** new authenticated nav page (route shape copied from
  `/dashboard` at `app.py:4836`) with the fuller live view (aggregated GPU card,
  temperature + decoded throttle reasons shown separately and labeled
  instantaneous, cumulative `max_temp_since_start`/`throttle_events_since_start`,
  per-lane queue depth/backlog, admission/saturation, and the existing
  stage-duration/throughput analytics). htmx polling; a React island only if an
  interactive chart genuinely earns it.
- **Thermal is warn-only** in v1 (driver already protects hardware; a software
  admission loop would fight it) - deferred with rationale; every surfaced warning
  carries a one-step remedy.

### W4 - Hardware-aware conservative auto-defaults (#96)
- Host-side detection in `scripts/install.sh` (the host can see the GPU; the app
  container cannot). Detection is **profile matching**, not a VRAM formula: a
  small table of tested profiles keyed by GPU identity + model/compute-type, with
  an explicit **conservative unknown-hardware fallback**.
- **Auto-apply** the conservative baseline (worker `--concurrency`,
  `MAX_PENDING_REQUESTS`, and `BATCH_SIZE` only from a parity-passed profile) via a
  generated `compose.override.yaml` with dry-run output, atomic write, and
  conflict detection; advisory `voxint hardware` re-recommends thereafter.
- Every profile that sets `BATCH_SIZE` must pass `tests/parity/` + an OOM soak
  before it ships. Fold the manual #96 levers in `docs/operations.md` into this.

## Config surface
- App-side `resource_*` Settings (e.g. `resource_status_ttl_seconds`) documented in
  a new `.env.example` section + `tests/contracts/test_resource_config.py`
  (mirror `test_watch_folder_config.py`: env-doc + default + bounds; assert absent
  from `TIER_SCALED_TIMING_FIELDS`). Keep the knob count minimal - poll cadence
  stays a single source of truth shared with the htmx template.
- Service-side telemetry env is separate and fail-soft (a malformed value must not
  crash-loop a service or fail `/healthz`).

## Testing strategy
- **Unit (fake NVML):** library-absent, permission-denied, unsupported fields,
  GPU-lost mid-run, multi-GPU/MIG, unknown throttle bits, UUID resolution -> all
  degrade to honest null without touching readiness.
- **Contract:** additive/nested `resources` accepted + optional; finite/bounded
  numbers; integer-byte units; old-service body (no `resources`) -> "unavailable";
  malformed values -> unavailable not crash; identical vendored `resource_probe`
  across the three images; compose env-propagation render; `test_resource_config.py`.
- **Trio agreement:** one fixture -> `voxint stats` / `/dashboard` / `/metrics`
  emit the same resource numbers.
- **Lane split (highest-risk):** a contract/integration test proving
  ENHANCE_MATCH/FINALIZE execute on the `post` worker and GPU stages on the GPU
  worker; crash-at-handoff resumes via recovery to the *correct* lane;
  cancel-at-handoff stays clean; retry budget and per-run LLM-client lifecycle
  preserved across the lane boundary. Reuse the CAS/claim/recovery test patterns
  around `engine.py`.
- **Integration:** dashboard strip + resource page (full page + htmx fragment
  carry the numbers; degrade honestly when telemetry absent; one aggregated card
  for a shared GPU); mixed-version deploy (new app + one old service).
- **Parity/soak (W4):** every batch-size profile passes `tests/parity/` and an OOM
  soak; a long GPU soak samples `/healthz` during inference and shows no
  throughput/allocator/output regression.

## Verification (end to end)
- `ruff`/`mypy`/`pytest` clean; `VOXINT_PARITY_REQUIRED=1` green for any W4 profile.
- On GPU maintainer hardware: bring up the two-lane stack, submit a run, confirm
  GPU stages run on the GPU worker and post stages on the post worker (logs + DB
  `stage_runs.worker_id`), the resource page shows one aggregated GPU with live
  VRAM/temp, and `voxint stats`/`/metrics`/`/dashboard` agree.
- `install.sh` on a known GPU picks the expected profile and writes a valid
  `compose.override.yaml`; unknown hardware falls back conservatively.

## Risks / open questions
- **Lane-split correctness** is the dominant risk: crash-safe handoff, lane-aware
  recovery, and preserving the retry/cancellation invariants. Treat W0 as its own
  reviewed, parity-adjacent change; it must land and bake before W4 tunes sizing.
- **Per-field telemetry gaps** (e.g. WSL2/driver states with util but no temp) -
  handled by the tri-state `availability`, but worth confirming on target hardware.
- **`nvidia-ml-py` container plumbing** needs the NVIDIA `utility` capability +
  mounted driver lib preserved across overlays (not just a requirements entry).
- **Dedicated page vs alarm fatigue:** the operator wants the page; W3 mitigates by
  curating the strip and labeling raw metrics as instantaneous on the page.

## Review notes (3-model consult paper trail)

Drafted, then critiqued in parallel by **codex** (clink planner), **deepseek-v4-pro**,
and **moonshotai/kimi-k3**. Strong convergence; folded in:

- **Telemetry shape** (all three): nest as one optional cached `resources` block,
  not flat fields; background-sample, never per-request NVML; parse 503 bodies;
  integer bytes; normalized throttle reasons. **Accepted.**
- **Shared-GPU dedup by UUID** (all three; codex/kimi strongest): three services
  report one card; index is unstable. **Accepted** (W1/W2/W3).
- **Contention is unsourced** (all three): added an `admission` block + made
  lane/queue depth a real W2/W3 work item rather than an unbacked UI promise.
  **Accepted.**
- **task_routes cannot move in-task stages** (codex, decisive): `run_pipeline` is
  monolithic. This turned "optional small fix" into the W0 full stage-lane split -
  which the operator chose. **Accepted, promoted to foundation.**
- **Batch-size touches numerics/parity** (codex): auto-applied sizing must come
  from parity-gated tested profiles + OOM soak. **Accepted** (W4 gate).
- **Sequential probe blocking** (deepseek/codex): bounded-concurrent + TTL cache;
  `/metrics` never probes live at render. **Accepted** (W2/W3).
- **App vs service config split** (codex/kimi): service telemetry is fail-soft env,
  forwarded by compose, tested separately. **Accepted.**
- **Cut config knobs** (kimi): dropped a user-tunable temp threshold and poll knob;
  surface the driver throttle flag, keep cadence single-sourced. **Accepted.**
- **Scope inversion - #96 is the product, page is gold-plating** (kimi): raised to
  the operator, who chose defaults-first sequencing but still wants both strip and
  page. **Resolved by decision:** W4 conservative auto-apply prioritized; strip
  curated to avoid alarm fatigue.
- **Defer automatic thermal control** (all three): **Accepted** - warn-only v1 with
  remedy copy + cumulative counters; precise event-reason wording, not a generic
  throttle badge.
