# Changelog

All notable changes to Voxint. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (0.x — expect breaking changes between minors).

## [Unreleased]

### Changed
- **No Hugging Face account or token needed** (#24): the
  `speaker-diarization-3.1` pipeline weights are now vendored into the
  pyannote images — sha256-pinned from the standing `pyannote-models-v1`
  asset release (`services/pyannote/models/provenance.json`; segmentation-3.0
  MIT, WeSpeaker embedding CC-BY-4.0, redistributed with attribution) and
  loaded offline by default. Vendored-vs-HF parity verified byte-identical on
  the parity clip. `HF_TOKEN` is demoted to an optional override for a custom
  `DIARIZER_MODEL_NAME`; the installer no longer prompts for a token, the
  compute overlays start without one (the `${HF_TOKEN:?}` guard is gone), the
  setup wizard drops its token row, and pyannote's CI smoke runs
  unconditionally (the secret-absent SKIP lane is deleted).

## [0.5.1] — 2026-08-14

Robustness patch for burst-load resilience (#23). No inference or contract
changes.

### Fixed
- **All long-running services now carry `restart: unless-stopped`** (core
  stack + every model-service overlay; `migrate` keeps its deliberate
  `"no"`): a transient model-service crash self-heals instead of staying
  down until a human runs `up -d` (#23).
- **Connection failures to a model service now say what they mean**: when
  the service DNS name stops resolving or the connection is refused —
  inside the compose network this almost always means the container is
  down — the worker's ledger error names the service host and says the
  service is likely down or restarting (pointing compose deployments at
  `docker compose ps`), instead of surfacing a raw resolver error that
  reads as a network problem (#23).

## [0.5.0] — 2026-08-14

AMD-GPU acceleration for ASR (#4). The ROCm tier is a hybrid: whisper runs on
the AMD GPU, pyannote/titanet stay on CPU. No numerical changes to existing
flavors.

### Added
- **whisper `-rocm` image** (`services/whisper/Dockerfile.rocm`, amd64):
  same faster-whisper 1.2.1 / CTranslate2 engine and code path as CUDA —
  the CTranslate2 4.8.1 **ROCm build** (GitHub release wheel, sha256-pinned;
  not on PyPI) on ubuntu:24.04 with the minimal measured ROCm 7.0.2
  runtime-library set. Torch-free (the 1.2.x Silero VAD is
  onnxruntime-based). Measured on RDNA4 (RX 9060 XT, gfx1200): warm
  transcription 4.8× the CPU baseline on the parity corpus clip (this
  image's smoke measured faster still); host needs only the amdgpu kernel
  driver.
- **`compose.rocm.yaml` overlay**: whisper on the GPU (`/dev/kfd` +
  `/dev/dri` passthrough + the owning host gid via `VOXINT_RENDER_GID`;
  no `video` group, no `seccomp:unconfined` — both verified unnecessary on
  real hardware), pyannote/titanet on the `-cpu` images,
  `COMPUTE_TIER=rocm` timing profile. Pin-parity contract test now covers it.
- **Installer AMD tier**: `[A]` in the compute-tier prompt (suggested when
  `/dev/kfd` exists and no NVIDIA driver is), records
  `VOXINT_COMPOSE_TIER=rocm` and auto-detects + records the gid owning
  `/dev/kfd` in `.env` (`VOXINT_RENDER_GID`); kept-`.env` re-runs re-detect
  and refresh it (the gid is per-host).
- **Honest `/healthz` device reporting without torch**: the CT2 ROCm build
  masquerades as CUDA and the `-rocm` image carries no torch, so
  `resolve_device_name` now also detects the loaded HIP runtime
  (`libamdhip64` in `/proc/self/maps`) and reports `device: "rocm"`.
- **`release.yml` `publish-whisper-rocm` lane** — build-only in CI (GitHub
  has no AMD-GPU runners); the real-GPU inference gate is a maintainer step
  on AMD hardware before tagging (Gate R, `docs/release-process.md`).
- Docs: `docs/operations.md` ROCm-tier section (incl. why pyannote/titanet
  stay CPU — MIOpen convolutions fail on current AMD consumer GPUs in both
  shipping torch-ROCm wheel lines), README AMD callout,
  `docs/gpu-contracts.md` device-reporting note, whisper README image matrix.

### Changed
- `cleanup_memory` in the whisper service tolerates a torch-free image
  (guarded import; CT2 manages its own device memory).

## [0.4.1] — 2026-08-14

Onboarding patch: closes the v0.4.0 first-run traps (#17–#22). No model
service, pipeline, or numerical changes — images rebuild, numerics untouched.

### Added
- **Installer compute-tier selection** (GPU / CPU / none-for-now; suggests GPU
  when `nvidia-smi` is present), remembered in `.env` as
  `VOXINT_COMPOSE_TIER`; one helper owns the tier → compose-file mapping and
  every installer Compose invocation goes through it, so the pull/up/status
  commands can never disagree about the active overlay (#18).
- **Installer Hugging Face token prompt** (hidden input, both pyannote gate
  URLs explained) with an advisory two-stage check — token validity, then
  access to each gated repo (terms accepted). Warnings only, never blocks;
  the token reaches curl via stdin config, never argv. Skipping the token
  records the tier but starts the core stack only (both compute overlays
  refuse to interpolate without `HF_TOKEN`), and the completion notice spells
  out the three steps to finish (#17).
- **Setup wizard SERVICES step**: a Hugging Face token presence row (never
  the value) and guidance covering both compute tiers, not just GPU (#17, #18).
- **Run page**: static guidance when a run failed at a model stage — start a
  compute tier, wait for it, requeue (#18).
- **`docs/interpreting-diarization.md`**: segment labels are a
  dominant-overlap projection and can under-report speakers (the turn ledger
  is the source of truth); short clips can over-split; honest note that
  `min/max_speakers` is service-API-only today (#22).
- **Offline installer test suite** (33 tests) driving the
  `VOXINT_INSTALL_LIB=1` seam with fake `docker`/`curl` on PATH: tier
  mapping, port-collision handling (#21), `.env` render/update/backup/0600,
  dotenv normalization, and secret non-disclosure (token never in
  stdout/stderr/argv).

### Fixed
- **Installer port-collision prompts were invisible**: after the first
  detected collision, a stray `exec … 2>/dev/null` in `port_in_use`
  permanently redirected the whole script's stderr to /dev/null — every
  later prompt and message vanished (#21).
- Installer re-runs that switch tier (or defer on a removed token) no longer
  strand the previous overlay's model containers
  (`docker compose up --remove-orphans`).
- Kept-`.env` reads now match Compose dotenv semantics (trailing CR,
  surrounding blanks, matched single/double quotes) — a hand-edited
  `HF_TOKEN=""` no longer defeats the skip-token deferral or produces a
  false "token rejected" warning.
- `.env` backups are forced to mode 0600 (`cp -p` had preserved a loose
  source mode).
- The false "a run simply waits on any service it needs" claim (wizard +
  onboarding docs) replaced with the real behavior: retry with backoff
  (about five attempts over roughly an hour and a half), then FAILED, then
  requeue from the run's page.

### Changed
- README leads non-NVIDIA users to the CPU tier from the top of the
  quickstart ("No NVIDIA GPU? Start here too"), and the CPU section is a
  linkable heading (#20).
- README and `voxint score --help` now state exactly what the harness
  scores: speaker attribution (name accuracy / agreement / ensemble) — ASR
  accuracy / WER is out of scope (#19).
- The installer handoff is honest about readiness: only the API is
  health-checked; model services are reported as *started* with the ps
  command to check them, not "enabled".

## [0.4.0] — 2026-08-13

CPU tier: run Voxint's full pipeline with **no NVIDIA GPU** — on plain
servers, AMD boxes, and Apple Silicon (Docker Desktop). Closes the
container-path ask of #1 (Apple Silicon) and #4 (AMD); accelerated ROCm and
native-Metal tiers are tracked separately.

### Added
- **Multi-arch (amd64 + arm64) `-cpu` image flavor** for all three model
  services (`voxint-{whisper,pyannote,titanet}:X.Y.Z-cpu`), built natively
  per arch (no QEMU) and merged into one manifest list. Unsuffixed
  model-service tags remain CUDA, unchanged.
- **`compose.cpu.yaml`** overlay: the whole stack on CPU with
  `docker compose -f compose.yaml -f compose.cpu.yaml up -d`. Sets
  `COMPUTE_TIER=cpu`, which scales default inference timeouts, stage leases,
  and the Celery visibility horizon so slow-but-healthy CPU runs are never
  reclaimed as hung. Honest expectation: long recordings take **hours** on
  CPU.
- **titanet ONNX Runtime engine in the shipped `-cpu` image**
  (`EMBED_ENGINE=onnx`, torch- and NeMo-free): same embedding space id
  (`titanet-large-v1`), kept on the measured three-level parity gate
  (mel / vector / decision) against the CUDA engine — verdict recorded in
  `docs/gpu-contracts.md`. The build verifies the model artifact's sha256
  against the committed export provenance; the ~100 MB `.onnx` ships via the
  standing `titanet-onnx-v1` model-asset release, never git.
- **pyannote device cascade** (`cuda → mps → cpu`) with a real-tensor-op
  startup probe that checks device output against a CPU reference — a backend
  that computes silently-wrong results (the historical MPS failure mode) is
  rejected, not trusted. MPS is inert in containers; the branch serves the
  future Apple host-process path.
- **Release gates in `release.yml`**: the strict titanet parity harness
  (`VOXINT_PARITY_REQUIRED=1`) runs on amd64 **and** arm64 runners and blocks
  the multi-arch builds; the per-arch smoke (`tools/smoke_cpu_services.py`)
  runs against the **untagged digest images before any tag exists** and
  requires healthz identity fields, a real corpus transcription, and a
  titanet embedding within cosine 0.999 of the committed CUDA reference
  (pyannote's smoke needs an `HF_TOKEN` secret and SKIPs explicitly when
  absent); tags are only ever attached to smoke-passed digests, and each
  manifest list is verified to expose exactly amd64+arm64.

### Changed
- **The app image (`voxint`) is now multi-arch** (amd64 + arm64).
- The whisper CUDA image's engine, pins, and behavior are untouched; the
  `-cpu` flavor runs the same faster-whisper/CTranslate2 int8 engine with
  CPU-appropriate defaults (`BATCH_SIZE=4`).

## [0.3.0] — 2026-08-13

Non-technical onboarding: get from a fresh clone to a first successful,
adjudicated run without editing config by hand.

### Added
- **Guided installer** (`scripts/install.sh`): one command that takes a fresh
  clone to a running core stack for non-technical users. Prompts only for an
  admin password and a media folder; auto-generates `CSRF_SECRET`, detects
  host-port collisions and offers a free alternate, and renders `.env` from
  `.env.example` (never overwriting an existing one without a timestamped
  backup). Preflights Docker + the Compose plugin (≥ 2.24), pulls the pinned
  images, starts the stack, and polls the API container's healthcheck — then
  prints the console URL and states plainly that the core stack is the control
  plane only (audio processing needs the GPU overlay). Bash 3.2+, macOS/Linux,
  no runtime dependency beyond Docker. (#2)
- **First-run setup wizard** (`/setup`): a guided, operator-authenticated flow
  that takes a fresh install to a configured state. Choose media folders (with
  an optional bounded scan that previews and batch-registers existing media),
  define a domain vocabulary that feeds both the Whisper `initial_prompt` and
  the LLM enhancement context, toggle optional LLM transcript enhancement, and
  check GPU service health honestly — core-only when the GPU overlay is absent,
  with no silent fallback. Preferences apply per run with no worker restart. An
  onboarding gate holds the console at the wizard until setup is finished, then
  releases the full app. Backed by an `app_settings` singleton (alembic
  revision 0006). (#3)
- **Guided 3-speaker tutorial**: a bundled synthetic 3-speaker sample and an
  idempotent `voxint tutorial seed` command that stages a ready-to-adjudicate
  run. Server-rendered `?tutorial=<step>` banners walk through the
  run → review → transcript flow on the real console pages, and a new
  **Settings** page re-runs the wizard and starts, replays (non-destructively),
  or completes the tutorial. (#3)

## [0.2.0] — 2026-08-12

### Added
- **Browser console** served from the same FastAPI app: a keyset-paged `/runs`
  execution-history browser (orthogonal `status=` / `review=` filters), a
  `/runs/{id}` run-detail page with the per-stage attempt ledger, and a
  resolver-attributed transcript view (`raw`/`enhanced`).
- **File upload** (`POST /submit`): bounded, streamed enforcement of
  `UPLOAD_MAX_BYTES` (default 5 GiB); each upload lands under a server-issued,
  uuid-namespaced immutable path, with idempotent form replay.
- **URL ingestion** via yt-dlp: `voxint fetch <url>` and `POST /fetch` register a
  `MediaItem.source_url` and enqueue a run. A new **ACQUIRE** stage —
  `STAGE_ORDER[0]`, a no-op for local/uploaded media — downloads it on the worker
  (alembic revision 0005 adds `source_url` and the `acquire` stage). Toggle with
  `YTDLP_ENABLED` (default on).
- **CAS requeue route** (`POST /runs/{id}/requeue`): the browser equivalent of
  `voxint requeue`, guarded by exact-revision compare-and-swap.

### Security
- **Two-gate SSRF model** for URL ingestion: a string-level check at submit and a
  host re-resolution check in the worker before download, sharing one
  public-address rule that unwraps IPv4-in-IPv6 embeddings and rejects site-local.
  Documented as authenticated admin egress with a residual that needs network
  policy (see `docs/architecture.md`).
- **yt-dlp lockdown**: `--no-config`, `--no-plugin-dirs`, `--no-exec`,
  `--no-playlist --max-downloads 1`, a size cap, hard wall-clock timeouts, and
  explicit proxy handling; proxy/cookies are treated as credentials and scrubbed
  from surfaced errors.
- **CSRF** on the mutation forms (`POST /submit`, `/fetch`, `/runs/{id}/requeue`,
  and `POST /review/{id}/claim`): a stateless, action-bound HMAC token keyed by a
  dedicated `CSRF_SECRET`.

### Changed
- Submission mutations are **durable-first**: the run is committed before the
  Celery task is published, so a broker outage leaves the run `QUEUED` (never
  `FAILED`) for the recovery sweep instead of failing the request.

## [0.1.0] — 2026-08-12

First public release.

### Added
- End-to-end pipeline: preprocess → transcribe (faster-whisper) + diarize
  (pyannote) + embed (TitaNet) → optional LLM transcript enhancement →
  speaker matching → human adjudication.
- Compare-and-swap run/stage state machine in Postgres with leased stage
  claims, retry budgets, and a beat-scheduled crash-recovery sweep.
- Adjudication web console (review queue, guarded slot workbench,
  decision-resolved transcript export) served as Jinja + htmx from the API.
- pgvector cosine speaker matching with a strict *named ≠ grounded* invariant;
  machine proposals kept separate from human rulings (append-only ledger).
- Scoring harness `voxint score` (name-accuracy, acoustic agreement, ensemble
  fusion) — DB-free, installable standalone from PyPI; synthetic walkthrough
  under `examples/`.
- Three GPU model services with frozen v1 HTTP contracts
  (`/v1/transcribe`, `/v1/diarize`, `/v1/embed`).
- Compose-first deployment: pinned GHCR release images by default,
  build-from-source overlays (`compose.build.yaml`, `compose.gpu.build.yaml`),
  one-shot `migrate` gate, swappable domain pack.

[Unreleased]: https://github.com/bengizmo/voxint/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/bengizmo/voxint/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bengizmo/voxint/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/bengizmo/voxint/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bengizmo/voxint/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bengizmo/voxint/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bengizmo/voxint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bengizmo/voxint/releases/tag/v0.1.0
