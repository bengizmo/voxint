# Changelog

All notable changes to Voxint. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (0.x — expect breaking changes between minors).

## [Unreleased]

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

[Unreleased]: https://github.com/bengizmo/voxint/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/bengizmo/voxint/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bengizmo/voxint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bengizmo/voxint/releases/tag/v0.1.0
