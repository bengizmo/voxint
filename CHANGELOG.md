# Changelog

All notable changes to Voxint. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/) (0.x — expect breaking changes between minors).

## [Unreleased]

### Added
- **Offline speaker-name suggestions** (#38): a new `names.offline`
  enrichment producer mines evidence-backed name candidates from stored
  source metadata (title/description/channel/tags) and transcript text
  (self- and host-introductions) — fully offline, deterministic regex with
  explicit false-positive guards, no LLM and no network. Cluster-level
  (per-diarization-label) suggestions come **only** from self-introductions
  inside that cluster's own segments; everything else stays a run-level
  hint, so a title mention can never masquerade as cluster identity.
  Scoring is explainable (max pattern reliability + small corroboration/
  diversity/domain-pack-seed bonuses, capped at 0.95) with the full
  component breakdown stored per candidate. Reruns supersede cleanly via an
  input-signature idempotency key. Invoke with `voxint enrich names
  <run_id>` (or `--all-completed`), or from the workbench. New settings:
  `ENRICHMENT_NAMES_ENABLED` (default true), `ENRICHMENT_NAMES_LLM_ENABLED`
  (default false, requires `LLM_ENABLED`).
- **Name-suggestion review surface** (#38): the adjudication workbench now
  shows a "Name hints" block (run-level) and per-label "Self-introduced
  (unverified)" suggestions, each with its evidence snippet and score.
  Operators can trigger/re-run the sweep (claim-gated, synchronous) and
  accept or reject each suggestion — accepting records a profile-review
  decision only, never a speaker, assignment, or adjudication ruling. An
  accepted per-label suggestion prefills the Enroll input (editable, never
  auto-submitted). Rerun duplicates group under their decided history
  instead of re-presenting as new.
- **Enrichment draft schema** (#37): machine-derived claims about speakers
  and runs now live as reviewable, evidence-backed drafts. Four new tables
  (migration 0010): `enrichment_producer_runs` (one row per completed
  producer invocation — scope, covered fields, monotonic generation, and an
  explicit `outcome='none'` when a producer looked and found nothing),
  immutable `enrichment_candidates` (claim field/value, producer-local score
  with visible components, write-once supersession stamp), normalized
  `enrichment_candidate_evidence` (one claim can cite a metadata field,
  transcript segments, and several URLs together), and the append-only
  `profile_review_decisions` human trail — deliberately separate from the
  attribution ledger. Review state is derived at read time (decision >
  superseded > proposed), never stored. Single sanctioned writers in
  `voxint.enrichment` (atomic per-scope finalization under an advisory lock;
  terminal accept/reject with idempotent replay). Invariant unchanged: drafts
  are suggestions *about* identity — accepting a name claim never touches
  `speakers.display_name`, machine proposals, or attribution resolution.
  Schema + writer layer only; the producers (#38, #40) and their console
  surface come separately.
- **Source media metadata capture** (#36, schema slice): new write-once
  `media_source_metadata` table (1:1 with `media_items`) holding normalized
  extractor context — title, uploader/channel (+URLs), description, upload
  date, source-claimed duration, tags, canonical URL, extractor
  name/version — plus a bounded, allowlisted, schema-versioned `raw` JSONB
  subset and the acquisition timestamp. Metadata is context, not identity:
  a MediaItem is per-acquisition, so a snapshot can never rewrite the
  context a past adjudication was made against. New nullable
  `pipeline_runs.operator_notes` keeps human input structurally apart from
  scraped metadata. Migration 0009 (additive, clean downgrade).
- **Metadata capture at acquisition** (#36): the yt-dlp invocation now also
  writes a clean info-JSON (`--write-info-json --clean-info-json
  --no-write-playlist-metafiles`, typed `infojson:` output — same invocation,
  no extra network exposure); ACQUIRE sanitizes it through a strict allowlist
  (secret-bearing keys — `formats`, `http_headers`, `cookies`, signed URLs —
  are never copied), publishes a hash-addressed replay sidecar before the
  media file, and inserts the write-once snapshot row. Best-effort: bad
  metadata logs a warning, never fails an acquisition.
- **Operator notes + surfacing** (#36): run detail gains a Source-metadata
  section and an editable Operator-notes form (`POST /runs/{id}/notes`,
  CSRF-gated, 10K-char cap); the runs browser shows the source title with
  media-path fallback; new `GET /runs/{id}/export.json` returns a versioned
  envelope (run + source_metadata + operator_notes + segments) while the
  pinned bare-array `/review/{id}/export.json` contract stays frozen.
- **macOS arm64 CI lane** (`.github/workflows/metal-lane.yml`, issue #34):
  nightly + manual-dispatch partial Gate M automation on `macos-15` runners
  (real MPS) — launcher unit tests on real macOS, then the whisper/pyannote/
  titanet parity modules from the launcher's own sha-verified per-service
  venvs, with provenance-keyed weight caches, an MPS tensor-op probe, and a
  junit guard that fails the lane if an expected module green-boards
  fully-skipped. Maintainer Gate M (per-chip verdict refreshes) is
  unchanged — this catches regressions between refreshes.
- **Metal-tier log rotation** (metal review follow-up): `voxint-metal.sh up`
  now installs a daily launchd job (`com.voxint.metal.logrotate`) that
  copy-truncates any service log over 50 MB to a timestamped archive,
  keeping the newest 5 — launchd's `StandardOutPath` never rotates and
  `KeepAlive` keeps services up for months. `VOXINT_METAL_LOG_MAX_MB` /
  `VOXINT_METAL_LOG_ARCHIVES` override; new `rotate-logs` subcommand runs a
  pass by hand; `logs -f` now follows with `tail -F`.
- **Parity references now record the exact request payloads** they were
  generated with (`tools/generate_parity_references.py` writes a
  `meta.request` block per reference): parity lanes replay hardcoded
  "service-default" params, and a regenerated reference could otherwise pair
  silently with different params than the lanes measure. Takes effect on the
  next reference regeneration; committed references predate the field
  (metal review follow-up).
- **Contract test binding `compose.metal.yaml` ports to the metal launcher's
  `service_port()`**: the overlay's `host.docker.internal:<port>` URLs and
  the native services' bind ports were each pinned to their own literals —
  a port moved in only one place would have kept both tests green while the
  worker called a dead port (metal review follow-up).

### Changed
- **Metal parity bounds ratcheted from Gate M evidence** (slice 9, panel
  consult recorded in the commit): pyannote boundary drift ≤ 0.10 s (was
  0.25), agreement vs reference ≥ 0.97 (was 0.95), MPS-vs-CPU ≥ 0.995 (was
  0.99); whisper transcript similarity ≥ 0.96 (was 0.95), confidence drift
  ≤ 0.05 (was 0.15). Repeat/segment/count bounds unchanged. Three deferred
  decisions closed as measured no-ops: CoreML EP default stays off (no
  speedup), no metal timeout factor (0.38–0.45× RT transcribe fits
  GPU-class budgets), no committed metal reference oracle (re-affirmed).
  See docs/gpu-contracts.md metal verdict table.

## [0.9.0] — 2026-08-14

The Apple Silicon "metal" compute tier (#1): native macOS model services
under launchd with diarization on the Apple GPU via torch-MPS, measured
against the committed CUDA references (maintainer Gate M PASS on an M1 Pro),
plus the tier-independent device-control contracts (`DIARIZER_DEVICE`,
`TITANET_ORT_PROVIDERS`) and multi-model review hardening of the launcher.

### Added
- **Apple Silicon "metal" compute tier**: the core stack stays in Docker
  (`compose.metal.yaml` rewires api/worker to `host.docker.internal`) while
  the three model services run natively on macOS — set up, sha-verified,
  and supervised under launchd by the new `scripts/metal/voxint-metal.sh`
  (`setup / up / down / status / logs / doctor / run --foreground`) — so
  diarization runs on the Apple GPU via torch-MPS (~5× native-CPU
  diarization measured on an M1 Pro, identical outputs; transcription stays
  on host CPU in v1 and remains the bottleneck). The installer grew an `[M]`
  option (default on Apple Silicon) that starts the core and hands off
  honestly. New device-control contracts, both tier-independent:
  `DIARIZER_DEVICE=auto|cuda|mps|cpu` (a forced device must pass the sanity
  probe or the service refuses to start) and `TITANET_ORT_PROVIDERS`
  (requested ONNX EPs must be verifiably active — no silent fallback
  anywhere). Metal parity lanes gate against the committed CUDA references
  (no metal oracle by design): `tests/parity/test_pyannote_metal.py`,
  `test_whisper_metal.py`, `VOXINT_PARITY_ORT_PROVIDERS` threading for the
  titanet 3-level gate, and `tools/generate_parity_references.py --tier
  metal`. Maintainer-run Gate M documented in the release process.

### Fixed
- **Metal tier review hardening** (pre-landing multi-model review): the
  installer's metal handoff no longer claims model services "were started";
  whisper's runtime load is pinned to the same HF revision setup downloads
  (`WHISPER_REVISION`, launcher-set — unset keeps image behavior), the local
  manifest records that revision and excludes HF cache bookkeeping, and a
  stale/corrupt cache is cleared before re-download instead of being
  re-blessed; `voxint-metal.sh up` preflights venvs/weights/config and waits
  out the launchd bootout-vs-bootstrap race instead of crash-looping under
  KeepAlive; `VOXINT_METAL_DIARIZER_DEVICE` accepts only `mps`/`cpu` (`auto`
  would re-open silent CPU fallback); vendored-config generation escapes sed
  metacharacters in the destination path and fails explicitly under
  `PYTHONOPTIMIZE`; doctor now verifies whisper weights; sha verifiers
  distinguish unreadable provenance from weight mismatch; metal parity lanes
  fail closed on empty diarizations, pin the whisper snapshot, and shed
  ambient `TITANET_ORT_PROVIDERS` / `PYANNOTE_*` env.
- **Metal launcher `.env` reading**: `voxint-metal.sh` read `MEDIA_ROOT`
  verbatim from `.env`, but the installer writes it single-quoted — the
  launcher hard-failed on every installer-generated file ("does not resolve
  to an existing directory"). Values are now normalized exactly like the
  installer reads them back (strip CR, blanks, and one matched pair of
  quotes), matching what Compose interpolation passes to the containers.

## [0.8.0] — 2026-08-14

Runs search (#8) plus CLI/observability ergonomics (#25, #32): the runs
browser gains transcript full-text search and facets, and the CLI grows
export, list, doctor, stats, and watch alongside a Prometheus `/metrics`
endpoint. Also carries the cross-platform / dev-experience hardening bundle
(#26, #27, #28, #29).

### Added
- **Search on the runs browser** (`/runs`, #8): transcript full-text search
  (`q=`, Postgres `websearch_to_tsquery` syntax — quotes, `-word`, `OR`) with
  a highlighted first-hit snippet per run, a speaker facet (runs whose
  read-time attribution — human ruling or grounded cosine, merge tombstones
  canonicalized — is the selected speaker; archived speakers stay listed,
  marked), a source-path substring facet, and UTC date-range bounds. All
  facets AND-compose with the existing status/review filters and keyset
  pagination. Backed by two GIN expression indexes (migration 0008) over
  `raw_text` AND `enhanced_text` separately — enhancement never makes the raw
  rendering of a term unfindable, and vice versa. Dictionary is `english`
  (stemming recall); a stopword-only query matches nothing by design. Results
  stay newest-first — no relevance ranking pre-1.0 — and the search document
  is one segment (terms split across segments of a run don't AND-match).
- **Structured & subtitle transcript exports.** The review console now offers
  SubRip (`.srt`), WebVTT (`.vtt`), JSON, and diarization RTTM (`.rttm`)
  alongside the existing plain-text export, at
  `GET /review/{run_id}/export.{srt,vtt,json,rttm}` (all accept `?text=raw|
  enhanced`, default enhanced; RTTM carries raw diarization labels). SRT/VTT/
  JSON/TXT share one set of pure formatters (`voxint.export`) with the CLI, so a
  downloaded file and a piped export are byte-identical.
- **`voxint export <run_id> --format srt|vtt|json|rttm|txt`** — headless
  transcript export to stdout or `-o PATH` (refuses to overwrite without
  `--force`); `--text raw|enhanced` selects the transcript variant.
- **`voxint list`** — a CLI run browser (newest first) mirroring the `/runs`
  query, with `--status`, `--limit` (1–500, default `runs_page_size`), and
  `--json`.
- **`voxint doctor`** — read-only preflight diagnostics: Postgres, Redis, and
  each model service's `/healthz` (reporting the compute `device`) are hard
  checks (exit 1 if any is down); the Hugging Face token and LLM endpoint are
  advisory (reported, never fail the exit). Credentials are never printed.
- **`voxint stats`** — an aggregate, read-only system summary: run counts by
  status, failed stage attempts by stage, average per-stage duration (over
  finished attempts), roster size, and runs created in a window (`--since`,
  accepting `<n>h`/`<n>d`/ISO-8601, default 24h). `--json` emits a stable object.
- **`GET /metrics`** — a Prometheus text-exposition endpoint (format 0.0.4)
  built on the same query module, on the authenticated router (scrape it with
  `basic_auth`, keeping the "everything but `/healthz` authenticates" invariant).
  Every `RunStatus`/`Stage` series is zero-filled so a series never disappears
  between scrapes; the one windowed gauge bakes its window into its name
  (`voxint_runs_created_24h`).
- **`voxint watch <run_id>`** — follow a run until it stops advancing, with a
  live progress line on stderr. Exit codes: `0` completed, `1` failed/cancelled,
  `2` missing run, `3` awaiting adjudication (paused — needs a human ruling),
  `124` timeout. `--interval` (default 2s) and `--timeout` (default 3600s) tune
  the poll.
- **`voxint submit --wait`** — enqueue, then follow the new run to a stop state
  with the same poll loop and exit codes (the run id stays alone on stdout;
  progress goes to stderr).

### Fixed
- **macOS/BSD media-download teardown raised the wrong error (#26).** On a
  download timeout, if the yt-dlp process-group leader had already been reaped
  and the survivor was a zombie reparented to launchd, `killpg` returns `EPERM`
  (not `ESRCH`); the raw `PermissionError` escaped the teardown and replaced the
  intended redacted `AcquisitionError`. Both teardown signals now suppress
  `PermissionError` alongside `ProcessLookupError`. (Linux returns `ESRCH`, so
  this was macOS/BSD-only; validated by a new monkeypatched unit test — a real
  Mac run is the true confirmation.)
- **Installer could offer the busy port as its own "alternate" (#27).** On
  macOS/BSD a listener with a full accept-backlog refuses further connects, so
  the `/dev/tcp` probe can misread a bound port as free; `resolve_port` then
  re-scanned starting *at* the known-busy default and could suggest it right
  back. It now searches strictly above the busy port, so the offered alternate
  is always distinct. The probe stays advisory (Compose remains the collision
  authority); its residual limitation is now documented in-script.

### Changed
- **Fresh `uv sync --extra dev` checkout is green again (#28).** The loopback
  default-credentials test is now hermetic (`_env_file=None`, so an on-disk
  `.env` can't override the code default), and the two librosa-dependent mel
  contract tests `importorskip("librosa")` (it ships only in the `parity`
  extra) — they still run in the parity lane, and no assertions were weakened.
- **Documented the CPU-tier host-RAM floor (#29).** The CPU tier holds the
  models in RAM (~6 GiB idle; whisper alone ~4.8 GiB) and needs **≥ 8 GB**
  available to the container host — on Docker Desktop the VM's memory limit, not
  the physical machine — or services are OOM-killed with an opaque exit. Noted
  in `docs/operations.md`, `docs/onboarding.md`, and the installer's tier prompt.

## [0.7.0] — 2026-08-14

Speaker roster management (#7): the roster is no longer write-only.

### Added
- **Speaker roster page** (`/speakers`, #7): view every enrolled speaker with
  its enrollment provenance, machine-proposal count, and a deterministic
  voiceprint strip derived from its own centroid. Curation actions: rename,
  merge duplicates, archive/restore, and remove a bad enrollment embedding —
  all without ever rewriting the append-only decision ledger. Merges keep the
  source speaker as a tombstone (`merged_into_id`, migration 0007) and readers
  canonicalize at read time, so historical rulings render under the merge
  target while the ledger rows stay byte-identical.

### Changed
- Speaker matching, the workbench assign dropdown, and the decide route now
  consider **active** speakers only — merged and archived speakers stop
  attracting proposals and decisions (archiving also removes the speaker's
  machine proposals; restore does not resurrect them).

### Fixed
- Enrollment replay now validates against durable provenance (run, label,
  operator) instead of the current display name, so renaming a speaker can no
  longer make a replayed enrollment POST falsely conflict.

## [0.6.0] — 2026-08-14

Token-free onboarding: the diarization weights are vendored (#24). No
numerical changes — vendored-vs-HF diarization verified byte-identical.

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
- `DIARIZER_MODEL_NAME` is now interpolated from `.env` by every compute
  overlay, so the documented override works without editing compose files.

### Fixed
- **CUDA pyannote image**: `setuptools` pinned `>=70,<81` with a build-time
  `pkg_resources` canary — the unpinned upgrade would have shipped an image
  that crashes on boot at the next rebuild (setuptools 81 removed
  `pkg_resources`, which pyannote.database imports; the CPU flavor already
  carried the pin).
- `/healthz` keeps reporting the canonical `pyannote/speaker-diarization-3.1`
  identity for the vendored default; an explicitly configured
  `VOXINT_VENDORED_PIPELINE` that does not exist now fails fast instead of
  silently degrading to a gated network fetch.

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

[Unreleased]: https://github.com/bengizmo/voxint/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/bengizmo/voxint/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bengizmo/voxint/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bengizmo/voxint/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bengizmo/voxint/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/bengizmo/voxint/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bengizmo/voxint/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/bengizmo/voxint/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bengizmo/voxint/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bengizmo/voxint/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bengizmo/voxint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bengizmo/voxint/releases/tag/v0.1.0
