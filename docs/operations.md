# Operations

Running Voxint day to day: deployment, schema migrations, pipeline
operations, recovery, and the adjudication workflow. Architecture and data
model: [architecture.md](architecture.md); gate semantics:
[quality-gates.md](quality-gates.md).

## Deployment

Prerequisites: Docker Engine with the **Compose plugin ≥ 2.24** (`docker
compose version`; the legacy v1 `docker-compose` binary cannot parse this
stack, and older v2 plugins lack its optional-`env_file` syntax), and for the
GPU overlay an NVIDIA GPU with the NVIDIA container toolkit. The three model
services share one GPU and total ~3.5–4.5 GB of loaded VRAM; budget ~6–8 GB for
Whisper batch headroom (per-service figures in `services/*/README.md`). No
NVIDIA GPU? See [the CPU tier](#running-without-an-nvidia-gpu-cpu-tier) below —
slower, but runs anywhere, including arm64/Apple Silicon.

For a first run, `./scripts/install.sh` is the recommended path — it renders
`.env`, generates secrets, resolves port collisions, brings the core stack up, and
hands off to the in-browser setup wizard (see [onboarding.md](onboarding.md)). The
manual equivalent:

```bash
cp .env.example .env          # set at least VOXINT_PASSWORD
mkdir -p media                # pre-create the media mount so Docker doesn't create it root-owned
docker compose up -d          # core: Postgres+pgvector, Redis, migrate, API, worker, beat
docker compose -f compose.yaml -f compose.gpu.yaml up -d   # + GPU model services
```

Everything publishes on `127.0.0.1` only. If a default port is already taken
on the host, override the published side in `.env` (`POSTGRES_PORT`,
`REDIS_PORT`, `API_PORT`) — container-internal ports never change.

### Release images vs. building from source

The default compose files run the **pinned release images** from GHCR
(`ghcr.io/bengizmo/voxint*`), even on a `main` checkout — `VOXINT_IMAGE_TAG`
in `.env` overrides the pin. To run the code you actually checked out, layer
the build overlays (build first, then up — services that don't own the build
use `pull_policy: never` on the local tag and fail fast if it's missing):

```bash
# app image only (core stack):
docker compose -f compose.yaml -f compose.build.yaml build api
docker compose -f compose.yaml -f compose.build.yaml up -d
# app + GPU model services. The pyannote build needs the vendored checkpoints
# in place first (they are gitignored; sha256s in services/pyannote/models/provenance.json):
gh release download pyannote-models-v1 -R bengizmo/voxint \
   --pattern '*.bin' --dir services/pyannote/models
docker compose -f compose.yaml -f compose.gpu.yaml \
               -f compose.build.yaml -f compose.gpu.build.yaml \
               build api whisper pyannote titanet
docker compose -f compose.yaml -f compose.gpu.yaml \
               -f compose.build.yaml -f compose.gpu.build.yaml up -d
```

Exactly one service (`api`) declares the app `build:`; migrate/worker/beat
consume the tag it produces. Do not give several `build:` services a shared
`image:` tag — concurrent BuildKit writers race on it ("already exists").

### Running without an NVIDIA GPU (CPU tier)

Every model service also ships a **`-cpu` image flavor** — multi-arch
(amd64 + arm64), no GPU, no NVIDIA container toolkit. This is the supported
path for Apple Silicon (via Docker Desktop) and plain CPU servers (AMD-GPU
boxes have the faster ROCm tier below):

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

What changes relative to the GPU overlay — and what doesn't:

- **Speed — set expectations honestly.** CPU inference is orders of magnitude
  slower than GPU: transcribing a multi-hour recording takes **hours**, not
  minutes. This is fine for overnight/batch use and correctness-identical; it
  is not an interactive experience.
- **Host RAM floor — ≥ 8 GB.** The CPU tier holds the models in RAM instead of
  VRAM: whisper alone is ~4.8 GiB resident (large-v2 int8 + CTranslate2 arenas)
  and the tier idles around ~6 GiB total. Give the container host — on **Docker
  Desktop (macOS/Windows) this is the VM's memory limit**, not the physical
  machine's — at least **8 GB *including the core stack*** (Postgres, Redis, api,
  worker share the same VM); **16 GB is comfortable**. Under the floor the
  services are OOM-killed with an opaque exit, not a clear message.
- **`COMPUTE_TIER=cpu` is load-bearing.** The overlay sets it on the api and
  worker: it multiplies the default inference timeouts, stage leases, and the
  Celery visibility horizon so a healthy 4-hour CPU transcription is never
  reclaimed as a hung task mid-stage (the reclaim would duplicate work). See
  [timeouts-and-leases.md](timeouts-and-leases.md). Timeout env vars you set
  explicitly are never scaled.
- **Same contracts, same quality.** `/v1/*` request/response schemas and the
  quality gates are identical. whisper runs the same faster-whisper/CTranslate2
  engine (int8) on CPU; pyannote runs the same pipeline on torch-CPU; titanet
  runs on **ONNX Runtime** (`/healthz` reports `engine: onnxruntime`) under the
  **same embedding space id** (`titanet-large-v1`) — kept on a measured
  three-level parity gate against the CUDA engine, not on faith
  (see [gpu-contracts.md](gpu-contracts.md)).
- **No `HF_TOKEN` needed** — the diarization weights are vendored into the
  pyannote image (sha256-pinned from the `pyannote-models-v1` asset release).
- **Mixing tiers is fine.** The overlays are per-service compositions; an
  accelerated tier swaps individual services without touching the others —
  the ROCm tier below is exactly that (GPU whisper + CPU pyannote/titanet).

### Running on an AMD GPU (ROCm tier)

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d
```

The ROCm overlay is a **hybrid tier**: ASR (whisper) runs on the AMD GPU via
the `-rocm` image; diarization (pyannote) and speaker embedding (titanet) run
the `-cpu` images. Honest expectations and constraints:

- **What accelerates.** whisper keeps the exact faster-whisper/CTranslate2
  engine and code path — the `-rocm` image swaps only the CTranslate2 build
  (the 4.8.1 ROCm wheel, published as a GitHub release asset, absent from
  PyPI). Measured on an RDNA4 card (RX 9060 XT, gfx1200): **4.8× the CPU
  baseline** on the parity corpus clip. `/healthz` reports `device: "rocm"`.
- **Why pyannote and titanet stay CPU.** MIOpen convolutions fail on current
  AMD consumer GPUs (verified on RDNA4 in BOTH shipping torch-ROCm wheel
  lines — rocm6.4 and rocm7.2); everything conv-based dies at inference while
  GEMM-based engines (CTranslate2) work. titanet's CPU path is already far
  faster than real-time and does not need a GPU. Tracked upstream in
  issue #4 — the overlay swaps per service, so a working pyannote `-rocm`
  can slot in later without touching the rest.
- **Host requirements: amdgpu kernel driver only.** No host ROCm install, no
  container toolkit — the `-rocm` image carries its own ROCm runtime
  libraries. The overlay passes `/dev/kfd` + `/dev/dri` through and adds the
  gid that owns them (allocated per host, so it cannot be a baked default).
  The installer detects and records it; **manual setups must set it in
  `.env`**: `VOXINT_RENDER_GID=$(stat -c %g /dev/kfd)` — a wrong gid shows
  up as the whisper service failing to open the GPU, not as a clear
  permission error. Do **not** set `HSA_OVERRIDE_GFX_VERSION` for
  natively supported GPUs — it corrupts kernel selection.
- **`COMPUTE_TIER=rocm`** scales timeouts/leases for the still-CPU stages
  (GPU-class timing for ASR; see [timeouts-and-leases.md](timeouts-and-leases.md)).
- **No `HF_TOKEN` needed**, same as every tier — the diarization weights are
  vendored into the pyannote image.
- **amd64 only** (the CT2 ROCm wheels are x86_64), and CI builds this image
  without a GPU — the real-GPU gate runs on maintainer AMD hardware before a
  release.

### Running on Apple Silicon (metal tier)

```bash
./scripts/install.sh                 # choose [M] — starts the Docker core stack
./scripts/metal/voxint-metal.sh setup   # venvs + sha-verified weights (~3.2 GB)
./scripts/metal/voxint-metal.sh up      # native services under launchd
./scripts/metal/voxint-metal.sh status  # whisper cpu / pyannote mps / titanet cpu
```

Docker Desktop on macOS has no GPU passthrough, so the Docker CPU tier
cannot touch the Apple GPU. The metal tier splits the deployment instead:
the core stack (postgres/redis/api/worker/beat) stays in Docker via the
`compose.metal.yaml` rewiring overlay, and the three model services run
**natively on the host**, bound to 127.0.0.1 and supervised by launchd
(`KeepAlive` restarts crashes — the native analogue of
`restart: unless-stopped`). api/worker reach them through
`host.docker.internal`, which is Docker-Desktop-specific — `voxint-metal.sh
doctor` verifies the loopback path from the worker container. Honest
expectations:

- **What accelerates: diarization only (v1).** pyannote runs on the Apple
  GPU via torch-MPS — the Phase 0 spike measured warm MPS diarization about
  **5× native-CPU speed** on an M1 Pro with decision outputs identical to
  CPU. Whisper runs CTranslate2 on the host CPU (a native ASR Metal engine
  is a tracked follow-up), so end-to-end runs stay transcribe-bound —
  expect roughly 1.5–1.8× media duration overall against ~2.5× for the
  Docker CPU tier on the same hardware (Gate M confirms per chip). titanet
  runs the ONNX CPU EP — already far faster than real-time.
- **No silent device fallback.** The launcher forces `DIARIZER_DEVICE=mps`:
  if MPS is missing or fails the tensor-op sanity probe, pyannote refuses
  to start rather than quietly landing on CPU (`logs pyannote` shows why;
  `VOXINT_METAL_DIARIZER_DEVICE=cpu` overrides deliberately).
  `TITANET_ORT_PROVIDERS=CoreMLExecutionProvider` enables the CoreML
  experiment for titanet — requested providers must be verifiably active or
  the service fails at load.
- **Weights are fetched, sha-verified, never trusted blind**: the same
  release assets and provenance sha256s the images bake
  (`pyannote-models-v1`, `titanet-onnx-v1`), plus whisper large-v2 at the
  same pinned HF revision as the images with a local drift-detection
  manifest. No HF account or token.
- **Version skew is visible, not prevented**: native services run your
  working tree; the core runs pinned images. `voxint-metal.sh status`
  prints both (`git describe` vs `VOXINT_IMAGE_TAG`) — pair a tagged tree
  with the matching image tag for supported runs.
- **`COMPUTE_TIER=metal`** keeps GPU-class timing
  ([timeouts-and-leases.md](timeouts-and-leases.md)); note v1 runs whisper
  on CPU under those budgets — very long recordings on slow Macs may need
  the CPU-tier profile instead until the post-measurement pass settles a
  metal factor.
- **Memory budget (16 GB Macs)**: whisper int8 ~4 GB + pyannote/MPS ~3 GB +
  titanet ~1 GB run natively; cap the Docker Desktop VM around 4 GB — it
  only runs the core stack.
- `voxint-metal.sh doctor` checks the lot: weights vs provenance, vendored
  config params, MEDIA_ROOT agreement with `.env` (physically resolved),
  port collisions (a leftover CPU-tier stack on 8021/8022/8024 is the
  classic one), the MPS probe, ORT providers, and worker→host loopback.

### Schema migrations

A one-shot `migrate` service runs `alembic upgrade head` after Postgres is
healthy; the API and worker only start once it exits successfully. `migrate`
showing `Exited (0)` in `docker compose ps -a` is the **success** state, not
a crash. It re-runs on every `docker compose up` and no-ops when the schema
is already at head — which is how you apply new revisions after pulling a new
image: `docker compose up -d` again. Note this is startup ordering, not a
zero-downtime upgrade protocol: an `up` that doesn't recreate an
already-running API/worker won't restart them around the migration.

Troubleshooting:

```bash
docker compose logs migrate            # why did it fail?
docker compose run --rm migrate        # run the migration step by hand
```

## Operating the pipeline

Media lives under `MEDIA_ROOT` (compose mounts `./media` by default). The
CLI runs inside the api/worker image or on a bare host with the same env:

```bash
docker compose exec api voxint submit path/to/file.mp3   # local path relative to MEDIA_ROOT; prints run id
docker compose exec api voxint fetch <url>               # yt-dlp URL ingestion (reads the URL from stdin if omitted)
docker compose exec api voxint status <run-id>           # run state + per-stage attempt ledger
docker compose exec api voxint requeue <run-id>          # re-enter a FAILED run at its failed stage
docker compose exec api voxint list                      # recent runs, newest first (--status, --limit, --json)
docker compose exec api voxint export <run-id> --format srt   # export a transcript (see below)
docker compose exec api voxint doctor                    # read-only preflight for every dependency
docker compose exec api voxint stats                     # aggregate health/throughput (--since, --json)
docker compose exec api voxint watch <run-id>            # follow a run until it stops (--interval, --timeout)
```

`submit` records the media item, creates a run, and enqueues it for the
worker — the CLI never executes a stage itself. `fetch` does the same for a
remote URL, recording it as `MediaItem.source_url`; the worker's ACQUIRE stage
downloads it (see below). `status` shows the run's current state plus every
stage attempt with its error, straight from the persisted ledger. `list`
enumerates runs (the same query as the `/runs` page); `--json` prints a machine-
readable array.

**`voxint doctor`** probes every dependency without changing anything: Postgres
(`SELECT 1`), Redis (`PING`), and each model service's `/healthz` (reporting its
compute `device`, e.g. `rocm`/`cpu`) are **hard** checks — the command exits
non-zero if any is down. The Hugging Face token (`HF_TOKEN`, validated via
whoami) and the LLM endpoint (only when `LLM_ENABLED`) are **advisory**: reported
but never failing the exit code, because the default install needs neither. No
credentials, tokens, or connection URLs are printed.

**`voxint watch <run-id>`** follows a run until it stops advancing, printing a
live status line to **stderr** (so the run id stays clean on stdout). It exits
`0` completed, `1` failed/cancelled, `2` on a missing run, `3` awaiting
adjudication (the automated stages finished and a human ruling is needed — the
state machine can still resume it, so it is not "success"), and `124` on
timeout. `--interval` (default 2s) and `--timeout` (default 3600s) tune the
poll; each poll opens a fresh session so it observes the worker's commits.
**`voxint submit --wait`** composes submit + watch: it enqueues, prints the run
id, then follows the new run with the same loop and exit codes.

### Metrics & monitoring

**`voxint stats`** prints an aggregate, read-only snapshot: run counts by status,
failed stage *attempts* per stage, average per-stage duration over finished
attempts, roster size, and runs created within a window. `--since` accepts a
relative span (`24h`, `7d`) or an ISO-8601 datetime (default 24h); `--json`
emits a stable object for scripting.

```bash
docker compose exec api voxint stats --since 7d          # human table
docker compose exec -T api voxint stats --json           # machine-readable
```

The same aggregates are exposed for Prometheus at **`GET /metrics`** (text
exposition format 0.0.4). It sits on the authenticated router, so scrape it with
Basic Auth — that keeps the "everything but `/healthz` authenticates" invariant
with no extra flag or token. Every `RunStatus`/`Stage` series is zero-filled so a
series never vanishes between scrapes; the one windowed gauge names its window
(`voxint_runs_created_24h`). A scrape config on the monitoring host:

```yaml
scrape_configs:
  - job_name: voxint
    metrics_path: /metrics
    basic_auth:
      username: ${VOXINT_USER}
      password: ${VOXINT_PASSWORD}
    static_configs:
      - targets: ["voxint-host:8090"]   # the operator's voxint API address
```

Exposed series (all gauges — recomputed from the database per scrape, so none
carry a counter-style `_total` suffix): `voxint_runs{status}`,
`voxint_stage_failures{stage}`, `voxint_stage_duration_seconds{stage}` with a
companion `voxint_stage_duration_attempts{stage}` (so "no finished attempts" is
distinguishable from a genuinely 0-second average), `voxint_roster_speakers`, and
`voxint_runs_created_24h`.

### Exporting transcripts

Every run's speaker-attributed transcript exports in five formats, from the CLI
or over HTTP. Both paths share the same formatters, so a downloaded file and a
piped export are byte-identical.

```bash
docker compose exec -T api voxint export <run-id> --format srt   > out.srt
docker compose exec -T api voxint export <run-id> --format vtt   > out.vtt
docker compose exec -T api voxint export <run-id> --format json  > out.json
docker compose exec -T api voxint export <run-id> --format rttm  > out.rttm
docker compose exec -T api voxint export <run-id> --format txt   > out.txt
```

- `--format` — `srt` (SubRip), `vtt` (WebVTT), `json` (array of
  `{start_seconds, end_seconds, speaker, text}`), `rttm` (NIST diarization
  format), or `txt` (bracketed plain text). Default `txt`.
- `--text raw|enhanced` — which transcript variant to render (default
  `enhanced`, the LLM-cleaned text; `raw` is the immutable ASR output). Ignored
  for `rttm`, which carries raw diarization labels, not attributed text.
- `-o PATH` — write to a file instead of stdout (refuses to overwrite an
  existing file unless `--force`).

The same exports are available over HTTP at
`GET /review/{run_id}/export.{txt,srt,vtt,json,rttm}` (add `?text=raw` for the
raw variant). RTTM uses the run's UUID as the file id and the raw diarization
labels (`SPEAKER_00` …), so it round-trips against diarization scoring tools —
it deliberately does **not** substitute adjudicated speaker names.

### The browser console

The same API serves a browser console (HTTP Basic, `VOXINT_USER` /
`VOXINT_PASSWORD`) for operators who prefer not to shell into a container:

- **`GET /runs`** — an execution-history browser: newest-first, keyset-paged
  (`RUNS_PAGE_SIZE`, default 50), with **orthogonal** filters `status=` and
  `review=needed|resolved|claimed`. **`GET /runs/{id}`** shows the run detail and
  the per-stage attempt ledger (the same data as `voxint status`), with
  transcript and audio links when present.
- **`POST /submit`** — a bounded **file upload**. `UPLOAD_MAX_BYTES` (default
  5 GiB) is enforced *while streaming* (never a single unbounded read); the file
  lands under a server-issued, uuid-namespaced `incoming/{submission_id}/…` path,
  so re-uploading a name yields a distinct immutable media item and never
  overwrites history. A hidden `submission_id` makes form replay idempotent.
- **`POST /fetch`** — the browser equivalent of `voxint fetch` (URL ingestion).
- **`POST /runs/{id}/requeue`** — an exact-revision (CAS) requeue of a FAILED run,
  the browser equivalent of `voxint requeue` (covers failed downloads).

The console is deliberately **append-only**: there is no delete, no cancel, and
no speaker-roster editing from these pages (roster changes happen only through
adjudication). The pipeline-state surface (`/runs*`) and the adjudication surface
(`/review*`) stay separate.

**Broker-degraded submission.** `/submit`, `/fetch`, and `/runs/{id}/requeue`
commit the durable run *before* publishing the Celery task. If Redis is down at
that moment the mutation still succeeds — the run stays `QUEUED` (never `FAILED`)
with a clear linked note, and the recovery sweep re-enqueues it once the broker
returns. Read pages (`/runs*`) render from Postgres only and never touch Redis.

### Failure lanes and recovery

Failures split into two lanes (see architecture.md):

- **Transient** (service outage, timeouts) — retried automatically with
  exponential backoff up to `STAGE_MAX_ATTEMPTS`; nothing to do.
- **Deterministic** (`inference_failed`, protocol violations, bad media) —
  the run stays FAILED until a human decides; `voxint requeue` is the
  explicit override after fixing the cause.

The compose stack's dedicated `beat` service schedules
`voxint.recovery_sweep` (every `RECOVERY_SWEEP_SECONDS`; the worker executes
it), which reclaims runs whose stage lease expired (worker crash, node loss)
and re-enqueues QUEUED runs whose broker task evaporated. Crash recovery is
therefore automatic **as long as `beat` is running** — a bare-host deployment
without a beat process has no automatic recovery; a run is then stranded
until a beat/sweep runs, not just on deterministic failures.

### URL ingestion & egress security

`voxint fetch <url>` / `POST /fetch` download a URL with yt-dlp on the worker
(the ACQUIRE stage). Toggle the capability with `YTDLP_ENABLED` (default on); when
off, the fetch route/CLI/form refuse and no row is created (an already-queued URL
run still completes). yt-dlp is always run with `--proxy`: `YTDLP_PROXY` when set,
otherwise an empty value meaning explicit direct egress — so an ambient
`HTTP(S)_PROXY` in the worker env is **ignored** (set `YTDLP_PROXY` to route
through a proxy on purpose). `YTDLP_COOKIES_FILE` is wired only when set. Both are
treated as **credentials** — never surfaced in errors, so keep them out of logs
and shared configs.

Voxint applies two SSRF gates (string-level at submit, host re-resolution in the
worker before download; see architecture.md) that reject non-public **resolved**
addresses. They are **not** a sandbox: yt-dlp re-resolves independently and its
generic extractor follows redirects, so a rebind-after-check, an HTTP redirect to
a private address, or an extractor-constructed private URL is **not** covered.
**For untrusted-URL ingestion, run the worker with restricted egress** — no route
to RFC1918 / link-local / `169.254.169.254` (egress firewall or dedicated egress
network). A blocked/refused download is a clean FAILED @ acquire the operator
**Requeues**; it never loops.

The mutation forms that require a CSRF token are `POST /submit`, `/fetch`,
`/runs/{id}/requeue`, and `POST /review/{id}/claim` (claiming mints the run's claim
token, so it has none of its own to gate a forged POST). Set `CSRF_SECRET` to a
persistent random value
(`python -c "import secrets; print(secrets.token_urlsafe(32))"`) — otherwise a
random per-process secret is used, which invalidates open forms on restart and
mismatches across multiple workers.

## Adjudication workflow

The review console is served by the API at `http://127.0.0.1:8080/` (or your
`API_PORT` override; basic auth, `VOXINT_USER`/`VOXINT_PASSWORD` — single
reviewer credential). On a **fresh install** the onboarding gate redirects every
authenticated page to the first-run setup wizard (`/setup`) until setup is
finished — so `/review` below becomes reachable only after onboarding completes
(see [onboarding.md](onboarding.md)):

1. **Queue** (`/review`) — runs that finished matching and await human
   review.
2. **Claim** — claiming a run gives you an exclusive slot for
   `REVIEW_CLAIM_TTL_SECONDS` (default 30 min); a closed tab self-releases
   when the TTL lapses, so the queue never dams.
3. **Workbench** (`/review/{run_id}`) — per-label transcript previews and
   audio playback; record a **decision** per diarization label (confirm /
   correct / reject the machine proposal) or **enroll** a label's audio as a
   new speaker in the roster. Human rulings are an immutable ledger kept
   separate from machine proposals; adjudication precedence is defined in
   quality-gates.md.
4. **Release / export** — release the claim for the next reviewer, or export
   the speaker-attributed transcript (`/review/{run_id}/export.{txt,srt,vtt,
   json,rttm}`, or `voxint export` — see "Exporting transcripts" above).

## HTTP endpoints

Every route but `/healthz` sits behind HTTP Basic; the core mutation forms
(`POST /submit`, `/fetch`, `/runs/{id}/requeue`, `POST /review/{id}/claim`, and
the wizard/settings forms with their own `CSRF_SETUP` / `CSRF_SETTINGS` tokens)
additionally require a CSRF token (see above), and the remaining review-workbench
mutations are gated by their per-run claim token.

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness (no DB access — schema readiness is the migrate gate's job) |
| `GET /metrics` | Prometheus text exposition (aggregate gauges; authenticated — scrape with `basic_auth`) |
| `GET /runs` | Execution-history browser (keyset-paged; `status=` / `review=` filters) |
| `GET /runs/{run_id}` | Run detail + per-stage attempt ledger |
| `GET /runs/{run_id}/transcript?text=raw\|enhanced` | Resolver-attributed transcript (HTML) |
| `POST /submit` | Bounded browser file upload → immutable uuid-namespaced media item |
| `POST /fetch` | yt-dlp URL ingestion (create media item + run, enqueue) |
| `POST /runs/{run_id}/requeue` | Exact-revision (CAS) requeue of a FAILED run |
| `GET /review` | Review queue |
| `POST /review/{run_id}/claim` · `/release` | Claim / release an exclusive review slot |
| `GET /review/{run_id}` | Adjudication workbench |
| `POST /review/{run_id}/labels/{label}/decision` | Record a human ruling for a label |
| `POST /review/{run_id}/labels/{label}/enroll` | Enroll a label's audio as a roster speaker |
| `GET /review/{run_id}/export.{txt,srt,vtt,json}?text=raw\|enhanced` | Speaker-attributed transcript export (plain text, SubRip, WebVTT, JSON) |
| `GET /review/{run_id}/export.rttm` | Diarization RTTM (raw labels, run-UUID file id) |
| `GET /media/{run_id}` | Gated media serving (Range-aware) for the workbench player |
| `GET /setup` · `POST /setup/{media,scan,vocabulary,llm,finish}` | First-run setup wizard; held by the onboarding gate until finished (own `CSRF_SETUP` token) |
| `GET /settings` | Post-onboarding settings: re-run the wizard, start/replay/complete the tutorial |
| `POST /settings/tutorial/{complete,replay}` | Complete / non-destructively replay the guided tutorial (own `CSRF_SETTINGS` token) |

## Backup

State worth backing up: the Postgres volume (`pgdata` — runs, transcripts,
roster, decision ledger) and your media tree (`MEDIA_ROOT`). Redis holds only
in-flight queue state; losing it is recoverable — the recovery sweep
re-enqueues interrupted runs, and `voxint requeue` covers FAILED ones.
Dump the database over the published port (credentials default to
`voxint`/`voxint`/db `voxint` unless you set `POSTGRES_USER` /
`POSTGRES_PASSWORD` / `POSTGRES_DB` in the environment):

```bash
pg_dump -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" -U voxint voxint > voxint.sql
```
