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
NVIDIA GPU? See [the CPU tier](#running-without-an-nvidia-gpu-cpu-tier) below.
It is slower, but runs anywhere, including arm64/Apple Silicon.

For a first run, `./scripts/install.sh` is the recommended path. It renders
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
`REDIS_PORT`, `API_PORT`); container-internal ports never change.

### Release images vs. building from source

The default compose files run the **pinned release images** from GHCR
(`ghcr.io/bengizmo/voxint*`), even on a `main` checkout. `VOXINT_IMAGE_TAG`
in `.env` overrides the pin. To run the code you actually checked out, layer
the build overlays (build first, then up; services that don't own the build
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
`image:` tag: concurrent BuildKit writers race on it ("already exists").

### Offline / air-gapped hosts

Model weights are baked into the service images, and the whisper images set
`HF_HUB_OFFLINE=1` plus a sha-pinned `WHISPER_REVISION`, so **no model service
makes an outbound call at startup**, and the stack comes up on a host with no
internet access. (Earlier whisper images made an unadvertised Hugging Face
revision check at startup, which stalled on restricted networks.)
Network access is still required for URL ingestion (`yt-dlp`) and, if you
opt into the online diarizer path, for `DIARIZER_MODEL_NAME`/`HF_TOKEN`
downloads. On the metal tier, `voxint-metal.sh setup` downloads the pinned
weights once; the running services are offline-clean the same way.

### Running without an NVIDIA GPU (CPU tier)

Every model service also ships a **`-cpu` image flavor**: multi-arch
(amd64 + arm64), no GPU, no NVIDIA container toolkit. This is the supported
path for Apple Silicon (via Docker Desktop) and plain CPU servers (AMD-GPU
boxes have the faster ROCm tier below):

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

What changes relative to the GPU overlay, and what doesn't:

- **Speed.** CPU inference is orders of magnitude slower than GPU: transcribing
  a multi-hour recording takes **hours**, not minutes. Fine for overnight or
  batch use, and correctness-identical, but not an interactive experience.
- **Host RAM floor: ≥ 8 GB.** The CPU tier holds the models in RAM instead of
  VRAM: whisper alone is ~4.8 GiB resident (large-v2 int8 + CTranslate2 arenas)
  and the tier idles around ~6 GiB total. Give the container host at least
  **8 GB *including the core stack*** (Postgres, Redis, api, worker share the
  same VM). On **Docker Desktop (macOS/Windows) that means the VM's memory
  limit**, not the physical machine's. **16 GB is comfortable.** Under the floor
  the services are OOM-killed with an opaque exit, not a clear message.
- **`COMPUTE_TIER=cpu` matters.** The overlay sets it on the api and worker,
  multiplying the default inference timeouts, stage leases, and the Celery
  visibility horizon so a healthy 4-hour CPU transcription is never reclaimed as
  a hung task mid-stage (the reclaim would duplicate work). See
  [timeouts-and-leases.md](timeouts-and-leases.md). Timeout env vars you set
  explicitly are never scaled.
- **Same contracts, same quality.** `/v1/*` request/response schemas and the
  quality gates are identical. whisper runs the same faster-whisper/CTranslate2
  engine (int8) on CPU, pyannote runs the same pipeline on torch-CPU, and
  titanet runs on **ONNX Runtime** (`/healthz` reports `engine: onnxruntime`)
  under the **same embedding space id** (`titanet-large-v1`). That id is held by
  a measured three-level parity gate against the CUDA engine
  (see [gpu-contracts.md](gpu-contracts.md)).
- **No `HF_TOKEN` needed.** The diarization weights are vendored into the
  pyannote image (sha256-pinned from the `pyannote-models-v1` asset release).
- **Mixing tiers is fine.** The overlays are per-service compositions; an
  accelerated tier swaps individual services without touching the others. The
  ROCm tier below is exactly that (GPU whisper + CPU pyannote/titanet).

### Running on an AMD GPU (ROCm tier)

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d
```

The ROCm overlay is a **hybrid tier**: ASR (whisper) runs on the AMD GPU via
the `-rocm` image; diarization (pyannote) and speaker embedding (titanet) run
the `-cpu` images. What that buys you, and its constraints:

- **What accelerates.** whisper keeps the exact faster-whisper/CTranslate2
  engine and code path; the `-rocm` image swaps only the CTranslate2 build
  (the 4.8.1 ROCm wheel, published as a GitHub release asset, absent from
  PyPI). Measured on an RDNA4 card (RX 9060 XT, gfx1200): **4.8× the CPU
  baseline** on the parity corpus clip. `/healthz` reports `device: "rocm"`.
- **Why pyannote and titanet stay CPU.** MIOpen convolutions fail on current
  AMD consumer GPUs (verified on RDNA4 in BOTH shipping torch-ROCm wheel
  lines, rocm6.4 and rocm7.2): everything conv-based dies at inference while
  GEMM-based engines like CTranslate2 work. titanet's CPU path is already far
  faster than real-time and does not need a GPU. Tracked upstream in
  issue #4. The overlay swaps per service, so a working pyannote `-rocm`
  can slot in later without touching the rest.
- **Host requirements: amdgpu kernel driver only.** No host ROCm install, no
  container toolkit; the `-rocm` image carries its own ROCm runtime
  libraries. The overlay passes `/dev/kfd` + `/dev/dri` through and adds the
  gid that owns them (allocated per host, so it cannot be a baked default).
  The installer detects and records it; **manual setups must set it in
  `.env`**: `VOXINT_RENDER_GID=$(stat -c %g /dev/kfd)`. A wrong gid shows
  up as the whisper service failing to open the GPU, not as a clear
  permission error. Do **not** set `HSA_OVERRIDE_GFX_VERSION` for
  natively supported GPUs; it corrupts kernel selection.
- **`COMPUTE_TIER=rocm`** scales timeouts/leases for the still-CPU stages
  (GPU-class timing for ASR; see [timeouts-and-leases.md](timeouts-and-leases.md)).
- **No `HF_TOKEN` needed**, same as every tier: the diarization weights are
  vendored into the pyannote image.
- **amd64 only** (the CT2 ROCm wheels are x86_64), and CI builds this image
  without a GPU; the real-GPU gate runs on maintainer AMD hardware before a
  release.

### Running on Apple Silicon (metal tier)

```bash
./scripts/install.sh                 # choose [M]; starts the Docker core stack
./scripts/metal/voxint-metal.sh setup   # venvs + sha-verified weights (~3.2 GB)
./scripts/metal/voxint-metal.sh up      # native services under launchd
./scripts/metal/voxint-metal.sh status  # whisper cpu / pyannote mps / titanet cpu
```

Docker Desktop on macOS has no GPU passthrough, so the Docker CPU tier
cannot touch the Apple GPU. The metal tier splits the deployment instead:
the core stack (postgres/redis/api/worker/beat) stays in Docker via the
`compose.metal.yaml` rewiring overlay, and the three model services run
**natively on the host**, bound to 127.0.0.1 and supervised by launchd
(`KeepAlive` restarts crashes, the native analogue of
`restart: unless-stopped`). api/worker reach them through
`host.docker.internal`, which is Docker-Desktop-specific; `voxint-metal.sh
doctor` verifies the loopback path from the worker container. What to
expect:

- **What accelerates: diarization only (v1).** pyannote runs on the Apple
  GPU via torch-MPS. The Phase 0 spike measured warm MPS diarization about
  **5× native-CPU speed** on an M1 Pro, with decision outputs identical to
  CPU. Whisper runs CTranslate2 on the host CPU (a native ASR Metal engine
  is a tracked follow-up), so end-to-end runs stay transcribe-bound: expect
  roughly 1.5–1.8× media duration overall against ~2.5× for the Docker CPU
  tier on the same hardware (Gate M confirms per chip). titanet runs the ONNX
  CPU EP, already far faster than real-time.
- **No silent device fallback.** The launcher forces `DIARIZER_DEVICE=mps`:
  if MPS is missing or fails the tensor-op sanity probe, pyannote refuses
  to start rather than quietly landing on CPU (`logs pyannote` shows why;
  `VOXINT_METAL_DIARIZER_DEVICE=cpu` overrides deliberately).
  `TITANET_ORT_PROVIDERS=CoreMLExecutionProvider` enables the CoreML
  experiment for titanet; requested providers must be verifiably active or
  the service fails at load.
- **Weights are fetched, sha-verified, never trusted blind**: the same
  release assets and provenance sha256s the images bake
  (`pyannote-models-v1`, `titanet-onnx-v1`), plus whisper large-v2 at the
  same pinned HF revision as the images with a local drift-detection
  manifest. No HF account or token.
- **Version skew is visible, not prevented**: native services run your
  working tree; the core runs pinned images. `voxint-metal.sh status`
  prints both (`git describe` vs `VOXINT_IMAGE_TAG`); pair a tagged tree
  with the matching image tag for supported runs.
- **`COMPUTE_TIER=metal`** keeps GPU-class timing
  ([timeouts-and-leases.md](timeouts-and-leases.md)). Gate M clocked v1's CPU
  whisper at 0.38–0.45× RT, well inside those budgets, so the choice is
  measured rather than a placeholder. Unusually slow Macs can still override
  individual timeouts via env (never silently scaled).
- **Memory budget (16 GB Macs)**: whisper int8 ~4 GB + pyannote/MPS ~3 GB +
  titanet ~1 GB run natively; cap the Docker Desktop VM around 4 GB, since it
  only runs the core stack.
- **Logs rotate themselves**: launchd never rotates `StandardOutPath`, and
  `KeepAlive` keeps services up for months, so `up` installs a fourth
  launchd job (`com.voxint.metal.logrotate`, daily) that copy-truncates any
  `~/.voxint-metal/logs/*.log` over 50 MB to a timestamped archive, keeping
  the newest 5 (`VOXINT_METAL_LOG_MAX_MB` / `VOXINT_METAL_LOG_ARCHIVES`
  override; `voxint-metal.sh rotate-logs` runs one pass by hand).
  Copy-truncate, not rename: the running service keeps its open fd, so a
  renamed log would just keep growing under the archive name.
- `voxint-metal.sh doctor` checks the lot: weights vs provenance, vendored
  config params, MEDIA_ROOT agreement with `.env` (physically resolved),
  port collisions (a leftover CPU-tier stack on 8021/8022/8024 is the
  classic one), the MPS probe, ORT providers, and worker→host loopback.

### Schema migrations

A one-shot `migrate` service runs `alembic upgrade head` after Postgres is
healthy; the API and worker only start once it exits successfully. `migrate`
showing `Exited (0)` in `docker compose ps -a` is the **success** state, not
a crash. It re-runs on every `docker compose up` and no-ops when the schema
is already at head. That is how you apply new revisions after pulling a new
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
worker; the CLI never executes a stage itself. `fetch` does the same for a
remote URL, recording it as `MediaItem.source_url`, and the worker's ACQUIRE
stage downloads it (see below). `status` shows the run's current state plus every
stage attempt with its error, straight from the persisted ledger. `list`
enumerates runs (the same query as the `/runs` page); `--json` prints a machine-
readable array.

`submit`, `fetch`, and `requeue` all commit the durable run **before**
publishing the broker task, and degrade cleanly if the broker (Redis) is down,
exactly like the HTTP API. The run id (or `requeued <id>`) is printed to
stdout *before* the publish, so a broker outage never costs you the id; the
publish failure is reported as a `warning:` on **stderr** and the command still
exits `0`, leaving the run `QUEUED` for the recovery sweep to re-enqueue once
the broker returns (see "Failure lanes and recovery"). A genuine bug in the
publish path still raises. With `submit --wait`, a deferred enqueue prints a
note that polling will wait for the sweep before the run advances.

**`voxint doctor`** probes every dependency without changing anything: Postgres
(`SELECT 1`), Redis (`PING`), and each model service's `/healthz` (reporting its
compute `device`, e.g. `rocm`/`cpu`) are **hard** checks, and the command exits
non-zero if any is down. The Hugging Face token (`HF_TOKEN`, validated via
whoami) and the LLM endpoint (only when `LLM_ENABLED`) are **advisory**: reported
but never failing the exit code, because the default install needs neither. No
credentials, tokens, or connection URLs are printed.

**`voxint watch <run-id>`** follows a run until it stops advancing, printing a
live status line to **stderr** (so the run id stays clean on stdout). It exits
`0` completed, `1` failed/cancelled, `2` on a missing run, `3` awaiting
adjudication (the automated stages finished and a human ruling is needed; the
state machine can still resume it, so it is not "success"), and `124` on
timeout. `--interval` (default 2s) and `--timeout` (default 3600s) tune the
poll; each poll opens a fresh session so it observes the worker's commits.
**`voxint submit --wait`** composes submit + watch: it enqueues, prints the run
id, then follows the new run with the same loop and exit codes.

### Domain packs (issue #11)

A **domain pack** supplies domain vocabulary, speaker name seeds, and LLM prompt
fragments (see `docs/domain-packs.md` for the manifest and what each field
shapes). The bundled `generic` pack is the zero-config default, so this is
optional. Two env knobs are the shipped operator surface:

```bash
# One default pack applied to every run:
DOMAIN_PACK_PATH=/data/voxint/packs/newsroom

# Or a library of named packs (one child folder each, resolved by manifest name):
DOMAIN_PACKS_DIR=/data/voxint/packs
```

With `DOMAIN_PACK_PATH` unset, only the bundled `generic` pack is used. Each run
**freezes** the pack it was submitted with (`pipeline_runs.domain_pack`, migration
0017), so editing a manifest on disk never changes a past run's transcription or
enrichment — a manifest change takes effect on the *next* run.

> Per-**folder** assignment (`{media_folder → pack_name}`) is editable in the
> console (issue #63): the setup wizard's media step and **Settings → Media
> folders** host a folder browser with a per-folder domain-pack picker. A
> per-**submission** pack override remains a backend-only capability. The default
> pack (`DOMAIN_PACK_PATH`) stays the installation-wide fallback.

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
Basic Auth, which keeps the "everything but `/healthz` authenticates" invariant
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

Exposed series (all gauges, recomputed from the database per scrape, so none
carry a counter-style `_total` suffix): `voxint_runs{status}`,
`voxint_stage_failures{stage}`, `voxint_stage_duration_seconds{stage}` with a
companion `voxint_stage_duration_attempts{stage}` (so "no finished attempts" is
distinguishable from a genuinely 0-second average), `voxint_roster_speakers`, and
`voxint_runs_created_24h`.

For a human at the console, the **Dashboard** page (`GET /dashboard`, first in the
top nav) renders the *same* aggregates as a read-only page: runs by status, the
review backlog, per-stage timing and failures, roster size, and runs created in
the window. It is authenticated like every non-`/healthz` page and shares the
`stats_query` data layer with `/metrics` and `voxint stats`, so the three surfaces
always agree. Stage names render in plain language ("Diarize & embed") while the
machine `/metrics`, JSON, and `voxint stats` outputs keep their raw identifiers.
The page auto-refreshes every 15 seconds (an htmx fragment poll, no external
assets). The throughput window is a **24h / 7d / 30d picker on the page**; the
same `?since=` query param still overrides it directly (any span/ISO-8601 syntax
`voxint stats --since` accepts), degrading to 24h if malformed.

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

- `--format`: `srt` (SubRip), `vtt` (WebVTT), `json` (array of
  `{start_seconds, end_seconds, speaker, text}`), `rttm` (NIST diarization
  format), or `txt` (bracketed plain text). Default `txt`.
- `--text raw|enhanced`: which transcript variant to render (default
  `enhanced`, the LLM-cleaned text; `raw` is the immutable ASR output). Ignored
  for `rttm`, which carries raw diarization labels, not attributed text.
- `-o PATH`: write to a file instead of stdout (refuses to overwrite an
  existing file unless `--force`).

The same exports are available over HTTP at
`GET /review/{run_id}/export.{txt,srt,vtt,json,rttm}` (add `?text=raw` for the
raw variant). RTTM uses the run's UUID as the file id and the raw diarization
labels (`SPEAKER_00` …), so it round-trips against diarization scoring tools,
and it deliberately does **not** substitute adjudicated speaker names.

### The browser console

The same API serves a browser console (HTTP Basic, `VOXINT_USER` /
`VOXINT_PASSWORD`) for operators who prefer not to shell into a container:

- **`GET /runs`**: an execution-history browser, newest-first, keyset-paged
  (`RUNS_PAGE_SIZE`, default 50), with **orthogonal** filters `status=` and
  `review=needed|resolved|claimed`. **`GET /runs/{id}`** shows the run detail and
  the per-stage attempt ledger (the same data as `voxint status`), with
  transcript and audio links when present.
- **`GET /review`**: the adjudication queue — completed runs with at least one
  voice still needing a human ruling. Each row shows a **friendly title**, the
  recording **duration** and **age**, and a **resolved-of-total** progress bar,
  so it is clear at a glance both what a recording is and how much is left to
  adjudicate. `?sort=` chooses the order: `oldest` (default, FIFO) or
  `unresolved` ("Most voices to resolve"). The **Review** button claims the run
  and opens the workbench.
- **`POST /submit`**: a bounded **file upload**. `UPLOAD_MAX_BYTES` (default
  5 GiB) is enforced *while streaming* (never a single unbounded read); the file
  lands under a server-issued, uuid-namespaced `incoming/{submission_id}/…` path,
  so re-uploading a name yields a distinct immutable media item and never
  overwrites history. A hidden `submission_id` makes form replay idempotent.
- **`POST /fetch`**: the browser equivalent of `voxint fetch` (URL ingestion).
- **`POST /runs/{id}/requeue`**: an exact-revision (CAS) requeue of a FAILED run,
  the browser equivalent of `voxint requeue` (covers failed downloads).
- **`POST /runs/{id}/cancel`**: an exact-revision (CAS) cancel of a *live* run
  (`QUEUED` / `RUNNING` / `AWAITING_ADJUDICATION`), from a button on the run
  detail page. Cancellation is **cooperative and pure DB state**: it drives the
  run to `CANCELLED` and publishes nothing. A `QUEUED` run cancelled before
  dispatch never starts; a `RUNNING` run's currently executing stage finishes
  first (cancel is not an immediate process kill), then no further stages run.
  Re-cancelling an already-cancelled run is an idempotent success, not an error.
  Cancelling leaves media and any partial results in place; **delete/archive is
  a separate action** (below).
- **`POST /runs/{id}/archive`** and **`POST /runs/{id}/unarchive`**: soft-archive
  a *terminal* run (`COMPLETED` / `FAILED` / `CANCELLED`), from the run detail
  page. Archiving stamps `pipeline_runs.archived_at` and **hides** the run from
  `/runs` and the `/review` queue while keeping **every row intact** (including
  the append-only adjudication ledger); un-archive reverses it. Archive
  is operator-visibility metadata: last-write-wins, orthogonal to `status`, no
  CAS/revision bump (like operator notes), and idempotent. A *live* run refuses
  archive (`409`, cancel it first), and an **archived run refuses requeue/claim**
  so a stale tab can't drive a hidden run back to live. `/runs` hides archived by
  default; `?archived=1` shows the archived-only view. Dashboard, `/metrics`, and
  `voxint stats` exclude archived runs from their counts.
- **`POST /runs/{id}/media/delete`**: **destructive**, terminal-only. Deletes
  only *this run's* derived audio (its `AudioArtifact` + `AudioChunk` rows and
  files) to reclaim disk; files are unlinked **after** the DB delete commits,
  path-confined under `MEDIA_ROOT`, and the operation is idempotent (an
  already-gone file is not an error). It **never** touches the original
  `MediaItem.source_path`: that file is shared by every run of the media item,
  so removing it is a separate, refcount-guarded action (a future slice). The
  evidence ledger (adjudication / transcript / diarization rows) is untouched.
  This is the **manual** counterpart to the scheduled media-retention GC (issue
  #15, below): the manual action deletes the rows and files on demand, while the
  GC sweep keeps the `AudioArtifact` row and stamps `reclaimed_at` for audit —
  use whichever fits, they compose (deleting a run already GC-reclaimed just
  finds its file already gone).

Beyond these, the console stays **append-only** for evidence: archive hides but
never deletes rows, media-delete only removes re-derivable audio files (never the
ledger), and there is no speaker-roster editing from these pages (roster changes
happen only through adjudication). The pipeline-state surface (`/runs*`) and the
adjudication surface (`/review*`) stay separate.

**Verify-and-advance transcript review** (`GET /review/{id}/transcript`, reached
from the claimed workbench). A keyboard-first loop for confirming the transcript
one segment at a time: **`v`** verifies the current segment and jumps to the next
unverified one (segments faster-whisper was uncertain about carry an "uncertain"
chip so they draw the eye first), **`e`** edits its words in an inline box (save
with `⌘/Ctrl+Enter`), **`n`** skips, **`p`** replays the segment's audio; a live
"N of M verified" readout tracks progress. Keys are typing-guarded (they never
fire while you are typing, and Space/scroll keys stay with the native player). A
correction is stored **beside** the immutable `raw_text` and clears that
segment's verified mark (edited text must be re-checked); reverting to the
pipeline wording removes it. All writes are claim-gated and carry the workbench's
claim token — the page never re-claims (a fresh claim would evict the workbench
tab). With JavaScript off the same page lists every segment, with a plain
**Verify** form on each one still unverified (inline editing needs the browser
island, stated plainly), and it renders read-only with a prompt to claim when
this tab does not hold the run's claim.

**Split a segment at a word boundary.** When faster-whisper merges two speakers'
words into one segment, the review page can cut it at a word. Press the **⎇ Split
at a word** button to enter split mode; the segment under the review cursor then
shows its individual words, and clicking a word cuts the segment *before* it (you
cannot cut before the first word). The cut is stored as an append-only boundary
on the immutable segment — the original `raw_text` and its word timings are never
altered — and the segment renders from then on as the derived child lines, each
inheriting the parent's speaker and review state. A split is only offered when the
segment's words reconcatenate exactly to its `raw_text` and its text has not been
materially enhanced; otherwise split mode reports plainly that the segment cannot
be split rather than guessing at boundaries. Splitting and inline editing are
**mutually exclusive**: a segment that has been split cannot then be edited (and a
segment with an operator correction cannot be split) — the box is disabled with a
short note, because a split's text is word-derived and a free-form edit would have
nowhere faithful to live. A segment can be cut once (into two children); splitting
an already-split segment into more parts is refused in this release, and there is
no un-split control — a mis-split is cleared by re-transcribing the run. Like
inline editing, splitting needs the browser island — with JavaScript off the
transcript still lists any already-derived child lines, but no new split can be
made.

**Reassign a split child to the right speaker.** A split's two halves start out
sharing the parent segment's single resolved speaker, which is rarely what you
want — the point of splitting a mis-merged segment is to give each half its own
speaker. After a split, each derived child line shows a small **speaker:**
dropdown listing your active roster; pick a speaker to reassign just that child,
or pick **↺ inherit (follow the segment)** to clear that child's own speaker so
it follows the segment again (a whole-segment reassignment if one exists,
otherwise the diarization label). The choice is scoped to that child's exact
word-range and stored as an append-only ruling on the immutable parent — it
survives a later whole-label decision and, because it is append-only, an
`inherit` reset tracks later rulings live rather than freezing a copy. A
word-range reassignment takes precedence over a whole-segment reassignment,
which takes precedence over the label. Only active roster identities are offered (a merged or archived
speaker cannot attract a new ruling); a run with no roster yet shows only the
inherit option. Like the rest of the review loop, the picker needs the browser
island and a held claim — with JavaScript off the child lines still render with
their resolved speakers, but there is no per-child picker. (Correcting a split
child's *text* per-range, and un-splitting an already-reassigned segment, are not
yet available; clear a mistaken reassignment with **↺ inherit**, or re-transcribe
to clear the split entirely.)

**Waveform strip (who spoke when).** The transcript pages (read-only and
review) draw a compact waveform under the audio player, tinted per speaker with
the same colors as the segment list — the colored regions come from the
diarization turns themselves, so overlapping speech shows a hatched marker and
diarized-but-untranscribed stretches still appear. Clicking the strip jumps to
that segment in the list (and plays it, when seeking is trusted); the review
page also underlines the segment under the review cursor. The amplitude data is
computed once per run on first view (a second or two for long recordings) and
cached; the strip keeps rendering as a static who-spoke-when map even after the
run's processed audio has been reclaimed to free disk space — though when
seeking is disabled (untrusted timeline, missing media) a strip click only
selects the segment, never seeks, and no playhead is shown. If the amplitude
data cannot be computed (e.g. the media file is gone and nothing was cached)
the strip simply does not appear; the transcript list is unaffected.

**Broker-degraded submission.** `/submit`, `/fetch`, and `/runs/{id}/requeue`
commit the durable run *before* publishing the Celery task. If Redis is down at
that moment the mutation still succeeds: the run stays `QUEUED` (never `FAILED`)
with a clear linked note, and the recovery sweep re-enqueues it once the broker
returns. Read pages (`/runs*`) render from Postgres only and never touch Redis.

### Failure lanes and recovery

Failures split into two lanes (see architecture.md):

- **Transient** (service outage, timeouts): retried automatically with
  exponential backoff up to `STAGE_MAX_ATTEMPTS`; nothing to do.
- **Deterministic** (`inference_failed`, protocol violations, bad media):
  the run stays FAILED until a human decides, and `voxint requeue` is the
  explicit override after fixing the cause.

The compose stack's dedicated `beat` service schedules
`voxint.recovery_sweep` (every `RECOVERY_SWEEP_SECONDS`; the worker executes
it), which reclaims runs whose stage lease expired (worker crash, node loss)
and re-enqueues QUEUED runs whose broker task evaporated. Crash recovery is
therefore automatic **as long as `beat` is running**. A bare-host deployment
without a beat process has no automatic recovery; a run is then stranded
until a beat/sweep runs, not just on deterministic failures.

The same `beat` schedule carries the opt-in sweeps: the media-retention GC
(when `MEDIA_RETENTION_ENABLED`), the webhook delivery sweep (when
`NOTIFY_ENABLED`), and the **watch-folder ingest sweep** `voxint.watch_sweep`
(issue #60). The watch sweep is registered unconditionally at
`WATCH_FOLDER_SWEEP_SECONDS` but re-checks its **effective** gate each run —
the env `WATCH_FOLDER_ENABLED` default overridden by the runtime
`app_settings.watch_folder_enabled` toggle (Settings → Media folders) — so a
disabled installation only pays one DB read per sweep and enabling it needs no
restart. When on, it walks the operator's registered `media_folders`, submits
each new file (skipping ones already ingested), waits out
`WATCH_FOLDER_SETTLE_SECONDS` so a file still being copied in is not read
mid-write, and records a one-line status summary shown in Settings. Like the
other sweeps it needs `beat` running; a bare-host deployment without a beat
process never ingests automatically.

### URL ingestion & egress security

`voxint fetch <url>` / `POST /fetch` download a URL with yt-dlp on the worker
(the ACQUIRE stage). Toggle the capability with `YTDLP_ENABLED` (default on); when
off, the fetch route/CLI/form refuse and no row is created (an already-queued URL
run still completes). yt-dlp is always run with `--proxy`: `YTDLP_PROXY` when set,
otherwise an empty value meaning explicit direct egress, so an ambient
`HTTP(S)_PROXY` in the worker env is **ignored** (set `YTDLP_PROXY` to route
through a proxy on purpose). `YTDLP_COOKIES_FILE` is wired only when set. Both are
treated as **credentials**, never surfaced in errors, so keep them out of logs
and shared configs.

Voxint applies two SSRF gates (string-level at submit, host re-resolution in the
worker before download; see architecture.md) that reject non-public **resolved**
addresses. They are **not** a sandbox: yt-dlp re-resolves independently and its
generic extractor follows redirects, so a rebind-after-check, an HTTP redirect to
a private address, or an extractor-constructed private URL is **not** covered.
**For untrusted-URL ingestion, run the worker with restricted egress**: no route
to RFC1918 / link-local / `169.254.169.254` (egress firewall or dedicated egress
network). A blocked/refused download is a clean FAILED @ acquire the operator
**Requeues**; it never loops.

#### Restricted URL-download overlay (opt-in, recommended for untrusted URLs)

The `compose.ytdlp-egress.yaml` overlay ships this restricted egress so you don't
have to build it by hand. Stack it after the base file (and after any GPU/CPU
tier):

```bash
docker compose -f compose.yaml -f compose.ytdlp-egress.yaml up -d
# with a GPU tier, for example:
docker compose -f compose.yaml -f compose.gpu.yaml -f compose.ytdlp-egress.yaml up -d
```

It runs a small filtering forward proxy (the same Voxint image, no extra
download) on an internal network, points yt-dlp's always-passed `--proxy` at it
(`YTDLP_PROXY`), and applies the **same** public-address policy the worker gate
uses (`voxint.media.netcheck.ip_is_public`) **at the connection boundary**,
connecting only to the vetted public IP. Because the proxy — not yt-dlp — makes
the outbound connection, this closes the rebind window and refuses redirect /
extractor destinations that resolve to a private address. The worker keeps its
normal network, so Postgres, Redis, the model services, and the LLM/research/
notify endpoints are unaffected.

**What it does and does not cover** (state it honestly): it constrains the
HTTP(S) traffic that honours yt-dlp's `--proxy`, *including* its redirects and
extractor-constructed URLs. It is **not** a kernel-level route removal and **not**
a sandbox — a helper yt-dlp spawns that ignores the proxy (e.g. ffmpeg for some
streaming formats), or the worker container's own routable network, is beyond it.
For that last mile, additionally deny the worker a route to RFC1918 / link-local /
`169.254.169.254` with a host-level egress firewall. A refused destination is the
same clean FAILED @ acquire you **Requeue**.

### Web research retrieval (issue #39; off by default)

`voxint research search "<query>"` and `voxint research read <url>` are the
operator surface of the controlled retrieval capability (`voxint.research`,
the library the future research loop, issue #40, will consume). Everything is
**off** until `VOXINT_WEB_RESEARCH=true`, and the capability is independent of
`LLM_ENABLED` in both directions.

Minimal enablement (SearxNG is the built-in provider; a LAN/private instance
is the expected setup and explicitly allowed, and every *result* URL is still held
to the full public-address egress policy):

```bash
VOXINT_WEB_RESEARCH=true
WEB_SEARCH_BASE_URL=http://<your-searxng-host>:8888   # must serve format=json
# WEB_SEARCH_API_KEY=...   # only if your instance sits behind an auth proxy
```

These four settings — the web-research master toggle, the enrichment-producer
toggle (below), the endpoint, and the API key — are also editable from
**Settings → Sources & research** (issue #76): a saved value wins over the
environment and takes effect on the next job with no restart. The env values above
are the fallback when the settings row is blank. The `WEB_READ_*` /
`WEB_SEARCH_MAX_RESULTS` / timeout caps stay env-only (they are per-job budget
knobs, deliberately not runtime-editable).

Verify the egress policy by hand after enabling:

```bash
voxint research search "test query"          # normalized title/url/snippet list
voxint research read https://example.com/    # extracted text
voxint research read http://169.254.169.254/ # must refuse: invalid_input
```

Unlike yt-dlp ingestion, `read_url` owns its connections, so redirects and DNS
rebinding ARE covered on this path: every redirect hop is re-validated and the
connection is pinned to the vetted address (details in architecture.md, "Web
research egress"). Caps (`WEB_READ_MAX_BYTES`, `WEB_READ_MAX_REDIRECTS`,
`WEB_READ_TOTAL_SECONDS`, `WEB_READ_MAX_TEXT_CHARS`,
`WEB_SEARCH_MAX_RESULTS`, timeouts) are all env-tunable; see `.env.example`.
Compressed responses are refused by design (identity-only), and every fetch
writes one attribution log line (feature, reason, host, verdict, bytes,
duration; never the URL or query).

### Web-research speaker enrichment (issue #40; off by default)

With retrieval verified, the `web_researcher` producer adds the research loop
on top. It needs **all three** gates (refused at startup otherwise, and
re-checked by the worker before any queued job runs):

```bash
ENRICHMENT_WEB_RESEARCH_ENABLED=true
VOXINT_WEB_RESEARCH=true      # + WEB_SEARCH_BASE_URL, as above
LLM_ENABLED=true              # + LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
```

> The LLM endpoint, model, and API key can also be set in the UI (setup wizard /
> Settings, issue #10) — a UI-saved value wins over these env vars, and the key is
> resolved live per job. The **`LLM_ENABLED` gate above is env-only for enrichment**:
> the UI enhancement toggle governs per-run transcript enhancement, not the
> enrichment producers, so web research and run assets still require
> `LLM_ENABLED=true` in the environment.

Operate it from the speaker's card on `/speakers`: the "Web research" block
shows the exact budget a job will run under (`RESEARCH_MAX_SEARCHES=3`,
`RESEARCH_MAX_READS=5`, `RESEARCH_MAX_ROUNDS=5`,
`RESEARCH_DEADLINE_SECONDS=300`; env-tunable, never raisable per job), an
optional note passed to the researcher as seed context, and a Start button.
While a job runs the block polls every 3 s with live search/read/round
counters and a Cancel control (cooperative: the loop stops before its next
round, and an in-flight fetch finishes first). Jobs run on the Celery worker;
`voxint research speaker <speaker-uuid> [--note …]` runs one inline for
headless use.

Findings land as **drafts** under the same block (field, value, source URL
with fetch time, and the verbatim supporting quote), and nothing touches the
speaker until you accept a draft, field by field. Accepted claims remain a
reviewable collection (never a canonical profile, never identity; see
`docs/quality-gates.md`). A job that ends `found=false` records an
authoritative "looked, found nothing" generation that retires prior
proposals; a **failed or cancelled** job records nothing. There are no
automatic retries: a failed job shows its error on the card, so fix the cause
(provider down, LLM endpoint unreachable, model too weak to follow the
protocol) and start a fresh job. If the broker is down at start, the job
stays queued; cancel it and retry once the broker is back.

### Run-level assets (issue #41; off by default)

The run detail page can carry three machine-generated assets: a **summary**,
a **topic list**, and **entity mentions** (grounded spans referencing the
transcript segments they occur in). Generation is on demand via the
configured enhancement LLM and needs (refused at startup otherwise, and
re-checked by the worker before any queued job runs):

```bash
ENRICHMENT_RUN_ASSETS_ENABLED=true
LLM_ENABLED=true              # + LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
```

Operate it from the run's detail page: "Generate all" or per-kind
Generate/Regenerate buttons; while a job runs the block polls every 3 s with
a Cancel control. The three kinds succeed and fail independently: one
failing shows its error on its own card and never blocks or retires the
others. Every asset shows when it was generated, by which model, and whether
it is **stale** (the transcript, metadata, or notes changed since, e.g.
enhancement rewrote segment text); regeneration supersedes, never edits.
Long transcripts are head+tail truncated to `RUN_ASSETS_MAX_INPUT_CHARS`
(the card says so). Assets ride the `/runs/{id}/export.json` envelope under
the additive `enrichment_assets` key. `voxint enrich assets <run-uuid>
[--kind summary|topics|entity_mentions]` runs generations inline for
headless use. Optionally, `ENRICHMENT_RUN_ASSETS_AUTOGENERATE=true` enqueues
generation automatically when a run completes (kinds whose current asset
already matches the source are skipped; a broker outage defers, never fails
the run). Summaries and topics are the model's reading, not a verified
record, so the UI labels them machine-generated; entity spans that cannot be
located verbatim in their segment are dropped and counted rather than shown.

The mutation forms that require a CSRF token are `POST /submit`, `/fetch`,
`/runs/{id}/requeue`, `POST /review/{id}/claim` (claiming mints the run's claim
token, so it has none of its own to gate a forged POST), the web-research
forms on `/speakers` (start, cancel, and per-draft accept/reject, each under
its own token action), and the run-asset forms on `/runs/{id}` (generate and
cancel, each under its own token action). Set `CSRF_SECRET` to a
persistent random value
(`python -c "import secrets; print(secrets.token_urlsafe(32))"`); otherwise a
random per-process secret is used, which invalidates open forms on restart and
mismatches across multiple workers.

### LLM endpoint timeouts: local models and proxies

Everything LLM-backed (transcript enhancement, the name pass, run assets,
web research) shares one per-attempt timeout, `LLM_TIMEOUT_SECONDS`
(default 300 s). The default is sized for **local models**: on maintainer
hardware, entity-mention extraction with a ~35B model routinely takes
180–300 s per call, and shorter timeouts made the default configuration fail
for exactly the self-hosted deployments this project targets. Cloud
endpoints answer in seconds and never wait out the timeout on a healthy
connection (connection establishment has its own short cap, so an
unreachable endpoint still fails fast). What the generous default does cost:
an endpoint that accepts connections but hangs mid-response is detected
slowly. For transcript enhancement the worst case before the circuit
breaker stops calling is `LLM_CONSECUTIVE_FAILURE_LIMIT ×
LLM_ATTEMPTS_PER_BATCH × LLM_TIMEOUT_SECONDS` (30 minutes at the defaults).
If you only use a fast cloud endpoint, lowering `LLM_TIMEOUT_SECONDS` tightens
that.

Two ceilings the client timeout **cannot** override:

- **A proxy in front of your endpoint.** Proxies impose their own request
  ceilings and return HTTP 408 when a call exceeds them, regardless of what
  Voxint is configured to wait. On the maintainer's deployment, an
  OpenAI-compatible proxy 408'd at its own **180 s** ceiling despite a
  higher client timeout; defaults vary by product, version, and
  configuration, so check yours. If long local-model calls fail with 408
  despite a high `LLM_TIMEOUT_SECONDS`, raise the proxy's own
  request-timeout setting. The effective limit is always the *lower* of the
  client timeout and every proxy/backend ceiling between Voxint and the
  model.
- **The web-research deadline.** `RESEARCH_DEADLINE_SECONDS` (default 300 s)
  is checked between research rounds, never mid-round: a round's model call
  (and, on a malformed reply, its one repair call) always finishes first. A
  single slow local-model call can consume most of the deadline, leaving the
  researcher one round before it is forced to conclude. With a local model
  in the 180–300 s-per-call range, raise the deadline to several multiples
  of your typical call time if you want multi-round research.

## Adjudication workflow

The review console is served by the API at `http://127.0.0.1:8080/` (or your
`API_PORT` override; basic auth, `VOXINT_USER`/`VOXINT_PASSWORD`, a single
reviewer credential). On a **fresh install** the onboarding gate redirects every
authenticated page to the first-run setup wizard (`/setup`) until setup is
finished, so `/review` below becomes reachable only after onboarding completes
(see [onboarding.md](onboarding.md)):

1. **Queue** (`/review`): runs that finished matching and await human
   review.
2. **Claim**: claiming a run gives you an exclusive slot for
   `REVIEW_CLAIM_TTL_SECONDS` (default 30 min); a closed tab self-releases
   when the TTL lapses, so the queue never dams.
3. **Workbench** (`/review/{run_id}`): per-label transcript previews and
   audio playback; record a **decision** per diarization label (confirm /
   correct / reject the machine proposal) or **enroll** a label's audio as a
   new speaker in the roster. Human rulings are an immutable ledger kept
   separate from machine proposals; adjudication precedence is defined in
   quality-gates.md.
4. **Release / export**: release the claim for the next reviewer, or export
   the speaker-attributed transcript (`/review/{run_id}/export.{txt,srt,vtt,
   json,rttm}`, or `voxint export`; see "Exporting transcripts" above).

## HTTP endpoints

Every route but `/healthz` sits behind HTTP Basic; the core mutation forms
(`POST /submit`, `/fetch`, `/runs/{id}/requeue`, `POST /review/{id}/claim`, and
the wizard/settings forms with their own `CSRF_SETUP` / `CSRF_SETTINGS` tokens)
additionally require a CSRF token (see above), and the remaining review-workbench
mutations are gated by their per-run claim token.

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness (no DB access; schema readiness is the migrate gate's job) |
| `GET /metrics` | Prometheus text exposition (aggregate gauges; authenticated, scrape with `basic_auth`) |
| `GET /dashboard` | Operator dashboard: read-only HTML render of the `/metrics` aggregates; optional `?since=` window, 15s htmx auto-refresh |
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

## Media retention / garbage collection (issue #15; off by default)

Storage grows with every run: the pipeline writes a normalized 16 kHz WAV
intermediate (`artifacts/{run_id}/normalized.wav`) that transcription and
diarization read, and nothing reclaims it. When enabled, a beat-scheduled GC
sweep reclaims that intermediate for **old terminal runs** — it unlinks the WAV
and stamps the `audio_artifacts` row (`reclaimed_at`, `reclaimed_bytes`) as an
audit record; the row itself is never deleted.

**What is reclaimed:** only the normalized-audio intermediate of runs that are
`completed` or `cancelled` and untouched for `MEDIA_RETENTION_SECONDS`.

**What is always kept:** the **source media** (so a reclaimed run stays
re-processable — re-submit it to regenerate the intermediate and downstream
results), the transcript, diarization turns, speaker assignments, and the
immutable adjudication decision ledger. Runs mid-pipeline, `failed`/requeue-able
runs, the guided-tutorial run, and any file that is also registered as a run's
source are all excluded.

**It is off by default** — no bytes are reclaimed until you opt in. In the
console, a run whose intermediate was reclaimed shows a "Media reclaimed on
`<date>`" notice instead of the audio link, and `GET /media/{run_id}` returns
`410 Gone`.

```dotenv
# .env — enable and tune (defaults shown)
MEDIA_RETENTION_ENABLED=true      # off unless set
MEDIA_RETENTION_SECONDS=2592000   # 30 d; floor 3600 (1 h)
GC_SWEEP_SECONDS=3600             # how often the sweep runs
GC_BATCH_LIMIT=500                # rows per sweep, oldest-first
```

A backlog drains at `GC_BATCH_LIMIT` rows per `GC_SWEEP_SECONDS` (the sweep
processes one bounded, oldest-first batch per run and is safe to run
concurrently — rows are claimed with `FOR UPDATE ... SKIP LOCKED`). To reclaim a
large accumulated backlog faster, raise `GC_BATCH_LIMIT` or lower
`GC_SWEEP_SECONDS` until it catches up.

To reclaim a single run's derived audio **immediately** (rather than waiting for
the sweep), use the manual **Delete derived audio files** action on the run
detail page (`POST /runs/{id}/media/delete`, issue #5, above). The manual action
deletes the rows and files outright; the scheduled sweep keeps the row and
stamps `reclaimed_at` as an audit record. Archived runs remain eligible for the
sweep — archiving only hides a run from the console, it does not exempt its
intermediate from reclamation.

## Run notifications / webhooks (issue #12; off by default)

Voxint can POST a **signed webhook** to an endpoint you control when a run
reaches a **notifiable transition** — `completed` or `failed`. It is opt-in and
off by default; nothing is emitted or delivered until you enable it, and
enabling later never back-fills runs that finished while it was off.

**How it is delivered (at-least-once).** The notification is recorded as an
outbox row **in the same database transaction as the run's state change**, so
delivery intent is atomic with the transition — a run never "completes" without
its notification queued, and a rolled-back transition takes its notification with
it. A beat-scheduled sweep then delivers each row **outside any transaction**, so
a slow or down receiver never blocks the pipeline. Because delivery is retried
until it succeeds, your receiver **can see the same delivery more than once** —
**deduplicate on the `delivery_id`** in the body (equivalently the
`X-Voxint-Delivery` header). A `failed` notification whose run is requeued before
delivery is *suppressed* (you are not paged about a failure the system already
recovered from); the short `NOTIFY_FAILED_INITIAL_DELAY_SECONDS` hold gives that
requeue time to settle.

**Egress posture.** The endpoint must be a public http/https URL (a
private/loopback/link-local target is refused, same as URL ingestion). Every
attempt re-resolves the host and connects to the vetted address (closing DNS
rebinding), redirects are **not** followed, and an ambient `HTTP(S)_PROXY` is
ignored. No URL, secret, or payload is ever written to a log or a stored error.

```dotenv
# .env — enable and tune (defaults shown)
NOTIFY_ENABLED=true
NOTIFY_WEBHOOK_URL=https://your-host.example.com/voxint-hook
NOTIFY_WEBHOOK_SECRET=<a random string, >= 16 chars>   # keep secret
NOTIFY_SWEEP_SECONDS=30            # how often delivery runs
NOTIFY_MAX_ATTEMPTS=8             # then the row is marked "dead"
NOTIFY_RETENTION_SECONDS=604800   # purge delivered/suppressed rows after 7 d
```

**The request.** A `POST` with a compact JSON body and these headers:

| Header | Meaning |
| --- | --- |
| `X-Voxint-Delivery` | The `delivery_id` — your idempotency key. |
| `X-Voxint-Timestamp` | Unix seconds when the POST was signed. |
| `X-Voxint-Signature` | `sha256=<hex>` — HMAC-SHA256 of `timestamp + "." + body`. |

The body is `{"schema_version", "event", "run_id", "transition_revision",
"occurred_at", "delivery_id"}`. It deliberately **omits the run's error text**
(it can carry sensitive detail); look up the run by `run_id` for detail.

**Verifying a delivery (receiver side).** Recompute the signature over the
**raw request bytes** — parse-and-re-serialize would change them and the check
would fail — compare in constant time, and reject a stale timestamp:

```python
import hashlib, hmac, time

def verify(raw_body: bytes, headers: dict[str, str], secret: str, *, skew: int = 300) -> bool:
    ts = headers["X-Voxint-Timestamp"]
    if abs(time.time() - int(ts)) > skew:      # replay window
        return False
    want = hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, hashlib.sha256).hexdigest()
    got = headers["X-Voxint-Signature"].removeprefix("sha256=")
    return hmac.compare_digest(want, got)      # then dedupe on X-Voxint-Delivery
```

Return any `2xx` to acknowledge; anything else (or a timeout) is retried with
capped exponential backoff until `NOTIFY_MAX_ATTEMPTS`, after which the row is
left `dead` for inspection (never auto-deleted). The sweep is safe to run
concurrently and purges old `delivered`/`suppressed` rows after
`NOTIFY_RETENTION_SECONDS` to bound table growth.

## Backup

State worth backing up: the Postgres volume (`pgdata`, holding runs,
transcripts, roster, and the decision ledger) and your media tree
(`MEDIA_ROOT`). Redis holds only in-flight queue state; losing it is
recoverable, since the recovery sweep re-enqueues interrupted runs and
`voxint requeue` covers FAILED ones. Dump the database over the published
port (credentials default to
`voxint`/`voxint`/db `voxint` unless you set `POSTGRES_USER` /
`POSTGRES_PASSWORD` / `POSTGRES_DB` in the environment):

```bash
pg_dump -h 127.0.0.1 -p "${POSTGRES_PORT:-5432}" -U voxint voxint > voxint.sql
```
