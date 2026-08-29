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

Prerequisites beyond Docker Desktop: **[Homebrew](https://brew.sh)** and
**[`uv`](https://docs.astral.sh/uv/)** (`brew install uv`); `voxint-metal.sh setup`
hard-fails without `uv`. Docker Desktop is required specifically (Colima/OrbStack/
plain `dockerd` break the `host.docker.internal` loopback the overlay depends on).

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

### GPU memory on a single, modest GPU (issue #96)

The default GPU overlay is tuned for headroom, not for the smallest card. On a
host with one modest GPU (for example a single 12 GB consumer card shared by
whisper, pyannote, titanet, and any co-resident LLM), the stock settings can
exhaust VRAM. The first out-of-memory error can poison that service's CUDA
context, so every request after it fails with a cascade of
`CUDA error: out of memory` and then `invalid device ordinal` until the service
is restarted. Shrinking that first allocation is the fix.

> ⚠ A poisoned CUDA context does not recover on its own. If a GPU service starts
> reporting `invalid device ordinal`, restart it (`docker compose restart
> whisper`) to clear the context, then apply the settings below so it does not
> recur.

#### What the installer applies automatically

On the GPU tier, `scripts/install.sh` reads the host GPU with `nvidia-smi`,
writes a generated `compose.hardware.yaml` with a conservative baseline, and
merges it into the compose chain it launches. For any GPU it does not yet have a
measured profile for, that baseline caps two schedule-only levers:

```yaml
services:
  worker:
    command: celery -A voxint.worker.app worker --loglevel=INFO --concurrency=1
  whisper:
    environment:
      MAX_PENDING_REQUESTS: "1"
```

`--concurrency=1` serializes runs at the worker so the GPU services are never
asked to hold several transcriptions at once, and `MAX_PENDING_REQUESTS=1`
bounds whisper's own admission queue. Neither changes transcription output:
whisper serializes inference behind a single model lock, so concurrency and
queue depth set scheduling, not numerics. The baseline deliberately does **not**
set `BATCH_SIZE`, which does move whisper's output (it feeds the decode config),
so an automatic value has to come from a per-GPU profile that has passed the
parity gate and an out-of-memory soak on real hardware. Until such a profile
exists for your card, `BATCH_SIZE` stays at the image default and you tune it by
hand (below) if a run exhausts VRAM.

The file is installer-owned: its first line carries a `# voxint:hardware-override`
marker, it is regenerated on every install run, and it is refreshed if you swap
cards. A `compose.hardware.yaml` you wrote yourself (no marker) is left untouched
and is not loaded. To preview what the installer would write, changing nothing,
run:

```bash
./scripts/install.sh --hardware-dry-run
```

Put your own hand-tuning in `compose.override.yaml`, not in the generated file.
The installer launches with an explicit file list, so it now merges your
`compose.override.yaml` last of all, letting it win over the base stack, the tier
overlay, and the hardware baseline. Precedence runs lowest to highest:
`compose.yaml`, the tier overlay (`compose.gpu.yaml`), `compose.hardware.yaml`,
then `compose.override.yaml`.

For overrides that are purely local to one host (GPU device IDs, port remaps to
avoid conflicts with other services on the same box), a gitignored
`compose.local.yaml` keeps deployment-specific details out of version control.
Add it to your `.gitignore` and append it to your `-f` chain after the other
overlays so it wins last.

#### Multi-GPU hosts

The GPU overlay (`compose.gpu.yaml`) requests one GPU per model service with
`count: 1` and lets the Docker NVIDIA runtime pick which card. On a host with
several GPUs, the runtime may assign a card that is already under memory
pressure from other workloads.

Pin all three model services to a specific card by overriding the `devices`
list with `device_ids`. Compose v2.24+ supports the `!override` tag to replace
a sequence instead of appending to it:

```yaml
# compose.local.yaml (gitignored)
services:
  whisper:
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
  pyannote:
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
  titanet:
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
```

`device_ids` takes GPU indices (from `nvidia-smi`) or full UUIDs. Without the
`!override` tag, Compose appends to the existing `devices` list, producing two
reservation entries and unpredictable assignment.

#### Port conflicts with other services

The GPU overlay publishes each model service on `127.0.0.1` for local debug
access (`:8021` titanet, `:8022` whisper, `:8024` pyannote). If another service
on the host already occupies one of those ports, Compose fails with "port is
already allocated" and the container stays in `Created` state.

The model services communicate over the Docker network by service name, so the
host port binding is optional. Remap it in a local override:

```yaml
# compose.local.yaml (gitignored)
services:
  pyannote:
    ports: !override
      - "127.0.0.1:18024:8024"
  titanet:
    ports: !override
      - "127.0.0.1:18021:8021"
```

GPU pins and port remaps can share the same `compose.local.yaml`.

To watch the pressure these settings fight, run `voxint doctor` or read a
service's `/healthz`: both surface live per-GPU VRAM used against total,
temperature, and throttle state (see "Metrics & monitoring"). A card sitting
near its VRAM limit, or reporting a thermal or power throttle during a run, is
the signal to lower the levers below.

Three settings drive peak VRAM. Lower them when a single card is the constraint;
raise them for throughput when the card has room.

| Setting | Where | Default | What it does |
|---|---|---|---|
| Worker concurrency | worker `command:` (`--concurrency=N`) | host CPU count | How many runs the Celery worker processes at once. The stock command sets no `--concurrency`, so it defaults to the host CPU count and can drive many runs onto the GPU services in parallel. |
| `BATCH_SIZE` | whisper service `environment:` | `16` | whisper's CTranslate2 decode batch. Peak transcription VRAM is set by this, not by recording length, so it is the largest single lever. |
| `MAX_PENDING_REQUESTS` | whisper service `environment:` | `8` | How many requests whisper queues before refusing new work with `503`. A lower cap bounds how much the worker can pile onto whisper at once (see [gpu-contracts.md](gpu-contracts.md)). |

whisper reads `BATCH_SIZE` and `MAX_PENDING_REQUESTS` from the environment at
startup, so a compose `environment:` override changes them without rebuilding the
image. Worker concurrency is a command-line flag, so it is set by overriding the
worker `command:`.

To go beyond the installer's baseline, put your own settings in
`compose.override.yaml` (it merges last, so it wins). A fuller conservative
profile for one very small card also lowers `BATCH_SIZE`:

```yaml
services:
  worker:
    command: celery -A voxint.worker.app worker --loglevel=INFO --concurrency=1
  whisper:
    environment:
      BATCH_SIZE: "4"
      MAX_PENDING_REQUESTS: "1"
```

`--concurrency=1` serializes runs so the GPU services are never asked to hold
several transcriptions at once; `BATCH_SIZE=4` and `MAX_PENDING_REQUESTS=1` cap
whisper's own peak and queue. This trades throughput for stability: a run that
would have shared the card now waits its turn. whisper already serializes
inference behind a single model lock, so lowering `--concurrency` changes
scheduling, not the transcription result. On a card with spare VRAM, raise the
batch and concurrency back up.

`--concurrency=1` on a single worker also serializes the stages that never
touch the GPU: while one run's LLM enhancement is in flight, the card sits
idle and the next run cannot start transcribing. The two execution lanes (see
[architecture.md](architecture.md#execution-lanes-and-queues)) remove that
cost. Split the worker in the same override: the GPU lane keeps
`--concurrency=1`, and a second worker drains the `post` queue (LLM
enhancement, finalize, and the LLM-bound asset/research jobs) concurrently:

```yaml
services:
  worker:
    command: celery -A voxint.worker.app worker --loglevel=INFO -Q celery --concurrency=1
  worker-post:
    image: ghcr.io/bengizmo/voxint:${VOXINT_IMAGE_TAG:-0.29.0}
    pull_policy: missing
    command: celery -A voxint.worker.app worker --loglevel=INFO -Q post --concurrency=2
    restart: unless-stopped
    env_file:
      - path: .env
        required: false
    environment:
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-voxint}:${POSTGRES_PASSWORD:-voxint}@postgres:5432/${POSTGRES_DB:-voxint}
      REDIS_URL: redis://redis:6379/0
      MEDIA_ROOT: /data/media
    volumes:
      - ${MEDIA_ROOT:-./media}:/data/media
    depends_on:
      migrate:
        condition: service_completed_successfully
      redis:
        condition: service_healthy
```

The new `worker-post` service inherits nothing from the base `worker` service,
so the override must repeat its image, environment, volume, and dependencies.
With no `-Q` flag at all (the stock command), one worker consumes both queues
and behavior is unchanged; the split is purely a deployment choice.
If your existing worker command sets `-Q`, drop the flag or use
`-Q celery,post`; otherwise the `post` queue has no consumer.

The installer's conservative baseline (above) is the safe-by-default sizing from
issue #96 for the two schedule-only levers. Measured per-GPU `BATCH_SIZE` profiles
are still landing behind the parity gate; until one exists for your card, tune
`BATCH_SIZE` by hand when the stock overlay runs out of memory.

#### PyTorch allocator assert on a shared GPU (issue #111)

Under heavy VRAM pressure (for example a card that also serves another CUDA
workload), a titanet or pyannote request can fail with an internal PyTorch
assert instead of a plain out-of-memory error:

```text
RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp"
```

The service returns `500` for that request and the run fails at its current
stage. This is an upstream PyTorch bug in the `expandable_segments` allocator
mode, which both CUDA images enabled through 0.21.0. The reproduced case was
titanet on a card shared with another CUDA workload, where long recordings hit
the assert on every retry; pyannote shipped the same allocator mode and
carries the same exposure. Retrying the run or restarting the service does not
help while the mode is enabled.

Releases after 0.21.0 drop `expandable_segments` from both images and keep
the rest of each image's allocator tuning. On an affected image, apply the
same change without rebuilding by overriding the allocator mode in your
compose override:

```yaml
services:
  titanet:
    environment:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256,garbage_collection_threshold:0.7"
  pyannote:
    environment:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512,garbage_collection_threshold:0.8"
```

Then recreate the two services with your full compose file chain (the GPU
services are defined in `compose.gpu.yaml`, and using `-f` at all means the
override must be listed explicitly):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml -f compose.override.yaml up -d titanet pyannote
```

and requeue the failed run.

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
docker compose exec api voxint speakers auto-enroll-backfill --dry-run  # preview auto-enrollment on existing runs
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
credentials, tokens, or connection URLs are printed. After the checks it prints
an advisory hardware-telemetry section (aggregated GPU utilization, VRAM,
temperature and throttle state, plus each service's admission depth) read from
the same `/healthz` `resources` block described under "Metrics & monitoring". A
telemetry failure never changes the doctor verdict.

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
enrichment. A manifest change takes effect on the *next* run.

> Per-**folder** assignment (`{media_folder → pack_name}`) is editable in the
> console (issue #63): the setup wizard's media step and **Settings → Media
> folders** host a folder browser with a per-folder domain-pack picker. A
> per-**submission** pack override remains a backend-only capability. The default
> pack (`DOMAIN_PACK_PATH`) stays the installation-wide fallback.

**Deterministic corrections (epic #78).** A pack may also declare a `corrections:`
list of literal `find → replace` rules that the `enhance_match` stage applies with
no model, composed with the optional LLM enhancement through a raw-gated dual pass
(#82). An operator can author their **own** rules from **Settings → Corrections**
(#84) without editing a manifest; those live per deployment in
`app_settings.corrections` (migration **0029**) and, at submit time, are **unioned
onto the resolved pack and frozen** into the same `pipeline_runs.domain_pack`
snapshot, so one frozen pack drives both the correction and its review-console
provenance (#83). The per-segment trail the console reads back is
`transcript_segments.correction_trace` + `corrector_version` (migration **0028**).
A colliding operator rule is refused: at author time against the default pack,
and visibly at submit-time freeze for a differently-scoped folder pack (never a
silent drop; on the ingest routes this surfaces as a plain-language 422, on the
CLI as exit 2, and the watch-folder sweep logs-and-skips the offending file rather
than stalling). See `docs/domain-packs.md` for the rule schema and semantics.

### Metrics & monitoring

**`voxint stats`** prints an aggregate, read-only snapshot: run counts by status,
failed stage *attempts* per stage, average per-stage duration over finished
attempts, roster size, and runs created within a window. `--since` accepts a
relative span (`24h`, `7d`) or an ISO-8601 datetime (default 24h); `--json`
emits a stable object for scripting. It also appends the aggregated hardware
snapshot (see "Hardware resource telemetry" below), under a `resources` key in
`--json`.

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

For a human at the console, the **Home** page (`GET /`) leads with a task-first
spine: needs-attention cards for **Continue review** (the canonical
review-backlog count), **Unidentified voices**, and **Failed runs** (each count
derived from the queue or figures it links to, so they cannot disagree), then
quick actions and the activity counts (**media added, runs started, speakers
enrolled**) with an hour / day / week / all-time window switch. The counts share
their query functions with `voxint stats`, so the two surfaces agree for the
same cutoff; a malformed `?window=` degrades to the day window and says so. A
recent-activity list (runs started, runs finished or failed, speakers enrolled)
closes the page. The cards render on page load and refresh only on a full
reload, so every count is a plain figure, never styled as live.

The old `/dashboard` page folded into Home and now issues a permanent redirect
to `/`. Its per-status and stage-timing HTML tables are retired; the same
figures remain on `GET /metrics` and `voxint stats` (raw identifiers there,
plain language in the console).

#### Hardware resource telemetry

Separately from the database aggregates above, each model service reports its own
hardware state on `GET /healthz` as an additive `resources` block: GPU
utilization, VRAM used against total, temperature, throttle state and decoded
reasons, an `admission` block (in-flight and rejected request counts), and a
host-visible CPU advisory. The full wire shape and its guarantees are in
[gpu-contracts.md](gpu-contracts.md); the two service-side knobs are:

| Env var | Where | Default | What it does |
|---|---|---|---|
| `VOXINT_TELEMETRY_ENABLED` | model service `environment:` | `1` | Set to `0` to turn the background sampler off; GPU telemetry then reports `disabled`. |
| `VOXINT_TELEMETRY_INTERVAL_SECONDS` | model service `environment:` | `5` | Background sample cadence, clamped to 0.5-3600. `/healthz` always serves the cached sample, never a live probe. |

Telemetry is fail-soft and never affects readiness: with no NVIDIA GPU or no NVML
the GPU fields report `unsupported`, and a healthy model still answers `200`. The
GPU is resolved by UUID, so three services sharing one physical card report the
same device rather than three.

The app aggregates the three services' blocks into one view (deduplicating a
shared GPU by UUID) behind a short single-flight cache, so a browser poll across
several tabs never fans out concurrent live probes. One app-side knob controls
the cache:

| Env var | Default | What it does |
|---|---|---|
| `RESOURCE_STATUS_TTL_SECONDS` | `10` | How long the aggregated resource view is cached before the app re-probes the services. `0` probes live on every read. |

This one aggregated snapshot renders on every surface: `voxint doctor` and
`voxint stats` print it (the per-GPU readout plus each service's admission
depth), `voxint stats --json` carries it under a `resources` key, `GET /metrics`
appends `voxint_gpu_*` and `voxint_service_admission_*` gauges, and the console
renders it in two places.

The **Resources** page (sidebar: **Hardware**) opens with a compact hardware
strip, refreshed on its 15s poll. It is deliberately quiet: it shows each GPU's activity
(idle / working / busy, never an alarm, since 100% during a transcription is
healthy) and raises an amber warning only for the two conditions an operator can
act on, each with one plain-language remedy:

- The driver reports the GPU is thermally throttling (improve airflow, let it
  cool). A high temperature that is not throttling is shown on the resource page
  but does not warn.
- A service's admission queue is currently full (wait for current work to
  finish). A past rejection with a now-idle queue does not warn.

High VRAM is not a warning: the models hold a resident footprint, so a
percentage is not an honest predictor of an out-of-memory failure. When no
service reports telemetry the strip says "hardware status unavailable" rather
than claiming all-clear.

The **Status** page (`GET /settings/status`, authenticated, 15s htmx refresh) is
the fuller live view behind the strip. It shows up to five hardware gauges:
Processor (load-average percentage), Memory (used / total), and Disk (media
root partition used / total) are always present; Graphics card (GPU
utilization) and Graphics memory (VRAM used / total) appear when a GPU is
available. CPU, memory, and disk are read from the host via stdlib
(`os.getloadavg`, `/proc/meminfo`, `shutil.disk_usage`); GPU metrics come from
the model services' `/healthz` telemetry. A "Parts of Voxint" component
list shows live health for the console, each model service, the database, the
task queue, and the Local AI model (with a primary "Turn on" button when
disabled). The banner includes an install summary with GPU acceleration status
and a "Check for updates" link to the GitHub releases page. The page is
reachable from the sidebar "Hardware" shortcut; the older `GET /resources`
address still works and redirects here. Warnings are warn-only in v1; the NVIDIA
driver already protects the hardware, so Voxint advises rather than pausing
work.

### Exporting transcripts

Every run's speaker-attributed transcript exports in six formats, from the CLI
or over HTTP. Both paths share the same formatters, so a downloaded file and a
piped export are byte-identical.

```bash
docker compose exec -T api voxint export <run-id> --format srt   > out.srt
docker compose exec -T api voxint export <run-id> --format vtt   > out.vtt
docker compose exec -T api voxint export <run-id> --format json  > out.json
docker compose exec -T api voxint export <run-id> --format rttm  > out.rttm
docker compose exec -T api voxint export <run-id> --format txt   > out.txt
docker compose exec -T api voxint export <run-id> --format md    > out.md
docker compose exec -T api voxint export <run-id> --format md --no-timestamps > reading.md
```

- `--format`: `srt` (SubRip), `vtt` (WebVTT), `json` (array of
  `{start_seconds, end_seconds, speaker, text}`), `rttm` (NIST diarization
  format), `txt` (bracketed plain text), or `md` (Markdown: a `##` speaker
  heading per contiguous same-speaker run over a `>` blockquote paragraph, with
  a per-paragraph `[start-end]` time range gated by the timestamps flag).
  Default `txt`.
- `--text corrected|enhanced|raw`: which transcript variant to render (default
  `corrected`, the operator-effective text with review corrections applied over
  the enhanced or raw fallback; `enhanced` is the LLM-cleaned pipeline text
  before corrections; `raw` is the immutable ASR output). Ignored for `rttm`,
  which carries raw diarization labels, not attributed text.
- `--no-timestamps`: drop the per-line time column (`txt`) or per-paragraph time
  range (`md`) for a clean reading copy. Ignored for the other formats, whose
  timing is structural.
- `-o PATH`: write to a file instead of stdout (refuses to overwrite an
  existing file unless `--force`).

The same exports are available over HTTP at
`GET /review/{run_id}/export.{txt,md,srt,vtt,json,rttm}` (add `?text=raw` for the
raw variant, or `?timestamps=false` on `txt`/`md` for the reading copy). RTTM
uses the run's UUID as the file id and the raw diarization labels (`SPEAKER_00`
…), so it round-trips against diarization scoring tools, and it deliberately
does **not** substitute adjudicated speaker names.

The transcript view route serves the same attributed text as an on-screen
**read mode**: `GET /runs/{run_id}/transcript?read=1` renders the transcript as
prose (one speaker heading over a merged paragraph), server-side with no
JavaScript, gated by `&timestamps=false` for a timestamp-free reading view. Read
mode and the Markdown export share one grouping helper (`paragraphize_transcript`)
with the presentation seam, so neither can drift from the other exports.

### The browser console

The same API serves a browser console (HTTP Basic, `VOXINT_USER` /
`VOXINT_PASSWORD`) for operators who prefer not to shell into a container:

- **`GET /runs`**: an execution-history browser, newest-first, keyset-paged
  (`RUNS_PAGE_SIZE`, default 50), with **orthogonal** filters `status=`,
  `review=needed|resolved|claimed`, and `language=` (an exact match on the
  language whisper detected, its options limited to codes some run actually
  carries; #124). Each row carries a **Language** column showing the detected
  language, or a dash for a run not yet transcribed or one that predates the
  feature. **`GET /runs/{id}`** shows the run detail and
  the per-stage attempt ledger (the same data as `voxint status`), with
  transcript and audio links when present. A **Pipeline models** block renders
  the per-attempt model identity recorded for the transcription and diarization
  stages (from each stage's latest completed attempt, so a retried stage shows
  the model that produced the result); runs that predate this provenance read
  "Not recorded". A **Detected language** block names the language and, when one
  was recorded, whisper's own **language-detection score**, labelled as the
  model's confidence in its own guess rather than a measure of transcript
  accuracy; runs from before the feature read as unrecorded. A **Glossary
  applied** block shows the exact bounded vocabulary hint whisper decoded with for
  that run (the operator glossary unioned with the run's domain-pack words, from
  the `pipeline_runs.initial_prompt` column stamped once the transcribe stage
  succeeds; #123); a run not yet transcribed, one with no vocabulary, or one that
  predates the column shows an honest empty state. See
  [Changing pipeline models](how-to/changing-pipeline-models.md).
- **`GET /search`**: the transcript **meaning-search** page (#121), reached from
  the **Meaning** tab beside the `/runs` search box. It ranks passages from across
  the whole corpus by a reciprocal-rank fusion of pgvector cosine similarity, a
  `simple` full-text arm, and an exact-quote arm (a `"quoted phrase"` is matched
  verbatim and floated to the top), reading `segment_embeddings` directly in one
  `REPEATABLE READ` transaction. Each hit deep-links the transcript at the passage
  start. The page reports its own state honestly: `off` (semantic search disabled),
  `unavailable` (the embedding weights are not installed), or `indexing` (on, but
  nothing indexed yet, pointing at `voxint embed backfill`). See
  [semantic-search.md](semantic-search.md).
- **`GET /review`**: the adjudication queue of completed runs with at least one
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
  When `console_media_enabled` is on, this route redirects to `/media` (303)
  without processing; uploads go through `/media/submit` instead.
- **`POST /fetch`**: the browser equivalent of `voxint fetch` (URL ingestion).
  When `console_media_enabled` is on, this route redirects to `/media` (303);
  URL fetches go through `/media/fetch` instead.
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
  so a stale tab can't drive a hidden run back to live. The guard is enforced at
  both the route level (API) and the service level (`RunArchivedError` in
  `requeue_failed_run`), so the CLI path is also covered. `/runs` hides archived by
  default; `?archived=1` shows the archived-only view. Home, `/metrics`, and
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
  GC sweep keeps the `AudioArtifact` row and stamps `reclaimed_at` for audit.
  Use whichever fits, they compose (deleting a run already GC-reclaimed just
  finds its file already gone).

**Media library file management** (`/media`, behind `console_media_enabled`,
ADR 0007). The library page offers three file operations through a journaled
operations system that survives crashes at any filesystem boundary:

- **Trash** (`POST /media/trash`): bulk-moves selected files into a managed
  `_trash/` tree inside `MEDIA_ROOT`. The file stays playable (playback resolves
  `current_path`, which follows the move). The watcher skips the trash tree.
  `media_items.trashed_at` is set on completion.
- **Restore** (`POST /media/restore`, from the trash view): moves each file back
  to its original location. Refuses if the destination is occupied (an honest
  "destination occupied" error, never an overwrite).
- **Empty trash** (`POST /media/empty-trash`): permanently deletes every trashed
  file and all its derived artifacts (preprocessed WAV, chunks, peaks, clips).
  Builds a durable per-file manifest first, deletes children one at a time with
  per-child commits, then removes artifact DB rows and sets `purged_at`. A partial
  purge is recoverable (the reconciler retries failed children on the next sweep).

Each operation is recorded in the `media_operations` journal. The route plans the
operation (committing the journal row), then executes it inline. If execution is
interrupted (crash, I/O error), the `voxint.media_reconcile` beat task drives the
operation to a consistent terminal state on its next sweep.

The active library view hides trashed and purged items. The trash view
(`/media?trashed=1`) shows trashed-not-yet-purged items. A "File missing" badge
appears on any library row whose `current_path` does not resolve to a regular file
on disk.

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
claim token; the page never re-claims (a fresh claim would evict the workbench
tab). With JavaScript off the same page lists every segment, with a plain
**Verify** form on each one still unverified (inline editing needs the browser
island, stated plainly), and it renders read-only with a prompt to claim when
this tab does not hold the run's claim.

**Split a segment at a word boundary.** When faster-whisper merges two speakers'
words into one segment, the review page can cut it at a word. Press the **⎇ Split
at a word** button to enter split mode; the segment under the review cursor then
shows its individual words, and clicking a word cuts the segment *before* it (you
cannot cut before the first word). The cut is stored as an append-only boundary
on the immutable segment (the original `raw_text` and its word timings are never
altered), and the segment renders from then on as the derived child lines, each
inheriting the parent's speaker and review state. A split is only offered when the
segment's words reconcatenate exactly to its `raw_text` and its text has not been
materially enhanced; otherwise split mode reports plainly that the segment cannot
be split rather than guessing at boundaries. Splitting and inline editing are
**mutually exclusive**: a segment that has been split cannot then be edited (and a
segment with an operator correction cannot be split): the box is disabled with a
short note, because a split's text is word-derived and a free-form edit would have
nowhere faithful to live. A segment can be cut once (into two children); splitting
an already-split segment into more parts is refused in this release, and there is
no un-split control. A mis-split is cleared by re-transcribing the run. Like
inline editing, splitting needs the browser island: with JavaScript off the
transcript still lists any already-derived child lines, but no new split can be
made.

**Reassign a split child to the right speaker.** A split's two halves start out
sharing the parent segment's single resolved speaker, which is rarely what you
want. The point of splitting a mis-merged segment is to give each half its own
speaker. After a split, each derived child line shows a small **speaker:**
dropdown listing your active roster; pick a speaker to reassign just that child,
or pick **↺ inherit (follow the segment)** to clear that child's own speaker so
it follows the segment again (a whole-segment reassignment if one exists,
otherwise the diarization label). The choice is scoped to that child's exact
word-range and stored as an append-only ruling on the immutable parent. It
survives a later whole-label decision and, because it is append-only, an
`inherit` reset tracks later rulings live rather than freezing a copy. A
word-range reassignment takes precedence over a whole-segment reassignment,
which takes precedence over the label. Only active roster identities are offered (a merged or archived
speaker cannot attract a new ruling); a run with no roster yet shows only the
inherit option. Like the rest of the review loop, the picker needs the browser
island and a held claim: with JavaScript off the child lines still render with
their resolved speakers, but there is no per-child picker. (Correcting a split
child's *text* per-range, and un-splitting an already-reassigned segment, are not
yet available; clear a mistaken reassignment with **↺ inherit**, or re-transcribe
to clear the split entirely.)

**Waveform strip (who spoke when).** The transcript pages (read-only and
review) draw a compact waveform under the audio player, tinted per speaker with
the same colors as the segment list. The colored regions come from the
diarization turns themselves, so overlapping speech shows a hatched marker and
diarized-but-untranscribed stretches still appear. Clicking the strip jumps to
that segment in the list (and plays it, when seeking is trusted); the review
page also underlines the segment under the review cursor. The amplitude data is
computed once per run on first view (a second or two for long recordings) and
cached; the strip keeps rendering as a static who-spoke-when map even after the
run's processed audio has been reclaimed to free disk space, though when
seeking is disabled (untrusted timeline, missing media) a strip click only
selects the segment, never seeks, and no playhead is shown. If the amplitude
data cannot be computed (e.g. the media file is gone and nothing was cached)
the strip simply does not appear; the transcript list is unaffected.

**Broker-degraded submission.** `/submit` (or `/media/submit` when the Media
library is enabled), `/fetch` (or `/media/fetch`), and `/runs/{id}/requeue`
commit the durable run *before* publishing the Celery task. If Redis is down at
that moment the mutation still succeeds: the run stays `QUEUED` (never `FAILED`)
with a clear linked note, and the recovery sweep re-enqueues it once the broker
returns. Read pages (`/runs*`) render from Postgres only and never touch Redis.

**Orphaned incoming cleanup.** At app startup, `reconcile_orphaned_incoming`
scans `media_root/incoming/` and removes files that have no committed
`MediaItem` row. These are crash orphans from the brief window between
`os.replace` and the DB commit during upload. The reconciler runs once per
process start and logs each removal.

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
`WATCH_FOLDER_SWEEP_SECONDS` but re-checks its **effective** gate each run:
the env `WATCH_FOLDER_ENABLED` default overridden by the runtime
`app_settings.watch_folder_enabled` toggle (Settings → Media folders). A
disabled installation only pays one DB read per sweep, and enabling it needs no
restart. When on, it walks the operator's registered `media_folders`, submits
each new file (skipping ones already ingested), waits out
`WATCH_FOLDER_SETTLE_SECONDS` so a file still being copied in is not read
mid-write, and records a one-line status summary shown in Settings. Like the
other sweeps it needs `beat` running; a bare-host deployment without a beat
process never ingests automatically.

### URL ingestion & egress security

`voxint fetch <url>` / `POST /fetch` download a URL with yt-dlp on the worker
(the ACQUIRE stage). Toggle the capability with `YTDLP_ENABLED` (default on) or
from **Settings → Features**, where a saved console choice wins over the
environment and applies with no restart; when off, the fetch route/CLI/form refuse
and no row is created (an already-queued URL run still completes). yt-dlp is always run with `--proxy`: `YTDLP_PROXY` when set,
otherwise an empty value meaning explicit direct egress, so an ambient
`HTTP(S)_PROXY` in the worker env is **ignored** (set `YTDLP_PROXY` to route
through a proxy on purpose). `YTDLP_COOKIES_FILE` is wired only when set. Both are
treated as **credentials**, never surfaced in errors, so keep them out of logs
and shared configs.

Voxint layers four controls, each closing what the one before it cannot. The
first two are userland gates inside the app; the last two are network policy you
deploy around it (see architecture.md for the gate internals):

1. **Submit gate** (`ingest.validate_ingest_url`). Rejects a non-http(s) or
   malformed URL, embedded credentials, `localhost`, or a private IP *literal* at
   submit time, before a row exists. It does not resolve DNS, so a name that looks
   public now can still rebind later.
2. **Resolved-host gate** (`media.netcheck`, download time). Re-resolves the host
   in the worker just before the download and refuses it if any resolved address
   is non-public, closing the rebind-after-submit window for DNS names. It cannot
   follow yt-dlp's own independent re-resolution, its redirects, or an
   extractor-constructed URL.
3. **Egress proxy overlay** (`compose.ytdlp-egress.yaml`, opt-in, below). Vets and
   pins every yt-dlp connection at the connection boundary, so redirects and
   extractor-built destinations that resolve to private space are refused too. It
   cannot constrain a helper process that ignores the proxy.
4. **Network policy** (a host egress firewall, or a Kubernetes `NetworkPolicy`,
   below). Denies the download path any kernel-level route to RFC1918 /
   link-local / `169.254.169.254`, covering even a proxy-ignoring helper.

Gates 1 and 2 do not close the residual on their own, and no doc should read as if
they do: a rebind between the worker's check and yt-dlp's fetch, an HTTP redirect
to a private address, or an extractor-constructed private URL all slip past a
userland check. The proxy overlay (3) closes them for yt-dlp's own HTTP(S)
traffic, and the network-policy layer (4) is what closes the residual for
genuinely untrusted URLs. A blocked or refused download is a clean FAILED @
acquire the operator **Requeues**; it never loops.

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
connecting only to the vetted public IP. Because the proxy, not yt-dlp, makes
the outbound connection, this closes the rebind window and refuses redirect /
extractor destinations that resolve to a private address. The worker keeps its
normal network, so Postgres, Redis, the model services, and the LLM/research/
notify endpoints are unaffected.

**What it does and does not cover**: it constrains the
HTTP(S) traffic that honours yt-dlp's `--proxy`, *including* its redirects and
extractor-constructed URLs. It is **not** a kernel-level route removal and **not**
a sandbox: a helper yt-dlp spawns that ignores the proxy (e.g. ffmpeg for some
streaming formats), or the worker container's own routable network, is beyond it.
For that last mile, additionally deny the worker a route to RFC1918 / link-local /
`169.254.169.254` with a host-level egress firewall. A refused destination is the
same clean FAILED @ acquire you **Requeue**.

#### Kubernetes: restricted egress with a NetworkPolicy

On Kubernetes the layer-4 firewall above is a `NetworkPolicy`, and it has to bind
to the pod where the download and any helper it spawns actually run: the
**worker** pod. yt-dlp and an ffmpeg helper share the worker's network namespace,
so a policy on any other pod (the proxy included) never sees the helper's traffic,
and only a worker-scoped policy can stop a proxy-ignoring helper reaching private
space. The catch is that the worker also needs its in-cluster dependencies
(Postgres, Redis, the model services, the egress proxy, DNS), which sit on private
addresses, and so does any other private target it legitimately reaches: the
bundled `voxint-llm` pod, a LAN LLM endpoint, or a LAN SearxNG for web research.
Once any egress rule exists the pod is default-deny for everything else, and the
public-internet rule below deliberately carves private space out, so each of these
needs its own explicit allow or it is denied with no hint why. NetworkPolicy
egress rules are additive: the allow rules and the public-internet rule combine,
and their order does not matter.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: voxint-worker-restricted-egress
  namespace: voxint
spec:
  podSelector:
    matchLabels:
      app: voxint-worker
  policyTypes:
    - Egress
  egress:
    # 1. DNS. Shown for CoreDNS in kube-system; substitute your cluster's DNS
    #    namespace and pod labels.
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - { protocol: UDP, port: 53 }
        - { protocol: TCP, port: 53 }
    # 2. The worker's private-address dependencies. Match the real labels of your
    #    own pods. Any other private target the worker must reach (the bundled
    #    voxint-llm pod, a LAN LLM endpoint, a LAN SearxNG) needs its own allow
    #    here too, or the public rule below denies it.
    - to:
        - podSelector: { matchLabels: { app: voxint-postgres } }
        - podSelector: { matchLabels: { app: voxint-redis } }
        - podSelector: { matchLabels: { app: voxint-whisper } }
        - podSelector: { matchLabels: { app: voxint-pyannote } }
        - podSelector: { matchLabels: { app: voxint-titanet } }
        - podSelector: { matchLabels: { app: voxint-egress-proxy } }
    # 3. The public internet, with private, shared, and link-local space carved
    #    out. 169.254.0.0/16 covers the cloud metadata endpoint (169.254.169.254);
    #    100.64.0.0/10 is carrier-grade NAT some overlay networks assign from. For
    #    a proxy-ignoring helper this list is the ONLY egress control, so add any
    #    other special-use range your network uses.
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 10.0.0.0/8
              - 172.16.0.0/12
              - 192.168.0.0/16
              - 100.64.0.0/10
              - 169.254.0.0/16
        - ipBlock:
            cidr: ::/0
            except:
              - fc00::/7
              - fe80::/10
              - fec0::/10
```

Caveats before you rely on it:

- **The cluster needs a working `NetworkPolicy` implementation** (Calico, Cilium,
  or Flannel with its policy controller enabled). Plain Flannel with no policy
  controller accepts the manifest and enforces nothing, so confirm enforcement
  before treating this as a control.
- **Match every selector to your cluster, then test.** The DNS rule targets
  CoreDNS in kube-system; the component labels are placeholders you replace with
  your own pods' labels. Some CNIs evaluate egress against a destination Service
  ClusterIP before it is rewritten to the backing pod, so a `podSelector` allow may
  not match Service traffic; if DNS or a dependency fails closed after you apply
  the policy, add an `ipBlock` allow for your cluster's Service CIDR. Then verify
  UDP and TCP DNS and one real download from the worker.
- **The `ipBlock` list is a coarse kernel-level backstop, not the precise policy.**
  It denies the well-known ranges by CIDR; the shared
  `voxint.media.netcheck.ip_is_public` check (used by both the worker gate and the
  proxy) stays authoritative per-address and also unwraps the IPv4-in-IPv6
  embeddings a static CIDR list cannot express (mapped, 6to4, Teredo, NAT64),
  judging the private IPv4 inside. For yt-dlp's own traffic the proxy applies that
  precise check, so run both; a proxy-ignoring helper has only this list, so widen
  the `except` entries to every special-use range your network uses.

Keep the egress proxy running alongside this policy. On Kubernetes, deploy the
proxy in-cluster and point the worker's `YTDLP_PROXY` at it (rule 2's
`voxint-egress-proxy` allow is the worker reaching it). Under a cluster-wide
default-deny, give the proxy pod its own egress policy too: DNS and the same
public-except-private rule, with no in-cluster allows. The proxy (layer 3) does
the precise per-connection vetting and IP pinning for yt-dlp's own HTTP(S)
traffic; this worker policy (layer 4) is what contains a helper that ignores the
proxy.

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

These four settings (the web-research master toggle, the enrichment-producer
toggle below, the endpoint, and the API key) are also editable from
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
> Settings, issue #10); a UI-saved value wins over these env vars, and the key is
> resolved live per job. The **`LLM_ENABLED` gate above is env-only for enrichment**:
> the UI enhancement toggle governs per-run transcript enhancement, not the
> enrichment producers, so web research and run assets still require
> `LLM_ENABLED=true` in the environment.

Operate it from the speaker's detail page (`/speakers/{id}`): the "Web research" block
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

### Bundled local LLM (issue #67; optional, no API key)

By default, transcript enhancement and run-asset generation call a
bring-your-own OpenAI-compatible endpoint (`LLM_BASE_URL` / `LLM_MODEL` /
`LLM_API_KEY`). The opt-in `compose.llm.yaml` overlay instead ships a vendored,
Apache-2.0 **Qwen3-4B-Instruct-2507** served locally by llama.cpp, so an
operator gets working enrichment with **no external key**. Layer it on top of
your compute tier:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.llm.yaml up -d
```

Then enable it in **Settings → Features → "Use the bundled local model"** (or
`LLM_BUNDLED_ENABLED=true`); `LLM_ENABLED` must also be on (it is the master
enhancement gate). The bundle is **scoped**: it powers **only transcript
enhancement and run-asset summaries + entity mentions**. Web research, LLM
speaker-name suggestions, and run-asset **topics** stay on the BYO endpoint and
never fall back to the bundle; #66 measured that a small local model isn't
reliable at those. When the bundle is active for a run-asset job and no distinct
BYO endpoint is configured, topics is silently skipped rather than generated
badly; when the bundle is active **and** a distinct BYO endpoint is also
configured, topics generate on that BYO endpoint (#106). The overlay publishes
**no host port**: only the worker reaches `voxint-llm` by service DNS.

⚠ CPU is a slow backstop for a dense 4B model: enhancement (~20 s) and
small/medium run-assets are fine, but large transcripts are not, so the bundled
run-asset input is clamped to `LLM_BUNDLED_RUN_ASSETS_MAX_INPUT_CHARS`
(16k, vs the BYO `RUN_ASSETS_MAX_INPUT_CHARS=48000`). A GPU is strongly
recommended: uncomment the `-ngl 99` + device-reservation block in
`compose.llm.yaml`. The pinned serving profile and provenance are in
[gpu-contracts.md](gpu-contracts.md).

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
cancel, each under its own token action). Since v0.27.0, the app auto-generates
a CSRF secret on first start and persists it to the data directory
(`DATA_DIR/csrf_secret`), so forms survive restarts and work across workers
without manual configuration. Set `CSRF_SECRET` explicitly to override the
auto-generated value (useful when multiple app instances share no filesystem).

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

### Reasoning models: turning thinking off

A reasoning model (Qwen3 and similar) emits a chain-of-thought before its
answer. On the heavy calls, entity-mention extraction and multi-round research,
those traces can consume the whole `LLM_TIMEOUT_SECONDS` window before any answer
begins, so the call fails with a read timeout even on a fast local GPU. Set
`LLM_DISABLE_THINKING=true` to send the vLLM chat-template switch
(`chat_template_kwargs.enable_thinking=false`) on every request, BYO and bundled
alike, which skips the thinking phase. It is off by default because a BYO or
OpenAI endpoint rejects the unknown field, so enable it only when both endpoints
honor it (vLLM does). If a heavy job times out against a reasoning model, prefer
this over raising the timeout.

## Auto-enrollment

When a pipeline run completes and its diarization labels have no matching
enrolled speaker, auto-enrollment creates unnamed speaker profiles ("Voice 1",
"Voice 2", ...) and resolves those labels automatically. Labels that match an
existing enrolled speaker are assigned to that speaker. Labels that do not meet
the grounding-tier quality gates (minimum turns, duration, cosine similarity,
margin, vote agreement) are skipped and left for manual review.

Auto-enrollment is on by default (`AUTO_ENROLL=true` in `.env`). Set
`AUTO_ENROLL=false` to disable it and leave all labels for human adjudication.
Auto-enrolled speakers appear in the roster's "Unnamed voices" section with
sequential display names. They carry full embeddings and participate in
cross-run matching the same way manually enrolled speakers do.

Adjudication decisions from auto-enrollment use `Decision.AUTO_ENROLL` and
`Resolution.AUTO_ENROLL`, keeping them distinct from human rulings in the
ledger.

### Backfill

Existing completed runs with unresolved labels can be retroactively processed:

```bash
docker compose exec api voxint speakers auto-enroll-backfill --dry-run   # preview: rolls back all changes
docker compose exec api voxint speakers auto-enroll-backfill --limit 10  # process at most 10 runs
docker compose exec api voxint speakers auto-enroll-backfill             # process all eligible runs
docker compose exec api voxint speakers auto-enroll-backfill --run-id <uuid>  # single run
```

`--dry-run` runs the full enrollment logic inside a transaction and rolls back,
printing per-run counts (created, matched, skipped) without persisting anything.
The backfill processes runs oldest-first and is safe to interrupt: each run
commits independently.

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
   the speaker-attributed transcript (`/review/{run_id}/export.{txt,md,srt,vtt,
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
| `GET /metrics` | Prometheus text exposition (aggregate DB gauges plus `voxint_gpu_*` / `voxint_service_admission_*` hardware gauges; authenticated, scrape with `basic_auth`) |
| `GET /` | Home: needs-attention cards (continue review, unidentified voices, failed runs), quick actions, windowed activity counts (`?window=hour|day|week|all`), recent activity |
| `GET /dashboard` | Permanent 303 redirect to `/` (the dashboard folded into Home) |
| `GET /resources` | Permanent 303 redirect to `/settings/status` (the hardware view folded into Settings) |
| `GET /runs` | Execution-history browser (keyset-paged; `status=` / `review=` filters) |
| `GET /runs/{run_id}` | Run detail + per-stage attempt ledger |
| `GET /runs/{run_id}/transcript?text=raw\|enhanced` | Resolver-attributed transcript (HTML); `&read=1&timestamps=false` renders the on-screen read-mode prose view |
| `POST /submit` | Bounded browser file upload → immutable uuid-namespaced media item (redirects to `/media` when `console_media_enabled` is on) |
| `POST /fetch` | yt-dlp URL ingestion (redirects to `/media` when `console_media_enabled` is on) |
| `POST /runs/{run_id}/requeue` | Exact-revision (CAS) requeue of a FAILED run |
| `GET /review` | Review queue |
| `POST /review/{run_id}/claim` · `/release` | Claim / release an exclusive review slot |
| `GET /review/{run_id}` | Adjudication workbench |
| `POST /review/{run_id}/labels/{label}/decision` | Record a human ruling for a label |
| `POST /review/{run_id}/labels/{label}/enroll` | Enroll a label's audio as a roster speaker |
| `GET /review/{run_id}/export.{txt,md,srt,vtt,json}?text=corrected\|raw\|enhanced` | Speaker-attributed transcript export (plain text, Markdown, SubRip, WebVTT, JSON); `txt`/`md` accept `&timestamps=false` for the reading copy |
| `GET /review/{run_id}/export.rttm` | Diarization RTTM (raw labels, run-UUID file id) |
| `GET /media/{id}/editor` | Media detail page: run selection (latest completed by default, `?run=` override), claim-token verification (stale/absent = read-only), transcript with speaker palette and verified-progress counter, run chooser, media metadata rail (#156) |
| `GET /media/{run_id}` | Gated media serving (Range-aware) for the workbench player |
| `GET /setup` · `POST /setup/{media,scan,vocabulary,llm,finish}` | First-run setup wizard; held by the onboarding gate until finished (own `CSRF_SETUP` token) |
| `GET /settings` | Post-onboarding settings hub: edit features / media folders / corrections / LLM / sources, re-run the wizard, start/replay/complete the tutorial, and reach the read-only sub-pages below |
| `GET /settings/status` | Status and health: install kind, live component health (Postgres / Redis / model services), and the live hardware snapshot (absorbs the old `/resources`; answers an `HX-Request` poll with just the hardware fragment, 15s auto-refresh) |
| `GET /settings/features` | 303 redirect to `/settings#features` (the Features section on the hub page) |
| `GET /settings/{hardware,database,plugins}`, `GET /settings/plugins/{id}` | Read-only settings sub-pages: effective hardware config, database size/retention, and the plugin registry |
| `GET /activity/events?since={id}` | Activity feed poll (JSON): run-completion and speaker-identification events after a cursor plus the live-jobs badge count. Dark-shipped behind `CONSOLE_ACTIVITY_ENABLED` (answers 404 until on); no `since` bootstraps at the high-water mark so a fresh tab does not replay history (#162) |
| `POST /settings/tutorial/{complete,replay}` | Complete / non-destructively replay the guided tutorial (own `CSRF_SETTINGS` token) |
| `POST /settings/corrections` | Replace the operator's console-authored correction rules (#84; whole list validated through the pack #80 gate, own `CSRF_SETTINGS` token; a pack collision returns a plain-language 422) |

## Media retention / garbage collection (issue #15; off by default)

Storage grows with every run: the pipeline writes a normalized 16 kHz WAV
intermediate (`artifacts/{run_id}/normalized.wav`) that transcription and
diarization read, and nothing reclaims it. When enabled, a beat-scheduled GC
sweep reclaims that intermediate for **old terminal runs**: it unlinks the WAV
and stamps the `audio_artifacts` row (`reclaimed_at`, `reclaimed_bytes`) as an
audit record; the row itself is never deleted.

**What is reclaimed:** the normalized-audio intermediate of runs that are
`completed` or `cancelled` and untouched for `MEDIA_RETENTION_SECONDS`, plus any
extracted audio clips (issue #88) aged past the same horizon. A clip is aged by
its **own** `created_at`, not the run's, so a clip freshly extracted on an old
terminal run is not immediately reclaimed; its row survives reclamation and the
clip serve route then answers `410 Gone`.

**What is always kept:** the **source media** (so a reclaimed run stays
re-processable; re-submit it to regenerate the intermediate and downstream
results), the transcript, diarization turns, speaker assignments, and the
immutable adjudication decision ledger. Runs mid-pipeline, `failed`/requeue-able
runs, the guided-tutorial run, and any file that is also registered as a run's
source are all excluded.

**It is off by default**: no bytes are reclaimed until you opt in. In the
console, a run whose intermediate was reclaimed shows a "Media reclaimed on
`<date>`" notice instead of the audio link, and `GET /media/{run_id}` returns
`410 Gone`.

```dotenv
# .env: enable and tune (defaults shown)
MEDIA_RETENTION_ENABLED=true      # off unless set
MEDIA_RETENTION_SECONDS=2592000   # 30 d; floor 3600 (1 h)
GC_SWEEP_SECONDS=3600             # how often the sweep runs
GC_BATCH_LIMIT=500                # rows per sweep, oldest-first
```

A backlog drains at `GC_BATCH_LIMIT` rows per `GC_SWEEP_SECONDS` (the sweep
processes one bounded, oldest-first batch per run and is safe to run
concurrently; rows are claimed with `FOR UPDATE ... SKIP LOCKED`). To reclaim a
large accumulated backlog faster, raise `GC_BATCH_LIMIT` or lower
`GC_SWEEP_SECONDS` until it catches up.

To reclaim a single run's derived audio **immediately** (rather than waiting for
the sweep), use the manual **Delete derived audio files** action on the run
detail page (`POST /runs/{id}/media/delete`, issue #5, above). The manual action
deletes the rows and files outright; the scheduled sweep keeps the row and
stamps `reclaimed_at` as an audit record. Archived runs remain eligible for the
sweep: archiving only hides a run from the console, it does not exempt its
intermediate from reclamation.

## Run notifications / webhooks (issue #12; off by default)

Voxint can POST a **signed webhook** to an endpoint you control when a run
reaches a **notifiable transition** (`completed` or `failed`). It is opt-in and
off by default; nothing is emitted or delivered until you enable it, and
enabling later never back-fills runs that finished while it was off.

**How it is delivered (at-least-once).** The notification is recorded as an
outbox row **in the same database transaction as the run's state change**, so
delivery intent is atomic with the transition: a run never "completes" without
its notification queued, and a rolled-back transition takes its notification with
it. A beat-scheduled sweep then delivers each row **outside any transaction**, so
a slow or down receiver never blocks the pipeline. Because delivery is retried
until it succeeds, your receiver **can see the same delivery more than once**, so
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
# .env: enable and tune (defaults shown)
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
| `X-Voxint-Delivery` | The `delivery_id`, your idempotency key. |
| `X-Voxint-Timestamp` | Unix seconds when the POST was signed. |
| `X-Voxint-Signature` | `sha256=<hex>`, HMAC-SHA256 of `timestamp + "." + body`. |

The body is `{"schema_version", "event", "run_id", "transition_revision",
"occurred_at", "delivery_id"}`. It deliberately **omits the run's error text**
(it can carry sensitive detail); look up the run by `run_id` for detail.

**Verifying a delivery (receiver side).** Recompute the signature over the
**raw request bytes** (parse-and-re-serialize would change them and the check
would fail), compare in constant time, and reject a stale timestamp:

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
