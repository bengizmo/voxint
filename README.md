# Voxint

**Self-hosted audio intelligence: turn any recording into a speaker-attributed
transcript, then review it by hand.** Transcription, diarization, and speaker
identity, running entirely on your own hardware.

Built for individuals and small teams (researchers, journalists, educators) who
need audio work to stay local: no cloud account, no per-minute fees, no
recordings leaving the room.

Voxint takes an audio or video file and produces an enhanced transcript with
speakers attributed:

```
local file · upload · URL → acquire → preprocess
    → transcribe (Whisper) + diarize (pyannote) + embed (TitaNet)
    → LLM transcript enhancement → speaker matching → human adjudication
```

Point it at a local file (`voxint submit`), upload one through the browser, or
hand it a URL (`voxint fetch` / `POST /fetch`), which runs a yt-dlp download as
the pipeline's first stage. URL ingestion is authenticated admin egress, not a
sandbox. Fetch only trusted URLs unless you run the worker with restricted
egress (no route to private, link-local, or metadata addresses). See
[docs/operations.md](docs/operations.md#url-ingestion--egress-security).

## What it does that a bare pipeline doesn't

Most of Voxint is the orchestration around the models, the part that usually
gets left as an exercise for the reader.

- **Quality gates.** Non-speech and digital-silence triage before you spend GPU
  time, hallucination soft-tagging and stripping, chunk-completeness checks, and
  an outage-vs-data-defect taxonomy with explicit retry budgets.
- **Durable state.** A compare-and-swap run/stage state machine in Postgres. A
  crash at any stage is recoverable, and a human pause is a database row, not a
  held task.
- **Speaker identity with a paper trail.** pgvector cosine matching against a
  speaker roster that grows as you use it, a strict *named ≠ grounded*
  invariant, and machine proposals kept separate from human rulings.
- **A built-in review console.** Review queue, guarded slot workbench, and an
  immutable decision ledger, served as Jinja + htmx from the same FastAPI app.
  No Node toolchain.
- **Operable from the browser.** A keyset-paged `/runs` history browser (with a
  per-stage attempt ledger), bounded file upload, and yt-dlp URL ingestion, all
  from the same app. Submission is durable-first: a broker outage leaves the run
  queued for the recovery sweep rather than dropping it. You can cancel a live
  run (cooperative, exact-revision CAS), soft-archive a terminal one (reversibly
  hidden, ledger intact), and delete its derived audio to reclaim disk without
  touching the shared original.
- **Measurement harnesses.** Speaker-attribution scoring you can run as CLIs:
  name-accuracy against ground truth (McNemar / bootstrap / Wilson), acoustic
  agreement verdicts, and verdict-level ensemble fusion (worked example under
  [`examples/`](examples/README.md)). These score *who spoke*, not *what was
  transcribed*. ASR accuracy and WER measurement are out of scope for now.

## The review console

Machine proposals stay separate from human rulings. The review queue lists
completed runs with voices still needing a decision, and the slot workbench
shows each voice's evidence (grounded cosine matches, unverified LLM-heard
names, transcript previews) with assign / enroll / exclude / unknown actions.
Synthetic demo data pictured.

![Adjudication queue](docs/images/review-queue.png)

![Slot workbench](docs/images/slot-workbench.png)

## Status

**Pre-alpha.** APIs, schema, and layout may change without notice through the
0.x series.

## Quickstart

Requires Docker Engine with the Compose plugin ≥ 2.24 (`docker compose version`;
the legacy v1 `docker-compose` binary cannot parse this stack).

> **No NVIDIA GPU? Start here too.** Voxint does not need one. The CPU tier runs
> the full pipeline on plain amd64/arm64 servers and Apple Silicon with zero GPU
> configuration. Follow the same quickstart, then use the `compose.cpu.yaml`
> overlay wherever the GPU one appears (see
> [No NVIDIA GPU? (CPU tier)](#no-nvidia-gpu-cpu-tier)). **AMD GPU?** The ROCm
> tier accelerates transcription on it (4.8x the CPU baseline; the amdgpu kernel
> driver is the only host requirement) via `compose.rocm.yaml` (see
> [AMD GPU? (ROCm tier)](#amd-gpu-rocm-tier)). **Apple Silicon Mac?** The metal
> tier runs the model services natively so diarization uses the Apple GPU (see
> [Apple Silicon Mac? (metal tier)](#apple-silicon-mac-metal-tier)).

```bash
git clone https://github.com/bengizmo/voxint.git && cd voxint
```

**Guided install (recommended for a first run):**

```bash
./scripts/install.sh
```

It asks for an admin password, a media folder, and a compute tier for the model
services (GPU / CPU / none for now). That's the whole interview. All model
weights, diarization included, are vendored into the images, so no Hugging Face
account or token is involved. The installer generates everything else (including
a random `CSRF_SECRET`), pulls the pinned release images, starts the core stack
plus your chosen tier's model services, waits for the API to report healthy, and
prints the console URL. Re-running it is safe: an existing `.env` is kept unless
you ask to regenerate it (which backs the old one up first), and your tier
choice is remembered (`VOXINT_COMPOSE_TIER`).

**Or configure by hand:**

```bash
cp .env.example .env          # then edit at least VOXINT_PASSWORD
mkdir -p media                # media mount; pre-create so it isn't root-owned
docker compose pull           # prebuilt release images from GHCR
docker compose up -d          # Postgres+pgvector, Redis, migrate, API + review UI, worker, beat
curl http://127.0.0.1:8080/healthz   # default port; matches API_PORT if you changed it
```

The default compose files run the pinned release images even from a `main`
checkout (set `VOXINT_IMAGE_TAG` in `.env` to run a different release). A
one-shot `migrate` service brings the schema to head before the API and worker
start. Seeing it report `Exited (0)` in `docker compose ps -a` is success, not a
crash. If a default port is already in use on your host, override the published
side in `.env` (`POSTGRES_PORT`, `REDIS_PORT`, `API_PORT`). Details and day-2
operations live in [docs/operations.md](docs/operations.md).

Open the console at `http://127.0.0.1:8080/` (HTTP Basic, the `VOXINT_USER` /
`VOXINT_PASSWORD` you set). On a fresh install the console holds you at a
first-run setup wizard (`/setup`): configure media folders, vocabulary, and
optional LLM enhancement in the browser, then finish into a short guided
tutorial on a bundled three-speaker sample. Full walkthrough:
[docs/onboarding.md](docs/onboarding.md).

Once onboarding is complete, browse runs at `/runs` and adjudicate at `/review`.
Feed it work by uploading a file, pointing it at a URL (`docker compose exec api
voxint fetch <url>`), or submitting a local path (`docker compose exec api
voxint submit path/to/file.mp3`, relative to `MEDIA_ROOT`).

To run the GPU model services too (one NVIDIA GPU assumed), bring up the GPU
overlay. The diarization weights are vendored into the pyannote image
(sha256-pinned from the `pyannote-models-v1` asset release), so no Hugging Face
token is needed (see `services/pyannote/README.md`).

All three services share the one GPU. Their loaded weights total roughly 3.5-4.5
GB of VRAM (whisper large-v2 int8 ~1.5 GB, pyannote ~1-2 GB, TitaNet ~1 GB);
budget ~6-8 GB in practice for Whisper's batch/decode headroom and three
separate CUDA contexts. An 8 GB card is comfortable. Per-service figures live in
each `services/*/README.md`.

Then:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Per-service details, env tunables, and image matrices are in
`services/*/README.md`; wire contracts are in
[docs/gpu-contracts.md](docs/gpu-contracts.md).

### No NVIDIA GPU? (CPU tier)

The same three model services ship as multi-arch (amd64 + arm64) `-cpu` images:
no GPU, no NVIDIA toolkit, runs on plain servers and Apple Silicon via Docker
Desktop.

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

Set your expectations accordingly: CPU inference is orders of magnitude slower.
A long recording that takes minutes on a GPU takes hours on CPU. The overlay
sets `COMPUTE_TIER=cpu`, which scales the pipeline's timeouts and stage leases so
slow-but-healthy runs aren't reclaimed as hung. Same contracts, same embedding
space (TitaNet runs on ONNX Runtime under a measured-equivalence parity gate).
Details:
[docs/operations.md](docs/operations.md#running-without-an-nvidia-gpu-cpu-tier).

### AMD GPU? (ROCm tier)

A hybrid tier for amd64 hosts with an AMD GPU. Transcription (whisper) runs on
the GPU via the CTranslate2 ROCm build (same engine, same code path, measured
4.8x the CPU baseline on RDNA4), while diarization and speaker embedding run the
`-cpu` images. MIOpen convolutions currently fail on AMD consumer GPUs, tracked
in [#4](https://github.com/bengizmo/voxint/issues/4):

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d
```

The host needs only the amdgpu kernel driver: no ROCm install, no container
toolkit, since the `-rocm` image carries its own ROCm runtime. The overlay sets
`COMPUTE_TIER=rocm` (GPU-speed ASR, CPU-scaled leases for the rest). Details:
[docs/operations.md](docs/operations.md#running-on-an-amd-gpu-rocm-tier).

### Apple Silicon Mac? (metal tier)

Docker Desktop has no GPU passthrough, so on a Mac the containerized tiers are
CPU-only. The metal tier keeps the core stack in Docker but runs the three model
services natively so diarization uses the Apple GPU (torch-MPS, measured ~5x
native-CPU diarization on an M1 Pro, identical outputs). Transcription stays on
the host CPU in v1, so runs remain transcribe-bound: faster than the Docker CPU
tier, not GPU-stack fast.

```bash
./scripts/install.sh                  # choose [M]
./scripts/metal/voxint-metal.sh setup # native venvs + sha-verified weights
./scripts/metal/voxint-metal.sh up    # services under launchd
```

Weights come from the same sha-pinned release assets the images use, so still no
Hugging Face account or token. Details:
[docs/operations.md](docs/operations.md#running-on-apple-silicon-metal-tier).

To run the source you checked out instead of the release images, layer the build
overlays (exactly one service owns each build; see
[docs/operations.md](docs/operations.md)):

```bash
docker compose -f compose.yaml -f compose.build.yaml build api
docker compose -f compose.yaml -f compose.build.yaml up -d
```

For development without Docker:

```bash
uv sync --extra dev
uv run pytest tests/unit
uv run uvicorn voxint.api.app:app --reload
```

The scoring harness needs none of the stack. `pip install voxint` gives you the
`voxint score` CLI (pure file-in/file-out, no database or GPU services;
speaker-attribution metrics only, no ASR/WER). See
[`examples/`](examples/README.md).

## Deployment model

Docker-compose-first on a single Linux machine with one NVIDIA GPU:

- `compose.yaml`: Postgres (+pgvector), Redis, one-shot `migrate`, API (+ review
  UI), Celery worker, Celery beat (crash-recovery sweep scheduler)
- `compose.gpu.yaml`: the GPU model services (faster-whisper, pyannote, TitaNet)
- `compose.build.yaml` / `compose.gpu.build.yaml`: build-from-source overlays for
  development

Kubernetes is not required. It may become an optional enhancement later.

## Modularity

ASR, diarizer, embedder, and LLM providers sit behind typed protocols with
versioned HTTP contracts (`/v1/transcribe`, `/v1/diarize`, `/v1/embed`). The LLM
enhancement stage speaks to any OpenAI-compatible endpoint and is optional
(`LLM_ENABLED=false` by default). Domain-specific vocabulary and prompts load
from a swappable domain pack (`DOMAIN_PACK_PATH`); a neutral meeting/podcast pack
ships as the default.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Vendored model weights
are redistributed under their own licenses with attribution (titanet:
CC-BY-4.0; pyannote segmentation: MIT; WeSpeaker embedding: CC-BY-4.0). See the
provenance files under `services/*/models/` and the model-asset releases.
