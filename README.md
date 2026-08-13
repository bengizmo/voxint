# Voxint

**From sound to intelligence: end-to-end transcription, diarization, and speaker identity with
human-grade quality gates.**

Voxint turns any audio or video file into an enhanced, speaker-attributed transcript:

```
local file · upload · URL → acquire → preprocess
    → transcribe (Whisper) + diarize (pyannote) + embed (TitaNet)
    → LLM transcript enhancement → speaker matching → human adjudication
```

Point it at a local file (`voxint submit`), upload one through the browser, or hand it a
URL (`voxint fetch` / `POST /fetch`) — a yt-dlp download runs as the pipeline's first stage.
URL ingestion is authenticated **admin egress, not a sandbox**: fetch only trusted URLs
unless you run the worker with restricted egress (no route to private / link-local /
metadata addresses). See [docs/operations.md](docs/operations.md#url-ingestion--egress-security).

What makes it different is the orchestration "glue" most pipelines skip:

- **Quality gates at every stage** — non-speech/digital-silence triage before you burn GPU time,
  hallucination soft-tagging and stripping, chunk-completeness checks, outage-vs-data-defect
  taxonomy with explicit retry budgets.
- **Durable state, not vibes** — a compare-and-swap'd run/stage state machine in Postgres; a crash
  at any stage is recoverable, and human pauses are database state, never a held task.
- **Speaker identity done honestly** — pgvector cosine matching against a grown speaker roster,
  a strict *named ≠ grounded* invariant, and machine proposals kept separate from human rulings.
- **A built-in adjudication web UI** — review queue, guarded slot workbench, and an immutable
  decision ledger, served as Jinja + htmx from the same FastAPI app (no Node toolchain).
- **Operable from the browser** — a keyset-paged `/runs` execution-history browser (with a
  per-stage attempt ledger), bounded file upload, and yt-dlp URL ingestion, from the same app.
  Submission is durable-first: a broker outage leaves the run queued for the recovery sweep,
  never lost. The console is append-only — no delete, no cancel.
- **Measurement harnesses** — name-accuracy scoring (McNemar / bootstrap / Wilson) and a
  golden-dataset agreement labeler, runnable as CLIs (worked example under
  [`examples/`](examples/README.md)).

## The adjudication console

Machine proposals stay separate from human rulings: the review queue lists completed runs
with voices still needing a ruling, and the slot workbench shows each voice's evidence —
grounded cosine matches, unverified LLM-heard names, transcript previews — with
assign / enroll / exclude / unknown actions. (Synthetic demo data pictured.)

![Adjudication queue](docs/images/review-queue.png)

![Slot workbench](docs/images/slot-workbench.png)

## Status

**Pre-alpha.** APIs, schema, and layout may change without notice through the 0.x series.

## Quickstart

Requires Docker Engine with the **Compose plugin ≥ 2.24** (`docker compose
version` — the legacy v1 `docker-compose` binary cannot parse this stack).

```bash
git clone https://github.com/bengizmo/voxint.git && cd voxint
```

**Guided install (recommended for a first run):**

```bash
./scripts/install.sh
```

It asks only for an admin password and a media folder, generates everything
else (including a random `CSRF_SECRET`), pulls the pinned release images, starts
the stack, waits for the API to report healthy, and prints the console URL. It
is safe to re-run — an existing `.env` is kept unless you ask to regenerate it
(which backs the old one up first). This brings up the **core control plane**
(console, review UI, durable pipeline state) — enough to open the console and
adjudicate; audio processing additionally needs the GPU model services (below).

**Or configure by hand:**

```bash
cp .env.example .env          # then edit at least VOXINT_PASSWORD
mkdir -p media                # media mount; pre-create so it isn't root-owned
docker compose pull           # prebuilt release images from GHCR
docker compose up -d          # Postgres+pgvector, Redis, migrate, API + review UI, worker, beat
curl http://127.0.0.1:8080/healthz   # default port; matches API_PORT if you changed it
```

The default compose files run the **pinned release images** — even from a
`main` checkout (set `VOXINT_IMAGE_TAG` in `.env` to run a different
release). A one-shot `migrate` service brings the schema to head before the
API and worker start — it showing `Exited (0)` in `docker compose ps -a` is
success, not a crash. If a default port is already in use on your host,
override the published side in `.env` (`POSTGRES_PORT`, `REDIS_PORT`,
`API_PORT`). Details and day-2 operations:
[docs/operations.md](docs/operations.md).

Open the console at `http://127.0.0.1:8080/` (HTTP Basic, the `VOXINT_USER` /
`VOXINT_PASSWORD` you set). On a **fresh install** the console holds you at a
first-run **setup wizard** (`/setup`) — configure media folders, vocabulary, and
optional LLM enhancement in the browser, then finish into a short guided tutorial
on a bundled three-speaker sample. Full walkthrough:
[docs/onboarding.md](docs/onboarding.md).

Once onboarding is complete, browse runs at `/runs` and adjudicate at `/review`.
Feed it work by uploading a file, pointing it at a URL (`docker compose exec api
voxint fetch <url>`), or submitting a local path (`docker compose exec api voxint
submit path/to/file.mp3`, relative to `MEDIA_ROOT`).

To run the GPU model services too (one NVIDIA GPU assumed), first set
`HF_TOKEN` in `.env` — the pyannote service's diarization weights are
HF-gated, so you need a Hugging Face token with access to the pyannote
models accepted (see `services/pyannote/README.md`); compose refuses the GPU
overlay without it.

All three services share the one GPU. Their loaded weights total roughly
**3.5–4.5 GB** of VRAM (whisper large-v2 int8 ~1.5 GB, pyannote ~1–2 GB,
TitaNet ~1 GB); budget **~6–8 GB in practice** for Whisper's batch/decode
headroom and three separate CUDA contexts. An 8 GB card is comfortable.
(Per-service figures live in each `services/*/README.md`.)

Then:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Per-service details, env tunables, and image matrices:
`services/*/README.md`; wire contracts:
[docs/gpu-contracts.md](docs/gpu-contracts.md).

To run the source you checked out instead of the release images, layer the
build overlays (exactly one service owns each build — see
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

The scoring harness needs none of the stack — `pip install voxint` gives you
the `voxint score` CLI (pure file-in/file-out, no database or GPU services);
see [`examples/`](examples/README.md).

## Deployment model

Docker-compose-first on a single Linux machine with one NVIDIA GPU:

- `compose.yaml` — Postgres (+pgvector), Redis, one-shot `migrate`, API (+ review UI),
  Celery worker, Celery beat (crash-recovery sweep scheduler)
- `compose.gpu.yaml` — the GPU model services: faster-whisper, pyannote, TitaNet
- `compose.build.yaml` / `compose.gpu.build.yaml` — build-from-source overlays for development

Kubernetes is explicitly **not** required (a future optional enhancement).

## Modularity

ASR, diarizer, embedder, and LLM providers sit behind typed protocols with versioned HTTP
contracts (`/v1/transcribe`, `/v1/diarize`, `/v1/embed`). The LLM enhancement stage speaks to any
OpenAI-compatible endpoint and is optional (`LLM_ENABLED=false` by default). Domain-specific
vocabulary and prompts load from a swappable **domain pack** (`DOMAIN_PACK_PATH`); a neutral
meeting/podcast pack ships as the default.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) — model weights (e.g. pyannote's
HF-gated checkpoints) are subject to their own terms and are downloaded with **your** credentials;
Voxint never vendors them.
