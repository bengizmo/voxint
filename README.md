# Voxint

**From sound to intelligence: end-to-end transcription, diarization, and speaker identity with
human-grade quality gates.**

Voxint turns any audio or video file into an enhanced, speaker-attributed transcript:

```
media file → preprocess → transcribe (Whisper) + diarize (pyannote) + embed (TitaNet)
           → LLM transcript enhancement → speaker matching → human adjudication
```

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

To run the GPU model services too (one NVIDIA GPU assumed), first set
`HF_TOKEN` in `.env` — the pyannote service's diarization weights are
HF-gated, so you need a Hugging Face token with access to the pyannote
models accepted (see `services/pyannote/README.md`); compose refuses the GPU
overlay without it. Then:

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

## Deployment model

Docker-compose-first on a single Linux machine with one NVIDIA GPU:

- `compose.yaml` — Postgres (+pgvector), Redis, API (+ review UI), Celery worker
- `compose.gpu.yaml` — the GPU model services: faster-whisper, pyannote, TitaNet

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
