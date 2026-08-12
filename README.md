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

## Status

**Pre-alpha — under active extraction.** APIs, schema, and layout will change without notice
until `v0.1.0`.

## Quickstart

Requires Docker Engine with the **Compose plugin ≥ 2.24** (`docker compose
version` — the legacy v1 `docker-compose` binary cannot parse this stack).

```bash
cp .env.example .env          # then edit at least VOXINT_PASSWORD
mkdir -p media                # media mount; pre-create so it isn't root-owned
docker compose up -d          # Postgres+pgvector, Redis, migrate, API + review UI, worker, beat
curl http://127.0.0.1:8080/healthz   # default port; matches API_PORT if you changed it
```

A one-shot `migrate` service brings the schema to head before the API and
worker start — it showing `Exited (0)` in `docker compose ps -a` is success,
not a crash. If a default port is already in use on your host, override the
published side in `.env` (`POSTGRES_PORT`, `REDIS_PORT`, `API_PORT`). Details
and day-2 operations: [docs/operations.md](docs/operations.md).

To run the GPU model services too (one NVIDIA GPU assumed):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

The pyannote service needs `HF_TOKEN` in `.env` (its diarization weights are
HF-gated — see `services/pyannote/README.md`). Per-service details, env
tunables, and image matrices: `services/*/README.md`; wire contracts:
[docs/gpu-contracts.md](docs/gpu-contracts.md).

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
