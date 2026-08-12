# Operations

Running Voxint day to day: deployment, schema migrations, pipeline
operations, recovery, and the adjudication workflow. Architecture and data
model: [architecture.md](architecture.md); gate semantics:
[quality-gates.md](quality-gates.md).

## Deployment

Prerequisites: Docker Engine with the **Compose plugin ≥ 2.24** (`docker
compose version`; the legacy v1 `docker-compose` binary cannot parse this
stack, and older v2 plugins lack its optional-`env_file` syntax), and for the
GPU overlay an NVIDIA GPU with the NVIDIA container toolkit.

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
# app + GPU model services:
docker compose -f compose.yaml -f compose.gpu.yaml \
               -f compose.build.yaml -f compose.gpu.build.yaml \
               build api whisper pyannote titanet
docker compose -f compose.yaml -f compose.gpu.yaml \
               -f compose.build.yaml -f compose.gpu.build.yaml up -d
```

Exactly one service (`api`) declares the app `build:`; migrate/worker/beat
consume the tag it produces. Do not give several `build:` services a shared
`image:` tag — concurrent BuildKit writers race on it ("already exists").

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
docker compose exec api voxint submit path/to/file.mp3   # path relative to MEDIA_ROOT; prints run id
docker compose exec api voxint status <run-id>           # run state + per-stage attempt ledger
docker compose exec api voxint requeue <run-id>          # re-enter a FAILED run at its failed stage
```

`submit` records the media item, creates a run, and enqueues it for the
worker — the CLI never executes a stage itself. `status` shows the run's
current state plus every stage attempt with its error, straight from the
persisted ledger.

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

## Adjudication workflow

The review console is served by the API at `http://127.0.0.1:8080/` (or your
`API_PORT` override; basic auth, `VOXINT_USER`/`VOXINT_PASSWORD` — single
reviewer credential):

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
   the speaker-attributed transcript (`/review/{run_id}/export.txt`).

### HTTP endpoints

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness (no DB access — schema readiness is the migrate gate's job) |
| `GET /review` | Review queue |
| `POST /review/{run_id}/claim` · `/release` | Claim / release an exclusive review slot |
| `GET /review/{run_id}` | Adjudication workbench |
| `POST /review/{run_id}/labels/{label}/decision` | Record a human ruling for a label |
| `POST /review/{run_id}/labels/{label}/enroll` | Enroll a label's audio as a roster speaker |
| `GET /review/{run_id}/export.txt` | Speaker-attributed transcript export |
| `GET /media/{run_id}` | Gated media serving (Range-aware) for the workbench player |

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
