# Testing

How Voxint is tested, how to run each layer locally, and the manual procedure for
browser-verifying the review console. Numerics changes have their own, stricter
doctrine — see [`gpu-contracts.md`](gpu-contracts.md) and the parity notes below.

## Test layers

| Layer | Path | What it covers | Needs |
|---|---|---|---|
| Unit | `tests/unit/` | Pure logic: config parsing, CLI, API helpers, formatters, scoring, redaction, review-auth, validation. No database. | nothing |
| Contracts | `tests/contracts/` | Invariants that would rot silently: version-pin parity across pyproject/compose/`.env.example`, Dockerfile sha ARGs ↔ provenance, restart policies, routes/schemas, the **frontend build/island wiring** (`test_frontend_build.py`). | nothing |
| Integration | `tests/integration/` | Real Postgres + the alembic chain. Every API/console behaviour is exercised here (submission, adjudication, verify-and-advance, run-assets, dashboard, migrations). | a pgvector database |
| Parity | `tests/parity/` | Model-output equivalence gates (mel / vector / decision) against committed CUDA references. Real audio fixtures live under `tests/parity/fixtures/`. | strict mode: `VOXINT_PARITY_REQUIRED=1` |
| E2E | `tests/e2e/` | The **real** pipeline against the **real** model services (faster-whisper + pyannote + TitaNet in their containers): submit the tutorial clip, run every stage, assert the persistence invariants. Plus a **real-LLM** enrichment lane (real `HttpLLMClient` → real endpoint) that gates the summary chain. Maintainer-run, opt-in gate — **never public CI**. | `VOXINT_E2E=1` + `VOXINT_TEST_DATABASE_URL` + the model services running; the LLM lane also needs the enrichment LLM env (see below) |

The layout is the standard `pytest` tree; add a test in the same commit that adds
the behaviour or invariant it guards (a new island → a row in
`tests/contracts/test_frontend_build.py`; a new contract → a `tests/contracts/`
test).

## Running the suite

Unit and contract tests need nothing external:

```bash
uv run --extra dev pytest tests/unit tests/contracts -q
```

Integration tests need a Postgres with the `vector` extension. They read
**`VOXINT_TEST_DATABASE_URL`** and are **skipped entirely when it is unset** (so a
bare `pytest` run still passes without a database, and CI supplies the service).
The `engine` fixture drops and recreates the `public` schema then runs
`alembic upgrade head`; each test truncates all tables afterward, so the suite is
safe to point at a throwaway database — **never the live `voxint` database**:

```bash
# One-time: a disposable database beside your dev one.
docker compose exec -T postgres psql -U voxint -d voxint \
  -c "CREATE DATABASE voxint_dev_test"
docker compose exec -T postgres psql -U voxint -d voxint_dev_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector"

export VOXINT_TEST_DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_dev_test"
uv run --extra dev pytest tests -q -p no:warnings
```

Static gates (run these before landing anything non-trivial):

```bash
cd frontend && npm run typecheck && npm run lint && npm run build && cd ..
uv run ruff check src tests
uv run mypy            # CI form: packages=voxint (do NOT add tests — the parity
                       # stub files carry pre-existing, tolerated stub errors)
```

The current integration tests exercise the real stage implementations against
real Postgres and real ffmpeg but with **fake model providers** (`tests/fakes.py`:
`FakeASR` / `FakeDiarizer` / `FakeEmbedder` / `FakeLLM` / `FailingLLM`) — see
`tests/integration/test_real_stages_e2e.py`. There is deliberately **no frontend
test runner** (no vitest/jest): island behaviour is covered by the Python
integration tests plus the manual browser pass below. Don't add one without
discussing it — it is bloat this single-operator app does not need.

## Manual browser verification of the review console

Interactive island behaviour (the #53/#58 verify-and-advance loop, click-to-edit,
the unsaved-edit discard warning) is confirmed by driving a real browser against a
**local** instance. This is a manual procedure today; the browser lane of the
[automated E2E suite](#automated-e2e-testse2e) is planned to replace it (the
real-pipeline lane has landed; the browser lane has not).

The dockerized `api` service runs the **released** image, not your working tree,
so browser-verifying a local change means running a fresh local instance:

1. **Build the islands and stage them where the app serves them.** The app reads
   the Vite manifest once at import, so copy *before* starting the server. These
   are build artifacts — **do not commit them**; restore the `.gitkeep` and remove
   them afterward.
   ```bash
   cd frontend && npm run build && cd ..
   cp -r frontend/dist/. src/voxint/api/static/app/     # overlays .vite/ + assets/
   ```
2. **Create a throwaway database, migrate it, and complete onboarding** (with the
   LLM off unless you are testing enrichment):
   ```bash
   docker compose exec -T postgres psql -U voxint -d voxint -c "CREATE DATABASE voxint_e2e"
   docker compose exec -T postgres psql -U voxint -d voxint_e2e -c "CREATE EXTENSION IF NOT EXISTS vector"
   export DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_e2e"
   uv run alembic upgrade head
   uv run python -c "from sqlalchemy import create_engine; from sqlalchemy.orm import Session; \
   from voxint.app_settings import complete_onboarding; import os; \
   e=create_engine(os.environ['DATABASE_URL']); \
   s=Session(e); complete_onboarding(s, llm_enabled_default=False); s.commit()"
   ```
3. **Seed a completed run** with an audio artifact and a handful of segments at
   varied confidence (some below the low-confidence threshold, so the "uncertain"
   chips appear). **Set the media item's `duration_seconds`** — without it
   `playback_capability` gates seeking off and the player cannot follow along.
   (The current seed is an ad-hoc script; the planned automated suite commits a
   shared fixture set. Mirror the existing shape in
   `tests/integration/test_review_api.py` — `seed_run` /
   `_seed_run_with_confidences`.)
4. **Serve locally on a spare port** with its own media root and basic-auth:
   ```bash
   DATABASE_URL="…voxint_e2e" MEDIA_ROOT="$PWD/media-e2e" \
   VOXINT_USER=admin VOXINT_PASSWORD=e2epass API_PORT=8099 \
   uv run voxint serve
   ```
5. **Drive it.** Navigate once with the credentials embedded
   (`http://admin:e2epass@127.0.0.1:8099/`) to cache basic-auth, then **re-navigate
   to the clean URL** (no embedded credentials) — an island `fetch()` throws
   "URL includes credentials" if the document URL carries them (a test-harness
   artifact, not a product bug). Claim the run from the workbench → **Review
   transcript →** → exercise the loop: `v` verify-and-advance, `e` edit +
   `⌘/Ctrl+Enter` save, `n` skip, `p` replay; click a line to move the edit cursor
   (verified lines are re-reachable); type an unsaved edit then verify to see the
   discard warning; focus the playback-speed `<select>` and press `v` to confirm
   the keymap does **not** fire from a form control.
6. **Clean up.** Kill the local server **by port** — `fuser -k 8099/tcp` — **not**
   `pkill -f "voxint serve"`, which also matches and restarts the dockerized `api`
   container. Then drop the throwaway database, remove the copied build artifacts,
   and restore the placeholder:
   ```bash
   fuser -k 8099/tcp
   docker compose exec -T postgres psql -U voxint -d voxint -c "DROP DATABASE IF EXISTS voxint_e2e"
   rm -rf src/voxint/api/static/app/.vite src/voxint/api/static/app/assets media-e2e
   git checkout -- src/voxint/api/static/app/.gitkeep
   ```

## Automated E2E (`tests/e2e/`)

`tests/e2e/` is a **maintainer-run, opt-in** gate that exercises the whole
pipeline end to end against the *real* model services — no fakes. It is
**never** part of public CI (GitHub has no GPU runners and no model weights) and
never operator ceremony; it runs on maintainer hardware before a release (see
[`release-process.md`](release-process.md)).

It is built in lanes; **landed so far:**

- **Real pipeline** (`test_real_pipeline.py`) — submits `sample-3speaker.wav`,
  runs PREPARE → transcribe → diarize → embed in-process against the running
  services, and asserts the persistence invariants: run COMPLETED, exactly one
  `preprocessed_audio` artifact normalized to 16 kHz mono, non-empty transcript
  segments, diarization turns all embedded in `titanet-large-v1`, and a
  `duration_seconds` populated by the real PREPARE stage. Assertions are on
  *ranges and shape*, never exact transcript text (real ASR is not
  bit-deterministic). Two serial runs are checked for clean repetition with no
  cross-run leakage.

- **Real LLM — enrichment summary** (`test_enrich_assets_real_llm.py`) — the one
  lane that drives a real `HttpLLMClient` against a real OpenAI-compatible
  endpoint (every other enrichment test injects a `FakeLLM`). It gates the
  **chain, not the prose**: a seeded COMPLETED run's transcript is fed to
  `voxint enrich assets`' code path (`create_jobs` → `execute_job` with the real
  client), and it asserts the endpoint is reachable, the durable job reaches
  `succeeded` with `error` NULL, a *current* summary asset persists with the
  expected producer/prompt version + model alias + `config` snapshot + a
  well-formed `source_content_hash`, the asset is non-stale immediately after
  generation, one real operator correction re-stales it, and a malformed model
  reply yields an honest `failed` job (no asset, no partial success). The
  summary's semantic quality is *characterized* (printed), never asserted — a
  real, nondeterministic model produces the text, so an assertion on it would be
  a flake. The transcript is seeded (not produced by the pipeline) to isolate the
  LLM boundary — a failure names the LLM chain, not an upstream model service.

**Still upcoming** (tracked in the maintainer plan, not yet committed): a browser
lane that drives the review-console UI and reconciles durable state — that lane
will replace the **manual browser pass above** when it lands.

### Gate semantics

The directory keys off `VOXINT_E2E`, deliberately asymmetric so an explicit run
can never go green by skipping itself:

- **`VOXINT_E2E` unset** → the whole directory is skipped at collection, so a
  bare `pytest` run (or CI) stays green with no model services present.
- **`VOXINT_E2E=1`** → any missing prerequisite (the test DB, model-service
  health, or a wrong `/healthz` device identity) is a **hard failure, not a
  skip**.

The real-LLM lane is an **optional sub-lane** with one extra rung: it is skipped
(never failed) when the LLM env is not configured — an operator may run the
pipeline lane without wiring an LLM — but once configured (`LLM_ENABLED=true`,
`ENRICHMENT_RUN_ASSETS_ENABLED=true`, a model alias set), an unreachable endpoint
or an alias that does not resolve is a **hard failure**. The gate records the
concrete backend the alias resolved to, so a silent reroute to different weights
is a named signal rather than an invisible change in the summary text.

### Running it

Bring up the model services on a lane your host supports (host-specific
bring-up — compose overlays, CPU limits, the AMD render gid — lives outside this
public repo). Then, against a **disposable** database (its schema is dropped and
rebuilt from the alembic chain — never the live `voxint` DB):

```bash
export VOXINT_TEST_DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_e2e"
VOXINT_E2E=1 uv run --extra dev pytest tests/e2e -q
```

To include the real-LLM lane, also set the enrichment LLM env (endpoint URL,
model alias, and key live in your own environment, never in the repo):

```bash
export LLM_ENABLED=true ENRICHMENT_RUN_ASSETS_ENABLED=true
export LLM_BASE_URL=... LLM_MODEL=... LLM_API_KEY=...
VOXINT_E2E=1 uv run --extra dev pytest tests/e2e -q
```

Keep it **serial / low concurrency** — the pipeline is heavy and the lane is not
built for parallel fan-out. The suite stages audio under `MEDIA_ROOT` (the same
host directory the containers mount at `/data/media`) and cleans up after
itself.
