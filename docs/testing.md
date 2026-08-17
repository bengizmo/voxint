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
**local** instance. This is a manual procedure today; an automated end-to-end
suite that seeds fixtures, drives the UI, and asserts the results is planned to
replace it (see *Planned: automated E2E* below).

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

### Planned: automated E2E

The manual pass above is the current state. A committed end-to-end suite is
planned to seed a set of audio fixtures covering the common cases, drive the
review-console UI, and assert both existing and changed behaviour in a disposable
test environment — including a real-LLM lane for the enrichment producers (the
run-asset summary/topics/entity generators bypass the UI's LLM toggle and read
`LLM_ENABLED` + `LLM_BASE_URL/MODEL/API_KEY` from the environment; see
[`operations.md`](operations.md) and `src/voxint/enrichment/`). When it lands,
this section is updated to point at it and the manual recipe becomes the fallback.
