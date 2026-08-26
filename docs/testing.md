# Testing

How Voxint is tested, how to run each layer locally, and the manual procedure for
browser-verifying the review console. Numerics changes have their own, stricter
doctrine: see [`gpu-contracts.md`](gpu-contracts.md) and the parity notes below.

## Test layers

| Layer | Path | What it covers | Needs |
|---|---|---|---|
| Unit | `tests/unit/` | Pure logic: config parsing, CLI, API helpers, formatters, scoring, redaction, review-auth, validation. No database. | nothing |
| Contracts | `tests/contracts/` | Invariants that would rot silently: version-pin parity across pyproject/compose/`.env.example`, Dockerfile sha ARGs ↔ provenance, restart policies, routes/schemas, the **frontend build/island wiring** (`test_frontend_build.py`). | nothing |
| Integration | `tests/integration/` | Real Postgres + the alembic chain. Every API/console behaviour is exercised here (submission, adjudication, verify-and-advance, run-assets, Home, migrations). | a pgvector database |
| Parity | `tests/parity/` | Model-output equivalence gates (mel / vector / decision) against committed CUDA references. Real audio fixtures live under `tests/parity/fixtures/`. | strict mode: `VOXINT_PARITY_REQUIRED=1` |
| E2E | `tests/e2e/` | The **real** pipeline against the **real** model services (faster-whisper + pyannote + TitaNet in their containers): submit the tutorial clip, run every stage, assert the persistence invariants. Plus a **real-LLM** enrichment lane (real `HttpLLMClient` → real endpoint) that gates the summary chain. Maintainer-run, opt-in gate, **never public CI**. | `VOXINT_E2E=1` + `VOXINT_TEST_DATABASE_URL` + the model services running; the LLM lane also needs the enrichment LLM env (see below) |

The layout is the standard `pytest` tree; add a test in the same commit that adds
the behaviour or invariant it guards (a new island → a row in
`tests/contracts/test_frontend_build.py`; a new contract → a `tests/contracts/`
test).

### Keeping a test in the right layer

A test belongs in `tests/unit/` when the function it exercises is pure: it takes
plain data (or lightweight duck-typed fakes) and returns a value, with no
`Session`, no HTTP route, no filesystem or network dependency. Some tests land in
`tests/integration/` only because they were written there, not because they need
Postgres; moving those down to the unit lane keeps the integration suite focused
on what actually needs a database. When you relocate one, follow this checklist so
the move is provably behaviour-preserving:

- **Move, do not rewrite.** Copy each test verbatim; never weaken or restate an
  assertion to fit the unit lane. If a test cannot assert the same behaviour
  without a `Session` or a `TestClient`, it stays put.
- **Prove the subject is pure.** The function under test takes no `Session`
  parameter and reaches no database, and importing its module opens no
  connection. Construct its inputs in memory (a `SimpleNamespace` fake carrying
  only the attributes the function reads is the house pattern, see
  `tests/unit/test_effective_text.py`).
- **Guard early-exit paths.** A CLI or handler test that is a unit test only
  because it returns before touching a database should assert that fact: wire the
  engine builder (or DB entry point) to raise, so a regression that reaches it
  fails loudly instead of silently connecting.
- **Prove the set is preserved.** Record the original test names, then confirm the
  new-unit and remaining-integration name sets are disjoint and their union equals
  the original. `pytest --collect-only` on both files together also confirms a
  shared basename resolves to two distinct modules (both `tests/` subdirectories
  are packages, so this is safe).
- **Validate in the real lanes.** Run the moved tests in the unit lane with
  `VOXINT_TEST_DATABASE_URL` and `DATABASE_URL` unset (this is the load-bearing
  proof they need no database), and run the retained tests against Postgres.
  `ruff` catches any now-unused import left in the shrunk file; `mypy src` does
  not cover `tests/`, so do not rely on it there.
- **Update both docstrings** to describe each file's narrowed scope.

Report performance as a directional result: one relocated module is a small
fraction of the integration suite, and the value is the smaller, more focused
lane plus a repeatable pattern, not a single headline speedup. Measure whole-lane
wall-clock over several runs (medians, fixed worker count) rather than timing one
module, whose cost is dominated by per-worker database setup.

## Running the suite

Unit and contract tests need nothing external, and parallelize cleanly:

```bash
uv run --extra dev pytest tests/unit tests/contracts -n auto
```

Integration tests need a Postgres with the `vector` extension. They read
**`VOXINT_TEST_DATABASE_URL`** and are **skipped entirely when it is unset** (so a
bare `pytest` run still passes without a database, and CI supplies the service).

The `engine` fixture is xdist-aware. Run serially (no `-n`) and it keeps the
historical single-database behaviour: drop and recreate the `public` schema on
`VOXINT_TEST_DATABASE_URL`, then `alembic upgrade head`. Run under `-n` and each
xdist worker instead gets **its own** disposable database
(`voxint_test_<runid>_<worker>`), created fresh, migrated to head, and dropped at
teardown; each test truncates its tables afterward. Folding the per-run id into
the name means two concurrent `pytest` invocations no longer collide on the same
database (the old "one invocation at a time or they deadlock on `DROP SCHEMA`"
foot-gun is gone). A fail-closed guard refuses any base name without a `test`/
`e2e` marker, so the suite can only ever touch a throwaway database, **never the
live `voxint` database**:

```bash
# One-time: a disposable database beside your dev one.
docker compose exec -T postgres psql -U voxint -d voxint \
  -c "CREATE DATABASE voxint_dev_test"
docker compose exec -T postgres psql -U voxint -d voxint_dev_test \
  -c "CREATE EXTENSION IF NOT EXISTS vector"

export VOXINT_TEST_DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_dev_test"
# -n 8 is the default worker count CI uses; the per-worker databases make this
# ~6x faster than the serial run on a multi-core box.
uv run --extra dev pytest tests/integration -n 8
```

The per-worker databases need a role that can `CREATE`/`DROP DATABASE` (the CI
`voxint` role and a local superuser both qualify); the maintenance connection
targets the server's `postgres` database.

Static gates (run these before landing anything non-trivial):

```bash
cd frontend && npm run typecheck && npm run lint && npm run build && cd ..
uv run ruff check src tests
uv run mypy            # CI form: packages=voxint (do NOT add tests; the parity
                       # stub files carry pre-existing, tolerated stub errors)
```

The current integration tests exercise the real stage implementations against
real Postgres and real ffmpeg but with **fake model providers** (`tests/fakes.py`:
`FakeASR` / `FakeDiarizer` / `FakeEmbedder` / `FakeLLM` / `FailingLLM`); see
`tests/integration/test_real_stages_e2e.py`. There is deliberately **no frontend
test runner** (no vitest/jest): island behaviour is covered by the Python
integration tests plus the manual browser pass below. Don't add one without
discussing it; it is bloat this single-operator app does not need.

## Choosing review depth and the browser lane

Review effort is matched to a change's blast radius, not to which files it edits.
`CLAUDE.md` states the policy; this section is the worked reference. Two gates are
judged independently:

- **Code-review depth.** Judge possible impact first. A change that could touch
  inference numerics, security, auth or CSRF, concurrency or locking, a DB
  migration, a public contract or seam, a released artifact, a dependency, or the
  strength of a test or CI gate is high-risk and gets a full multi-model panel,
  whatever files it edits. Real design choices or a new cross-cutting seam get a
  multi-model review. A clear fix in a familiar pattern gets a single-model
  review. A change with no plausible blast radius gets no formal panel, only the
  standard gates that already run on every change (local `ruff`, `mypy`, and
  `pytest`, plus the required CI checks `lint-test`, `secrets-scan`, and
  `coverage`). When two rows both fit, take the deeper one.
- **Browser lane.** Run the [browser E2E lane](#automated-e2e-testse2e) when a
  change alters observable review-console behaviour or a delivery, data, or auth
  contract a console island depends on, or when it changes the browser acceptance
  harness or its fixtures. Skip it for backend, pipeline, service, docs, CI, or
  test-only changes that leave island behaviour unchanged.

File type is an illustration, not the classifier. A "config tweak" that changes a
decode parameter is a numerics change; a "test-only" edit that loosens an
assertion weakens a gate; a "docs" edit to install or release copy can change
what operators do. Classify by what the change can affect.

| Example change | Impact class | Code-review depth | Browser lane | Why |
|---|---|---|---|---|
| Typo in a doc or code comment | none | no formal panel | no | No blast radius. |
| Loosen or delete a test assertion | high (gate strength) | full panel | no | A gate is never weakened to pass; escalates regardless of the file. |
| Change a whisper or pyannote model pin | high (numerics) | full panel | no | Parity evidence is mandatory and independent of review depth. |
| Change a config default that feeds inference (decode or batch param) | high (numerics) | full panel | no | A "config" change that moves numerics still needs measured equivalence. |
| Add a DB migration | high (migration) | full panel | only if an island reads the changed shape | Schema and data-integrity blast radius. |
| Edit auth or CSRF middleware | high (security) and island-facing | full panel | yes | Security escalates; islands depend on the auth contract. |
| Restyle a console island (CSS, build, or asset) | low | single-model review | yes | No invariant risk, but observable island behaviour changes. |
| Backend refactor with strong existing coverage, no numerics | routine | single-model review | no | Familiar pattern, coverage backs it, no island surface. |
| New cross-cutting backend seam or public API | non-trivial | multi-model review | only if an island consumes it | Design choices and a new contract. |
| Bump a dependency | high (supply chain) | full panel | yes if it is a frontend, build, or island runtime dependency | Supply-chain escalation; the lane only if island runtime can change. |
| Edit a release or CI workflow (`release.yml`, required-check wiring) | high (released artifact, gate strength) | full panel | no | Changes the supply chain or the gate set. |

This choice is about the slice in front of you. It does not replace the release
process's **Gate E**, which runs its own browser acceptance lane before tagging a
release whenever the review console or the island build path changed, under its
own diff-scoped carry-over rule (see
[`release-process.md`](release-process.md)). A slice that skipped the lane can
still oblige a Gate-E run at release time. Record both classifications, the gates
you ran, and each applied fix or deliberate skip in the commit message or PR, and
reclassify against the final landing diff if it grew.

## Browser verification of the review console

Interactive island behaviour (the #53/#58 verify-and-advance loop, click-to-edit,
the unsaved-edit discard warning, keymap suppression) is confirmed by driving a
real browser against a **local** instance. This is now automated as the
[browser E2E lane](#automated-e2e-testse2e): a canonical lifecycle tool
(`tools/e2e_browser_lifecycle.py`) plus the `voxint-e2e-review` skill that drives
Playwright and reconciles durable state. The manual steps below remain the
fallback (and document exactly what the tool automates) for a hand-run pass.

The dockerized `api` service runs the **released** image, not your working tree,
so browser-verifying a local change means running a fresh local instance:

1. **Build the islands and stage them where the app serves them.** The app reads
   the Vite manifest once at import, so copy *before* starting the server. These
   are build artifacts, so **do not commit them**; restore the `.gitkeep` and remove
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
   chips appear). **Set the media item's `duration_seconds`**; without it
   `playback_capability` gates seeking off and the player cannot follow along.
   (The current seed is an ad-hoc script; the planned automated suite commits a
   shared fixture set. Mirror the existing shape in
   `tests/integration/test_review_api.py` (`seed_run` /
   `_seed_run_with_confidences`).)
4. **Serve locally on a spare port** with its own media root and basic-auth:
   ```bash
   DATABASE_URL="…voxint_e2e" MEDIA_ROOT="$PWD/media-e2e" \
   VOXINT_USER=admin VOXINT_PASSWORD=e2epass API_PORT=8099 \
   uv run voxint serve
   ```
5. **Drive it.** Navigate once with the credentials embedded
   (`http://admin:e2epass@127.0.0.1:8099/`) to cache basic-auth, then **re-navigate
   to the clean URL** (no embedded credentials); an island `fetch()` throws
   "URL includes credentials" if the document URL carries them (a test-harness
   artifact, not a product bug). Claim the run from the workbench → **Review
   transcript →** → exercise the loop: `v` verify-and-advance, `e` edit +
   `⌘/Ctrl+Enter` save, `n` skip, `p` replay; click a line to move the edit cursor
   (verified lines are re-reachable); type an unsaved edit then verify to see the
   discard warning; focus the playback-speed `<select>` and press `v` to confirm
   the keymap does **not** fire from a form control.
6. **Clean up.** Kill the local server **by port** (`fuser -k 8099/tcp`), **not**
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
pipeline end to end against the *real* model services, no fakes. It is
**never** part of public CI (GitHub has no GPU runners and no model weights) and
never operator ceremony; it runs on maintainer hardware before a release (see
[`release-process.md`](release-process.md)).

It is built in lanes; **landed so far:**

- **Real pipeline** (`test_real_pipeline.py`): submits `sample-3speaker.wav`,
  runs PREPARE → transcribe → diarize → embed in-process against the running
  services, and asserts the persistence invariants: run COMPLETED, exactly one
  `preprocessed_audio` artifact normalized to 16 kHz mono, non-empty transcript
  segments, diarization turns all embedded in `titanet-large-v1`, and a
  `duration_seconds` populated by the real PREPARE stage. Assertions are on
  *ranges and shape*, never exact transcript text (real ASR is not
  bit-deterministic). Two serial runs are checked for clean repetition with no
  cross-run leakage. **This lane is AMD-only**: `EXPECTED_SERVICES` hardcodes
  whisper `device: rocm` (fail-not-skip, no env override), so run it on an
  AMD/ROCm box.

- **Real LLM, enrichment summary** (`test_enrich_assets_real_llm.py`): the one
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
  summary's semantic quality is *characterized* (printed), never asserted: a
  real, nondeterministic model produces the text, so an assertion on it would be
  a flake. The transcript is seeded (not produced by the pipeline) to isolate the
  LLM boundary: a failure names the LLM chain, not an upstream model service.

- **Browser runtime acceptance** (the `voxint-e2e-review` skill +
  `tools/e2e_browser_lifecycle.py`). This is the one lane that is **not** a pytest
  module: Playwright MCP is a Claude-Code capability, not a test dependency, and
  the durable-state check is post-hoc (it runs only after a browser has driven
  the UI). The lifecycle tool builds and stages the islands, seeds a disposable
  database with a COMPLETED run shaped for the loop (an audio artifact,
  `duration_seconds` set, and varied-confidence segments including sub-threshold
  ones so the "uncertain" chips appear), and serves a working-tree instance. The
  skill then drives the review-console islands: `v` verify-and-advance, `e` +
  `⌘/Ctrl+Enter` save, `n` skip, `p` replay, click-to-edit, the type-then-verify
  discard warning (warn on the first `v`, advance on the second), and the keymap
  suppression while a `<select>`/`<textarea>` has focus, asserting the DOM and
  network behaviour of each immediately (only verify and save touch the wire).
  Finally the tool's `reconcile` subcommand is a **fail-closed** verifier over
  `segment_review_states`: the browser was the sole writer, so the verified rows,
  corrected text, and the N-of-M progress must match exactly what was driven, or
  it exits non-zero. This replaces the manual browser pass above; run it serially
  on maintainer hardware (issue #23).

- **Native (docker-free) install + usage** (the `voxint-native-e2e` skill +
  `tools/native_e2e_lifecycle.py`). This is the lane for epic #68's no-Docker path. The
  launchd-supervised launcher `scripts/native/voxint-native.sh` stands up brew
  Postgres+pgvector + Redis + api/worker/beat (no containers) and delegates to the
  metal launcher for the model services. A fast `--no-models` **smoke** inner gate
  proves the install: `/healthz` 200, `doctor` PASS, `/setup` references the hashed
  island bundles, and every bundle in the Vite manifest serves 200. The **full
  usage** lane then submits `media/diarize-3speaker.wav` over the real HTTP surface
  (mint CSRF from `state.env` → onboard → `/submit` → poll `export.json`), so it
  exercises the API→enqueue→**Celery worker** path the in-process pipeline lane
  above never touches, and reads back the durable invariants (run + all six stages
  `completed`, non-empty ASR text, diarization turns embedded in `titanet-large-v1`
  at 192 dims, and zero operator-enrollment rows). Finally it checks `backup` +
  **restart-survival** persistence (down → up → re-verify), and in the opt-in
  **`--with-restore`** rung (Part C) an honest destructive-recovery gate:
  `voxint-native.sh restore --fresh <dump>` takes an automatic pre-drop safety
  backup (prints `SAFETY_BACKUP <path>`), drops the DB, proves it empty
  (`EMPTY_DB PASS`), rebuilds it from a backup as the sole schema source, then
  re-verifies the same run. Unlike every other lane it runs against the launcher's
  **live** `voxint` database (the native install is throwaway), so the verifier is
  **read-back / SELECT-only**: there is no schema-drop path in the tool (the
  destructive DDL is launcher-owned, behind the explicit `--fresh` flag), and the
  generated `DB_PASSWORD`/`CSRF_SECRET` are read from `state.env` internally, never
  passed on argv. macOS/Apple-Silicon only; serial (issue #23).
  - *Maintainer self-test:* `voxint-native.sh upgrade-db --rehearse` forces a
    same-major dump/restore cycle to exercise the `upgrade-db` machinery
    mechanically (no real version change). It is a maintainer aid, deliberately
    kept out of the operator preview guide.

### Gate semantics

The `tests/e2e/` directory keys off `VOXINT_E2E`, deliberately asymmetric so an
explicit run can never go green by skipping itself:

- **`VOXINT_E2E` unset** → the whole directory is skipped at collection, so a
  bare `pytest` run (or CI) stays green with no model services present.
- **`VOXINT_E2E=1`** → any missing prerequisite (the test DB, model-service
  health, or a wrong `/healthz` device identity) is a **hard failure, not a
  skip**.

The **native install + usage** lane is a skill, not a pytest module, so its
fail-not-skip is enforced by the skill: it keys off `VOXINT_NATIVE_E2E=1` and
stops (never silently green) when macOS/Apple-Silicon, the native install, or the
model tier is absent. Its tool's own unit + integration tests (`parse_state_env`
parity, the DSN composer, the manifest extractor, and the read-back verifier with
one negative case per invariant) run in the normal suites against a disposable
`voxint_e2e`; no model tier needed.

The real-LLM lane is an **optional sub-lane** with one extra rung: it is skipped
(never failed) when the LLM env is not configured, since an operator may run the
pipeline lane without wiring an LLM. Once configured (`LLM_ENABLED=true`,
`ENRICHMENT_RUN_ASSETS_ENABLED=true`, a model alias set), an unreachable endpoint
or an alias that does not resolve is a **hard failure**. The gate records the
concrete backend the alias resolved to, so a silent reroute to different weights
is a named signal rather than an invisible change in the summary text.

### Running it

Bring up the model services on a lane your host supports. The host-specific
bring-up (compose overlays, CPU limits, the AMD render gid) lives outside this
public repo. The real-pipeline lane needs whisper on **ROCm** (see the AMD-only
note above); the real-LLM and browser lanes are hardware-agnostic. Then, against
a **disposable** database (its schema is dropped and rebuilt from the alembic
chain, never the live `voxint` DB):

```bash
export VOXINT_TEST_DATABASE_URL="postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint_e2e"
VOXINT_E2E=1 uv run --extra dev pytest tests/e2e -q
```

> ⚠ The browser lane and this pytest lane default to the **same** disposable
> database name (`voxint_e2e`), and the browser lane's teardown (or the manual
> cleanup below) can drop it. The pytest lane expects the database to already
> exist and fails with "database does not exist" rather than creating it. If
> the browser lane ran first with `--drop-db`, recreate the database and its
> `vector` extension before this lane (`CREATE DATABASE voxint_e2e;` then
> `CREATE EXTENSION vector;` in it), or run the pytest lane first.

To include the real-LLM lane, also set the enrichment LLM env (endpoint URL,
model alias, and key live in your own environment, never in the repo):

```bash
export LLM_ENABLED=true ENRICHMENT_RUN_ASSETS_ENABLED=true
export LLM_BASE_URL=... LLM_MODEL=... LLM_API_KEY=...
VOXINT_E2E=1 uv run --extra dev pytest tests/e2e -q
```

Keep it **serial / low concurrency**: the pipeline is heavy and the lane is not
built for parallel fan-out. The suite stages audio under `MEDIA_ROOT` (the same
host directory the containers mount at `/data/media`) and cleans up after
itself.

## Eval-quality harness (offline, maintainer)

`tools/eval_quality.py` (issue #97) scores the pipeline's diarization and
transcript output against public, hand-annotated ground truth (AMI and
VoxConverse): Diarization Error Rate (DER) and Jaccard Error Rate (JER) via the
vetted `pyannote.metrics` accumulators, pooled Word Error Rate reusing the frozen
Whisper-bakeoff WER stack, and concatenated minimum-permutation WER (cpWER) via
`meeteval`. It is a **tripwire, not a benchmark**: the small subset can catch
gross breakage when the GPU knobs change, but it cannot prove non-regression, so
every threshold is measured from a zero-change noise floor rather than reasoned.
It is a maintainer instrument, never shipped to users and never installed into a
service image. No baseline-scores report is committed to `docs/reports/` yet.

The harness lives in its own **`eval-quality`** dependency extra, kept isolated
from `dev` because `pyannote.metrics` 4.1 pulls a `pyannote.core`/`numpy`/`scipy`
set that conflicts with the diarizer service's pinned `pyannote.core==5.0.0`. The
two never share an environment (the service is containerized; this harness runs in
the host `uv` env), and `tests/contracts/test_eval_quality_extra.py` asserts the
isolation and the dependency closure. Run it with the extra isolated, alongside
`parity` (which carries the frozen `jiwer` WER stack):

```bash
# Score a prepared hypotheses+reference manifest into metrics JSON:
uv run --isolated --extra parity --extra eval-quality \
  tools/eval_quality.py score --manifest paths.json --out metrics.json

# Render one or more scored metrics JSONs to a dated Markdown report (house style):
uv run --isolated --extra parity --extra eval-quality \
  tools/eval_quality.py report --run ami=metrics.json --date YYYY-MM-DD \
  --out docs/reports/eval-quality-baseline-YYYY-MM-DD.md
```

The `score` and `report` steps need no worker, database, or GPU. The `run`
subcommand is the live driver (submit to the real pipeline, poll, read the DB,
export the relabelled hypothesis RTTM/text that `score` consumes); it needs an
idle worker and a disposable database, the same way the E2E lanes do. Ground
truth is prepared off-repo: `tools/build_ami_wer_reference.py` freezes AMI's
per-speaker word-aligned XML into one chronological, UEM-cropped raw reference
stream once, so the harness never re-parses the XML. Nothing here is normalized
at rest; per the numerics doctrine, WER normalization is applied to raw reference
and raw hypothesis together at scoring time, and the harness records which
normalizer scored it.
