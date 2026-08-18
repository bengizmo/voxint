---
name: voxint-native-e2e
description: >-
  Run the maintainer, opt-in acceptance lane for the docker-free NATIVE macOS
  install + usage path (#69): the launchd-supervised core-stack launcher
  scripts/native/voxint-native.sh brings up brew Postgres+pgvector + Redis +
  api/worker/beat (no containers) and delegates to the metal launcher for the
  model services. A fast --no-models "smoke" inner gate proves the install stands
  up (healthz + doctor + every island bundle serves 200); the full usage lane
  submits a media file over HTTP, drives the real Metal pipeline through the
  launchd-supervised Celery worker, reads back the durable DB invariants (all six
  stages completed, real titanet-large-v1 embeddings), then checks backup +
  restart-survival persistence. macOS/Apple-Silicon only; serial; never public
  CI; fail-not-skip if a prerequisite is absent.
---

# Voxint native (docker-free) install + usage E2E lane

This is a **thin adapter** over two things it does NOT own: the native launcher
`scripts/native/voxint-native.sh` (which owns setup/up/down/backup/restore/doctor)
and the canonical driver `tools/native_e2e_lifecycle.py` (which owns the HTTP
submit→poll steps and the read-back DB verifier). This skill only sequences them
and asserts each step. Keep interaction and its assertion together.

**Audience & scope:** Voxint is a self-hosted, single-operator app. Epic #68 runs
it **without Docker**; its macOS MVP (#69) is this launcher. Every server-side
test is green while the *native orchestration* — launchd supervision, private
brew Postgres/Redis on collision-free ports, alembic gating, secret hygiene,
frontend staging, the API→enqueue→Celery→worker path — could still be broken. This
lane makes that gate-able. It complements, not duplicates, the in-process
`tests/e2e/` pipeline lane (which never exercises the worker/HTTP path).

**Why a skill, not a pytest module:** the lane mutates real launchd + brew state
and runs against the launcher's **live** `voxint` database (not a disposable
`voxint_e2e`), so the driver is **read-back only** — no DDL/drop path exists in it.

## Preconditions (fail, do not skip)

- Repo root, under `uv`. **macOS on Apple Silicon** with Homebrew + the native
  install already provisioned (`~/.voxint-native/state.env` present — run
  `scripts/native/voxint-native.sh setup` first if not).
- `VOXINT_NATIVE_E2E=1` in the environment (the opt-in gate; if unset, stop and
  report — do not run silently).
- The **smoke** lane needs only the core stack (`--no-models`). The **full usage**
  lane additionally needs the **model tier** reachable (the metal launcher's
  whisper/pyannote/titanet, or `VOXINT_NATIVE_ASR_URL`/`DIARIZER_URL`/`EMBEDDER_URL`
  pointed elsewhere).
- If a required prerequisite is absent when the lane is asked for, it must **fail,
  not skip** — an operator who asked for the native gate must never get a green
  result because the stack was absent.

The driver reads `~/.voxint-native/state.env` itself and composes the DSN / mints
CSRF tokens **internally** — the generated `DB_PASSWORD`/`CSRF_SECRET` are never
passed on argv or printed. Override the state file with `--state-file` if needed.

## Part A — smoke inner gate (`--no-models`, cheap; run this alone first)

```bash
scripts/native/voxint-native.sh setup --no-models   # idempotent; skip if already set up
scripts/native/voxint-native.sh up --no-models
uv run python tools/native_e2e_lifecycle.py env      # prints BASE_URL=… (non-secret)
```

Wait for `http://127.0.0.1:<API_PORT>/healthz` to return 200 (the `env` output
names the port; on a box already running brew PG/Redis it is 8081, not 8080).
Then:

```bash
scripts/native/voxint-native.sh doctor --no-models   # assert overall PASS
uv run python tools/native_e2e_lifecycle.py smoke     # healthz + /setup wiring + every bundle 200
scripts/native/voxint-native.sh down --no-models
```

- **doctor**: treat overall **PASS** as green. `--no-models` makes it emit a
  model-tier **SKIP** line — that is expected, not a failure.
- **smoke**: `/healthz` 200; `/setup` (auth, no onboarding) 200 and its HTML
  references the manifest's hashed `main`/`tailwind` bundles (catches template↔
  manifest drift); every hashed bundle in `.vite/manifest.json` serves 200 with
  non-empty bytes. A non-zero exit fails the gate.

## Part B — full usage lane (with the model tier)

```bash
scripts/native/voxint-native.sh setup        # fresh install → invariant 5 (0 enrollment rows) holds
scripts/native/voxint-native.sh up           # core + delegates to the metal model launcher
uv run python tools/native_e2e_lifecycle.py env   # wait for /healthz 200
```

Drive a real run through the HTTP surface and verify the durable state:

```bash
# onboard → submit → poll in one call (or run the three subcommands separately):
uv run python tools/native_e2e_lifecycle.py drive --file media/diarize-3speaker.wav
#   → prints RUN_ID=<uuid>; DRIVE PASS when the run reaches 'completed'
uv run python tools/native_e2e_lifecycle.py verify --run-id "<RUN_ID>"
#   → VERIFY PASS: run completed; all 6 stages completed; non-empty ASR text;
#     ≥1 diarization turn embedded in titanet-large-v1 (192-dim); 0 enrollment rows
```

`submit` fails loudly if the run comes back `?enqueue=deferred` — that means the
Celery broker/worker is not up, and the run would never progress. Poll timeout is
`--timeout` (default 600 s; raise it on a slow host — the first Metal run
cold-loads weights). Assertions are shape/identity only (never exact transcript
text), so real ASR sampling noise never trips them.

Then check backup + **restart-survival** persistence:

```bash
scripts/native/voxint-native.sh backup       # pg_dump -Fc into ~/.voxint-native/backups/
scripts/native/voxint-native.sh down
scripts/native/voxint-native.sh up
uv run python tools/native_e2e_lifecycle.py env   # wait for /healthz 200
uv run python tools/native_e2e_lifecycle.py verify --run-id "<RUN_ID>"   # survived the restart
scripts/native/voxint-native.sh down
```

## Part C — destructive restore rung (`--with-restore`, opt-in)

Runs **after** Part B, reusing its green `<RUN_ID>`. This is the honest
disaster-recovery gate: `restore --fresh` **drops the database, proves it is
genuinely empty, then rebuilds it from a dump** (the dump is the sole schema
source; the vendored pgvector extension is preinstalled by the superuser and
excluded from the restore — never recreated as the unprivileged `voxint` role).
The unchanged Python `verify` then proves the same run survived a full rebuild.

```bash
scripts/native/voxint-native.sh backup     # capture the EXACT printed dump path
scripts/native/voxint-native.sh down        # services down (required for --fresh)
scripts/native/voxint-native.sh restore --fresh "<that-dump-path>"
#   → prints `SAFETY_BACKUP <path>` (automatic pre-drop 0600 backup) then
#     `EMPTY_DB PASS (old_oid=… new_oid=…, 0 public tables)` then rebuilds
scripts/native/voxint-native.sh up          # alembic no-ops at head; app starts
uv run python tools/native_e2e_lifecycle.py env      # wait /healthz 200
uv run python tools/native_e2e_lifecycle.py verify --run-id "<RUN_ID>"
#   → VERIFY PASS: the run survived a destructive drop + rebuild-from-dump
scripts/native/voxint-native.sh down
```

- **Capture the dump path from `backup`'s own output** (`backup complete: …`),
  not a newest-file heuristic.
- `restore --fresh` refuses to run while api/worker/beat are supervised (the flag
  is destructive consent, not license to force past a live app). **Before dropping
  anything** it validates the archive (`pg_restore --list`), requires it to *be* a
  voxint dump (an `alembic_version` table entry in the TOC — a valid dump of some
  other database is refused, not restored over your data), and confirms the
  postmaster on the port is the managed cluster (`SHOW data_directory`). It then
  takes an automatic pre-drop **safety backup** (a `0600` dump, printed as
  `SAFETY_BACKUP <path>`) and aborts before dropping anything if that dump fails,
  so a failed restore is always recoverable. On a
  restore failure the single transaction rolls back and the dump is left untouched;
  it prints the exact retry command (the prior DB is already dropped by then, so
  the message says so).
- **Recovery scope is DB-only.** `pg_dump`/`restore --fresh` cover database state,
  **not** external media files or model weights.
- **Plain `restore <file>` (non-`--fresh`) now shares the same fail-closed
  preflight** (#71): services-down gate, archive + `alembic_version` identity,
  managed-postmaster `data_directory` check, and vector-TOC filtering — but it
  *replaces in place* (`--clean --if-exists` in one transaction) rather than
  dropping the database, so objects absent from the archive survive. `--fresh`
  remains the exact-rebuild path this Part C exercises. New `backup` dumps are
  taken with `--exclude-extension=vector`.

## Cleanup

The native install is **throwaway** and the driver never mutates the DB, so
teardown is just bringing the stack down (above). To reset from scratch:
`rm -rf ~/.voxint-native && scripts/native/voxint-native.sh setup`. Leave the
machine in the state you found it (stack down unless the operator wants it up).

## Notes

- **Serial only** (issue #23): keep concurrency at one on maintainer hardware.
- The honest restore-into-fresh rung is **Part C** (`restore --fresh`): down →
  drop/recreate DB from `template0` (proving emptiness) → restore the dump as the
  sole schema source → up (alembic no-ops at head) → verify. It is launcher-owned
  and destructive; the Python driver stays SELECT-only and unchanged. `backup` +
  restart-survival (Part B) remains the non-destructive persistence check.
- **Invariant 5** (`speakers`/`speaker_embeddings` = 0 rows) is table-level and
  assumes a *fresh* install — which the full lane always is (it starts from
  `setup`). Those rows are operator-enrollment centroids from human adjudication,
  never pipeline output.
