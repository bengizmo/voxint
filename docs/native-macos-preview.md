# Native macOS preview (no Docker) — technical preview

**Status: technical preview.** This runs the Voxint stack on macOS/arm64
**without Docker Desktop**, under `launchd`, from a single launcher. It is aimed
at operators for whom Docker Desktop is a barrier. It is **not** the packaged,
signed, non-technical release — that is a later child of the epic (#73). Expect
to run a few shell commands and to read `doctor` output when something is off.

It keeps the current architecture unchanged — PostgreSQL 17 + pgvector, Redis,
the API server, the Celery worker, and Celery beat — and only proves the native
lifecycle end to end. Nothing about the pipeline, the models, or the numerics
differs from the Docker stack; only the process supervisor does.

The launcher is `scripts/native/voxint-native.sh`. It is the core-stack
counterpart to `scripts/metal/voxint-metal.sh` (which supervises the three
**model** services on Apple Silicon). By default `voxint-native` also drives the
metal launcher, so **one command brings up the whole preview** — core plus
whisper/pyannote/titanet.

## What you need

- macOS on **Apple Silicon** (arm64).
- [Homebrew](https://brew.sh) — the launcher `brew install`s `postgresql@17`,
  `pgvector`, and `redis` (their binaries only; the cluster it runs is its own).
- [uv](https://docs.astral.sh/uv/) — builds the Python venvs.
- `ffmpeg` / `ffprobe` on `PATH` (`brew install ffmpeg`) — the PREPARE stage.
- **Node + npm** (`brew install node`) — builds the review-console islands. If
  absent, `setup` still completes but the console will not hydrate until you
  build the frontend and re-run `setup`.

Run `scripts/native/voxint-native.sh doctor` at any time; it checks all of the
above (plus the cluster, the pgvector extension, port collisions, staged
islands, and MEDIA_ROOT) and exits non-zero if anything is wrong.

## Install and run

```bash
# 1. One-time setup: brew datastore binaries, the core venv, the review-console
#    islands, a PRIVATE Postgres cluster, generated secrets, AND the model tier
#    (weights download — needs network). Re-runnable and idempotent.
scripts/native/voxint-native.sh setup

# 2. Bring the whole preview up under launchd (core + models). This provisions
#    the role/db/extension, runs `alembic upgrade head` BEFORE the app starts,
#    then opens the console in your browser.
scripts/native/voxint-native.sh up

# 3. Check readiness (supervision state, /healthz, datastore reachability, and
#    the delegated model-service status).
scripts/native/voxint-native.sh status

# 4. Stop everything (data is left intact).
scripts/native/voxint-native.sh down
```

Everything the launcher owns lives under `~/.voxint-native/`
(`venv`, `pgdata`, `logs`, `run`, `backups`, and `state.env`). Uninstalling is
`down` followed by `rm -rf ~/.voxint-native`.

**Secret hygiene.** The `~/.voxint-native` tree is created mode `0700`, and every
file that carries a secret is `0600`: `state.env` (the generated `DB_PASSWORD` /
`VOXINT_PASSWORD` / `CSRF_SECRET`), the launchd plists under `run/` (they bake the
`DATABASE_URL` — password included — plus `VOXINT_PASSWORD` and `CSRF_SECRET` into
their env), and the `pg_dump` archives under `backups/` (a full copy of the
database in the clear). So credentials and data stay readable only by your own
account, not by other local users. This hardening is tracked in
[the 2026-08-18 security audit](security/audit-2026-08-18.md).

### Ports and isolation

The private cluster and Redis default to Postgres `5432` / Redis `6379` /
API `8080`, but `setup` moves each off a collision and records the chosen value
in `state.env`, so the native stack never fights an existing brew Postgres or a
running Docker stack. `status` and `doctor` report the ports actually in use.

### Running the models elsewhere (`--no-models`)

Pass `--no-models` (or set `VOXINT_NATIVE_WITH_MODELS=0`) on
`setup`/`up`/`down`/`status`/`doctor` to skip driving the metal launcher — for
when the model services run on other hardware. Point the core at them by setting
`VOXINT_NATIVE_ASR_URL` / `VOXINT_NATIVE_DIARIZER_URL` /
`VOXINT_NATIVE_EMBEDDER_URL` before `up` (these default to the metal launcher's
loopback ports and are baked into the service plist — launchd inherits no
ambient env, so a bare `ASR_URL` in your shell would be ignored). **Submissions
fail until the model services are reachable**, whichever way you run them.

Both the flag and any URL override are read on every `up`, so re-run `up` after
changing them.

The launcher validates the operator-settable `VOXINT_NATIVE_*` values on every
subcommand and **fails closed with a clear message** on an unsafe one: ports must
be `1..65535`, `VOXINT_NATIVE_DB_USER` / `VOXINT_NATIVE_DB_NAME` must be plain SQL
identifiers (`[A-Za-z_][A-Za-z0-9_]*`), the log-rotation sizes must be positive
integers, and no override that is written into a plist (the model URLs, the DB
password, `VOXINT_NATIVE_HOME`, the brew prefix / PG bindir) may contain a
newline. This keeps a stray or malformed value from reaching a shell, the
database, or a launchd env record — you will get an `ERROR:` line instead of a
surprising failure deep in `up`.

### MEDIA_ROOT

Both launchers read `MEDIA_ROOT` from the repo `.env` and resolve it against the
repo root (physically, via `pwd -P`), so the core and the models land on the
**identical** media directory by construction. The native launcher falls back to
`./media` when there is no `.env`; the metal launcher **requires** the `.env`
(create one with `scripts/install.sh`). Without it, the core comes up but the
delegated model `up` fails with the metal launcher's own message — the core
stays healthy; submissions just fail until the models are up.

## Honest runtime cost

On the Apple-Silicon metal tier the model services run on CPU/Metal, not a
datacenter GPU. **Long recordings take a while** — transcription and diarization
of an hour of audio are minutes of compute, not seconds, and the first pyannote
run pays a one-time Metal shader warm-up. This preview is about proving the
native lifecycle, not about matching GPU throughput. Submit a short clip first
to confirm the pipeline flows before queuing anything long.

## Upgrade path

Upgrading is re-running the same two commands after pulling new code:

```bash
git pull
scripts/native/voxint-native.sh setup   # rebuilds venv + islands (idempotent)
scripts/native/voxint-native.sh up       # alembic upgrade head runs first
```

`setup` reuses the existing ports and secrets from `state.env` and only rebuilds
what changed; `up` runs `alembic upgrade head`, which **no-ops when already at
head**, so your data survives the upgrade untouched. Take a backup first if you
want a safety net (below).

`setup` **overlays** the rebuilt console islands rather than wiping the old ones,
so a console open in your browser keeps working until `up` restarts the API with
the new build; reload the page after `up` to pick it up. (Superseded hashed
bundles linger unreferenced under `static/app/` — harmless; the manifest only
ever points at the current build.)

### Postgres major-version mismatch

The two commands above upgrade the *application*. They do **not** move your data
to a new **Postgres major** (e.g. 17 → 18). Skew happens when the Postgres
binaries the launcher runs are a different major than the private cluster on
disk — for example after a launcher release that targets a newer
`postgresql@NN`, or when `VOXINT_NATIVE_PG_BINDIR` is pointed at a different
major. The server would then refuse to start against the old data directory.
`up` now catches this **before** starting anything and stops with a plain
message, and `doctor` reports it as a failure naming both versions. You then have
two choices:

- **Stay on your current data** — point the launcher back at the matching major:
  install it if needed (`brew install postgresql@17`) and set
  `VOXINT_NATIVE_PG_BINDIR="$(brew --prefix postgresql@17)/bin"`, then `up`.
- **Move your data forward one major** (e.g. 17 → 18) — run `upgrade-db` (below).

### Upgrading the Postgres major (`upgrade-db`)

`upgrade-db` performs a **dump/restore** upgrade that preserves your real data,
one major forward at a time (the first certified edge is 17 → 18):

```bash
scripts/native/voxint-native.sh down        # the whole stack must be stopped
scripts/native/voxint-native.sh upgrade-db   # dump old -> initdb new -> restore
scripts/native/voxint-native.sh up
```

What it does, fail-closed and validate-before-destroy:

1. Confirms the move is exactly one major forward (same-major is a no-op;
   downgrades and skipped majors are refused).
2. Runs your **old** cluster briefly on a private socket (it needs the old
   `postgresql@NN` binaries present — install them with `brew install
   postgresql@NN`, or point `VOXINT_NATIVE_OLD_PG_BINDIR` at their `bin`), dumps
   the `voxint` database with the **new** `pg_dump`, and **proves the dump is
   restorable before touching your data directory**.
3. Renames the old cluster aside as a rollback (`pgdata.pg<old>-<stamp>`),
   `initdb`s the new major, and rebuilds from the dump (the same tested
   `restore --fresh` machinery: pgvector-safe, single-transaction, `alembic
   upgrade head`).

The old cluster is **kept** at `pgdata.pg<old>-<stamp>` — delete it (`rm -rf`)
only once you have confirmed a good run. It is a cleanly-stopped, logically
intact copy (starting and stopping it writes WAL/control state, so it is not
byte-for-byte identical to before), and running it again needs the old
`postgresql@<old>` binaries.

**If anything goes wrong** the command auto-rolls-back: it sets the partial new
cluster aside as `pgdata.failed-<stamp>` (never deleted) and restores the old
cluster to `pgdata`. You can also trigger this yourself:

```bash
scripts/native/voxint-native.sh upgrade-db --rollback
```

After a rollback the old data is live again, but you must repoint
`VOXINT_NATIVE_PG_BINDIR` at the old `postgresql@<old>` binaries (rolling data
back without the matching binaries just re-creates the skew). If `up` finds a
set-aside cluster but no live one — an upgrade interrupted mid-cutover — it
refuses and points you at `upgrade-db --rollback`.

> Bundling the Postgres distribution itself (so no `brew install postgresql@NN`
> is needed) remains the other, still-deferred half of the bundled-Postgres
> child (#71).

## Backup and restore

```bash
scripts/native/voxint-native.sh backup           # pg_dump -Fc (vector-free) -> backups/voxint-<stamp>.dump
scripts/native/voxint-native.sh restore <file>   # in-place replace (services down)
scripts/native/voxint-native.sh restore --fresh <file>   # exact rebuild / disaster recovery
```

Single-operator `pg_dump`/`pg_restore` of the `voxint` database. Both restore
paths run with the **app services down** (only maintenance Postgres up) and are
pgvector-safe: the privileged `vector` extension is preinstalled by the cluster
superuser and its dump entries are filtered out, so the unprivileged `voxint`
role never tries to (re)create it. Before touching the database each path proves
the archive is a real voxint dump (its `alembic_version` table entry) and that
the listener on the port is the managed cluster, then restores inside a single
transaction (any error rolls back and leaves the dump untouched). New backups are
taken with `--exclude-extension=vector`, so fresh dumps omit the extension
entirely; older dumps are still filtered at restore time.

- **`restore <file>`** — *replacement in place*. Runs `pg_restore --clean
  --if-exists`, so objects the archive carries are replaced; objects present in
  the current database but **absent from the archive survive**.
- **`restore --fresh <file>`** — *exact rebuild*. Because this one is destructive,
  it first takes an **automatic pre-drop safety backup** of the current database
  (a `0600` dump under `backups/`) and prints a `SAFETY_BACKUP <path>` line; if
  that dump fails it **aborts before dropping anything**, so you are never left
  without a fallback. It then drops the database, proves it empty
  (`EMPTY_DB PASS`), and rebuilds it from `<file>` as the sole schema source. Use
  this for disaster recovery when you want the database to match the dump exactly —
  and if a restore ever goes wrong, recover with `restore --fresh` on the
  `SAFETY_BACKUP` path it printed.

Postgres major-version **skew is detected** at `up`/`doctor`, and a guided,
data-preserving major-version **upgrade** is available via `upgrade-db` (see
[Upgrading the Postgres major](#upgrading-the-postgres-major-upgrade-db) above).
Bundling the Postgres distribution itself remains deferred to the
bundled-Postgres child (#71).

## Verifying the install (E2E)

A maintainer, opt-in acceptance lane gates this whole path — the `voxint-native-e2e`
skill driving `tools/native_e2e_lifecycle.py`. It has two rungs:

- **Smoke inner gate** (`--no-models`, fast): after `up --no-models`,
  `doctor` reports PASS and
  `uv run python tools/native_e2e_lifecycle.py smoke` confirms `/healthz` plus every
  hashed island bundle in the Vite manifest serves 200 — proof the install stood up
  and the frontend staged, no model tier required.
- **Full usage lane** (with models): `drive --file media/diarize-3speaker.wav`
  submits over the real HTTP surface (CSRF minted from `state.env`), drives the
  launchd-supervised Celery worker through the real Metal pipeline, and
  `verify --run-id <id>` reads back the durable invariants (run + all six stages
  `completed`, non-empty transcript text, diarization turns embedded in
  `titanet-large-v1`, and zero operator-enrollment rows). It then checks `backup`
  and restart-survival persistence.

The verifier is **read-back only** (SELECT against the live `voxint` database — no
schema-drop path), and the generated `DB_PASSWORD`/`CSRF_SECRET` are read from
`state.env` internally, never passed on the command line. See
[testing.md](testing.md) for the full lane description. Set `VOXINT_NATIVE_E2E=1`
to run it; it is macOS/Apple-Silicon only and never part of public CI.

## Logs

Each service logs to `~/.voxint-native/logs/<service>.log`. Follow one with
`scripts/native/voxint-native.sh logs <api|worker|beat|postgres|redis|logrotate> -f`.
Logs are `copytruncate`-rotated once over 50 MB, keeping 5 archives; `up`
installs a daily `launchd` rotation job (`com.voxint.native.logrotate`) and you
can rotate on demand with `scripts/native/voxint-native.sh rotate-logs`.

## Troubleshooting

- **`up` cannot reach Postgres** — run `doctor`. A fresh cluster that failed to
  start is usually a locale trap; the launcher bakes `LC_ALL=C`/`LANG=C` into the
  Postgres job for exactly this reason.
- **`CREATE EXTENSION vector` failed** — `pgvector` must be built against
  `postgresql@17`: `brew reinstall pgvector`.
- **The console loads but does not hydrate** — the islands are not staged.
  Install Node and re-run `setup`; `doctor` reports whether
  `static/app/.vite/manifest.json` is present.
- **Submissions fail** — the model services are not up. `status` shows the
  delegated model state; `scripts/metal/voxint-metal.sh doctor` diagnoses them.
