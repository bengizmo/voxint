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
(`venv`, `pgdata`, `logs`, `run`, `backups`, and `state.env` with the generated
secrets at mode `0600`). Uninstalling is `down` followed by
`rm -rf ~/.voxint-native`.

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

## Backup and restore

```bash
scripts/native/voxint-native.sh backup           # pg_dump -Fc -> backups/voxint-<stamp>.dump
scripts/native/voxint-native.sh restore <file>   # pg_restore --clean --if-exists
```

This is acceptance-lite: a single-operator `pg_dump`/`pg_restore` of the `voxint`
database. Hardened backup/restore and Postgres major-version upgrades are
deferred to the bundled-Postgres child (#71).

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
