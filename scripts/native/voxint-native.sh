#!/usr/bin/env bash
# voxint-native.sh -- native (no-Docker) core control plane for macOS/arm64.
#
# The counterpart to scripts/metal/voxint-metal.sh: that script supervises the
# three MODEL services natively; this one supervises the CORE control plane --
# a launcher-managed PostgreSQL 17 + pgvector, Redis, the api server, the Celery
# worker, and Celery beat -- all under launchd, with no Docker at all. Issue #69
# (the MVP of epic #68, "run without Docker").
#
# This is a TECHNICAL PREVIEW, not the non-technical packaged release (#73). It
# keeps the current architecture (Postgres+pgvector, Redis, Celery, beat) to
# prove the native lifecycle end-to-end before any dependency is removed.
#
#   setup                 brew-install the datastore binaries, create the core
#                         venv, initdb a PRIVATE cluster, generate secrets
#   up | down             start/stop the whole core stack under launchd
#                         (KeepAlive restarts a crashed service -- the native
#                         equivalent of the containers' restart policy). `up`
#                         provisions the role/db/extension and runs
#                         `alembic upgrade head` BEFORE starting api/worker,
#                         reproducing compose's migrate gate.
#   status                per-service supervision state + api /healthz +
#                         Postgres/Redis reachability
#   logs <svc> [-f]       show (or follow) a service's log
#   doctor                environment checks: tooling, brew formulae, venv,
#                         ffmpeg/ffprobe, cluster + pgvector, ports
#   backup                pg_dump -Fc the voxint database into backups/
#   restore <file>        pg_restore a dump into the voxint database
#   run <svc> --foreground  run one service in the foreground for debugging
#   rotate-logs           copytruncate-rotate oversized logs (also daily via launchd)
#
# By default setup/up/down/status/doctor ALSO drive scripts/metal/voxint-metal.sh
# so one command runs the whole preview (core + whisper/pyannote/titanet). Pass
# --no-models (or set VOXINT_NATIVE_WITH_MODELS=0) to manage the models yourself.
#
# Layout (override the root with VOXINT_NATIVE_HOME):
#   $HOME/.voxint-native/{venv,pgdata,logs,run,backups,state.env}
#
# A launcher-managed PRIVATE Postgres+Redis is the default: brew provides the
# binaries, but the cluster + data live under VOXINT_NATIVE_HOME on their own
# ports, so nothing collides with an operator's existing brew Postgres. If the
# cluster has not been initialized, `up` falls back to operator-provided
# datastores on 127.0.0.1 (doctor reports which mode is in effect).
#
# Requirements: macOS on Apple Silicon, uv, Homebrew. Bash 3.2 compatible.
#
# Sourcing with VOXINT_NATIVE_LIB=1 loads the functions without running main
# (tests/unit/test_native_launcher.py exercises the pure logic that way).

set -eu

# ---------------------------------------------------------------------------
# Locate the checkout. When sourced for tests, $0 is the shell -- fall back to
# BASH_SOURCE (same guard as the metal launcher).
# ---------------------------------------------------------------------------
NATIVE_SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
case $NATIVE_SCRIPT_DIR in
  */scripts/native) : ;;
  *) NATIVE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) ;;
esac
REPO_ROOT=$(cd "$NATIVE_SCRIPT_DIR/../.." && pwd -P)
if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
  printf 'ERROR: cannot locate the Voxint checkout from %s (symlinked? run the script from scripts/native/ in the checkout)\n' "$0" >&2
  exit 1
fi

VOXINT_NATIVE_HOME=${VOXINT_NATIVE_HOME:-$HOME/.voxint-native}

# Supervised processes. Datastores come up (and go down) as a group distinct
# from the core services, with the migrate gate between them in `up`.
NATIVE_DATASTORES="postgres redis"
NATIVE_SERVICES="api worker beat"
LAUNCHD_PREFIX=com.voxint.native

# Core-service connection config. All overridable; the defaults match the app's
# own config.py fallbacks with the host/port made explicit. `setup` may move the
# ports off a collision and persists the chosen values to state.env.
NATIVE_DB_USER=${VOXINT_NATIVE_DB_USER:-voxint}
NATIVE_DB_NAME=${VOXINT_NATIVE_DB_NAME:-voxint}
NATIVE_DB_PASSWORD=${VOXINT_NATIVE_DB_PASSWORD:-voxint}
NATIVE_PG_PORT=${VOXINT_NATIVE_PG_PORT:-5432}
NATIVE_REDIS_PORT=${VOXINT_NATIVE_REDIS_PORT:-6379}
NATIVE_API_HOST=${VOXINT_NATIVE_API_HOST:-127.0.0.1}
NATIVE_API_PORT=${VOXINT_NATIVE_API_PORT:-8080}
# Homebrew prefix -- put its bin on each service's PATH so the worker finds
# ffmpeg/ffprobe (the app resolves the bare names on PATH, and launchd inherits
# none of the login shell's PATH). Arm64 default.
NATIVE_BREW_PREFIX=${VOXINT_NATIVE_BREW_PREFIX:-/opt/homebrew}
# postgresql@17 is keg-only, so its binaries live under opt/. Overridable.
NATIVE_PG_BINDIR=${VOXINT_NATIVE_PG_BINDIR:-$NATIVE_BREW_PREFIX/opt/postgresql@17/bin}
NATIVE_PGDATA=$VOXINT_NATIVE_HOME/pgdata
NATIVE_STATE=$VOXINT_NATIVE_HOME/state.env

# Remember which connection knobs the operator set explicitly: load_state and
# the setup port-picker must never clobber an explicit override with a
# persisted or auto-picked value.
_pg_port_explicit=${VOXINT_NATIVE_PG_PORT:+1}
_redis_port_explicit=${VOXINT_NATIVE_REDIS_PORT:+1}
_api_port_explicit=${VOXINT_NATIVE_API_PORT:+1}
_db_password_explicit=${VOXINT_NATIVE_DB_PASSWORD:+1}

# Secrets, threaded through native_service_env when set. load_state fills these
# from state.env; setup generates them on a fresh install.
VOXINT_NATIVE_PASSWORD=${VOXINT_NATIVE_PASSWORD:-}
VOXINT_NATIVE_CSRF_SECRET=${VOXINT_NATIVE_CSRF_SECRET:-}

# Model-service ports the api/worker reach over loopback. These are owned by
# scripts/metal/voxint-metal.sh (service_port there); a contract test binds the
# two so a port moved in one place cannot silently orphan the other.
WHISPER_PORT=8022
PYANNOTE_PORT=8024
TITANET_PORT=8021

# The model-service URLs the api/worker reach. Default to the metal launcher's
# loopback ports (the delegated, same-box case). Overridable so `--no-models`
# with the models on OTHER hardware actually works: launchd bakes these into the
# plist and inherits no ambient env, so a bare `ASR_URL` in the operator's shell
# would be ignored -- the override MUST flow through here.
NATIVE_ASR_URL=${VOXINT_NATIVE_ASR_URL:-http://127.0.0.1:$WHISPER_PORT}
NATIVE_DIARIZER_URL=${VOXINT_NATIVE_DIARIZER_URL:-http://127.0.0.1:$PYANNOTE_PORT}
NATIVE_EMBEDDER_URL=${VOXINT_NATIVE_EMBEDDER_URL:-http://127.0.0.1:$TITANET_PORT}

# One-command preview: by default setup/up/down/status/doctor also drive the
# metal launcher (the three model services), so a single `voxint-native up`
# brings up the WHOLE preview -- core + whisper/pyannote/titanet. `--no-models`
# (or VOXINT_NATIVE_WITH_MODELS=0) skips that delegation for operators running
# the models elsewhere.
NATIVE_WITH_MODELS=${VOXINT_NATIVE_WITH_MODELS:-1}

# Frontend island build + staging. `setup` runs `npm ci && npm run build` in the
# frontend dir and stages frontend/dist -> the api's asset dir (app.py reads
# _APP_ASSETS_DIR = src/voxint/api/static/app, manifest at .vite/manifest.json).
# Both are overridable so the offline tests can exercise staging against a
# throwaway tree without dirtying the repo.
NATIVE_FRONTEND_DIR=${VOXINT_NATIVE_FRONTEND_DIR:-$REPO_ROOT/frontend}
NATIVE_APP_ASSETS_DIR=${VOXINT_NATIVE_APP_ASSETS_DIR:-$REPO_ROOT/src/voxint/api/static/app}

# Log rotation (copytruncate + a daily launchd job), lifted from the metal
# launcher: the core services run for months under KeepAlive, so their stdout
# logs need bounding. Same knobs as the metal tier.
VOXINT_NATIVE_LOG_MAX_MB=${VOXINT_NATIVE_LOG_MAX_MB:-50}
VOXINT_NATIVE_LOG_ARCHIVES=${VOXINT_NATIVE_LOG_ARCHIVES:-5}

say()  { printf '%s\n' "$*" >&2; }
step() { printf '\n== %s\n' "$*" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested through the library seam)
# ---------------------------------------------------------------------------
core_venv()   { printf '%s/venv' "$VOXINT_NATIVE_HOME"; }
plist_label() { printf '%s.%s' "$LAUNCHD_PREFIX" "$1"; }
plist_path()  { printf '%s/run/%s.plist' "$VOXINT_NATIVE_HOME" "$(plist_label "$1")"; }
service_log() { printf '%s/logs/%s.log' "$VOXINT_NATIVE_HOME" "$1"; }

# Model-service delegation. metal_script is the sibling launcher; models_delegated
# is the pure predicate `up`/`down`/etc. consult before driving it.
metal_script()    { printf '%s/scripts/metal/voxint-metal.sh' "$REPO_ROOT"; }
models_delegated() { [ "${NATIVE_WITH_MODELS:-1}" = 1 ]; }

# --no-models parsing, split into a pure predicate + filter so main can both
# flip the module-level flag and strip the token from the positional args. A
# command substitution runs in a subshell (its global writes are invisible),
# hence the two-function shape rather than one mutating filter.
no_models_flag_present() {
  local a
  for a in "$@"; do [ "x$a" = "x--no-models" ] && return 0; done
  return 1
}
args_without_no_models() {
  local a
  for a in "$@"; do [ "x$a" = "x--no-models" ] || printf '%s\n' "$a"; done
}

# Frontend island build + staging paths (overridable for offline tests).
frontend_dir()     { printf '%s' "$NATIVE_FRONTEND_DIR"; }
app_assets_dir()   { printf '%s' "$NATIVE_APP_ASSETS_DIR"; }
app_manifest_path() { printf '%s/.vite/manifest.json' "$NATIVE_APP_ASSETS_DIR"; }

native_database_url() {
  printf 'postgresql+psycopg://%s:%s@127.0.0.1:%s/%s' \
    "$NATIVE_DB_USER" "$NATIVE_DB_PASSWORD" "$NATIVE_PG_PORT" "$NATIVE_DB_NAME"
}

native_redis_url() {
  printf 'redis://127.0.0.1:%s/0' "$NATIVE_REDIS_PORT"
}

# 32 bytes of /dev/urandom as 64 hex chars (256 bits). Lifted from
# scripts/install.sh: no openssl/python dependency, hex needs no dotenv
# escaping, and it exceeds the app's 16-char CSRF minimum.
generate_secret() {
  local s
  s=$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n') || return 1
  [ "${#s}" -eq 64 ] || return 1
  printf '%s' "$s"
}

# True if something on 127.0.0.1:$1 accepts a TCP connection or leaves it
# hanging. Same watchdog probe as the metal launcher / scripts/install.sh --
# macOS drops the SYN silently on a full accept queue, so a plain /dev/tcp
# probe would hang and then misreport a wedged listener as free.
port_in_use() {
  case $1 in ''|*[!0-9]*) return 1 ;; esac
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1 &
  local probe_pid=$! probe_tries=0
  while kill -0 "$probe_pid" 2>/dev/null; do
    probe_tries=$((probe_tries + 1))
    if [ "$probe_tries" -ge 20 ]; then
      kill "$probe_pid" 2>/dev/null
      wait "$probe_pid" 2>/dev/null
      return 0
    fi
    sleep 0.1
  done
  wait "$probe_pid"
}

next_free_port() {
  local p=$1
  while port_in_use "$p"; do
    p=$((p + 1))
    [ "$p" -le 65535 ] || { printf '%s' "$1"; return; }
  done
  printf '%s' "$p"
}

# ---------------------------------------------------------------------------
# MEDIA_ROOT resolution. Reuses the metal launcher's dotenv-quote-stripping so
# a native run agrees with whatever the Docker world mounted. Physically
# resolved (pwd -P): APFS /tmp -> /private/tmp symlinks would otherwise break
# the MEDIA_ROOT-relative path contract the worker relies on.
# ---------------------------------------------------------------------------
env_value_from_file() {
  # $1 = KEY, $2 = env file. Last assignment wins (dotenv semantics); trailing
  # CR, surrounding blanks, and ONE matched pair of quotes are stripped.
  local raw
  raw=$(grep -E "^${1}=" "$2" 2>/dev/null | tail -1 | cut -d= -f2-) || raw=""
  raw=${raw%$'\r'}
  raw=${raw#"${raw%%[![:blank:]]*}"}
  raw=${raw%"${raw##*[![:blank:]]}"}
  if [ "${#raw}" -ge 2 ]; then
    case $raw in
      \'*\') raw=${raw#\'}; raw=${raw%\'} ;;
      \"*\") raw=${raw#\"}; raw=${raw%\"} ;;
    esac
  fi
  printf '%s' "$raw"
}

resolve_media_root() {
  # $1 = raw MEDIA_ROOT (relative resolves against the repo root, as compose
  # does). Echoes the physical absolute path; returns 1 if it is not a dir.
  local raw=$1 dir
  case $raw in
    /*) dir=$raw ;;
    *)  dir=$REPO_ROOT/$raw ;;
  esac
  [ -d "$dir" ] || return 1
  (cd "$dir" && pwd -P)
}

resolved_media_root_or_fail() {
  local envf=$REPO_ROOT/.env raw dir resolved
  raw=""
  [ -f "$envf" ] && raw=$(env_value_from_file MEDIA_ROOT "$envf")
  [ -n "$raw" ] || raw=${VOXINT_NATIVE_MEDIA_ROOT:-./media}
  case $raw in /*) dir=$raw ;; *) dir=$REPO_ROOT/$raw ;; esac
  mkdir -p "$dir" 2>/dev/null || true
  resolved=$(resolve_media_root "$raw") \
    || fail "MEDIA_ROOT '$raw' does not resolve to a directory"
  printf '%s' "$resolved"
}

# ---------------------------------------------------------------------------
# Persisted state. setup writes the chosen ports + generated secrets to
# state.env (mode 0600); up/status/doctor/backup load it before doing anything.
# An explicit env override always wins over the persisted value.
# ---------------------------------------------------------------------------
load_state() {
  local sf=$NATIVE_STATE v
  [ -f "$sf" ] || return 0
  if [ -z "$_pg_port_explicit" ]; then
    v=$(env_value_from_file PG_PORT "$sf"); [ -n "$v" ] && NATIVE_PG_PORT=$v
  fi
  if [ -z "$_redis_port_explicit" ]; then
    v=$(env_value_from_file REDIS_PORT "$sf"); [ -n "$v" ] && NATIVE_REDIS_PORT=$v
  fi
  if [ -z "$_api_port_explicit" ]; then
    v=$(env_value_from_file API_PORT "$sf"); [ -n "$v" ] && NATIVE_API_PORT=$v
  fi
  if [ -z "$_db_password_explicit" ]; then
    v=$(env_value_from_file DB_PASSWORD "$sf"); [ -n "$v" ] && NATIVE_DB_PASSWORD=$v
  fi
  v=$(env_value_from_file VOXINT_PASSWORD "$sf"); [ -n "$v" ] && VOXINT_NATIVE_PASSWORD=$v
  v=$(env_value_from_file CSRF_SECRET "$sf"); [ -n "$v" ] && VOXINT_NATIVE_CSRF_SECRET=$v
  return 0
}

write_state_env() {
  # umask 077 -> the temp is 0600 from birth; atomic mv; explicit chmod. Secrets
  # never touch a world-readable inode. Mirrors scripts/install.sh write_env.
  local tmp
  ( umask 077
    tmp=$(mktemp "$VOXINT_NATIVE_HOME/state.env.XXXXXX") || exit 1
    {
      printf '# Written by voxint-native.sh setup -- ports + secrets (mode 0600).\n'
      printf 'PG_PORT=%s\n' "$NATIVE_PG_PORT"
      printf 'REDIS_PORT=%s\n' "$NATIVE_REDIS_PORT"
      printf 'API_PORT=%s\n' "$NATIVE_API_PORT"
      printf 'DB_PASSWORD=%s\n' "$NATIVE_DB_PASSWORD"
      printf 'VOXINT_PASSWORD=%s\n' "$VOXINT_NATIVE_PASSWORD"
      printf 'CSRF_SECRET=%s\n' "$VOXINT_NATIVE_CSRF_SECRET"
    } > "$tmp"
    mv -f "$tmp" "$NATIVE_STATE"
    chmod 600 "$NATIVE_STATE"
  ) || fail "could not write $NATIVE_STATE"
}

# ---------------------------------------------------------------------------
# Per-service argv and environment. ONE assembly point each, used by both the
# launchd plist generator and `run --foreground`, so the supervised and debug
# paths cannot drift. native_program_args prints one argv element per line;
# native_service_env prints KEY=VALUE lines (datastores carry their config as
# flags and need no baked env). launchd inherits no shell environment.
# ---------------------------------------------------------------------------
native_program_args() {
  local svc=$1 venv
  venv=$(core_venv)
  case $svc in
    # Managed datastores. Bind 127.0.0.1 only; data + socket live under our home.
    postgres) printf '%s\n' "$NATIVE_PG_BINDIR/postgres" \
                "-D" "$NATIVE_PGDATA" "-p" "$NATIVE_PG_PORT" \
                "-k" "$VOXINT_NATIVE_HOME/run" "-c" "listen_addresses=127.0.0.1" ;;
    redis)    printf '%s\n' "$NATIVE_BREW_PREFIX/bin/redis-server" \
                "--port" "$NATIVE_REDIS_PORT" "--bind" "127.0.0.1" \
                "--dir" "$VOXINT_NATIVE_HOME" ;;
    # `voxint serve` reads API_HOST/API_PORT from the env below and calls
    # uvicorn.run("voxint.api.app:app", ...) -- byte-identical to the image CMD.
    api)    printf '%s\n' "$venv/bin/voxint" "serve" ;;
    worker) printf '%s\n' "$venv/bin/celery" "-A" "voxint.worker.app" "worker" "--loglevel=INFO" ;;
    # beat is a dedicated process (not worker --beat), matching compose; its
    # schedule file must sit on a writable path we own.
    beat)   printf '%s\n' "$venv/bin/celery" "-A" "voxint.worker.app" "beat" "--loglevel=INFO" \
              "-s" "$VOXINT_NATIVE_HOME/celerybeat-schedule" ;;
    *) return 1 ;;
  esac
}

native_service_env() {
  local svc=$1 media_root=$2
  case $svc in
    postgres)
      # macOS launchd inherits no locale; without a valid LC_ALL the postmaster
      # "becomes multithreaded during startup" and dies (a known Darwin trap).
      # Match the cluster's initdb --locale=C.
      printf 'LC_ALL=C\n'
      printf 'LANG=C\n'
      return 0
      ;;
    redis) return 0 ;;
    api|worker|beat) : ;;
    *) return 1 ;;
  esac
  printf 'DATABASE_URL=%s\n' "$(native_database_url)"
  printf 'REDIS_URL=%s\n' "$(native_redis_url)"
  printf 'MEDIA_ROOT=%s\n' "$media_root"
  # api/worker reach the model services at these URLs (loopback + the metal
  # launcher's ports by default; overridable for models on other hardware).
  # COMPUTE_TIER=metal picks the timing profile the Apple-Silicon tier was tuned
  # for.
  printf 'ASR_URL=%s\n' "$NATIVE_ASR_URL"
  printf 'DIARIZER_URL=%s\n' "$NATIVE_DIARIZER_URL"
  printf 'EMBEDDER_URL=%s\n' "$NATIVE_EMBEDDER_URL"
  printf 'COMPUTE_TIER=metal\n'
  printf 'PYTHONUNBUFFERED=1\n'
  # venv/bin first, then Homebrew (ffmpeg/ffprobe), then the system dirs.
  printf 'PATH=%s/bin:%s/bin:/usr/bin:/bin:/usr/sbin:/sbin\n' \
    "$(core_venv)" "$NATIVE_BREW_PREFIX"
  # Secrets are threaded through only when provided. Emitting an empty value
  # would override the app default with a weaker one.
  [ -n "$VOXINT_NATIVE_PASSWORD" ] \
    && printf 'VOXINT_PASSWORD=%s\n' "$VOXINT_NATIVE_PASSWORD"
  [ -n "$VOXINT_NATIVE_CSRF_SECRET" ] \
    && printf 'CSRF_SECRET=%s\n' "$VOXINT_NATIVE_CSRF_SECRET"
  case $svc in
    api)
      printf 'API_HOST=%s\n' "$NATIVE_API_HOST"
      printf 'API_PORT=%s\n' "$NATIVE_API_PORT"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# launchd plist generation. KeepAlive/SuccessfulExit=false restarts crashed
# services but lets `down` (bootout) and clean exits stay down -- the native
# analogue of the containers' contract-tested `restart: unless-stopped`.
# ---------------------------------------------------------------------------
xml_escape() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

render_plist() {
  # $1 = service, $2 = resolved MEDIA_ROOT, $3 = output path.
  local svc=$1 media_root=$2 out=$3 line key value env_block args_block arg
  # Captured OUTSIDE the output block: a failure inside the while-subshell pipes
  # below would be swallowed and ship a partial plist; here it aborts under set -e.
  env_block=$(native_service_env "$svc" "$media_root")
  args_block=$(native_program_args "$svc")
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '<plist version="1.0">\n<dict>\n'
    printf '  <key>Label</key><string>%s</string>\n' "$(plist_label "$svc")"
    printf '  <key>ProgramArguments</key>\n  <array>\n'
    printf '%s\n' "$args_block" | while IFS= read -r arg; do
      printf '    <string>%s</string>\n' "$(xml_escape "$arg")"
    done
    printf '  </array>\n'
    printf '  <key>WorkingDirectory</key><string>%s</string>\n' \
      "$(xml_escape "$REPO_ROOT")"
    printf '  <key>EnvironmentVariables</key>\n  <dict>\n'
    printf '%s\n' "$env_block" | while IFS= read -r line; do
      [ -n "$line" ] || continue
      key=${line%%=*}
      value=${line#*=}
      printf '    <key>%s</key><string>%s</string>\n' \
        "$(xml_escape "$key")" "$(xml_escape "$value")"
    done
    printf '  </dict>\n'
    printf '  <key>RunAtLoad</key><true/>\n'
    printf '  <key>KeepAlive</key>\n  <dict>\n'
    printf '    <key>SuccessfulExit</key><false/>\n'
    printf '  </dict>\n'
    printf '  <key>StandardOutPath</key><string>%s</string>\n' \
      "$(xml_escape "$(service_log "$svc")")"
    printf '  <key>StandardErrorPath</key><string>%s</string>\n' \
      "$(xml_escape "$(service_log "$svc")")"
    printf '</dict>\n</plist>\n'
  } > "$out"
}

# ---------------------------------------------------------------------------
# Model-service delegation. The metal launcher owns whisper/pyannote/titanet
# under com.voxint.metal.*; we drive it with the bare subcommand so a single
# native command controls the whole preview. MEDIA_ROOT agreement is by
# construction: both launchers read $REPO_ROOT/.env's MEDIA_ROOT and resolve it
# against $REPO_ROOT with `pwd -P`, so they land on the identical physical dir.
# (Metal's `up` REQUIRES that .env; native's own MEDIA_ROOT falls back to
# ./media, so without an .env the delegated model `up` fails with metal's own
# clear message while the core stack stays up -- an honest, non-fatal outcome.)
# ---------------------------------------------------------------------------
delegate_metal() {
  local cmd=$1 ms
  models_delegated || { say "  --no-models: leaving the model services to you ('$cmd' skipped)"; return 0; }
  ms=$(metal_script)
  # Test presence, not the executable bit, and invoke via `bash` (as the
  # logrotate plist does): a checkout that lost the +x bit (zip export,
  # core.fileMode=false) still has a runnable script. A genuinely absent
  # launcher when models WERE meant to be managed is a failure, not a skip.
  if [ ! -f "$ms" ]; then
    say "  model launcher not found at $ms -- cannot manage the model services"
    return 1
  fi
  step "Model services -> voxint-metal.sh $cmd"
  bash "$ms" "$cmd"
}

# ---------------------------------------------------------------------------
# Frontend island build + staging. Vite builds frontend/dist (base=/static/app/,
# manifest emitted at dist/.vite/manifest.json); the api serves those files from
# _APP_ASSETS_DIR. `setup` builds and stages; without node/npm it degrades
# gracefully (the console will not hydrate until the islands are built).
# ---------------------------------------------------------------------------
stage_frontend_dist() {
  local app dist
  app=$(app_assets_dir); dist=$(frontend_dir)/dist
  [ -f "$dist/.vite/manifest.json" ] \
    || fail "no built frontend at $dist (missing .vite/manifest.json) -- build it first"
  mkdir -p "$app"
  # OVERLAY the new build rather than wipe-then-copy. The api parses its manifest
  # once at import, so deleting the old hashed bundles before it restarts would
  # 404 the still-running console mid-upgrade; and a wipe that partially failed
  # could serve a new manifest against stale files. Copying the dist CONTENTS
  # (trailing /. = "contents of") overwrites the manifest + adds the new hashed
  # files, which is all the freshly-restarted api references. Old fingerprinted
  # bundles linger unreferenced (negligible for a single-operator preview);
  # the tracked .gitkeep is untouched.
  cp -R "$dist"/. "$app"/ || fail "staging frontend dist into $app failed"
}

build_frontend() {
  local fe app
  fe=$(frontend_dir); app=$(app_assets_dir)
  if [ ! -d "$fe" ]; then
    say "  no frontend/ directory at $fe -- skipping island build"
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1 || ! command -v node >/dev/null 2>&1; then
    say "  node/npm not found -- skipping island build; the console will not"
    say "  hydrate until you build $fe and re-run setup (see docs/operations/native-macos-preview.md)"
    return 0
  fi
  say "  npm ci && npm run build in $fe"
  ( cd "$fe" && npm ci >&2 && npm run build >&2 ) || fail "frontend build failed"
  stage_frontend_dist
  say "  islands staged -> $app"
}

# ---------------------------------------------------------------------------
# Log rotation. launchd opened each service's StandardOutPath fd ONCE at
# bootstrap, so an mv-style rotation would leave the running service writing
# into the archive's inode. copytruncate (cp then truncate the live inode)
# matches that fd reality; the handful of bytes written between the cp and the
# truncate are accepted losses for single-operator stdout logs. Lifted from the
# metal launcher, which supervises its services the same way.
# ---------------------------------------------------------------------------
rotate_log_file() {
  # $1 = live log path, $2 = max size in MB, $3 = newest archives to keep.
  local log=$1 max_mb=$2 keep=$3 size stamp archive
  [ -f "$log" ] || return 0
  size=$(wc -c < "$log")
  [ "$size" -ge $((max_mb * 1024 * 1024)) ] || return 0
  stamp=$(date +%Y-%m-%d-%H-%M-%S)
  archive=${log%.log}_$stamp.log
  cp "$log" "$archive" || return 1
  # Guard the truncate: rotate_log_file runs on the left of `||` (cmd_rotate_logs),
  # which disables errexit inside it, so a failed `: > "$log"` (e.g. a read-only
  # log) would otherwise be masked and the log would grow unbounded while we
  # claim success.
  : > "$log" || return 1
  say "rotated $(basename "$log") ($size bytes) -> $(basename "$archive")"
  prune_log_archives "$log" "$keep"
}

prune_log_archives() {
  # $1 = live log path, $2 = newest archives to keep. The timestamp format sorts
  # lexicographically == chronologically, so `sort -r | tail +N` drops the
  # oldest. macOS head has no negative -n; tail -n +K is POSIX on both platforms.
  local log=$1 keep=$2 dir base old
  dir=$(dirname "$log")
  base=$(basename "$log" .log)
  ls -1 "$dir" 2>/dev/null \
    | grep -E "^${base}_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}\.log$" \
    | sort -r | tail -n +$((keep + 1)) | while IFS= read -r old; do
      rm -f "$dir/$old"
    done
}

cmd_rotate_logs() {
  # Rotates every service log AND the rotation job's own log.
  local svc rc=0
  mkdir -p "$VOXINT_NATIVE_HOME/logs"
  for svc in $NATIVE_DATASTORES $NATIVE_SERVICES logrotate; do
    rotate_log_file "$(service_log "$svc")" \
      "$VOXINT_NATIVE_LOG_MAX_MB" "$VOXINT_NATIVE_LOG_ARCHIVES" || rc=1
  done
  return $rc
}

render_logrotate_plist() {
  # $1 = output path. Daily one-shot (03:17 -- an arbitrary quiet minute, off the
  # exact hour where periodic jobs pile up); launchd coalesces intervals missed
  # while the Mac slept into one run on wake. No KeepAlive: a clean exit stays
  # exited until the next calendar fire.
  local out=$1
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '<plist version="1.0">\n<dict>\n'
    printf '  <key>Label</key><string>%s</string>\n' "$(plist_label logrotate)"
    printf '  <key>ProgramArguments</key>\n  <array>\n'
    printf '    <string>/bin/bash</string>\n'
    printf '    <string>%s</string>\n' "$(xml_escape "$NATIVE_SCRIPT_DIR/voxint-native.sh")"
    printf '    <string>rotate-logs</string>\n'
    printf '  </array>\n'
    printf '  <key>EnvironmentVariables</key>\n  <dict>\n'
    printf '    <key>VOXINT_NATIVE_HOME</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_NATIVE_HOME")"
    printf '    <key>VOXINT_NATIVE_LOG_MAX_MB</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_NATIVE_LOG_MAX_MB")"
    printf '    <key>VOXINT_NATIVE_LOG_ARCHIVES</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_NATIVE_LOG_ARCHIVES")"
    printf '  </dict>\n'
    printf '  <key>RunAtLoad</key><false/>\n'
    printf '  <key>StartCalendarInterval</key>\n  <dict>\n'
    printf '    <key>Hour</key><integer>3</integer>\n'
    printf '    <key>Minute</key><integer>17</integer>\n'
    printf '  </dict>\n'
    printf '  <key>StandardOutPath</key><string>%s</string>\n' \
      "$(xml_escape "$(service_log logrotate)")"
    printf '  <key>StandardErrorPath</key><string>%s</string>\n' \
      "$(xml_escape "$(service_log logrotate)")"
    printf '</dict>\n</plist>\n'
  } > "$out"
}

install_logrotate() {
  # Rotate oversized logs now AND (re)install the daily job -- under KeepAlive the
  # services can run for months without another `up`, so the inline rotation
  # alone would not bound growth. Same bounded bootout-race dance as services.
  local plist label i
  cmd_rotate_logs || true
  plist=$(plist_path logrotate)
  label=$(plist_label logrotate)
  render_logrotate_plist "$plist"
  plutil -lint -s "$plist" || fail "generated plist failed plutil lint: $plist"
  launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1 || true
  i=0
  while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -le 50 ] || fail "$label did not unload within 10s (launchctl bootout race)"
    sleep 0.2
  done
  launchctl bootstrap "gui/$(id -u)" "$plist" \
    || fail "launchctl bootstrap failed for $label (see $(service_log logrotate))"
  say "installed $label (daily, keeps $VOXINT_NATIVE_LOG_ARCHIVES archives over ${VOXINT_NATIVE_LOG_MAX_MB}MB)"
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
require_macos() {
  [ "$(uname -s)" = "Darwin" ] || fail "the native tier is macOS-only (this is $(uname -s))"
  [ "$(uname -m)" = "arm64" ] || fail "the native tier needs Apple Silicon (this is $(uname -m))"
}

require_tools() {
  command -v uv >/dev/null 2>&1 \
    || fail "uv is required (https://docs.astral.sh/uv/ or: brew install uv)"
  command -v brew >/dev/null 2>&1 \
    || fail "Homebrew is required to provision Postgres/Redis (https://brew.sh)"
}

brew_install_datastores() {
  local f
  for f in postgresql@17 pgvector redis; do
    if brew list --versions "$f" >/dev/null 2>&1; then
      say "  $f already installed"
    else
      say "  brew install $f"
      brew install "$f" >&2 || fail "brew install $f failed"
    fi
  done
  # Prefer brew's own answer for the (keg-only) bindir when we can get it.
  local prefix
  prefix=$(brew --prefix postgresql@17 2>/dev/null) || prefix=""
  [ -n "$prefix" ] && [ -x "$prefix/bin/initdb" ] && NATIVE_PG_BINDIR=$prefix/bin
  [ -x "$NATIVE_PG_BINDIR/initdb" ] \
    || fail "cannot find initdb under $NATIVE_PG_BINDIR (set VOXINT_NATIVE_PG_BINDIR)"
}

init_cluster() {
  if [ -f "$NATIVE_PGDATA/PG_VERSION" ]; then
    say "  cluster already initialized at $NATIVE_PGDATA"
    return 0
  fi
  say "  initdb $NATIVE_PGDATA"
  # trust auth on the loopback-only cluster: this is a single-operator local
  # preview, not a shared server (#71 hardens the DB story). UTF-8 / C locale.
  "$NATIVE_PG_BINDIR/initdb" -D "$NATIVE_PGDATA" \
    --encoding=UTF8 --locale=C \
    --auth-local=trust --auth-host=trust >&2 \
    || fail "initdb failed"
}

ensure_database() {
  # Idempotent role/db/extension provisioning. Runs as the OS superuser initdb
  # created (the current user), over loopback TCP. The extension is created here
  # by the superuser so migration 0001's `CREATE EXTENSION IF NOT EXISTS vector`
  # (which the unprivileged voxint role could not run) simply no-ops.
  local psql=$NATIVE_PG_BINDIR/psql su
  su=$(id -un)
  "$psql" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" -d postgres -v ON_ERROR_STOP=1 -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='$NATIVE_DB_USER'" | grep -q 1 \
    || "$psql" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" -d postgres -v ON_ERROR_STOP=1 -c \
       "CREATE ROLE \"$NATIVE_DB_USER\" LOGIN PASSWORD '$NATIVE_DB_PASSWORD'" >&2
  "$psql" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" -d postgres -v ON_ERROR_STOP=1 -tAc \
    "SELECT 1 FROM pg_database WHERE datname='$NATIVE_DB_NAME'" | grep -q 1 \
    || "$NATIVE_PG_BINDIR/createdb" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" \
       -O "$NATIVE_DB_USER" "$NATIVE_DB_NAME" >&2
  "$psql" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" -d "$NATIVE_DB_NAME" -v ON_ERROR_STOP=1 -c \
    "CREATE EXTENSION IF NOT EXISTS vector" >&2 \
    || fail "CREATE EXTENSION vector failed -- is pgvector built against postgresql@17? (brew reinstall pgvector)"
}

cmd_setup() {
  require_macos
  require_tools
  local venv fresh=0 metal_setup_rc=0
  step "Directories under $VOXINT_NATIVE_HOME"
  mkdir -p "$VOXINT_NATIVE_HOME/logs" "$VOXINT_NATIVE_HOME/run" "$VOXINT_NATIVE_HOME/backups"

  step "Datastore binaries (Homebrew)"
  brew_install_datastores

  # Load any prior state first so re-running setup keeps the same ports/secrets;
  # only a fresh install picks ports and mints secrets.
  load_state
  [ -f "$NATIVE_STATE" ] || fresh=1
  if [ "$fresh" = 1 ]; then
    step "Choosing ports (moving off collisions) and generating secrets"
    [ -n "$_pg_port_explicit" ]    || NATIVE_PG_PORT=$(next_free_port "$NATIVE_PG_PORT")
    [ -n "$_redis_port_explicit" ] || NATIVE_REDIS_PORT=$(next_free_port "$NATIVE_REDIS_PORT")
    [ -n "$_api_port_explicit" ]   || NATIVE_API_PORT=$(next_free_port "$NATIVE_API_PORT")
    [ -n "$_db_password_explicit" ] || NATIVE_DB_PASSWORD=$(generate_secret) \
      || fail "secret generation failed"
    [ -n "$VOXINT_NATIVE_PASSWORD" ] || VOXINT_NATIVE_PASSWORD=$(generate_secret) \
      || fail "secret generation failed"
    [ -n "$VOXINT_NATIVE_CSRF_SECRET" ] || VOXINT_NATIVE_CSRF_SECRET=$(generate_secret) \
      || fail "secret generation failed"
    write_state_env
    say "  postgres :$NATIVE_PG_PORT  redis :$NATIVE_REDIS_PORT  api :$NATIVE_API_PORT"
    say "  secrets written to $NATIVE_STATE (mode 0600)"
  else
    say "  reusing ports + secrets from $NATIVE_STATE"
  fi

  step "Core Python 3.11 venv (uv)"
  venv=$(core_venv)
  if [ ! -x "$venv/bin/python" ]; then
    say "  creating venv: $venv"
    uv venv --python 3.11 "$venv" >&2
  fi
  say "  installing voxint (editable) into the core venv"
  uv pip install --quiet --python "$venv/bin/python" -e "$REPO_ROOT" >&2

  step "Frontend islands (npm build + stage)"
  build_frontend

  step "Private PostgreSQL 17 cluster"
  init_cluster

  # Bring the model tier's own setup along (weights download; network required),
  # unless --no-models. The core install is complete either way and metal setup
  # can be re-run, so we do NOT abort -- but we record the failure and return it
  # so a scripted `setup && up` sees the incomplete preview.
  delegate_metal setup || metal_setup_rc=$?
  [ "$metal_setup_rc" -eq 0 ] \
    || say "  model setup did not complete (network needed) -- retry: $(metal_script) setup"

  step "Setup complete"
  if models_delegated; then
    say "Start the whole preview (core + models) with: $0 up"
  else
    say "Start the core stack with: $0 up   (models run elsewhere: --no-models)"
    say "Note: submissions will fail until the model services are also up"
    say "(scripts/metal/voxint-metal.sh up)."
  fi
  return "$metal_setup_rc"
}

# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------
run_alembic() {
  # Migrations read DATABASE_URL via get_settings(); alembic.ini lives at the
  # repo root (prepend_sys_path=src), so run from there.
  ( cd "$REPO_ROOT" \
    && DATABASE_URL="$(native_database_url)" "$(core_venv)/bin/alembic" "$@" )
}

wait_for_postgres() {
  local i=0
  while [ "$i" -lt 60 ]; do
    if [ -x "$NATIVE_PG_BINDIR/pg_isready" ]; then
      "$NATIVE_PG_BINDIR/pg_isready" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -q && return 0
    elif command -v pg_isready >/dev/null 2>&1; then
      pg_isready -h 127.0.0.1 -p "$NATIVE_PG_PORT" -q && return 0
    else
      port_in_use "$NATIVE_PG_PORT" && return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  return 1
}

wait_for_redis() {
  local i=0 cli=$NATIVE_BREW_PREFIX/bin/redis-cli
  [ -x "$cli" ] || cli=redis-cli
  while [ "$i" -lt 60 ]; do
    if command -v "$cli" >/dev/null 2>&1 || [ -x "$cli" ]; then
      [ "$("$cli" -h 127.0.0.1 -p "$NATIVE_REDIS_PORT" ping 2>/dev/null)" = "PONG" ] \
        && return 0
    else
      port_in_use "$NATIVE_REDIS_PORT" && return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  return 1
}

bootstrap_service() {
  # Render + (re)load one service under launchd. Idempotent: bootout is a no-op
  # when not loaded, but RETURNS BEFORE the job is fully gone -- an immediate
  # bootstrap races it and fails with EIO -- so wait (bounded) for the unload.
  local svc=$1 media_root=$2 plist label i
  plist=$(plist_path "$svc")
  label=$(plist_label "$svc")
  render_plist "$svc" "$media_root" "$plist"
  plutil -lint -s "$plist" || fail "generated plist failed plutil lint: $plist"
  launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1 || true
  i=0
  while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -le 50 ] || fail "$label did not unload within 10s (launchctl bootout race)"
    sleep 0.2
  done
  launchctl bootstrap "gui/$(id -u)" "$plist" \
    || fail "launchctl bootstrap failed for $label (see $(service_log "$svc"))"
  say "started $label"
}

managed_cluster() { [ -f "$NATIVE_PGDATA/PG_VERSION" ]; }

cmd_up() {
  require_macos
  load_state
  local media_root svc managed=0 metal_up_rc=0
  [ -x "$(core_venv)/bin/voxint" ] || fail "core venv missing -- run: $0 setup"
  managed_cluster && managed=1
  media_root=$(resolved_media_root_or_fail)
  mkdir -p "$VOXINT_NATIVE_HOME/run" "$VOXINT_NATIVE_HOME/logs"

  if [ "$managed" = 1 ]; then
    step "Starting managed datastores under launchd"
    for svc in $NATIVE_DATASTORES; do
      bootstrap_service "$svc" "$media_root"
    done
  else
    step "Using operator-provided Postgres + Redis (no managed cluster found)"
  fi

  step "Waiting for Postgres + Redis"
  wait_for_postgres || fail "Postgres not reachable at 127.0.0.1:$NATIVE_PG_PORT (see: $0 doctor)"
  say "  Postgres reachable on :$NATIVE_PG_PORT"
  wait_for_redis || fail "Redis not reachable at 127.0.0.1:$NATIVE_REDIS_PORT"
  say "  Redis reachable on :$NATIVE_REDIS_PORT"

  if [ "$managed" = 1 ]; then
    step "Provisioning role / database / pgvector extension"
    ensure_database
    say "  role, database, and vector extension present"
  fi

  step "Applying migrations (alembic upgrade head) BEFORE starting api/worker"
  run_alembic upgrade head || fail "alembic upgrade head failed (see the error above)"
  say "  database at head"

  step "Starting core services under launchd"
  for svc in $NATIVE_SERVICES; do
    bootstrap_service "$svc" "$media_root"
  done

  step "Log rotation"
  install_logrotate

  # Bring up the model services (unless --no-models) so a single `up` yields the
  # whole preview. The core stack is NOT torn down on a model failure -- a
  # missing weight or .env leaves the console usable, submissions just fail until
  # the models are up. But we record the failure and return it, so `up && submit`
  # does not proceed against a half-up preview claiming success.
  delegate_metal up || metal_up_rc=$?
  [ "$metal_up_rc" -eq 0 ] \
    || say "  model services did not all start -- submissions will fail until they do (retry: $(metal_script) up; check: $(metal_script) doctor)"

  step "Ready"
  say "Core stack starting. Console: http://$NATIVE_API_HOST:$NATIVE_API_PORT"
  say "Check readiness with: $0 status"
  # Open the console in the default browser (macOS `open`); harmless if absent.
  # Wait (bounded) for /healthz first so the browser does not race the api's
  # bind and land on a connection-refused page.
  if command -v open >/dev/null 2>&1; then
    wait_for_api
    open "http://$NATIVE_API_HOST:$NATIVE_API_PORT" >/dev/null 2>&1 || true
  fi
  return "$metal_up_rc"
}

wait_for_api() {
  # Poll /healthz briefly (best-effort; never fails the caller). Used only to
  # avoid opening the browser before the api has bound its port.
  local i=0
  while [ "$i" -lt 20 ]; do
    curl -fsS -m 2 "http://127.0.0.1:$NATIVE_API_PORT/healthz" >/dev/null 2>&1 && return 0
    i=$((i + 1))
    sleep 0.5
  done
  return 0
}

cmd_down() {
  local svc label i
  # Core services first, then the datastores they depend on, then the daily
  # rotation job. Order among these does not matter for teardown.
  for svc in $NATIVE_SERVICES $NATIVE_DATASTORES logrotate; do
    label=$(plist_label "$svc")
    if launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1; then
      # bootout returns before the job is fully torn down; wait (bounded) so a
      # following `status`/`up` sees the real state, not a mid-teardown ghost.
      i=0
      while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do
        i=$((i + 1))
        [ "$i" -le 50 ] || break
        sleep 0.2
      done
      say "stopped $label"
    else
      say "$label was not running"
    fi
  done
  # Stop the model services too (unless --no-models). Non-fatal.
  delegate_metal down || true
}

cmd_status() {
  load_state
  local svc label state health
  step "Managed datastores"
  for svc in $NATIVE_DATASTORES; do
    label=$(plist_label "$svc")
    if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
      state="supervised"
    else
      state="NOT loaded"
    fi
    if [ "$svc" = postgres ]; then
      printf '%-9s %-12s :%s  %s\n' "$svc" "[$state]" "$NATIVE_PG_PORT" \
        "$(port_in_use "$NATIVE_PG_PORT" && echo listening || echo 'not reachable')"
    else
      printf '%-9s %-12s :%s  %s\n' "$svc" "[$state]" "$NATIVE_REDIS_PORT" \
        "$(port_in_use "$NATIVE_REDIS_PORT" && echo listening || echo 'not reachable')"
    fi
  done

  step "Core services"
  for svc in $NATIVE_SERVICES; do
    label=$(plist_label "$svc")
    if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
      state="supervised"
    else
      state="NOT loaded"
    fi
    if [ "$svc" = api ]; then
      health=$(curl -fsS -m 3 "http://127.0.0.1:$NATIVE_API_PORT/healthz" 2>/dev/null) \
        || health="unreachable"
      printf '%-9s %-12s :%s  %s\n' "$svc" "[$state]" "$NATIVE_API_PORT" "$health"
    else
      printf '%-9s %-12s\n' "$svc" "[$state]"
    fi
  done

  step "Version"
  say "working tree: $(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)"

  # Append the model tier's own status (unless --no-models). Non-fatal.
  delegate_metal status || true
}

cmd_logs() {
  local svc=${1:-} follow=${2:-}
  [ -n "$svc" ] || fail "usage: $0 logs <postgres|redis|api|worker|beat|logrotate> [-f]"
  case $svc in postgres|redis|api|worker|beat|logrotate) : ;; *) fail "unknown service: $svc" ;; esac
  if [ "$follow" = "-f" ]; then
    tail -F "$(service_log "$svc")"
  else
    tail -n 100 "$(service_log "$svc")"
  fi
}

# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------
cmd_backup() {
  load_state
  managed_cluster || fail "no managed cluster to back up (see: $0 setup)"
  wait_for_postgres || fail "Postgres not reachable on :$NATIVE_PG_PORT -- is the stack up?"
  local stamp out
  stamp=$(date +%Y-%m-%d-%H-%M-%S)
  out=$VOXINT_NATIVE_HOME/backups/voxint-$stamp.dump
  say "pg_dump -> $out"
  "$NATIVE_PG_BINDIR/pg_dump" -Fc -h 127.0.0.1 -p "$NATIVE_PG_PORT" \
    -U "$NATIVE_DB_USER" "$NATIVE_DB_NAME" > "$out" \
    || fail "pg_dump failed"
  say "backup complete: $out"
}

cmd_restore() {
  load_state
  local file=${1:-}
  [ -n "$file" ] || fail "usage: $0 restore <dump-file>"
  [ -f "$file" ] || fail "no such dump: $file"
  managed_cluster || fail "no managed cluster to restore into (see: $0 setup)"
  wait_for_postgres || fail "Postgres not reachable on :$NATIVE_PG_PORT -- is the stack up?"
  say "pg_restore <- $file (into $NATIVE_DB_NAME)"
  # --clean --if-exists so a restore over an existing schema replaces it.
  "$NATIVE_PG_BINDIR/pg_restore" --clean --if-exists --no-owner \
    -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$NATIVE_DB_USER" -d "$NATIVE_DB_NAME" "$file" \
    || fail "pg_restore reported errors (see above)"
  say "restore complete"
}

# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------
doctor_report() {
  # $1 = PASS/FAIL/SKIP, $2 = message
  printf '  [%s] %s\n' "$1" "$2" >&2
  [ "$1" = "FAIL" ] && DOCTOR_RC=1
  return 0
}

cmd_doctor() {
  DOCTOR_RC=0
  load_state
  local venv bin su

  step "Tooling"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    doctor_report PASS "macOS on Apple Silicon"
  else
    doctor_report FAIL "not macOS/arm64 ($(uname -s)/$(uname -m)) -- the native tier cannot run here"
  fi
  command -v uv >/dev/null 2>&1 \
    && doctor_report PASS "uv present" || doctor_report FAIL "uv missing"
  command -v brew >/dev/null 2>&1 \
    && doctor_report PASS "Homebrew present" || doctor_report FAIL "Homebrew missing"
  for bin in ffmpeg ffprobe; do
    if PATH="$NATIVE_BREW_PREFIX/bin:$PATH" command -v "$bin" >/dev/null 2>&1; then
      doctor_report PASS "$bin present"
    else
      doctor_report FAIL "$bin missing -- brew install ffmpeg (the PREPARE stage needs it)"
    fi
  done

  step "Datastore binaries"
  [ -x "$NATIVE_PG_BINDIR/postgres" ] \
    && doctor_report PASS "postgresql@17 binaries at $NATIVE_PG_BINDIR" \
    || doctor_report FAIL "postgresql@17 binaries missing -- run: $0 setup"
  [ -x "$NATIVE_BREW_PREFIX/bin/redis-server" ] \
    && doctor_report PASS "redis-server present" \
    || doctor_report FAIL "redis-server missing -- run: $0 setup"

  step "Core venv"
  venv=$(core_venv)
  if [ -x "$venv/bin/python" ]; then
    doctor_report PASS "core venv present ($venv)"
    "$venv/bin/python" -c "import voxint" >/dev/null 2>&1 \
      && doctor_report PASS "voxint importable in the core venv" \
      || doctor_report FAIL "voxint not importable in the core venv -- run: $0 setup"
    [ -x "$venv/bin/alembic" ] \
      && doctor_report PASS "alembic present" \
      || doctor_report FAIL "alembic missing in the core venv -- run: $0 setup"
    [ -x "$venv/bin/celery" ] \
      && doctor_report PASS "celery present" \
      || doctor_report FAIL "celery missing in the core venv -- run: $0 setup"
  else
    doctor_report FAIL "core venv missing ($venv) -- run: $0 setup"
  fi

  step "Cluster + datastores"
  if managed_cluster; then
    doctor_report PASS "private cluster initialized ($NATIVE_PGDATA)"
  else
    doctor_report SKIP "no managed cluster -- $0 up will expect operator-provided datastores"
  fi
  if wait_for_postgres_once; then
    doctor_report PASS "Postgres reachable on :$NATIVE_PG_PORT"
    su=$(id -un)
    if "$NATIVE_PG_BINDIR/psql" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -U "$su" -d postgres -tAc \
        "SELECT 1 FROM pg_available_extensions WHERE name='vector'" 2>/dev/null | grep -q 1; then
      doctor_report PASS "pgvector available to the cluster"
    else
      doctor_report FAIL "pgvector NOT available to the cluster -- brew reinstall pgvector (must build against postgresql@17)"
    fi
  else
    doctor_report SKIP "Postgres not running on :$NATIVE_PG_PORT -- $0 up"
  fi
  if redis_ping_once; then
    doctor_report PASS "Redis reachable on :$NATIVE_REDIS_PORT"
  else
    doctor_report SKIP "Redis not running on :$NATIVE_REDIS_PORT -- $0 up"
  fi

  step "MEDIA_ROOT"
  local envf=$REPO_ROOT/.env raw resolved
  raw=""
  [ -f "$envf" ] && raw=$(env_value_from_file MEDIA_ROOT "$envf")
  [ -n "$raw" ] || raw=${VOXINT_NATIVE_MEDIA_ROOT:-./media}
  if resolved=$(resolve_media_root "$raw" 2>/dev/null); then
    doctor_report PASS "MEDIA_ROOT=$raw resolves to $resolved"
  else
    doctor_report SKIP "MEDIA_ROOT=$raw not created yet ($0 up creates it)"
  fi
  # metal reads the SAME .env MEDIA_ROOT and resolves it against the SAME repo
  # root, so the two launchers land on the identical physical dir by construction.
  say "  (the model launcher reads the same MEDIA_ROOT from $envf)"

  step "Frontend islands"
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    doctor_report PASS "node + npm present (needed to build the console islands)"
  else
    doctor_report FAIL "node/npm missing -- the review console cannot be built (brew install node)"
  fi
  if [ -f "$(app_manifest_path)" ]; then
    doctor_report PASS "console islands staged ($(app_manifest_path))"
  else
    doctor_report FAIL "console islands not staged -- run: $0 setup (the console will not hydrate)"
  fi

  step "Model services"
  if models_delegated; then
    if [ -f "$(metal_script)" ]; then
      doctor_report PASS "delegating to $(metal_script) doctor (output below)"
      bash "$(metal_script)" doctor || DOCTOR_RC=1
    else
      doctor_report FAIL "model launcher not found at $(metal_script)"
    fi
  else
    doctor_report SKIP "--no-models: model services are managed elsewhere"
  fi

  if [ "$DOCTOR_RC" -eq 0 ]; then
    step "doctor: all checks passed"
  else
    step "doctor: FAILURES above"
  fi
  return "$DOCTOR_RC"
}

# Single-shot reachability probes for doctor (the up-path versions block up to
# 60s; doctor must report current state, not wait).
wait_for_postgres_once() {
  if [ -x "$NATIVE_PG_BINDIR/pg_isready" ]; then
    "$NATIVE_PG_BINDIR/pg_isready" -h 127.0.0.1 -p "$NATIVE_PG_PORT" -q && return 0
    return 1
  fi
  port_in_use "$NATIVE_PG_PORT"
}

redis_ping_once() {
  local cli=$NATIVE_BREW_PREFIX/bin/redis-cli
  [ -x "$cli" ] || cli=redis-cli
  if command -v "$cli" >/dev/null 2>&1 || [ -x "$cli" ]; then
    [ "$("$cli" -h 127.0.0.1 -p "$NATIVE_REDIS_PORT" ping 2>/dev/null)" = "PONG" ]
    return
  fi
  port_in_use "$NATIVE_REDIS_PORT"
}

# ---------------------------------------------------------------------------
# Foreground run (debugging)
# ---------------------------------------------------------------------------
cmd_run() {
  local svc=${1:-} mode=${2:-} media_root line oldifs
  [ -n "$svc" ] || fail "usage: $0 run <api|worker|beat> --foreground"
  case $svc in api|worker|beat) : ;; *) fail "run supports api|worker|beat (datastores go through: $0 up)" ;; esac
  [ "$mode" = "--foreground" ] \
    || fail "only --foreground is supported (background runs go through: $0 up)"
  load_state
  media_root=$(resolved_media_root_or_fail)
  # Same env assembly the plists use -- the debug path may not drift.
  while IFS= read -r line; do
    [ -n "$line" ] && export "${line?}"
  done <<EOF
$(native_service_env "$svc" "$media_root")
EOF
  cd "$REPO_ROOT"
  # Rebuild argv from the one-per-line output. IFS=newline keeps args with
  # spaces intact (a venv path under a spaced $HOME); noglob stops a stray
  # metachar in a path from expanding.
  oldifs=$IFS
  set -f
  IFS='
'
  # shellcheck disable=SC2046
  set -- $(native_program_args "$svc")
  set +f
  IFS=$oldifs
  exec "$@"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
  local cmd oldifs
  # Strip a LEADING --no-models (the common `tool --flag cmd` habit) before we
  # pick the subcommand, so `voxint-native --no-models up` is not mistaken for a
  # command named --no-models.
  while [ "${1:-}" = "--no-models" ]; do
    NATIVE_WITH_MODELS=0
    shift
  done
  cmd=${1:-}
  [ $# -gt 0 ] && shift
  # Also pull a TRAILING/interspersed --no-models out of the remaining args (it
  # applies to the delegating commands and is harmless elsewhere): flip the
  # module flag, then rebuild the positional args without it. Same IFS=newline
  # rebuild cmd_run uses -- array-free for Bash 3.2.
  if no_models_flag_present "$@"; then
    NATIVE_WITH_MODELS=0
    oldifs=$IFS
    set -f
    IFS='
'
    # shellcheck disable=SC2046
    set -- $(args_without_no_models "$@")
    set +f
    IFS=$oldifs
  fi
  case $cmd in
    setup)   cmd_setup "$@" ;;
    up)      cmd_up "$@" ;;
    down)    cmd_down "$@" ;;
    status)  cmd_status "$@" ;;
    logs)    cmd_logs "$@" ;;
    doctor)  cmd_doctor "$@" ;;
    backup)  cmd_backup "$@" ;;
    restore) cmd_restore "$@" ;;
    run)     cmd_run "$@" ;;
    rotate-logs) cmd_rotate_logs "$@" ;;
    *)
      say "voxint-native.sh -- native (no-Docker) core control plane for macOS/arm64"
      say "usage: $0 <setup|up|down|status|logs|doctor|backup|restore|run|rotate-logs> [--no-models]"
      say "  setup                 brew datastores + core venv + islands + cluster + secrets"
      say "  up / down             start/stop Postgres+Redis+api/worker/beat under launchd"
      say "                        (up provisions the db and runs alembic upgrade head first)"
      say "  status                supervision state + /healthz + datastore reachability"
      say "  logs <svc> [-f]       show/follow a service log"
      say "  doctor                environment checks"
      say "  backup                pg_dump -Fc into backups/"
      say "  restore <file>        pg_restore a dump"
      say "  run <svc> --foreground  debug api|worker|beat in the foreground"
      say "  rotate-logs           copytruncate-rotate oversized service logs"
      say "                        (also runs daily via launchd once 'up' has run)"
      say "  --no-models           skip driving scripts/metal/voxint-metal.sh"
      say "                        (setup/up/down/status/doctor); models run elsewhere"
      exit 1
      ;;
  esac
}

if [ "${VOXINT_NATIVE_LIB:-}" != "1" ]; then
  main "$@"
fi
