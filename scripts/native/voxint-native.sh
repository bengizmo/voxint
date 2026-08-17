#!/usr/bin/env bash
# voxint-native.sh -- native (no-Docker) core control plane for macOS/arm64.
#
# The counterpart to scripts/metal/voxint-metal.sh: that script supervises the
# three MODEL services natively; this one supervises the CORE control plane --
# the api server, the Celery worker, and Celery beat -- under launchd, with no
# Docker at all. Issue #69 (the MVP of epic #68, "run without Docker").
#
# This is a TECHNICAL PREVIEW, not the non-technical packaged release (#73). It
# keeps the current architecture (Postgres+pgvector, Redis, Celery, beat) to
# prove the native lifecycle end-to-end before any dependency is removed.
#
#   setup                 create the core venv (uv), install voxint editable
#   up | down             start/stop api/worker/beat under launchd (KeepAlive
#                         restarts them after a crash -- the native equivalent
#                         of the containers' restart policy). `up` runs
#                         `alembic upgrade head` BEFORE starting api/worker,
#                         reproducing compose's migrate gate.
#   status                per-service supervision state + api /healthz +
#                         Postgres/Redis reachability
#   logs <svc> [-f]       show (or follow) a service's log
#   doctor                environment checks: tooling, venv, ffmpeg/ffprobe,
#                         Postgres/Redis reachability, ports
#   run <svc> --foreground  run one service in the foreground for debugging
#
# Layout (override the root with VOXINT_NATIVE_HOME):
#   $HOME/.voxint-native/{venv,logs,run,backups}
#
# SLICE 1 SCOPE: Postgres 17 + pgvector and Redis are OPERATOR-PROVIDED and
# already running on 127.0.0.1 (doctor checks reachability). A later slice adds
# a launcher-managed private cluster; model-service delegation, the frontend
# island build, and log rotation land alongside it.
#
# Requirements: macOS on Apple Silicon, uv. Bash 3.2 compatible.
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

# Supervised core processes. Model services stay under voxint-metal.sh.
NATIVE_SERVICES="api worker beat"
LAUNCHD_PREFIX=com.voxint.native

# Core-service connection config. All overridable; the defaults match the app's
# own config.py fallbacks (localhost Postgres/Redis) with the host/port made
# explicit. The DB password default mirrors compose's voxint:voxint dev pair;
# a managed cluster (later slice) generates a real one into state.env.
NATIVE_DB_USER=${VOXINT_NATIVE_DB_USER:-voxint}
NATIVE_DB_NAME=${VOXINT_NATIVE_DB_NAME:-voxint}
NATIVE_DB_PASSWORD=${VOXINT_NATIVE_DB_PASSWORD:-voxint}
NATIVE_PG_PORT=${VOXINT_NATIVE_PG_PORT:-5432}
NATIVE_REDIS_PORT=${VOXINT_NATIVE_REDIS_PORT:-6379}
NATIVE_API_HOST=${VOXINT_NATIVE_API_HOST:-127.0.0.1}
NATIVE_API_PORT=${VOXINT_NATIVE_API_PORT:-8080}
# Homebrew prefix -- put its bin on each service's PATH so the worker finds
# ffmpeg/ffprobe (the app resolves the bare names "ffmpeg"/"ffprobe" on PATH,
# and launchd inherits none of the login shell's PATH). Arm64 default.
NATIVE_BREW_PREFIX=${VOXINT_NATIVE_BREW_PREFIX:-/opt/homebrew}

# Model-service ports the api/worker reach over loopback. These are owned by
# scripts/metal/voxint-metal.sh (service_port there); a contract test binds the
# two so a port moved in one place cannot silently orphan the other.
WHISPER_PORT=8022
PYANNOTE_PORT=8024
TITANET_PORT=8021

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

native_database_url() {
  printf 'postgresql+psycopg://%s:%s@127.0.0.1:%s/%s' \
    "$NATIVE_DB_USER" "$NATIVE_DB_PASSWORD" "$NATIVE_PG_PORT" "$NATIVE_DB_NAME"
}

native_redis_url() {
  printf 'redis://127.0.0.1:%s/0' "$NATIVE_REDIS_PORT"
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
# Per-service argv and environment. ONE assembly point each, used by both the
# launchd plist generator and `run --foreground`, so the supervised and debug
# paths cannot drift. native_program_args prints one argv element per line;
# native_service_env prints KEY=VALUE lines. launchd inherits no shell
# environment -- everything a service needs must be listed explicitly.
# ---------------------------------------------------------------------------
native_program_args() {
  local svc=$1 venv
  venv=$(core_venv)
  case $svc in
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
  case $svc in api|worker|beat) : ;; *) return 1 ;; esac
  printf 'DATABASE_URL=%s\n' "$(native_database_url)"
  printf 'REDIS_URL=%s\n' "$(native_redis_url)"
  printf 'MEDIA_ROOT=%s\n' "$media_root"
  # api/worker reach the model services over loopback (metal launcher supervises
  # them). COMPUTE_TIER=metal picks the timing profile the Apple-Silicon tier
  # was tuned for.
  printf 'ASR_URL=http://127.0.0.1:%s\n' "$WHISPER_PORT"
  printf 'DIARIZER_URL=http://127.0.0.1:%s\n' "$PYANNOTE_PORT"
  printf 'EMBEDDER_URL=http://127.0.0.1:%s\n' "$TITANET_PORT"
  printf 'COMPUTE_TIER=metal\n'
  printf 'PYTHONUNBUFFERED=1\n'
  # venv/bin first, then Homebrew (ffmpeg/ffprobe), then the system dirs.
  printf 'PATH=%s/bin:%s/bin:/usr/bin:/bin:/usr/sbin:/sbin\n' \
    "$(core_venv)" "$NATIVE_BREW_PREFIX"
  # Secrets are threaded through only when provided (a managed deployment sets
  # them; on loopback the app tolerates its own defaults). Emitting an empty
  # value would override the app default with a weaker one.
  [ -n "${VOXINT_NATIVE_PASSWORD:-}" ] \
    && printf 'VOXINT_PASSWORD=%s\n' "$VOXINT_NATIVE_PASSWORD"
  [ -n "${VOXINT_NATIVE_CSRF_SECRET:-}" ] \
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
# Setup
# ---------------------------------------------------------------------------
require_macos() {
  [ "$(uname -s)" = "Darwin" ] || fail "the native tier is macOS-only (this is $(uname -s))"
  [ "$(uname -m)" = "arm64" ] || fail "the native tier needs Apple Silicon (this is $(uname -m))"
}

require_tools() {
  command -v uv >/dev/null 2>&1 \
    || fail "uv is required (https://docs.astral.sh/uv/ or: brew install uv)"
}

cmd_setup() {
  require_macos
  require_tools
  local venv
  step "Directories under $VOXINT_NATIVE_HOME"
  mkdir -p "$VOXINT_NATIVE_HOME/logs" "$VOXINT_NATIVE_HOME/run" "$VOXINT_NATIVE_HOME/backups"

  step "Core Python 3.11 venv (uv)"
  venv=$(core_venv)
  if [ ! -x "$venv/bin/python" ]; then
    say "  creating venv: $venv"
    uv venv --python 3.11 "$venv" >&2
  fi
  say "  installing voxint (editable) into the core venv"
  uv pip install --quiet --python "$venv/bin/python" -e "$REPO_ROOT" >&2

  step "Setup complete"
  say "This preview slice expects an operator-provided PostgreSQL 17 + pgvector"
  say "and Redis on 127.0.0.1. Verify with: $0 doctor"
  say "Then start the core stack with: $0 up"
  say "Note: submissions will fail until the model services are also up"
  say "(scripts/metal/voxint-metal.sh up)."
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
    if command -v pg_isready >/dev/null 2>&1; then
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
  local i=0
  while [ "$i" -lt 60 ]; do
    if command -v redis-cli >/dev/null 2>&1; then
      [ "$(redis-cli -h 127.0.0.1 -p "$NATIVE_REDIS_PORT" ping 2>/dev/null)" = "PONG" ] \
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

cmd_up() {
  require_macos
  local media_root svc
  [ -x "$(core_venv)/bin/voxint" ] || fail "core venv missing -- run: $0 setup"
  media_root=$(resolved_media_root_or_fail)
  mkdir -p "$VOXINT_NATIVE_HOME/run" "$VOXINT_NATIVE_HOME/logs"

  step "Waiting for Postgres + Redis (operator-provided in this slice)"
  wait_for_postgres || fail "Postgres not reachable at 127.0.0.1:$NATIVE_PG_PORT -- start PostgreSQL 17 + pgvector, then re-run (see: $0 doctor)"
  say "  Postgres reachable on :$NATIVE_PG_PORT"
  wait_for_redis || fail "Redis not reachable at 127.0.0.1:$NATIVE_REDIS_PORT -- start Redis, then re-run"
  say "  Redis reachable on :$NATIVE_REDIS_PORT"

  step "Applying migrations (alembic upgrade head) BEFORE starting api/worker"
  run_alembic upgrade head || fail "alembic upgrade head failed (see the error above)"
  say "  database at head"

  step "Starting core services under launchd"
  for svc in $NATIVE_SERVICES; do
    bootstrap_service "$svc" "$media_root"
  done
  say "Core stack starting. Console: http://$NATIVE_API_HOST:$NATIVE_API_PORT"
  say "Check readiness with: $0 status"
}

cmd_down() {
  local svc label
  for svc in $NATIVE_SERVICES; do
    label=$(plist_label "$svc")
    if launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1; then
      say "stopped $label"
    else
      say "$label was not running"
    fi
  done
}

cmd_status() {
  local svc label state health pg redis
  step "Native core services"
  for svc in $NATIVE_SERVICES; do
    label=$(plist_label "$svc")
    if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
      state="supervised"
    else
      state="NOT loaded"
    fi
    if [ "$svc" = "api" ]; then
      health=$(curl -fsS -m 3 "http://127.0.0.1:$NATIVE_API_PORT/healthz" 2>/dev/null) \
        || health="unreachable"
      printf '%-7s %-12s :%s  %s\n' "$svc" "[$state]" "$NATIVE_API_PORT" "$health"
    else
      printf '%-7s %-12s\n' "$svc" "[$state]"
    fi
  done

  step "Datastores"
  if port_in_use "$NATIVE_PG_PORT"; then pg="listening"; else pg="not reachable"; fi
  if port_in_use "$NATIVE_REDIS_PORT"; then redis="listening"; else redis="not reachable"; fi
  printf 'postgres  :%s  %s\n' "$NATIVE_PG_PORT" "$pg"
  printf 'redis     :%s  %s\n' "$NATIVE_REDIS_PORT" "$redis"

  step "Version"
  say "working tree: $(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)"
}

cmd_logs() {
  local svc=${1:-} follow=${2:-}
  [ -n "$svc" ] || fail "usage: $0 logs <api|worker|beat> [-f]"
  case $svc in api|worker|beat) : ;; *) fail "unknown service: $svc" ;; esac
  if [ "$follow" = "-f" ]; then
    # -F, not -f: a manually deleted/recreated log would strand a plain -f follower.
    tail -F "$(service_log "$svc")"
  else
    tail -n 100 "$(service_log "$svc")"
  fi
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
  local venv bin

  step "Tooling"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    doctor_report PASS "macOS on Apple Silicon"
  else
    doctor_report FAIL "not macOS/arm64 ($(uname -s)/$(uname -m)) -- the native tier cannot run here"
  fi
  command -v uv >/dev/null 2>&1 \
    && doctor_report PASS "uv present" || doctor_report FAIL "uv missing"
  for bin in ffmpeg ffprobe; do
    if PATH="$NATIVE_BREW_PREFIX/bin:$PATH" command -v "$bin" >/dev/null 2>&1; then
      doctor_report PASS "$bin present"
    else
      doctor_report FAIL "$bin missing -- brew install ffmpeg (the PREPARE stage needs it)"
    fi
  done

  step "Core venv"
  venv=$(core_venv)
  if [ -x "$venv/bin/python" ]; then
    doctor_report PASS "core venv present ($venv)"
    if "$venv/bin/python" -c "import voxint" >/dev/null 2>&1; then
      doctor_report PASS "voxint importable in the core venv"
    else
      doctor_report FAIL "voxint not importable in the core venv -- run: $0 setup"
    fi
    [ -x "$venv/bin/alembic" ] \
      && doctor_report PASS "alembic present" \
      || doctor_report FAIL "alembic missing in the core venv -- run: $0 setup"
    [ -x "$venv/bin/celery" ] \
      && doctor_report PASS "celery present" \
      || doctor_report FAIL "celery missing in the core venv -- run: $0 setup"
  else
    doctor_report FAIL "core venv missing ($venv) -- run: $0 setup"
  fi

  step "Datastores (operator-provided in this slice)"
  if command -v pg_isready >/dev/null 2>&1 \
      && pg_isready -h 127.0.0.1 -p "$NATIVE_PG_PORT" -q; then
    doctor_report PASS "Postgres ready on :$NATIVE_PG_PORT"
  elif port_in_use "$NATIVE_PG_PORT"; then
    doctor_report PASS "something is listening on :$NATIVE_PG_PORT (pg_isready unavailable to confirm)"
  else
    doctor_report FAIL "Postgres not reachable on :$NATIVE_PG_PORT -- start PostgreSQL 17 + pgvector"
  fi
  if command -v redis-cli >/dev/null 2>&1 \
      && [ "$(redis-cli -h 127.0.0.1 -p "$NATIVE_REDIS_PORT" ping 2>/dev/null)" = "PONG" ]; then
    doctor_report PASS "Redis ready on :$NATIVE_REDIS_PORT"
  elif port_in_use "$NATIVE_REDIS_PORT"; then
    doctor_report PASS "something is listening on :$NATIVE_REDIS_PORT (redis-cli unavailable to confirm)"
  else
    doctor_report FAIL "Redis not reachable on :$NATIVE_REDIS_PORT -- start Redis"
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

  if [ "$DOCTOR_RC" -eq 0 ]; then
    step "doctor: all checks passed"
  else
    step "doctor: FAILURES above"
  fi
  return "$DOCTOR_RC"
}

# ---------------------------------------------------------------------------
# Foreground run (debugging)
# ---------------------------------------------------------------------------
cmd_run() {
  local svc=${1:-} mode=${2:-} media_root line oldifs
  [ -n "$svc" ] || fail "usage: $0 run <api|worker|beat> --foreground"
  native_program_args "$svc" >/dev/null || fail "unknown service: $svc"
  [ "$mode" = "--foreground" ] \
    || fail "only --foreground is supported (background runs go through: $0 up)"
  media_root=$(resolved_media_root_or_fail)
  # Same env assembly the plists use -- the debug path may not drift.
  while IFS= read -r line; do
    export "${line?}"
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
  local cmd=${1:-}
  [ $# -gt 0 ] && shift
  case $cmd in
    setup)  cmd_setup "$@" ;;
    up)     cmd_up "$@" ;;
    down)   cmd_down "$@" ;;
    status) cmd_status "$@" ;;
    logs)   cmd_logs "$@" ;;
    doctor) cmd_doctor "$@" ;;
    run)    cmd_run "$@" ;;
    *)
      say "voxint-native.sh -- native (no-Docker) core control plane for macOS/arm64"
      say "usage: $0 <setup|up|down|status|logs|doctor|run>"
      say "  setup                 create the core venv, install voxint editable"
      say "  up / down             start/stop api/worker/beat under launchd"
      say "                        (up runs alembic upgrade head first)"
      say "  status                supervision state + /healthz + datastore reachability"
      say "  logs <svc> [-f]       show/follow a service log"
      say "  doctor                environment checks"
      say "  run <svc> --foreground  debug one service in the foreground"
      exit 1
      ;;
  esac
}

if [ "${VOXINT_NATIVE_LIB:-}" != "1" ]; then
  main "$@"
fi
