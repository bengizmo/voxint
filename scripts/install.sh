#!/usr/bin/env bash
#
# Voxint guided installer.
#
# One command that takes a fresh clone to a running core stack: it asks the
# minimum questions (admin password, media folder), generates the rest, renders
# a .env, pulls the pinned release images, starts the stack, waits for the API
# to report healthy, and prints the console URL. It orchestrates the documented
# Quickstart -- it does not reimplement Compose.
#
# What this installs: the CORE control plane (Postgres, Redis, the API + review
# console, the Celery worker and beat). That is enough to open the console,
# browse runs, and adjudicate -- but NOT to process audio. Transcription,
# diarization, and speaker embedding need the GPU model services (compose.gpu.yaml)
# or external model endpoints; the completion notice says so. GPU setup is a
# separate, guided step (see the printed docs pointer).
#
# Requirements: Docker Engine with the Compose plugin >= 2.24. No other runtime
# dependency. Tested on Linux and macOS; Bash 3.2 compatible.
#
# Safe to re-run: an existing .env defaults to keep-and-run; regeneration always
# backs up first. The script never deletes containers, volumes, or media.

set -eu

# ---------------------------------------------------------------------------
# Small output helpers. Everything human-facing goes to stderr so stdout stays
# clean for the rare `$(...)`-captured value; secrets are NEVER printed.
# ---------------------------------------------------------------------------
say()  { printf '%s\n' "$*" >&2; }
step() { printf '\n==> %s\n' "$*" >&2; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

# Remove the temp .env (which briefly holds the password + CSRF secret) on any
# exit, including SIGINT/SIGTERM/SIGHUP. Only ever removes the exact mktemp path
# recorded in _CLEANUP_TMP -- never user data. The trap is armed in main().
_CLEANUP_TMP=""
_cleanup() { [ -n "${_CLEANUP_TMP:-}" ] && rm -f "$_CLEANUP_TMP"; return 0; }

# ---------------------------------------------------------------------------
# Repo root: resolve relative to this script so the installer works from any CWD.
# ---------------------------------------------------------------------------
# ${BASH_SOURCE[0]} (not $0) so the path is correct when the script is sourced
# in library mode by the test harness, as well as when executed directly.
SCRIPT_SELF=${BASH_SOURCE[0]:-$0}
SCRIPT_DIR=$(CDPATH= cd -P -- "$(dirname -- "$SCRIPT_SELF")" && pwd)
REPO_ROOT=$(CDPATH= cd -P -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

[ -f compose.yaml ]  || fail "compose.yaml not found in $REPO_ROOT -- run this from a Voxint checkout."
[ -f .env.example ]  || fail ".env.example not found in $REPO_ROOT -- run this from a Voxint checkout."

# A .env that exists but is not a regular file (a directory or a dangling
# symlink) would make `[ -f .env ]` read as "absent" and later corrupt that
# path (mv into a dir, chmod the dir). Refuse rather than mangle it.
if { [ -e .env ] || [ -L .env ]; } && [ ! -f .env ]; then
  fail ".env exists but is not a regular file. Move it aside and re-run."
fi

# Managed keys we own in .env. Unset any inherited shell exports of these so
# they cannot silently shadow the .env values during Compose interpolation
# (shell environment outranks the .env file). We do not touch DATABASE_URL etc.
unset VOXINT_PASSWORD MEDIA_ROOT CSRF_SECRET POSTGRES_PORT REDIS_PORT API_PORT HF_TOKEN 2>/dev/null || true

# ---------------------------------------------------------------------------
# Preflight: Docker CLI, a running daemon, and the Compose plugin >= 2.24.
# ---------------------------------------------------------------------------
preflight() {
  step "Checking Docker and the Compose plugin"

  command -v docker >/dev/null 2>&1 || fail \
    "Docker is not installed. Install Docker Desktop (macOS) or Docker Engine (Linux): https://docs.docker.com/engine/install/"

  docker info >/dev/null 2>&1 || fail \
    "The Docker daemon is not reachable. Start Docker Desktop (macOS) or the Docker service (Linux), then re-run."

  local version major rest minor
  version=$(docker compose version --short 2>/dev/null) || fail \
    "The Docker Compose plugin is missing. This stack needs the v2 plugin ('docker compose'), not the legacy 'docker-compose'. See https://docs.docker.com/compose/install/"

  version=${version#v}
  major=${version%%.*}
  rest=${version#*.}
  minor=${rest%%.*}
  case $major in ''|*[!0-9]*) fail "Could not parse Docker Compose version '$version'." ;; esac
  case $minor in ''|*[!0-9]*) fail "Could not parse Docker Compose version '$version'." ;; esac
  if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 24 ]; }; then
    fail "Docker Compose >= 2.24 required (found $version). Update Docker Desktop, or the compose plugin: https://docs.docker.com/compose/install/"
  fi

  say "Docker OK; Compose plugin $version."
}

# ---------------------------------------------------------------------------
# .env value encoding. Values are written single-quoted, which is fully literal
# under Compose dotenv rules (no interpolation, no escapes) and so safe for $,
# ", \, spaces, and #. Single quotes cannot be represented inside a
# single-quoted dotenv value, so we reject them at input time instead of
# emitting something Compose would mis-parse.
# ---------------------------------------------------------------------------
has_single_quote() { case $1 in *\'*) return 0 ;; *) return 1 ;; esac; }
# A value ending in a backslash cannot be single-quoted for Compose dotenv: the
# trailing `\` escapes the closing quote (`'abc\'` -> "unterminated quoted
# value"). Internal backslashes are literal and fine; only the trailing one bites.
ends_with_backslash() { case $1 in *\\) return 0 ;; *) return 1 ;; esac; }
dotenv_squote()    { printf "'%s'" "$1"; }  # caller guarantees no embedded ' or trailing \

valid_port() {
  case $1 in ''|*[!0-9]*) return 1 ;; esac
  [ "$1" -ge 1 ] && [ "$1" -le 65535 ]
}

# Advisory only: true if something already accepts a TCP connection on
# 127.0.0.1:$1. Uses bash /dev/tcp (no netcat/lsof dependency). Compose remains
# the authority -- this just lets us ask about a collision before we hit it.
port_in_use() {
  case $1 in ''|*[!0-9]*) return 1 ;; esac
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1 || return 1
  exec 3>&- 2>/dev/null || true
  return 0
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
# CSRF secret: 32 bytes of /dev/urandom as 64 hex chars (256 bits). No openssl
# or python dependency, no base64/SIGPIPE portability traps. Exceeds the app's
# 16-char minimum; hex needs no dotenv escaping.
# ---------------------------------------------------------------------------
generate_secret() {
  local s
  s=$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n') || return 1
  [ "${#s}" -eq 64 ] || return 1
  printf '%s' "$s"
}

# ---------------------------------------------------------------------------
# Prompts. A TTY is required whenever we need input; fail early and clearly if
# stdin is not interactive rather than looping on EOF.
# ---------------------------------------------------------------------------
require_tty() {
  [ -t 0 ] || fail "This installer needs an interactive terminal for setup. Run it directly in a terminal (not piped)."
}

prompt_password() {
  local pw confirm
  while :; do
    printf 'Admin password (for the web console login): ' >&2
    IFS= read -r -s pw || fail "No input."
    printf '\n' >&2
    printf 'Confirm password: ' >&2
    IFS= read -r -s confirm || fail "No input."
    printf '\n' >&2

    if [ -z "$pw" ]; then
      say "  Password cannot be empty (the API refuses an empty password)."; continue
    fi
    if [ "$pw" != "$confirm" ]; then
      say "  Passwords did not match -- try again."; continue
    fi
    case $pw in
      change-me) say "  Please choose something other than the placeholder 'change-me'."; continue ;;
    esac
    if has_single_quote "$pw"; then
      say "  Please choose a password without a single-quote (') character."; continue
    fi
    if ends_with_backslash "$pw"; then
      say "  Please choose a password that does not end with a backslash (\\)."; continue
    fi
    PASSWORD=$pw
    return 0
  done
}

prompt_media_root() {
  local ans
  printf "Media folder for ingested audio/video [./media]: " >&2
  IFS= read -r ans || fail "No input."
  ans=${ans:-./media}
  # Expand a leading ~/ (the shell does not, inside a read value).
  case $ans in
    "~/"*) ans="$HOME/${ans#\~/}" ;;
    "~")   ans="$HOME" ;;
  esac
  # `read` never expands $VAR, so a typed `$HOME/media` would create a literal
  # directory named `$HOME`. A ':' breaks the Compose short-volume syntax. Refuse
  # both rather than silently use the wrong directory.
  case $ans in
    *'$'*) fail "Media path must not contain '\$' (shell variables are not expanded here) -- type a literal path." ;;
    *:*)   fail "Media path must not contain ':' (it breaks the Docker volume mount): $ans" ;;
  esac
  if has_single_quote "$ans"; then
    fail "Media path must not contain a single-quote (') character: $ans"
  fi
  if ends_with_backslash "$ans"; then
    fail "Media path must not end with a backslash (\\): $ans"
  fi
  MEDIA_ROOT_VALUE=$ans
}

# Only prompts when the default host port is already busy (minimum questions).
# Echoes the chosen port on stdout; leaves it at the default otherwise.
resolve_port() {
  local label=$1 def=$2 suggested ans
  if ! port_in_use "$def"; then
    printf '%s' "$def"; return 0
  fi
  suggested=$(next_free_port "$def")
  say "  Host port $def ($label) is already in use."
  while :; do
    printf '  Alternate %s port [%s]: ' "$label" "$suggested" >&2
    IFS= read -r ans || fail "No input."
    ans=${ans:-$suggested}
    if valid_port "$ans" && ! port_in_use "$ans"; then
      printf '%s' "$ans"; return 0
    fi
    say "  '$ans' is not a free, valid port (1-65535)."
  done
}

# ---------------------------------------------------------------------------
# Render .env by streaming .env.example and replacing only the keys we manage
# (matching commented defaults too), then validate the candidate with Compose
# before an atomic, mode-0600 move into place. Never sed raw input; never
# source .env.
#
# Managed values are read from these globals (empty = leave the template line
# untouched): PASSWORD, MEDIA_ROOT_VALUE, CSRF_VALUE, PG_PORT, RD_PORT, API_PORT_VALUE.
# Ports are only substituted when they differ from the template default.
# ---------------------------------------------------------------------------
managed_replacement() {
  # $1 = KEY; echoes the full replacement line, or nothing to pass through.
  case $1 in
    VOXINT_PASSWORD) printf 'VOXINT_PASSWORD=%s' "$(dotenv_squote "$PASSWORD")" ;;
    MEDIA_ROOT)      printf 'MEDIA_ROOT=%s'      "$(dotenv_squote "$MEDIA_ROOT_VALUE")" ;;
    CSRF_SECRET)     printf 'CSRF_SECRET=%s'     "$CSRF_VALUE" ;;
    POSTGRES_PORT)   if [ -n "${PG_PORT:-}" ];        then printf 'POSTGRES_PORT=%s' "$PG_PORT"; fi ;;
    REDIS_PORT)      if [ -n "${RD_PORT:-}" ];        then printf 'REDIS_PORT=%s'    "$RD_PORT"; fi ;;
    API_PORT)        if [ -n "${API_PORT_VALUE:-}" ]; then printf 'API_PORT=%s'     "$API_PORT_VALUE"; fi ;;
  esac
  return 0  # never let a non-matching / empty branch fail under `set -e`
}

# The keys we intend to write on this run (password/media/csrf always; ports
# only when overridden). Used as a safety net so an override can never be
# silently dropped if .env.example's template line is reformatted or removed.
managed_keys_with_values() {
  printf '%s\n' VOXINT_PASSWORD MEDIA_ROOT CSRF_SECRET
  if [ -n "${PG_PORT:-}" ];        then printf '%s\n' POSTGRES_PORT; fi
  if [ -n "${RD_PORT:-}" ];        then printf '%s\n' REDIS_PORT; fi
  if [ -n "${API_PORT_VALUE:-}" ]; then printf '%s\n' API_PORT; fi
}

write_env() {
  local tmp line stripped key repl emitted=" " k
  # Restrict permissions on the secret-bearing file from creation.
  local old_umask; old_umask=$(umask); umask 077
  tmp=$(mktemp "$REPO_ROOT/.env.tmp.XXXXXX") || { umask "$old_umask"; fail "Could not create a temp file in $REPO_ROOT."; }
  _CLEANUP_TMP=$tmp  # the EXIT/signal trap removes this if we die mid-write

  # Read the template line by line, preserving everything we do not manage.
  # IFS= and -r keep whitespace and backslashes intact; the [ -n ] guard emits
  # a final line that has no trailing newline.
  while IFS= read -r line || [ -n "$line" ]; do
    stripped=${line#\#}                              # drop one optional leading '#'
    stripped=${stripped#"${stripped%%[![:blank:]]*}"} # then any run of blanks (tolerate reformatting)
    case $stripped in
      [A-Za-z_]*=*)
        key=${stripped%%=*}
        repl=$(managed_replacement "$key")
        if [ -n "$repl" ]; then
          printf '%s\n' "$repl" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
          emitted="$emitted$key "
        else
          printf '%s\n' "$line" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
        fi
        ;;
      *)
        printf '%s\n' "$line" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
        ;;
    esac
  done < .env.example

  # Safety net: append any managed key that the template didn't carry, so an
  # override can never be silently dropped (a matched line is already replaced
  # above, so this only fires for a genuinely missing key).
  for k in $(managed_keys_with_values); do
    case $emitted in
      *" $k "*) : ;;
      *) printf '%s\n' "$(managed_replacement "$k")" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; } ;;
    esac
  done
  umask "$old_umask"

  # Let Compose parse the candidate as the authority on dotenv correctness
  # (this also proves the required VOXINT_PASSWORD interpolation resolves).
  if ! docker compose --env-file "$tmp" config --quiet >/dev/null 2>&1; then
    fail "Generated .env failed Compose validation. This is a bug -- please report it."
  fi

  mv -f "$tmp" .env || fail "Could not move the generated .env into place."
  _CLEANUP_TMP=""  # .env is in place; nothing left to clean
  chmod 600 .env 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Decide what to do about an existing .env. Default is keep-and-run.
# ---------------------------------------------------------------------------
ENV_ACTION=""  # "generate" | "keep"
decide_existing_env() {
  if [ ! -f .env ]; then
    ENV_ACTION="generate"; return 0
  fi

  step "An existing .env was found"
  say "  [K]eep it and (re)start the stack   (default)"
  say "  [B]ack it up and generate a fresh one"
  say "  [A]bort without changing anything"
  local ans
  printf 'Choice [K/b/a]: ' >&2
  IFS= read -r ans || ans=""
  case $ans in
    ''|k|K|keep|Keep)
      if ! docker compose config --quiet >/dev/null 2>&1; then
        say "  The existing .env does not pass Compose validation (it may be missing VOXINT_PASSWORD)."
        printf '  Back it up and generate a fresh one now? [y/N]: ' >&2
        IFS= read -r ans || ans=""
        case $ans in
          y|Y|yes|Yes) ENV_ACTION="generate" ;;
          *) fail "Kept an invalid .env. Fix it (see .env.example) or re-run and choose regenerate." ;;
        esac
      else
        ENV_ACTION="keep"
      fi
      ;;
    b|B|backup|Backup) ENV_ACTION="generate" ;;
    a|A|abort|Abort)   say "Aborted; nothing changed."; exit 0 ;;
    *) fail "Unrecognized choice '$ans'." ;;
  esac
}

backup_env() {
  # Timestamped, collision-proof backup before we replace .env.
  local ts base
  ts=$(date +%Y%m%d-%H%M%S 2>/dev/null || printf 'backup')
  base=".env.backup.$ts"
  local dest=$base i=1
  while [ -e "$dest" ]; do dest="$base.$i"; i=$((i + 1)); done
  cp -p .env "$dest" || fail "Could not back up existing .env to $dest."
  say "  Backed up existing .env -> $dest"
}

# ---------------------------------------------------------------------------
# Collect answers and render .env (generate path only).
# ---------------------------------------------------------------------------
configure() {
  require_tty
  step "Configuring Voxint (minimum questions)"

  prompt_password
  prompt_media_root

  step "Checking host ports"
  PG_PORT=$(resolve_port "PostgreSQL" 5432)
  RD_PORT=$(resolve_port "Redis" 6379)
  API_PORT_VALUE=$(resolve_port "API / web console" 8080)
  # Only record an override line when a port actually changed from its default
  # (explicit if/fi: a bare `test && x` would abort the script under `set -e`
  # whenever the test is false).
  if [ "$PG_PORT" = "5432" ];        then PG_PORT=""; fi
  if [ "$RD_PORT" = "6379" ];        then RD_PORT=""; fi
  if [ "$API_PORT_VALUE" = "8080" ]; then API_PORT_VALUE=""; fi

  CSRF_VALUE=$(generate_secret) || fail "Could not generate a random secret from /dev/urandom."

  # Pre-create the media directory so Compose does not create it root-owned.
  mkdir -p -- "$MEDIA_ROOT_VALUE" || fail "Could not create media folder: $MEDIA_ROOT_VALUE"
  # Canonicalize to an absolute path: a bare relative name (e.g. "media") is read
  # by Compose as a *named volume*, not a host bind mount. An absolute path always
  # binds. (pwd resolves ./, ~/, and symlinks.)
  MEDIA_ROOT_VALUE=$(cd -- "$MEDIA_ROOT_VALUE" && pwd) || fail "Could not resolve media folder path."
  case $MEDIA_ROOT_VALUE in *:*) fail "Resolved media path contains ':' and cannot be mounted: $MEDIA_ROOT_VALUE" ;; esac
  if [ "$MEDIA_ROOT_VALUE" = "$REPO_ROOT/.env" ]; then
    fail "The media folder cannot be the .env path."
  fi
  say "  Media folder: $MEDIA_ROOT_VALUE"

  if [ -f .env ]; then backup_env; fi
  write_env
  say "  Wrote .env (mode 0600)."
}

# ---------------------------------------------------------------------------
# Pull + start. Compose is authoritative for port binds and config errors; we
# do not parse its messages or auto-retry with different ports.
# ---------------------------------------------------------------------------
pull_and_start() {
  step "Pulling release images (first run can take a few minutes)"
  docker compose pull || fail "Image pull failed. Check your connection and re-run."

  step "Starting the stack"
  if ! docker compose up -d; then
    say ""
    say "Startup failed. Current services:"
    docker compose ps -a >&2 || true
    say ""
    say "Inspect the one-shot migration and API logs with:"
    say "  docker compose logs migrate"
    say "  docker compose logs api"
    fail "docker compose up did not succeed."
  fi
}

# ---------------------------------------------------------------------------
# Readiness: poll the API container's own healthcheck (no curl dependency).
# The one-shot 'migrate' service runs first; api only starts once it completes,
# so a generous window covers migrate + API warmup.
# ---------------------------------------------------------------------------
wait_for_health() {
  step "Waiting for the API to become healthy"
  local deadline=$((SECONDS + 180)) cid raw lifecycle health mid mstate mcode
  while [ "$SECONDS" -lt "$deadline" ]; do
    # -aq so a container that has already exited is still visible (a running-only
    # query would make a crash-on-start look like a hang until the timeout).
    cid=$(docker compose ps -aq api 2>/dev/null || true)
    if [ -n "$cid" ]; then
      # Inspect lifecycle AND health separately: a container can exit while its
      # health still reads 'starting', which the health field alone would hide.
      raw=$(docker inspect --format '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid" 2>/dev/null || true)
      lifecycle=${raw%%|*}
      health=${raw#*|}
      case $lifecycle in
        exited|dead)
          say ""
          say "The API container is '$lifecycle'. Recent logs:"
          docker compose logs --tail 40 migrate api >&2 2>/dev/null || true
          fail "The API did not come up cleanly."
          ;;
      esac
      case $health in
        healthy)
          say "  API is healthy."
          return 0
          ;;
        unhealthy)
          say ""
          say "The API container is 'unhealthy'. Recent logs:"
          docker compose logs --tail 40 api >&2 2>/dev/null || true
          fail "The API did not come up cleanly."
          ;;
      esac
    else
      # No API container yet -- if the one-shot 'migrate' already failed, the API
      # will never start, so surface that now instead of waiting out the timeout.
      mid=$(docker compose ps -aq migrate 2>/dev/null || true)
      if [ -n "$mid" ]; then
        mstate=$(docker inspect --format '{{.State.Status}}|{{.State.ExitCode}}' "$mid" 2>/dev/null || true)
        mcode=${mstate#*|}
        case ${mstate%%|*} in
          exited|dead)
            if [ "$mcode" != "0" ]; then
              say ""
              say "The one-shot 'migrate' service failed (exit $mcode). Recent logs:"
              docker compose logs --tail 40 migrate >&2 2>/dev/null || true
              fail "Database migration failed; the API cannot start."
            fi
            ;;
        esac
      fi
    fi
    sleep 2
  done

  say ""
  say "Timed out after 180s waiting for the API to report healthy. It may still be"
  say "starting (a slow first pull/migrate). Check status and logs with:"
  say "  docker compose ps -a"
  say "  docker compose logs migrate api"
  fail "API health check timed out."
}

# ---------------------------------------------------------------------------
# Handoff. Print the URL and username (never the password), explain the
# migrate one-shot, and state the core-only limitation honestly.
# ---------------------------------------------------------------------------
print_handoff() {
  local bind url port
  bind=$(docker compose port api 8080 2>/dev/null || true)
  case $bind in
    *:*) url="http://$bind/" ;;
    *)
      # Fallback if `docker compose port` fails: read the configured API_PORT from
      # .env rather than assuming the default (which would be wrong on an override).
      port=$(grep -m1 '^API_PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d "'\" ")
      [ -n "$port" ] || port=8080
      url="http://127.0.0.1:$port/"
      ;;
  esac

  printf '\n' >&2
  say "======================================================================"
  say " Voxint core stack is up."
  say ""
  say "   Console:  $url"
  say "   Sign in:  username 'admin' (or the VOXINT_USER you set) + the"
  say "             password you just chose."
  say ""
  say " Open the console, browse runs at ${url}runs, and adjudicate at ${url}review."
  say ""
  say " Note: the one-shot 'migrate' service showing 'Exited (0)' in"
  say " 'docker compose ps -a' is SUCCESS (it applied the schema and stopped),"
  say " not a crash."
  say ""
  say " IMPORTANT -- this is the CORE control plane only. You can open the"
  say " console and adjudicate, but transcription, diarization, and speaker"
  say " embedding need the GPU model services (or external model endpoints)."
  say " Submitting audio now will fail at those stages. To enable processing,"
  say " bring up the GPU overlay (needs an NVIDIA GPU + a Hugging Face token):"
  say "   docker compose -f compose.yaml -f compose.gpu.yaml up -d"
  say " See the Quickstart in README.md and docs/operations.md."
  say ""
  say " Stop the stack with:  docker compose down"
  say "======================================================================"
}

# ---------------------------------------------------------------------------
main() {
  # Armed here (not at global scope) so sourcing in library mode leaves the
  # caller's traps untouched. Removes the temp .env on Ctrl+C / kill / hangup.
  trap '_cleanup; exit 130' INT
  trap '_cleanup; exit 143' TERM HUP
  trap _cleanup EXIT

  say "Voxint installer"
  preflight
  decide_existing_env
  if [ "$ENV_ACTION" = "generate" ]; then
    configure
  else
    say "Keeping the existing .env."
  fi
  pull_and_start
  wait_for_health
  print_handoff
}

# Sourcing with VOXINT_INSTALL_LIB=1 loads the functions without running the
# installer -- used by the test harness to exercise pure-shell logic (.env
# rendering, port/version parsing) offline.
if [ "${VOXINT_INSTALL_LIB:-}" != "1" ]; then
  main "$@"
fi
