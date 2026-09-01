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
# console, the Celery worker and beat) plus, if you choose one, a compute tier
# for the model services -- GPU (compose.gpu.yaml, NVIDIA), ROCm
# (compose.rocm.yaml, AMD GPU for ASR), CPU (compose.cpu.yaml, runs
# anywhere), or Metal (compose.metal.yaml, Apple Silicon: core in Docker,
# model services native via scripts/metal/voxint-metal.sh so diarization can
# use the Apple GPU). Transcription, diarization, and speaker embedding need
# a tier; all model weights (diarization included) are vendored into the
# images -- and sha-verified downloads for the native metal services -- so no
# Hugging Face account or token is involved.
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
# No `--` for dirname: it is a GNU-ism some BSD dirnames reject, and
# SCRIPT_SELF derives from BASH_SOURCE so it cannot begin with a dash.
SCRIPT_DIR=$(CDPATH= cd -P -- "$(dirname "$SCRIPT_SELF")" && pwd)
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
unset VOXINT_PASSWORD MEDIA_ROOT CSRF_SECRET POSTGRES_PORT REDIS_PORT API_PORT HF_TOKEN VOXINT_COMPOSE_TIER VOXINT_RENDER_GID \
  WHISPER_MODEL WHISPER_REVISION WHISPER_ALLOW_DOWNLOAD DIARIZER_MODEL_NAME DIARIZER_REVISION 2>/dev/null || true

# ---------------------------------------------------------------------------
# Compute-tier state. COMPUTE_TIER_VALUE is what the user chose
# (cpu|gpu|rocm|metal|none, persisted in .env as VOXINT_COMPOSE_TIER).
# EFFECTIVE_TIER is what this run actually starts (same as the choice — the
# model weights are vendored into the images, nothing to defer on). The
# metal tier starts core-only in Docker: its model services run natively,
# set up AFTER this installer by scripts/metal/voxint-metal.sh (the handoff
# says so explicitly).
# ---------------------------------------------------------------------------
COMPUTE_TIER_VALUE=""
EFFECTIVE_TIER="none"
RENDER_GID_VALUE=""
COMPOSE_FILE_ARGS="-f compose.yaml"
# Optional alternate-model overrides (advanced entry; empty = write nothing, keep
# the validated defaults). Collected by prompt_advanced_models, written verbatim.
WHISPER_MODEL_VALUE=""
WHISPER_REVISION_VALUE=""
WHISPER_ALLOW_DOWNLOAD_VALUE=""
DIARIZER_MODEL_NAME_VALUE=""
DIARIZER_REVISION_VALUE=""
HF_TOKEN_VALUE=""

# Installer-managed hardware defaults (issue #96). A generated, marker-headed
# compose.hardware.yaml carries conservative single-GPU sizing; it is wired into
# the compose chain only for the GPU tier and only when WE generated it. The
# operator's own overrides belong in compose.override.yaml (honored separately).
HARDWARE_OVERRIDE_FILE="compose.hardware.yaml"
HARDWARE_OVERRIDE_MARKER="# voxint:hardware-override"  # first-line ownership marker

# Base Celery worker command, kept in ONE place. The generated hardware override
# must restate the whole command (Compose replaces command: wholesale, never
# merges it), so a unit test pins this to compose.yaml's worker command: if the
# base ever changes, that test fails and forces this to move in lockstep.
WORKER_BASE_COMMAND="celery -A voxint.worker.app worker --loglevel=INFO"

# True only when compose.hardware.yaml is a regular file whose first line is our
# ownership marker -- i.e. the installer generated it. A prefix match keeps a
# future format-version bump (…-override v2) still recognizable as managed.
hardware_override_is_managed() {
  # A symlink is never installer-managed: we never create one, and following it
  # would let a link decide what gets folded into the compose chain.
  [ -L "$HARDWARE_OVERRIDE_FILE" ] && return 1
  [ -f "$HARDWARE_OVERRIDE_FILE" ] || return 1
  # Initialized so an unreadable-but-present file (e.g. root-owned mode 600 from a
  # prior sudo run) degrades to "unmanaged" instead of tripping set -u on $first.
  local first=""
  IFS= read -r first < "$HARDWARE_OVERRIDE_FILE" 2>/dev/null || true
  case $first in "$HARDWARE_OVERRIDE_MARKER"*) return 0 ;; *) return 1 ;; esac
}

# ROCm tier: the gid owning /dev/kfd + /dev/dri/renderD* is allocated per
# host, so the rocm overlay interpolates it from .env (VOXINT_RENDER_GID).
# Prefer the ACTUAL owner of /dev/kfd (ground truth) over the "render" group
# name; fall back to getent when the device is absent (e.g. pre-driver
# install). Empty when neither works — the overlay then falls back to its
# default and this NOTE tells the user what to set.
detect_render_gid() {
  RENDER_GID_VALUE=""
  if [ -e /dev/kfd ]; then
    RENDER_GID_VALUE=$(stat -c %g /dev/kfd 2>/dev/null || true)
  fi
  if [ -z "$RENDER_GID_VALUE" ]; then
    RENDER_GID_VALUE=$(getent group render 2>/dev/null | cut -d: -f3)
  fi
  case $RENDER_GID_VALUE in *[!0-9]*) RENDER_GID_VALUE="" ;; esac
  if [ -z "$RENDER_GID_VALUE" ]; then
    say "  NOTE: could not detect the gid owning /dev/kfd; if the whisper"
    say "  service cannot open the GPU, set VOXINT_RENDER_GID in .env to the"
    say "  owning group (stat -c %g /dev/kfd)."
  fi
}

normalize_tier() {
  case ${1:-} in cpu|gpu|rocm|metal|none) printf '%s' "$1" ;; *) printf '%s' "" ;; esac
}

# One helper owns the tier -> Compose-file mapping, and EVERY Compose
# invocation goes through dc(), so pull/up/ps/logs/port can never disagree
# about which overlay is active.
compose_file_args_for_tier() {
  local tier args skip_hardware
  tier=$(normalize_tier "${1:-}")
  # $2 == "skip-hardware" omits the managed compose.hardware.yaml even on the GPU
  # tier -- used when validating a kept .env, so a stale managed file cannot
  # masquerade as a bad .env before configure_hardware_defaults regenerates it.
  skip_hardware=${2:-}
  case $tier in
    cpu)  args='-f compose.yaml -f compose.cpu.yaml' ;;
    gpu)  args='-f compose.yaml -f compose.gpu.yaml' ;;
    rocm) args='-f compose.yaml -f compose.rocm.yaml' ;;
    metal) args='-f compose.yaml -f compose.metal.yaml' ;;
    *)   args='-f compose.yaml' ;;
  esac
  # Conservative GPU/ROCm sizing (issue #96): fold in the installer-generated
  # compose.hardware.yaml, after the tier overlay so it wins. GPU and ROCm tiers
  # only, and only when we generated it -- an unmarked hand-written file is left
  # alone and never auto-loaded here.
  if { [ "$tier" = "gpu" ] || [ "$tier" = "rocm" ]; } && [ "$skip_hardware" != "skip-hardware" ] && hardware_override_is_managed; then
    args="$args -f $HARDWARE_OVERRIDE_FILE"
  fi
  # An operator's own compose.override.yaml is NOT auto-merged when explicit -f
  # args are passed (as we always do), so honor it explicitly and LAST, so it
  # wins over base, the tier overlay, and the hardware defaults -- matching what
  # a bare `docker compose up` would merge.
  if [ -f compose.override.yaml ]; then
    args="$args -f compose.override.yaml"
  fi
  printf '%s' "$args"
}

dc() {
  # shellcheck disable=SC2086  # intentional word-splitting of the -f arguments
  docker compose $COMPOSE_FILE_ARGS "$@"
}

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

# Advisory only: true if something on 127.0.0.1:$1 accepts a TCP connection
# OR leaves it hanging (a listener with a full accept queue -- macOS drops the
# SYN silently instead of refusing, so a hung service would otherwise read as
# free; the bounded watchdog below treats a still-pending connect as in-use).
# Uses bash /dev/tcp (no netcat/lsof dependency). Compose remains the
# authority -- this just lets us ask about a collision before we hit it, and
# every probe has a TOCTOU race before Compose actually binds. resolve_port
# compensates by only ever OFFERING ports strictly above a known-busy default.
port_in_use() {
  case $1 in ''|*[!0-9]*) return 1 ;; esac
  # The probe runs in a background subshell so a SYN-drop cannot hang the
  # installer: refused connections fail in milliseconds, so anything still
  # pending after ~2s is a wedged/backlogged listener -- treat it as in use.
  # The probe fd is opened INSIDE the subshell and dies with it -- nothing to
  # close here. (A previous `exec 3>&- 2>/dev/null` cleanup line was a bug: a
  # bare `exec` with redirections rebinds the CURRENT shell's stderr, so after
  # the first detected collision every later prompt went to /dev/null.)
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
    "~"*)  fail "Media path '~user/...' forms are not expanded here -- type the absolute path instead." ;;
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
  # $def is known busy here, so search STRICTLY ABOVE it. Starting the scan at
  # $def would let a macOS/BSD backlog-full misread (see port_in_use) return the
  # busy port right back as the "alternate". +1 guarantees a distinct suggestion.
  if [ "$def" -ge 65535 ]; then
    # No port above 65535 exists to offer. Leave the suggestion EMPTY rather than
    # defaulting the prompt to the known-busy $def (which would violate the
    # "offered alternate never equals the busy port" invariant); the loop below
    # then requires the operator to type a valid free port.
    suggested=
  else
    suggested=$(next_free_port "$((def + 1))")
  fi
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

# Which compute tier should run the model services. Uses the GPU inventory's
# VRAM-aware recommendation when available, with presence-based detection as a
# backward-compatible fallback (advisory only -- the user decides).
prompt_compute_tier() {
  local def
  # Use inventory-based recommendation when available
  if [ -s "${GPU_INVENTORY_FILE:-}" ]; then
    recommend_compute_tier
    def=$RECOMMENDED_TIER
  else
    # Fallback: presence-based detection (no inventory created yet)
    def=c
    if [ -e /dev/kfd ]; then def=a; fi
    if command -v "${VOXINT_NVIDIA_SMI:-nvidia-smi}" >/dev/null 2>&1; then def=g; fi
  fi
  # Checked LAST so it wins on a Mac even if a stray nvidia-smi is on PATH:
  # Apple Silicon cannot run the CUDA containers, and the metal tier is what
  # actually uses the Apple GPU.
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then def=m; fi

  # Map tier names to prompt letters
  case $def in gpu) def=g ;; rocm) def=a ;; cpu) def=c ;; metal) def=m ;; esac

  step "Choosing a compute tier for the model services"

  # Show GPU inventory if available
  if [ -s "${GPU_INVENTORY_FILE:-}" ]; then
    print_gpu_inventory "$GPU_INVENTORY_FILE"
    if [ -n "${RECOMMENDATION_REASON:-}" ]; then
      case $def in
        g) say "  Recommending GPU tier: $RECOMMENDATION_REASON" ;;
        a) say "  Recommending AMD tier: $RECOMMENDATION_REASON" ;;
        c) say "  Recommending CPU tier: $RECOMMENDATION_REASON" ;;
      esac
    fi
    say ""
  fi

  say "  Transcription, diarization, and speaker embedding run as model services."
  say "  Pick how they should run:"
  say "    [G] GPU tier  -- needs an NVIDIA GPU + driver (fastest; ~6-8 GB VRAM)"
  say "    [A] AMD tier  -- needs an AMD GPU (amdgpu driver only; transcription"
  say "                     runs on the GPU, diarization/embedding on CPU)"
  say "    [C] CPU tier  -- no GPU needed; works on any amd64/arm64 host"
  say "                     (much slower: long recordings take hours, not minutes;"
  say "                      needs >=8 GB RAM for the container host / Docker VM)"
  say "    [M] Apple tier -- native model services on this Mac (Apple Silicon):"
  say "                     diarization uses the Apple GPU; needs a separate"
  say "                     native setup step AFTER this installer (uv, ~3.2 GB)"
  say "    [N] None for now -- core console only; audio processing disabled"
  local label ans
  case $def in
    g) label='[G/a/c/m/n]' ;;
    a) label='[A/g/c/m/n]' ;;
    m) label='[M/g/a/c/n]' ;;
    *) label='[C/g/a/m/n]' ;;
  esac
  while :; do
    printf 'Compute tier %s: ' "$label" >&2
    IFS= read -r ans || fail "No input."
    ans=${ans:-$def}
    case $ans in
      g|G|gpu|GPU)          COMPUTE_TIER_VALUE=gpu;  return 0 ;;
      a|A|amd|AMD|rocm|ROCm|ROCM) COMPUTE_TIER_VALUE=rocm; detect_render_gid; return 0 ;;
      c|C|cpu|CPU)          COMPUTE_TIER_VALUE=cpu;  return 0 ;;
      m|M|metal|Metal|METAL) COMPUTE_TIER_VALUE=metal; return 0 ;;
      n|N|none|None|NONE)   COMPUTE_TIER_VALUE=none; return 0 ;;
    esac
    say "  Please answer g, a, c, m, or n."
  done
}

# Refuse characters that would break dotenv / Compose parsing of an unquoted
# value, so a typo in an advanced model field surfaces here with a clear message
# instead of as an opaque "Compose validation failed" after the fact.
_reject_unsafe_env_value() {
  # $1 = human label, $2 = value.
  case $2 in
    *[[:space:]]*) fail "$1 must not contain spaces: '$2'." ;;
    *\'*)          fail "$1 must not contain a single-quote (')." ;;
    *'$'*)         fail "$1 must not contain a '\$' character." ;;
    *'`'*)         fail "$1 must not contain a backtick." ;;
  esac
  case $2 in *\\) fail "$1 must not end with a backslash (\\)." ;; esac
}

_is_full_sha() {
  # $1 = candidate; success iff exactly 40 lowercase hex chars (the whisper
  # service's revision gate). Empty or any non-hex character fails. Bash 3.2.
  # Spell out abcdef instead of a-f: on macOS Bash 3.2 the default locale
  # makes [a-f] include uppercase letters, so A-F would pass the check.
  case $1 in
    *[!0-9abcdef]*) return 1 ;;
  esac
  [ "${#1}" -eq 40 ]
}

# ---------------------------------------------------------------------------
# Optional, advanced: point transcription or diarization at an alternate model.
# Skipped by default -- the shipped whisper large-v2 and pyannote
# speaker-diarization-3.1 are the only VALIDATED configs and need no setup, so
# a default install answers 'no' and writes no new keys. Accepting writes the
# operator's raw values verbatim; the model services enforce their own gates at
# startup (a non-default whisper needs an explicit download opt-in plus a
# full-SHA revision), and only large-v2 / pyannote-3.1 carry Voxint's numerics
# guarantees -- whisper v3 and turbo hallucinate. HF_TOKEN is read hidden and
# never echoed. Every field defaults to blank, meaning leave that key unset.
# ---------------------------------------------------------------------------
prompt_advanced_models() {
  local ans model_ans
  step "Alternate pipeline models (advanced, optional)"
  say "  The shipped models -- whisper large-v2 and pyannote speaker-diarization-3.1"
  say "  -- are the only validated ones and need no setup. Most installs skip this."
  say "  You can point transcription or diarization at a different model, but its"
  say "  accuracy is not measured (whisper v3 and turbo are known to hallucinate)."
  printf 'Configure alternate models now? [y/N]: ' >&2
  IFS= read -r ans || ans=""
  case $ans in
    y|Y|yes|Yes) : ;;
    *) return 0 ;;  # default: skip -- write no model keys, keep the defaults
  esac

  # Transcription (whisper).
  printf 'Alternate transcription model (blank = keep validated large-v2): ' >&2
  IFS= read -r ans || ans=""
  if [ -n "$ans" ]; then
    _reject_unsafe_env_value "Transcription model" "$ans"
    model_ans=$ans
    say "  A non-default model must be downloaded once and pinned to an exact commit."
    printf 'Transcription model revision (full 40-character lowercase commit SHA): ' >&2
    IFS= read -r ans || ans=""
    _reject_unsafe_env_value "Transcription revision" "$ans"
    if _is_full_sha "$ans"; then
      WHISPER_MODEL_VALUE=$model_ans
      WHISPER_REVISION_VALUE=$ans
      # The service refuses to start without BOTH a full-SHA revision (validated
      # above) and this download opt-in, so record it only once the SHA is valid.
      # This never writes a config the whisper container would reject at startup.
      WHISPER_ALLOW_DOWNLOAD_VALUE=1
      say "  Recorded WHISPER_ALLOW_DOWNLOAD=1 to permit the one-time download."
    else
      say "  That is not a full 40-character lowercase commit SHA. Keeping the"
      say "  validated large-v2 instead; re-run the installer to set an alternate"
      say "  model with its exact revision."
    fi
  fi

  # Diarization (pyannote).
  printf 'Alternate diarization model (blank = keep validated speaker-diarization-3.1): ' >&2
  IFS= read -r ans || ans=""
  if [ -n "$ans" ]; then
    _reject_unsafe_env_value "Diarization model" "$ans"
    DIARIZER_MODEL_NAME_VALUE=$ans
    printf 'Diarization model revision (optional commit pin; blank = repo default): ' >&2
    IFS= read -r ans || ans=""
    if [ -n "$ans" ]; then
      _reject_unsafe_env_value "Diarization revision" "$ans"
      DIARIZER_REVISION_VALUE=$ans
    fi
    # A gated Hugging Face pipeline needs a token: read hidden, never echoed, and
    # single-quoted when written so an odd character cannot break dotenv.
    printf 'Hugging Face token for a gated diarization model (optional; hidden): ' >&2
    IFS= read -r -s ans || ans=""
    printf '\n' >&2
    if [ -n "$ans" ]; then
      # dotenv_squote (used to write it) requires no embedded single-quote or
      # trailing backslash; enforce that here rather than emit a malformed .env.
      _reject_unsafe_env_value "Hugging Face token" "$ans"
      HF_TOKEN_VALUE=$ans
    fi
  fi
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
    VOXINT_COMPOSE_TIER) if [ -n "${COMPUTE_TIER_VALUE:-}" ]; then printf 'VOXINT_COMPOSE_TIER=%s' "$COMPUTE_TIER_VALUE"; fi ;;
    VOXINT_RENDER_GID) if [ -n "${RENDER_GID_VALUE:-}" ]; then printf 'VOXINT_RENDER_GID=%s' "$RENDER_GID_VALUE"; fi ;;
    # Advanced alternate-model overrides: written verbatim, and only when the
    # operator opted in and supplied a value. HF_TOKEN is single-quoted like the
    # password so a token with shell metacharacters cannot break dotenv parsing.
    WHISPER_MODEL) if [ -n "${WHISPER_MODEL_VALUE:-}" ]; then printf 'WHISPER_MODEL=%s' "$WHISPER_MODEL_VALUE"; fi ;;
    WHISPER_REVISION) if [ -n "${WHISPER_REVISION_VALUE:-}" ]; then printf 'WHISPER_REVISION=%s' "$WHISPER_REVISION_VALUE"; fi ;;
    WHISPER_ALLOW_DOWNLOAD) if [ -n "${WHISPER_ALLOW_DOWNLOAD_VALUE:-}" ]; then printf 'WHISPER_ALLOW_DOWNLOAD=%s' "$WHISPER_ALLOW_DOWNLOAD_VALUE"; fi ;;
    DIARIZER_MODEL_NAME) if [ -n "${DIARIZER_MODEL_NAME_VALUE:-}" ]; then printf 'DIARIZER_MODEL_NAME=%s' "$DIARIZER_MODEL_NAME_VALUE"; fi ;;
    DIARIZER_REVISION) if [ -n "${DIARIZER_REVISION_VALUE:-}" ]; then printf 'DIARIZER_REVISION=%s' "$DIARIZER_REVISION_VALUE"; fi ;;
    HF_TOKEN) if [ -n "${HF_TOKEN_VALUE:-}" ]; then printf 'HF_TOKEN=%s' "$(dotenv_squote "$HF_TOKEN_VALUE")"; fi ;;
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
  if [ -n "${COMPUTE_TIER_VALUE:-}" ]; then printf '%s\n' VOXINT_COMPOSE_TIER; fi
  if [ -n "${RENDER_GID_VALUE:-}" ];   then printf '%s\n' VOXINT_RENDER_GID; fi
  if [ -n "${WHISPER_MODEL_VALUE:-}" ];          then printf '%s\n' WHISPER_MODEL; fi
  if [ -n "${WHISPER_REVISION_VALUE:-}" ];       then printf '%s\n' WHISPER_REVISION; fi
  if [ -n "${WHISPER_ALLOW_DOWNLOAD_VALUE:-}" ]; then printf '%s\n' WHISPER_ALLOW_DOWNLOAD; fi
  if [ -n "${DIARIZER_MODEL_NAME_VALUE:-}" ];    then printf '%s\n' DIARIZER_MODEL_NAME; fi
  if [ -n "${DIARIZER_REVISION_VALUE:-}" ];      then printf '%s\n' DIARIZER_REVISION; fi
  if [ -n "${HF_TOKEN_VALUE:-}" ];               then printf '%s\n' HF_TOKEN; fi
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
  # Validated against the EFFECTIVE file set, so a tier overlay that is about
  # to be started is proven interpolable before we commit.
  if ! dc --env-file "$tmp" config --quiet >/dev/null 2>&1; then
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

# Read one key's value from the kept .env without sourcing it. Echoes the
# value of the LAST uncommented KEY= line (Compose dotenv semantics: last one
# wins), normalized: a trailing CR (CRLF-edited files), surrounding blanks,
# and one MATCHED pair of single or double quotes are stripped -- so a
# hand-edited HF_TOKEN="" reads as empty and HF_TOKEN="hf_x" as hf_x, matching
# what Compose interpolation sees. (export-prefixed or indented lines are not
# matched; installer-managed files never contain them.)
read_env_value() {
  local raw
  raw=$(grep -E "^${1}=" .env 2>/dev/null | tail -n1 | cut -d= -f2-) || raw=""
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

# Minimal atomic editor for a KEPT .env: replaces (or appends) ONLY
# VOXINT_COMPOSE_TIER, plus VOXINT_RENDER_GID when the rocm tier detected one.
# Backs up first, validates with Compose, and preserves mode 0600. Everything
# else in the file (an optional HF_TOKEN included) passes through
# byte-for-byte.
update_env_keys() {
  backup_env
  local tmp line wrote_tier=0 wrote_gid=0
  local old_umask; old_umask=$(umask); umask 077
  tmp=$(mktemp "$REPO_ROOT/.env.tmp.XXXXXX") || { umask "$old_umask"; fail "Could not create a temp file in $REPO_ROOT."; }
  _CLEANUP_TMP=$tmp
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in
      VOXINT_COMPOSE_TIER=*)
        # Rewrite the first occurrence; drop later duplicates (Compose is
        # last-wins, so keeping them would leave a stale-looking earlier line).
        if [ "$wrote_tier" = 0 ]; then
          printf 'VOXINT_COMPOSE_TIER=%s\n' "$COMPUTE_TIER_VALUE" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
          wrote_tier=1
        fi
        ;;
      VOXINT_RENDER_GID=*)
        if [ -n "$RENDER_GID_VALUE" ]; then
          if [ "$wrote_gid" = 0 ]; then
            printf 'VOXINT_RENDER_GID=%s\n' "$RENDER_GID_VALUE" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
            wrote_gid=1
          fi
        else
          printf '%s\n' "$line" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
        fi
        ;;
      *)
        printf '%s\n' "$line" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
        ;;
    esac
  done < .env
  if [ "$wrote_tier" = 0 ]; then
    printf 'VOXINT_COMPOSE_TIER=%s\n' "$COMPUTE_TIER_VALUE" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
  fi
  if [ -n "$RENDER_GID_VALUE" ] && [ "$wrote_gid" = 0 ]; then
    printf 'VOXINT_RENDER_GID=%s\n' "$RENDER_GID_VALUE" >>"$tmp" || { umask "$old_umask"; fail "Failed writing candidate .env."; }
  fi
  umask "$old_umask"
  # Validate with the file set this .env will actually run: the recorded
  # tier's overlay (startable unconditionally — nothing gates on a token).
  # Mirrors write_env.
  local vargs
  vargs=$(compose_file_args_for_tier "$COMPUTE_TIER_VALUE")
  # shellcheck disable=SC2086  # intentional word-splitting of the -f arguments
  if ! docker compose $vargs --env-file "$tmp" config --quiet >/dev/null 2>&1; then
    fail "Updated .env failed Compose validation. This is a bug -- please report it."
  fi
  mv -f "$tmp" .env || fail "Could not move the updated .env into place."
  _CLEANUP_TMP=""
  chmod 600 .env 2>/dev/null || true
  say "  Updated .env (recorded compute tier)."
}

# Kept-.env path: honor a recorded VOXINT_COMPOSE_TIER; on a legacy .env
# (pre-0.4.1, no tier key -- or an unrecognized value) ask once and record it.
resolve_kept_env_tier() {
  local tier
  tier=$(normalize_tier "$(read_env_value VOXINT_COMPOSE_TIER)")
  if [ -z "$tier" ]; then
    require_tty
    say "  This .env predates tier selection (no VOXINT_COMPOSE_TIER) -- one question:"
    prompt_compute_tier
    tier=$COMPUTE_TIER_VALUE
    update_env_keys
  fi
  COMPUTE_TIER_VALUE=$tier
  # Kept rocm tier: re-detect the render gid every run — the recorded value
  # goes stale when the checkout moves hosts (the gid is per-host), and a
  # hand-set VOXINT_COMPOSE_TIER=rocm never went through the prompt at all.
  if [ "$tier" = "rocm" ]; then
    local recorded_gid
    recorded_gid=$(read_env_value VOXINT_RENDER_GID)
    detect_render_gid
    if [ -n "$RENDER_GID_VALUE" ] && [ "$RENDER_GID_VALUE" != "$recorded_gid" ]; then
      if [ -n "$recorded_gid" ]; then
        say "  Recorded VOXINT_RENDER_GID ($recorded_gid) does not match this host"
        say "  ($RENDER_GID_VALUE) -- updating .env."
      fi
      update_env_keys
    fi
  fi
  EFFECTIVE_TIER=$tier
  COMPOSE_FILE_ARGS=$(compose_file_args_for_tier "$EFFECTIVE_TIER")
  # Validate the kept .env against the file set that will actually start, so a
  # hand-edited .env fails here with a clear message instead of mid-`up`. Exclude
  # any installer-managed compose.hardware.yaml from THIS check: a stale managed
  # file (e.g. left by an older voxint whose compose changed) would otherwise fail
  # validation here and be reported as a bad .env, blocking the very regeneration
  # that configure_hardware_defaults performs (and re-validates) right after.
  local validate_args
  validate_args=$(compose_file_args_for_tier "$EFFECTIVE_TIER" skip-hardware)
  # shellcheck disable=SC2086  # intentional word-splitting of the -f arguments
  if ! docker compose $validate_args config --quiet >/dev/null 2>&1; then
    fail "The existing .env failed Compose validation for the recorded tier ($tier). Fix it or re-run and choose to regenerate."
  fi
}

backup_env() {
  # Timestamped, collision-proof backup before we replace .env.
  local ts base
  ts=$(date +%Y%m%d-%H%M%S 2>/dev/null || printf 'backup')
  base=".env.backup.$ts"
  local dest=$base i=1
  while [ -e "$dest" ]; do dest="$base.$i"; i=$((i + 1)); done
  cp -p .env "$dest" || fail "Could not back up existing .env to $dest."
  # cp -p copies the SOURCE mode -- a hand-created 0644 .env would yield a
  # world-readable backup full of secrets. Force 0600 regardless.
  chmod 600 "$dest" 2>/dev/null || true
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
  create_gpu_inventory
  prompt_compute_tier
  # GPU/ROCm: select a specific device and confirm VRAM budget
  case $COMPUTE_TIER_VALUE in
    gpu|rocm)
      prompt_gpu_device "$COMPUTE_TIER_VALUE"
      if ! confirm_gpu_budget "$COMPUTE_TIER_VALUE"; then
        say "  Falling back to CPU tier."
        COMPUTE_TIER_VALUE=cpu
      fi
      ;;
  esac
  cleanup_gpu_inventory
  EFFECTIVE_TIER=$COMPUTE_TIER_VALUE
  COMPOSE_FILE_ARGS=$(compose_file_args_for_tier "$EFFECTIVE_TIER")

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

  prompt_advanced_models

  if [ -f .env ]; then backup_env; fi
  write_env
  say "  Wrote .env (mode 0600)."
}

# ---------------------------------------------------------------------------
# Hardware-aware conservative defaults (issue #96). The HOST can see the GPU
# (via nvidia-smi); the app container cannot. Detection is PROFILE MATCHING on
# GPU identity, never a VRAM formula. Only the conservative unknown fallback
# ships today: it caps worker concurrency and whisper's pending queue and NEVER
# touches BATCH_SIZE (which feeds whisper's decode_config_hash and can move
# transcription outputs -- any auto BATCH_SIZE must come from a tests/parity-
# passed profile + a real-GPU OOM soak). All functions here are pure/offline so
# the library-mode test seam can exercise them with a fake nvidia-smi.
# ---------------------------------------------------------------------------
GPU_NAME=""
GPU_VRAM_MIB=""
GPU_SIGNATURE=""

# ---------------------------------------------------------------------------
# GPU inventory: vendor-neutral enumeration of all accelerators on the host.
# The inventory is a cleanup-managed temporary TSV file with one record per
# physical GPU. Foundation for VRAM-aware tier suggestion (Phase 2) and
# per-device compose targeting (Phase 3).
# ---------------------------------------------------------------------------

# VRAM budget thresholds for tier recommendations. Advisory only -- BATCH_SIZE
# is NEVER derived from these (numerics doctrine: only parity-passed profiles
# touch BATCH_SIZE).
GPU_BUDGET_RECOMMENDED_MIB=8192    # NVIDIA full suite: comfortable headroom
GPU_BUDGET_MINIMUM_MIB=6144        # NVIDIA full suite: tight but workable
ROCM_BUDGET_RECOMMENDED_MIB=14336  # ROCm whisper: measured 13.06 GiB peak
ROCM_BUDGET_MINIMUM_MIB=6144       # ROCm whisper: lower batch or small model

# Inventory file: tab-separated, one record per physical GPU.
# Fields: vendor index stable_id name total_mib used_mib free_mib render_node mem_confidence
#   vendor:         nvidia | amd
#   index:          display-only ordinal (nvidia-smi index or renderD number)
#   stable_id:      NVIDIA UUID (GPU-xxxx) or AMD PCI BDF (targeting identity)
#   name:           human-readable GPU name
#   total_mib:      integer; -1 when unknown
#   used_mib:       integer; -1 when unknown
#   free_mib:       integer; -1 when unknown
#   render_node:    empty for NVIDIA; /dev/dri/renderDN for AMD
#   mem_confidence: measured (nvidia-smi) | sysfs (AMD dGPU) | shared (AMD iGPU)
GPU_INVENTORY_FILE=""

# Selected GPU device for compose targeting (set by prompt_gpu_device).
SEL_GPU_VENDOR=""
SEL_GPU_STABLE_ID=""
SEL_GPU_INDEX=""
SEL_GPU_NAME=""
SEL_GPU_TOTAL_MIB=""
SEL_GPU_FREE_MIB=""
SEL_GPU_RENDER_NODE=""
SEL_GPU_MEM_CONF=""

# Tier recommendation (set by recommend_compute_tier).
RECOMMENDED_TIER=""
RECOMMENDATION_REASON=""

# Create the GPU inventory temp file and populate it from all vendors.
# Sets GPU_INVENTORY_FILE. Caller must call cleanup_gpu_inventory when done.
create_gpu_inventory() {
  GPU_INVENTORY_FILE=$(mktemp "${TMPDIR:-/tmp}/voxint-gpu-inv.XXXXXX")
  enumerate_nvidia_gpus "$GPU_INVENTORY_FILE"
  enumerate_amd_gpus "$GPU_INVENTORY_FILE"
}

cleanup_gpu_inventory() {
  [ -n "${GPU_INVENTORY_FILE:-}" ] && rm -f "$GPU_INVENTORY_FILE"
  GPU_INVENTORY_FILE=""
}

# Portable tab-safe inventory line reader. Sets the nine inventory globals:
# _inv_vendor _inv_idx _inv_stable_id _inv_name _inv_total _inv_used _inv_free
# _inv_render_node _inv_conf
# Usage: _read_inv_line "$line"
_read_inv_line() {
  local _rl
  _rl=$(printf '%s' "$1" | tr '\t' '\034')
  IFS=$(printf '\034') read -r _inv_vendor _inv_idx _inv_stable_id _inv_name \
    _inv_total _inv_used _inv_free _inv_render_node _inv_conf <<EOF
$_rl
EOF
}

# Append one record per NVIDIA GPU to the inventory file.
# Uses VOXINT_NVIDIA_SMI as a test seam (same as detect_nvidia_gpu).
enumerate_nvidia_gpus() {
  local inv_file=$1
  local smi=${VOXINT_NVIDIA_SMI:-nvidia-smi}
  command -v "$smi" >/dev/null 2>&1 || return 0
  local out
  out=$("$smi" --query-gpu=index,uuid,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits 2>/dev/null) || return 0
  [ -n "$out" ] || return 0

  local line idx uuid name total used free
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    # Skip truly blank lines
    local trimmed=${line#"${line%%[![:blank:]]*}"}
    trimmed=${trimmed%"${trimmed##*[![:blank:]]}"}
    [ -n "$trimmed" ] || continue

    # Parse 6 CSV fields: index, uuid, name, memory.total, memory.used, memory.free
    # Each field may have leading/trailing whitespace from nvidia-smi CSV output.
    local rest=$line
    idx=${rest%%,*}; rest=${rest#*,}
    uuid=${rest%%,*}; rest=${rest#*,}
    name=${rest%%,*}; rest=${rest#*,}
    total=${rest%%,*}; rest=${rest#*,}
    used=${rest%%,*}
    free=${rest#*,}

    # Trim whitespace from all fields
    idx=${idx#"${idx%%[![:blank:]]*}"}; idx=${idx%"${idx##*[![:blank:]]}"}
    uuid=${uuid#"${uuid%%[![:blank:]]*}"}; uuid=${uuid%"${uuid##*[![:blank:]]}"}
    name=${name#"${name%%[![:blank:]]*}"}; name=${name%"${name##*[![:blank:]]}"}
    total=${total#"${total%%[![:blank:]]*}"}; total=${total%"${total##*[![:blank:]]}"}
    used=${used#"${used%%[![:blank:]]*}"}; used=${used%"${used##*[![:blank:]]}"}
    free=${free#"${free%%[![:blank:]]*}"}; free=${free%"${free##*[![:blank:]]}"}

    # Skip MIG instances and malformed rows
    case $name in *MIG*|*mig*) continue ;; esac
    case $total in ''|*[!0-9]*) continue ;; esac
    case $used in ''|*[!0-9]*) used=-1 ;; esac
    case $free in ''|*[!0-9]*) free=-1 ;; esac

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "nvidia" "$idx" "$uuid" "$name" "$total" "$used" "$free" "" "measured" \
      >> "$inv_file"
  done < <(printf '%s\n' "$out")
}

# Append one record per AMD GPU to the inventory file.
# Test seams: VOXINT_SYSFS_ROOT (default /sys), VOXINT_DEV_ROOT (default /dev).
enumerate_amd_gpus() {
  local inv_file=$1
  local sysfs_root=${VOXINT_SYSFS_ROOT:-/sys}
  local dev_root=${VOXINT_DEV_ROOT:-/dev}

  # /dev/kfd must exist for ROCm compute
  [ -e "${dev_root}/kfd" ] || return 0

  local render_path dev_path render_name idx
  local vendor total_bytes used_bytes gtt_bytes total_mib used_mib free_mib mem_conf
  local unique_id pci_slot name

  for render_path in "${sysfs_root}"/class/drm/renderD*/device; do
    [ -d "$render_path" ] || continue

    # Filter: only AMD GPUs (vendor 0x1002)
    [ -f "${render_path}/vendor" ] || continue
    vendor=$(cat "${render_path}/vendor" 2>/dev/null) || continue
    case $vendor in *1002*) ;; *) continue ;; esac

    # Extract renderD node name and number
    dev_path=${render_path%/device}
    render_name=${dev_path##*/}
    idx=${render_name#renderD}

    # Stable ID: prefer unique_id (dGPU), fall back to PCI slot
    unique_id=""
    [ -f "${render_path}/unique_id" ] && unique_id=$(cat "${render_path}/unique_id" 2>/dev/null) || true
    if [ -z "$unique_id" ]; then
      pci_slot=""
      [ -f "${render_path}/uevent" ] && \
        pci_slot=$(grep -o 'PCI_SLOT_NAME=[^ ]*' "${render_path}/uevent" 2>/dev/null | cut -d= -f2) || true
      [ -n "$pci_slot" ] && unique_id=$pci_slot || unique_id="renderD${idx}"
    fi

    # GPU name: product_name or marketing_name from sysfs
    name=""
    local name_file
    for name_file in "${render_path}/product_name" "${render_path}/marketing_name"; do
      [ -f "$name_file" ] && name=$(cat "$name_file" 2>/dev/null) && [ -n "$name" ] && break
    done
    [ -n "$name" ] || name="AMD GPU (renderD${idx})"

    # VRAM from sysfs (bytes -> MiB)
    total_mib=-1; used_mib=-1; free_mib=-1; mem_conf="shared"
    if [ -f "${render_path}/mem_info_vram_total" ]; then
      total_bytes=$(cat "${render_path}/mem_info_vram_total" 2>/dev/null) || true
      case $total_bytes in ''|*[!0-9]*) total_bytes="" ;; esac
      if [ -n "$total_bytes" ] && [ "$total_bytes" -gt 0 ] 2>/dev/null; then
        total_mib=$((total_bytes / 1048576))
        mem_conf="sysfs"
        if [ -f "${render_path}/mem_info_vram_used" ]; then
          used_bytes=$(cat "${render_path}/mem_info_vram_used" 2>/dev/null) || true
          case $used_bytes in ''|*[!0-9]*) used_bytes="" ;; esac
          if [ -n "$used_bytes" ]; then
            used_mib=$((used_bytes / 1048576))
            free_mib=$((total_mib - used_mib))
            [ "$free_mib" -lt 0 ] && free_mib=0
          fi
        fi
        # Detect shared memory (iGPU/APU): GTT >> VRAM means system-backed
        if [ -f "${render_path}/mem_info_gtt_total" ]; then
          gtt_bytes=$(cat "${render_path}/mem_info_gtt_total" 2>/dev/null) || true
          case $gtt_bytes in ''|*[!0-9]*) gtt_bytes="" ;; esac
          if [ -n "$gtt_bytes" ]; then
            local gtt_mib=$((gtt_bytes / 1048576))
            if [ "$gtt_mib" -gt "$((total_mib * 2))" ]; then
              mem_conf="shared"
            fi
          fi
        fi
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "amd" "$idx" "$unique_id" "$name" "$total_mib" "$used_mib" "$free_mib" \
      "${dev_root}/dri/${render_name}" "$mem_conf" \
      >> "$inv_file"
  done
}

# Print the GPU inventory as a human-readable table to stderr.
print_gpu_inventory() {
  local inv_file=$1
  [ -s "$inv_file" ] || { say "  No GPUs detected."; return 0; }

  say "  Detected GPUs:"
  local line status free_display
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    # Build status annotation
    if [ "$_inv_free" = "-1" ]; then
      free_display="unknown"
      status="(free VRAM unknown)"
    elif [ "$_inv_free" -eq 0 ] || { [ "$_inv_vendor" = "nvidia" ] && [ "$_inv_free" -lt 256 ]; }; then
      free_display="${_inv_free} MiB free"
      status="(occupied)"
    else
      free_display="${_inv_free} MiB free"
      status=""
    fi

    local total_display
    [ "$_inv_total" = "-1" ] && total_display="unknown" || total_display="${_inv_total} MiB"

    if [ "$_inv_vendor" = "nvidia" ]; then
      say "    [nv:${_inv_idx}] ${_inv_name}: ${total_display} total, ${free_display} ${status}"
    else
      local conf_note=""
      [ "$_inv_conf" = "shared" ] && conf_note=" (shared system memory)"
      say "    [amd:${_inv_idx}] ${_inv_name}: ${total_display} total, ${free_display}${conf_note} ${status}"
    fi
  done < "$inv_file"
}

# Classify a GPU's VRAM against budget thresholds for a given tier.
# Usage: classify_gpu FREE_MIB TOTAL_MIB MEM_CONFIDENCE TIER
# Prints one of: recommended, constrained, insufficient, unknown
classify_gpu() {
  local free_mib=$1 total_mib=$2 confidence=$3 tier=$4
  local budget_rec budget_min

  case $tier in
    gpu)
      budget_rec=$GPU_BUDGET_RECOMMENDED_MIB
      budget_min=$GPU_BUDGET_MINIMUM_MIB
      ;;
    rocm)
      budget_rec=$ROCM_BUDGET_RECOMMENDED_MIB
      budget_min=$ROCM_BUDGET_MINIMUM_MIB
      ;;
    *) printf 'unknown'; return 0 ;;
  esac

  # Unknown free VRAM: fall back to total for a rough estimate
  if [ "$free_mib" = "-1" ]; then
    if [ "$total_mib" = "-1" ]; then
      printf 'unknown'; return 0
    fi
    # Use total as the upper bound; actual free could be anything
    if [ "$total_mib" -ge "$budget_rec" ]; then
      printf 'unknown'; return 0
    elif [ "$total_mib" -ge "$budget_min" ]; then
      printf 'unknown'; return 0
    else
      printf 'insufficient'; return 0
    fi
  fi

  if [ "$free_mib" -ge "$budget_rec" ]; then
    printf 'recommended'
  elif [ "$free_mib" -ge "$budget_min" ]; then
    printf 'constrained'
  else
    printf 'insufficient'
  fi
}

# Analyze the GPU inventory and pick the best tier. Sets RECOMMENDED_TIER and
# RECOMMENDATION_REASON. Apple Silicon override is handled by the caller.
recommend_compute_tier() {
  RECOMMENDED_TIER="cpu"
  RECOMMENDATION_REASON="no eligible GPU detected"

  [ -s "$GPU_INVENTORY_FILE" ] || return 0

  local line
  local best_nvidia_free=-1 best_nvidia_name="" best_nvidia_class="" best_nvidia_class_rank=-1
  local best_amd_free=-1 best_amd_name="" best_amd_class="" best_amd_total=-1 best_amd_class_rank=-1

  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    if [ "$_inv_vendor" = "nvidia" ]; then
      local class class_rank=0 effective_free=$_inv_free
      class=$(classify_gpu "$_inv_free" "$_inv_total" "$_inv_conf" "gpu")
      [ "$effective_free" = "-1" ] && effective_free=$_inv_total
      case $class in recommended) class_rank=3 ;; constrained) class_rank=2 ;; unknown) class_rank=1 ;; esac
      if [ "$class_rank" -gt "$best_nvidia_class_rank" ] || \
         { [ "$class_rank" -eq "$best_nvidia_class_rank" ] && [ "$effective_free" -gt "$best_nvidia_free" ] 2>/dev/null; }; then
        best_nvidia_free=$effective_free
        best_nvidia_name=$_inv_name
        best_nvidia_class=$class
        best_nvidia_class_rank=$class_rank
      fi
    elif [ "$_inv_vendor" = "amd" ]; then
      local class class_rank=0 effective_free=$_inv_free
      class=$(classify_gpu "$_inv_free" "$_inv_total" "$_inv_conf" "rocm")
      [ "$effective_free" = "-1" ] && effective_free=$_inv_total
      case $class in recommended) class_rank=3 ;; constrained) class_rank=2 ;; unknown) class_rank=1 ;; esac
      if [ "$class_rank" -gt "$best_amd_class_rank" ] || \
         { [ "$class_rank" -eq "$best_amd_class_rank" ] && [ "$effective_free" -gt "$best_amd_free" ] 2>/dev/null; }; then
        best_amd_free=$effective_free
        best_amd_name=$_inv_name
        best_amd_class=$class
        best_amd_total=$_inv_total
        best_amd_class_rank=$class_rank
      fi
    fi
  done < "$GPU_INVENTORY_FILE"

  # Prefer NVIDIA if any card is at least "constrained"
  case $best_nvidia_class in
    recommended|constrained)
      RECOMMENDED_TIER="gpu"
      RECOMMENDATION_REASON="${best_nvidia_name}: ${best_nvidia_free} MiB free"
      return 0
      ;;
  esac

  # Then AMD if at least "constrained"
  case $best_amd_class in
    recommended|constrained)
      RECOMMENDED_TIER="rocm"
      if [ "$best_amd_free" = "-1" ]; then
        RECOMMENDATION_REASON="${best_amd_name}: ${best_amd_total} MiB total (free unknown)"
      else
        RECOMMENDATION_REASON="${best_amd_name}: ${best_amd_free} MiB free"
      fi
      return 0
      ;;
  esac

  # All GPUs are insufficient or unknown -- explain why
  local reasons=""
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    local tier_for_vendor
    [ "$_inv_vendor" = "nvidia" ] && tier_for_vendor="gpu" || tier_for_vendor="rocm"
    local class
    class=$(classify_gpu "$_inv_free" "$_inv_total" "$_inv_conf" "$tier_for_vendor")
    case $class in
      insufficient)
        if [ "$_inv_free" != "-1" ] && [ "$_inv_free" -lt 256 ] 2>/dev/null; then
          reasons="${reasons}${reasons:+; }${_inv_name}: ${_inv_free} MiB free (occupied)"
        elif [ "$_inv_free" != "-1" ]; then
          reasons="${reasons}${reasons:+; }${_inv_name}: ${_inv_free} MiB free (below budget)"
        else
          reasons="${reasons}${reasons:+; }${_inv_name}: ${_inv_total} MiB total (below budget)"
        fi
        ;;
      unknown)
        reasons="${reasons}${reasons:+; }${_inv_name}: free VRAM unknown"
        ;;
    esac
  done < "$GPU_INVENTORY_FILE"
  [ -n "$reasons" ] && RECOMMENDATION_REASON="$reasons"
}

# When the chosen tier has >1 eligible device, prompt for which one.
# When exactly one, auto-select it. Sets SEL_GPU_* globals.
# Usage: prompt_gpu_device TIER
prompt_gpu_device() {
  local tier=$1

  # Reset selection
  SEL_GPU_VENDOR=""; SEL_GPU_STABLE_ID=""; SEL_GPU_INDEX=""
  SEL_GPU_NAME=""; SEL_GPU_TOTAL_MIB=""; SEL_GPU_FREE_MIB=""
  SEL_GPU_RENDER_NODE=""; SEL_GPU_MEM_CONF=""

  [ -s "$GPU_INVENTORY_FILE" ] || return 0

  local target_vendor
  case $tier in
    gpu)  target_vendor="nvidia" ;;
    rocm) target_vendor="amd" ;;
    *)    return 0 ;;
  esac

  # Collect matching devices into parallel arrays (Bash 3.2: no associative arrays)
  local count=0
  local vendors=() indices=() stable_ids=() names=() totals=() frees=() render_nodes=() confs=()

  local line
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    [ "$_inv_vendor" = "$target_vendor" ] || continue
    count=$((count + 1))
    vendors+=("$_inv_vendor")
    indices+=("$_inv_idx")
    stable_ids+=("$_inv_stable_id")
    names+=("$_inv_name")
    totals+=("$_inv_total")
    frees+=("$_inv_free")
    render_nodes+=("$_inv_render_node")
    confs+=("$_inv_conf")
  done < "$GPU_INVENTORY_FILE"

  if [ "$count" -eq 0 ]; then
    say "  No ${target_vendor} GPUs detected."
    return 0
  fi

  local sel=0
  if [ "$count" -gt 1 ]; then
    say ""
    say "  Multiple ${target_vendor} GPUs detected. Pick one for Voxint:"
    local i=0
    while [ "$i" -lt "$count" ]; do
      local f_disp
      [ "${frees[$i]}" = "-1" ] && f_disp="unknown" || f_disp="${frees[$i]} MiB free"
      local t_disp="${totals[$i]} MiB"
      local status=""
      if [ "${frees[$i]}" != "-1" ] && [ "${frees[$i]}" -lt 256 ] 2>/dev/null; then
        status=" (occupied)"
      fi
      say "    [$((i + 1))] ${names[$i]}: ${t_disp} total, ${f_disp}${status}"
      i=$((i + 1))
    done
    # Default to the device with most free VRAM
    local best_idx=0 best_free=-1
    i=0
    while [ "$i" -lt "$count" ]; do
      if [ "${frees[$i]}" != "-1" ] && [ "${frees[$i]}" -gt "$best_free" ] 2>/dev/null; then
        best_free=${frees[$i]}
        best_idx=$i
      fi
      i=$((i + 1))
    done
    local def_num=$((best_idx + 1))
    local ans
    printf '  Device [%s]: ' "$def_num" >&2
    IFS= read -r ans || ans=""
    ans=${ans:-$def_num}
    case $ans in *[!0-9]*) ans=$def_num ;; esac
    if [ "$ans" -lt 1 ] || [ "$ans" -gt "$count" ] 2>/dev/null; then
      ans=$def_num
    fi
    sel=$((ans - 1))
  fi

  SEL_GPU_VENDOR=${vendors[$sel]}
  SEL_GPU_INDEX=${indices[$sel]}
  SEL_GPU_STABLE_ID=${stable_ids[$sel]}
  SEL_GPU_NAME=${names[$sel]}
  SEL_GPU_TOTAL_MIB=${totals[$sel]}
  SEL_GPU_FREE_MIB=${frees[$sel]}
  SEL_GPU_RENDER_NODE=${render_nodes[$sel]}
  SEL_GPU_MEM_CONF=${confs[$sel]}

  say "  Selected: ${SEL_GPU_NAME} (${SEL_GPU_STABLE_ID})"
}

# Warn if the selected GPU is below the VRAM budget for the chosen tier.
# Returns 0 to proceed, 1 to fall back to CPU. Requires explicit 'y' for
# below-budget devices; Enter defaults to falling back.
confirm_gpu_budget() {
  local tier=$1

  # Nothing selected or not a GPU tier
  [ -n "$SEL_GPU_STABLE_ID" ] || return 0
  case $tier in gpu|rocm) ;; *) return 0 ;; esac

  local class
  class=$(classify_gpu "$SEL_GPU_FREE_MIB" "$SEL_GPU_TOTAL_MIB" "$SEL_GPU_MEM_CONF" "$tier")

  case $class in
    recommended) return 0 ;;
    constrained)
      say "  Note: ${SEL_GPU_NAME} has ${SEL_GPU_FREE_MIB} MiB free -- tight but workable."
      return 0
      ;;
    insufficient)
      say ""
      say "  WARNING: ${SEL_GPU_NAME} has insufficient free VRAM for the $tier tier."
      if [ "$SEL_GPU_FREE_MIB" != "-1" ]; then
        say "  Currently ${SEL_GPU_FREE_MIB} MiB free; the $tier stack needs at least"
      else
        say "  Free VRAM is unknown; the $tier stack needs at least"
      fi
      case $tier in
        gpu)  say "  ${GPU_BUDGET_MINIMUM_MIB} MiB (recommended: ${GPU_BUDGET_RECOMMENDED_MIB} MiB)." ;;
        rocm) say "  ${ROCM_BUDGET_MINIMUM_MIB} MiB (recommended: ${ROCM_BUDGET_RECOMMENDED_MIB} MiB)." ;;
      esac
      say "  Model services may crash with out-of-memory errors."
      say ""
      local ans
      printf '  Proceed with %s anyway? Type "y" to continue, Enter to fall back to CPU: ' "$tier" >&2
      IFS= read -r ans || ans=""
      case $ans in
        y|Y) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    unknown)
      say "  Note: Could not determine free VRAM on ${SEL_GPU_NAME}."
      say "  If the GPU is occupied by another process, model services may run out of memory."
      return 0
      ;;
  esac
}

# Re-query the selected GPU's free VRAM. Called before compose up to warn if
# availability has changed since device selection. Does not block unattended starts.
refresh_selected_gpu() {
  [ -n "$SEL_GPU_STABLE_ID" ] || return 0

  if [ "$SEL_GPU_VENDOR" = "nvidia" ]; then
    local smi=${VOXINT_NVIDIA_SMI:-nvidia-smi}
    command -v "$smi" >/dev/null 2>&1 || return 0
    local out
    out=$("$smi" --query-gpu=uuid,memory.free --format=csv,noheader,nounits 2>/dev/null) || return 0
    local line uuid free
    while IFS= read -r line || [ -n "$line" ]; do
      uuid=${line%%,*}
      free=${line#*,}
      uuid=${uuid#"${uuid%%[![:blank:]]*}"}; uuid=${uuid%"${uuid##*[![:blank:]]}"}
      free=${free#"${free%%[![:blank:]]*}"}; free=${free%"${free##*[![:blank:]]}"}
      if [ "$uuid" = "$SEL_GPU_STABLE_ID" ]; then
        case $free in *[!0-9]*) return 0 ;; esac
        SEL_GPU_FREE_MIB=$free
        return 0
      fi
    done < <(printf '%s\n' "$out")
  elif [ "$SEL_GPU_VENDOR" = "amd" ]; then
    local sysfs_root=${VOXINT_SYSFS_ROOT:-/sys}
    local render_path
    for render_path in "${sysfs_root}"/class/drm/renderD*/device; do
      [ -d "$render_path" ] || continue
      local uid=""
      [ -f "${render_path}/unique_id" ] && uid=$(cat "${render_path}/unique_id" 2>/dev/null) || true
      local pci_slot=""
      [ -f "${render_path}/uevent" ] && \
        pci_slot=$(grep -o 'PCI_SLOT_NAME=[^ ]*' "${render_path}/uevent" 2>/dev/null | cut -d= -f2) || true
      local dev_path=${render_path%/device}
      local render_name=${dev_path##*/}
      local idx=${render_name#renderD}
      local cand=${uid:-$pci_slot}
      [ -z "$cand" ] && cand="renderD${idx}"
      if [ "$cand" = "$SEL_GPU_STABLE_ID" ]; then
        local total_bytes used_bytes
        [ -f "${render_path}/mem_info_vram_total" ] || return 0
        total_bytes=$(cat "${render_path}/mem_info_vram_total" 2>/dev/null) || return 0
        case $total_bytes in ''|*[!0-9]*) return 0 ;; esac
        [ "$total_bytes" -gt 0 ] 2>/dev/null || return 0
        local total_mib=$((total_bytes / 1048576))
        [ -f "${render_path}/mem_info_vram_used" ] || return 0
        used_bytes=$(cat "${render_path}/mem_info_vram_used" 2>/dev/null) || return 0
        case $used_bytes in ''|*[!0-9]*) return 0 ;; esac
        local used_mib=$((used_bytes / 1048576))
        SEL_GPU_FREE_MIB=$((total_mib - used_mib))
        [ "$SEL_GPU_FREE_MIB" -lt 0 ] && SEL_GPU_FREE_MIB=0
        return 0
      fi
    done
  fi
}

# Lowercase, collapse whitespace to single underscores, drop everything but
# [a-z0-9_], and trim leading/trailing underscores -- a stable signature token.
normalize_gpu_name() {
  local s
  s=$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' '_' | tr -cd 'a-z0-9_')
  s=${s#"${s%%[!_]*}"}   # strip a leading run of underscores
  s=${s%"${s##*[!_]}"}   # strip a trailing run of underscores
  printf '%s' "$s"
}

# Query the host NVIDIA driver for ONE unambiguous GPU signature. Sets GPU_NAME/
# GPU_VRAM_MIB/GPU_SIGNATURE, or leaves them empty ("unknown") when nvidia-smi is
# absent or fails, the output is malformed, MIG is enabled, or more than one
# DISTINCT GPU signature is present -- we do not guess for mixed cards.
detect_nvidia_gpu() {
  GPU_NAME=""; GPU_VRAM_MIB=""; GPU_SIGNATURE=""
  # VOXINT_NVIDIA_SMI overrides the tool path (a test seam, like VOXINT_INSTALL_LIB;
  # also lets an operator point at a non-default location). Defaults to nvidia-smi.
  local smi=${VOXINT_NVIDIA_SMI:-nvidia-smi}
  command -v "$smi" >/dev/null 2>&1 || return 0
  local out
  out=$("$smi" --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null) || return 0
  [ -n "$out" ] || return 0

  # Process substitution (not a pipe) keeps this loop in the current shell so the
  # GPU_* assignments persist; feeding "$out" through printf avoids the here-doc
  # re-expansion of driver-supplied names (no `$(...)`/backtick evaluation).
  local line trimmed name vram sig first_sig="" count=0
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    trimmed=${line#"${line%%[![:blank:]]*}"}
    trimmed=${trimmed%"${trimmed##*[![:blank:]]}"}
    [ -n "$trimmed" ] || continue
    name=${line%%,*}
    vram=${line#*,}
    name=${name#"${name%%[![:blank:]]*}"}; name=${name%"${name##*[![:blank:]]}"}
    vram=${vram#"${vram%%[![:blank:]]*}"}; vram=${vram%"${vram##*[![:blank:]]}"}
    # nounits means a plain integer; MIG rows carry "N/A" or "[MIG …]" names.
    # NOTE: this name-substring check catches MIG-instance rows but not a physical
    # card with MIG merely enabled (that needs mig.mode.current). Harmless today --
    # every unknown card gets the same conservative fallback -- but a future
    # per-GPU BATCH_SIZE profile (gated on parity + an OOM soak) must query
    # mig.mode.current before it can trust the name+VRAM signature.
    case $vram in ''|*[!0-9]*) GPU_NAME=""; GPU_VRAM_MIB=""; GPU_SIGNATURE=""; return 0 ;; esac
    case $name in *MIG*|*mig*) GPU_NAME=""; GPU_VRAM_MIB=""; GPU_SIGNATURE=""; return 0 ;; esac
    sig="$(normalize_gpu_name "$name")|$vram"
    count=$((count + 1))
    if [ "$count" -eq 1 ]; then
      first_sig=$sig; GPU_NAME=$name; GPU_VRAM_MIB=$vram
    elif [ "$sig" != "$first_sig" ]; then
      GPU_NAME=""; GPU_VRAM_MIB=""; GPU_SIGNATURE=""; return 0
    fi
  done < <(printf '%s\n' "$out")
  [ -n "$GPU_NAME" ] && GPU_SIGNATURE=$first_sig
  return 0
}

PROFILE_ID=""
PROFILE_CONCURRENCY=""
PROFILE_MAX_PENDING=""
PROFILE_BATCH_SIZE=""
# Set by write_hardware_override so the caller can report what actually happened:
# "written" (created/refreshed), "unchanged" (already current), or "refused"
# (a non-managed/non-regular/symlinked target left untouched).
HARDWARE_WRITE_RESULT=""

# Map a detected GPU signature to a TESTED sizing profile. Profile matching,
# never a VRAM formula. Only the conservative unknown fallback ships today;
# concrete per-GPU arms (which may set BATCH_SIZE) are added ONLY after they pass
# tests/parity + a real-GPU OOM soak. The empty-schema case statement is trivial
# to extend and locks no premature schema.
select_hardware_profile() {
  local compute_type=${1:-int8}
  case "${GPU_SIGNATURE}|${compute_type}" in
    # Parity-gated tested profiles go here, e.g.:
    #   nvidia_geforce_rtx_3060|12288|int8) PROFILE_ID='rtx-3060-12g-int8-v1'; PROFILE_CONCURRENCY=1; PROFILE_MAX_PENDING=1; PROFILE_BATCH_SIZE='8' ;;
    *)
      PROFILE_ID="nvidia-unknown-v1"
      PROFILE_CONCURRENCY=1
      PROFILE_MAX_PENDING=1
      PROFILE_BATCH_SIZE=""   # never auto-set without a parity-passed profile
      ;;
  esac
}

# Emit the compose.hardware.yaml body on stdout for the selected PROFILE_*.
# Deterministic. The worker command is the base command + --concurrency; whisper
# gets MAX_PENDING_REQUESTS (additive env merge). BATCH_SIZE is emitted ONLY when
# a profile set it -- never for the unknown fallback.
render_hardware_override() {
  # Sanitize identities before YAML emission (defense-in-depth).
  if [ -n "$SEL_GPU_STABLE_ID" ]; then
    case $SEL_GPU_STABLE_ID in
      GPU-*|[0-9]*:*.*|0x*|renderD[0-9]*) ;;
      *) say "  WARNING: unexpected GPU id format: $SEL_GPU_STABLE_ID -- skipping device pinning."
         SEL_GPU_STABLE_ID=""; SEL_GPU_RENDER_NODE="" ;;
    esac
  fi
  if [ -n "$SEL_GPU_RENDER_NODE" ]; then
    case $SEL_GPU_RENDER_NODE in
      /dev/dri/renderD[0-9]*) ;;
      *) say "  WARNING: unexpected render node format: $SEL_GPU_RENDER_NODE -- skipping device pinning."
         SEL_GPU_STABLE_ID=""; SEL_GPU_RENDER_NODE="" ;;
    esac
  fi

  printf '%s\n' "${HARDWARE_OVERRIDE_MARKER} v2"
  printf '%s\n' "# profile: ${PROFILE_ID}"
  if [ -n "$SEL_GPU_STABLE_ID" ]; then
    printf '%s\n' "# device: ${SEL_GPU_NAME} (${SEL_GPU_TOTAL_MIB} MiB)"
  fi
  printf '%s\n' "# Generated by scripts/install.sh -- regenerated on every install run."
  printf '%s\n' "# Do not edit; put your own overrides in compose.override.yaml instead."
  printf '%s\n' "services:"
  printf '%s\n' "  worker:"
  printf '%s\n' "    command: ${WORKER_BASE_COMMAND} --concurrency=${PROFILE_CONCURRENCY}"

  if [ "${EFFECTIVE_TIER:-}" = "rocm" ] && [ -n "$SEL_GPU_RENDER_NODE" ]; then
    # ROCm: pin whisper to one renderD node; container sees one GPU as index 0
    printf '%s\n' "  whisper:"
    printf '%s\n' "    devices: !override"
    printf '%s\n' "      - /dev/kfd"
    printf '%s\n' "      - ${SEL_GPU_RENDER_NODE}"
    printf '%s\n' "    environment:"
    printf '%s\n' "      MAX_PENDING_REQUESTS: \"${PROFILE_MAX_PENDING}\""
    printf '%s\n' "      HIP_VISIBLE_DEVICES: \"0\""
    printf '%s\n' "      ROCR_VISIBLE_DEVICES: \"0\""
    if [ -n "$PROFILE_BATCH_SIZE" ]; then
      printf '%s\n' "      BATCH_SIZE: \"${PROFILE_BATCH_SIZE}\""
    fi
  elif [ "${EFFECTIVE_TIER:-}" = "gpu" ] && [ -n "$SEL_GPU_STABLE_ID" ]; then
    # NVIDIA: pin all three model services to one UUID
    printf '%s\n' "  whisper:"
    printf '%s\n' "    environment:"
    printf '%s\n' "      MAX_PENDING_REQUESTS: \"${PROFILE_MAX_PENDING}\""
    if [ -n "$PROFILE_BATCH_SIZE" ]; then
      printf '%s\n' "      BATCH_SIZE: \"${PROFILE_BATCH_SIZE}\""
    fi
    printf '%s\n' "    deploy:"
    printf '%s\n' "      resources:"
    printf '%s\n' "        reservations:"
    printf '%s\n' "          devices: !override"
    printf '%s\n' "            - driver: nvidia"
    printf '%s\n' "              device_ids: [\"${SEL_GPU_STABLE_ID}\"]"
    printf '%s\n' "              capabilities: [gpu]"
    printf '%s\n' "  pyannote:"
    printf '%s\n' "    deploy:"
    printf '%s\n' "      resources:"
    printf '%s\n' "        reservations:"
    printf '%s\n' "          devices: !override"
    printf '%s\n' "            - driver: nvidia"
    printf '%s\n' "              device_ids: [\"${SEL_GPU_STABLE_ID}\"]"
    printf '%s\n' "              capabilities: [gpu]"
    printf '%s\n' "  titanet:"
    printf '%s\n' "    deploy:"
    printf '%s\n' "      resources:"
    printf '%s\n' "        reservations:"
    printf '%s\n' "          devices: !override"
    printf '%s\n' "            - driver: nvidia"
    printf '%s\n' "              device_ids: [\"${SEL_GPU_STABLE_ID}\"]"
    printf '%s\n' "              capabilities: [gpu]"
  else
    # No device selected: scheduling caps only (backward compatible)
    printf '%s\n' "  whisper:"
    printf '%s\n' "    environment:"
    printf '%s\n' "      MAX_PENDING_REQUESTS: \"${PROFILE_MAX_PENDING}\""
    if [ -n "$PROFILE_BATCH_SIZE" ]; then
      printf '%s\n' "      BATCH_SIZE: \"${PROFILE_BATCH_SIZE}\""
    fi
  fi
}

# Ownership-safe atomic write of compose.hardware.yaml. Refuses a non-regular or
# operator-authored (unmarked) target; no-ops when the managed content is already
# identical; otherwise renders to a temp, validates it as part of the FULL
# effective GPU chain + .env, then moves it into place. Returns non-zero (without
# failing the install) when it deliberately left an existing file untouched.
write_hardware_override() {
  HARDWARE_WRITE_RESULT="refused"
  if [ -L "$HARDWARE_OVERRIDE_FILE" ]; then
    say "  NOTE: $HARDWARE_OVERRIDE_FILE is a symlink -- leaving it untouched"
    say "  (put custom overrides in compose.override.yaml instead)."
    return 1
  fi
  if [ -e "$HARDWARE_OVERRIDE_FILE" ] && [ ! -f "$HARDWARE_OVERRIDE_FILE" ]; then
    say "  NOTE: $HARDWARE_OVERRIDE_FILE exists but is not a regular file -- leaving it untouched."
    return 1
  fi
  if [ -f "$HARDWARE_OVERRIDE_FILE" ] && ! hardware_override_is_managed; then
    say "  NOTE: $HARDWARE_OVERRIDE_FILE exists and was not generated by the installer;"
    say "  leaving it untouched (put custom overrides in compose.override.yaml)."
    return 1
  fi

  local candidate current
  candidate=$(render_hardware_override)
  if hardware_override_is_managed; then
    current=$(cat "$HARDWARE_OVERRIDE_FILE" 2>/dev/null || true)
    if [ "$current" = "$candidate" ]; then
      HARDWARE_WRITE_RESULT="unchanged"
      return 0  # idempotent: nothing changed
    fi
  fi

  local tmp old_umask rc
  old_umask=$(umask); umask 077
  tmp=$(mktemp "$REPO_ROOT/.compose.hardware.tmp.XXXXXX") || { umask "$old_umask"; fail "Could not create a temp file in $REPO_ROOT."; }
  _CLEANUP_TMP=$tmp
  printf '%s\n' "$candidate" >"$tmp" || { umask "$old_umask"; fail "Failed writing candidate hardware override."; }
  umask "$old_umask"

  # Validate against the file set that will actually run (base + tier overlay +
  # this candidate + any operator override), so a bad render fails here, not at up.
  # $tmp is an absolute path (REPO_ROOT can contain spaces on macOS), so pass every
  # -f as a discrete quoted argument rather than word-splitting one string.
  local tier_file
  case ${EFFECTIVE_TIER:-gpu} in
    gpu)  tier_file="compose.gpu.yaml" ;;
    rocm) tier_file="compose.rocm.yaml" ;;
    *)    tier_file="compose.gpu.yaml" ;;
  esac
  if [ -f compose.override.yaml ]; then
    docker compose -f compose.yaml -f "$tier_file" -f "$tmp" -f compose.override.yaml --env-file .env config --quiet >/dev/null 2>&1 && rc=0 || rc=$?
  else
    docker compose -f compose.yaml -f "$tier_file" -f "$tmp" --env-file .env config --quiet >/dev/null 2>&1 && rc=0 || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    rm -f "$tmp"; _CLEANUP_TMP=""
    fail "Generated $HARDWARE_OVERRIDE_FILE failed Compose validation. This is a bug -- please report it."
  fi

  mv -f "$tmp" "$HARDWARE_OVERRIDE_FILE" || fail "Could not move $HARDWARE_OVERRIDE_FILE into place."
  _CLEANUP_TMP=""
  chmod 644 "$HARDWARE_OVERRIDE_FILE" 2>/dev/null || true
  HARDWARE_WRITE_RESULT="written"
  return 0
}

# GPU/ROCm tiers only: pick conservative defaults and generate/refresh
# compose.hardware.yaml, then refold COMPOSE_FILE_ARGS so pull/up/ps/handoff all
# agree on the effective chain. Requires .env in place (the write validates it).
configure_hardware_defaults() {
  case $EFFECTIVE_TIER in gpu|rocm) ;; *) return 0 ;; esac

  # Kept-env path: SEL_GPU_* may be empty (device selection only runs during
  # configure/generate). Auto-select the best device to preserve pinning.
  if [ -z "$SEL_GPU_STABLE_ID" ]; then
    create_gpu_inventory
    local target_vendor
    case $EFFECTIVE_TIER in gpu) target_vendor="nvidia" ;; rocm) target_vendor="amd" ;; esac
    local best_free=-1 line
    while IFS= read -r line || [ -n "$line" ]; do
      _read_inv_line "$line"
      [ "$_inv_vendor" = "$target_vendor" ] || continue
      local eff_free=$_inv_free
      [ "$eff_free" = "-1" ] && eff_free=$_inv_total
      if [ "$eff_free" -gt "$best_free" ] 2>/dev/null; then
        best_free=$eff_free
        SEL_GPU_VENDOR=$_inv_vendor; SEL_GPU_INDEX=$_inv_idx; SEL_GPU_STABLE_ID=$_inv_stable_id
        SEL_GPU_NAME=$_inv_name; SEL_GPU_TOTAL_MIB=$_inv_total; SEL_GPU_FREE_MIB=$_inv_free
        SEL_GPU_RENDER_NODE=$_inv_render_node; SEL_GPU_MEM_CONF=$_inv_conf
      fi
    done < "$GPU_INVENTORY_FILE"
    cleanup_gpu_inventory
  fi
  step "Applying conservative hardware defaults"

  # For GPU tier, detect the card for profile matching.
  # For ROCm, use the conservative unknown profile.
  if [ "$EFFECTIVE_TIER" = "gpu" ]; then
    detect_nvidia_gpu
    select_hardware_profile
    if [ -n "$GPU_NAME" ]; then
      say "  Detected GPU: $GPU_NAME (${GPU_VRAM_MIB} MiB)."
    else
      say "  Could not read a single, unambiguous NVIDIA GPU from nvidia-smi;"
      say "  using the conservative default that suits any modest card."
    fi
  else
    # ROCm: no per-GPU profile matching yet
    PROFILE_ID="rocm-unknown-v1"
    PROFILE_CONCURRENCY=1
    PROFILE_MAX_PENDING=1
    PROFILE_BATCH_SIZE=""
  fi

  if [ -n "$SEL_GPU_STABLE_ID" ]; then
    say "  Device: ${SEL_GPU_NAME} (${SEL_GPU_STABLE_ID})"
  fi

  if write_hardware_override; then
    say "  Profile '$PROFILE_ID' applied: worker --concurrency=$PROFILE_CONCURRENCY,"
    say "  whisper MAX_PENDING_REQUESTS=$PROFILE_MAX_PENDING. This is a scheduling-only"
    say "  baseline: it reduces overlapping GPU load but does not lower peak per-request"
    say "  memory, so a card too small for a single job can still run out of memory."
    if [ -z "$PROFILE_BATCH_SIZE" ]; then
      say "  Transcription batch size stays at the image default (unchanged): auto-tuning"
      say "  it needs a measured per-GPU profile, so the installer does not set it. If a"
      say "  run hits out-of-memory, lower BATCH_SIZE by hand in compose.override.yaml."
    fi
    if [ -n "$SEL_GPU_STABLE_ID" ]; then
      if [ "$EFFECTIVE_TIER" = "gpu" ]; then
        say "  NVIDIA device pinned: ${SEL_GPU_STABLE_ID}"
      else
        say "  ROCm device pinned: ${SEL_GPU_RENDER_NODE}"
      fi
    fi
    if [ "$HARDWARE_WRITE_RESULT" = "unchanged" ]; then
      say "  $HARDWARE_OVERRIDE_FILE was already current. See docs/operations.md (#96)."
    else
      say "  Wrote $HARDWARE_OVERRIDE_FILE. See docs/operations.md (#96)."
    fi
  else
    say "  Conservative hardware defaults were NOT applied: $HARDWARE_OVERRIDE_FILE is present"
    say "  but not installer-managed, so it is left as-is and is NOT merged into the"
    say "  compose chain. Put your own overrides in compose.override.yaml instead."
  fi
  COMPOSE_FILE_ARGS=$(compose_file_args_for_tier "$EFFECTIVE_TIER")
  if [ -f compose.override.yaml ]; then
    say "  Note: your compose.override.yaml is merged last -- it wins over these defaults."
  fi
}

# `install.sh --hardware-dry-run`: report the inventory, recommendation, selected
# devices, and exact GPU/ROCm compose.hardware.yaml bodies that WOULD be written.
hardware_dry_run() {
  create_gpu_inventory
  say "=== GPU Inventory ==="
  print_gpu_inventory "$GPU_INVENTORY_FILE"
  say ""
  recommend_compute_tier
  say "Recommended tier: $RECOMMENDED_TIER"
  say "Reason: $RECOMMENDATION_REASON"
  say ""

  # Show what hardware_override would look like for gpu tier
  detect_nvidia_gpu
  select_hardware_profile
  if [ -n "$GPU_NAME" ]; then
    say "NVIDIA profile: $PROFILE_ID (worker --concurrency=$PROFILE_CONCURRENCY, whisper MAX_PENDING_REQUESTS=$PROFILE_MAX_PENDING)"
  fi
  if [ -z "$PROFILE_BATCH_SIZE" ]; then
    say "BATCH_SIZE: unchanged (needs a measured per-GPU profile before it is auto-set)."
  fi

  # Retain the prospective chain in the report for existing CLI consumers.
  local chain="-f compose.yaml -f compose.gpu.yaml -f $HARDWARE_OVERRIDE_FILE"
  if [ -f compose.override.yaml ]; then chain="$chain -f compose.override.yaml"; fi
  say "Effective GPU compose chain would be:"
  say "  docker compose $chain"

  # Simulate GPU tier with best NVIDIA device
  local saved_tier=$EFFECTIVE_TIER
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    if [ "$_inv_vendor" = "nvidia" ]; then
      SEL_GPU_VENDOR=$_inv_vendor; SEL_GPU_INDEX=$_inv_idx; SEL_GPU_STABLE_ID=$_inv_stable_id
      SEL_GPU_NAME=$_inv_name; SEL_GPU_TOTAL_MIB=$_inv_total; SEL_GPU_FREE_MIB=$_inv_free
      SEL_GPU_RENDER_NODE=""; SEL_GPU_MEM_CONF=$_inv_conf
      break
    fi
  done < "$GPU_INVENTORY_FILE"
  # Always render the conservative GPU scheduling caps. When inventory has a
  # device, the same rendering also demonstrates UUID pinning.
  EFFECTIVE_TIER=gpu
  say ""
  say "--- GPU tier compose.hardware.yaml would contain: ---"
  render_hardware_override
  say "-----------------------------------------------------"

  # Simulate ROCm tier with best AMD device
  SEL_GPU_VENDOR=""; SEL_GPU_STABLE_ID=""; SEL_GPU_RENDER_NODE=""
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    if [ "$_inv_vendor" = "amd" ]; then
      SEL_GPU_VENDOR=$_inv_vendor; SEL_GPU_INDEX=$_inv_idx; SEL_GPU_STABLE_ID=$_inv_stable_id
      SEL_GPU_NAME=$_inv_name; SEL_GPU_TOTAL_MIB=$_inv_total; SEL_GPU_FREE_MIB=$_inv_free
      SEL_GPU_RENDER_NODE=$_inv_render_node; SEL_GPU_MEM_CONF=$_inv_conf
      PROFILE_ID="rocm-unknown-v1"; PROFILE_CONCURRENCY=1; PROFILE_MAX_PENDING=1; PROFILE_BATCH_SIZE=""
      break
    fi
  done < "$GPU_INVENTORY_FILE"
  if [ -n "$SEL_GPU_STABLE_ID" ]; then
    EFFECTIVE_TIER=rocm
    say ""
    say "--- ROCm tier compose.hardware.yaml would contain: ---"
    render_hardware_override
    say "------------------------------------------------------"
  fi

  EFFECTIVE_TIER=$saved_tier
  cleanup_gpu_inventory
  say ""
  say "(dry run -- nothing was written or started.)"
}

# `install.sh --gpu-check`: read-only GPU inventory and classification.
# No Docker daemon, .env, or file writes required.
gpu_check() {
  create_gpu_inventory
  say "=== GPU Inventory ==="
  print_gpu_inventory "$GPU_INVENTORY_FILE"
  say ""
  recommend_compute_tier
  say "Recommended tier: $RECOMMENDED_TIER"
  say "Reason: $RECOMMENDATION_REASON"
  say ""

  # Classify each GPU for both tiers
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    _read_inv_line "$line"
    local gpu_class rocm_class
    gpu_class=$(classify_gpu "$_inv_free" "$_inv_total" "$_inv_conf" "gpu")
    rocm_class=$(classify_gpu "$_inv_free" "$_inv_total" "$_inv_conf" "rocm")
    say "  ${_inv_name}: gpu=$gpu_class, rocm=$rocm_class"
  done < "$GPU_INVENTORY_FILE"

  cleanup_gpu_inventory
  say ""
  say "(read-only check -- nothing was written or started.)"
}

# ---------------------------------------------------------------------------
# Pull + start. Compose is authoritative for port binds and config errors; we
# do not parse its messages or auto-retry with different ports.
# ---------------------------------------------------------------------------
pull_and_start() {
  # Refresh selected GPU VRAM before starting services
  if [ -n "${SEL_GPU_STABLE_ID:-}" ]; then
    local old_free=$SEL_GPU_FREE_MIB
    refresh_selected_gpu
    if [ "$SEL_GPU_FREE_MIB" != "$old_free" ] && [ "$SEL_GPU_FREE_MIB" != "-1" ]; then
      local class
      class=$(classify_gpu "$SEL_GPU_FREE_MIB" "$SEL_GPU_TOTAL_MIB" "$SEL_GPU_MEM_CONF" "$EFFECTIVE_TIER")
      case $class in
        insufficient)
          say "  NOTE: ${SEL_GPU_NAME} now has ${SEL_GPU_FREE_MIB} MiB free (was ${old_free} MiB)."
          say "  VRAM availability changed; model services may run out of memory."
          ;;
      esac
    fi
  fi

  step "Pulling release images (first run can take a few minutes)"
  dc pull || fail "Image pull failed. Check your connection and re-run."

  if [ "$EFFECTIVE_TIER" = "none" ]; then
    step "Starting the stack (core services)"
  elif [ "$EFFECTIVE_TIER" = "metal" ]; then
    # The metal overlay only rewires api/worker -- the model services run
    # natively and are NOT started by Compose (see the handoff below).
    step "Starting the stack (core services, wired for native metal model services)"
  else
    step "Starting the stack (core + $EFFECTIVE_TIER model services)"
  fi
  # --remove-orphans: on a re-run that switched tier (e.g. gpu -> none, any
  # tier switch), containers from the previously-active overlay are no
  # longer in the file set and would otherwise keep running while the handoff
  # claims core-only. Compose removes exactly those; same-name services on a
  # gpu<->cpu switch are recreated, not orphaned.
  if ! dc up -d --remove-orphans; then
    say ""
    say "Startup failed. Current services:"
    dc ps -a >&2 || true
    say ""
    say "Inspect the one-shot migration and API logs with:"
    say "  docker compose $COMPOSE_FILE_ARGS logs migrate"
    say "  docker compose $COMPOSE_FILE_ARGS logs api"
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
    cid=$(dc ps -aq api 2>/dev/null || true)
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
          dc logs --tail 40 migrate api >&2 2>/dev/null || true
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
          dc logs --tail 40 api >&2 2>/dev/null || true
          fail "The API did not come up cleanly."
          ;;
      esac
    else
      # No API container yet -- if the one-shot 'migrate' already failed, the API
      # will never start, so surface that now instead of waiting out the timeout.
      mid=$(dc ps -aq migrate 2>/dev/null || true)
      if [ -n "$mid" ]; then
        mstate=$(docker inspect --format '{{.State.Status}}|{{.State.ExitCode}}' "$mid" 2>/dev/null || true)
        mcode=${mstate#*|}
        case ${mstate%%|*} in
          exited|dead)
            if [ "$mcode" != "0" ]; then
              say ""
              say "The one-shot 'migrate' service failed (exit $mcode). Recent logs:"
              dc logs --tail 40 migrate >&2 2>/dev/null || true
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
  say "  docker compose $COMPOSE_FILE_ARGS ps -a"
  say "  docker compose $COMPOSE_FILE_ARGS logs migrate api"
  fail "API health check timed out."
}

# ---------------------------------------------------------------------------
# Handoff. Print the URL and username (never the password), explain the
# migrate one-shot, and state honestly what is and is not running.
# ---------------------------------------------------------------------------
print_handoff() {
  local bind url port
  bind=$(dc port api 8080 2>/dev/null || true)
  case $bind in
    *:*) url="http://$bind/" ;;
    *)
      # Fallback if `docker compose port` fails: read the configured API_PORT from
      # .env rather than assuming the default (which would be wrong on an override).
      port=$(read_env_value API_PORT)
      [ -n "$port" ] || port=8080
      url="http://127.0.0.1:$port/"
      ;;
  esac

  printf '\n' >&2
  say "======================================================================"
  if [ "$EFFECTIVE_TIER" = "none" ]; then
    say " Voxint core stack is up."
  elif [ "$EFFECTIVE_TIER" = "metal" ]; then
    # The metal overlay starts NO model services (they run natively) -- the
    # headline must not claim otherwise; the tier section below hands off.
    say " Voxint core is up; the native metal model services are NOT running yet."
  else
    # "core is up" is what wait_for_health verified; the model services were
    # started but are NOT health-checked here (first boot downloads weights,
    # which can take minutes) -- the tier section below says how to check them.
    say " Voxint core is up; $EFFECTIVE_TIER-tier model services were started."
  fi
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
  case $EFFECTIVE_TIER in
    cpu)
      say " The CPU-tier model services were STARTED alongside the core stack."
      say " They may still be booting and loading models on a first run --"
      say " check their status before submitting audio:"
      say "   docker compose $COMPOSE_FILE_ARGS ps"
      say " Expect CPU inference to be much slower than a GPU -- a long recording"
      say " can take hours. Slow-but-healthy runs are protected by the CPU timing"
      say " profile (COMPUTE_TIER=cpu)."
      say ""
      say " Stop the stack with:"
      say "   docker compose $COMPOSE_FILE_ARGS down"
      ;;
    gpu)
      say " The GPU-tier model services were STARTED alongside the core stack."
      say " They may still be booting and loading models on a first run --"
      say " check their status before submitting audio:"
      say "   docker compose $COMPOSE_FILE_ARGS ps"
      say " A run submitted before they are ready retries with backoff, so a"
      say " brief warmup is harmless; if a run does land failed, requeue it from"
      say " the run's page once the services are up."
      say ""
      say " These commands include the conservative compose.hardware.yaml the"
      say " installer generated; keep the full flag list so re-runs stay capped."
      say " Stop the stack with:"
      say "   docker compose $COMPOSE_FILE_ARGS down"
      ;;
    metal)
      say " IMPORTANT -- the core stack is wired for the metal tier, but the"
      say " native model services are NOT running yet: Docker only runs the"
      say " core here. Submitting audio now will fail at the processing stages."
      say ""
      say " Finish the native side (creates local venvs and downloads ~3.2 GB"
      say " of sha-verified model weights, then starts the services under"
      say " launchd):"
      say "   ./scripts/metal/voxint-metal.sh setup"
      say "   ./scripts/metal/voxint-metal.sh up"
      say "   ./scripts/metal/voxint-metal.sh status"
      say " Expected status: whisper device cpu, pyannote device mps (Apple"
      say " GPU), titanet device cpu."
      say ""
      say " Stop everything with:"
      say "   ./scripts/metal/voxint-metal.sh down"
      say "   docker compose $COMPOSE_FILE_ARGS down"
      ;;
    rocm)
      say " The AMD (ROCm) model services were STARTED alongside the core stack."
      say " Transcription runs on the AMD GPU; diarization and speaker embedding"
      say " run on CPU (see docs/operations.md for why). Services may still be"
      say " booting and loading models on a first run -- check their status"
      say " before submitting audio:"
      say "   docker compose $COMPOSE_FILE_ARGS ps"
      say " The CPU-bound stages are protected by the rocm timing profile"
      say " (COMPUTE_TIER=rocm)."
      say ""
      say " Stop the stack with:"
      say "   docker compose $COMPOSE_FILE_ARGS down"
      ;;
    *)
      say " IMPORTANT -- this is the CORE control plane only. You can open the"
      say " console and adjudicate, but transcription, diarization, and speaker"
      say " embedding need the model services. Submitting audio now will fail at"
      say " those stages. To enable processing later, bring up a compute tier:"
      say "   docker compose -f compose.yaml -f compose.gpu.yaml up -d    # NVIDIA GPU"
      say "   docker compose -f compose.yaml -f compose.rocm.yaml up -d   # AMD GPU"
      say "   docker compose -f compose.yaml -f compose.cpu.yaml up -d    # no GPU"
      say " (or re-run this installer and pick a tier)."
      say ""
      say " Stop the stack with:  docker compose down"
      ;;
  esac
  say " See the Quickstart in README.md and docs/operations.md."
  say "======================================================================"
}

# ---------------------------------------------------------------------------
main() {
  # Armed here (not at global scope) so sourcing in library mode leaves the
  # caller's traps untouched. Removes the temp .env on Ctrl+C / kill / hangup.
  trap '_cleanup; exit 130' INT
  trap '_cleanup; exit 143' TERM HUP
  trap _cleanup EXIT

  # Read-only advisory mode: show what the hardware defaults would do, touch
  # nothing. No Docker daemon or .env required.
  case ${1:-} in
    --hardware-dry-run) hardware_dry_run; return 0 ;;
    --gpu-check) gpu_check; return 0 ;;
  esac

  say "Voxint installer"
  preflight
  decide_existing_env
  if [ "$ENV_ACTION" = "generate" ]; then
    configure
  else
    say "Keeping the existing .env."
    resolve_kept_env_tier
  fi
  configure_hardware_defaults
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
