#!/usr/bin/env bash
# voxint-metal.sh -- native model services for the bare-metal Apple Silicon
# ("metal") compute tier.
#
# The guided installer (scripts/install.sh, option [M]) starts ONLY the Docker
# core stack (postgres/redis/api/worker/beat via compose.yaml +
# compose.metal.yaml) and hands off here. This script sets up and supervises
# the three model services natively on macOS so pyannote can use the Apple GPU
# through MPS -- Docker Desktop has no GPU passthrough. Native services bind
# 127.0.0.1 only; api/worker reach them through host.docker.internal.
#
#   setup                 create venvs, fetch + sha-verify weights, generate
#                         the vendored pyannote config (network required)
#   up | down             start/stop all three services under launchd
#                         (KeepAlive restarts them after a crash -- the native
#                         equivalent of the containers' restart policy)
#   status                per-service /healthz + supervision state + a
#                         working-tree vs image-tag skew warning
#   logs <svc> [-f]       show (or follow) a service's log
#   doctor                environment checks: weights, config, MEDIA_ROOT
#                         agreement, port collisions, MPS probe, ORT providers
#   run <svc> --foreground  run one service in the foreground for debugging
#
# Layout (override the root with VOXINT_METAL_HOME):
#   $HOME/.voxint-metal/{venvs,models,logs,run}
#
# Requirements: macOS on Apple Silicon, uv, curl. Python 3.11 venvs are
# created per service to match the container images. Bash 3.2 compatible.
#
# Sourcing with VOXINT_METAL_LIB=1 loads the functions without running main
# (tests/unit/test_metal_launcher.py exercises the pure logic that way).

set -eu

# ---------------------------------------------------------------------------
# Constants. The weight releases and shas are the SAME sources the container
# images bake at build time -- provenance JSONs are the single authority.
# ---------------------------------------------------------------------------
METAL_SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
# When sourced for tests, $0 is the shell -- fall back to BASH_SOURCE.
case $METAL_SCRIPT_DIR in
  */scripts/metal) : ;;
  *) METAL_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) ;;
esac
REPO_ROOT=$(cd "$METAL_SCRIPT_DIR/../.." && pwd -P)
# A symlinked invocation (~/bin/voxint-metal.sh) resolves $0 to the link's
# directory and lands REPO_ROOT somewhere wrong; fail with the cause instead
# of surfacing it later as missing-requirements noise.
if [ ! -f "$REPO_ROOT/compose.metal.yaml" ]; then
  printf 'ERROR: cannot locate the Voxint checkout from %s (symlinked? run the script from scripts/metal/ in the checkout)\n' "$0" >&2
  exit 1
fi

VOXINT_METAL_HOME=${VOXINT_METAL_HOME:-$HOME/.voxint-metal}

METAL_SERVICES="whisper pyannote titanet"
GITHUB_REPO=${VOXINT_GITHUB_REPO:-bengizmo/voxint}
PYANNOTE_MODELS_RELEASE=pyannote-models-v1
TITANET_ONNX_RELEASE=titanet-onnx-v1
# Same pinned HF revision Dockerfile.cpu bakes -- keeps every deployment of
# large-v2 byte-identical (docs/gpu-contracts.md).
WHISPER_HF_REVISION=f0fe81560cb8b68660e564f55dd99207059c092e
WHISPER_MODEL_REPO=Systran/faster-whisper-large-v2
LAUNCHD_PREFIX=com.voxint.metal
# Log rotation knobs (launchd StandardOutPath never rotates; KeepAlive keeps
# services up for months). copytruncate rotation runs from cmd_up and from a
# daily launchd job (com.voxint.metal.logrotate).
VOXINT_METAL_LOG_MAX_MB=${VOXINT_METAL_LOG_MAX_MB:-50}
VOXINT_METAL_LOG_ARCHIVES=${VOXINT_METAL_LOG_ARCHIVES:-5}

say()  { printf '%s\n' "$*" >&2; }
step() { printf '\n== %s\n' "$*" >&2; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested through the library seam)
# ---------------------------------------------------------------------------
service_port() {
  case $1 in
    whisper)  printf '8022' ;;
    pyannote) printf '8024' ;;
    titanet)  printf '8021' ;;
    *) return 1 ;;
  esac
}

service_requirements() {
  # titanet reuses the CPU image's file VERBATIM on purpose: the committed
  # ONNX parity verdict binds to exactly that dependency chain.
  case $1 in
    whisper)  printf 'services/whisper/requirements.metal.txt' ;;
    pyannote) printf 'services/pyannote/requirements.metal.txt' ;;
    titanet)  printf 'services/titanet/requirements.cpu.txt' ;;
    *) return 1 ;;
  esac
}

service_venv()  { printf '%s/venvs/%s' "$VOXINT_METAL_HOME" "$1"; }
plist_label()   { printf '%s.%s' "$LAUNCHD_PREFIX" "$1"; }
plist_path()    { printf '%s/run/%s.plist' "$VOXINT_METAL_HOME" "$(plist_label "$1")"; }
service_log()   { printf '%s/logs/%s.log' "$VOXINT_METAL_HOME" "$1"; }

# Python used for JSON/YAML verification work. Defaults to the pyannote venv
# (guaranteed by setup order); tests override with VOXINT_METAL_PYTHON.
metal_python() {
  if [ -n "${VOXINT_METAL_PYTHON:-}" ]; then
    printf '%s' "$VOXINT_METAL_PYTHON"
  else
    printf '%s/bin/python' "$(service_venv pyannote)"
  fi
}

sha256_of() {
  # shasum ships on macOS; sha256sum covers Linux (CI running the unit tests).
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

# True if something on 127.0.0.1:$1 accepts a TCP connection or leaves it
# hanging. Same watchdog probe as scripts/install.sh port_in_use() -- macOS
# drops the SYN silently on a full accept queue, so a plain /dev/tcp probe
# would hang and then misreport a wedged listener as free.
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
# Log rotation. launchd opened each service's StandardOutPath fd ONCE at
# bootstrap: an mv-style rotation would leave the running service writing
# into the archive's inode until the next restart. copytruncate (cp then
# truncate the live inode) matches that fd reality; the few bytes a service
# may write between the cp and the truncate are accepted losses for
# single-operator stdout logs. No lock file: the only invokers are cmd_up and
# the daily launchd job, a collision at worst duplicates an archive (pruned
# next round), while a stale lock would silently stop rotation forever.
# ---------------------------------------------------------------------------
rotate_log_file() {
  # $1 = live log path, $2 = max size in MB, $3 = newest archives to keep.
  # Below-threshold and missing logs are left untouched.
  local log=$1 max_mb=$2 keep=$3 size stamp archive
  [ -f "$log" ] || return 0
  size=$(wc -c < "$log")
  [ "$size" -ge $((max_mb * 1024 * 1024)) ] || return 0
  stamp=$(date +%Y-%m-%d-%H-%M-%S)
  archive=${log%.log}_$stamp.log
  cp "$log" "$archive" || return 1
  : > "$log"
  say "rotated $(basename "$log") ($size bytes) -> $(basename "$archive")"
  prune_log_archives "$log" "$keep"
}

prune_log_archives() {
  # $1 = live log path, $2 = newest archives to keep. The timestamp format
  # sorts lexicographically == chronologically, so `sort -r | tail +N` drops
  # exactly the oldest ones. macOS head has no negative -n; tail -n +K is
  # POSIX on both platforms.
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
  # Rotates the three service logs AND the rotation job's own log.
  local svc rc=0
  mkdir -p "$VOXINT_METAL_HOME/logs"
  for svc in $METAL_SERVICES logrotate; do
    rotate_log_file "$(service_log "$svc")" \
      "$VOXINT_METAL_LOG_MAX_MB" "$VOXINT_METAL_LOG_ARCHIVES" || rc=1
  done
  return $rc
}

render_logrotate_plist() {
  # $1 = output path. Daily one-shot (03:17 — an arbitrary quiet minute, off
  # the exact hour where periodic jobs pile up); launchd coalesces intervals
  # missed while the Mac slept into one run on wake. No KeepAlive: a clean
  # exit stays exited until the next calendar fire.
  local out=$1
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '<plist version="1.0">\n<dict>\n'
    printf '  <key>Label</key><string>%s</string>\n' "$(plist_label logrotate)"
    printf '  <key>ProgramArguments</key>\n  <array>\n'
    printf '    <string>/bin/bash</string>\n'
    printf '    <string>%s</string>\n' "$(xml_escape "$METAL_SCRIPT_DIR/voxint-metal.sh")"
    printf '    <string>rotate-logs</string>\n'
    printf '  </array>\n'
    printf '  <key>EnvironmentVariables</key>\n  <dict>\n'
    printf '    <key>VOXINT_METAL_HOME</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_METAL_HOME")"
    printf '    <key>VOXINT_METAL_LOG_MAX_MB</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_METAL_LOG_MAX_MB")"
    printf '    <key>VOXINT_METAL_LOG_ARCHIVES</key><string>%s</string>\n' \
      "$(xml_escape "$VOXINT_METAL_LOG_ARCHIVES")"
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

# ---------------------------------------------------------------------------
# MEDIA_ROOT -- the ONE path both worlds must agree on. The worker container
# mounts it as /data/media and sends MEDIA_ROOT-relative paths; the native
# services resolve those same relative paths against this directory. Always
# physically resolved (pwd -P): APFS /tmp -> /private/tmp symlinks and
# case-insensitive aliases would otherwise break the services' path contract.
# ---------------------------------------------------------------------------
env_value_from_file() {
  # $1 = KEY, $2 = env file. Last assignment wins, matching dotenv semantics.
  # Normalized exactly like install.sh read_env_value: trailing CR, surrounding
  # blanks, and ONE matched pair of single or double quotes are stripped — the
  # installer single-quotes MEDIA_ROOT (dotenv_squote), and Compose
  # interpolation sees the unquoted value, so a raw read here would disagree
  # with what the containers mount.
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

media_root_from_env_file() {
  env_value_from_file MEDIA_ROOT "$1"
}

image_tag_from_env_file() {
  env_value_from_file VOXINT_IMAGE_TAG "$1"
}

resolve_media_root() {
  # $1 = raw MEDIA_ROOT value from .env (may be relative to the repo root,
  # which is where compose resolves it). Echoes the physical absolute path.
  local raw=$1 dir
  case $raw in
    /*) dir=$raw ;;
    *)  dir=$REPO_ROOT/$raw ;;
  esac
  [ -d "$dir" ] || return 1
  (cd "$dir" && pwd -P)
}

# ---------------------------------------------------------------------------
# Per-service environment. ONE assembly point used by both the launchd plist
# generator and `run --foreground`, so the two paths cannot drift. Prints
# KEY=VALUE lines. launchd inherits no shell environment -- everything a
# service needs must be listed here explicitly.
# ---------------------------------------------------------------------------
# The pyannote device override exists for CPU A/B parity measurement ONLY.
# 'auto' (or anything else) is refused: auto would cascade to CPU when MPS is
# missing, silently -- exactly the honesty failure DIARIZER_DEVICE=mps exists
# to prevent. Called by cmd_up and cmd_run BEFORE env assembly, because a
# failure inside the plist render pipe would not abort the script.
validate_diarizer_override() {
  case ${VOXINT_METAL_DIARIZER_DEVICE:-mps} in
    mps|cpu) : ;;
    *) fail "VOXINT_METAL_DIARIZER_DEVICE must be 'mps' or 'cpu' (got '${VOXINT_METAL_DIARIZER_DEVICE}'): 'auto' would re-open silent CPU fallback" ;;
  esac
}

service_env() {
  local svc=$1 media_root=$2
  printf 'MEDIA_ROOT=%s\n' "$media_root"
  printf 'PYTHONUNBUFFERED=1\n'
  printf 'PATH=%s/bin:/usr/bin:/bin:/usr/sbin:/sbin\n' "$(service_venv "$svc")"
  case $svc in
    whisper)
      # v1 metal runs CT2 on CPU (int8, like the CPU image); the Metal ASR
      # engine is a tracked follow-up behind a pre-registered gate.
      printf 'DEVICE=cpu\n'
      printf 'WHISPER_MODEL=large-v2\n'
      printf 'COMPUTE_TYPE=int8\n'
      printf 'WHISPER_DOWNLOAD_ROOT=%s/models/whisper\n' "$VOXINT_METAL_HOME"
      # Pin the runtime load to the snapshot setup downloaded -- without it
      # faster-whisper resolves "latest", which could re-download a different
      # revision than the one the parity verdict measured.
      printf 'WHISPER_REVISION=%s\n' "$WHISPER_HF_REVISION"
      # Setup already downloaded the pinned snapshot; the running service
      # must never phone home (sha-pinned revisions resolve offline).
      printf 'HF_HUB_OFFLINE=1\n'
      ;;
    pyannote)
      printf 'VOXINT_VENDORED_PIPELINE=%s/models/pyannote/vendored/config.yaml\n' \
        "$VOXINT_METAL_HOME"
      # Forced, not auto: the tier exists to run diarization on the Apple
      # GPU, and DIARIZER_DEVICE=mps refuses to start if MPS is missing or
      # fails the tensor probe -- no silent CPU degradation. Override with
      # VOXINT_METAL_DIARIZER_DEVICE=cpu for A/B parity measurement.
      printf 'DIARIZER_DEVICE=%s\n' "${VOXINT_METAL_DIARIZER_DEVICE:-mps}"
      ;;
    titanet)
      printf 'EMBED_ENGINE=onnx\n'
      printf 'TITANET_ONNX_PATH=%s/models/titanet/titanet-large.onnx\n' "$VOXINT_METAL_HOME"
      # Unset/blank means the CPU EP default the parity verdict measured;
      # CoreML stays an explicit experiment (docs/gpu-contracts.md).
      if [ -n "${TITANET_ORT_PROVIDERS:-}" ]; then
        printf 'TITANET_ORT_PROVIDERS=%s\n' "$TITANET_ORT_PROVIDERS"
      fi
      ;;
    *) return 1 ;;
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
  # $1 = service, $2 = resolved MEDIA_ROOT, $3 = output path
  local svc=$1 media_root=$2 out=$3 venv port line key value env_block
  venv=$(service_venv "$svc")
  port=$(service_port "$svc")
  # Captured OUTSIDE the output block: a service_env failure inside the pipe
  # below would be swallowed by the while-subshell and ship a partial env
  # dict; here it aborts under set -e.
  env_block=$(service_env "$svc" "$media_root")
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '<plist version="1.0">\n<dict>\n'
    printf '  <key>Label</key><string>%s</string>\n' "$(plist_label "$svc")"
    printf '  <key>ProgramArguments</key>\n  <array>\n'
    printf '    <string>%s</string>\n' "$(xml_escape "$venv/bin/python")"
    printf '    <string>-m</string>\n    <string>uvicorn</string>\n'
    printf '    <string>app.main:app</string>\n'
    printf '    <string>--host</string>\n    <string>127.0.0.1</string>\n'
    printf '    <string>--port</string>\n    <string>%s</string>\n' "$port"
    printf '  </array>\n'
    printf '  <key>WorkingDirectory</key><string>%s</string>\n' \
      "$(xml_escape "$REPO_ROOT/services/$svc")"
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
# Weight verification -- the provenance JSONs are the single sha authority.
# ---------------------------------------------------------------------------
verify_pyannote_checkpoints() {
  # $1 = dir holding the two .bin files, $2 = provenance.json path.
  # Prints nothing on success; returns 1 with a message on mismatch.
  local dir=$1 prov=$2 name expected actual rc=0
  for name in segmentation-3.0.bin wespeaker-voxceleb-resnet34-LM.bin; do
    [ -f "$dir/$name" ] || { say "missing checkpoint: $dir/$name"; rc=1; continue; }
    expected=$("$(metal_python)" -c "import json,sys; print(json.load(open(sys.argv[1]))['files'][sys.argv[2]]['sha256'])" "$prov" "$name" 2>/dev/null) || expected=""
    # A broken venv/provenance must not masquerade as a weights mismatch --
    # the operator would re-download gigabytes to fix the wrong problem.
    [ "${#expected}" -eq 64 ] || { say "cannot read expected sha256 for $name from $prov (venv or provenance problem, not a weights mismatch)"; rc=1; continue; }
    actual=$(sha256_of "$dir/$name")
    if [ "$expected" != "$actual" ]; then
      say "sha256 mismatch for $name: expected $expected got $actual"
      rc=1
    fi
  done
  return $rc
}

verify_titanet_onnx() {
  # $1 = .onnx path, $2 = provenance.json path.
  local file=$1 prov=$2 expected actual
  [ -f "$file" ] || { say "missing ONNX graph: $file"; return 1; }
  expected=$("$(metal_python)" -c "import json,sys; print(json.load(open(sys.argv[1]))['onnx_sha256'])" "$prov" 2>/dev/null) || expected=""
  [ "${#expected}" -eq 64 ] || { say "cannot read expected sha256 from $prov (venv or provenance problem, not a weights mismatch)"; return 1; }
  actual=$(sha256_of "$file")
  if [ "$expected" != "$actual" ]; then
    say "sha256 mismatch for $(basename "$file"): expected $expected got $actual"
    return 1
  fi
}

whisper_manifest_path() { printf '%s/models/whisper/.voxint-manifest.sha256' "$1"; }

whisper_weights_ok() {
  # $1 = VOXINT_METAL_HOME. True when a NON-EMPTY manifest exists, its
  # revision header matches the pinned WHISPER_HF_REVISION, and every
  # recorded sha still verifies. Comment lines are stripped before shasum -c.
  local mh=$1 manifest
  manifest=$(whisper_manifest_path "$mh")
  [ -s "$manifest" ] || return 1
  head -1 "$manifest" | grep -qF "# revision: $WHISPER_HF_REVISION" || return 1
  grep -cv '^#' "$manifest" >/dev/null 2>&1 || return 1
  (cd "$mh/models/whisper" \
    && grep -v '^#' "$manifest" | shasum -a 256 -c - >/dev/null 2>&1)
}

write_whisper_manifest() {
  # $1 = VOXINT_METAL_HOME. Records the pinned revision plus per-file sha256s
  # of the payload, excluding hub bookkeeping (.cache/.locks -- lockfiles and
  # download metadata churn without the weights changing).
  local mh=$1 manifest
  manifest=$(whisper_manifest_path "$mh")
  (cd "$mh/models/whisper" && {
    printf '# revision: %s\n' "$WHISPER_HF_REVISION"
    find . \( -name '.cache' -o -name '.locks' \) -prune -o \
      -type f ! -name '.voxint-manifest.sha256' -exec shasum -a 256 {} +
  } > "$manifest")
}

# ---------------------------------------------------------------------------
# Vendored pyannote config generation. The checkpoints live under a
# "pyannote"-named directory ON PURPOSE: pyannote.audio 3.1.1 dispatches its
# embedding loader on path substrings, and a wespeaker-named path without
# "pyannote" routes to the uninstalled ONNX loader (contract-tested upstream;
# see services/pyannote/models/config.vendored.yaml).
# ---------------------------------------------------------------------------
generate_vendored_config() {
  # $1 = committed config.vendored.yaml, $2 = dest dir (checkpoints already
  # in $2/pyannote/). Writes $2/config.yaml, then verifies the result against
  # the provenance params -- a silent sed slip must not ship a config whose
  # numerics differ from the images'.
  local src=$1 dest=$2 prov=$3 dest_escaped nl
  nl=$(printf '\nx'); nl=${nl%x}  # $() strips trailing newlines; keep one
  case $dest in
    *'|'*) fail "VOXINT_METAL_HOME must not contain '|' (used as sed delimiter)" ;;
    *"$nl"*) fail "VOXINT_METAL_HOME must not contain a newline" ;;
  esac
  # In sed REPLACEMENT text '&' expands to the matched string and '\' starts
  # an escape -- a legal path like "/Volumes/Ben & Co" would silently corrupt
  # the rewritten checkpoint paths without this escaping.
  dest_escaped=$(printf '%s' "$dest" | sed -e 's/[&\\]/\\&/g')
  sed "s|/app/vendored/pyannote/|$dest_escaped/pyannote/|g" "$src" > "$dest/config.yaml"
  "$(metal_python)" - "$dest/config.yaml" "$prov" <<'PYEOF'
import json
import os
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
prov = json.load(open(sys.argv[2]))["pipeline_params"]
pp = cfg["pipeline"]["params"]
emb, seg = pp["embedding"], pp["segmentation"]


# Explicit raises, not assert: PYTHONOPTIMIZE would strip asserts and let an
# unverified config pass as "param-verified".
def check(ok: bool, msg: str) -> None:
    if not ok:
        raise SystemExit(f"vendored config verification failed: {msg}")


check(
    "pyannote" in os.path.dirname(emb),
    f"embedding path lost the load-bearing 'pyannote' substring: {emb}",
)
check(os.path.isfile(emb), f"embedding checkpoint missing: {emb}")
check(os.path.isfile(seg), f"segmentation checkpoint missing: {seg}")
checks = {
    "version": str(cfg["version"]),
    "pipeline_name": cfg["pipeline"]["name"],
    "clustering": pp["clustering"],
    "embedding_batch_size": pp["embedding_batch_size"],
    "embedding_exclude_overlap": pp["embedding_exclude_overlap"],
    "segmentation_batch_size": pp["segmentation_batch_size"],
    "clustering_method": cfg["params"]["clustering"]["method"],
    "clustering_min_cluster_size": cfg["params"]["clustering"]["min_cluster_size"],
    "clustering_threshold": cfg["params"]["clustering"]["threshold"],
    "segmentation_min_duration_off": cfg["params"]["segmentation"]["min_duration_off"],
}
for key, value in checks.items():
    check(
        str(prov[key]) == str(value),
        f"{key}: generated {value!r} != provenance {prov[key]!r}",
    )
PYEOF
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
require_macos() {
  [ "$(uname -s)" = "Darwin" ] || fail "the metal tier is macOS-only (this is $(uname -s))"
  [ "$(uname -m)" = "arm64" ] || fail "the metal tier needs Apple Silicon (this is $(uname -m))"
}

require_tools() {
  command -v uv >/dev/null 2>&1 || fail "uv is required (https://docs.astral.sh/uv/ or: brew install uv)"
  command -v curl >/dev/null 2>&1 || fail "curl is required"
}

fetch_asset() {
  # $1 = release tag, $2 = asset name, $3 = output path
  local url="https://github.com/$GITHUB_REPO/releases/download/$1/$2"
  say "  fetching $2 from $1 ..."
  curl -fL --retry 3 -o "$3" "$url" || fail "download failed: $url"
}

cmd_setup() {
  require_macos
  require_tools
  local mh=$VOXINT_METAL_HOME svc venv reqs

  step "Directories under $mh"
  mkdir -p "$mh/venvs" "$mh/models/whisper" "$mh/models/pyannote/vendored/pyannote" \
    "$mh/models/titanet" "$mh/logs" "$mh/run"

  step "Per-service Python 3.11 venvs (uv)"
  for svc in $METAL_SERVICES; do
    venv=$(service_venv "$svc")
    reqs=$REPO_ROOT/$(service_requirements "$svc")
    if [ ! -x "$venv/bin/python" ]; then
      say "  creating venv: $venv"
      uv venv --python 3.11 "$venv" >&2
    fi
    say "  installing $(basename "$reqs") into $svc venv"
    uv pip install --quiet --python "$venv/bin/python" -r "$reqs" >&2
  done

  step "pyannote checkpoints ($PYANNOTE_MODELS_RELEASE, sha-verified)"
  local pdir=$mh/models/pyannote/vendored/pyannote
  local pprov=$REPO_ROOT/services/pyannote/models/provenance.json
  if ! verify_pyannote_checkpoints "$pdir" "$pprov" 2>/dev/null; then
    fetch_asset "$PYANNOTE_MODELS_RELEASE" segmentation-3.0.bin "$pdir/segmentation-3.0.bin"
    fetch_asset "$PYANNOTE_MODELS_RELEASE" wespeaker-voxceleb-resnet34-LM.bin \
      "$pdir/wespeaker-voxceleb-resnet34-LM.bin"
    verify_pyannote_checkpoints "$pdir" "$pprov" \
      || fail "downloaded pyannote checkpoints do not match the committed provenance sha256s"
  fi
  say "  checkpoints verified against provenance.json"

  step "Vendored pipeline config (generated, param-verified)"
  generate_vendored_config \
    "$REPO_ROOT/services/pyannote/models/config.vendored.yaml" \
    "$mh/models/pyannote/vendored" "$pprov"
  say "  $mh/models/pyannote/vendored/config.yaml OK"

  step "TitaNet ONNX graph ($TITANET_ONNX_RELEASE, sha-verified)"
  local tfile=$mh/models/titanet/titanet-large.onnx
  local tprov=$REPO_ROOT/tests/parity/fixtures/onnx/provenance.json
  if ! verify_titanet_onnx "$tfile" "$tprov" 2>/dev/null; then
    fetch_asset "$TITANET_ONNX_RELEASE" titanet-large.onnx "$tfile"
    verify_titanet_onnx "$tfile" "$tprov" \
      || fail "downloaded titanet-large.onnx does not match the committed provenance sha256"
  fi
  say "  ONNX graph verified against provenance.json"

  step "Whisper large-v2 (pinned HF revision ${WHISPER_HF_REVISION})"
  if whisper_weights_ok "$mh"; then
    say "  already present and matching the local manifest"
  else
    # A stale or corrupted cache must be CLEARED before re-downloading:
    # huggingface_hub trusts existing blobs (no content re-hash), so a
    # re-download over a corrupt tree would no-op and the fresh manifest
    # would bless the corruption.
    if [ -n "$(ls -A "$mh/models/whisper" 2>/dev/null)" ]; then
      say "  cache missing, stale, or corrupt -- clearing and re-downloading"
      rm -rf "$mh/models/whisper"
      mkdir -p "$mh/models/whisper"
    fi
    "$(service_venv whisper)/bin/python" -c "
from faster_whisper import WhisperModel
WhisperModel('$WHISPER_MODEL_REPO', device='cpu',
             download_root='$mh/models/whisper',
             revision='$WHISPER_HF_REVISION')
" >&2 || fail "whisper model download failed"
    # No committed sha authority exists for the HF blobs (revision-pinned
    # instead); record what we downloaded so later drift is detectable.
    # The revision header binds the manifest to the pin; hub bookkeeping
    # (.cache/.locks metadata) is excluded -- it is not stable content.
    write_whisper_manifest "$mh" || fail "could not record the whisper manifest"
    say "  downloaded and recorded to the local manifest"
  fi

  step "Setup complete"
  say "Next: $0 up   (then: $0 status)"
  say "Note: submissions through the console will fail until the services are up."
}

# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------
resolved_media_root_or_fail() {
  local envf=$REPO_ROOT/.env raw resolved
  [ -f "$envf" ] || fail "no .env at $envf -- run scripts/install.sh first (option [M])"
  raw=$(media_root_from_env_file "$envf")
  [ -n "$raw" ] || fail ".env has no MEDIA_ROOT line"
  resolved=$(resolve_media_root "$raw") \
    || fail "MEDIA_ROOT '$raw' (from .env) does not resolve to an existing directory"
  printf '%s' "$resolved"
}

cmd_up() {
  require_macos
  validate_diarizer_override
  local media_root svc plist label i
  media_root=$(resolved_media_root_or_fail)
  # Preflight the setup artifacts: bootstrapping a service with a missing
  # venv or weights would crash-loop under launchd KeepAlive with only a
  # cryptic log to show for it -- fail fast with the actual remedy instead.
  mkdir -p "$VOXINT_METAL_HOME/run" "$VOXINT_METAL_HOME/logs"
  for svc in $METAL_SERVICES; do
    [ -x "$(service_venv "$svc")/bin/python" ] \
      || fail "$svc venv missing -- run: $0 setup"
  done
  [ -f "$VOXINT_METAL_HOME/models/pyannote/vendored/config.yaml" ] \
    || fail "vendored pyannote config missing -- run: $0 setup"
  [ -f "$VOXINT_METAL_HOME/models/titanet/titanet-large.onnx" ] \
    || fail "titanet ONNX graph missing -- run: $0 setup"
  whisper_weights_ok "$VOXINT_METAL_HOME" \
    || fail "whisper weights missing or failing the local manifest -- run: $0 setup"
  # Rotate oversized logs now AND (re)install the daily rotation job — under
  # KeepAlive the services can run for months without another `up`, so the
  # inline rotation alone would not bound log growth.
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
  say "installed $label (daily, keeps $VOXINT_METAL_LOG_ARCHIVES archives over ${VOXINT_METAL_LOG_MAX_MB}MB)"
  for svc in $METAL_SERVICES; do
    plist=$(plist_path "$svc")
    label=$(plist_label "$svc")
    render_plist "$svc" "$media_root" "$plist"
    plutil -lint -s "$plist" || fail "generated plist failed plutil lint: $plist"
    # Idempotent restart: bootout is a no-op error when not loaded. It also
    # RETURNS BEFORE the job is fully unloaded -- an immediate bootstrap
    # races it and fails with EIO -- so wait (bounded) for the unload.
    launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1 || true
    i=0
    while launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; do
      i=$((i + 1))
      [ "$i" -le 50 ] || fail "$label did not unload within 10s (launchctl bootout race)"
      sleep 0.2
    done
    launchctl bootstrap "gui/$(id -u)" "$plist" \
      || fail "launchctl bootstrap failed for $label (see $(service_log "$svc"))"
    say "started $label (port $(service_port "$svc"))"
  done
  say "All three services starting. First pyannote MPS run pays a ~10s Metal"
  say "shader warm-up. Check readiness with: $0 status"
}

cmd_down() {
  local svc label rc=0
  for svc in $METAL_SERVICES logrotate; do
    label=$(plist_label "$svc")
    if launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1; then
      say "stopped $label"
    else
      say "$label was not running"
    fi
  done
  return $rc
}

cmd_status() {
  local svc port label state health tag envf
  step "Native model services"
  for svc in $METAL_SERVICES; do
    port=$(service_port "$svc")
    label=$(plist_label "$svc")
    if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
      state="supervised"
    else
      state="NOT loaded"
    fi
    health=$(curl -fsS -m 3 "http://127.0.0.1:$port/healthz" 2>/dev/null) \
      || health="unreachable"
    printf '%-9s %-11s :%s  %s\n' "$svc" "[$state]" "$port" "$health"
  done

  step "Version skew"
  # Native services run the WORKING TREE; the core stack runs pinned images.
  # A mismatch is not an error, but the operator should know about it.
  envf=$REPO_ROOT/.env
  tag=""
  [ -f "$envf" ] && tag=$(image_tag_from_env_file "$envf")
  [ -n "$tag" ] || tag="(compose default)"
  say "working tree: $(git -C "$REPO_ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)"
  say "core images:  VOXINT_IMAGE_TAG=$tag"
  say "Pair a tagged working tree with the matching image tag for supported runs."
}

cmd_logs() {
  local svc=${1:-} follow=${2:-}
  [ -n "$svc" ] || fail "usage: $0 logs <whisper|pyannote|titanet|logrotate> [-f]"
  [ "$svc" = "logrotate" ] || service_port "$svc" >/dev/null \
    || fail "unknown service: $svc"
  if [ "$follow" = "-f" ]; then
    # -F, not -f: rotation truncates in place (same inode), but a manually
    # deleted/recreated log would strand a plain -f follower silently.
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
  local mh=$VOXINT_METAL_HOME svc venv port

  step "Tooling"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    doctor_report PASS "macOS on Apple Silicon"
  else
    doctor_report FAIL "not macOS/arm64 ($(uname -s)/$(uname -m)) -- the metal tier cannot run here"
  fi
  command -v uv >/dev/null 2>&1 \
    && doctor_report PASS "uv present" || doctor_report FAIL "uv missing"

  step "Venvs + weights"
  for svc in $METAL_SERVICES; do
    venv=$(service_venv "$svc")
    [ -x "$venv/bin/python" ] \
      && doctor_report PASS "$svc venv" \
      || doctor_report FAIL "$svc venv missing ($venv) -- run: $0 setup"
  done
  if verify_pyannote_checkpoints "$mh/models/pyannote/vendored/pyannote" \
      "$REPO_ROOT/services/pyannote/models/provenance.json" >/dev/null 2>&1; then
    doctor_report PASS "pyannote checkpoints match provenance sha256s"
  else
    doctor_report FAIL "pyannote checkpoints missing or sha-mismatched -- run: $0 setup"
  fi
  if verify_titanet_onnx "$mh/models/titanet/titanet-large.onnx" \
      "$REPO_ROOT/tests/parity/fixtures/onnx/provenance.json" >/dev/null 2>&1; then
    doctor_report PASS "titanet ONNX graph matches provenance sha256"
  else
    doctor_report FAIL "titanet ONNX graph missing or sha-mismatched -- run: $0 setup"
  fi
  if [ -f "$mh/models/pyannote/vendored/config.yaml" ]; then
    doctor_report PASS "vendored pyannote config present"
  else
    doctor_report FAIL "vendored pyannote config missing -- run: $0 setup"
  fi
  if whisper_weights_ok "$mh"; then
    doctor_report PASS "whisper weights match the local manifest (revision $WHISPER_HF_REVISION)"
  else
    doctor_report FAIL "whisper weights missing, stale, or failing the local manifest -- run: $0 setup"
  fi

  step "MEDIA_ROOT agreement (.env vs native services)"
  local envf=$REPO_ROOT/.env raw resolved
  if [ -f "$envf" ]; then
    raw=$(media_root_from_env_file "$envf")
    if [ -n "$raw" ] && resolved=$(resolve_media_root "$raw"); then
      doctor_report PASS "MEDIA_ROOT=$raw resolves to $resolved (physical)"
    else
      doctor_report FAIL "MEDIA_ROOT '$raw' from .env does not resolve to a directory"
    fi
  else
    doctor_report FAIL "no .env at $envf -- run scripts/install.sh (option [M]) first"
  fi

  step "Ports (native services bind 127.0.0.1 only)"
  for svc in $METAL_SERVICES; do
    port=$(service_port "$svc")
    if curl -fsS -m 3 "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      doctor_report PASS "$svc responding on :$port"
    elif port_in_use "$port"; then
      doctor_report FAIL ":$port is occupied by something that is not $svc /healthz -- a leftover cpu-tier container stack? (docker ps)"
    else
      doctor_report SKIP "$svc not running (:$port free) -- $0 up"
    fi
  done

  step "MPS (pyannote venv)"
  venv=$(service_venv pyannote)
  if [ -x "$venv/bin/python" ]; then
    if "$venv/bin/python" - <<'PYEOF' >/dev/null 2>&1
import torch

assert torch.backends.mps.is_available(), "MPS backend not available"
gen = torch.Generator().manual_seed(0)
x = torch.randn(64, 64, generator=gen)
ref = x @ x
out = (x.to("mps") @ x.to("mps")).cpu()
assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), "MPS computed wrong output"
PYEOF
    then
      doctor_report PASS "MPS available and passes the tensor-op probe"
    else
      doctor_report FAIL "MPS unavailable or failing the tensor-op probe -- pyannote (DIARIZER_DEVICE=mps) will refuse to start"
    fi
  else
    doctor_report SKIP "pyannote venv missing; cannot probe MPS"
  fi

  step "ONNX Runtime providers (titanet venv)"
  venv=$(service_venv titanet)
  if [ -x "$venv/bin/python" ]; then
    if "$venv/bin/python" -c "
import os
import onnxruntime as ort

available = ort.get_available_providers()
assert 'CPUExecutionProvider' in available, available
requested = [p.strip() for p in os.getenv('TITANET_ORT_PROVIDERS', '').split(',') if p.strip()]
missing = [p for p in requested if p not in available]
assert not missing, f'TITANET_ORT_PROVIDERS requests {missing}; available: {available}'
print(','.join(available))
" >/dev/null 2>&1; then
      doctor_report PASS "onnxruntime providers OK for current TITANET_ORT_PROVIDERS"
    else
      doctor_report FAIL "onnxruntime provider check failed (TITANET_ORT_PROVIDERS=${TITANET_ORT_PROVIDERS:-unset})"
    fi
  else
    doctor_report SKIP "titanet venv missing; cannot check ORT providers"
  fi

  step "Worker -> host loopback (Docker Desktop specific)"
  if command -v docker >/dev/null 2>&1 \
      && docker compose -f "$REPO_ROOT/compose.yaml" ps --status running worker 2>/dev/null | grep -q worker; then
    if docker compose -f "$REPO_ROOT/compose.yaml" exec -T worker \
        python -c "import httpx; httpx.get('http://host.docker.internal:8024/healthz', timeout=5)" \
        >/dev/null 2>&1; then
      doctor_report PASS "worker container reaches host.docker.internal"
    else
      doctor_report FAIL "worker cannot reach host.docker.internal -- non-Docker-Desktop engine? Native services are only reachable from Docker Desktop's loopback proxy"
    fi
  else
    doctor_report SKIP "core stack not running; loopback check skipped"
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
  local svc=${1:-} mode=${2:-} media_root line
  [ -n "$svc" ] || fail "usage: $0 run <whisper|pyannote|titanet> --foreground"
  service_port "$svc" >/dev/null || fail "unknown service: $svc"
  [ "$mode" = "--foreground" ] \
    || fail "only --foreground is supported (background runs go through: $0 up)"
  validate_diarizer_override
  media_root=$(resolved_media_root_or_fail)
  # Same env assembly the plists use -- the debug path may not drift.
  while IFS= read -r line; do
    export "${line?}"
  done <<EOF
$(service_env "$svc" "$media_root")
EOF
  cd "$REPO_ROOT/services/$svc"
  exec "$(service_venv "$svc")/bin/python" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$(service_port "$svc")"
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
    rotate-logs) cmd_rotate_logs "$@" ;;
    *)
      say "voxint-metal.sh -- native model services for the metal tier"
      say "usage: $0 <setup|up|down|status|logs|doctor|run|rotate-logs>"
      say "  setup                 venvs + weights (network required)"
      say "  up / down             start/stop under launchd"
      say "  status                healthz + supervision + version skew"
      say "  logs <svc> [-f]       show/follow a service log"
      say "  doctor                environment checks"
      say "  run <svc> --foreground  debug one service in the foreground"
      say "  rotate-logs           copytruncate-rotate oversized service logs"
      say "                        (also runs daily via launchd once 'up' has run)"
      exit 1
      ;;
  esac
}

if [ "${VOXINT_METAL_LIB:-}" != "1" ]; then
  main "$@"
fi
