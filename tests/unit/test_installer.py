"""Offline tests for scripts/install.sh via its library-mode seam.

The installer sources with ``VOXINT_INSTALL_LIB=1`` (no main), which lets these
tests exercise the pure-shell logic — tier→Compose-file mapping, port-collision
handling, .env rendering/updating, and the HF-token preflight — without a
Docker daemon or network. Every test runs in a throwaway fixture "repo" (the
script resolves ``REPO_ROOT`` from its own path, so copying it into tmp_path
fully isolates the real checkout's .env), with fake ``docker`` / ``curl``
executables on PATH that log their argv.
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import pytest

REAL_REPO = Path(__file__).resolve().parents[2]

FAKE_DOCKER = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${FAKE_DOCKER_LOG:?}"
exit 0
"""

# Emits an http_code like curl -w '%{http_code}' would; drains stdin (the
# -K - config carrying the Authorization header) so the pipe never breaks.
# FAKE_CURL_CODES (a file of one code per line, consumed head-first) lets a
# test sequence per-call responses (whoami vs each gated repo); otherwise
# every call answers ${FAKE_CURL_CODE:-200}.
FAKE_CURL = """#!/usr/bin/env bash
cat > /dev/null
printf '%s\\n' "$*" >> "${FAKE_CURL_LOG:?}"
if [ -n "${FAKE_CURL_CODES:-}" ] && [ -s "${FAKE_CURL_CODES}" ]; then
  code=$(head -n1 "$FAKE_CURL_CODES")
  tail -n +2 "$FAKE_CURL_CODES" > "$FAKE_CURL_CODES.tmp"
  mv "$FAKE_CURL_CODES.tmp" "$FAKE_CURL_CODES"
else
  code="${FAKE_CURL_CODE:-200}"
fi
printf '%s' "$code"
exit 0
"""


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(REAL_REPO / "scripts" / "install.sh", tmp_path / "scripts" / "install.sh")
    shutil.copy(REAL_REPO / ".env.example", tmp_path / ".env.example")
    for name in ("compose.yaml", "compose.cpu.yaml", "compose.gpu.yaml"):
        shutil.copy(REAL_REPO / name, tmp_path / name)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for exe, body in (("docker", FAKE_DOCKER), ("curl", FAKE_CURL)):
        p = fakebin / exe
        p.write_text(body)
        p.chmod(0o755)
    return tmp_path


def run_lib(
    repo: Path,
    script: str,
    stdin: str = "",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VOXINT_INSTALL_LIB"] = "1"
    env["PATH"] = f"{repo / 'fakebin'}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(repo / "docker.log")
    env["FAKE_CURL_LOG"] = str(repo / "curl.log")
    if extra_env:
        env.update(extra_env)
    full = f'source "{repo}/scripts/install.sh"\n{script}'
    return subprocess.run(
        ["bash", "-c", full],
        cwd=repo,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


# --------------------------------------------------------------------------- #
# Tier -> Compose file mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("cpu", "-f compose.yaml -f compose.cpu.yaml"),
        ("gpu", "-f compose.yaml -f compose.gpu.yaml"),
        ("none", "-f compose.yaml"),
        ("junk", "-f compose.yaml"),
        ("", "-f compose.yaml"),
    ],
)
def test_compose_file_args_for_tier(repo: Path, tier: str, expected: str) -> None:
    proc = run_lib(repo, f'compose_file_args_for_tier "{tier}"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("cpu", "cpu"), ("gpu", "gpu"), ("none", "none"), ("CPU", ""), ("banana", ""), ("", "")],
)
def test_normalize_tier(repo: Path, value: str, expected: str) -> None:
    proc = run_lib(repo, f'normalize_tier "{value}"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


def test_dc_routes_overlay_args_to_docker(repo: Path) -> None:
    proc = run_lib(
        repo,
        'COMPOSE_FILE_ARGS=$(compose_file_args_for_tier cpu)\ndc config --quiet',
    )
    assert proc.returncode == 0, proc.stderr
    log = (repo / "docker.log").read_text()
    assert "compose -f compose.yaml -f compose.cpu.yaml config --quiet" in log


# --------------------------------------------------------------------------- #
# Port collision handling (issue #21)
# --------------------------------------------------------------------------- #
def test_port_collision_detected_and_alternative_offered(repo: Path) -> None:
    # Occupy a real port, then ask resolve_port about it: the collision must be
    # announced and the suggested free alternative accepted via a bare Enter.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        proc = run_lib(repo, f'resolve_port "API / web console" {busy}', stdin="\n")
    assert proc.returncode == 0, proc.stderr
    assert f"Host port {busy} (API / web console) is already in use" in proc.stderr
    chosen = int(proc.stdout)
    assert chosen != busy
    assert 1 <= chosen <= 65535


def test_free_port_passes_through_without_prompting(repo: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free = sock.getsockname()[1]
    # Socket closed -> port free again. No stdin provided: a prompt would fail.
    proc = run_lib(repo, f'resolve_port "API / web console" {free}')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == str(free)
    assert "already in use" not in proc.stderr


def test_port_in_use_true_for_bound_port(repo: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        proc = run_lib(repo, f"port_in_use {busy} && echo USED || echo FREE")
    assert proc.stdout.strip() == "USED"


# --------------------------------------------------------------------------- #
# .env rendering (generate path)
# --------------------------------------------------------------------------- #
WRITE_ENV_PREAMBLE = (
    "PASSWORD='pw-for-test'\n"
    "MEDIA_ROOT_VALUE='/tmp/voxint-media'\n"
    "CSRF_VALUE='deadbeef'\n"
)


def test_write_env_records_tier_and_token(repo: Path) -> None:
    token = "hf_TESTSECRETTOKEN123"
    proc = run_lib(
        repo,
        WRITE_ENV_PREAMBLE
        + f"HF_TOKEN_VALUE='{token}'\n"
        "COMPUTE_TIER_VALUE=cpu\n"
        "COMPOSE_FILE_ARGS=$(compose_file_args_for_tier cpu)\n"
        "write_env",
    )
    assert proc.returncode == 0, proc.stderr
    env_file = repo / ".env"
    content = env_file.read_text()
    assert f"HF_TOKEN='{token}'" in content
    assert "VOXINT_COMPOSE_TIER=cpu" in content
    assert "VOXINT_PASSWORD='pw-for-test'" in content
    # mode 0600
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    # The secret never reaches stdout/stderr.
    assert token not in proc.stdout
    assert token not in proc.stderr
    # Compose validation ran against the EFFECTIVE (cpu) file set.
    log = (repo / "docker.log").read_text()
    assert "-f compose.cpu.yaml" in log


def test_write_env_without_token_keeps_template_line(repo: Path) -> None:
    proc = run_lib(
        repo,
        WRITE_ENV_PREAMBLE + "COMPUTE_TIER_VALUE=none\nwrite_env",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "HF_TOKEN=\n" in content  # untouched template line
    assert "VOXINT_COMPOSE_TIER=none" in content


# --------------------------------------------------------------------------- #
# Kept-.env update path (legacy migration)
# --------------------------------------------------------------------------- #
LEGACY_ENV = "VOXINT_PASSWORD='old-pw'\nMEDIA_ROOT='/data'\nHF_TOKEN=\n"


def test_update_env_keys_appends_tier_and_backs_up(repo: Path) -> None:
    (repo / ".env").write_text(LEGACY_ENV)
    proc = run_lib(repo, "COMPUTE_TIER_VALUE=gpu\nHF_TOKEN_VALUE=\nupdate_env_keys")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "VOXINT_COMPOSE_TIER=gpu" in content
    assert "VOXINT_PASSWORD='old-pw'" in content  # untouched passthrough
    assert "HF_TOKEN=\n" in content  # not overwritten when no new token given
    backups = list(repo.glob(".env.backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == LEGACY_ENV


def test_update_env_keys_replaces_existing_tier_and_token(repo: Path) -> None:
    (repo / ".env").write_text(LEGACY_ENV + "VOXINT_COMPOSE_TIER=none\n")
    token = "hf_NEWSECRET456"
    proc = run_lib(
        repo,
        f"COMPUTE_TIER_VALUE=cpu\nHF_TOKEN_VALUE='{token}'\nupdate_env_keys",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert content.count("VOXINT_COMPOSE_TIER=") == 1
    assert "VOXINT_COMPOSE_TIER=cpu" in content
    assert f"HF_TOKEN='{token}'" in content
    assert "HF_TOKEN=\n" not in content
    assert token not in proc.stdout
    assert token not in proc.stderr
    assert stat.S_IMODE((repo / ".env").stat().st_mode) == 0o600


def test_read_env_value_last_wins_and_strips_quotes(repo: Path) -> None:
    (repo / ".env").write_text("HF_TOKEN='first'\nHF_TOKEN='second'\n")
    proc = run_lib(repo, "read_env_value HF_TOKEN")
    assert proc.stdout == "second"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('HF_TOKEN="hf_dq"', "hf_dq"),  # double quotes stripped like Compose does
        ('HF_TOKEN=""', ""),  # empty double-quoted value reads EMPTY (defers tier)
        ("HF_TOKEN=''", ""),
        ("HF_TOKEN=hf_plain", "hf_plain"),
        ("HF_TOKEN=hf_crlf\r", "hf_crlf"),  # CRLF-edited .env
        ("HF_TOKEN=  hf_pad  ", "hf_pad"),  # surrounding blanks trimmed
        ("HF_TOKEN='", "'"),  # lone quote is a literal, not a pair
    ],
)
def test_read_env_value_normalization(repo: Path, raw: str, expected: str) -> None:
    (repo / ".env").write_text(raw + "\n")
    proc = run_lib(repo, "read_env_value HF_TOKEN")
    assert proc.stdout == expected


def test_update_env_keys_dedupes_duplicate_key_lines(repo: Path) -> None:
    (repo / ".env").write_text(
        "VOXINT_PASSWORD='pw'\nHF_TOKEN='a'\nHF_TOKEN='b'\nVOXINT_COMPOSE_TIER=gpu\n"
    )
    proc = run_lib(repo, "COMPUTE_TIER_VALUE=cpu\nHF_TOKEN_VALUE='hf_new'\nupdate_env_keys")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert content.count("HF_TOKEN=") == 1
    assert "HF_TOKEN='hf_new'" in content
    assert content.count("VOXINT_COMPOSE_TIER=") == 1
    # Validation ran against the recorded tier's overlay (token present).
    log = (repo / "docker.log").read_text()
    assert "-f compose.cpu.yaml" in log


# --------------------------------------------------------------------------- #
# HF token preflight (advisory; secret hygiene)
# --------------------------------------------------------------------------- #
def test_hf_preflight_valid_token_never_puts_secret_in_argv(repo: Path) -> None:
    token = "hf_PREFLIGHTSECRET789"
    proc = run_lib(
        repo,
        f"PREFLIGHT_TOKEN='{token}'\nhf_preflight",
        extra_env={"FAKE_CURL_CODE": "200"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Token is valid" in proc.stderr
    assert "Gated model access looks good" in proc.stderr
    # The token reaches curl via stdin config only — argv must be clean.
    curl_log = (repo / "curl.log").read_text()
    assert token not in curl_log
    assert token not in proc.stdout
    assert token not in proc.stderr


def test_hf_preflight_rejected_token_warns_but_does_not_fail(repo: Path) -> None:
    proc = run_lib(
        repo,
        "PREFLIGHT_TOKEN='hf_bad'\nhf_preflight\necho SURVIVED",
        extra_env={"FAKE_CURL_CODE": "401"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "rejected the token" in proc.stderr
    assert "SURVIVED" in proc.stdout


def test_hf_preflight_network_failure_is_nonfatal(repo: Path) -> None:
    proc = run_lib(
        repo,
        "PREFLIGHT_TOKEN='hf_x'\nhf_preflight\necho SURVIVED",
        extra_env={"FAKE_CURL_CODE": "000"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Could not verify the token" in proc.stderr
    assert "SURVIVED" in proc.stdout


def test_hf_preflight_gate_not_accepted_warns_per_repo(repo: Path) -> None:
    # whoami 200, first gated repo 403, second 200: warn about the first repo
    # only, never claim access looks good, and never fail the install.
    codes = repo / "curl.codes"
    codes.write_text("200\n403\n200\n")
    proc = run_lib(
        repo,
        "PREFLIGHT_TOKEN='hf_x'\nhf_preflight\necho SURVIVED",
        extra_env={"FAKE_CURL_CODES": str(codes)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Token is valid" in proc.stderr
    assert "cannot access pyannote/speaker-diarization-3.1" in proc.stderr
    assert "Gated model access looks good" not in proc.stderr
    assert "SURVIVED" in proc.stdout


def test_hf_preflight_skipped_without_token(repo: Path) -> None:
    proc = run_lib(repo, "PREFLIGHT_TOKEN=''\nhf_preflight\necho SURVIVED")
    assert proc.returncode == 0, proc.stderr
    assert "SURVIVED" in proc.stdout
    assert not (repo / "curl.log").exists()
