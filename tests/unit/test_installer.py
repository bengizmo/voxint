"""Offline tests for scripts/install.sh via its library-mode seam.

The installer sources with ``VOXINT_INSTALL_LIB=1`` (no main), which lets these
tests exercise the pure-shell logic — tier→Compose-file mapping, port-collision
handling, .env rendering/updating, and render-gid detection — without a Docker
daemon or network. Every test runs in a throwaway fixture "repo" (the script
resolves ``REPO_ROOT`` from its own path, so copying it into tmp_path fully
isolates the real checkout's .env), with a fake ``docker`` executable on PATH
that logs its argv.
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

@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(REAL_REPO / "scripts" / "install.sh", tmp_path / "scripts" / "install.sh")
    shutil.copy(REAL_REPO / ".env.example", tmp_path / ".env.example")
    for name in (
        "compose.yaml",
        "compose.cpu.yaml",
        "compose.gpu.yaml",
        "compose.rocm.yaml",
        "compose.metal.yaml",
    ):
        shutil.copy(REAL_REPO / name, tmp_path / name)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    p = fakebin / "docker"
    p.write_text(FAKE_DOCKER)
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
        ("rocm", "-f compose.yaml -f compose.rocm.yaml"),
        ("metal", "-f compose.yaml -f compose.metal.yaml"),
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
    [
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        ("rocm", "rocm"),
        ("metal", "metal"),
        ("none", "none"),
        ("CPU", ""),
        ("ROCM", ""),
        ("METAL", ""),
        ("banana", ""),
        ("", ""),
    ],
)
def test_normalize_tier(repo: Path, value: str, expected: str) -> None:
    proc = run_lib(repo, f'normalize_tier "{value}"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


def test_prompt_compute_tier_accepts_metal(repo: Path) -> None:
    proc = run_lib(
        repo, 'prompt_compute_tier\nprintf %s "$COMPUTE_TIER_VALUE"', stdin="m\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "metal"
    assert "[M] Apple tier" in proc.stderr


def test_prompt_compute_tier_defaults_to_metal_on_apple_silicon(repo: Path) -> None:
    # A fake uname on the fixture PATH models an Apple Silicon host, so this
    # runs (and stays meaningful) on Linux CI too. Bare Enter must take the
    # suggested default.
    fake_uname = repo / "fakebin" / "uname"
    fake_uname.write_text(
        "#!/usr/bin/env bash\n"
        'case "${1:-}" in -s) echo Darwin ;; -m) echo arm64 ;; *) echo Darwin ;; esac\n'
    )
    fake_uname.chmod(0o755)
    proc = run_lib(
        repo, 'prompt_compute_tier\nprintf %s "$COMPUTE_TIER_VALUE"', stdin="\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "metal"
    assert "[M/g/a/c/n]" in proc.stderr


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


def test_resolve_port_never_offers_busy_default_on_flaky_probe(
    repo: Path, tmp_path: Path
) -> None:
    # Regression for #27's real failure mode. A real bound socket reads busy
    # *consistently* on Linux, so it can't tell the fix from the pre-fix code
    # (both then scan to def+1). The actual macOS/BSD bug is an INCONSISTENT
    # probe: a backlog-full listener refuses the re-probe, so port_in_use flips
    # the busy default to "free" the second time it is asked. We model exactly
    # that with a stub — busy on the first probe of $BUSY_PORT, free after — which
    # the pre-fix `next_free_port "$def"` re-scan would hand right back as the
    # "alternate" (chosen == busy), and which the fix (`next_free_port def+1`)
    # sidesteps by never re-probing the known-busy default.
    marker = tmp_path / "probe_marker"  # must not exist yet
    busy = 8080
    script = (
        "port_in_use() {\n"
        '  if [ "$1" = "$BUSY_PORT" ] && [ ! -e "$PROBE_MARKER" ]; then\n'
        '    : > "$PROBE_MARKER"; return 0\n'  # busy: first probe of the default
        "  fi\n"
        "  return 1\n"  # free: the re-probe of the default, and every other port
        "}\n"
        'resolve_port "API / web console" "$BUSY_PORT"'
    )
    proc = run_lib(
        repo,
        script,
        stdin="\n",  # accept the suggested alternate
        extra_env={"BUSY_PORT": str(busy), "PROBE_MARKER": str(marker)},
    )
    assert proc.returncode == 0, proc.stderr
    assert f"Host port {busy} (API / web console) is already in use" in proc.stderr
    chosen = int(proc.stdout)
    assert chosen != busy, "must never offer the known-busy default back as the alternate"
    assert 1 <= chosen <= 65535


# --------------------------------------------------------------------------- #
# .env rendering (generate path)
# --------------------------------------------------------------------------- #
WRITE_ENV_PREAMBLE = (
    "PASSWORD='pw-for-test'\n"
    "MEDIA_ROOT_VALUE='/tmp/voxint-media'\n"
    "CSRF_VALUE='deadbeef'\n"
)


def test_write_env_records_tier(repo: Path) -> None:
    proc = run_lib(
        repo,
        WRITE_ENV_PREAMBLE
        + "COMPUTE_TIER_VALUE=cpu\n"
        "COMPOSE_FILE_ARGS=$(compose_file_args_for_tier cpu)\n"
        "write_env",
    )
    assert proc.returncode == 0, proc.stderr
    env_file = repo / ".env"
    content = env_file.read_text()
    assert "VOXINT_COMPOSE_TIER=cpu" in content
    assert "VOXINT_PASSWORD='pw-for-test'" in content
    # mode 0600
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    # Compose validation ran against the EFFECTIVE (cpu) file set.
    log = (repo / "docker.log").read_text()
    assert "-f compose.cpu.yaml" in log


def test_write_env_keeps_hf_token_template_line(repo: Path) -> None:
    # HF_TOKEN is an optional override the installer never manages: the
    # commented .env.example line must pass through untouched.
    proc = run_lib(
        repo,
        WRITE_ENV_PREAMBLE + "COMPUTE_TIER_VALUE=none\nwrite_env",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "# HF_TOKEN=\n" in content  # untouched template line
    assert "VOXINT_COMPOSE_TIER=none" in content


# --------------------------------------------------------------------------- #
# Kept-.env update path (legacy migration)
# --------------------------------------------------------------------------- #
LEGACY_ENV = "VOXINT_PASSWORD='old-pw'\nMEDIA_ROOT='/data'\nHF_TOKEN=\n"


def test_update_env_keys_appends_tier_and_backs_up(repo: Path) -> None:
    (repo / ".env").write_text(LEGACY_ENV)
    proc = run_lib(repo, "COMPUTE_TIER_VALUE=gpu\nupdate_env_keys")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "VOXINT_COMPOSE_TIER=gpu" in content
    assert "VOXINT_PASSWORD='old-pw'" in content  # untouched passthrough
    assert "HF_TOKEN=\n" in content  # optional-override line passes through untouched
    backups = list(repo.glob(".env.backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == LEGACY_ENV


def _fake_getent(repo: Path, body: str) -> None:
    p = repo / "fakebin" / "getent"
    p.write_text(body)
    p.chmod(0o755)


def _fake_stat_failing(repo: Path) -> None:
    # On a host with a real GPU, /dev/kfd exists and its gid would satisfy
    # detect_render_gid before the getent fallback these tests exercise. A
    # failing stat neutralizes that probe so the tests behave the same on
    # GPU and GPU-less machines.
    p = repo / "fakebin" / "stat"
    p.write_text("#!/bin/sh\nexit 1\n")
    p.chmod(0o755)


def test_detect_render_gid_parses_getent(repo: Path) -> None:
    _fake_stat_failing(repo)
    _fake_getent(repo, "#!/bin/sh\necho 'render:x:990:ben'\n")
    proc = run_lib(repo, 'detect_render_gid\nprintf "%s" "$RENDER_GID_VALUE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "990"


def test_detect_render_gid_rejects_non_numeric_and_notes(repo: Path) -> None:
    _fake_stat_failing(repo)
    _fake_getent(repo, "#!/bin/sh\necho 'garbage output'\n")
    proc = run_lib(repo, 'detect_render_gid\nprintf "[%s]" "$RENDER_GID_VALUE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[]"
    assert "VOXINT_RENDER_GID" in proc.stderr  # the NOTE tells the user what to set


def test_detect_render_gid_empty_when_getent_fails(repo: Path) -> None:
    _fake_stat_failing(repo)
    _fake_getent(repo, "#!/bin/sh\nexit 2\n")
    proc = run_lib(repo, 'detect_render_gid\nprintf "[%s]" "$RENDER_GID_VALUE"')
    assert proc.returncode == 0, proc.stderr  # set -eu must survive the failure
    assert proc.stdout == "[]"


def test_update_env_keys_writes_render_gid_for_rocm(repo: Path) -> None:
    (repo / ".env").write_text(LEGACY_ENV + "VOXINT_COMPOSE_TIER=cpu\nVOXINT_RENDER_GID=111\n")
    proc = run_lib(
        repo,
        "COMPUTE_TIER_VALUE=rocm\nRENDER_GID_VALUE=990\nupdate_env_keys",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert content.count("VOXINT_RENDER_GID=") == 1
    assert "VOXINT_RENDER_GID=990" in content
    assert "VOXINT_COMPOSE_TIER=rocm" in content


def test_update_env_keys_keeps_existing_render_gid_when_not_detected(repo: Path) -> None:
    (repo / ".env").write_text(LEGACY_ENV + "VOXINT_RENDER_GID=111\n")
    proc = run_lib(
        repo,
        "COMPUTE_TIER_VALUE=cpu\nRENDER_GID_VALUE=\nupdate_env_keys",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "VOXINT_RENDER_GID=111" in content  # untouched passthrough


def test_update_env_keys_replaces_tier_and_passes_token_through(repo: Path) -> None:
    # HF_TOKEN is an optional override the installer never manages: an existing
    # line must survive a tier rewrite byte-for-byte.
    (repo / ".env").write_text(LEGACY_ENV + "VOXINT_COMPOSE_TIER=none\nHF_TOKEN='hf_KEEPME'\n")
    proc = run_lib(repo, "COMPUTE_TIER_VALUE=cpu\nupdate_env_keys")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert content.count("VOXINT_COMPOSE_TIER=") == 1
    assert "VOXINT_COMPOSE_TIER=cpu" in content
    assert "HF_TOKEN='hf_KEEPME'" in content
    assert stat.S_IMODE((repo / ".env").stat().st_mode) == 0o600


def test_read_env_value_last_wins_and_strips_quotes(repo: Path) -> None:
    (repo / ".env").write_text("HF_TOKEN='first'\nHF_TOKEN='second'\n")
    proc = run_lib(repo, "read_env_value HF_TOKEN")
    assert proc.stdout == "second"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('HF_TOKEN="hf_dq"', "hf_dq"),  # double quotes stripped like Compose does
        ('HF_TOKEN=""', ""),  # empty double-quoted value reads EMPTY
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


def test_update_env_keys_dedupes_duplicate_tier_lines(repo: Path) -> None:
    (repo / ".env").write_text(
        "VOXINT_PASSWORD='pw'\nVOXINT_COMPOSE_TIER=gpu\nVOXINT_COMPOSE_TIER=none\n"
    )
    proc = run_lib(repo, "COMPUTE_TIER_VALUE=cpu\nupdate_env_keys")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert content.count("VOXINT_COMPOSE_TIER=") == 1
    assert "VOXINT_COMPOSE_TIER=cpu" in content
    # Validation ran against the recorded tier's overlay.
    log = (repo / "docker.log").read_text()
    assert "-f compose.cpu.yaml" in log


# --------------------------------------------------------------------------- #
# Hardware-aware conservative defaults (issue #96): host GPU detection, profile
# selection, compose.hardware.yaml generation, and compose-chain wiring. All via
# the library seam with a fake nvidia-smi on PATH -- no GPU or Docker daemon.
# --------------------------------------------------------------------------- #
NO_NVIDIA = {"VOXINT_NVIDIA_SMI": "/nonexistent/nvidia-smi"}


def _fake_nvidia_smi(repo: Path, body: str) -> None:
    # A fake nvidia-smi in fakebin shadows any real one (fakebin is prepended to
    # PATH), so these tests are deterministic on GPU and GPU-less hosts alike.
    p = repo / "fakebin" / "nvidia-smi"
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)


def run_script(
    repo: Path, args: list[str], extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run install.sh as a real program (main runs), not via the library seam."""
    env = os.environ.copy()
    env["PATH"] = f"{repo / 'fakebin'}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(repo / "docker.log")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(repo / "scripts" / "install.sh"), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NVIDIA GeForce RTX 3060", "nvidia_geforce_rtx_3060"),
        ("  NVIDIA   GeForce  RTX 3060  ", "nvidia_geforce_rtx_3060"),
        ("Quadro RTX 4000", "quadro_rtx_4000"),
        ("Tesla T4", "tesla_t4"),
        ("weird!!name@@here", "weirdnamehere"),
        ("", ""),
    ],
)
def test_normalize_gpu_name(repo: Path, raw: str, expected: str) -> None:
    proc = run_lib(repo, f'normalize_gpu_name "{raw}"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


def test_detect_nvidia_gpu_single_card(repo: Path) -> None:
    _fake_nvidia_smi(repo, "printf 'NVIDIA GeForce RTX 3060, 12288\\n'")
    proc = run_lib(
        repo,
        'detect_nvidia_gpu\nprintf "%s|%s|%s" "$GPU_NAME" "$GPU_VRAM_MIB" "$GPU_SIGNATURE"',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "NVIDIA GeForce RTX 3060|12288|nvidia_geforce_rtx_3060|12288"


def test_detect_nvidia_gpu_absent_is_unknown(repo: Path) -> None:
    proc = run_lib(
        repo,
        'detect_nvidia_gpu\nprintf "[%s]" "$GPU_SIGNATURE"',
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[]"


def test_detect_nvidia_gpu_command_failure_is_unknown(repo: Path) -> None:
    _fake_nvidia_smi(repo, "exit 9")
    proc = run_lib(repo, 'detect_nvidia_gpu\nprintf "[%s]" "$GPU_SIGNATURE"')
    assert proc.returncode == 0, proc.stderr  # set -eu must survive the failure
    assert proc.stdout == "[]"


def test_detect_nvidia_gpu_mixed_multi_gpu_is_unknown(repo: Path) -> None:
    _fake_nvidia_smi(
        repo,
        "printf 'NVIDIA GeForce RTX 3060, 12288\\nQuadro RTX 4000, 8192\\n'",
    )
    proc = run_lib(repo, 'detect_nvidia_gpu\nprintf "[%s]" "$GPU_SIGNATURE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[]"  # differing signatures -> we do not guess


def test_detect_nvidia_gpu_identical_multi_gpu_shares_signature(repo: Path) -> None:
    _fake_nvidia_smi(
        repo,
        "printf 'NVIDIA GeForce RTX 3060, 12288\\nNVIDIA GeForce RTX 3060, 12288\\n'",
    )
    proc = run_lib(repo, 'detect_nvidia_gpu\nprintf "%s" "$GPU_SIGNATURE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "nvidia_geforce_rtx_3060|12288"


def test_detect_nvidia_gpu_mig_is_unknown(repo: Path) -> None:
    _fake_nvidia_smi(repo, "printf 'NVIDIA A100-SXM4-40GB MIG 1g.5gb, 4864\\n'")
    proc = run_lib(repo, 'detect_nvidia_gpu\nprintf "[%s]" "$GPU_SIGNATURE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[]"


def test_detect_nvidia_gpu_malformed_vram_is_unknown(repo: Path) -> None:
    _fake_nvidia_smi(repo, "printf 'NVIDIA GeForce RTX 3060, N/A\\n'")
    proc = run_lib(repo, 'detect_nvidia_gpu\nprintf "[%s]" "$GPU_SIGNATURE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "[]"


def test_select_hardware_profile_unknown_fallback(repo: Path) -> None:
    # No detection -> empty signature -> conservative fallback: cap concurrency
    # and pending queue, and NEVER set BATCH_SIZE (a numerics knob).
    proc = run_lib(
        repo,
        "select_hardware_profile\n"
        'printf "%s|%s|%s|[%s]" "$PROFILE_ID" "$PROFILE_CONCURRENCY" '
        '"$PROFILE_MAX_PENDING" "$PROFILE_BATCH_SIZE"',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "nvidia-unknown-v1|1|1|[]"


def test_render_hardware_override_has_no_batch_size(repo: Path) -> None:
    proc = run_lib(repo, "select_hardware_profile\nrender_hardware_override")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert out.startswith("# voxint:hardware-override v1\n")
    assert "# profile: nvidia-unknown-v1" in out
    assert "--concurrency=1" in out
    assert 'MAX_PENDING_REQUESTS: "1"' in out
    assert "BATCH_SIZE" not in out  # the load-bearing numerics guarantee


def test_render_hardware_override_is_deterministic(repo: Path) -> None:
    proc = run_lib(
        repo,
        "select_hardware_profile\na=$(render_hardware_override)\n"
        'b=$(render_hardware_override)\n[ "$a" = "$b" ] && echo SAME || echo DIFF',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "SAME"


def test_generated_worker_command_matches_base_compose(repo: Path) -> None:
    # DRIFT GUARD: the override restates the whole worker command (Compose
    # replaces command: wholesale), so WORKER_BASE_COMMAND must stay byte-equal
    # to compose.yaml's worker command. If the base command ever changes, this
    # fails and forces the generator to move in lockstep.
    import yaml

    compose = yaml.safe_load((repo / "compose.yaml").read_text())
    base_command = compose["services"]["worker"]["command"]

    proc = run_lib(repo, 'printf "%s" "$WORKER_BASE_COMMAND"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == base_command, (
        "compose.yaml worker command drifted from WORKER_BASE_COMMAND in install.sh"
    )

    rendered = run_lib(repo, "select_hardware_profile\nrender_hardware_override")
    assert f"command: {base_command} --concurrency=1" in rendered.stdout


def test_write_hardware_override_first_write(repo: Path) -> None:
    proc = run_lib(
        repo,
        "select_hardware_profile\nwrite_hardware_override\necho RC=$?",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=0" in proc.stdout
    target = repo / "compose.hardware.yaml"
    assert target.exists()
    assert target.read_text().startswith("# voxint:hardware-override v1\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert "BATCH_SIZE" not in target.read_text()


def test_write_hardware_override_idempotent(repo: Path) -> None:
    # Writing twice leaves identical content and, crucially, the second call
    # returns BEFORE re-validating (only one docker invocation is logged).
    proc = run_lib(
        repo,
        "select_hardware_profile\nwrite_hardware_override\nwrite_hardware_override\necho RC=$?",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=0" in proc.stdout
    log = (repo / "docker.log").read_text()
    assert log.count("config --quiet") == 1  # second write was a no-op


def test_write_hardware_override_refuses_unmarked_operator_file(repo: Path) -> None:
    operator = "services:\n  worker:\n    command: my own thing\n"
    (repo / "compose.hardware.yaml").write_text(operator)
    proc = run_lib(
        repo,
        "select_hardware_profile\nif write_hardware_override; then echo RC=0; else echo RC=$?; fi",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=1" in proc.stdout  # deliberately declined, not a hard failure
    assert (repo / "compose.hardware.yaml").read_text() == operator  # untouched
    assert "was not generated by the installer" in proc.stderr


def test_write_hardware_override_refuses_symlink(repo: Path, tmp_path: Path) -> None:
    victim = tmp_path / "victim.yaml"
    victim.write_text("services: {}\n")
    (repo / "compose.hardware.yaml").symlink_to(victim)
    proc = run_lib(
        repo,
        "select_hardware_profile\nif write_hardware_override; then echo RC=0; else echo RC=$?; fi",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=1" in proc.stdout
    assert victim.read_text() == "services: {}\n"  # symlink target unharmed


def test_write_hardware_override_validation_failure_is_atomic(repo: Path) -> None:
    # A failing compose validation must abort the write with a clear error and
    # leave no target and no temp file behind.
    (repo / "fakebin" / "docker").write_text("#!/usr/bin/env bash\nexit 1\n")
    (repo / "fakebin" / "docker").chmod(0o755)
    proc = run_lib(
        repo,
        "select_hardware_profile\nwrite_hardware_override",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode != 0
    assert "failed Compose validation" in proc.stderr
    assert not (repo / "compose.hardware.yaml").exists()
    assert not list(repo.glob(".compose.hardware.tmp.*"))  # temp cleaned up


def _write_managed_hardware_file(repo: Path) -> None:
    run_lib(
        repo,
        "select_hardware_profile\nwrite_hardware_override",
        extra_env=NO_NVIDIA,
    )


def test_compose_file_args_appends_hardware_for_gpu_only(repo: Path) -> None:
    _write_managed_hardware_file(repo)
    assert (repo / "compose.hardware.yaml").exists()
    gpu = run_lib(repo, "compose_file_args_for_tier gpu")
    assert gpu.stdout == "-f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml"
    for tier, base in [
        ("cpu", "-f compose.yaml -f compose.cpu.yaml"),
        ("rocm", "-f compose.yaml -f compose.rocm.yaml"),
        ("metal", "-f compose.yaml -f compose.metal.yaml"),
        ("none", "-f compose.yaml"),
    ]:
        proc = run_lib(repo, f"compose_file_args_for_tier {tier}")
        assert proc.stdout == base, f"{tier} must not fold in the GPU hardware file"


def test_compose_file_args_ignores_unmarked_hardware_file(repo: Path) -> None:
    (repo / "compose.hardware.yaml").write_text("services: {}\n")  # no marker
    proc = run_lib(repo, "compose_file_args_for_tier gpu")
    assert proc.stdout == "-f compose.yaml -f compose.gpu.yaml"


def test_compose_file_args_appends_operator_override_last(repo: Path) -> None:
    _write_managed_hardware_file(repo)
    (repo / "compose.override.yaml").write_text(
        'services:\n  api:\n    environment:\n      FOO: "bar"\n'
    )
    gpu = run_lib(repo, "compose_file_args_for_tier gpu")
    assert gpu.stdout == (
        "-f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml -f compose.override.yaml"
    )
    cpu = run_lib(repo, "compose_file_args_for_tier cpu")
    assert cpu.stdout == "-f compose.yaml -f compose.cpu.yaml -f compose.override.yaml"


def test_dc_passes_hardware_file_to_docker(repo: Path) -> None:
    _write_managed_hardware_file(repo)
    proc = run_lib(
        repo,
        "COMPOSE_FILE_ARGS=$(compose_file_args_for_tier gpu)\ndc config --quiet",
    )
    assert proc.returncode == 0, proc.stderr
    log = (repo / "docker.log").read_text()
    assert "-f compose.gpu.yaml -f compose.hardware.yaml config --quiet" in log


def test_hardware_dry_run_prints_yaml_and_writes_nothing(repo: Path) -> None:
    _fake_nvidia_smi(repo, "printf 'NVIDIA GeForce RTX 3060, 12288\\n'")
    proc = run_script(repo, ["--hardware-dry-run"])
    assert proc.returncode == 0, proc.stderr
    # YAML body on stdout (so `2>/dev/null` yields just the file).
    assert "# voxint:hardware-override v1" in proc.stdout
    assert 'MAX_PENDING_REQUESTS: "1"' in proc.stdout
    assert "BATCH_SIZE" not in proc.stdout
    # Human annotations on stderr; nothing written or started.
    assert "dry run" in proc.stderr
    assert not (repo / "compose.hardware.yaml").exists()
    assert not (repo / ".env").exists()


def test_hardware_dry_run_unknown_gpu_is_conservative(repo: Path) -> None:
    proc = run_script(repo, ["--hardware-dry-run"], extra_env=NO_NVIDIA)
    assert proc.returncode == 0, proc.stderr
    assert "# profile: nvidia-unknown-v1" in proc.stdout
    assert "--concurrency=1" in proc.stdout
