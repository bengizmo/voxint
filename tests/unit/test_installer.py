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

# Like FAKE_DOCKER but, for `config`, asserts every `-f` argument names a real
# file. A word-split absolute path (e.g. a repo dir with a space) reaches this as
# a broken token and fails -- so it catches quoting regressions the always-pass
# fake cannot.
VALIDATING_FAKE_DOCKER = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${FAKE_DOCKER_LOG:?}"
is_config=0
for a in "$@"; do [ "$a" = "config" ] && is_config=1; done
if [ "$is_config" = "1" ]; then
  prev=""
  for a in "$@"; do
    if [ "$prev" = "-f" ] && [ ! -f "$a" ]; then
      printf 'no such compose file: %s\\n' "$a" >&2
      exit 1
    fi
    prev="$a"
  done
fi
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
    # A developer's own VOXINT_NVIDIA_SMI must not override the fakebin/NO_NVIDIA
    # seam these tests rely on; each test sets it explicitly when it wants one.
    env.pop("VOXINT_NVIDIA_SMI", None)
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
# Advanced alternate-model overrides (B2): the one skippable advanced entry
# --------------------------------------------------------------------------- #
def test_write_env_omits_model_keys_when_not_opted_in(repo: Path) -> None:
    # A default install sets none of the advanced globals, so no uncommented model
    # key is written and the optional HF_TOKEN template line stays commented.
    proc = run_lib(repo, WRITE_ENV_PREAMBLE + "COMPUTE_TIER_VALUE=none\nwrite_env")
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    for key in (
        "WHISPER_MODEL=",
        "WHISPER_REVISION=",
        "WHISPER_ALLOW_DOWNLOAD=",
        "DIARIZER_MODEL_NAME=",
        "DIARIZER_REVISION=",
    ):
        assert f"\n{key}" not in content  # never written uncommented
    assert "# HF_TOKEN=\n" in content


def test_write_env_writes_advanced_model_overrides(repo: Path) -> None:
    sha = "a" * 40
    proc = run_lib(
        repo,
        WRITE_ENV_PREAMBLE
        + "COMPUTE_TIER_VALUE=gpu\n"
        "WHISPER_MODEL_VALUE=large-v3\n"
        f"WHISPER_REVISION_VALUE={sha}\n"
        "WHISPER_ALLOW_DOWNLOAD_VALUE=1\n"
        "DIARIZER_MODEL_NAME_VALUE=pyannote/community-1\n"
        "DIARIZER_REVISION_VALUE=abc123\n"
        "HF_TOKEN_VALUE='hf_s3cret'\n"
        "write_env",
    )
    assert proc.returncode == 0, proc.stderr
    content = (repo / ".env").read_text()
    assert "WHISPER_MODEL=large-v3" in content
    assert f"WHISPER_REVISION={sha}" in content
    assert "WHISPER_ALLOW_DOWNLOAD=1" in content
    assert "DIARIZER_MODEL_NAME=pyannote/community-1" in content
    assert "DIARIZER_REVISION=abc123" in content
    # HF_TOKEN is single-quoted like the password so an odd character is dotenv-safe.
    assert "HF_TOKEN='hf_s3cret'" in content


def test_prompt_advanced_models_skip_writes_nothing(repo: Path) -> None:
    # Answering 'no' (or just pressing enter) leaves every override global empty.
    proc = run_lib(
        repo,
        "prompt_advanced_models\n"
        'echo "M=${WHISPER_MODEL_VALUE}"\n'
        'echo "D=${DIARIZER_MODEL_NAME_VALUE}"\n'
        'echo "T=${HF_TOKEN_VALUE}"\n',
        stdin="\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "M=\n" in proc.stdout
    assert "D=\n" in proc.stdout
    assert "T=\n" in proc.stdout


def test_prompt_advanced_models_collects_values_and_hides_token(repo: Path) -> None:
    sha = "b" * 40
    stdin = f"y\nlarge-v3\n{sha}\npyannote/community-1\nrev9\nhf_topsecret\n"
    proc = run_lib(
        repo,
        "prompt_advanced_models\n"
        'echo "M=${WHISPER_MODEL_VALUE}"\n'
        'echo "R=${WHISPER_REVISION_VALUE}"\n'
        'echo "A=${WHISPER_ALLOW_DOWNLOAD_VALUE}"\n'
        'echo "D=${DIARIZER_MODEL_NAME_VALUE}"\n'
        'echo "DR=${DIARIZER_REVISION_VALUE}"\n'
        'echo "T=${HF_TOKEN_VALUE}"\n',
        stdin=stdin,
    )
    assert proc.returncode == 0, proc.stderr
    assert "M=large-v3\n" in proc.stdout
    assert f"R={sha}\n" in proc.stdout
    assert "A=1\n" in proc.stdout  # opting in auto-records the download permission
    assert "D=pyannote/community-1\n" in proc.stdout
    assert "DR=rev9\n" in proc.stdout
    assert "T=hf_topsecret\n" in proc.stdout
    # The token is never echoed by the installer's own prompts (only surfaced by
    # the test's explicit echo above, which prints once).
    assert proc.stderr.count("hf_topsecret") == 0
    assert proc.stdout.count("hf_topsecret") == 1


def test_prompt_advanced_models_drops_whisper_override_without_full_sha(repo: Path) -> None:
    # A non-technical operator who gives a model but a short/typo'd revision would
    # otherwise get WHISPER_MODEL + WHISPER_ALLOW_DOWNLOAD=1 with no valid SHA, and
    # the whisper service refuses to start (crash-loop after a "successful" install).
    # The installer now drops the override and keeps the validated large-v2 instead.
    for bad_rev in ("", "abc123", "A" * 40):  # blank, too-short, uppercase
        proc = run_lib(
            repo,
            "prompt_advanced_models\n"
            'echo "M=${WHISPER_MODEL_VALUE}"\n'
            'echo "R=${WHISPER_REVISION_VALUE}"\n'
            'echo "A=${WHISPER_ALLOW_DOWNLOAD_VALUE}"\n',
            stdin=f"y\nlarge-v3\n{bad_rev}\n\n",
        )
        assert proc.returncode == 0, proc.stderr
        assert "M=\n" in proc.stdout, f"model should be dropped for revision {bad_rev!r}"
        assert "R=\n" in proc.stdout
        assert "A=\n" in proc.stdout  # never records the download opt-in without a SHA
        assert "Keeping the" in proc.stderr


def test_prompt_advanced_models_rejects_unsafe_token(repo: Path) -> None:
    # A token with an embedded single-quote would break dotenv_squote; reject it at
    # input rather than emit a malformed .env that fails opaquely at Compose time.
    proc = run_lib(repo, "prompt_advanced_models\n", stdin="y\n\npyannote/x\n\nbad'tok\n")
    assert proc.returncode != 0
    assert "must not contain a single-quote" in proc.stderr


def test_prompt_advanced_models_rejects_unsafe_value(repo: Path) -> None:
    # A model id with a space cannot be written into dotenv safely: reject it here
    # with a clear message rather than fail opaquely at Compose validation.
    proc = run_lib(repo, "prompt_advanced_models\n", stdin="y\nbad model\n")
    assert proc.returncode != 0
    assert "must not contain spaces" in proc.stderr


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
NO_AMD = {"VOXINT_SYSFS_ROOT": "/nonexistent/sysfs", "VOXINT_DEV_ROOT": "/nonexistent/dev"}


def _fake_nvidia_smi(repo: Path, body: str) -> None:
    # A fake nvidia-smi in fakebin shadows any real one (fakebin is prepended to
    # PATH), so these tests are deterministic on GPU and GPU-less hosts alike.
    p = repo / "fakebin" / "nvidia-smi"
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)


def _fake_amd_sysfs(
    repo: Path,
    render_num: int = 129,
    *,
    vendor: str = "0x1002\n",
    vram_total: int | None = 4294967296,  # 4 GiB in bytes
    vram_used: int | None = 1048576000,  # ~1000 MiB
    gtt_total: int | None = 14331686912,  # ~13.7 GiB
    product_name: str | None = None,
    unique_id: str | None = None,
    pci_slot: str | None = "0000:c5:00.0",
) -> tuple[Path, Path]:
    """Create fake AMD GPU sysfs tree + /dev/kfd. Returns (fakesys, fakedev)."""
    fakesys = repo / "fakesys"
    fakedev = repo / "fakedev"
    dev_dir = fakesys / "class" / "drm" / f"renderD{render_num}" / "device"
    dev_dir.mkdir(parents=True, exist_ok=True)
    (dev_dir / "vendor").write_text(vendor)
    if vram_total is not None:
        (dev_dir / "mem_info_vram_total").write_text(str(vram_total) + "\n")
    if vram_used is not None:
        (dev_dir / "mem_info_vram_used").write_text(str(vram_used) + "\n")
    if gtt_total is not None:
        (dev_dir / "mem_info_gtt_total").write_text(str(gtt_total) + "\n")
    if product_name is not None:
        (dev_dir / "product_name").write_text(product_name + "\n")
    if unique_id is not None:
        (dev_dir / "unique_id").write_text(unique_id + "\n")
    if pci_slot is not None:
        uevent = f"PCI_SLOT_NAME={pci_slot}\n"
        (dev_dir / "uevent").write_text(uevent)
    # /dev/kfd must exist
    fakedev.mkdir(exist_ok=True)
    (fakedev / "kfd").touch()
    # /dev/dri/renderDN must be referenced (but needn't be a real device)
    (fakedev / "dri").mkdir(exist_ok=True)
    (fakedev / "dri" / f"renderD{render_num}").touch()
    return fakesys, fakedev


def run_script(
    repo: Path, args: list[str], extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run install.sh as a real program (main runs), not via the library seam."""
    env = os.environ.copy()
    env["PATH"] = f"{repo / 'fakebin'}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(repo / "docker.log")
    # See run_lib: keep an ambient VOXINT_NVIDIA_SMI from leaking into the seam.
    env.pop("VOXINT_NVIDIA_SMI", None)
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


# --------------------------------------------------------------------------- #
# GPU inventory: enumerate_nvidia_gpus
# --------------------------------------------------------------------------- #


def test_enumerate_nvidia_single_gpu(repo: Path) -> None:
    """Single NVIDIA GPU with 6-field CSV format."""
    _fake_nvidia_smi(
        repo,
        r"printf '0, GPU-abc-123, NVIDIA GeForce RTX 3060, 12288, 2048, 10240\n'",
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert len(fields) == 9
    assert fields[0] == "nvidia"  # vendor
    assert fields[1] == "0"  # index
    assert fields[2] == "GPU-abc-123"  # stable_id (UUID)
    assert fields[3] == "NVIDIA GeForce RTX 3060"  # name
    assert fields[4] == "12288"  # total_mib
    assert fields[5] == "2048"  # used_mib
    assert fields[6] == "10240"  # free_mib
    assert fields[7] == ""  # render_node (empty for NVIDIA)
    assert fields[8] == "measured"  # mem_confidence


def test_enumerate_nvidia_occupied_gpu(repo: Path) -> None:
    """NVIDIA GPU with nearly zero free VRAM (occupied by another process)."""
    _fake_nvidia_smi(
        repo,
        r"printf '0, GPU-5090-uuid, NVIDIA GeForce RTX 5090, 32607, 31974, 177\n'",
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert fields[4] == "32607"
    assert fields[6] == "177"  # nearly zero free


def test_enumerate_nvidia_multi_gpu(repo: Path) -> None:
    """Multiple NVIDIA GPUs, including one occupied."""
    _fake_nvidia_smi(
        repo,
        r"printf '0, GPU-5090-uuid, NVIDIA GeForce RTX 5090, "
        r"32607, 31974, 177\n1, GPU-4060-uuid, NVIDIA GeForce RTX 4060, "
        r"8192, 512, 7680\n'",
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == 2
    # First GPU: occupied 5090
    f1 = lines[0].split("\t")
    assert f1[2] == "GPU-5090-uuid"
    assert f1[6] == "177"
    # Second GPU: available 4060
    f2 = lines[1].split("\t")
    assert f2[2] == "GPU-4060-uuid"
    assert f2[6] == "7680"


def test_enumerate_nvidia_absent(repo: Path) -> None:
    """No nvidia-smi -> empty inventory."""
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        wc -l < "$GPU_INVENTORY_FILE" | tr -d ' '
        cleanup_gpu_inventory
    """,
        extra_env={**NO_NVIDIA, **NO_AMD},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


def test_enumerate_nvidia_mig_rows_skipped(repo: Path) -> None:
    """MIG instance rows are skipped but non-MIG rows pass."""
    _fake_nvidia_smi(
        repo,
        r"printf '0, GPU-a100-uuid, NVIDIA A100-SXM4-40GB MIG 1g.5gb, "
        r"4864, 0, 4864\n1, GPU-3060-uuid, NVIDIA GeForce RTX 3060, "
        r"12288, 0, 12288\n'",
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == 1  # MIG row skipped
    assert "RTX 3060" in lines[0]


def test_enumerate_nvidia_malformed_total_skipped(repo: Path) -> None:
    """Rows with non-numeric total VRAM are skipped."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-bad, Bad Card, N/A, 0, 0\n'")
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        wc -l < "$GPU_INVENTORY_FILE" | tr -d ' '
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


def test_enumerate_nvidia_malformed_free_is_minus_one(repo: Path) -> None:
    """Non-numeric free VRAM is recorded as -1 (unknown), row not skipped."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-ok, Good Card, 12288, N/A, N/A\n'")
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert fields[4] == "12288"  # total is valid
    assert fields[5] == "-1"  # used is unknown
    assert fields[6] == "-1"  # free is unknown


# --------------------------------------------------------------------------- #
# GPU inventory: enumerate_amd_gpus
# --------------------------------------------------------------------------- #


def test_enumerate_amd_igpu_shared(repo: Path) -> None:
    """AMD iGPU with GTT >> VRAM -> shared memory confidence."""
    fakesys, fakedev = _fake_amd_sysfs(
        repo,
        129,
        vram_total=4294967296,
        vram_used=1048576000,
        gtt_total=14331686912,
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert fields[0] == "amd"
    assert fields[1] == "129"
    assert fields[4] == "4096"  # 4294967296 / 1048576
    assert fields[5] == "1000"  # 1048576000 / 1048576
    assert fields[6] == "3096"  # 4096 - 1000
    assert "renderD129" in fields[7]
    assert fields[8] == "shared"


def test_enumerate_amd_dgpu_sysfs(repo: Path) -> None:
    """AMD discrete GPU with unique_id and small GTT -> sysfs confidence."""
    fakesys, fakedev = _fake_amd_sysfs(
        repo,
        128,
        vram_total=16 * 1024 * 1024 * 1024,  # 16 GiB
        vram_used=2 * 1024 * 1024 * 1024,  # 2 GiB
        gtt_total=8 * 1024 * 1024 * 1024,  # 8 GiB (< 2*VRAM = 32 GiB)
        unique_id="0x00abcdef12345678",
        product_name="Radeon RX 7900 XTX",
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert fields[0] == "amd"
    assert fields[2] == "0x00abcdef12345678"  # unique_id
    assert fields[3] == "Radeon RX 7900 XTX"  # product_name
    assert fields[4] == "16384"  # 16 GiB
    assert fields[8] == "sysfs"  # NOT shared (GTT < 2*VRAM)


def test_enumerate_amd_no_kfd(repo: Path) -> None:
    """No /dev/kfd -> empty AMD inventory."""
    fakesys = repo / "fakesys"
    fakedev = repo / "fakedev"
    fakesys.mkdir()
    fakedev.mkdir()
    # No kfd file
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        wc -l < "$GPU_INVENTORY_FILE" | tr -d ' '
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


def test_enumerate_amd_nvidia_device_skipped(repo: Path) -> None:
    """NVIDIA renderD nodes (vendor 0x10de) in sysfs are not added as AMD."""
    fakesys = repo / "fakesys"
    fakedev = repo / "fakedev"
    dev_dir = fakesys / "class" / "drm" / "renderD128" / "device"
    dev_dir.mkdir(parents=True)
    (dev_dir / "vendor").write_text("0x10de\n")  # NVIDIA vendor
    fakedev.mkdir()
    (fakedev / "kfd").touch()
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        wc -l < "$GPU_INVENTORY_FILE" | tr -d ' '
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0"


def test_enumerate_amd_no_vram_files(repo: Path) -> None:
    """AMD GPU with no mem_info files -> -1 for all memory, shared confidence."""
    fakesys, fakedev = _fake_amd_sysfs(
        repo, 129, vram_total=None, vram_used=None, gtt_total=None
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    fields = proc.stdout.strip().split("\t")
    assert fields[4] == "-1"
    assert fields[5] == "-1"
    assert fields[6] == "-1"
    assert fields[8] == "shared"


# --------------------------------------------------------------------------- #
# GPU inventory: mixed NVIDIA + AMD
# --------------------------------------------------------------------------- #


def test_enumerate_mixed_nvidia_and_amd(repo: Path) -> None:
    """Both NVIDIA and AMD GPUs appear in the same inventory."""
    _fake_nvidia_smi(
        repo,
        r"printf '0, GPU-5090-uuid, NVIDIA GeForce RTX 5090, 32607, 31974, 177\n'",
    )
    fakesys, fakedev = _fake_amd_sysfs(
        repo,
        129,
        vram_total=4294967296,
        vram_used=1048576000,
        gtt_total=14331686912,
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        cat "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("nvidia\t")
    assert lines[1].startswith("amd\t")


# --------------------------------------------------------------------------- #
# GPU inventory: classify_gpu
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("free", "total", "confidence", "tier", "expected"),
    [
        # NVIDIA tier
        ("8192", "8192", "measured", "gpu", "recommended"),
        ("7000", "8192", "measured", "gpu", "constrained"),
        ("3000", "8192", "measured", "gpu", "insufficient"),
        ("177", "32607", "measured", "gpu", "insufficient"),
        ("-1", "32607", "measured", "gpu", "unknown"),
        ("-1", "4096", "shared", "gpu", "insufficient"),
        ("-1", "-1", "shared", "gpu", "unknown"),
        # ROCm tier
        ("15000", "16384", "sysfs", "rocm", "recommended"),
        ("8000", "16384", "sysfs", "rocm", "constrained"),
        ("2500", "4096", "shared", "rocm", "insufficient"),
        # Unknown tier
        ("8192", "8192", "measured", "cpu", "unknown"),
    ],
)
def test_classify_gpu(
    repo: Path,
    free: str,
    total: str,
    confidence: str,
    tier: str,
    expected: str,
) -> None:
    proc = run_lib(repo, f"classify_gpu {free} {total} {confidence} {tier}")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


# --------------------------------------------------------------------------- #
# GPU inventory: print_gpu_inventory
# --------------------------------------------------------------------------- #


def test_print_gpu_inventory_occupied_nvidia(repo: Path) -> None:
    """Occupied NVIDIA GPU shows (occupied) annotation."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-5090, RTX 5090, 32607, 31974, 177\n'")
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        print_gpu_inventory "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env=NO_AMD,
    )
    assert proc.returncode == 0, proc.stderr
    assert "(occupied)" in proc.stderr


def test_print_gpu_inventory_shared_amd(repo: Path) -> None:
    """AMD iGPU shows (shared system memory) annotation."""
    fakesys, fakedev = _fake_amd_sysfs(
        repo,
        129,
        vram_total=4294967296,
        vram_used=1048576000,
        gtt_total=14331686912,
    )
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        print_gpu_inventory "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={
            **NO_NVIDIA,
            "VOXINT_SYSFS_ROOT": str(fakesys),
            "VOXINT_DEV_ROOT": str(fakedev),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "(shared system memory)" in proc.stderr


def test_print_gpu_inventory_empty(repo: Path) -> None:
    """Empty inventory says 'No GPUs detected'."""
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        print_gpu_inventory "$GPU_INVENTORY_FILE"
        cleanup_gpu_inventory
    """,
        extra_env={**NO_NVIDIA, **NO_AMD},
    )
    assert proc.returncode == 0, proc.stderr
    assert "No GPUs detected" in proc.stderr


def test_cleanup_gpu_inventory(repo: Path) -> None:
    """cleanup_gpu_inventory removes the temp file and clears the global."""
    proc = run_lib(
        repo,
        """
        create_gpu_inventory
        file="$GPU_INVENTORY_FILE"
        [ -f "$file" ] && echo "exists"
        cleanup_gpu_inventory
        [ -f "$file" ] && echo "still-exists" || echo "removed"
        echo "var=[$GPU_INVENTORY_FILE]"
    """,
        extra_env={**NO_NVIDIA, **NO_AMD},
    )
    assert proc.returncode == 0, proc.stderr
    assert "exists" in proc.stdout
    assert "removed" in proc.stdout
    assert "var=[]" in proc.stdout


# --------------------------------------------------------------------------- #
# Phase 3: compose generation with device pinning
# --------------------------------------------------------------------------- #

def test_render_hardware_override_nvidia_device_pinning(repo: Path) -> None:
    """NVIDIA device pinning emits device_ids with the UUID for all 3 services."""
    proc = run_lib(repo, """
        SEL_GPU_VENDOR=nvidia
        SEL_GPU_STABLE_ID="GPU-test-1234"
        SEL_GPU_NAME="RTX Test"
        SEL_GPU_TOTAL_MIB=8192
        SEL_GPU_FREE_MIB=7000
        SEL_GPU_RENDER_NODE=""
        SEL_GPU_MEM_CONF=measured
        EFFECTIVE_TIER=gpu
        select_hardware_profile
        render_hardware_override
    """, extra_env=NO_NVIDIA)
    assert proc.returncode == 0, proc.stderr
    yaml_out = proc.stdout
    assert "device_ids:" in yaml_out
    assert '"GPU-test-1234"' in yaml_out
    assert "!override" in yaml_out
    # All three services should have the UUID
    assert yaml_out.count("GPU-test-1234") == 3
    # Marker is v2
    assert "# voxint:hardware-override v2" in yaml_out
    # BATCH_SIZE must NOT appear (unknown profile)
    assert "BATCH_SIZE" not in yaml_out


def test_render_hardware_override_rocm_device_pinning(repo: Path) -> None:
    """ROCm device pinning emits renderD node + HIP/ROCR visibility vars."""
    proc = run_lib(repo, """
        SEL_GPU_VENDOR=amd
        SEL_GPU_STABLE_ID="0000:c5:00.0"
        SEL_GPU_NAME="Test AMD GPU"
        SEL_GPU_TOTAL_MIB=16384
        SEL_GPU_FREE_MIB=14000
        SEL_GPU_RENDER_NODE="/dev/dri/renderD129"
        SEL_GPU_MEM_CONF=sysfs
        EFFECTIVE_TIER=rocm
        PROFILE_ID="rocm-unknown-v1"
        PROFILE_CONCURRENCY=1
        PROFILE_MAX_PENDING=1
        PROFILE_BATCH_SIZE=""
        render_hardware_override
    """, extra_env=NO_NVIDIA)
    assert proc.returncode == 0, proc.stderr
    yaml_out = proc.stdout
    assert "/dev/kfd" in yaml_out
    assert "/dev/dri/renderD129" in yaml_out
    assert "!override" in yaml_out
    assert 'HIP_VISIBLE_DEVICES: "0"' in yaml_out
    assert 'ROCR_VISIBLE_DEVICES: "0"' in yaml_out
    assert "BATCH_SIZE" not in yaml_out
    # Should NOT have device_ids (that's NVIDIA)
    assert "device_ids" not in yaml_out


def test_render_hardware_override_no_device_selected(repo: Path) -> None:
    """No device selected -> scheduling caps only, no device_ids or renderD."""
    proc = run_lib(repo, """
        SEL_GPU_STABLE_ID=""
        EFFECTIVE_TIER=gpu
        select_hardware_profile
        render_hardware_override
    """, extra_env=NO_NVIDIA)
    assert proc.returncode == 0, proc.stderr
    yaml_out = proc.stdout
    assert "device_ids" not in yaml_out
    assert "/dev/kfd" not in yaml_out
    assert "MAX_PENDING_REQUESTS" in yaml_out
    assert "BATCH_SIZE" not in yaml_out


def test_render_hardware_override_never_emits_batch_size_for_unknown(repo: Path) -> None:
    """Device pinning does NOT add BATCH_SIZE (numerics doctrine)."""
    for tier, vendor, stable_id, render_node in [
        ("gpu", "nvidia", "GPU-test", ""),
        ("rocm", "amd", "0000:xx:00.0", "/dev/dri/renderD128"),
        ("gpu", "", "", ""),  # no device selected
    ]:
        proc = run_lib(repo, f"""
            SEL_GPU_VENDOR="{vendor}"
            SEL_GPU_STABLE_ID="{stable_id}"
            SEL_GPU_RENDER_NODE="{render_node}"
            SEL_GPU_NAME="Test"
            SEL_GPU_TOTAL_MIB=8192
            SEL_GPU_FREE_MIB=7000
            SEL_GPU_MEM_CONF=measured
            EFFECTIVE_TIER={tier}
            PROFILE_ID="test"
            PROFILE_CONCURRENCY=1
            PROFILE_MAX_PENDING=1
            PROFILE_BATCH_SIZE=""
            render_hardware_override
        """, extra_env=NO_NVIDIA)
        assert "BATCH_SIZE" not in proc.stdout, (
            f"BATCH_SIZE leaked for tier={tier}, vendor={vendor}"
        )


def test_compose_file_args_rocm_includes_managed_hardware(repo: Path) -> None:
    """ROCm tier includes managed compose.hardware.yaml in the chain."""
    # Create a managed file
    hw = repo / "compose.hardware.yaml"
    hw.write_text("# voxint:hardware-override v2\nservices: {}\n")
    proc = run_lib(repo, 'compose_file_args_for_tier rocm')
    assert proc.returncode == 0, proc.stderr
    assert "compose.hardware.yaml" in proc.stdout


# --------------------------------------------------------------------------- #
# Phase 4: recommendation, dry-run, gpu-check
# --------------------------------------------------------------------------- #

def test_recommend_tier_occupied_nvidia_undersized_amd(repo: Path) -> None:
    """The golden scenario: occupied 5090 + 4GB 780M -> CPU."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-5090, RTX 5090, 32607, 31974, 177\n'")
    fakesys, fakedev = _fake_amd_sysfs(
        repo,
        129,
        vram_total=4294967296,
        vram_used=1048576000,
        gtt_total=14331686912,
    )
    proc = run_lib(repo, """
        create_gpu_inventory
        recommend_compute_tier
        printf "%s|%s" "$RECOMMENDED_TIER" "$RECOMMENDATION_REASON"
        cleanup_gpu_inventory
    """, extra_env={"VOXINT_SYSFS_ROOT": str(fakesys), "VOXINT_DEV_ROOT": str(fakedev)})
    assert proc.returncode == 0, proc.stderr
    parts = proc.stdout.split("|", 1)
    assert parts[0] == "cpu"
    assert "occupied" in parts[1].lower() or "177" in parts[1]


def test_recommend_tier_available_nvidia(repo: Path) -> None:
    """An NVIDIA GPU with plenty of free VRAM -> GPU tier."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-free, RTX 4060, 8192, 512, 7680\n'")
    proc = run_lib(repo, """
        create_gpu_inventory
        recommend_compute_tier
        printf "%s" "$RECOMMENDED_TIER"
        cleanup_gpu_inventory
    """, extra_env=NO_AMD)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "gpu"


def test_gpu_check_mode(repo: Path) -> None:
    """--gpu-check prints inventory and recommendation without writing."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-test, Test GPU, 8192, 512, 7680\n'")
    result = run_script(repo, ["--gpu-check"], extra_env=NO_AMD)
    assert result.returncode == 0, result.stderr
    assert "GPU Inventory" in result.stderr
    assert "Recommended tier" in result.stderr
    assert "read-only check" in result.stderr


def test_hardware_dry_run_shows_inventory(repo: Path) -> None:
    """--hardware-dry-run shows full GPU inventory."""
    _fake_nvidia_smi(repo, r"printf '0, GPU-test, Test GPU, 8192, 512, 7680\n'")
    result = run_script(repo, ["--hardware-dry-run"], extra_env=NO_AMD)
    assert result.returncode == 0, result.stderr
    assert "GPU Inventory" in result.stderr
    assert "Recommended tier" in result.stderr
    assert "dry run" in result.stderr


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
    assert out.startswith("# voxint:hardware-override v2\n")
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
    # The override restates the command as a single YAML scalar; a list/folded
    # form would need different handling, so pin the shape too.
    assert isinstance(base_command, str), (
        "worker command must stay a single YAML string for wholesale override restatement"
    )
    # The generated override is applied on top of compose.gpu.yaml, but it restates
    # only compose.yaml's command. If the gpu overlay ever sets its own worker
    # command, the override would silently clobber it -- fail loudly if that lands.
    gpu = yaml.safe_load((repo / "compose.gpu.yaml").read_text())
    assert "command" not in gpu.get("services", {}).get("worker", {}), (
        "compose.gpu.yaml worker gained a command; WORKER_BASE_COMMAND must account for it"
    )

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
    assert target.read_text().startswith("# voxint:hardware-override v2\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert "BATCH_SIZE" not in target.read_text()


def test_write_hardware_override_idempotent(repo: Path) -> None:
    # Writing twice leaves identical content and, crucially, the second call
    # returns BEFORE re-validating (only one docker invocation is logged) and
    # reports "unchanged" rather than "written".
    proc = run_lib(
        repo,
        "select_hardware_profile\nwrite_hardware_override\n"
        'printf "R1=%s\\n" "$HARDWARE_WRITE_RESULT"\n'
        "write_hardware_override\necho RC=$?\n"
        'printf "R2=%s\\n" "$HARDWARE_WRITE_RESULT"',
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=0" in proc.stdout
    assert "R1=written" in proc.stdout  # first call created the file
    assert "R2=unchanged" in proc.stdout  # second call was a content no-op
    log = (repo / "docker.log").read_text()
    assert log.count("config --quiet") == 1  # second write skipped re-validation
    # The validation line names the temp candidate and passes --env-file .env,
    # so a regression that validated the wrong file set would be caught here.
    assert ".compose.hardware.tmp." in log
    assert "--env-file .env" in log


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
    # A symlink -- even one pointing at a regular file -- is refused outright via
    # the -L guard, BEFORE the -f/marker checks (which follow the link). The
    # target is never written through, and the link is left in place.
    victim = tmp_path / "victim.yaml"
    victim.write_text("services: {}\n")
    (repo / "compose.hardware.yaml").symlink_to(victim)
    proc = run_lib(
        repo,
        "select_hardware_profile\nif write_hardware_override; then echo RC=0; else echo RC=$?; fi\n"
        'printf "RESULT=%s\\n" "$HARDWARE_WRITE_RESULT"',
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=1" in proc.stdout
    assert "RESULT=refused" in proc.stdout
    assert "is a symlink" in proc.stderr  # the dedicated symlink branch fired
    assert (repo / "compose.hardware.yaml").is_symlink()  # link untouched
    assert victim.read_text() == "services: {}\n"  # symlink target unharmed
    # A symlinked file is never treated as installer-managed, so it is not folded
    # into the compose chain either.
    chain = run_lib(repo, "compose_file_args_for_tier gpu")
    assert chain.stdout == "-f compose.yaml -f compose.gpu.yaml"


def test_write_hardware_override_refuses_non_regular_file(repo: Path) -> None:
    # A broken symlink (target missing) is not a regular file: -L catches it, and
    # even without that the "exists-but-not-regular" branch would. Either way the
    # installer must decline rather than try to write.
    (repo / "compose.hardware.yaml").symlink_to(repo / "nonexistent-target.yaml")
    proc = run_lib(
        repo,
        "select_hardware_profile\nif write_hardware_override; then echo RC=0; else echo RC=$?; fi",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=1" in proc.stdout
    assert not (repo / "compose.hardware.yaml").exists()  # broken link still there, unresolved
    assert (repo / "compose.hardware.yaml").is_symlink()


def test_write_hardware_override_survives_repo_path_with_spaces(tmp_path: Path) -> None:
    # Regression: the compose-validation call embeds the absolute temp path. A repo
    # directory containing a space (e.g. macOS "~/My Projects/voxint") must not
    # word-split that -f argument. The validating fake docker fails if any -f path
    # does not resolve, so a split token would be caught here.
    spaced = tmp_path / "dir with spaces"
    (spaced / "scripts").mkdir(parents=True)
    shutil.copy(REAL_REPO / "scripts" / "install.sh", spaced / "scripts" / "install.sh")
    for name in ("compose.yaml", "compose.gpu.yaml", ".env.example"):
        shutil.copy(REAL_REPO / name, spaced / name)
    fakebin = spaced / "fakebin"
    fakebin.mkdir()
    docker = fakebin / "docker"
    docker.write_text(VALIDATING_FAKE_DOCKER)
    docker.chmod(0o755)

    env = os.environ.copy()
    env["VOXINT_INSTALL_LIB"] = "1"
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(spaced / "docker.log")
    env["VOXINT_NVIDIA_SMI"] = "/nonexistent/nvidia-smi"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{spaced}/scripts/install.sh"\n'
            "select_hardware_profile\nwrite_hardware_override\necho RC=$?",
        ],
        cwd=spaced,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RC=0" in proc.stdout
    assert (spaced / "compose.hardware.yaml").exists()
    log = (spaced / "docker.log").read_text()
    assert "config --quiet" in log  # validation actually ran and passed


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


def test_compose_file_args_appends_hardware_for_gpu_and_rocm(repo: Path) -> None:
    _write_managed_hardware_file(repo)
    assert (repo / "compose.hardware.yaml").exists()
    gpu = run_lib(repo, "compose_file_args_for_tier gpu")
    assert gpu.stdout == "-f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml"
    rocm = run_lib(repo, "compose_file_args_for_tier rocm")
    assert rocm.stdout == "-f compose.yaml -f compose.rocm.yaml -f compose.hardware.yaml"
    for tier, base in [
        ("cpu", "-f compose.yaml -f compose.cpu.yaml"),
        ("metal", "-f compose.yaml -f compose.metal.yaml"),
        ("none", "-f compose.yaml"),
    ]:
        proc = run_lib(repo, f"compose_file_args_for_tier {tier}")
        assert proc.stdout == base, f"{tier} must not fold in the hardware file"


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
    assert "# voxint:hardware-override v2" in proc.stdout
    assert 'MAX_PENDING_REQUESTS: "1"' in proc.stdout
    assert "BATCH_SIZE" not in proc.stdout
    # Human annotations on stderr; nothing written or started. The advertised
    # "effective chain" must include the hardware file a real install would fold
    # in -- even though it does not exist yet on this dry run.
    assert "dry run" in proc.stderr
    assert (
        "docker compose -f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml"
        in proc.stderr
    )
    assert not (repo / "compose.hardware.yaml").exists()
    assert not (repo / ".env").exists()


def test_hardware_dry_run_unknown_gpu_is_conservative(repo: Path) -> None:
    proc = run_script(repo, ["--hardware-dry-run"], extra_env=NO_NVIDIA)
    assert proc.returncode == 0, proc.stderr
    assert "# profile: nvidia-unknown-v1" in proc.stdout
    assert "--concurrency=1" in proc.stdout


def test_compose_file_args_skip_hardware_excludes_managed_file(repo: Path) -> None:
    # The kept-.env validation path asks for the chain WITHOUT the hardware file so
    # a stale managed file cannot fail that check before it is regenerated.
    _write_managed_hardware_file(repo)
    with_hw = run_lib(repo, "compose_file_args_for_tier gpu")
    assert with_hw.stdout == "-f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml"
    without_hw = run_lib(repo, "compose_file_args_for_tier gpu skip-hardware")
    assert without_hw.stdout == "-f compose.yaml -f compose.gpu.yaml"


def test_configure_hardware_defaults_folds_hardware_into_chain(repo: Path) -> None:
    # End-to-end: on a first GPU install, configure_hardware_defaults must write the
    # hardware file AND refold COMPOSE_FILE_ARGS so pull/up/handoff use it. This
    # pins the single line that wires the generated file into the effective chain;
    # the component tests recompute the args by hand and would not catch its loss.
    proc = run_lib(
        repo,
        "EFFECTIVE_TIER=gpu\nCOMPOSE_FILE_ARGS=$(compose_file_args_for_tier gpu)\n"
        "configure_hardware_defaults\n"
        'printf "ARGS=%s" "$COMPOSE_FILE_ARGS"',
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert (repo / "compose.hardware.yaml").exists()
    assert proc.stdout == "ARGS=-f compose.yaml -f compose.gpu.yaml -f compose.hardware.yaml"
    # Honest UX: caps are announced as applied only after a successful write.
    assert "Profile 'nvidia-unknown-v1' applied" in proc.stderr


def test_configure_hardware_defaults_reports_not_applied_when_refused(repo: Path) -> None:
    # An operator-authored (unmarked) file is left as-is, is NOT folded into the
    # chain, and the copy must say so rather than implying the caps are active.
    (repo / "compose.hardware.yaml").write_text("services: {}\n")
    proc = run_lib(
        repo,
        "EFFECTIVE_TIER=gpu\nCOMPOSE_FILE_ARGS=$(compose_file_args_for_tier gpu)\n"
        "configure_hardware_defaults\n"
        'printf "ARGS=%s" "$COMPOSE_FILE_ARGS"',
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "were NOT applied" in proc.stderr
    assert "Profile 'nvidia-unknown-v1' applied" not in proc.stderr  # no false claim
    assert proc.stdout == "ARGS=-f compose.yaml -f compose.gpu.yaml"


def test_configure_hardware_defaults_noop_on_non_gpu_tier(repo: Path) -> None:
    # Non-GPU tiers must not detect, write, or announce anything.
    proc = run_lib(
        repo,
        "EFFECTIVE_TIER=cpu\nconfigure_hardware_defaults\necho DONE",
        extra_env=NO_NVIDIA,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DONE" in proc.stdout
    assert not (repo / "compose.hardware.yaml").exists()
    assert "conservative GPU defaults" not in proc.stderr.lower()
