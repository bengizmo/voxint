"""Cross-file contracts for the native (no-Docker) macOS launcher (#69).

The native launcher bakes command strings and model-service ports that MUST
stay in lock-step with two other sources of truth, or a native preview silently
diverges from the shipped stack:

  * ``compose.yaml`` — the api/worker/beat commands the Docker stack runs.
  * ``scripts/metal/voxint-metal.sh`` — the model launcher the native launcher
    delegates to, which owns the whisper/pyannote/titanet ports and shares the
    MEDIA_ROOT resolution both worlds must agree on.

These are drift guards, promoted from the unit suite to the contract layer where
the pin-parity invariants live. They drive each launcher through its
``*_LIB=1`` sourcing seam — no launchd, Postgres/Redis, network, or Docker.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "scripts" / "native" / "voxint-native.sh"
METAL = REPO / "scripts" / "metal" / "voxint-metal.sh"
COMPOSE = REPO / "compose.yaml"


def _run(script_path: Path, lib_var: str, snippet: str, env: dict[str, str]) -> str:
    full_env = os.environ.copy()
    full_env[lib_var] = "1"
    full_env.update(env)
    proc = subprocess.run(
        ["bash", "-c", f'source "{script_path}"\n{snippet}'],
        env=full_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def native(snippet: str, home: Path) -> str:
    return _run(NATIVE, "VOXINT_NATIVE_LIB", snippet, {"VOXINT_NATIVE_HOME": str(home)})


def metal(snippet: str, home: Path) -> str:
    return _run(METAL, "VOXINT_METAL_LIB", snippet, {"VOXINT_METAL_HOME": str(home)})


def native_argv(snippet: str, home: Path) -> list[str]:
    return [ln for ln in native(snippet, home).splitlines() if ln != ""]


def native_env(snippet: str, home: Path) -> dict[str, str]:
    return dict(
        ln.split("=", 1) for ln in native(snippet, home).splitlines() if "=" in ln
    )


# --------------------------------------------------------------------------- #
# Native argv ↔ compose.yaml
# --------------------------------------------------------------------------- #
def test_core_commands_match_compose(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(COMPOSE.read_text())

    def compose_cmd(service: str) -> list[str]:
        return str(doc["services"][service]["command"]).split()

    assert compose_cmd("api") == ["voxint", "serve"]
    api = native_argv("native_program_args api", tmp_path)
    assert api[0].endswith("/bin/voxint") and api[-1] == "serve"

    # No -Q flag is load-bearing since the two-lane split: a worker started
    # without one consumes every queue declared in task_queues (celery + post),
    # which is what keeps single-worker deployments whole-pipeline by default.
    assert compose_cmd("worker") == [
        "celery",
        "-A",
        "voxint.worker.app",
        "worker",
        "--loglevel=INFO",
    ]
    worker = native_argv("native_program_args worker", tmp_path)
    assert worker[1:] == ["-A", "voxint.worker.app", "worker", "--loglevel=INFO"]

    # beat: same app + subcommand; the -s schedule path legitimately differs
    # (compose writes /tmp, native writes under its own home).
    assert compose_cmd("beat")[:5] == [
        "celery",
        "-A",
        "voxint.worker.app",
        "beat",
        "--loglevel=INFO",
    ]
    beat = native_argv("native_program_args beat", tmp_path)
    assert beat[1:6] == ["-A", "voxint.worker.app", "beat", "--loglevel=INFO", "-s"]


# --------------------------------------------------------------------------- #
# Native model URLs ↔ metal launcher ports (the delegation target)
# --------------------------------------------------------------------------- #
def test_model_urls_match_metal_launcher_ports(tmp_path: Path) -> None:
    def metal_port(svc: str) -> str:
        return metal(f"service_port {svc}", tmp_path).strip()

    env = native_env("native_service_env api /media/root", tmp_path)
    assert env["ASR_URL"].rsplit(":", 1)[1] == metal_port("whisper")
    assert env["DIARIZER_URL"].rsplit(":", 1)[1] == metal_port("pyannote")
    assert env["EMBEDDER_URL"].rsplit(":", 1)[1] == metal_port("titanet")


def test_native_delegates_to_the_real_metal_launcher(tmp_path: Path) -> None:
    # The delegation target the native launcher will drive must be the committed
    # metal launcher — a renamed/moved metal script would orphan the models.
    resolved = native("metal_script", tmp_path).strip()
    assert resolved == str(METAL)
    assert Path(resolved).is_file()


# --------------------------------------------------------------------------- #
# MEDIA_ROOT parity — both launchers must land on the same physical dir
# --------------------------------------------------------------------------- #
def test_media_root_resolution_agrees(tmp_path: Path) -> None:
    # Both launchers read $REPO_ROOT/.env's MEDIA_ROOT and resolve it against the
    # repo root with `pwd -P`; a relative path must resolve identically in both,
    # or the worker's MEDIA_ROOT-relative paths break across the two tiers.
    media = tmp_path / "media dir"  # a space exercises the quoting on both sides
    media.mkdir()
    rel = os.path.relpath(media, REPO)
    n = native(f'resolve_media_root "{rel}"', tmp_path).strip()
    m = metal(f'resolve_media_root "{rel}"', tmp_path).strip()
    assert n == m == str(media.resolve())
