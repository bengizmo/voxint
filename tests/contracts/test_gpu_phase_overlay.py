"""The GPU phase overlay grants the Docker socket to the gpu-phase worker only (#748).

``compose.gpu-phase.yaml`` is the second, deliberate Docker socket grant in the
repo (the first is ``compose.service-controls.yaml``, pinned by
``test_service_control_overlay.py``). It adds a dedicated ``gpu-phase`` Celery
worker that starts and stops the model services. It leaves the regular worker
alone: the app does not declare the ``gpu_phase`` queue, so a worker started
without ``-Q`` never consumes an orchestrator tick.

These are YAML-level checks, like the service-controls contract: CI has no
Docker, so ``docker compose config`` is not run here. ``gpu-phase`` is built with
``extends`` from the ``worker`` service in ``compose.yaml``, so its effective
fields are that base service plus the overlay entry; the tests resolve it that
way.
"""

from __future__ import annotations

import shlex
from typing import Any

import pytest
import yaml

from tests.contracts.conftest import REPO_ROOT
from voxint.worker.app import POST_QUEUE, app

OVERLAY = "compose.gpu-phase.yaml"
SERVICE_CONTROLS = "compose.service-controls.yaml"
SOCKET = "/var/run/docker.sock:/var/run/docker.sock"
GPU_PHASE_QUEUE = "gpu_phase"
# Tiers whose model services are containers the gpu-phase process can control.
CONTAINER_TIERS = ("gpu", "rocm", "cpu")


def _services(filename: str) -> dict[str, Any]:
    services = yaml.safe_load((REPO_ROOT / filename).read_text())["services"]
    assert isinstance(services, dict)
    return services


def _flag(argv: list[str], *names: str) -> str | None:
    """Value of a CLI flag given as ``-X value``, ``--name value`` or ``--name=value``."""
    found = None
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name:
                assert index + 1 < len(argv), f"{name} has no value in {argv}"
                assert found is None, f"{name} given twice in {argv}"
                found = argv[index + 1]
            elif name.startswith("--") and arg.startswith(f"{name}="):
                assert found is None, f"{name} given twice in {argv}"
                found = arg.split("=", 1)[1]
    return found


def _queues(command: str) -> set[str]:
    argv = shlex.split(command)
    assert argv[:4] == ["celery", "-A", "voxint.worker.app", "worker"], argv
    value = _flag(argv, "-Q", "--queues")
    assert value is not None, f"worker command consumes every queue (no -Q): {command}"
    return set(value.split(","))


def _layers(tier: str, *, service_controls: bool) -> list[str]:
    files = ["compose.yaml", f"compose.{tier}.yaml"]
    if service_controls:
        files.append(SERVICE_CONTROLS)
    files.append(OVERLAY)
    return files


def test_overlay_shape() -> None:
    services = _services(OVERLAY)
    # The overlay adds gpu-phase and touches no other service, so it composes
    # with the installer's compose.hardware.yaml worker command in any order.
    assert set(services) == {"gpu-phase"}

    gpu_phase = services["gpu-phase"]
    # Inherit from the base worker so the image pin, .env, settings, media volume
    # and dependencies have one source; the overlay adds no image of its own.
    assert gpu_phase["extends"] == {"file": "compose.yaml", "service": "worker"}
    assert "image" not in gpu_phase and "build" not in gpu_phase
    assert gpu_phase["restart"] == "unless-stopped"
    assert gpu_phase["volumes"] == [SOCKET]
    assert gpu_phase["group_add"] == ["${DOCKER_GID:-999}"]
    # The host broker is reachable by a stable name on Linux too.
    assert gpu_phase["extra_hosts"] == ["host.docker.internal:host-gateway"]
    environment = gpu_phase["environment"]
    assert environment["VOXINT_SERVICE_CONTROL"] == "docker"
    # DockerController finds model containers by compose project label; the
    # project this container runs in is where they live.
    assert environment["VOXINT_COMPOSE_PROJECT"] == "${COMPOSE_PROJECT_NAME:-voxint}"
    # No GPU for the orchestrator itself.
    assert "deploy" not in gpu_phase and "devices" not in gpu_phase


def test_flagless_workers_never_consume_gpu_phase_ticks() -> None:
    # A worker without -Q consumes exactly the declared queues. gpu_phase must
    # stay undeclared so only the dedicated gpu-phase worker runs ticks.
    declared = {queue.name for queue in app.conf.task_queues}
    assert GPU_PHASE_QUEUE not in declared
    assert {"celery", POST_QUEUE} <= declared
    assert app.conf.task_create_missing_queues is True  # -Q gpu_phase creates it


def test_gpu_phase_consumes_only_its_queue_at_concurrency_one() -> None:
    command = _services(OVERLAY)["gpu-phase"]["command"]
    assert _queues(command) == {GPU_PHASE_QUEUE}
    argv = shlex.split(command)
    assert _flag(argv, "-c", "--concurrency") == "1"
    node = _flag(argv, "-n", "--hostname")
    assert node is not None and node.startswith("gpu-phase@"), node


@pytest.mark.parametrize("tier", CONTAINER_TIERS)
def test_tier_overlays_leave_the_overlay_in_charge(tier: str) -> None:
    tier_services = _services(f"compose.{tier}.yaml")
    # A tier entry named gpu-phase would merge into it (GPU reservations, tier
    # environment).
    assert "gpu-phase" not in tier_services
    # extends reads compose.yaml only, so tier fields never reach gpu-phase;
    # the base worker it copies must carry no GPU access of its own.
    base_worker = _services("compose.yaml")["worker"]
    assert "deploy" not in base_worker and "devices" not in base_worker
    # The readiness probe needs the same model-service URLs the tier gives the
    # worker.
    gpu_phase_env = _services(OVERLAY)["gpu-phase"]["environment"]
    for key in ("ASR_URL", "DIARIZER_URL", "EMBEDDER_URL"):
        assert gpu_phase_env[key] == tier_services["worker"]["environment"][key], key


@pytest.mark.parametrize("service_controls", [False, True])
@pytest.mark.parametrize("tier", [*CONTAINER_TIERS, "metal"])
def test_socket_is_gpu_phase_only_across_tiers(tier: str, service_controls: bool) -> None:
    # Metal is documented as unsupported (its model services are not
    # containers), but layering the overlay there by mistake must still not
    # widen the grant.
    files = _layers(tier, service_controls=service_controls)
    layers = {filename: _services(filename) for filename in files}
    base = layers["compose.yaml"]
    assert {"api", "worker", "beat"} <= base.keys()
    for filename in files[1:-1]:
        assert "gpu-phase" not in layers[filename], filename

    # Effective gpu-phase: the extended base worker, then the overlay entry.
    # Compose concatenates volume lists and merges environment mappings.
    extended = base[layers[OVERLAY]["gpu-phase"]["extends"]["service"]]
    gpu_phase_volumes = [*extended.get("volumes", []), *layers[OVERLAY]["gpu-phase"]["volumes"]]
    gpu_phase_environment = {
        **extended.get("environment", {}),
        **layers[OVERLAY]["gpu-phase"]["environment"],
    }
    assert gpu_phase_volumes.count(SOCKET) == 1
    assert len(gpu_phase_volumes) > 1  # The media volume survives the added socket.
    assert gpu_phase_environment["VOXINT_SERVICE_CONTROL"] == "docker"

    granted = {"gpu-phase", "api"} if service_controls else {"gpu-phase"}
    api_socket_mounts = 0
    for filename, services in layers.items():
        for name, service in services.items():
            mounts = [v for v in service.get("volumes", []) if "docker.sock" in str(v)]
            if name == "api":
                api_socket_mounts += len(mounts)
                continue
            if name == "gpu-phase":
                continue
            assert not mounts, f"{filename}:{name}"
            assert "VOXINT_SERVICE_CONTROL" not in service.get("environment", {}), (
                f"{filename}:{name}"
            )
            assert "${DOCKER_GID:-999}" not in service.get("group_add", []), f"{filename}:{name}"
    assert api_socket_mounts == (1 if "api" in granted else 0)
