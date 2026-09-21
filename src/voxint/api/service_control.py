"""Model-service lifecycle control (#556).

Lets the console start, stop, and restart model services (whisper, pyannote, titanet) from the
Status page. The controller dispatches to the right backend based on the
``VOXINT_SERVICE_CONTROL`` setting:

- ``docker``: talks to the Docker Engine API over a Unix socket (opt-in via
  ``compose.service-controls.yaml``).
- ``launchd``: reserved for native macOS installs (not yet implemented).
- unset/empty: controls disabled; the Status page shows terminal hints instead.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Protocol

import httpx

from voxint.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service registry
# ---------------------------------------------------------------------------

# Builds on the service identity in health_probe._SERVICES without importing
# it (that tuple is (human_label, settings_attr); we need compose names and
# stable URL-safe keys too).  Keep the two in sync via contract tests.

SERVICE_KEYS = frozenset({"transcription", "diarization", "speaker_embedding"})


@dataclass(frozen=True)
class ServiceDef:
    """One controllable model service."""

    key: str  # URL-safe key: transcription, diarization, speaker_embedding
    compose_service: str  # Docker Compose service name
    label: str  # friendly display name
    settings_url_attr: str  # Settings attribute holding the base URL
    launchd_label: str  # macOS launchd job label


SERVICES: dict[str, ServiceDef] = {
    "transcription": ServiceDef(
        key="transcription",
        compose_service="whisper",
        label="Transcriber",
        settings_url_attr="asr_url",
        launchd_label="com.voxint.metal.whisper",
    ),
    "diarization": ServiceDef(
        key="diarization",
        compose_service="pyannote",
        label="Voice separation",
        settings_url_attr="diarizer_url",
        launchd_label="com.voxint.metal.pyannote",
    ),
    "speaker_embedding": ServiceDef(
        key="speaker_embedding",
        compose_service="titanet",
        label="Voice identity",
        settings_url_attr="embedder_url",
        launchd_label="com.voxint.metal.titanet",
    ),
}


# ---------------------------------------------------------------------------
# Control result types
# ---------------------------------------------------------------------------


class ControlOutcome(enum.Enum):
    RESTARTED = "restarted"
    STARTED = "started"
    STOPPED = "stopped"
    ALREADY_RUNNING = "already_running"
    ALREADY_STOPPED = "already_stopped"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"
    BUSY = "busy"


@dataclass(frozen=True)
class ControlResult:
    outcome: ControlOutcome
    detail: str


class ServiceState(enum.Enum):
    RUNNING = "running"
    RESTARTING = "restarting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Controller protocol
# ---------------------------------------------------------------------------


class ServiceController(Protocol):
    @property
    def controllable(self) -> bool: ...

    @property
    def backend_name(self) -> str: ...

    def start(self, service_key: str) -> ControlResult: ...

    def stop(self, service_key: str) -> ControlResult: ...

    def restart(self, service_key: str) -> ControlResult: ...

    def inspect(self, service_key: str) -> ServiceState: ...

    def terminal_hint(self, service_key: str, action: str = "restart") -> str | None: ...


# ---------------------------------------------------------------------------
# Noop controller (unsupported deployment modes)
# ---------------------------------------------------------------------------


class NoopController:
    """Returns unavailable for all operations, with helpful terminal hints."""

    def __init__(self, reason: str, *, install_kind: str = "unknown") -> None:
        self._reason = reason
        self._install_kind = install_kind

    @property
    def controllable(self) -> bool:
        return False

    @property
    def backend_name(self) -> str:
        return "none"

    def start(self, service_key: str) -> ControlResult:
        return ControlResult(ControlOutcome.UNAVAILABLE, self._reason)

    def stop(self, service_key: str) -> ControlResult:
        return ControlResult(ControlOutcome.UNAVAILABLE, self._reason)

    def restart(self, service_key: str) -> ControlResult:
        return ControlResult(ControlOutcome.UNAVAILABLE, self._reason)

    def inspect(self, service_key: str) -> ServiceState:
        return ServiceState.UNKNOWN

    def terminal_hint(self, service_key: str, action: str = "restart") -> str | None:
        svc = SERVICES.get(service_key)
        if svc is None:
            return None
        if self._install_kind == "native":
            if action == "start":
                return (
                    f"launchctl bootstrap gui/$(id -u) "
                    f"~/.voxint-metal/run/{svc.launchd_label}.plist"
                )
            if action == "stop":
                return f"launchctl bootout gui/$(id -u)/{svc.launchd_label}"
            return f"launchctl kickstart -k gui/$(id -u)/{svc.launchd_label}"
        return f"docker compose {action} {svc.compose_service}"


# ---------------------------------------------------------------------------
# Docker controller (via Docker Engine API over Unix socket)
# ---------------------------------------------------------------------------

_DOCKER_API_VERSION = "v1.43"
_GRACEFUL_SECONDS = 10
_CONTROL_TIMEOUT_SECONDS = 30


class DockerController:
    """Controls model-service containers via the Docker Engine API."""

    def __init__(
        self,
        socket_path: str,
        compose_project: str,
    ) -> None:
        self._socket_path = socket_path
        self._compose_project = compose_project
        self._locks: dict[str, threading.Lock] = {key: threading.Lock() for key in SERVICE_KEYS}

    @property
    def controllable(self) -> bool:
        return True

    @property
    def backend_name(self) -> str:
        return "docker"

    def _client(self, timeout: float = 10.0) -> httpx.Client:
        transport = httpx.HTTPTransport(uds=self._socket_path)
        return httpx.Client(
            transport=transport,
            base_url=f"http://docker/{_DOCKER_API_VERSION}",
            timeout=httpx.Timeout(timeout),
        )

    def _find_container(self, client: httpx.Client, compose_service: str) -> str | None:
        """Find the container ID for a compose service. Returns None if not
        found or ambiguous (multiple matches)."""
        filters = json.dumps(
            {
                "label": [
                    f"com.docker.compose.project={self._compose_project}",
                    f"com.docker.compose.service={compose_service}",
                ],
            }
        )
        resp = client.get("/containers/json", params={"all": "true", "filters": filters})
        resp.raise_for_status()
        containers = resp.json()
        if len(containers) != 1:
            return None
        container_id: str = containers[0]["Id"]
        return container_id

    def _container_state(self, client: httpx.Client, container_id: str) -> ServiceState:
        resp = client.get(f"/containers/{container_id}/json")
        if resp.status_code == 404:
            return ServiceState.UNKNOWN
        resp.raise_for_status()
        state = resp.json().get("State", {})
        if state.get("Restarting"):
            return ServiceState.RESTARTING
        if state.get("Running"):
            return ServiceState.RUNNING
        return ServiceState.STOPPED

    def start(self, service_key: str) -> ControlResult:
        if service_key not in SERVICE_KEYS:
            return ControlResult(ControlOutcome.ERROR, f"unknown service: {service_key}")

        svc = SERVICES[service_key]
        lock = self._locks[service_key]
        if not lock.acquire(blocking=False):
            return ControlResult(ControlOutcome.BUSY, f"{svc.label} operation already in progress")
        try:
            return self._do_start(svc)
        finally:
            lock.release()

    def stop(self, service_key: str) -> ControlResult:
        if service_key not in SERVICE_KEYS:
            return ControlResult(ControlOutcome.ERROR, f"unknown service: {service_key}")

        svc = SERVICES[service_key]
        lock = self._locks[service_key]
        if not lock.acquire(blocking=False):
            return ControlResult(ControlOutcome.BUSY, f"{svc.label} operation already in progress")
        try:
            return self._do_stop(svc)
        finally:
            lock.release()

    def restart(self, service_key: str) -> ControlResult:
        if service_key not in SERVICE_KEYS:
            return ControlResult(ControlOutcome.ERROR, f"unknown service: {service_key}")

        svc = SERVICES[service_key]
        lock = self._locks[service_key]
        if not lock.acquire(blocking=False):
            return ControlResult(ControlOutcome.BUSY, f"{svc.label} operation already in progress")
        try:
            return self._do_restart(svc)
        finally:
            lock.release()

    def _do_start(self, svc: ServiceDef) -> ControlResult:
        try:
            with self._client(timeout=_CONTROL_TIMEOUT_SECONDS) as client:
                container_id = self._find_container(client, svc.compose_service)
                if container_id is None:
                    return ControlResult(
                        ControlOutcome.ERROR,
                        f"could not find a unique {svc.compose_service} container "
                        f"in project {self._compose_project}",
                    )

                resp = client.post(f"/containers/{container_id}/start")
                if resp.status_code == 204:
                    logger.info(
                        "service_start service=%s container=%s outcome=started",
                        svc.key,
                        container_id[:12],
                    )
                    return ControlResult(ControlOutcome.STARTED, f"{svc.label} started")
                if resp.status_code == 304:
                    return ControlResult(
                        ControlOutcome.ALREADY_RUNNING, f"{svc.label} is already running"
                    )
                return ControlResult(
                    ControlOutcome.ERROR,
                    f"Docker API returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.TimeoutException:
            return ControlResult(
                ControlOutcome.TIMEOUT,
                f"{svc.label} start timed out after {_CONTROL_TIMEOUT_SECONDS}s",
            )
        except httpx.HTTPError as exc:
            return ControlResult(ControlOutcome.ERROR, f"Docker API error: {exc}")
        except (OSError, ValueError, KeyError) as exc:
            return ControlResult(ControlOutcome.ERROR, f"cannot reach Docker socket: {exc}")

    def _do_stop(self, svc: ServiceDef) -> ControlResult:
        try:
            with self._client(timeout=_CONTROL_TIMEOUT_SECONDS) as client:
                container_id = self._find_container(client, svc.compose_service)
                if container_id is None:
                    return ControlResult(
                        ControlOutcome.ERROR,
                        f"could not find a unique {svc.compose_service} container "
                        f"in project {self._compose_project}",
                    )

                resp = client.post(
                    f"/containers/{container_id}/stop",
                    params={"t": str(_GRACEFUL_SECONDS)},
                )
                if resp.status_code == 204:
                    logger.info(
                        "service_stop service=%s container=%s outcome=stopped",
                        svc.key,
                        container_id[:12],
                    )
                    return ControlResult(ControlOutcome.STOPPED, f"{svc.label} stopped")
                if resp.status_code == 304:
                    return ControlResult(
                        ControlOutcome.ALREADY_STOPPED, f"{svc.label} is already stopped"
                    )
                return ControlResult(
                    ControlOutcome.ERROR,
                    f"Docker API returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.TimeoutException:
            return ControlResult(
                ControlOutcome.TIMEOUT,
                f"{svc.label} stop timed out after {_CONTROL_TIMEOUT_SECONDS}s",
            )
        except httpx.HTTPError as exc:
            return ControlResult(ControlOutcome.ERROR, f"Docker API error: {exc}")
        except (OSError, ValueError, KeyError) as exc:
            return ControlResult(ControlOutcome.ERROR, f"cannot reach Docker socket: {exc}")

    def _do_restart(self, svc: ServiceDef) -> ControlResult:
        try:
            with self._client(timeout=_CONTROL_TIMEOUT_SECONDS) as client:
                container_id = self._find_container(client, svc.compose_service)
                if container_id is None:
                    return ControlResult(
                        ControlOutcome.ERROR,
                        f"could not find a unique {svc.compose_service} container "
                        f"in project {self._compose_project}",
                    )

                resp = client.post(
                    f"/containers/{container_id}/restart",
                    params={"t": str(_GRACEFUL_SECONDS)},
                )
                if resp.status_code == 204:
                    logger.info(
                        "service_restart service=%s container=%s outcome=restarted",
                        svc.key,
                        container_id[:12],
                    )
                    return ControlResult(ControlOutcome.RESTARTED, f"{svc.label} restarted")
                return ControlResult(
                    ControlOutcome.ERROR,
                    f"Docker API returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.TimeoutException:
            return ControlResult(
                ControlOutcome.TIMEOUT,
                f"{svc.label} restart timed out after {_CONTROL_TIMEOUT_SECONDS}s",
            )
        except httpx.HTTPError as exc:
            return ControlResult(ControlOutcome.ERROR, f"Docker API error: {exc}")
        except (OSError, ValueError, KeyError) as exc:
            return ControlResult(ControlOutcome.ERROR, f"cannot reach Docker socket: {exc}")

    def inspect(self, service_key: str) -> ServiceState:
        if service_key not in SERVICE_KEYS:
            return ServiceState.UNKNOWN
        svc = SERVICES[service_key]
        try:
            with self._client() as client:
                container_id = self._find_container(client, svc.compose_service)
                if container_id is None:
                    return ServiceState.UNKNOWN
                return self._container_state(client, container_id)
        except (httpx.HTTPError, OSError, ValueError, KeyError):
            return ServiceState.UNKNOWN

    def terminal_hint(self, service_key: str, action: str = "restart") -> str | None:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_controller(settings: Settings) -> ServiceController:
    """Build the right controller for the current deployment."""
    mode = (settings.voxint_service_control or "").strip().lower()

    if mode == "docker":
        socket_path = settings.voxint_docker_socket
        if not os.path.exists(socket_path):
            logger.warning(
                "VOXINT_SERVICE_CONTROL=docker but socket %s not found; service controls disabled",
                socket_path,
            )
            return NoopController(
                f"Docker socket not found at {socket_path}. "
                "Check compose.service-controls.yaml is included.",
                install_kind="docker",
            )
        if not _socket_accessible(socket_path):
            logger.warning(
                "VOXINT_SERVICE_CONTROL=docker but socket %s is not accessible; "
                "check group_add in compose.service-controls.yaml",
                socket_path,
            )
            return NoopController(
                f"Docker socket at {socket_path} is not accessible. "
                "Check DOCKER_GID in compose.service-controls.yaml.",
                install_kind="docker",
            )
        logger.info(
            "service_control backend=docker socket=%s project=%s",
            socket_path,
            settings.voxint_compose_project,
        )
        return DockerController(
            socket_path=socket_path,
            compose_project=settings.voxint_compose_project,
        )

    if mode == "launchd":
        logger.info("service_control backend=launchd (not yet implemented)")
        return NoopController(
            "launchd service controls are not yet implemented",
            install_kind="native",
        )

    if mode and mode not in ("docker", "launchd"):
        logger.warning(
            "VOXINT_SERVICE_CONTROL=%r is not a recognized value "
            "(expected 'docker' or 'launchd'); service controls disabled",
            settings.voxint_service_control,
        )

    from voxint.api.settings_view import install_kind

    kind = install_kind(settings)
    return NoopController(
        "Service controls are not enabled. See docs/service-controls.md for setup.",
        install_kind=kind,
    )


def _socket_accessible(path: str) -> bool:
    """Check if the Docker socket is readable/writable."""
    try:
        return os.access(path, os.R_OK | os.W_OK)
    except OSError:
        return False
