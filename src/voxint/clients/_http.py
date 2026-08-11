"""Shared plumbing for the model-service HTTP clients.

One request = one synchronous inference run (see docs/gpu-contracts.md), so the
read timeout must accommodate media-length work — hours, not seconds. Requests
send MEDIA_ROOT-relative paths; the pipeline side holds absolute paths, so the
translation (and the escape check) happens here, before any bytes move.
"""

import math
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from voxint.clients.errors import ServiceError, error_from_response, error_from_transport

CONNECT_TIMEOUT_SECONDS = 10.0


def finite_interval(
    start: object, end: object, *, zero_length_ok: bool = False
) -> tuple[float, float]:
    """Validate a (start, end) seconds pair from a service response.

    NaN/inf or a reversed interval would otherwise surface later as a DB
    constraint error attributed to the wrong stage — reject at the seam.
    """
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("interval bounds must be numbers")
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        raise ValueError("interval bounds must be numbers")
    start_f, end_f = float(start), float(end)
    if not (math.isfinite(start_f) and math.isfinite(end_f)):
        raise ValueError("interval bounds must be finite")
    if start_f < 0 or (end_f < start_f if zero_length_ok else end_f <= start_f):
        raise ValueError(f"invalid interval ({start_f}, {end_f})")
    return start_f, end_f


class ServiceHttpClient:
    """Base for the concrete clients: owns transport, paths, and error mapping.

    Pass ``client`` to share a long-lived ``httpx.Client`` (it must carry the
    service's ``base_url``); the instance then never closes it. Otherwise one
    is created and owned here — call :meth:`close` (or use as a context
    manager) from the owning worker process.
    """

    def __init__(
        self,
        base_url: str,
        media_root: Path,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._media_root = media_root.resolve()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=timeout_seconds,
                write=timeout_seconds,
                pool=timeout_seconds,
            ),
        )

    def relative_path(self, audio_path: Path) -> str:
        """Translate an absolute pipeline path to the contract's relative form."""
        if not audio_path.is_absolute():
            raise ServiceError(
                "path_violation",
                f"audio path must be absolute: {audio_path}",
                retryable=False,
            )
        try:
            return audio_path.resolve().relative_to(self._media_root).as_posix()
        except ValueError as exc:
            raise ServiceError(
                "path_violation",
                f"{audio_path} escapes media root {self._media_root}",
                retryable=False,
            ) from exc

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise error_from_transport(exc) from exc
        if response.status_code >= 400:
            raise error_from_response(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise ServiceError(
                "protocol_violation",
                f"non-JSON 2xx body from {path}",
                retryable=False,
                status_code=response.status_code,
            ) from exc
        if not isinstance(body, dict):
            raise ServiceError(
                "protocol_violation",
                f"expected JSON object from {path}, got {type(body).__name__}",
                retryable=False,
                status_code=response.status_code,
            )
        return body

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ServiceHttpClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
