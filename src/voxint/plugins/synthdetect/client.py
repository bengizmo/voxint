"""HTTP client for the synthdetect model service."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SynthdetectServiceError(Exception):
    """The synthdetect service returned a non-200 response or is unreachable."""


class HttpSynthdetectClient:
    """Thin httpx wrapper around the synthdetect service API."""

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def healthz(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(f"{self._base_url}/healthz")
            resp.raise_for_status()
            return dict(resp.json())

    def score(
        self,
        path: str,
        intervals: list[dict[str, float]],
    ) -> dict[str, Any]:
        """POST /v1/score and return the parsed response.

        ``intervals`` is a list of ``{"start_seconds": ..., "end_seconds": ...}``.
        """
        body = {"path": path, "intervals": intervals}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/v1/score", json=body)
                resp.raise_for_status()
                return dict(resp.json())
        except httpx.HTTPStatusError as exc:
            raise SynthdetectServiceError(
                f"synthdetect service returned {exc.response.status_code}"
            ) from exc
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise SynthdetectServiceError(
                f"synthdetect service unreachable: {type(exc).__name__}"
            ) from exc
