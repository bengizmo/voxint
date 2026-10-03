"""Host broker hooks (all calls have a ten-second timeout).

POST acquire sends holder=voxint, ttl_seconds=max(10*tick_seconds, 600), and
lease_id (null for acquisition, current ID for renew-on-acquire). 200 returns
lease_id and optional ISO8601 expires_at; null/absent expiry means explicit
release only. 202 state=pending requests another tick; 409 holder plus optional
retry_after_seconds means busy. Renewal 404/409 means lost.
POST release sends lease_id; 200 and 404 both succeed. 202 state=pending means the
broker is still restoring the other service: ask again next tick. GET status returns
state=free|held, holder, lease_id and optional expires_at.
Optional bearer authorization is never copied into diagnostics. Bodies and
exception messages are deliberately excluded from failure reasons.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import httpx

from voxint.config import Settings

HOOK_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class AcquireResult:
    kind: Literal["held", "pending", "busy", "lost", "failed"]
    lease_id: str | None = field(default=None, repr=False)
    expires_at: datetime | None = None
    retry_after_seconds: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ReleaseResult:
    kind: Literal["released", "pending", "failed"]
    reason: str = ""


@dataclass(frozen=True)
class StatusResult:
    kind: Literal["free", "held", "failed"]
    holder: str | None = field(default=None, repr=False)
    lease_id: str | None = field(default=None, repr=False)
    expires_at: datetime | None = None
    reason: str = ""


def _expiry(body: dict[str, object]) -> datetime | None:
    value = body.get("expires_at")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError
    return parsed.astimezone(UTC)


def _string(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError
    return value


class LeaseClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._transport = transport
        self.ttl_seconds = max(10 * settings.gpu_phase_tick_seconds, 600)

    def _request(
        self,
        method: str,
        url: str,
        body: dict[str, object] | None = None,
        *,
        body_optional: bool = False,
    ) -> tuple[int, dict[str, object]]:
        headers = {}
        if self._settings.gpu_lease_token:
            headers["Authorization"] = f"Bearer {self._settings.gpu_lease_token}"

        def send(client: httpx.Client) -> httpx.Response:
            return client.request(
                method, url, json=body, headers=headers, timeout=HOOK_TIMEOUT_SECONDS
            )

        if self._client is not None:
            response = send(self._client)
        else:
            with httpx.Client(transport=self._transport) as client:
                response = send(client)
        # Release, unexpected status codes and renewal loss need no JSON body.
        if (
            body_optional
            or response.status_code not in (200, 202, 409)
            or (
                response.status_code == 409
                and body is not None
                and body.get("lease_id") is not None
            )
        ):
            return response.status_code, {}
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            if response.status_code != 200:
                return response.status_code, {}
            raise
        return response.status_code, data

    def acquire(self, lease_id: str | None = None) -> AcquireResult:
        code: int | None = None
        try:
            code, body = self._request(
                "POST",
                self._settings.gpu_lease_acquire_url,
                {
                    "holder": "voxint",
                    "ttl_seconds": self.ttl_seconds,
                    "lease_id": lease_id,
                },
            )
            if lease_id is not None and code in (404, 409):
                return AcquireResult("lost", reason=f"lease lost (HTTP {code})")
            if code == 200:
                return AcquireResult("held", _string(body.get("lease_id")), _expiry(body))
            if code == 202 and body.get("state") == "pending":
                return AcquireResult("pending")
            if code == 409:
                _string(body.get("holder"))
                delay = body.get("retry_after_seconds", 0)
                if type(delay) is not int or delay < 0:
                    raise ValueError
                return AcquireResult(
                    "busy", retry_after_seconds=delay, reason=f"GPU is busy (HTTP {code})"
                )
            return AcquireResult("failed", reason=f"unexpected broker response (HTTP {code})")
        except Exception:
            return AcquireResult(
                "failed",
                reason=(
                    f"invalid broker response (HTTP {code})"
                    if code is not None and code != 200
                    else "broker unavailable or invalid response"
                ),
            )

    def release(self, lease_id: str) -> ReleaseResult:
        try:
            code, _ = self._request(
                "POST",
                self._settings.gpu_lease_release_url,
                {"lease_id": lease_id},
                body_optional=True,
            )
            if code in (200, 404):
                return ReleaseResult("released")
            if code == 202:
                # The broker is still bringing the other service back up.
                return ReleaseResult("pending")
            return ReleaseResult("failed", f"broker refused release (HTTP {code})")
        except Exception:
            return ReleaseResult("failed", "broker unavailable or invalid response")

    def status(self) -> StatusResult:
        code: int | None = None
        try:
            code, body = self._request("GET", self._settings.gpu_lease_status_url)
            if code == 200:
                expiry = _expiry(body)
                if (
                    body.get("state") == "free"
                    and body.get("holder") is None
                    and body.get("lease_id") is None
                ):
                    return StatusResult("free")
                if body.get("state") == "held":
                    return StatusResult(
                        "held", _string(body.get("holder")), _string(body.get("lease_id")), expiry
                    )
            return StatusResult("failed", reason=f"unexpected broker response (HTTP {code})")
        except Exception:
            return StatusResult(
                "failed",
                reason=(
                    f"invalid broker response (HTTP {code})"
                    if code is not None and code != 200
                    else "broker unavailable or invalid response"
                ),
            )
