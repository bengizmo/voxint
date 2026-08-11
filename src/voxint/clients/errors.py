"""Errors raised by the HTTP model-service clients.

The v1 GPU contracts define structured error bodies
(``{"detail": {"code": ..., "message": ..., "retryable": bool}}``); the clients
translate those — and anything non-conforming a proxy or dead socket produces —
into :class:`ServiceError` so callers branch on one ``retryable`` flag instead
of HTTP minutiae. Retry policy itself lives in the worker, not here.
"""

import httpx


class ServiceError(Exception):
    """A model-service call failed.

    ``retryable`` follows the service's own verdict when the body conforms to
    the error contract; otherwise 5xx/transport failures are presumed
    transient and 4xx are not.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code


class ProtocolError(ServiceError):
    """The service answered 2xx but violated the contract (bad shape, count
    mismatch, wrong embedding dimension). Never retryable — the same request
    will keep failing until a human looks at it."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(
            "protocol_violation", message, retryable=False, status_code=status_code
        )


def error_from_response(response: httpx.Response) -> ServiceError:
    """Map an error response to a ServiceError, honoring the structured body."""
    try:
        detail = response.json()["detail"]
        code, message, retryable = detail["code"], detail["message"], detail["retryable"]
        # Strict types or it's not the structured contract: bool("false") is
        # True, so coercion would flip retry semantics on a sloppy body.
        if (
            isinstance(code, str)
            and isinstance(message, str)
            and isinstance(retryable, bool)
        ):
            return ServiceError(
                code, message, retryable=retryable, status_code=response.status_code
            )
    except (ValueError, KeyError, TypeError):
        pass
    if response.status_code == 422:
        # FastAPI-native validation body — a request-shape bug, never transient.
        return ServiceError(
            "validation_error",
            response.text[:500],
            retryable=False,
            status_code=response.status_code,
        )
    return ServiceError(
        "http_error",
        f"HTTP {response.status_code}: {response.text[:500]}",
        retryable=response.status_code >= 500,
        status_code=response.status_code,
    )


def error_from_transport(exc: httpx.HTTPError) -> ServiceError:
    """Timeouts, refused connections, dead sockets — presumed transient."""
    return ServiceError(
        "transport_error", f"{type(exc).__name__}: {exc}", retryable=True
    )
