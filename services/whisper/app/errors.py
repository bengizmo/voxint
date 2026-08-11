"""Structured error bodies — torch-free by design.

Every non-422 error is ``{"detail": {"code", "message", "retryable"}}`` with the
stable codes from docs/gpu-contracts.md. Duplicated across the three GPU
services on purpose — each image is self-contained.
"""

from fastapi import HTTPException


def http_error(status_code: int, code: str, message: str, *, retryable: bool) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retryable},
    )


def path_violation(message: str) -> HTTPException:
    return http_error(400, "path_violation", message, retryable=False)


def invalid_media(message: str) -> HTTPException:
    return http_error(400, "invalid_media", message, retryable=False)


def file_not_found(message: str) -> HTTPException:
    return http_error(404, "file_not_found", message, retryable=False)


def model_unavailable(message: str = "Model not loaded") -> HTTPException:
    return http_error(503, "model_unavailable", message, retryable=True)


def saturated(message: str = "Service at capacity; retry later") -> HTTPException:
    return http_error(503, "saturated", message, retryable=True)


def inference_failed(message: str) -> HTTPException:
    return http_error(500, "inference_failed", message, retryable=False)
