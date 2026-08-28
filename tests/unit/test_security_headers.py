"""The D1 security-header handler for unhandled 500s (`_security_headers_on_error`)
and the token-sensitive-path classifier (`_is_token_sensitive_path`).

Starlette's ServerErrorMiddleware wraps the app outside `_SecurityHeadersMiddleware`,
so unhandled 500s are covered by this handler instead -- unit-tested directly because
the middleware path is exercised end-to-end in the review-API integration tests.
"""

import pytest
from starlette.requests import Request

from voxint.api.app import _is_token_sensitive_path, _security_headers_on_error


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
        }
    )


# ---- _is_token_sensitive_path classifier ----


@pytest.mark.parametrize(
    "path",
    [
        "/review",
        "/review/abc123",
        "/review/abc123/transcript",
        "/review/abc123?token=t",
        "/media/00000000-0000-0000-0000-000000000001/editor",
        "/media/00000000-0000-0000-0000-000000000001/editor?run=x",
        "/media/AABBCCDD-1122-3344-5566-778899AABBCC/editor",
    ],
)
def test_token_sensitive_paths_match(path: str) -> None:
    assert _is_token_sensitive_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/media",
        "/media/submit",
        "/media/fetch",
        "/media/assign",
        "/media/folders",
        "/media/rerun",
        "/media/rerun/confirm",
        "/media/archive",
        "/media/unarchive",
        "/media/00000000-0000-0000-0000-000000000001",
        "/media/00000000-0000-0000-0000-000000000001/peaks",
        "/runs/abc123",
        "/settings",
        "/",
        "/media/not-a-uuid/editor",
    ],
)
def test_non_token_paths_do_not_match(path: str) -> None:
    assert _is_token_sensitive_path(path) is False


# ---- error handler ----


async def test_error_handler_stamps_referrer_policy_on_any_path() -> None:
    response = await _security_headers_on_error(_request("/runs/x"), RuntimeError("boom"))
    assert response.status_code == 500
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "cache-control" not in response.headers


async def test_error_handler_stamps_nosniff_on_any_path() -> None:
    response = await _security_headers_on_error(_request("/runs/x"), RuntimeError("boom"))
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_error_handler_adds_no_store_on_review_path() -> None:
    response = await _security_headers_on_error(
        _request("/review/abc123?token=t"), RuntimeError("boom")
    )
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"


async def test_error_handler_adds_no_store_on_media_editor_path() -> None:
    response = await _security_headers_on_error(
        _request("/media/00000000-0000-0000-0000-000000000001/editor"), RuntimeError("boom")
    )
    assert response.headers["cache-control"] == "no-store"


async def test_error_handler_no_store_absent_on_media_library_path() -> None:
    response = await _security_headers_on_error(
        _request("/media"), RuntimeError("boom")
    )
    assert "cache-control" not in response.headers
