"""The D1 security-header handler for unhandled 500s (`_security_headers_on_error`).

Starlette's ServerErrorMiddleware wraps the app outside `_SecurityHeadersMiddleware`,
so unhandled 500s are covered by this handler instead — unit-tested directly because
the middleware path is exercised end-to-end in the review-API integration tests.
"""

from starlette.requests import Request

from voxint.api.app import _security_headers_on_error


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


async def test_error_handler_stamps_referrer_policy_on_any_path() -> None:
    response = await _security_headers_on_error(_request("/runs/x"), RuntimeError("boom"))
    assert response.status_code == 500
    assert response.headers["referrer-policy"] == "no-referrer"
    # Non-/review 500s are not forced no-store.
    assert "cache-control" not in response.headers


async def test_error_handler_stamps_nosniff_on_any_path() -> None:
    # Issue #103: X-Content-Type-Options: nosniff is a baseline header on every
    # response, so it must survive an unhandled 500 like the D1 referrer policy.
    response = await _security_headers_on_error(_request("/runs/x"), RuntimeError("boom"))
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_error_handler_adds_no_store_on_review_path() -> None:
    response = await _security_headers_on_error(
        _request("/review/abc123?token=t"), RuntimeError("boom")
    )
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
