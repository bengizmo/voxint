"""Unit tests for the _validate_next open-redirect defense (issue #304)."""

import pytest

from voxint.api.routers.auth_pages import _validate_next


@pytest.mark.parametrize(
    ("input_url", "expected"),
    [
        # Valid local paths pass through.
        ("/", "/"),
        ("/review", "/review"),
        ("/runs/abc?tab=2", "/runs/abc?tab=2"),
        ("/settings#synthdetect", "/settings#synthdetect"),
        # None / empty / non-slash → fallback.
        (None, "/"),
        ("", "/"),
        ("http://evil.com", "/"),
        ("evil.com", "/"),
        ("javascript:alert(1)", "/"),
        # Double-slash (protocol-relative).
        ("//evil.com", "/"),
        ("//evil.com/path", "/"),
        # Backslash bypass: browsers normalize \ to /, producing //evil.com.
        ("/\\evil.com", "/"),
        ("/foo\\bar", "/"),
        # Control characters (response splitting / header injection).
        ("/foo\r\nLocation: evil.com", "/"),
        ("/foo\x00bar", "/"),
        ("/foo\x01", "/"),
        ("/foo\x1f", "/"),
        # Scheme-bearing paths that urlparse would parse with a scheme.
        ("http:///evil.com", "/"),
        ("https://evil.com/path", "/"),
        ("ftp://evil.com", "/"),
    ],
)
def test_validate_next(input_url: str | None, expected: str) -> None:
    assert _validate_next(input_url) == expected
