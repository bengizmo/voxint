"""Pure helper coverage for media-library search and status filters."""

import pytest

from voxint.api.media_query import _escape_like, status_is_known


@pytest.mark.parametrize(
    ("term", "escaped"),
    [
        ("100%", r"100\%"),
        ("file_name", r"file\_name"),
        (r"folder\file", r"folder\\file"),
        (r"100%_folder\file", r"100\%\_folder\\file"),
    ],
)
def test_escape_like(term: str, escaped: str) -> None:
    assert _escape_like(term) == escaped


@pytest.mark.parametrize("status", ["needs_review", "failed", "reviewed"])
def test_status_is_known_accepts_supported_statuses(status: str) -> None:
    assert status_is_known(status)


@pytest.mark.parametrize("status", [None, "invalid", ""])
def test_status_is_known_rejects_unsupported_statuses(status: str | None) -> None:
    assert not status_is_known(status)
