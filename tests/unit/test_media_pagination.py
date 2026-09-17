"""Media pagination token and URL contracts."""

import base64
import uuid
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlsplit

import pytest

from voxint.api.media_query import InvalidMediaCursorError, MediaCursor, media_url
from voxint.config import Settings


@pytest.mark.parametrize(
    "sort,value",
    [
        ("added", "2026-09-17T00:00:00+00:00"),
        ("name", "a café"),
        ("name", "artist | episode | track"),
        ("name", "pipes|||everywhere"),
        ("duration", "1.5"),
        ("size", "42"),
        ("duration", "__null__"),
        ("size", "__null__"),
    ],
)
def test_cursor_roundtrip(sort: str, value: str) -> None:
    cursor = MediaCursor(sort, value, datetime.now(UTC), uuid.uuid4())
    assert MediaCursor.decode(cursor.encode()) == cursor


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "added|value",
        "unknown|value|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
        "added|value|2026-09-17T00:00:00|" + str(uuid.uuid4()),
        "added|value|bad-date|" + str(uuid.uuid4()),
        "added|value|2026-09-17T00:00:00+00:00|bad-id",
        "duration|notanumber|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
        "duration|inf|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
        "duration|nan|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
        "size|1.5|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
        "size|notanumber|2026-09-17T00:00:00+00:00|" + str(uuid.uuid4()),
    ],
)
def test_invalid_cursor_fields(raw: str) -> None:
    with pytest.raises(InvalidMediaCursorError):
        MediaCursor.decode(base64.urlsafe_b64encode(raw.encode()).decode())


def test_invalid_base64() -> None:
    with pytest.raises(InvalidMediaCursorError):
        MediaCursor.decode("not base64")


def test_url_preserves_filters() -> None:
    cursor = MediaCursor("name", "a & b", datetime.now(UTC), uuid.uuid4())
    url = media_url(
        sort="name",
        view="cards",
        archived=True,
        trashed=True,
        search="a & b",
        status="failed",
        open_folder="/a b/",
        cursor=cursor,
    )
    assert urlsplit(url).path == "/media"
    assert parse_qs(urlsplit(url).query) == {
        "sort": ["name"],
        "view": ["cards"],
        "archived": ["1"],
        "trashed": ["1"],
        "q": ["a & b"],
        "status": ["failed"],
        "open": ["/a b/"],
        "cursor": [cursor.encode()],
    }
    assert media_url() == "/media?sort=added&view=table"


@pytest.mark.parametrize("size", [0, 501])
def test_page_size_bounds(size: int) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, media_page_size=size)
