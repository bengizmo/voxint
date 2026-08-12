"""Pure logic of the /runs browser: cursor codec, filter parsing, URL building."""

import uuid
from datetime import UTC, datetime

import pytest

from voxint.api.runs_query import (
    Cursor,
    InvalidCursorError,
    ReviewFilter,
    parse_review_filter,
    parse_status_filter,
    runs_url,
)
from voxint.db.models import RunStatus


def test_cursor_round_trips_full_precision() -> None:
    cursor = Cursor(
        created_at=datetime(2026, 8, 12, 9, 30, 15, 123456, tzinfo=UTC),
        run_id=uuid.uuid4(),
    )
    restored = Cursor.decode(cursor.encode())
    assert restored == cursor
    assert restored.created_at.microsecond == 123456


@pytest.mark.parametrize("token", ["", "!!!not-base64!!!", "Zm9v", "bm90LWEtY3Vyc29y"])
def test_cursor_decode_rejects_garbage(token: str) -> None:
    with pytest.raises(InvalidCursorError):
        Cursor.decode(token)


def test_cursor_decode_rejects_bad_uuid_or_timestamp() -> None:
    import base64

    bad = base64.urlsafe_b64encode(b"2026-08-12T09:30:15+00:00|not-a-uuid").decode()
    with pytest.raises(InvalidCursorError):
        Cursor.decode(bad)


def test_cursor_decode_rejects_naive_timestamp() -> None:
    import base64

    # A forged cursor with no UTC offset would compare ambiguously against the
    # tz-aware column under a non-UTC session timezone — reject it.
    naive = base64.urlsafe_b64encode(f"2026-08-12T09:30:15|{uuid.uuid4()}".encode()).decode()
    with pytest.raises(InvalidCursorError):
        Cursor.decode(naive)


@pytest.mark.parametrize("blank", [None, ""])
def test_parse_filters_blank_is_all(blank: str | None) -> None:
    assert parse_status_filter(blank) is None
    assert parse_review_filter(blank) is None


def test_parse_filters_valid() -> None:
    assert parse_status_filter("completed") is RunStatus.COMPLETED
    assert parse_review_filter("needed") is ReviewFilter.NEEDED


@pytest.mark.parametrize("bad", ["done", "COMPLETED ", "queued;drop"])
def test_parse_status_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown status"):
        parse_status_filter(bad)


def test_parse_review_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown review filter"):
        parse_review_filter("maybe")


def test_runs_url_omits_absent_filters() -> None:
    assert runs_url() == "/runs"
    assert runs_url(status=RunStatus.FAILED) == "/runs?status=failed"
    assert runs_url(review=ReviewFilter.NEEDED) == "/runs?review=needed"


def test_runs_url_preserves_filters_and_cursor() -> None:
    from urllib.parse import parse_qs, urlparse

    cursor = Cursor(created_at=datetime(2026, 1, 1, tzinfo=UTC), run_id=uuid.uuid4())
    url = runs_url(status=RunStatus.COMPLETED, review=ReviewFilter.RESOLVED, cursor=cursor)
    assert url.startswith("/runs?")
    params = parse_qs(urlparse(url).query)  # parse_qs undoes percent-encoding
    assert params["status"] == ["completed"]
    assert params["review"] == ["resolved"]
    assert Cursor.decode(params["cursor"][0]) == cursor
