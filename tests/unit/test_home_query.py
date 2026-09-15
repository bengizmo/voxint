"""Unit tests for the Home feed query helpers (issue #152, #478).

Tests that need a DB live in ``tests/integration/test_home_api.py``.  These
cover the pure-Python helpers: sort determinism and the group_activity
combiner, which are testable without Postgres.
"""

import uuid
from datetime import UTC, datetime

from voxint.api.home_query import ActivityItem, group_activity


def _item(
    kind: str = "run_started",
    at: datetime | None = None,
    title: str | None = None,
    run_id: uuid.UUID | None = None,
    speaker_id: uuid.UUID | None = None,
    pickup_count: int = 0,
) -> ActivityItem:
    return ActivityItem(
        at=at or datetime.now(UTC),
        kind=kind,
        title=title,
        source_path="test.wav",
        run_id=run_id,
        speaker_id=speaker_id,
        pickup_count=pickup_count,
    )


def test_sort_key_deterministic_for_watch_pickup_items() -> None:
    """Two watch_pickup items at the same timestamp sort stably by title."""
    ts = datetime(2026, 9, 14, tzinfo=UTC)
    items = [
        _item(kind="watch_pickup", at=ts, title="meetings"),
        _item(kind="watch_pickup", at=ts, title="interviews"),
    ]
    # The sort key from recent_activity: (at, kind, str(run_id or speaker_id or title or ""))
    def sort_key(i: ActivityItem) -> tuple[datetime, str, str]:
        return (i.at, i.kind, str(i.run_id or i.speaker_id or i.title or ""))

    sorted_items = sorted(items, key=sort_key, reverse=True)
    assert sorted_items[0].title == "meetings"
    assert sorted_items[1].title == "interviews"
    # Same order on a second sort (deterministic).
    assert sorted(items, key=sort_key, reverse=True) == sorted_items


def test_group_activity_passes_watch_pickup_through() -> None:
    """Watch-pickup items are not run_failed, so group_activity passes them through."""
    pickup = _item(kind="watch_pickup", title="interviews", pickup_count=3)
    started = _item(kind="run_started", run_id=uuid.uuid4())
    result = group_activity([started, pickup])
    assert len(result) == 2
    assert result[1] is pickup


def test_pickup_count_zero_for_non_pickup_items() -> None:
    """Non-pickup ActivityItems default to pickup_count=0."""
    item = _item(kind="run_started")
    assert item.pickup_count == 0
