"""ResearchBudget atomicity/deadline semantics + Attribution bounds (issue #39)."""

import threading

import pytest

from voxint.research.budget import Attribution, ResearchBudget


def test_attribution_accepts_bounded_identifiers() -> None:
    a = Attribution(feature="cli", reason="operator-search")
    assert a.feature == "cli"
    assert a.reason == "operator-search"


@pytest.mark.parametrize(
    ("feature", "reason"),
    [
        ("", "ok"),  # empty
        ("cli", ""),  # empty reason
        ("CLI", "ok"),  # uppercase
        ("has space", "ok"),  # whitespace
        ("inject\nline", "ok"),  # log injection
        ("x" * 65, "ok"),  # over length
        ("cli", "token=SECRET?"),  # non-identifier chars
    ],
)
def test_attribution_rejects_unbounded_text(feature: str, reason: str) -> None:
    with pytest.raises(ValueError, match="attribution"):
        Attribution(feature=feature, reason=reason)


def test_counters_consume_down_to_zero() -> None:
    b = ResearchBudget(max_searches=2, max_reads=1)
    assert b.try_consume_search() is True
    assert b.try_consume_search() is True
    assert b.try_consume_search() is False  # spent
    assert b.try_consume_read() is True
    assert b.try_consume_read() is False
    assert b.searches_left == 0
    assert b.reads_left == 0


def test_deadline_blocks_new_consumption() -> None:
    now = [0.0]
    b = ResearchBudget(
        max_searches=5, max_reads=5, deadline_seconds=10.0, clock=lambda: now[0]
    )
    assert b.try_consume_read() is True
    now[0] = 10.0  # exactly at the deadline — expired, not "one more"
    assert b.expired is True
    assert b.try_consume_read() is False
    assert b.try_consume_search() is False


def test_remaining_seconds_floors_at_zero_and_none_without_deadline() -> None:
    now = [0.0]
    b = ResearchBudget(
        max_searches=1, max_reads=1, deadline_seconds=5.0, clock=lambda: now[0]
    )
    assert b.remaining_seconds() == 5.0
    now[0] = 99.0
    assert b.remaining_seconds() == 0.0
    assert ResearchBudget(max_searches=1, max_reads=1).remaining_seconds() is None


def test_concurrent_consumption_never_overspends() -> None:
    b = ResearchBudget(max_searches=0, max_reads=50)
    granted: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(10):
            ok = b.try_consume_read()
            with lock:
                granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) == 50  # exactly the budget, no double-spend
    assert b.reads_left == 0


def test_invalid_construction_rejected() -> None:
    with pytest.raises(ValueError):
        ResearchBudget(max_searches=-1, max_reads=0)
    with pytest.raises(ValueError):
        ResearchBudget(max_searches=0, max_reads=0, deadline_seconds=0.0)


def test_nan_and_inf_deadlines() -> None:
    # NaN would make every deadline comparison False — the wall clock the #40
    # loop relies on would silently never fire (review finding).
    with pytest.raises(ValueError):
        ResearchBudget(max_searches=1, max_reads=1, deadline_seconds=float("nan"))
    # +inf is a harmless explicit "no deadline".
    b = ResearchBudget(max_searches=1, max_reads=1, deadline_seconds=float("inf"))
    assert b.expired is False
    assert b.try_consume_read() is True
