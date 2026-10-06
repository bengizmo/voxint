"""The Cleaned page's diff runs and reason labels (#758 slice 5)."""

import uuid
from typing import Any

import pytest

from voxint.api.routers.legacy_runs import _CLEANUP_REASON_LABELS
from voxint.enrichment.cleanup import REJECT_REASONS
from voxint.enrichment.cleanups import CleanupError, cleanup_diff


def _line(source: str, *ranges: tuple[int, int]) -> dict[str, Any]:
    return {
        "i": 0,
        "source": source,
        "deleted": [[0, 1, start, end] for start, end in ranges],
    }


def test_no_deletions_is_one_kept_run() -> None:
    assert cleanup_diff(_line("Okay then.")) == [("Okay then.", False)]


def test_empty_source_has_no_runs() -> None:
    assert cleanup_diff(_line("")) == []


def test_adjacent_removals_merge_across_whitespace() -> None:
    source = "I mean, I think it's fine."
    assert cleanup_diff(_line(source, (2, 7), (0, 1))) == [
        ("I mean,", True),
        (" I think it's fine.", False),
    ]


def test_removals_split_by_a_kept_word_stay_apart() -> None:
    source = "um so uh yes"
    assert cleanup_diff(_line(source, (0, 2), (6, 8))) == [
        ("um", True),
        (" so ", False),
        ("uh", True),
        (" yes", False),
    ]


def test_trailing_removal_and_round_trip() -> None:
    source = "We went, you know"
    parts = cleanup_diff(_line(source, (9, 12), (13, 17)))
    assert parts == [("We went, ", False), ("you know", True)]
    assert "".join(text for text, _ in parts) == source


@pytest.mark.parametrize(
    "ranges",
    [((3, 2),), ((0, 99),), ((-1, 2),), ((0, 4), (2, 6)), ((1, 1),), ((0, 4), (0, 4))],
)
def test_out_of_bounds_or_overlapping_ranges_fail_loudly(
    ranges: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(CleanupError, match="deleted range outside its source"):
        cleanup_diff(_line("abcdefgh", *ranges))


def test_every_reject_reason_has_a_label() -> None:
    assert set(_CLEANUP_REASON_LABELS) == set(REJECT_REASONS)


def test_publish_sends_the_job_and_reports_a_broker_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.exceptions import OperationalError

    from voxint.api.routers import legacy_runs
    from voxint.worker.tasks import cleanup_run

    sent: list[tuple[object, ...]] = []

    def _send(args: tuple[object, ...], **kwargs: object) -> None:
        assert kwargs == {"ignore_result": True}
        sent.append(args)

    job_id = uuid.uuid4()
    monkeypatch.setattr(cleanup_run, "apply_async", _send)
    assert legacy_runs._publish_cleanup_job(job_id) is True
    assert sent == [(str(job_id),)]

    def _down(args: tuple[object, ...], **kwargs: object) -> None:
        raise OperationalError("broker down")

    monkeypatch.setattr(cleanup_run, "apply_async", _down)
    assert legacy_runs._publish_cleanup_job(job_id) is False
