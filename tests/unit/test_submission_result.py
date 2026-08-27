"""Tests for :class:`voxint.ingest.service.SubmissionResult`."""

from __future__ import annotations

import uuid

import pytest

from voxint.ingest.service import SubmissionResult


def test_submission_result_is_frozen() -> None:
    result = SubmissionResult(run_id=uuid.uuid4())
    with pytest.raises(AttributeError):
        result.run_id = uuid.uuid4()  # type: ignore[misc]


def test_publish_returns_true_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = uuid.uuid4()
    result = SubmissionResult(run_id=rid)

    calls: list[tuple[str, ...]] = []

    import voxint.worker.tasks as tasks_mod

    monkeypatch.setattr(
        tasks_mod.run_pipeline, "apply_async", lambda args, **kw: calls.append(args)
    )
    assert result.publish() is True
    assert calls == [(str(rid),)]


def test_publish_returns_false_on_broker_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    from celery.exceptions import OperationalError

    import voxint.worker.tasks as tasks_mod

    def boom(args, **kwargs):  # type: ignore[no-untyped-def]
        raise OperationalError("broker down")

    monkeypatch.setattr(tasks_mod.run_pipeline, "apply_async", boom)
    result = SubmissionResult(run_id=uuid.uuid4())
    assert result.publish() is False
