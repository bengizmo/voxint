"""Characterization of the commit-before-publish contract (audit finding H5).

Every submit path is DB-only and returns a :class:`SubmissionResult`; the caller
commits, then calls :meth:`SubmissionResult.publish` to enqueue the pipeline.
Pinned here against real Postgres: a caller that commits but forgets
``publish()`` leaves a durable QUEUED run and sends nothing to the broker (the
recovery sweep's case), and ``publish()`` is the one call that enqueues it.

The spy replaces ``apply_async`` on Celery's base ``Task`` and ``send_task`` on
the app class, so an enqueue through any task or by name is caught, not only
one through ``run_pipeline``.
"""

import io
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from celery import Celery
from celery.app.task import Task
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import PipelineRun, RunStatus
from voxint.ingest import submit_media_item, submit_media_item_if_new, submit_upload
from voxint.ingest.service import SubmissionResult

Submitter = Callable[[Session, Path], SubmissionResult | None]


def _via_media_item(session: Session, _media_root: Path) -> SubmissionResult | None:
    return submit_media_item(session, f"incoming/{uuid.uuid4()}.wav")


def _via_media_item_if_new(session: Session, _media_root: Path) -> SubmissionResult | None:
    return submit_media_item_if_new(session, f"incoming/{uuid.uuid4()}.wav")


def _via_upload(session: Session, media_root: Path) -> SubmissionResult | None:
    return submit_upload(
        session,
        stream=io.BytesIO(b"audio-bytes"),
        filename="clip.wav",
        submission_id=uuid.uuid4().hex,
        media_root=media_root,
        max_bytes=1024 * 1024,
    )


@pytest.fixture()
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    # Same leading parameters as the real methods (args may be None, send_task
    # takes kwargs positionally), so any call shape is recorded, not rejected.
    def _apply_async(self: Task, args: tuple[Any, ...] | None = None, *_a: Any, **_kw: Any) -> None:
        calls.append((str(self.name), tuple(args or ())))

    def _send_task(
        _self: Celery,
        name: str,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        *_a: Any,
        **_kw: Any,
    ) -> None:
        calls.append((name, tuple(args or ())))

    monkeypatch.setattr(Task, "apply_async", _apply_async)
    monkeypatch.setattr(Celery, "send_task", _send_task)
    return calls


@pytest.mark.parametrize(
    "submitter",
    [_via_media_item, _via_media_item_if_new, _via_upload],
    ids=["submit_media_item", "submit_media_item_if_new", "submit_upload"],
)
def test_forgotten_publish_leaves_run_queued_with_no_task(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    enqueued: list[tuple[str, tuple[Any, ...]]],
    submitter: Submitter,
) -> None:
    with session_factory() as session:
        result = submitter(session, tmp_path)
        assert result is not None
        session.commit()
    # The caller stops here: committed, never published.

    with session_factory() as session:
        run = session.get(PipelineRun, result.run_id)
        assert run is not None
        assert run.status == RunStatus.QUEUED
        assert run.current_stage is None
    assert enqueued == []

    # Positive control: publish() is the call that enqueues, through the spy.
    assert result.publish() is True
    assert enqueued == [("voxint.run_pipeline", (str(result.run_id),))]
