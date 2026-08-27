"""requeue_failed_run refuses archived runs (M9).

The archived guard fires before the FAILED status check, so an archived run
of any status is refused with RunArchivedError.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from voxint.db.models import RunStatus
from voxint.ingest.service import RunArchivedError, requeue_failed_run


def _mock_session(*, archived: bool, status: str = RunStatus.FAILED.value) -> MagicMock:
    session = MagicMock()
    run = MagicMock()
    run.archived_at = datetime.now(UTC) if archived else None
    run.status = status
    run.current_stage = "transcribe"
    run.revision = 1
    session.get.return_value = run
    return session


def test_archived_run_raises_run_archived_error() -> None:
    session = _mock_session(archived=True)
    with pytest.raises(RunArchivedError):
        requeue_failed_run(session, uuid.uuid4())


def test_non_archived_failed_run_proceeds_past_guard() -> None:
    """A non-archived run should NOT raise RunArchivedError (it may raise
    other errors downstream from the mock, but the archive guard passes)."""
    session = _mock_session(archived=False)
    with pytest.raises(Exception) as exc_info:
        requeue_failed_run(session, uuid.uuid4())
    assert not isinstance(exc_info.value, RunArchivedError)
