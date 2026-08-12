"""Shared DB-only submission + requeue service used by the CLI and the API.

Every function here operates on a live SQLAlchemy ``Session`` and **never**
imports Celery: the caller owns the commit boundary and lazily publishes
``voxint.run_pipeline`` *after* the transaction commits (commit-before-publish).
Keeping the broker out of this module is what lets the API's read path stay
Postgres-only and guarantees a broker outage can never leave a half-written
run — the durable QUEUED row exists before anything is enqueued.

Failure modes surface as typed exceptions so each caller maps them to its own
surface (the CLI prints a message + exit 2; the API returns 404/409). The two
``cas_update_run`` errors (:class:`StaleRevisionError`,
:class:`InvalidTransitionError`) propagate unchanged.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.pipeline.engine import submit
from voxint.pipeline.transitions import (
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
    snapshot,
)


class IngestError(Exception):
    """Base for submission/requeue failures a caller maps to a UI/exit code."""


class RunNotFoundError(IngestError):
    def __init__(self, run_id: uuid.UUID) -> None:
        super().__init__(f"no run {run_id}")
        self.run_id = run_id


class RunNotFailedError(IngestError):
    """Requeue attempted on a run that is not FAILED (only FAILED may requeue)."""

    def __init__(self, run_id: uuid.UUID, status: RunStatus) -> None:
        super().__init__(f"run is {status.value}, only failed runs can be requeued")
        self.run_id = run_id
        self.status = status


class MissingStageError(IngestError):
    """A FAILED run carrying no current_stage — the state machine was violated.

    A FAILED run always carries its failed stage; ``None`` means corruption, so
    we refuse to guess a stage rather than requeue into an arbitrary one.
    """

    def __init__(self, run_id: uuid.UUID) -> None:
        super().__init__(f"run {run_id} is FAILED with no current_stage; refusing to guess")
        self.run_id = run_id


def submit_media_item(session: Session, source_path: str) -> PipelineRun:
    """Create-or-reuse the MediaItem for ``source_path`` and queue a fresh run.

    DB-only: the caller owns the commit and, once it commits, lazily publishes
    ``voxint.run_pipeline`` (commit-before-publish). ``source_path`` is UNIQUE,
    so a repeated local path reuses its MediaItem while every submission still
    mints a distinct run.
    """
    media = _get_or_create_media(session, source_path)
    return submit(session, media.id)


def _get_or_create_media(session: Session, source_path: str) -> MediaItem:
    """Return the MediaItem for ``source_path``, inserting it if absent.

    ``source_path`` is UNIQUE. Two callers can both observe no row and both try
    to insert; the loser's INSERT violates the constraint. Containing that INSERT
    in a SAVEPOINT means the conflict rolls back only the insert — not the
    caller's outer transaction — so we can re-read and adopt the row the winner
    committed. A concurrent submission thus still gets a MediaItem to mint its
    own run against, honouring "each submission mints a distinct run". (The API's
    uploads/URLs use uuid-namespaced paths that never collide; this guards the
    CLI's reuse-by-path and any future shared caller.)
    """
    existing = session.execute(
        select(MediaItem).where(MediaItem.source_path == source_path)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    media = MediaItem(source_path=source_path)
    try:
        # add() inside the savepoint so a rolled-back attempt is expunged and
        # cannot be re-INSERTed at the caller's later flush/commit.
        with session.begin_nested():
            session.add(media)
            session.flush()
    except IntegrityError:
        return session.execute(
            select(MediaItem).where(MediaItem.source_path == source_path)
        ).scalar_one()
    return media


def requeue_failed_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    expected_revision: int | None = None,
) -> RunSnapshot:
    """CAS-requeue a FAILED run at its failed stage, guarded by exact revision.

    Pass ``expected_revision`` to enforce exact-revision CAS from a caller that
    already knows the revision it means to act on (e.g. the API's requeue form):
    a mismatch raises :class:`StaleRevisionError` before any write, so a stale
    browser tab can never requeue a run that moved on. The CLI reads fresh and
    omits it — there is no gap to race within a single transaction.

    DB-only: the caller commits then lazily publishes ``voxint.run_pipeline``.
    Raises :class:`RunNotFoundError`, :class:`RunNotFailedError`,
    :class:`MissingStageError`, or — from the CAS — ``StaleRevisionError`` /
    ``InvalidTransitionError``, which callers map to their own responses.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    held = snapshot(run)
    if held.status is not RunStatus.FAILED:
        raise RunNotFailedError(run_id, held.status)
    if held.current_stage is None:
        raise MissingStageError(run_id)
    if expected_revision is not None and held.revision != expected_revision:
        raise StaleRevisionError(run_id, expected_revision)
    return cas_update_run(
        session,
        held,
        status=RunStatus.QUEUED,
        current_stage=held.current_stage,
    )
