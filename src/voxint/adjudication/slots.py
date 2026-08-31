"""The guarded reviewer slot: one adjudication claim per run.

Adjudication is post-hoc — only COMPLETED runs can be claimed, and claiming
never touches the pipeline state machine. The claim rides the same CAS
``revision`` column as pipeline transitions, so a concurrent claim, release,
or (hypothetically) pipeline write can never be lost — one of the writers gets
zero rows and re-reads.

The claim token is an opaque per-claim secret returned to the browser and
required on every mutation. A stale tab holding yesterday's token gets a
:class:`ClaimMismatchError` (HTTP 409 upstream) rather than silently acting on
a slot someone else re-claimed.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from voxint.db.models import PipelineRun, RunStatus


class ClaimUnavailableError(Exception):
    """The run is not claimable: not COMPLETED, actively claimed by someone
    else, or moved concurrently — re-read and re-render."""


class ClaimMismatchError(Exception):
    """The presented claim token does not match the run's live, unexpired
    claim. The tab must re-claim before deciding anything."""


def claim_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    reviewer: str,
    ttl_seconds: int,
) -> uuid.UUID:
    """Claim a COMPLETED run for review; returns the claim token.

    Re-claiming a run the same reviewer already holds issues a fresh token
    (the old tab's token dies with it — exactly one live token per run).
    """
    run = session.execute(
        select(PipelineRun).where(PipelineRun.id == run_id)
    ).scalar_one_or_none()
    if run is None:
        raise ClaimUnavailableError(f"no run {run_id}")
    now = datetime.now(tz=UTC)
    claim_live = (
        run.review_claim_token is not None
        and run.review_claim_expires_at is not None
        and run.review_claim_expires_at > now
    )
    if run.status != RunStatus.COMPLETED.value:
        raise ClaimUnavailableError(f"run {run_id} is {run.status}, not completed")
    if claim_live and run.review_claimed_by != reviewer:
        raise ClaimUnavailableError(
            f"run {run_id} is claimed by {run.review_claimed_by!r}"
        )
    token = uuid.uuid4()
    result = cast(
        CursorResult[Any],
        session.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id, PipelineRun.revision == run.revision)
            .values(
                review_claim_token=token,
                review_claimed_by=reviewer,
                review_claimed_at=now,
                review_claim_expires_at=now + timedelta(seconds=ttl_seconds),
                revision=run.revision + 1,
            )
        ),
    )
    if result.rowcount != 1:
        raise ClaimUnavailableError(f"run {run_id} moved concurrently; retry")
    return token


def verify_claim(
    session: Session, run_id: uuid.UUID, token: uuid.UUID, *, for_update: bool = False
) -> PipelineRun:
    """The gate every decision POST passes: run exists, is COMPLETED, and the
    presented token is its live unexpired claim.

    Mutating callers pass ``for_update=True``: the row lock is held until
    their transaction ends, so a concurrent re-claim/release serializes
    against the write — a token that verifies cannot go stale between the
    check and the commit (the re-claim waits, or this verify sees its token).
    """
    stmt = select(PipelineRun).where(PipelineRun.id == run_id)
    if for_update:
        stmt = stmt.with_for_update()
    run = session.execute(stmt).scalar_one_or_none()
    if run is None or run.status != RunStatus.COMPLETED.value:
        raise ClaimMismatchError(f"run {run_id} is not open for review")
    now = datetime.now(tz=UTC)
    if (
        run.review_claim_token != token
        or run.review_claim_expires_at is None
        or run.review_claim_expires_at <= now
    ):
        raise ClaimMismatchError(f"claim on run {run_id} is stale; re-claim")
    return run


def refresh_run_claim(
    session: Session,
    run_id: uuid.UUID,
    token: uuid.UUID,
    *,
    ttl_seconds: int,
) -> None:
    """Extend the TTL of an active claim without rotating the token.

    Used by the heartbeat: a stale tab whose token no longer matches gets
    ClaimMismatchError (409 upstream) and drops to claim-lost state.
    Unlike claim_run this never issues a new token and never bumps revision.
    """
    now = datetime.now(tz=UTC)
    result = cast(
        CursorResult[Any],
        session.execute(
            update(PipelineRun)
            .where(
                PipelineRun.id == run_id,
                PipelineRun.review_claim_token == token,
                PipelineRun.review_claim_expires_at > now,
                PipelineRun.status == RunStatus.COMPLETED.value,
            )
            .values(
                review_claim_expires_at=now + timedelta(seconds=ttl_seconds),
            )
        ),
    )
    if result.rowcount != 1:
        raise ClaimMismatchError(
            f"claim on run {run_id} is stale or expired; re-claim"
        )


def release_run(session: Session, run_id: uuid.UUID, token: uuid.UUID) -> None:
    """Release a held claim; token must match (idempotent for a dead claim)."""
    run = session.execute(
        select(PipelineRun).where(PipelineRun.id == run_id)
    ).scalar_one_or_none()
    if run is None or run.review_claim_token is None:
        return  # nothing held — release is idempotent
    if run.review_claim_token != token:
        raise ClaimMismatchError(f"claim on run {run_id} is held by another token")
    result = cast(
        CursorResult[Any],
        session.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id, PipelineRun.revision == run.revision)
            .values(
                review_claim_token=None,
                review_claimed_by=None,
                review_claimed_at=None,
                review_claim_expires_at=None,
                revision=run.revision + 1,
            )
        ),
    )
    if result.rowcount != 1:
        raise ClaimUnavailableError(f"run {run_id} moved concurrently; retry")
