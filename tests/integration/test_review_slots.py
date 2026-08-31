"""Reviewer slot lifecycle against real Postgres: claim, verify, release, expiry."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    release_run,
    renew_claim,
    verify_claim,
)
from voxint.db.models import MediaItem, PipelineRun, RunStatus

TTL = 600


def make_run(session: Session, status: RunStatus = RunStatus.COMPLETED) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=status.value)
    session.add(run)
    session.flush()
    return run.id


def test_claim_verify_release_roundtrip(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        run = verify_claim(session, run_id, token)
        assert run.review_claimed_by == "ben"
        release_run(session, run_id, token)
        session.commit()

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.review_claim_token is None
        assert run.review_claimed_by is None
        with pytest.raises(ClaimMismatchError):
            verify_claim(session, run_id, token)


def test_only_completed_runs_are_claimable(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session, status=RunStatus.RUNNING)
        with pytest.raises(ClaimUnavailableError, match="not completed"):
            claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)


def test_live_claim_blocks_other_reviewer(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    with session_factory() as session, pytest.raises(ClaimUnavailableError, match="claimed by"):
        claim_run(session, run_id, reviewer="mallory", ttl_seconds=TTL)


def test_same_reviewer_reclaim_rotates_token(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        first = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    with session_factory() as session:
        second = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    assert first != second
    with session_factory() as session:
        with pytest.raises(ClaimMismatchError):
            verify_claim(session, run_id, first)  # the old tab is dead
        verify_claim(session, run_id, second)


def test_expired_claim_is_reclaimable_and_unusable(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        stale = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.review_claim_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
        session.commit()

    with session_factory() as session:
        with pytest.raises(ClaimMismatchError, match="stale"):
            verify_claim(session, run_id, stale)
        fresh = claim_run(session, run_id, reviewer="mallory", ttl_seconds=TTL)
        session.commit()
    with session_factory() as session:
        assert verify_claim(session, run_id, fresh).review_claimed_by == "mallory"


def test_release_requires_matching_token(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    with session_factory() as session:
        with pytest.raises(ClaimMismatchError):
            release_run(session, run_id, uuid.uuid4())
        # Releasing an unclaimed run is a no-op, not an error.
        release_run(session, uuid.uuid4(), uuid.uuid4())


def test_concurrent_claim_one_winner(session_factory: sessionmaker[Session]) -> None:
    """Two sessions read the same unclaimed run; CAS lets exactly one claim it."""
    with session_factory() as session:
        run_id = make_run(session)
        session.commit()

    s1, s2 = session_factory(), session_factory()
    try:
        # Both sessions read revision 0 before either writes.
        r1 = s1.get(PipelineRun, run_id)
        r2 = s2.get(PipelineRun, run_id)
        assert r1 is not None and r2 is not None
        assert r1.revision == r2.revision == 0

        claim_run(s1, run_id, reviewer="first", ttl_seconds=TTL)
        s1.commit()
        with pytest.raises(ClaimUnavailableError):
            claim_run(s2, run_id, reviewer="second", ttl_seconds=TTL)
        s2.rollback()
    finally:
        s1.close()
        s2.close()


# ---- renew_claim tests ----


def test_renew_extends_expiry_keeps_token(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        run_before = session.get(PipelineRun, run_id)
        assert run_before is not None
        old_expires = run_before.review_claim_expires_at

    with session_factory() as session:
        new_expires = renew_claim(session, run_id, token, ttl_seconds=TTL)
        session.commit()

    assert new_expires > old_expires  # type: ignore[operator]
    with session_factory() as session:
        run_after = session.get(PipelineRun, run_id)
        assert run_after is not None
        assert run_after.review_claim_token == token
        assert run_after.review_claimed_by == "ben"


def test_renew_wrong_token_raises(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()

    with session_factory() as session, pytest.raises(ClaimMismatchError):
        renew_claim(session, run_id, uuid.uuid4(), ttl_seconds=TTL)


def test_renew_expired_token_raises(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.review_claim_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
        session.commit()

    with session_factory() as session, pytest.raises(ClaimMismatchError, match="stale"):
        renew_claim(session, run_id, token, ttl_seconds=TTL)


def test_renew_non_completed_raises(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session, status=RunStatus.RUNNING)
        session.commit()

    with session_factory() as session, pytest.raises(ClaimMismatchError):
        renew_claim(session, run_id, uuid.uuid4(), ttl_seconds=TTL)


def test_renew_then_verify_same_token(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        renew_claim(session, run_id, token, ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        run = verify_claim(session, run_id, token)
        assert run.review_claimed_by == "ben"
