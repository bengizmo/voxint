"""Reviewer slot lifecycle against real Postgres: claim, verify, release, expiry."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    refresh_run_claim,
    release_run,
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


# --- refresh_run_claim tests (#374) ---


def test_refresh_extends_expiry_same_token(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        before = session.get(PipelineRun, run_id)
        assert before is not None
        old_expiry = before.review_claim_expires_at
        old_rev = before.revision
        session.expunge(before)

    with session_factory() as session:
        refresh_run_claim(session, run_id, token, ttl_seconds=TTL)
        session.commit()

    with session_factory() as session:
        after = session.get(PipelineRun, run_id)
        assert after is not None
        assert after.review_claim_token == token
        assert after.review_claim_expires_at is not None
        assert old_expiry is not None
        assert after.review_claim_expires_at > old_expiry
        assert after.revision == old_rev


def test_refresh_rejects_stale_token(
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
    with session_factory() as session, pytest.raises(ClaimMismatchError):
        refresh_run_claim(session, run_id, first, ttl_seconds=TTL)


def test_refresh_rejects_expired_claim(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.review_claim_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
        session.commit()

    with session_factory() as session, pytest.raises(ClaimMismatchError):
        refresh_run_claim(session, run_id, token, ttl_seconds=TTL)


def test_refresh_rejects_non_completed_run(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = make_run(session)
        token = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.status = RunStatus.RUNNING.value
        session.commit()

    with session_factory() as session, pytest.raises(ClaimMismatchError):
        refresh_run_claim(session, run_id, token, ttl_seconds=TTL)


def test_two_claims_then_refresh_first_token_fails(
    session_factory: sessionmaker[Session],
) -> None:
    """Tab A claims, Tab B claims (rotates), Tab A's refresh gets 409."""
    with session_factory() as session:
        run_id = make_run(session)
        token_a = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    with session_factory() as session:
        token_b = claim_run(session, run_id, reviewer="ben", ttl_seconds=TTL)
        session.commit()
    assert token_a != token_b
    with session_factory() as session, pytest.raises(ClaimMismatchError):
        refresh_run_claim(session, run_id, token_a, ttl_seconds=TTL)
    with session_factory() as session:
        refresh_run_claim(session, run_id, token_b, ttl_seconds=TTL)
        session.commit()
