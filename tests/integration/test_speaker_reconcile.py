"""Integration tests for cold-start speaker reconcile discovery.

Covers embedding-space-aware detection, false-positive exclusion,
NULL roster_size handling, and the --run filter.  Per-run reconciliation,
idempotency, and CLI dry-run tests are a follow-up.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    MatchCandidate,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.reconcile import cold_start_affected_runs

SPACE = "test-reconcile-v1"
SPACE_B = "test-reconcile-v2"


def _unit(dim: int, index: int) -> list[float]:
    v = np.zeros(dim)
    v[index % dim] = 1.0
    return v.tolist()


E0 = _unit(192, 0)
E1 = _unit(192, 1)


def _add_speaker(
    session: Session, name: str, embedding: list[float], space: str = SPACE
) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    session.add(
        SpeakerEmbedding(
            speaker_id=speaker.id, embedding_space=space, embedding=embedding
        )
    )
    session.flush()
    return speaker.id


def _add_completed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4().hex}/reconcile.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(
        media_item_id=media.id,
        status=RunStatus.COMPLETED.value,
    )
    session.add(run)
    session.flush()
    return run.id


def _add_match_candidate(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    speaker_id: uuid.UUID | None,
    *,
    roster_size: int | None,
    space: str = SPACE,
    similarity: float | None = 0.95,
) -> None:
    if speaker_id is not None:
        decision = "accepted"
        reason = "cosine_above_threshold"
    else:
        decision = "ineligible"
        reason = "no_eligible_turns"
    session.add(MatchCandidate(
        pipeline_run_id=run_id,
        diarization_label=label,
        decision=decision,
        reason=reason,
        embedding_space=space,
        top_speaker_id=speaker_id if decision != "ineligible" else None,
        similarity=similarity if decision != "ineligible" else None,
        vote_agreement=1.0 if decision != "ineligible" else None,
        grounded=True if decision == "accepted" else None,
        eligible_turns=3 if decision != "ineligible" else 0,
        eligible_seconds=10.0 if decision != "ineligible" else 0.0,
        roster_size=roster_size,
    ))
    session.flush()


@pytest.fixture()
def session(session_factory: sessionmaker[Session]):  # type: ignore[no-untyped-def]
    with session_factory() as s:
        yield s


class TestColdStartDetection:
    def test_finds_run_with_smaller_roster(self, session: Session) -> None:
        alice = _add_speaker(session, "Alice", E0)
        _add_speaker(session, "Bob", E1)
        run_id = _add_completed_run(session)
        _add_match_candidate(session, run_id, "SPEAKER_00", alice, roster_size=1)
        session.flush()

        plan = cold_start_affected_runs(session)
        assert run_id in plan.affected_run_ids
        assert plan.max_deficit >= 1

    def test_excludes_run_at_current_size(self, session: Session) -> None:
        alice = _add_speaker(session, "Alice", E0)
        run_id = _add_completed_run(session)
        _add_match_candidate(session, run_id, "SPEAKER_00", alice, roster_size=1)
        session.flush()

        plan = cold_start_affected_runs(session)
        assert run_id not in plan.affected_run_ids

    def test_embedding_space_isolation(self, session: Session) -> None:
        alice = _add_speaker(session, "Alice", E0, space=SPACE)
        _add_speaker(session, "Bob", E1, space=SPACE_B)
        run_id = _add_completed_run(session)
        _add_match_candidate(
            session, run_id, "SPEAKER_00", alice, roster_size=1, space=SPACE
        )
        session.flush()

        plan = cold_start_affected_runs(session)
        assert run_id not in plan.affected_run_ids

    def test_null_roster_size_excluded(self, session: Session) -> None:
        alice = _add_speaker(session, "Alice", E0)
        _add_speaker(session, "Bob", E1)
        run_id = _add_completed_run(session)
        _add_match_candidate(session, run_id, "SPEAKER_00", alice, roster_size=None)
        session.flush()

        plan = cold_start_affected_runs(session)
        assert run_id not in plan.affected_run_ids

    def test_run_filter(self, session: Session) -> None:
        alice = _add_speaker(session, "Alice", E0)
        _add_speaker(session, "Bob", E1)
        run_a = _add_completed_run(session)
        run_b = _add_completed_run(session)
        _add_match_candidate(session, run_a, "SPEAKER_00", alice, roster_size=1)
        _add_match_candidate(session, run_b, "SPEAKER_00", alice, roster_size=1)
        session.flush()

        plan = cold_start_affected_runs(session, run_id=run_a)
        assert run_a in plan.affected_run_ids
        assert run_b not in plan.affected_run_ids
