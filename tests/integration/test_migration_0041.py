"""Migration 0041 (speaker_profiles), issue #159.

Real alembic up/down against the shared test database (head restored in
teardown): the table appears with its constraints; the backfill materializes
previously accepted bio/affiliation/link decisions — newest decision per
(canonical speaker, field) wins, speaker ids canonicalize through merge
tombstones, rejected and name decisions contribute nothing; the downgrade
drops the table and a re-upgrade re-backfills; the ORM model matches the
migrated DDL column-for-column.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import (
    EnrichmentCandidate,
    EnrichmentProducerRun,
    ProfileReviewDecision,
    Speaker,
    SpeakerProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        yield cfg
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")


def _producer_run(session: Session, speaker_id: uuid.UUID) -> uuid.UUID:
    now = datetime.now(UTC)
    run = EnrichmentProducerRun(
        producer="test-producer",
        producer_version="1",
        target_kind="speaker",
        speaker_id=speaker_id,
        covered_fields=["bio", "affiliation", "link"],
        generation=1,
        outcome="found",
        idempotency_key=f"prod-{uuid.uuid4()}",
        started_at=now,
        completed_at=now,
    )
    session.add(run)
    session.flush()
    return run.id


def _candidate(
    session: Session, speaker_id: uuid.UUID, field: str, value: str
) -> uuid.UUID:
    cand = EnrichmentCandidate(
        producer_run_id=_producer_run(session, speaker_id),
        target_kind="speaker",
        speaker_id=speaker_id,
        field=field,
        value=value,
    )
    session.add(cand)
    session.flush()
    return cand.id


def _decision(
    session: Session,
    candidate_id: uuid.UUID,
    *,
    decision: str = "accept",
    operator: str = "op",
    created_at: datetime,
) -> None:
    # Core INSERT: the test downgrades to a pre-0052 schema that lacks
    # user_id, so the ORM model (which includes user_id) would emit an
    # INSERT referencing a non-existent column.
    from sqlalchemy import insert

    session.execute(
        insert(ProfileReviewDecision).values(
            id=uuid.uuid4(),
            candidate_id=candidate_id,
            decision=decision,
            operator=operator,
            note=None,
            idempotency_key=f"dec-{uuid.uuid4()}",
            created_at=created_at,
        )
    )


def test_schema_constraints_hold(engine: Engine, alembic_cfg: Config) -> None:
    inspector = inspect(engine)
    assert inspector.has_table("speaker_profiles")
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        speaker = Speaker(display_name=f"S-{uuid.uuid4()}")
        session.add(speaker)
        session.flush()
        # name is not a profile field.
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(
                SpeakerProfile(
                    speaker_id=speaker.id,
                    field="name",
                    value="x",
                    provenance="manual",
                    operator="op",
                )
            )
            session.flush()
        # enrichment provenance requires a candidate reference (and vice versa).
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(
                SpeakerProfile(
                    speaker_id=speaker.id,
                    field="bio",
                    value="x",
                    provenance="enrichment",
                    operator="op",
                )
            )
            session.flush()
        # One row per (speaker, field).
        session.add(
            SpeakerProfile(
                speaker_id=speaker.id,
                field="bio",
                value="first",
                provenance="manual",
                operator="op",
            )
        )
        session.flush()
        with pytest.raises(IntegrityError), session.begin_nested():
            session.add(
                SpeakerProfile(
                    speaker_id=speaker.id,
                    field="bio",
                    value="second",
                    provenance="manual",
                    operator="op",
                )
            )
            session.flush()
        session.rollback()


def test_backfill_newest_accept_wins_and_canonicalizes(
    engine: Engine, alembic_cfg: Config
) -> None:
    command.downgrade(alembic_cfg, "0040")
    factory = sessionmaker(engine, expire_on_commit=False)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    with factory() as session:
        a = Speaker(display_name="A")
        b = Speaker(display_name="B")
        c = Speaker(display_name="C")
        session.add_all([a, b, c])
        session.flush()
        # A was merged into B: A's accepted claims must land under B.
        a.merged_into_id = b.id
        a.merged_at = base
        # bio: an older accept on A, a newer accept on B — B's value wins.
        _decision(
            session, _candidate(session, a.id, "bio", "old bio via A"), created_at=base
        )
        _decision(
            session,
            _candidate(session, b.id, "bio", "new bio via B"),
            operator="newer-op",
            created_at=base + timedelta(days=1),
        )
        # affiliation: accepted only on the tombstone A — lands under B.
        _decision(
            session,
            _candidate(session, a.id, "affiliation", "Acme"),
            created_at=base,
        )
        # link on C: accepted; a rejected bio on C contributes nothing.
        _decision(
            session,
            _candidate(session, c.id, "link", "https://example.org"),
            created_at=base,
        )
        _decision(
            session,
            _candidate(session, c.id, "bio", "rejected bio"),
            decision="reject",
            created_at=base,
        )
        session.commit()
        b_id, c_id = b.id, c.id

    command.upgrade(alembic_cfg, "head")

    with factory() as session:
        rows = session.query(SpeakerProfile).all()
        by_key = {(r.speaker_id, r.field): r for r in rows}
        assert set(by_key) == {
            (b_id, "bio"),
            (b_id, "affiliation"),
            (c_id, "link"),
        }
        bio = by_key[(b_id, "bio")]
        assert bio.value == "new bio via B"
        assert bio.operator == "newer-op"
        assert bio.provenance == "enrichment"
        assert bio.accepted_candidate_id is not None
        assert by_key[(b_id, "affiliation")].value == "Acme"

        # Idempotent + re-derivable: downgrade drops, re-upgrade re-backfills.
        session.close()
    command.downgrade(alembic_cfg, "0040")
    assert not inspect(engine).has_table("speaker_profiles")
    command.upgrade(alembic_cfg, "head")
    with factory() as session:
        assert session.query(SpeakerProfile).count() == 3


def test_orm_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    inspector = inspect(engine)
    reflected = {c["name"] for c in inspector.get_columns("speaker_profiles")}
    model = {c.name for c in SpeakerProfile.__table__.columns}
    assert reflected == model
