"""Migration 0032 (match_candidate_evidence), up/down + constraints (issue #113).

Real alembic up/down against the shared test database (head restored in
teardown): ``match_candidates`` appears with its full column set, an accepted
row round-trips, the UNIQUE(run, label) and decision/coherence CHECKs reject
malformed evidence, the run FK cascades and the speaker FK sets NULL on delete,
downgrade drops the table, and the ORM model stays in lockstep with the DDL.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import MatchCandidate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pg_type(type_: TypeEngine[object]) -> str:
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


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


def _seed_run_and_speaker(engine: Engine) -> tuple[uuid.UUID, uuid.UUID]:
    media_id, run_id, speaker_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :m, 'completed', 0)"
            ),
            {"id": run_id, "m": media_id},
        )
        conn.execute(
            text("INSERT INTO speakers (id, display_name) VALUES (:id, :n)"),
            {"id": speaker_id, "n": f"Speaker {speaker_id}"},
        )
        conn.commit()
    return run_id, speaker_id


def _insert(engine: Engine, **cols: object) -> None:
    keys = ", ".join(cols)
    binds = ", ".join(f":{k}" for k in cols)
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO match_candidates ({keys}) VALUES ({binds})"), cols
        )
        conn.commit()


def _accepted_row(run_id: uuid.UUID, speaker_id: uuid.UUID, label: str) -> dict[str, object]:
    return {
        "id": uuid.uuid4(),
        "pipeline_run_id": run_id,
        "diarization_label": label,
        "decision": "accepted",
        "reason": "accepted",
        "embedding_space": "titanet-large-v1",
        "top_speaker_id": speaker_id,
        "similarity": 0.92,
        "margin": 0.3,
        "vote_agreement": 1.0,
        "grounded": True,
        "eligible_turns": 3,
        "eligible_seconds": 12.0,
        "roster_size": 2,
    }


def test_upgrade_roundtrips_and_enforces_constraints(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, speaker_id = _seed_run_and_speaker(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("match_candidates")}
    assert cols == {
        "id",
        "pipeline_run_id",
        "diarization_label",
        "decision",
        "reason",
        "embedding_space",
        "top_speaker_id",
        "similarity",
        "margin",
        "vote_agreement",
        "grounded",
        "eligible_turns",
        "eligible_seconds",
        "roster_size",
        "created_at",
    }

    _insert(engine, **_accepted_row(run_id, speaker_id, "SPEAKER_00"))
    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT decision, similarity, grounded FROM match_candidates"
                " WHERE pipeline_run_id = :r"
            ),
            {"r": run_id},
        ).one()
    assert stored == ("accepted", 0.92, True)

    # UNIQUE(pipeline_run_id, diarization_label): one row per label per run.
    with pytest.raises(IntegrityError):
        _insert(engine, **_accepted_row(run_id, speaker_id, "SPEAKER_00"))

    # decision CHECK: only the three known verdicts.
    bad = _accepted_row(run_id, speaker_id, "SPEAKER_01")
    bad["decision"] = "maybe"
    with pytest.raises(IntegrityError):
        _insert(engine, **bad)


def test_ineligible_and_grounded_coherence_checks(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, speaker_id = _seed_run_and_speaker(engine)

    # An ineligible row must carry no candidate and no numbers.
    bad_ineligible = {
        "id": uuid.uuid4(),
        "pipeline_run_id": run_id,
        "diarization_label": "SPEAKER_00",
        "decision": "ineligible",
        "reason": "too_few_turns",
        "top_speaker_id": speaker_id,  # violates the ineligible-speaker CHECK
        "eligible_turns": 1,
        "eligible_seconds": 3.0,
    }
    with pytest.raises(IntegrityError):
        _insert(engine, **bad_ineligible)

    # An accepted row must resolve grounding (grounded NOT NULL).
    bad_accepted = _accepted_row(run_id, speaker_id, "SPEAKER_01")
    bad_accepted["grounded"] = None
    with pytest.raises(IntegrityError):
        _insert(engine, **bad_accepted)


def test_run_delete_cascades_and_speaker_delete_is_refused(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id, speaker_id = _seed_run_and_speaker(engine)
    _insert(engine, **_accepted_row(run_id, speaker_id, "SPEAKER_00"))

    # Speakers are soft-archived/merged, never hard-deleted; the plain FK refuses
    # a stray hard delete rather than nulling recorded evidence.
    with pytest.raises(IntegrityError), engine.connect() as conn:
        conn.execute(text("DELETE FROM speakers WHERE id = :s"), {"s": speaker_id})
        conn.commit()

    # Deleting the run cascades the evidence away.
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM pipeline_runs WHERE id = :r"), {"r": run_id})
        conn.commit()
        remaining = conn.execute(
            text("SELECT count(*) FROM match_candidates WHERE pipeline_run_id = :r"),
            {"r": run_id},
        ).scalar_one()
    assert remaining == 0


def test_downgrade_drops_table(engine: Engine, alembic_cfg: Config) -> None:
    run_id, speaker_id = _seed_run_and_speaker(engine)
    _insert(engine, **_accepted_row(run_id, speaker_id, "SPEAKER_00"))
    command.downgrade(alembic_cfg, "0031")
    assert "match_candidates" not in inspect(engine).get_table_names()
    command.upgrade(alembic_cfg, "head")
    assert "match_candidates" in inspect(engine).get_table_names()


def test_model_matches_migrated_schema(engine: Engine, alembic_cfg: Config) -> None:
    reflected = {
        col["name"]: _pg_type(col["type"])
        for col in inspect(engine).get_columns("match_candidates")
    }
    model = {
        col.name: _pg_type(col.type) for col in MatchCandidate.__table__.columns
    }
    assert reflected == model
