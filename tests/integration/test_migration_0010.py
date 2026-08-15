"""Migration 0010 (enrichment draft schema), up/down + model parity.

Mirrors test_migration_0009: runs the real alembic up/down against the shared
test database and restores head in the fixture teardown, asserts every named
CHECK/UNIQUE rejects bad rows, exercises the three integrity triggers, and
asserts the ORM models match the migrated schema (the suite builds its schema
from the alembic chain and no autogenerate parity test exists in-tree).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, Table, UniqueConstraint, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.types import TypeEngine

from voxint.db.models import (
    Base,
    EnrichmentCandidate,
    EnrichmentCandidateEvidence,
    EnrichmentProducerRun,
    ProfileReviewDecision,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

NEW_TABLES = (
    "enrichment_producer_runs",
    "enrichment_candidates",
    "enrichment_candidate_evidence",
    "profile_review_decisions",
)

SPEAKER_ID = "00000000-0000-0000-0000-00000000000a"
PRODUCER_RUN_ID = "00000000-0000-0000-0000-000000000010"
CANDIDATE_ID = "00000000-0000-0000-0000-000000000020"
EVIDENCE_ID = "00000000-0000-0000-0000-000000000030"
DECISION_ID = "00000000-0000-0000-0000-000000000040"


def _pg_type(type_: TypeEngine[object]) -> str:
    """Render a type as its postgres DDL string so model and reflected agree.

    Postgres has no distinct FLOAT type — ``FLOAT`` *is* ``DOUBLE PRECISION``
    (float8), so the model's ``Float`` compiles to "FLOAT" while reflection
    returns "DOUBLE PRECISION". Normalize that one server-side equivalence so
    the parity check compares real types, not spelling.
    """
    compiled = type_.compile(dialect=postgresql.dialect())
    return "DOUBLE PRECISION" if compiled == "FLOAT" else compiled


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    # Depending on `engine` guarantees DATABASE_URL is exported and the schema
    # starts at head. The finally restores a pristine head regardless of outcome
    # so a mid-suite migration test never leaves the DB downgraded for the tests
    # that follow.
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


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _seed_speaker(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO speakers (id, display_name) VALUES"
                f" ('{SPEAKER_ID}', 'Enrichment Parity Speaker')"
            )
        )
        conn.commit()


def _insert_producer_run(engine: Engine, row_id: str = PRODUCER_RUN_ID) -> None:
    """Minimal valid speaker-scope invocation, relying on created_at default."""
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO enrichment_producer_runs"
                " (id, producer, producer_version, target_kind, speaker_id,"
                "  covered_fields, generation, outcome, idempotency_key,"
                "  started_at, completed_at)"
                f" VALUES ('{row_id}', 'name_miner', '1.0', 'speaker',"
                f" '{SPEAKER_ID}', ARRAY['name'], 1, 'found', 'idem-{row_id}',"
                " now(), now())"
            )
        )
        conn.commit()


def _insert_candidate(engine: Engine, row_id: str = CANDIDATE_ID) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO enrichment_candidates"
                " (id, producer_run_id, target_kind, speaker_id, field, value)"
                f" VALUES ('{row_id}', '{PRODUCER_RUN_ID}', 'speaker',"
                f" '{SPEAKER_ID}', 'name', 'Jane Interviewee')"
            )
        )
        conn.commit()


def test_migration_0010_roundtrip(alembic_cfg: Config, engine: Engine) -> None:
    # --- at 0009: none of the four tables exists yet ---
    command.downgrade(alembic_cfg, "0009")
    present = _tables(engine)
    for table in NEW_TABLES:
        assert table not in present

    # --- upgrade to 0010: all four appear ---
    command.upgrade(alembic_cfg, "0010")
    present = _tables(engine)
    for table in NEW_TABLES:
        assert table in present

    _seed_speaker(engine)
    _insert_producer_run(engine)
    _insert_candidate(engine)
    with engine.connect() as conn:
        # a minimal candidate relies on the score_components server default
        row = conn.execute(
            text(
                "SELECT score, score_components, superseded_by_producer_run_id"
                " FROM enrichment_candidates"
            )
        ).one()
    assert row.score is None
    assert row.score_components == {}
    assert row.superseded_by_producer_run_id is None

    # --- downgrade is clean (purely additive migration) ---
    command.downgrade(alembic_cfg, "0009")
    present = _tables(engine)
    for table in NEW_TABLES:
        assert table not in present
    with engine.connect() as conn:
        functions = {
            r.proname
            for r in conn.execute(
                text("SELECT proname FROM pg_proc WHERE proname LIKE '%append_only%'"
                     " OR proname LIKE '%content_immutable%'")
            )
        }
    assert "profile_review_decisions_append_only" not in functions
    assert "enrichment_candidates_content_immutable" not in functions
    assert "enrichment_candidate_evidence_append_only" not in functions
    assert "enrichment_producer_runs_append_only" not in functions


@pytest.fixture()
def seeded(engine: Engine) -> Iterator[Engine]:
    """A speaker, one producer run, and one candidate at head schema."""
    _seed_speaker(engine)
    _insert_producer_run(engine)
    _insert_candidate(engine)
    try:
        yield engine
    finally:
        # TRUNCATE fires no row-level triggers, so it crosses the
        # immutability/append-only guards cleanly.
        with engine.connect() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
            conn.commit()


def test_producer_run_constraints(seeded: Engine) -> None:
    engine = seeded
    bad_rows = {
        # empty producer
        "producer_empty": "('00000000-0000-0000-0000-000000000011', '  ', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 1, 'found',"
        " 'k1', now(), now())",
        # speaker shape violated: run target with a speaker_id
        "shape": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'run', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 1, 'found',"
        " 'k1', now(), now())",
        # unknown covered field
        "covered_fields": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['salary'], 1, 'found',"
        " 'k1', now(), now())",
        # empty covered fields
        "covered_fields_empty": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY[]::text[], 1, 'found',"
        " 'k1', now(), now())",
        # generation < 1
        "generation": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 0, 'found',"
        " 'k1', now(), now())",
        # unknown outcome
        "outcome": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 1, 'partial',"
        " 'k1', now(), now())",
        # duplicate idempotency key (existing row used idem-<PRODUCER_RUN_ID>)
        "idempotency": "('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 2, 'found',"
        f" 'idem-{PRODUCER_RUN_ID}', now(), now())",
        # completed before started
        "completed_before_started": "('00000000-0000-0000-0000-000000000011', 'p',"
        f" '1.0', 'speaker', '{SPEAKER_ID}', NULL, NULL, ARRAY['name'], 1,"
        " 'found', 'k1', now(), now() - interval '1 second')",
    }
    for values in bad_rows.values():
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO enrichment_producer_runs"
                    " (id, producer, producer_version, target_kind, speaker_id,"
                    "  pipeline_run_id, diarization_label, covered_fields,"
                    "  generation, outcome, idempotency_key, started_at, completed_at)"
                    f" VALUES {values}"
                )
            )
    # config without its schema version violates the pairing CHECK
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO enrichment_producer_runs"
                " (id, producer, producer_version, target_kind, speaker_id,"
                "  covered_fields, generation, outcome, idempotency_key,"
                "  started_at, completed_at, config)"
                " VALUES ('00000000-0000-0000-0000-000000000011', 'p', '1.0',"
                f" 'speaker', '{SPEAKER_ID}', ARRAY['name'], 1, 'found', 'k1',"
                " now(), now(), '{}'::jsonb)"
            )
        )


def test_candidate_constraints(seeded: Engine) -> None:
    engine = seeded
    bad_rows = {
        # run_label shape without a label
        "shape": f"('00000000-0000-0000-0000-000000000021', '{PRODUCER_RUN_ID}',"
        f" 'run_label', '{SPEAKER_ID}', NULL, NULL, 'name', 'Jane', NULL)",
        # unknown claim field
        "field": f"('00000000-0000-0000-0000-000000000021', '{PRODUCER_RUN_ID}',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, 'salary', 'Jane', NULL)",
        # whitespace-only value
        "value": f"('00000000-0000-0000-0000-000000000021', '{PRODUCER_RUN_ID}',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, 'name', '   ', NULL)",
        # score out of range
        "score": f"('00000000-0000-0000-0000-000000000021', '{PRODUCER_RUN_ID}',"
        f" 'speaker', '{SPEAKER_ID}', NULL, NULL, 'name', 'Jane', 1.5)",
    }
    for values in bad_rows.values():
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO enrichment_candidates"
                    " (id, producer_run_id, target_kind, speaker_id,"
                    "  pipeline_run_id, diarization_label, field, value, score)"
                    f" VALUES {values}"
                )
            )
    # score_components must be a JSON object, not an array
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO enrichment_candidates"
                " (id, producer_run_id, target_kind, speaker_id, field, value,"
                "  score_components)"
                f" VALUES ('00000000-0000-0000-0000-000000000021', '{PRODUCER_RUN_ID}',"
                f" 'speaker', '{SPEAKER_ID}', 'name', 'Jane', '[]'::jsonb)"
            )
        )


def test_evidence_constraints(seeded: Engine) -> None:
    engine = seeded
    # valid url-kind evidence row
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO enrichment_candidate_evidence"
                " (id, candidate_id, ordinal, kind, url, snippet)"
                f" VALUES ('{EVIDENCE_ID}', '{CANDIDATE_ID}', 0, 'url',"
                " 'https://example.com/about', 'Jane hosts the show')"
            )
        )
        conn.commit()
    bad_rows = {
        # duplicate ordinal for the same candidate
        "ordinal_key": f"('00000000-0000-0000-0000-000000000031', '{CANDIDATE_ID}',"
        " 0, 'url', NULL, NULL, NULL, NULL, 'https://example.com', NULL)",
        # metadata kind without its columns
        "metadata_shape": f"('00000000-0000-0000-0000-000000000031', '{CANDIDATE_ID}',"
        " 1, 'metadata_field', NULL, NULL, NULL, NULL, NULL, NULL)",
        # url kind carrying transcript columns
        "url_shape": f"('00000000-0000-0000-0000-000000000031', '{CANDIDATE_ID}',"
        " 1, 'url', NULL, NULL, NULL, 12.5, 'https://example.com', NULL)",
        # negative ordinal
        "ordinal_negative": f"('00000000-0000-0000-0000-000000000031', '{CANDIDATE_ID}',"
        " -1, 'url', NULL, NULL, NULL, NULL, 'https://example.com', NULL)",
    }
    for values in bad_rows.values():
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO enrichment_candidate_evidence"
                    " (id, candidate_id, ordinal, kind, source_metadata_id,"
                    "  source_field, transcript_segment_id, timestamp_seconds,"
                    "  url, snippet)"
                    f" VALUES {values}"
                )
            )
    # oversized url (2049 chars)
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO enrichment_candidate_evidence"
                " (id, candidate_id, ordinal, kind, url)"
                f" VALUES ('00000000-0000-0000-0000-000000000031', '{CANDIDATE_ID}',"
                " 1, 'url', 'https://e.com/' || repeat('x', 2035))"
            )
        )
    # evidence is append-only: UPDATE and DELETE both raise via trigger
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(
            text("UPDATE enrichment_candidate_evidence SET snippet = 'edited'")
        )
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM enrichment_candidate_evidence"))


def test_producer_run_append_only_trigger(seeded: Engine) -> None:
    engine = seeded
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("UPDATE enrichment_producer_runs SET generation = 99"))
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM enrichment_producer_runs"))


def test_candidate_born_unsuperseded(seeded: Engine) -> None:
    """An initial non-NULL supersession stamp is rejected at INSERT."""
    engine = seeded
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(
            text(
                "INSERT INTO enrichment_candidates"
                " (id, producer_run_id, target_kind, speaker_id, field, value,"
                "  superseded_by_producer_run_id)"
                " VALUES ('00000000-0000-0000-0000-000000000023',"
                f" '{PRODUCER_RUN_ID}', 'speaker', '{SPEAKER_ID}', 'name',"
                f" 'Jane', '{PRODUCER_RUN_ID}')"
            )
        )


def test_candidate_immutability_trigger(seeded: Engine) -> None:
    engine = seeded
    # content edits and DELETE are blocked
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("UPDATE enrichment_candidates SET value = 'Someone Else'"))
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM enrichment_candidates"))
    # ... but stamping supersession once is allowed
    _insert_producer_run(engine, "00000000-0000-0000-0000-000000000012")
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE enrichment_candidates SET superseded_by_producer_run_id ="
                " '00000000-0000-0000-0000-000000000012'"
            )
        )
        conn.commit()
    # ... and only once: re-stamping is write-once
    _insert_producer_run(engine, "00000000-0000-0000-0000-000000000013")
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(
            text(
                "UPDATE enrichment_candidates SET superseded_by_producer_run_id ="
                " '00000000-0000-0000-0000-000000000013'"
            )
        )


def test_profile_review_decision_constraints(seeded: Engine) -> None:
    engine = seeded
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO profile_review_decisions"
                " (id, candidate_id, decision, operator, idempotency_key)"
                f" VALUES ('{DECISION_ID}', '{CANDIDATE_ID}', 'accept', 'ben', 'd1')"
            )
        )
        conn.commit()
    # terminal per candidate: a second decision violates the UNIQUE
    with engine.connect() as conn, pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO profile_review_decisions"
                " (id, candidate_id, decision, operator, idempotency_key)"
                " VALUES ('00000000-0000-0000-0000-000000000041',"
                f" '{CANDIDATE_ID}', 'reject', 'ben', 'd2')"
            )
        )
    # unknown decision / empty operator
    _insert_candidate(engine, "00000000-0000-0000-0000-000000000022")
    for values in (
        "('00000000-0000-0000-0000-000000000041',"
        " '00000000-0000-0000-0000-000000000022', 'maybe', 'ben', 'd3')",
        "('00000000-0000-0000-0000-000000000041',"
        " '00000000-0000-0000-0000-000000000022', 'accept', '  ', 'd3')",
    ):
        with engine.connect() as conn, pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO profile_review_decisions"
                    " (id, candidate_id, decision, operator, idempotency_key)"
                    f" VALUES {values}"
                )
            )
    # append-only: UPDATE and DELETE both raise via trigger
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("UPDATE profile_review_decisions SET operator = 'x'"))
    with engine.connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM profile_review_decisions"))


@pytest.mark.parametrize(
    "model",
    [
        EnrichmentProducerRun,
        EnrichmentCandidate,
        EnrichmentCandidateEvidence,
        ProfileReviewDecision,
    ],
)
def test_enrichment_models_match_migration(engine: Engine, model: type[Base]) -> None:
    """The ORM models and the migrated tables agree on columns + nullability.

    The suite builds its schema from the alembic chain, not create_all, and there
    is no autogenerate parity check — so assert it here for each new table.
    """
    table = model.__table__
    assert isinstance(table, Table)
    insp = inspect(engine)
    migrated = {c["name"]: c for c in insp.get_columns(table.name)}
    model_cols = {c.name: c for c in table.columns}
    assert set(migrated) == set(model_cols)
    for name, col in model_cols.items():
        assert migrated[name]["nullable"] == col.nullable, f"{name} nullability drift"
        assert _pg_type(migrated[name]["type"]) == _pg_type(col.type), (
            f"{name} type drift: migrated={_pg_type(migrated[name]['type'])} "
            f"model={_pg_type(col.type)}"
        )
    migrated_checks = {c["name"] for c in insp.get_check_constraints(table.name)}
    model_checks = {
        c.name for c in table.constraints if isinstance(c.name, str) and "check" in c.name
    }
    assert model_checks <= migrated_checks, (
        f"missing CHECKs in migration: {model_checks - migrated_checks}"
    )
    migrated_uniques = {
        tuple(u["column_names"]) for u in insp.get_unique_constraints(table.name)
    }
    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            cols = tuple(c.name for c in constraint.columns)
            assert cols in migrated_uniques, f"missing UNIQUE on {cols}"
    migrated_fks = {
        (tuple(f["constrained_columns"]), f["referred_table"])
        for f in insp.get_foreign_keys(table.name)
    }
    for fk in table.foreign_keys:
        pair = ((fk.parent.name,), fk.column.table.name)
        assert pair in migrated_fks, f"missing FK {pair}"
