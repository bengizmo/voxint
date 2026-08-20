"""Migration 0031 (operator annotation layer), up/down + constraints (issue #86).

Real alembic up/down against the shared test database (head restored in
teardown): the three new tables appear with their columns, a word_range
annotation round-trips, the per-kind shape / hash-hex / color / segment-order
CHECKs and the tag normalized-uniqueness constraint reject bad rows, the FKs
cascade on run / segment / annotation delete, downgrade drops all three tables,
and the ORM models stay in lockstep with the DDL.
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

from voxint.db.models import (
    AnnotationTag,
    AnnotationTagLink,
    TranscriptAnnotation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_HEX64 = "a" * 64


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


def _seed_run_with_segments(engine: Engine) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """One run with two segments (index 0 and 1). Returns (run, seg0, seg1)."""
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    seg0, seg1 = uuid.uuid4(), uuid.uuid4()
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
        for seg_id, idx, start in ((seg0, 0, 0.0), (seg1, 1, 1.0)):
            conn.execute(
                text(
                    "INSERT INTO transcript_segments"
                    " (id, pipeline_run_id, segment_index, start_seconds,"
                    " end_seconds, raw_text)"
                    " VALUES (:id, :r, :i, :s, :e, 'hello world')"
                ),
                {"id": seg_id, "r": run_id, "i": idx, "s": start, "e": start + 1.0},
            )
        conn.commit()
    return run_id, seg0, seg1


def _insert_word_range(
    engine: Engine,
    run_id: uuid.UUID,
    seg_id: uuid.UUID,
    **overrides: object,
) -> None:
    """Insert a valid single-segment word_range annotation, with overrides."""
    row: dict[str, object] = {
        "id": uuid.uuid4(),
        "pipeline_run_id": run_id,
        "anchor_schema_version": 1,
        "anchor_kind": "word_range",
        "start_segment_id": seg_id,
        "end_segment_id": seg_id,
        "start_segment_index": 0,
        "end_segment_index": 0,
        "start_word_index": 0,
        "end_word_index": 2,
        "start_char_offset": None,
        "end_char_offset": None,
        "source_text_hash": _HEX64,
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "quote_text": "hello world",
        "color_index": 0,
        "note": None,
        "operator": "ben",
        "idempotency_key": None,
        "request_fingerprint": None,
    }
    row.update(overrides)
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO transcript_annotations"
                " (id, pipeline_run_id, anchor_schema_version, anchor_kind,"
                " start_segment_id, end_segment_id, start_segment_index,"
                " end_segment_index, start_word_index, end_word_index,"
                " start_char_offset, end_char_offset, source_text_hash,"
                " start_seconds, end_seconds, quote_text, color_index, note,"
                " operator, idempotency_key, request_fingerprint)"
                " VALUES (:id, :pipeline_run_id, :anchor_schema_version,"
                " :anchor_kind, :start_segment_id, :end_segment_id,"
                " :start_segment_index, :end_segment_index, :start_word_index,"
                " :end_word_index, :start_char_offset, :end_char_offset,"
                " :source_text_hash, :start_seconds, :end_seconds, :quote_text,"
                " :color_index, :note, :operator, :idempotency_key,"
                " :request_fingerprint)"
            ),
            row,
        )
        conn.commit()


def _insert_tag(engine: Engine, name: str, normalized: str, color: int = 0) -> uuid.UUID:
    tag_id = uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO annotation_tags (id, name, name_normalized, color)"
                " VALUES (:id, :n, :nn, :c)"
            ),
            {"id": tag_id, "n": name, "nn": normalized, "c": color},
        )
        conn.commit()
    return tag_id


def test_upgrade_creates_tables_and_columns(engine: Engine, alembic_cfg: Config) -> None:
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert {"annotation_tags", "transcript_annotations", "annotation_tag_links"} <= tables

    assert {c["name"] for c in insp.get_columns("annotation_tags")} == {
        "id",
        "name",
        "name_normalized",
        "color",
        "created_at",
        "archived_at",
    }
    assert {c["name"] for c in insp.get_columns("transcript_annotations")} == {
        "id",
        "pipeline_run_id",
        "anchor_schema_version",
        "anchor_kind",
        "start_segment_id",
        "end_segment_id",
        "start_segment_index",
        "end_segment_index",
        "start_word_index",
        "end_word_index",
        "start_char_offset",
        "end_char_offset",
        "source_text_hash",
        "start_seconds",
        "end_seconds",
        "quote_text",
        "color_index",
        "note",
        "operator",
        "idempotency_key",
        "request_fingerprint",
        "created_at",
        "updated_at",
        "deleted_at",
    }
    assert {c["name"] for c in insp.get_columns("annotation_tag_links")} == {
        "annotation_id",
        "tag_id",
    }


def test_word_range_roundtrips(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg0, _ = _seed_run_with_segments(engine)
    _insert_word_range(engine, run_id, seg0)
    with engine.connect() as conn:
        kind, sw, ew, quote = conn.execute(
            text(
                "SELECT anchor_kind, start_word_index, end_word_index, quote_text"
                " FROM transcript_annotations WHERE pipeline_run_id = :r"
            ),
            {"r": run_id},
        ).one()
    assert (kind, sw, ew, quote) == ("word_range", 0, 2, "hello world")


@pytest.mark.parametrize(
    "overrides",
    [
        # word_range must NOT carry char offsets (per-kind shape CHECK).
        {"start_char_offset": 0, "end_char_offset": 3},
        # word_range must carry a word pair (kind shape CHECK).
        {"start_word_index": None, "end_word_index": None},
        # hash must be 64 lowercase hex chars.
        {"source_text_hash": "NOTHEX"},
        {"source_text_hash": "a" * 63},
        # color out of the 6-color palette.
        {"color_index": 6},
        # schema version pinned to 1.
        {"anchor_schema_version": 2},
        # same-segment half-open range must be non-empty (end > start).
        {"start_word_index": 2, "end_word_index": 2},
        # segment-index order.
        {"start_segment_index": 1, "end_segment_index": 0},
        # unknown anchor kind.
        {"anchor_kind": "line_range"},
        # seconds must be paired.
        {"start_seconds": 0.0, "end_seconds": None},
        # timing is kind-gated: a word_range MUST carry precise seconds.
        {"start_seconds": None, "end_seconds": None},
    ],
)
def test_check_constraints_reject(
    engine: Engine, alembic_cfg: Config, overrides: dict[str, object]
) -> None:
    run_id, seg0, _ = _seed_run_with_segments(engine)
    with pytest.raises(IntegrityError):
        _insert_word_range(engine, run_id, seg0, **overrides)


def test_seconds_are_gated_by_anchor_kind(engine: Engine, alembic_cfg: Config) -> None:
    # A non-word_range MUST NOT carry precise seconds (timing honesty): a
    # text_range with seconds populated is refused.
    run_id, seg0, _ = _seed_run_with_segments(engine)
    with pytest.raises(IntegrityError):
        _insert_word_range(
            engine,
            run_id,
            seg0,
            anchor_kind="text_range",
            start_word_index=None,
            end_word_index=None,
            start_char_offset=0,
            end_char_offset=5,
            start_seconds=0.0,
            end_seconds=1.0,
        )


def test_cross_segment_text_range_allows_descending_offsets(
    engine: Engine, alembic_cfg: Config
) -> None:
    # A legal cross-segment selection: the end offset indexes a DIFFERENT
    # segment's text than the start offset, so it may be numerically smaller.
    # The same-segment ordering CHECKs must NOT reject this (false-positive guard).
    run_id, seg0, seg1 = _seed_run_with_segments(engine)
    _insert_word_range(
        engine,
        run_id,
        seg0,
        anchor_kind="text_range",
        end_segment_id=seg1,
        end_segment_index=1,
        start_word_index=None,
        end_word_index=None,
        start_char_offset=9,
        end_char_offset=2,
        start_seconds=None,
        end_seconds=None,
    )
    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT start_char_offset, end_char_offset FROM transcript_annotations"
                " WHERE pipeline_run_id = :r"
            ),
            {"r": run_id},
        ).one()
    assert stored == (9, 2)


def test_segment_range_and_text_range_shapes(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg0, seg1 = _seed_run_with_segments(engine)
    # segment_range: all four offsets NULL, spanning two segments.
    _insert_word_range(
        engine,
        run_id,
        seg0,
        anchor_kind="segment_range",
        end_segment_id=seg1,
        end_segment_index=1,
        start_word_index=None,
        end_word_index=None,
        start_seconds=None,
        end_seconds=None,
    )
    # text_range: char pair present, word pair absent, no precise seconds.
    _insert_word_range(
        engine,
        run_id,
        seg0,
        anchor_kind="text_range",
        start_word_index=None,
        end_word_index=None,
        start_char_offset=0,
        end_char_offset=5,
        start_seconds=None,
        end_seconds=None,
    )
    with engine.connect() as conn:
        kinds = {
            k
            for (k,) in conn.execute(
                text("SELECT anchor_kind FROM transcript_annotations WHERE pipeline_run_id = :r"),
                {"r": run_id},
            )
        }
    assert kinds == {"segment_range", "text_range"}


def test_tag_normalized_uniqueness(engine: Engine, alembic_cfg: Config) -> None:
    _insert_tag(engine, "Key Point", "key point")
    with pytest.raises(IntegrityError):
        _insert_tag(engine, "key point", "key point")


@pytest.mark.parametrize("overrides", [{"color": 6}, {"name": "   "}])
def test_tag_check_constraints_reject(
    engine: Engine, alembic_cfg: Config, overrides: dict[str, object]
) -> None:
    with pytest.raises(IntegrityError):
        name = str(overrides.get("name", "tag"))
        color = int(overrides.get("color", 0))  # type: ignore[arg-type]
        _insert_tag(engine, name, name.strip().casefold() or "x", color)


def test_annotation_fk_ondelete_cascade_declared(engine: Engine, alembic_cfg: Config) -> None:
    # The run and both endpoint-segment FKs are ON DELETE CASCADE so a run
    # teardown / re-transcription (which mints new segment ids) leaves no orphan
    # annotation. The run FK cannot be exercised behaviourally in isolation
    # (transcript_segments RESTRICT the run), so its cascade is asserted on the
    # DDL here and proven through the segment path in test_segment_delete_cascades.
    fks = {
        tuple(fk["constrained_columns"]): (fk["referred_table"], fk["options"].get("ondelete"))
        for fk in inspect(engine).get_foreign_keys("transcript_annotations")
    }
    assert fks[("pipeline_run_id",)] == ("pipeline_runs", "CASCADE")
    assert fks[("start_segment_id",)] == ("transcript_segments", "CASCADE")
    assert fks[("end_segment_id",)] == ("transcript_segments", "CASCADE")

    link_fks = {
        tuple(fk["constrained_columns"]): (fk["referred_table"], fk["options"].get("ondelete"))
        for fk in inspect(engine).get_foreign_keys("annotation_tag_links")
    }
    # A deleted annotation drops its links; a tag has no delete in v1, so its FK
    # is a plain restrict (no cascade from tag → link).
    assert link_fks[("annotation_id",)] == ("transcript_annotations", "CASCADE")
    assert link_fks[("tag_id",)][1] in (None, "NO ACTION", "RESTRICT")


def test_segment_delete_cascades(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg0, _ = _seed_run_with_segments(engine)
    _insert_word_range(engine, run_id, seg0)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM transcript_segments WHERE id = :s"), {"s": seg0})
        conn.commit()
        remaining = conn.execute(
            text("SELECT count(*) FROM transcript_annotations WHERE start_segment_id = :s"),
            {"s": seg0},
        ).scalar_one()
    assert remaining == 0


def test_tag_link_cascades_with_annotation(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg0, _ = _seed_run_with_segments(engine)
    ann_id = uuid.uuid4()
    _insert_word_range(engine, run_id, seg0, id=ann_id)
    tag_id = _insert_tag(engine, "topic", "topic")
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO annotation_tag_links (annotation_id, tag_id) VALUES (:a, :t)"),
            {"a": ann_id, "t": tag_id},
        )
        conn.commit()
        # Deleting the annotation removes its links; the tag itself survives.
        conn.execute(text("DELETE FROM transcript_annotations WHERE id = :a"), {"a": ann_id})
        conn.commit()
        links = conn.execute(
            text("SELECT count(*) FROM annotation_tag_links WHERE tag_id = :t"),
            {"t": tag_id},
        ).scalar_one()
        tags = conn.execute(
            text("SELECT count(*) FROM annotation_tags WHERE id = :t"), {"t": tag_id}
        ).scalar_one()
    assert links == 0
    assert tags == 1


def test_downgrade_drops_tables(engine: Engine, alembic_cfg: Config) -> None:
    run_id, seg0, _ = _seed_run_with_segments(engine)
    _insert_word_range(engine, run_id, seg0)
    command.downgrade(alembic_cfg, "0030")
    tables = set(inspect(engine).get_table_names())
    assert not ({"annotation_tags", "transcript_annotations", "annotation_tag_links"} & tables)
    command.upgrade(alembic_cfg, "head")
    assert {
        "annotation_tags",
        "transcript_annotations",
        "annotation_tag_links",
    } <= set(inspect(engine).get_table_names())


@pytest.mark.parametrize(
    ("model", "table"),
    [
        (AnnotationTag, "annotation_tags"),
        (TranscriptAnnotation, "transcript_annotations"),
        (AnnotationTagLink, "annotation_tag_links"),
    ],
)
def test_models_match_migrated_schema(
    engine: Engine, alembic_cfg: Config, model: type, table: str
) -> None:
    # Compare type AND nullability so a drift like making source_text_hash
    # nullable on one side (but not the other) is caught.
    reflected = {
        col["name"]: (_pg_type(col["type"]), col["nullable"])
        for col in inspect(engine).get_columns(table)
    }
    declared = {col.name: (_pg_type(col.type), col.nullable) for col in model.__table__.columns}
    assert reflected == declared
