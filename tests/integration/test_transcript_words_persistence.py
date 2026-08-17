"""Persistence semantics of the transcript_segments.words JSONB column (#59).

The distinction that matters downstream (and for the future split UI): a run
with word timing stores an array on every segment; a wordless run stores real
SQL ``NULL`` — never JSONB ``'null'`` — so ``words IS NULL`` and the array-shape
CHECK stay honest. These assertions go through raw SQL because the ORM read path
hides the SQL-NULL vs JSONB-null difference the fix turns on.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import MediaItem, PipelineRun, RunStatus, TranscriptSegment


def _seed_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    return run.id


def _add_segment(
    session: Session, run_id: uuid.UUID, index: int, words: object
) -> uuid.UUID:
    seg = TranscriptSegment(
        pipeline_run_id=run_id,
        segment_index=index,
        start_seconds=float(index),
        end_seconds=float(index + 1),
        raw_text="seg",
        words=words,
    )
    session.add(seg)
    session.flush()
    return seg.id


def test_none_words_persist_as_sql_null_not_json_null(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        run_id = _seed_run(session)
        seg_id = _add_segment(session, run_id, 0, None)
        session.commit()
        row = session.execute(
            text(
                "SELECT words IS NULL AS is_null, jsonb_typeof(words) AS typeof "
                "FROM transcript_segments WHERE id = :id"
            ),
            {"id": seg_id},
        ).mappings().one()
    assert row["is_null"] is True
    assert row["typeof"] is None  # not the JSONB scalar 'null'


def test_array_words_persist_as_jsonb_array(
    session_factory: sessionmaker[Session],
) -> None:
    payload = [{"start": 0.0, "end": 0.5, "word": "hi", "confidence": 0.9}]
    with session_factory() as session:
        run_id = _seed_run(session)
        seg_id = _add_segment(session, run_id, 0, payload)
        empty_id = _add_segment(session, run_id, 1, [])  # word-timed but no bucket
        session.commit()
        rows = session.execute(
            text(
                "SELECT id, words IS NULL AS is_null, jsonb_typeof(words) AS typeof "
                "FROM transcript_segments WHERE id IN (:a, :b)"
            ),
            {"a": seg_id, "b": empty_id},
        ).mappings().all()
    by_id = {r["id"]: r for r in rows}
    assert by_id[seg_id]["is_null"] is False
    assert by_id[seg_id]["typeof"] == "array"
    # An empty array is distinguishable from NULL: word-timed run, empty bucket.
    assert by_id[empty_id]["is_null"] is False
    assert by_id[empty_id]["typeof"] == "array"


def test_non_array_words_rejected_by_check(
    session_factory: sessionmaker[Session],
) -> None:
    # Raw insert of a JSONB object bypasses the ORM shape; the DB CHECK is the
    # backstop that keeps words an array (or NULL).
    with session_factory() as session:
        run_id = _seed_run(session)
        with pytest.raises(IntegrityError):
            session.execute(
                text(
                    "INSERT INTO transcript_segments "
                    "(id, pipeline_run_id, segment_index, start_seconds, "
                    "end_seconds, raw_text, words) VALUES "
                    "(:id, :run, 0, 0, 1, 'seg', '{\"not\": \"an array\"}'::jsonb)"
                ),
                {"id": uuid.uuid4(), "run": run_id},
            )
