"""Activity outbox emission + read/prune helpers, against real Postgres (#162).

The write side of the console activity feed: a run reaching COMPLETED via
``cas_update_run`` inserts exactly one ``activity_events`` row IN the transition's
transaction, gated by ``console_activity_enabled``, atomic under rollback, and
idempotent on the occurrence key. Non-terminal transitions and the no-settings
path emit nothing. The ``events_since`` / ``high_water`` / ``prune`` helpers back
the poll endpoint and the retention sweep.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from voxint.activity import (
    events_since,
    high_water,
    prune_activity_events,
    record_activity_event,
    record_run_completed,
    retained_floor,
)
from voxint.config import Settings
from voxint.db.models import ActivityEvent, ActivityKind, PipelineRun, RunStatus, Stage
from voxint.pipeline.transitions import cas_update_run, snapshot


def _settings(*, enabled: bool = True) -> Settings:
    return Settings(_env_file=None, console_activity_enabled=enabled)


def _seed_running_run(session: Session, *, stage: str = "finalize") -> uuid.UUID:
    rid = uuid.uuid4()
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/recording.wav"},
    )
    session.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, current_stage,"
            " revision, created_at, updated_at)"
            " VALUES (:rid, :mid, 'running', :stage, 0, now(), now())"
        ),
        {"rid": rid, "mid": mid, "stage": stage},
    )
    return rid


def _events(session: Session, run_id: uuid.UUID) -> list[ActivityEvent]:
    return list(
        session.query(ActivityEvent)
        .filter_by(pipeline_run_id=run_id)
        .order_by(ActivityEvent.id)
    )


def test_completed_emits_one_row(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session, held, status=RunStatus.COMPLETED, current_stage=None, settings=_settings()
        )
        session.commit()
    with session_factory() as session:
        rows = _events(session, rid)
        assert len(rows) == 1
        row = rows[0]
        assert row.kind == ActivityKind.RUN_COMPLETED.value
        assert row.occurrence_key == f"run:{rid}:completed"
        assert row.href == f"/jobs/{rid}"
        # friendly_media_label falls back to the cleaned filename.
        assert row.title == "recording.wav"


def test_flag_off_emits_nothing(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session,
            held,
            status=RunStatus.COMPLETED,
            current_stage=None,
            settings=_settings(enabled=False),
        )
        session.commit()
    with session_factory() as session:
        assert _events(session, rid) == []


def test_no_settings_emits_nothing(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
        session.commit()
    with session_factory() as session:
        assert _events(session, rid) == []


def test_non_terminal_transition_emits_nothing(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session, stage="transcribe")
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session,
            held,
            status=RunStatus.RUNNING,
            current_stage=Stage.DIARIZE_EMBED,
            settings=_settings(),
        )
        session.commit()
    with session_factory() as session:
        assert _events(session, rid) == []


def test_rollback_removes_row_and_transition(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        session.commit()
    with session_factory() as session:
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session, held, status=RunStatus.COMPLETED, current_stage=None, settings=_settings()
        )
        session.rollback()
    with session_factory() as session:
        assert _events(session, rid) == []
        assert session.get_one(PipelineRun, rid).status == RunStatus.RUNNING.value


def test_idempotent_on_occurrence_key(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        session.commit()
    with session_factory() as session:
        record_run_completed(session, rid)
        record_run_completed(session, rid)  # same occurrence key => one row
        session.commit()
    with session_factory() as session:
        assert len(_events(session, rid)) == 1


def _seed_completed_run(session: Session) -> uuid.UUID:
    rid = _seed_running_run(session)
    session.execute(
        text("UPDATE pipeline_runs SET status = 'completed', current_stage = NULL WHERE id = :rid"),
        {"rid": rid},
    )
    return rid


def test_events_since_and_high_water(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_completed_run(session)
        for i in range(5):
            record_activity_event(
                session,
                kind=ActivityKind.RUN_COMPLETED,
                occurrence_key=f"k-{i}",
                pipeline_run_id=rid,
                title=f"t{i}",
                href=f"/jobs/{rid}",
            )
        session.commit()
    with session_factory() as session:
        hw = high_water(session)
        all_rows = events_since(session, after_id=0, limit=10)
        assert [r.id for r in all_rows] == sorted(r.id for r in all_rows)  # ascending
        assert len(all_rows) == 5
        assert hw == all_rows[-1].id
        # A cursor mid-stream returns only newer rows, capped by the limit.
        mid = all_rows[1].id
        page = events_since(session, after_id=mid, limit=2)
        assert [r.id for r in page] == [all_rows[2].id, all_rows[3].id]


def test_high_water_empty_is_zero(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert high_water(session) == 0


def test_prune_keeps_newest_n(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_completed_run(session)
        for i in range(6):
            record_activity_event(
                session,
                kind=ActivityKind.RUN_COMPLETED,
                occurrence_key=f"k-{i}",
                pipeline_run_id=rid,
                title=f"t{i}",
                href=f"/jobs/{rid}",
            )
        session.commit()
    with session_factory() as session:
        newest_ids = [r.id for r in events_since(session, after_id=0, limit=100)][-3:]
        removed = prune_activity_events(session, keep=3)
        session.commit()
        assert removed == 3
    with session_factory() as session:
        survivors = [r.id for r in events_since(session, after_id=0, limit=100)]
        assert survivors == newest_ids


def test_prune_below_cap_is_noop(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_completed_run(session)
        for i in range(2):
            record_activity_event(
                session,
                kind=ActivityKind.RUN_COMPLETED,
                occurrence_key=f"k-{i}",
                pipeline_run_id=rid,
                title=f"t{i}",
                href=f"/jobs/{rid}",
            )
        session.commit()
    with session_factory() as session:
        assert prune_activity_events(session, keep=3) == 0
        session.commit()
    with session_factory() as session:
        assert len(events_since(session, after_id=0, limit=100)) == 2


def test_prune_empty_is_noop(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert prune_activity_events(session, keep=3) == 0


def test_prune_keep_below_one_is_noop(session_factory: sessionmaker[Session]) -> None:
    # keep<1 would compile to OFFSET -1 (a Postgres error); guarded to a no-op.
    with session_factory() as session:
        rid = _seed_completed_run(session)
        record_activity_event(
            session,
            kind=ActivityKind.RUN_COMPLETED,
            occurrence_key="k-0",
            pipeline_run_id=rid,
            title="t",
            href=f"/jobs/{rid}",
        )
        session.commit()
    with session_factory() as session:
        assert prune_activity_events(session, keep=0) == 0
        assert len(events_since(session, after_id=0, limit=100)) == 1


def test_retained_floor(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert retained_floor(session) == 0  # empty
    with session_factory() as session:
        rid = _seed_completed_run(session)
        for i in range(3):
            record_activity_event(
                session,
                kind=ActivityKind.RUN_COMPLETED,
                occurrence_key=f"k-{i}",
                pipeline_run_id=rid,
                title=f"t{i}",
                href=f"/jobs/{rid}",
            )
        session.commit()
    with session_factory() as session:
        rows = events_since(session, after_id=0, limit=100)
        assert retained_floor(session) == rows[0].id  # the smallest id
        # After a prune the floor rises to the oldest surviving row.
        prune_activity_events(session, keep=2)
        session.commit()
    with session_factory() as session:
        rows = events_since(session, after_id=0, limit=100)
        assert retained_floor(session) == rows[0].id == min(r.id for r in rows)
