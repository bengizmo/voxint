"""Webhook emission via cas_update_run (issue #12, phase B), against real Postgres.

Covers the transactional-outbox emission side: a notifiable transition inserts
exactly one notification_deliveries row IN the transition's transaction, gated by
notify_enabled, atomic under rollback, idempotent per (run_id, transition_revision),
and with FAILED held for the initial delay. Non-notifiable transitions and the
no-settings path emit nothing. Delivery (the sweep) is a separate phase.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import NotificationStatus, PipelineRun, RunStatus, Stage
from voxint.pipeline.transitions import cas_update_run, snapshot


def _settings(*, enabled: bool = True, failed_delay: int = 15) -> Settings:
    return Settings(
        _env_file=None,
        notify_enabled=enabled,
        notify_webhook_url="https://hooks.example.com/voxint",
        notify_webhook_secret="a-sufficiently-long-secret",
        notify_failed_initial_delay_seconds=failed_delay,
    )


def _seed_running_run(session: Session, *, stage: str = "finalize") -> uuid.UUID:
    """Insert a media item + a RUNNING run at revision 0 on the given stage."""
    rid = uuid.uuid4()
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
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


def _rows(session: Session, run_id: uuid.UUID) -> list[dict[str, object]]:
    result = session.execute(
        text(
            "SELECT event, status, transition_revision, next_attempt_at, payload, id"
            " FROM notification_deliveries WHERE pipeline_run_id = :rid"
        ),
        {"rid": run_id},
    )
    return [dict(r._mapping) for r in result]


def test_completed_transition_emits_one_row(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        new = cas_update_run(
            session,
            held,
            status=RunStatus.COMPLETED,
            current_stage=None,
            settings=_settings(),
        )
        session.commit()

    with session_factory() as session:
        rows = _rows(session, rid)
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "completed"
    assert row["status"] == NotificationStatus.PENDING.value
    assert row["transition_revision"] == new.revision == 1
    payload = row["payload"]
    assert isinstance(payload, dict)
    assert payload["event"] == "completed"
    assert payload["run_id"] == str(rid)
    assert payload["transition_revision"] == 1
    assert payload["delivery_id"] == str(row["id"])
    assert payload["schema_version"] == 1
    assert "error" not in payload  # run.error is deliberately never included


def test_disabled_emits_nothing(session_factory: sessionmaker[Session]) -> None:
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
        assert _rows(session, rid) == []


def test_no_settings_emits_nothing(session_factory: sessionmaker[Session]) -> None:
    # The default (tests / ingest / CLI) path passes no settings — never emits.
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(session, held, status=RunStatus.COMPLETED, current_stage=None)
        session.commit()
    with session_factory() as session:
        assert _rows(session, rid) == []


def test_non_notifiable_transition_emits_nothing(session_factory: sessionmaker[Session]) -> None:
    # RUNNING -> RUNNING (stage advance) is not news, even with notify enabled.
    with session_factory() as session:
        rid = _seed_running_run(session, stage="transcribe")
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session,
            held,
            status=RunStatus.RUNNING,
            current_stage=Stage.DIARIZE_EMBED,  # the legal advance from transcribe
            settings=_settings(),
        )
        session.commit()
    with session_factory() as session:
        assert _rows(session, rid) == []


def test_rollback_removes_row_and_transition(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        rid = _seed_running_run(session)
        session.commit()
    with session_factory() as session:
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session, held, status=RunStatus.COMPLETED, current_stage=None, settings=_settings()
        )
        session.rollback()  # the outbox row must vanish with the transition
    with session_factory() as session:
        assert _rows(session, rid) == []
        run = session.get_one(PipelineRun, rid)
        assert run.status == RunStatus.RUNNING.value  # transition rolled back too


def test_failed_row_is_held_for_initial_delay(session_factory: sessionmaker[Session]) -> None:
    before = datetime.now(tz=UTC)
    with session_factory() as session:
        rid = _seed_running_run(session)
        held = snapshot(session.get_one(PipelineRun, rid))
        cas_update_run(
            session,
            held,
            status=RunStatus.FAILED,
            current_stage=Stage.FINALIZE,
            error="boom",
            settings=_settings(failed_delay=60),
        )
        session.commit()
    with session_factory() as session:
        rows = _rows(session, rid)
    assert len(rows) == 1
    next_at = rows[0]["next_attempt_at"]
    assert isinstance(next_at, datetime)
    # Held roughly the initial delay into the future (allow scheduling slack).
    assert (next_at - before).total_seconds() >= 55


def test_idempotent_on_run_revision(session_factory: sessionmaker[Session]) -> None:
    # A second emission for the same arrival (same run_id + transition_revision)
    # is a no-op — ON CONFLICT DO NOTHING keeps exactly one row.
    from voxint.notify import record_transition

    with session_factory() as session:
        rid = _seed_running_run(session)
        record_transition(
            session,
            run_id=rid,
            status=RunStatus.COMPLETED,
            transition_revision=1,
            settings=_settings(),
        )
        record_transition(
            session,
            run_id=rid,
            status=RunStatus.COMPLETED,
            transition_revision=1,
            settings=_settings(),
        )
        session.commit()
    with session_factory() as session:
        assert len(_rows(session, rid)) == 1
