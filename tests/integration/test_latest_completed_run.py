"""The dashboard's "last finished run" query (issue #117, Phase 0).

``latest_completed_run`` picks the newest COMPLETED, non-archived run by
*completion* time — the terminal FINALIZE stage's ``finished_at`` when stage
rows exist, falling back to ``updated_at`` for seeded or legacy runs that
predate stage tracking. These tests pin that ordering, the retry and legacy
fallbacks, the archived/non-completed exclusions, deterministic ties, the empty
case, and the display-title precedence.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from voxint.api.runs_query import latest_completed_run
from voxint.db.models import (
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    Stage,
    StageRun,
    StageStatus,
)

BASE = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)


def _run(
    session: Session,
    *,
    created_at: datetime,
    status: RunStatus = RunStatus.COMPLETED,
    updated_at: datetime | None = None,
    archived_at: datetime | None = None,
    sidecar: dict[str, object] | None = None,
    source_title: str | None = None,
) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    if source_title is not None:
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title=source_title,
                raw={"id": "x"},
                raw_schema_version=1,
                acquired_at=BASE,
            )
        )
    # ``sidecar`` is JSONB with a ``jsonb_typeof = 'object'`` check: passing
    # Python None would insert JSON ``null`` (not SQL NULL) and violate it, so
    # omit the column entirely when there is no sidecar.
    extra: dict[str, object] = {} if sidecar is None else {"sidecar": sidecar}
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
        created_at=created_at,
        updated_at=updated_at if updated_at is not None else created_at,
        archived_at=archived_at,
        **extra,
    )
    session.add(run)
    session.flush()
    return run.id


def _finalize(
    session: Session,
    run_id: uuid.UUID,
    finished_at: datetime,
    *,
    status: StageStatus = StageStatus.COMPLETED,
    attempt: int = 1,
) -> None:
    session.add(
        StageRun(
            pipeline_run_id=run_id,
            stage=Stage.FINALIZE.value,
            status=status.value,
            attempt=attempt,
            started_at=finished_at - timedelta(seconds=1),
            finished_at=finished_at,
        )
    )


def test_none_when_no_run_has_completed(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        assert latest_completed_run(session) is None


def test_orders_by_finalize_completion_not_submission(
    session_factory: sessionmaker[Session],
) -> None:
    """The pick is by when a run FINISHED, not when it was submitted."""
    with session_factory() as session:
        # A: submitted later but finished earlier. B: submitted earlier, finished later.
        a = _run(session, created_at=BASE + timedelta(hours=2))
        _finalize(session, a, BASE + timedelta(hours=3))
        b = _run(session, created_at=BASE)
        _finalize(session, b, BASE + timedelta(hours=5))
        session.commit()

        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == b
        assert latest.completed_at == BASE + timedelta(hours=5)


def test_retried_finalize_uses_latest_attempt(
    session_factory: sessionmaker[Session],
) -> None:
    """A failed-then-succeeded finalize completes at the later, successful attempt."""
    with session_factory() as session:
        retried = _run(session, created_at=BASE)
        _finalize(
            session, retried, BASE + timedelta(hours=1), status=StageStatus.FAILED, attempt=1
        )
        _finalize(session, retried, BASE + timedelta(hours=4), attempt=2)
        # A rival that finished between the two attempts must not win.
        rival = _run(session, created_at=BASE)
        _finalize(session, rival, BASE + timedelta(hours=3))
        session.commit()

        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == retried
        assert latest.completed_at == BASE + timedelta(hours=4)


def test_legacy_run_without_stage_rows_falls_back_to_updated_at(
    session_factory: sessionmaker[Session],
) -> None:
    """A run with no FINALIZE row is ordered by its updated_at fallback."""
    with session_factory() as session:
        legacy = _run(session, created_at=BASE, updated_at=BASE + timedelta(hours=9))
        tracked = _run(session, created_at=BASE)
        _finalize(session, tracked, BASE + timedelta(hours=6))
        session.commit()

        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == legacy
        assert latest.completed_at == BASE + timedelta(hours=9)


def test_archived_and_non_completed_are_excluded(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        # Archived completed run finished most recently — still excluded.
        archived = _run(session, created_at=BASE, archived_at=BASE + timedelta(hours=1))
        _finalize(session, archived, BASE + timedelta(hours=10))
        # A still-running run is not COMPLETED, regardless of any finalize row.
        running = _run(session, created_at=BASE, status=RunStatus.RUNNING)
        _finalize(session, running, BASE + timedelta(hours=9), status=StageStatus.FAILED)
        # The only eligible run, finished earliest of the three.
        done = _run(session, created_at=BASE)
        _finalize(session, done, BASE + timedelta(hours=2))
        session.commit()

        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == done


def test_ties_break_on_created_at_then_id(
    session_factory: sessionmaker[Session],
) -> None:
    """Equal completion times resolve deterministically to the newer run."""
    with session_factory() as session:
        finish = BASE + timedelta(hours=4)
        older = _run(session, created_at=BASE)
        _finalize(session, older, finish)
        newer = _run(session, created_at=BASE + timedelta(minutes=30))
        _finalize(session, newer, finish)
        session.commit()

        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == newer


def test_carries_display_title_precedence(
    session_factory: sessionmaker[Session],
) -> None:
    """Same title precedence as the queue/listing: sidecar over scraped over none.

    Each sub-case finishes strictly later than the last so it is the newest
    completed run in the shared disposable DB when its title is asserted.
    """
    # Sidecar operator title (issue #104) wins over the scraped metadata title.
    with session_factory() as session:
        run_id = _run(
            session,
            created_at=BASE,
            sidecar={"title": "Operator title"},
            source_title="Scraped title",
        )
        _finalize(session, run_id, BASE + timedelta(hours=1))
        session.commit()
        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == run_id
        assert latest.title == "Operator title"
        assert latest.source_path.startswith("incoming/")

    # No sidecar: the scraped acquisition-metadata title (issue #36) shows.
    with session_factory() as session:
        run_id = _run(session, created_at=BASE, source_title="Scraped title")
        _finalize(session, run_id, BASE + timedelta(hours=2))
        session.commit()
        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == run_id
        assert latest.title == "Scraped title"

    # Neither: title is None and the template falls back to source_path.
    with session_factory() as session:
        run_id = _run(session, created_at=BASE)
        _finalize(session, run_id, BASE + timedelta(hours=3))
        session.commit()
        latest = latest_completed_run(session)
        assert latest is not None
        assert latest.run_id == run_id
        assert latest.title is None
