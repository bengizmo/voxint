"""POST /media/archive and /media/unarchive — P2b bulk archive (issue #154).

Commit 5 adds the archived view (``/media?archived=1``) and two non-destructive
bulk routes that soft-archive or restore each selected file's latest run *in the
current view*: archive acts on the latest NON-archived run (the active view, where
its button lives), unarchive on the latest ARCHIVED run (the archived view). These
tests pin the wiring the per-run helpers cannot see: the CSRF gate, whole-selection
prevalidation, the skip-not-abort posture (a file with no run in the view — or, for
archive, a live latest run — is reported skipped, never failed), the archived-view
toggle's visibility, the archive->unarchive round-trip leaving the file's bytes and
path untouched (AC-3), and the flag-off 404.

Needs the real Postgres test DB (the archived-view window is Postgres behaviour), so
skipped without VOXINT_TEST_DATABASE_URL.
"""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.csrf import (
    CSRF_MEDIA_ARCHIVE,
    CSRF_MEDIA_UNARCHIVE,
    mint_csrf_token,
)
from voxint.api.media_query import MEDIA_LIBRARY_LIMIT
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "media-archive-test-csrf-key"

# A fixed base instant so every run's created_at is explicit and the "latest run"
# window is deterministic within one transaction (server-side now() would tie).
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _settings(tmp_path: Path, *, media_enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path / "media",
        console_media_enabled=media_enabled,
        csrf_secret=_CSRF_KEY,
    )


def _make_client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    from voxint.api.app import create_app

    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    s = _settings(tmp_path)
    s.media_root.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], settings: Settings
) -> TestClient:
    return _make_client(session_factory, settings)


def _data(csrf_action: str, **fields: object) -> dict[str, object]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, csrf_action), **fields}


def _add_media(session: Session, *, source_path: str) -> MediaItem:
    media = MediaItem(source_path=source_path, size_bytes=7)
    session.add(media)
    session.flush()
    return media


def _add_run(
    session: Session,
    media_id: uuid.UUID,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    archived: bool = False,
    order: int = 0,
) -> PipelineRun:
    """A run with an explicit created_at (``_T0 + order``) so recency is
    deterministic, and an optional archive stamp."""
    run = PipelineRun(
        media_item_id=media_id,
        status=status.value,
        created_at=_T0 + timedelta(minutes=order),
        archived_at=(_T0 if archived else None),
    )
    session.add(run)
    session.flush()
    return run


def _run(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> PipelineRun:
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        session.expunge(run)
        return run


def _baselines(
    session_factory: sessionmaker[Session],
    media_ids: list[uuid.UUID],
    *,
    archived: bool,
) -> list[str]:
    """The render-time ``media_id:run_id`` baselines the /media page emits for these
    files in the given view (latest run in view, or ``:none``) — the hidden
    ``run_baseline`` fields the archive/unarchive routes now require."""
    from voxint.api.routers.media import _latest_run_in_view

    with session_factory() as session:
        latest = _latest_run_in_view(session, media_ids, archived=archived)
    return [f"{mid}:{latest.get(mid) or 'none'}" for mid in media_ids]


# ---- archive: happy + latest-non-archived ------------------------------------


def test_archive_stamps_latest_terminal_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.COMPLETED)
        m_id, run_id = m.id, run.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=False),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "archive_done=1" in resp.headers["location"]
    assert "skipped" not in resp.headers["location"]
    assert _run(session_factory, run_id).archived_at is not None


def test_archive_acts_on_latest_non_archived_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """With an older archived run and a newer live-completed one, archive stamps the
    NEWER (latest non-archived) run — the one the active view shows."""
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        old = _add_run(session, m.id, archived=True, order=0)
        new = _add_run(session, m.id, status=RunStatus.COMPLETED, order=5)
        m_id, old_id, new_id = m.id, old.id, new.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=False),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "archive_done=1" in resp.headers["location"]
    # The already-archived older run keeps its original stamp; the newer run is now
    # archived too.
    assert _run(session_factory, new_id).archived_at is not None
    old_run = _run(session_factory, old_id)
    assert old_run.archived_at == _T0  # untouched (idempotent, unchanged)


# ---- archive: skip-not-abort -------------------------------------------------


def test_archive_skips_file_with_no_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")  # no run at all
        m_id = m.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=False),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # Reported as a skip, never a failure or a silent drop.
    assert "archive_done=0" in resp.headers["location"]
    assert "skipped=1" in resp.headers["location"]


def test_archive_skips_live_latest_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A live (QUEUED) latest run is not archivable — skipped, not failed, and its
    archived_at stays NULL."""
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.QUEUED)
        m_id, run_id = m.id, run.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=False),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "archive_done=0" in resp.headers["location"]
    # A live run is counted apart from a plain skip so the banner can say "cancel
    # it first" rather than "no run to archive".
    assert "live_skipped=1" in resp.headers["location"]
    assert _run(session_factory, run_id).archived_at is None


def test_archive_mixed_selection_reports_both_counts(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        ok = _add_media(session, source_path="incoming/ok.wav")
        _add_run(session, ok.id, status=RunStatus.COMPLETED)
        empty = _add_media(session, source_path="incoming/empty.wav")
        ok_id, empty_id = ok.id, empty.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(ok_id), str(empty_id)],
            run_baseline=_baselines(
                session_factory, [ok_id, empty_id], archived=False
            ),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "archive_done=1" in resp.headers["location"]
    assert "skipped=1" in resp.headers["location"]


# ---- idempotent replay + required baseline (issue #154 review) ---------------


def test_archive_double_submit_archives_only_the_previewed_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A replay of the SAME archive POST must not slide onto the next-older run.

    Two non-archived runs: the render-time baseline pins the newer one. The first
    submit archives it; the replay carries the now-stale baseline (the latest in
    view has moved), so it drifts and skips — the older run stays untouched.
    """
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        old = _add_run(session, m.id, status=RunStatus.COMPLETED, order=0)
        new = _add_run(session, m.id, status=RunStatus.COMPLETED, order=5)
        m_id, old_id, new_id = m.id, old.id, new.id
        session.commit()

    body = _data(
        CSRF_MEDIA_ARCHIVE,
        media_id=[str(m_id)],
        run_baseline=_baselines(session_factory, [m_id], archived=False),
    )
    first = client.post("/media/archive", data=body, follow_redirects=False)
    assert first.status_code == 303
    assert "archive_done=1" in first.headers["location"]

    # Replay the identical body — baseline now stale (new is archived): skip.
    second = client.post("/media/archive", data=body, follow_redirects=False)
    assert second.status_code == 303
    assert "archive_done=0" in second.headers["location"]
    assert "skipped=1" in second.headers["location"]

    # Only the previewed (newer) run was archived; the older run is left alone.
    assert _run(session_factory, new_id).archived_at is not None
    assert _run(session_factory, old_id).archived_at is None


def test_archive_selection_without_baseline_is_rejected(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A checked file with no carried run_baseline is a stale/forged form: reject
    the whole request with zero writes (the confirm route's discipline)."""
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.COMPLETED)
        m_id, run_id = m.id, run.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(CSRF_MEDIA_ARCHIVE, media_id=[str(m_id)]),  # no run_baseline
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert _run(session_factory, run_id).archived_at is None


# ---- unarchive: happy --------------------------------------------------------


def test_unarchive_clears_latest_archived_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.COMPLETED, archived=True)
        m_id, run_id = m.id, run.id
        session.commit()

    resp = client.post(
        "/media/unarchive",
        data=_data(
            CSRF_MEDIA_UNARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=True),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # Stays in the archived view so the operator watches it leave.
    assert "archived=1" in resp.headers["location"]
    assert "unarchive_done=1" in resp.headers["location"]
    assert _run(session_factory, run_id).archived_at is None


def test_unarchive_skips_file_with_no_archived_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        _add_run(session, m.id, status=RunStatus.COMPLETED)  # non-archived only
        m_id = m.id
        session.commit()

    resp = client.post(
        "/media/unarchive",
        data=_data(
            CSRF_MEDIA_UNARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=True),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "unarchive_done=0" in resp.headers["location"]
    assert "skipped=1" in resp.headers["location"]


# ---- archived view visibility ------------------------------------------------


def test_archived_view_shows_only_archived_items(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        active = _add_media(session, source_path="incoming/active.wav")
        _add_run(session, active.id, status=RunStatus.COMPLETED)
        archived = _add_media(session, source_path="incoming/archived.wav")
        _add_run(session, archived.id, status=RunStatus.COMPLETED, archived=True)
        session.commit()

    archived_view = client.get("/media?archived=1")
    assert archived_view.status_code == 200
    assert "archived.wav" in archived_view.text
    # The active-only file has no archived run, so the inner join drops it here.
    assert "active.wav" not in archived_view.text
    # The archived view is a restore surface: no upload form, no folder panel.
    assert 'action="/media/submit"' not in archived_view.text
    assert "/media/unarchive" in archived_view.text

    active_view = client.get("/media")
    assert active_view.status_code == 200
    # The archived-only file still lists (media items are never hidden) but its
    # archived run is hidden, so it reads "Not processed yet".
    assert "active.wav" in active_view.text
    assert "/media/archive" in active_view.text
    # The active view offers the archived-view toggle.
    assert "archived=1" in active_view.text


# ---- round-trip (AC-3: no filesystem / path / bytes change) ------------------


def test_archive_then_unarchive_round_trip_preserves_media(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.COMPLETED)
        m_id, run_id = m.id, run.id
        before = (m.source_path, m.current_path, m.size_bytes)
        session.commit()

    archive = client.post(
        "/media/archive",
        data=_data(
            CSRF_MEDIA_ARCHIVE,
            media_id=[str(m_id)],
            run_baseline=_baselines(session_factory, [m_id], archived=False),
        ),
        follow_redirects=False,
    )
    assert archive.status_code == 303
    assert _run(session_factory, run_id).archived_at is not None

    unarchive = client.post(
        "/media/unarchive",
        data=_data(
            CSRF_MEDIA_UNARCHIVE,
            media_id=[str(m_id)],
            # The run is archived now, so its baseline is taken in the archived view.
            run_baseline=_baselines(session_factory, [m_id], archived=True),
        ),
        follow_redirects=False,
    )
    assert unarchive.status_code == 303
    assert _run(session_factory, run_id).archived_at is None

    with session_factory() as session:
        after_media = session.get(MediaItem, m_id)
        assert after_media is not None
        assert (
            after_media.source_path,
            after_media.current_path,
            after_media.size_bytes,
        ) == before
        # No extra runs were minted by the archive round-trip.
        rows = session.execute(
            select(PipelineRun).where(PipelineRun.media_item_id == m_id)
        ).scalars().all()
        assert len(rows) == 1


# ---- prevalidation, CSRF, count-match, flag-off ------------------------------


@pytest.mark.parametrize("route", ["/media/archive", "/media/unarchive"])
@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, 400),  # empty selection
        ({"media_id": ["not-a-uuid"]}, 400),  # malformed id
    ],
)
def test_archive_prevalidation(
    client: TestClient, route: str, payload: dict[str, object], expected: int
) -> None:
    action = CSRF_MEDIA_UNARCHIVE if route.endswith("unarchive") else CSRF_MEDIA_ARCHIVE
    resp = client.post(route, data=_data(action, **payload), follow_redirects=False)
    assert resp.status_code == expected


def test_archive_over_cap_rejected(client: TestClient) -> None:
    too_many = [str(uuid.uuid4()) for _ in range(MEDIA_LIBRARY_LIMIT + 1)]
    resp = client.post(
        "/media/archive",
        data=_data(CSRF_MEDIA_ARCHIVE, media_id=too_many),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert f"at most {MEDIA_LIBRARY_LIMIT}" in resp.text


def test_archive_count_mismatch_rejects_with_zero_writes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """A selected id that no longer exists is a 409 with nothing changed."""
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        run = _add_run(session, m.id, status=RunStatus.COMPLETED)
        m_id, run_id = m.id, run.id
        session.commit()

    resp = client.post(
        "/media/archive",
        data=_data(CSRF_MEDIA_ARCHIVE, media_id=[str(m_id), str(uuid.uuid4())]),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    assert _run(session_factory, run_id).archived_at is None  # untouched


@pytest.mark.parametrize("route", ["/media/archive", "/media/unarchive"])
def test_archive_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session], route: str
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()
    resp = client.post(
        route,
        data={"csrf_token": "forged", "media_id": [str(m_id)]},
        follow_redirects=False,
    )
    assert resp.status_code == 403


@pytest.mark.parametrize("route", ["/media/archive", "/media/unarchive"])
def test_archive_flag_off_404(
    session_factory: sessionmaker[Session], tmp_path: Path, route: str
) -> None:
    settings = _settings(tmp_path, media_enabled=False)
    settings.media_root.mkdir(parents=True, exist_ok=True)
    client = _make_client(session_factory, settings)
    action = CSRF_MEDIA_UNARCHIVE if route.endswith("unarchive") else CSRF_MEDIA_ARCHIVE
    resp = client.post(
        route,
        data=_data(action, media_id=[str(uuid.uuid4())]),
        follow_redirects=False,
    )
    assert resp.status_code == 404
