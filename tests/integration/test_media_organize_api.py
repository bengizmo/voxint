"""POST /media/assign and /media/folders — the P2b organize surface (issue #154).

Commit 3 makes the read-only library operable: a multi-select drives a
non-destructive bulk **assign** (set each file's settings folder, including
clearing it), and a folder panel registers/unregisters folders through the shared
write service. These tests pin the wiring the pure helpers cannot see: the CSRF
gate, whole-selection prevalidation with zero writes on any failure, the ADR 0002
no-filesystem-touch invariant (only ``media_folder_id`` moves), the honest
"N files reverted to global settings" count on unregister, and the flag-off 404.

Needs the real Postgres test DB (the advisory lock / FK SET NULL are Postgres
behaviour), so skipped without VOXINT_TEST_DATABASE_URL.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_MEDIA_ASSIGN,
    CSRF_MEDIA_FOLDERS,
    mint_csrf_token,
)
from voxint.api.media_query import MEDIA_LIBRARY_LIMIT
from voxint.config import Settings
from voxint.db.models import MediaFolder, MediaItem

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "media-organize-test-csrf-key"


def _make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    media_enabled: bool = True,
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        console_media_enabled=media_enabled,
        csrf_secret=_CSRF_KEY,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    return _make_client(session_factory, tmp_path)


def _data(csrf_action: str, **fields: object) -> dict[str, object]:
    """Form fields with a valid CSRF token for ``csrf_action`` merged in.

    The keyword is ``csrf_action`` (not ``action``) so a ``folder``-panel form
    field named ``action`` can be passed through ``**fields`` without colliding.
    """
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, csrf_action), **fields}


def _add_media(
    session: Session, *, source_path: str, folder_id: uuid.UUID | None = None
) -> MediaItem:
    media = MediaItem(source_path=source_path, media_folder_id=folder_id, size_bytes=7)
    session.add(media)
    session.flush()
    return media


def _add_folder(session: Session, *, path: str) -> MediaFolder:
    folder = MediaFolder(path=path)
    session.add(folder)
    session.flush()
    return folder


# ---- bulk assign --------------------------------------------------------------


def test_assign_sets_folder_over_selection_without_touching_files(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        a = _add_media(session, source_path="incoming/a.wav")
        b = _add_media(session, source_path="incoming/b.wav")
        folder_id, a_id, b_id = folder.id, a.id, b.id
        before = {
            m.id: (m.source_path, m.current_path, m.size_bytes) for m in (a, b)
        }
        session.commit()

    resp = client.post(
        "/media/assign",
        data=_data(
            CSRF_MEDIA_ASSIGN,
            media_id=[str(a_id), str(b_id)],
            media_folder_id=str(folder_id),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/media?assigned=2")

    with session_factory() as session:
        for mid in (a_id, b_id):
            m = session.get(MediaItem, mid)
            assert m is not None
            assert m.media_folder_id == folder_id
            # AC-3: only the settings-folder pointer moved; identity, live path,
            # and bytes are untouched.
            assert (m.source_path, m.current_path, m.size_bytes) == before[mid]


def test_assign_clear_to_none_reverts_to_global(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        m = _add_media(session, source_path="incoming/a.wav", folder_id=folder.id)
        m_id = m.id
        session.commit()

    resp = client.post(
        "/media/assign",
        data=_data(CSRF_MEDIA_ASSIGN, media_id=[str(m_id)], media_folder_id=""),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    with session_factory() as session:
        assert session.get(MediaItem, m_id).media_folder_id is None  # type: ignore[union-attr]


def test_assign_empty_selection_rejected_zero_writes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        m = _add_media(session, source_path="incoming/a.wav")
        folder_id, m_id = folder.id, m.id
        session.commit()

    resp = client.post(
        "/media/assign",
        data=_data(CSRF_MEDIA_ASSIGN, media_folder_id=str(folder_id)),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "Select at least one file" in resp.text
    with session_factory() as session:
        assert session.get(MediaItem, m_id).media_folder_id is None  # type: ignore[union-attr]


def test_assign_stale_media_id_rejected_zero_writes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        m = _add_media(session, source_path="incoming/a.wav")
        folder_id, m_id = folder.id, m.id
        session.commit()

    # One real id + one that never existed: the whole batch is refused, nothing set.
    resp = client.post(
        "/media/assign",
        data=_data(
            CSRF_MEDIA_ASSIGN,
            media_id=[str(m_id), str(uuid.uuid4())],
            media_folder_id=str(folder_id),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 409
    assert "no longer exist" in resp.text
    with session_factory() as session:
        assert session.get(MediaItem, m_id).media_folder_id is None  # type: ignore[union-attr]


def test_assign_malformed_id_rejected(client: TestClient) -> None:
    resp = client.post(
        "/media/assign",
        data=_data(CSRF_MEDIA_ASSIGN, media_id=["not-a-uuid"], media_folder_id=""),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "not valid" in resp.text


def test_assign_stale_target_folder_rejected_zero_writes(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()

    resp = client.post(
        "/media/assign",
        data=_data(
            CSRF_MEDIA_ASSIGN,
            media_id=[str(m_id)],
            media_folder_id=str(uuid.uuid4()),  # a folder that does not exist
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "settings folder no longer exists" in resp.text
    with session_factory() as session:
        assert session.get(MediaItem, m_id).media_folder_id is None  # type: ignore[union-attr]


def test_assign_over_cap_rejected(client: TestClient) -> None:
    too_many = [str(uuid.uuid4()) for _ in range(MEDIA_LIBRARY_LIMIT + 1)]
    resp = client.post(
        "/media/assign",
        data=_data(CSRF_MEDIA_ASSIGN, media_id=too_many, media_folder_id=""),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert f"at most {MEDIA_LIBRARY_LIMIT}" in resp.text


def test_assign_requires_csrf_before_any_write(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        m = _add_media(session, source_path="incoming/a.wav")
        folder_id, m_id = folder.id, m.id
        session.commit()

    resp = client.post(
        "/media/assign",
        data={
            "csrf_token": "forged",
            "media_id": [str(m_id)],
            "media_folder_id": str(folder_id),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.get(MediaItem, m_id).media_folder_id is None  # type: ignore[union-attr]


def test_assign_error_rerender_preserves_the_selection(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        m = _add_media(session, source_path="incoming/a.wav")
        m_id = m.id
        session.commit()

    # A stale target folder rejects the batch; the re-render keeps the file checked
    # so the operator does not lose their selection.
    resp = client.post(
        "/media/assign",
        data=_data(
            CSRF_MEDIA_ASSIGN,
            media_id=[str(m_id)],
            media_folder_id=str(uuid.uuid4()),
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert f'value="{m_id}"' in resp.text
    # The checkbox for the submitted id is re-checked.
    checkbox = resp.text.split(f'value="{m_id}"', 1)[1].split(">", 1)[0]
    assert "checked" in checkbox


# ---- folder panel: register / unregister -------------------------------------


def test_register_folder(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    (tmp_path / "interviews").mkdir()
    resp = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="add", folder="interviews"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/media?folder=added")
    with session_factory() as session:
        rows = session.execute(select(MediaFolder.path)).scalars().all()
        assert "interviews" in rows


def test_register_missing_folder_rerenders_with_message(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="add", folder="does-not-exist"),
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "existing directory" in resp.text
    with session_factory() as session:
        assert session.execute(select(MediaFolder)).first() is None


def test_register_overlapping_folder_refused(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    ok = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="add", folder="a"),
        follow_redirects=False,
    )
    assert ok.status_code == 303
    nested = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="add", folder="a/b"),
        follow_redirects=False,
    )
    assert nested.status_code == 400
    assert "overlaps" in nested.text
    with session_factory() as session:
        paths = session.execute(select(MediaFolder.path)).scalars().all()
        assert paths == ["a"]


def test_unregister_reverts_assigned_media_and_reports_count(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        folder = _add_folder(session, path="interviews")
        a = _add_media(session, source_path="interviews/a.wav", folder_id=folder.id)
        b = _add_media(session, source_path="interviews/b.wav", folder_id=folder.id)
        folder_id, a_id, b_id = folder.id, a.id, b.id
        session.commit()

    resp = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="remove", folder_id=str(folder_id)),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/media?folder=removed&reverted=2")
    with session_factory() as session:
        assert session.get(MediaFolder, folder_id) is None
        for mid in (a_id, b_id):
            # FK ON DELETE SET NULL reverts each file to global settings.
            assert session.get(MediaItem, mid).media_folder_id is None  # type: ignore[union-attr]


def test_unregister_unknown_id_is_idempotent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    resp = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="remove", folder_id=str(uuid.uuid4())),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/media?folder=removed&reverted=0")


def test_folders_unknown_action_is_422(client: TestClient) -> None:
    resp = client.post(
        "/media/folders",
        data=_data(CSRF_MEDIA_FOLDERS, action="frobnicate", folder="x"),
        follow_redirects=False,
    )
    assert resp.status_code == 422


def test_folders_requires_csrf(
    client: TestClient, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    (tmp_path / "interviews").mkdir()
    resp = client.post(
        "/media/folders",
        data={"csrf_token": "forged", "action": "add", "folder": "interviews"},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    with session_factory() as session:
        assert session.execute(select(MediaFolder)).first() is None


def test_organize_routes_404_when_flag_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    off = _make_client(session_factory, tmp_path, media_enabled=False)
    assert (
        off.post(
            "/media/assign",
            data=_data(CSRF_MEDIA_ASSIGN, media_id=[str(uuid.uuid4())]),
            follow_redirects=False,
        ).status_code
        == 404
    )
    assert (
        off.post(
            "/media/folders",
            data=_data(CSRF_MEDIA_FOLDERS, action="add", folder="x"),
            follow_redirects=False,
        ).status_code
        == 404
    )


def test_folder_panel_renders_registered_folders(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        _add_folder(session, path="interviews")
        _add_media(session, source_path="incoming/a.wav")
        session.commit()
    body = client.get("/media").text
    assert 'action="/media/folders"' in body
    assert 'name="action" value="remove"' in body
    # The bulk-assign form only renders when there is at least one file.
    assert 'action="/media/assign"' in body
    assert "interviews" in body
