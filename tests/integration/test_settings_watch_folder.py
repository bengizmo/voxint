"""Settings → Automatic ingest toggle + status line (issue #60), end to end.

Covers the ``/settings/watch-folder`` route on the Settings mount against real
Postgres: the tri-state toggle writes the nullable ``app_settings.watch_folder_enabled``
column (On/Off/inherit → True/False/None), an unrecognized value is rejected, CSRF
is enforced, the toggle renders beside the folder panel, and the persisted last-sweep
summary renders as a plain-language status line.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-watch-folder-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session], media_root: Path, **overrides: object
) -> tuple[TestClient, Settings]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    return client, settings


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _flag(session_factory: sessionmaker[Session]) -> bool | None:
    with session_factory() as s:
        row = get_app_settings(s)
        return row.watch_folder_enabled if row else None


def test_toggle_section_renders_beside_folders(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    body = client.get("/settings").text
    assert 'id="watch-folder"' in body
    assert 'action="/settings/watch-folder"' in body
    assert 'name="watch_folder_enabled"' in body


def test_toggle_on_off_inherit_write_the_tristate(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)

    resp = client.post("/settings/watch-folder", data=_form(watch_folder_enabled="on"))
    assert resp.status_code == 200
    assert _flag(session_factory) is True

    client.post("/settings/watch-folder", data=_form(watch_folder_enabled="off"))
    assert _flag(session_factory) is False

    client.post("/settings/watch-folder", data=_form(watch_folder_enabled="inherit"))
    assert _flag(session_factory) is None  # cleared → inherit the installation default


def test_unrecognized_choice_is_rejected_and_writes_nothing(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    client.post("/settings/watch-folder", data=_form(watch_folder_enabled="on"))
    resp = client.post(
        "/settings/watch-folder",
        data=_form(watch_folder_enabled="sometimes"),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Unrecognized watch-folder setting" in resp.text
    assert _flag(session_factory) is True  # the prior value is untouched


def test_toggle_requires_csrf(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/watch-folder", data={"watch_folder_enabled": "on"}, follow_redirects=False
    )
    assert resp.status_code == 403
    assert _flag(session_factory) is None


def test_status_line_renders_persisted_summary(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, watch_folder_enabled=True)
    with session_factory() as s:
        row = get_or_create(s, llm_enabled_default=False)
        row.watch_folder_last_sweep = {
            "picked_up": 3,
            "already_known": 12,
            "settling": 2,
            "deferred": 0,
            "stat_errors": 0,
            "hit_entry_cap": False,
            "hit_file_cap": False,
            "completed_at": datetime(2026, 8, 18, 10, 42, tzinfo=UTC).isoformat(),
        }
        s.commit()

    body = client.get("/settings").text
    assert "picked up 3 new files" in body
    assert "12 already known" in body
    assert "2 waiting to settle" in body
    assert "2026-08-18 10:42 UTC" in body


def _cap_summary(**caps: bool) -> dict[str, object]:
    return {
        "picked_up": 0,
        "already_known": 0,
        "settling": 0,
        "deferred": 0,
        "stat_errors": 0,
        "hit_entry_cap": caps.get("hit_entry_cap", False),
        "hit_file_cap": caps.get("hit_file_cap", False),
        "root_missing": False,
        "completed_at": datetime(2026, 8, 18, 10, 42, tzinfo=UTC).isoformat(),
    }


def test_status_line_file_cap_warning_promises_eventual_pickup(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # hit_file_cap only bounds NET-NEW, so the leftover really is picked up next pass.
    client, _ = make_client(session_factory, media_root, watch_folder_enabled=True)
    with session_factory() as s:
        row = get_or_create(s, llm_enabled_default=False)
        row.watch_folder_last_sweep = _cap_summary(hit_file_cap=True)
        s.commit()
    body = client.get("/settings").text
    assert "only the first batch was queued" in body
    assert "picked up over the next few checks" in body


def test_status_line_entry_cap_warning_is_honest_about_starvation(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # hit_entry_cap spends the walk budget on ALL entries, so the tail can be starved
    # forever — the copy must NOT promise eventual pickup, unlike the file-cap case.
    client, _ = make_client(session_factory, media_root, watch_folder_enabled=True)
    with session_factory() as s:
        row = get_or_create(s, llm_enabled_default=False)
        row.watch_folder_last_sweep = _cap_summary(hit_entry_cap=True)
        s.commit()
    body = client.get("/settings").text
    assert "may not be picked up" in body
    assert "picked up over the next few checks" not in body


def test_status_line_root_missing_warning(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, watch_folder_enabled=True)
    with session_factory() as s:
        row = get_or_create(s, llm_enabled_default=False)
        summary = _cap_summary()
        summary["root_missing"] = True
        row.watch_folder_last_sweep = summary
        s.commit()
    body = client.get("/settings").text
    assert "media folder could not be found" in body
