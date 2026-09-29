"""The shell context processor and console area flags (Console 2.0 P1, #152)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from voxint.api.routers.deps import _shell_template_context, templates
from voxint.config import Settings


def _request_with(
    settings: Settings,
    *,
    projects_routed: bool | None = None,
    media_routed: bool | None = None,
    activity_routed: bool | None = None,
    csrf_secret: str = "test-secret",
) -> Request:
    state = SimpleNamespace(settings=settings, csrf_secret=csrf_secret)
    if projects_routed is not None:
        state.projects_routed = projects_routed
    if media_routed is not None:
        state.media_routed = media_routed
    if activity_routed is not None:
        state.activity_routed = activity_routed
    app = SimpleNamespace(state=state)
    return cast(Request, SimpleNamespace(app=app, state=SimpleNamespace()))


@pytest.fixture(autouse=True)
def _no_console_flags_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests reason about flag defaults, so no CONSOLE_* value may leak in
    from the process environment and mask a default regressing (#646)."""
    for name in list(os.environ):
        if name.upper().startswith("CONSOLE_"):
            monkeypatch.delenv(name)


def test_console_area_flags_shipped_default_on() -> None:
    """Shipped areas are live on a stock install. Media is among them (#646):
    its editor is the only review surface, so a dark media area made review
    unreachable on every default install."""
    settings = Settings(database_url="postgresql+psycopg://x/x", _env_file=None)
    assert settings.console_media_enabled is True
    assert settings.console_speakers_enabled is True
    assert settings.console_settings_enabled is True
    assert settings.console_palette_enabled is True


def test_console_area_flags_unshipped_default_off() -> None:
    """Unshipped areas stay dark on a stock install."""
    settings = Settings(database_url="postgresql+psycopg://x/x", _env_file=None)
    assert settings.console_projects_enabled is False
    assert settings.console_users_enabled is False
    assert settings.console_activity_enabled is False


def test_shell_context_requires_flag_and_route() -> None:
    """An area's links render only when its flag is on AND its routes exist —
    an early flag flip must never advertise a dead /projects link (review)."""
    on = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://x/x",
        console_projects_enabled=True,
        console_palette_enabled=False,
    )
    off = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://x/x",
        console_palette_enabled=False,
    )
    assert _shell_template_context(
        _request_with(on, projects_routed=True, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": True,
            "media_enabled": True,
            "activity_enabled": False,
            "users_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
            "can_write": False,
            "palette_enabled": False,
            "palette_destinations": [],
        }
    }
    # Flag on, no /projects route registered yet (today's reality): stays dark.
    assert _shell_template_context(
        _request_with(on, projects_routed=False, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": False,
            "media_enabled": True,
            "activity_enabled": False,
            "users_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
            "can_write": False,
            "palette_enabled": False,
            "palette_destinations": [],
        }
    }
    # A stale app with no stamp at all fails closed too.
    assert _shell_template_context(_request_with(on)) == {
        "shell": {
            "projects_enabled": False,
            "media_enabled": False,
            "activity_enabled": False,
            "users_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
            "can_write": False,
            "palette_enabled": False,
            "palette_destinations": [],
        }
    }
    assert _shell_template_context(
        _request_with(off, projects_routed=True, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": False,
            "media_enabled": True,
            "activity_enabled": False,
            "users_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
            "can_write": False,
            "palette_enabled": False,
            "palette_destinations": [],
        }
    }


def test_shell_context_media_discovery_flag() -> None:
    """Media (#154) dark-ships routed-but-undiscovered: the /media route always
    exists (media_routed True), so the sidebar Media entry and the "Add media"
    quick action follow the flag alone."""
    on = Settings(
        database_url="postgresql+psycopg://x/x", console_media_enabled=True, _env_file=None
    )
    off = Settings(
        database_url="postgresql+psycopg://x/x", console_media_enabled=False, _env_file=None
    )
    assert (
        _shell_template_context(_request_with(on, media_routed=True))["shell"][
            "media_enabled"
        ]
        is True
    )
    assert (
        _shell_template_context(_request_with(off, media_routed=True))["shell"][
            "media_enabled"
        ]
        is False
    )
    assert (
        _shell_template_context(_request_with(on))["shell"]["media_enabled"] is False
    )


def test_shell_context_activity_flag() -> None:
    """Activity (#162) depends on its own flag and route stamp."""
    on = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://x/x",
        console_activity_enabled=True,
    )
    off = Settings(database_url="postgresql+psycopg://x/x", _env_file=None)
    # Flag on + route present: surfaced.
    assert (
        _shell_template_context(
            _request_with(on, activity_routed=True)
        )["shell"]["activity_enabled"]
        is True
    )
    # Flag off: stays dark.
    assert (
        _shell_template_context(
            _request_with(off, activity_routed=True)
        )["shell"]["activity_enabled"]
        is False
    )
    # Defensive: a stale app missing the activity stamp fails closed.
    assert (
        _shell_template_context(_request_with(on))["shell"][
            "activity_enabled"
        ]
        is False
    )


def test_shell_context_multi_user_csrf_logout_token() -> None:
    """When multi-user is enabled the shell mints a non-empty CSRF logout token."""
    settings = Settings(
        database_url="postgresql+psycopg://x/x", voxint_multi_user=True, _env_file=None
    )
    ctx = _shell_template_context(
        _request_with(settings, media_routed=True)
    )
    token = ctx["shell"]["csrf_logout_token"]
    assert isinstance(token, str)
    assert len(token) > 0
    assert token.count(".") == 2


def test_shell_processor_is_registered() -> None:
    """The processor rides every TemplateResponse via the shared environment."""
    assert _shell_template_context in templates.context_processors
