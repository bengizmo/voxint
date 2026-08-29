"""The shell context processor and console area flags (Console 2.0 P1, #152)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi import Request

from voxint.api.routers.deps import _shell_template_context, templates
from voxint.config import Settings


def _request_with(
    settings: Settings,
    *,
    projects_routed: bool | None = None,
    jobs_routed: bool | None = None,
    media_routed: bool | None = None,
    activity_routed: bool | None = None,
    csrf_secret: str = "test-secret",
) -> Request:
    state = SimpleNamespace(settings=settings, csrf_secret=csrf_secret)
    if projects_routed is not None:
        state.projects_routed = projects_routed
    if jobs_routed is not None:
        state.jobs_routed = jobs_routed
    if media_routed is not None:
        state.media_routed = media_routed
    if activity_routed is not None:
        state.activity_routed = activity_routed
    app = SimpleNamespace(state=state)
    return cast(Request, SimpleNamespace(app=app, state=SimpleNamespace()))


def test_console_area_flags_default_off() -> None:
    """Unshipped areas stay dark on a stock install."""
    settings = Settings(database_url="postgresql+psycopg://x/x")
    assert settings.console_projects_enabled is False
    assert settings.console_jobs_enabled is False
    assert settings.console_media_enabled is False


def test_shell_context_requires_flag_and_route() -> None:
    """An area's links render only when its flag is on AND its routes exist —
    an early flag flip must never advertise a dead /projects link (review)."""
    on = Settings(database_url="postgresql+psycopg://x/x", console_projects_enabled=True)
    off = Settings(database_url="postgresql+psycopg://x/x")
    assert _shell_template_context(
        _request_with(on, projects_routed=True, jobs_routed=True, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": True,
            "jobs_enabled": False,
            "media_enabled": False,
            "activity_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
        }
    }
    # Flag on, no /projects route registered yet (today's reality): stays dark.
    assert _shell_template_context(
        _request_with(on, projects_routed=False, jobs_routed=True, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": False,
            "jobs_enabled": False,
            "media_enabled": False,
            "activity_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
        }
    }
    # A stale app with no stamp at all fails closed too.
    assert _shell_template_context(_request_with(on)) == {
        "shell": {
            "projects_enabled": False,
            "jobs_enabled": False,
            "media_enabled": False,
            "activity_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
        }
    }
    assert _shell_template_context(
        _request_with(off, projects_routed=True, jobs_routed=True, media_routed=True)
    ) == {
        "shell": {
            "projects_enabled": False,
            "jobs_enabled": False,
            "media_enabled": False,
            "activity_enabled": False,
            "multi_user": False,
            "current_user": None,
            "csrf_logout_token": "",
        }
    }


def test_shell_context_jobs_discovery_flag() -> None:
    """Jobs (#160) dark-ships routed-but-undiscovered: the /jobs routes always
    exist (jobs_routed True), so the sidebar entry follows the flag alone."""
    on = Settings(database_url="postgresql+psycopg://x/x", console_jobs_enabled=True)
    off = Settings(database_url="postgresql+psycopg://x/x")
    # Flag on + routes present (the always-true reality for Jobs): discoverable.
    assert (
        _shell_template_context(_request_with(on, jobs_routed=True))["shell"][
            "jobs_enabled"
        ]
        is True
    )
    # Flag off: the sidebar keeps pointing Jobs at the /runs placeholder.
    assert (
        _shell_template_context(_request_with(off, jobs_routed=True))["shell"][
            "jobs_enabled"
        ]
        is False
    )
    # Defensive: a stale app missing the stamp fails closed even with the flag on.
    assert (
        _shell_template_context(_request_with(on))["shell"]["jobs_enabled"] is False
    )


def test_shell_context_media_discovery_flag() -> None:
    """Media (#154) dark-ships routed-but-undiscovered: the /media route always
    exists (media_routed True), so the sidebar Media entry and the "Add media"
    quick action follow the flag alone."""
    on = Settings(database_url="postgresql+psycopg://x/x", console_media_enabled=True)
    off = Settings(database_url="postgresql+psycopg://x/x")
    # Flag on + route present (the always-true reality for Media): discoverable.
    assert (
        _shell_template_context(_request_with(on, media_routed=True))["shell"][
            "media_enabled"
        ]
        is True
    )
    # Flag off: the sidebar keeps pointing Media at the /runs placeholder.
    assert (
        _shell_template_context(_request_with(off, media_routed=True))["shell"][
            "media_enabled"
        ]
        is False
    )
    # Defensive: a stale app missing the stamp fails closed even with the flag on.
    assert (
        _shell_template_context(_request_with(on))["shell"]["media_enabled"] is False
    )


def test_shell_context_activity_requires_jobs_discovery() -> None:
    """Activity (#162) depends on Jobs discovery: the badge lives on the Jobs
    entry and the toast links target /jobs, so activity stays dark unless its own
    flag AND route stamp are set AND Jobs discovery is on."""
    both = Settings(
        database_url="postgresql+psycopg://x/x",
        console_activity_enabled=True,
        console_jobs_enabled=True,
    )
    activity_only = Settings(
        database_url="postgresql+psycopg://x/x", console_activity_enabled=True
    )
    # Activity flag + route + Jobs discovery all present: surfaced.
    assert (
        _shell_template_context(
            _request_with(both, jobs_routed=True, activity_routed=True)
        )["shell"]["activity_enabled"]
        is True
    )
    # Activity on but Jobs discovery off: stays dark (nav-honesty guard).
    assert (
        _shell_template_context(
            _request_with(activity_only, jobs_routed=True, activity_routed=True)
        )["shell"]["activity_enabled"]
        is False
    )
    # Defensive: a stale app missing the activity stamp fails closed.
    assert (
        _shell_template_context(_request_with(both, jobs_routed=True))["shell"][
            "activity_enabled"
        ]
        is False
    )


def test_shell_context_multi_user_csrf_logout_token() -> None:
    """When multi-user is enabled the shell mints a non-empty CSRF logout token."""
    settings = Settings(
        database_url="postgresql+psycopg://x/x", voxint_multi_user=True
    )
    ctx = _shell_template_context(
        _request_with(settings, jobs_routed=True, media_routed=True)
    )
    token = ctx["shell"]["csrf_logout_token"]
    assert isinstance(token, str)
    assert len(token) > 0
    assert token.count(".") == 2


def test_shell_processor_is_registered() -> None:
    """The processor rides every TemplateResponse via the shared environment."""
    assert _shell_template_context in templates.context_processors
