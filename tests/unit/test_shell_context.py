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
) -> Request:
    state = SimpleNamespace(settings=settings)
    if projects_routed is not None:
        state.projects_routed = projects_routed
    if jobs_routed is not None:
        state.jobs_routed = jobs_routed
    app = SimpleNamespace(state=state)
    return cast(Request, SimpleNamespace(app=app))


def test_console_area_flags_default_off() -> None:
    """Unshipped areas stay dark on a stock install."""
    settings = Settings(database_url="postgresql+psycopg://x/x")
    assert settings.console_projects_enabled is False
    assert settings.console_jobs_enabled is False


def test_shell_context_requires_flag_and_route() -> None:
    """An area's links render only when its flag is on AND its routes exist —
    an early flag flip must never advertise a dead /projects link (review)."""
    on = Settings(database_url="postgresql+psycopg://x/x", console_projects_enabled=True)
    off = Settings(database_url="postgresql+psycopg://x/x")
    assert _shell_template_context(
        _request_with(on, projects_routed=True, jobs_routed=True)
    ) == {"shell": {"projects_enabled": True, "jobs_enabled": False}}
    # Flag on, no /projects route registered yet (today's reality): stays dark.
    assert _shell_template_context(
        _request_with(on, projects_routed=False, jobs_routed=True)
    ) == {"shell": {"projects_enabled": False, "jobs_enabled": False}}
    # A stale app with no stamp at all fails closed too.
    assert _shell_template_context(_request_with(on)) == {
        "shell": {"projects_enabled": False, "jobs_enabled": False}
    }
    assert _shell_template_context(
        _request_with(off, projects_routed=True, jobs_routed=True)
    ) == {"shell": {"projects_enabled": False, "jobs_enabled": False}}


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


def test_shell_processor_is_registered() -> None:
    """The processor rides every TemplateResponse via the shared environment."""
    assert _shell_template_context in templates.context_processors
