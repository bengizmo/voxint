"""The shell context processor and console area flags (Console 2.0 P1, #152)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi import Request

from voxint.api.routers.deps import _shell_template_context, templates
from voxint.config import Settings


def _request_with(settings: Settings, *, projects_routed: bool | None = None) -> Request:
    state = SimpleNamespace(settings=settings)
    if projects_routed is not None:
        state.projects_routed = projects_routed
    app = SimpleNamespace(state=state)
    return cast(Request, SimpleNamespace(app=app))


def test_console_area_flags_default_off() -> None:
    """Unshipped areas stay dark on a stock install."""
    settings = Settings(database_url="postgresql+psycopg://x/x")
    assert settings.console_projects_enabled is False


def test_shell_context_requires_flag_and_route() -> None:
    """An area's links render only when its flag is on AND its routes exist —
    an early flag flip must never advertise a dead /projects link (review)."""
    on = Settings(database_url="postgresql+psycopg://x/x", console_projects_enabled=True)
    off = Settings(database_url="postgresql+psycopg://x/x")
    assert _shell_template_context(_request_with(on, projects_routed=True)) == {
        "shell": {"projects_enabled": True}
    }
    # Flag on, no /projects route registered yet (today's reality): stays dark.
    assert _shell_template_context(_request_with(on, projects_routed=False)) == {
        "shell": {"projects_enabled": False}
    }
    # A stale app with no stamp at all fails closed too.
    assert _shell_template_context(_request_with(on)) == {
        "shell": {"projects_enabled": False}
    }
    assert _shell_template_context(_request_with(off, projects_routed=True)) == {
        "shell": {"projects_enabled": False}
    }


def test_shell_processor_is_registered() -> None:
    """The processor rides every TemplateResponse via the shared environment."""
    assert _shell_template_context in templates.context_processors
