"""The shell context processor and console area flags (Console 2.0 P1, #152)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi import Request

from voxint.api.routers.deps import _shell_template_context, templates
from voxint.config import Settings


def _request_with(settings: Settings) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(settings=settings))
    return cast(Request, SimpleNamespace(app=app))


def test_console_area_flags_default_off() -> None:
    """Unshipped areas stay dark on a stock install."""
    settings = Settings(database_url="postgresql+psycopg://x/x")
    assert settings.console_projects_enabled is False


def test_shell_context_reflects_settings() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://x/x", console_projects_enabled=True
    )
    context = _shell_template_context(_request_with(settings))
    assert context == {"shell": {"projects_enabled": True}}

    settings_off = Settings(database_url="postgresql+psycopg://x/x")
    assert _shell_template_context(_request_with(settings_off)) == {
        "shell": {"projects_enabled": False}
    }


def test_shell_processor_is_registered() -> None:
    """The processor rides every TemplateResponse via the shared environment."""
    assert _shell_template_context in templates.context_processors
