"""Route-building dependencies handed to a plugin's ``build_router`` (issue #137).

A plugin's routes need the same primitives the core routes use — the Jinja
environment, the request-scoped session dependency, the CSRF helpers, and the
settings-page re-render helper — but must not reach into ``api/app.py`` for them
(import direction is law). :class:`PluginRouteDeps` is the frozen, capped bundle
the seam-wiring issue (#138) constructs once and passes to every plugin's
``build_router``. The surface is deliberately small: a new field needs
justification in review, so the plugin boundary cannot quietly grow into a second
copy of the app's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi import Request, Response
    from fastapi.templating import Jinja2Templates
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class PluginRouteDeps:
    """The primitives a plugin router may use, and nothing more.

    * ``templates`` — the shared Jinja environment (already wired with the
      plugin-namespacing ChoiceLoader), so a plugin renders ``<id>/template.html``.
    * ``get_session`` — the request-scoped session dependency (commit-on-success /
      rollback-on-exception), used verbatim as a FastAPI ``Depends`` target.
    * ``verify_csrf`` — the CSRF check a mutating route calls before acting.
    * ``render_settings_page`` — re-render the settings page (a plugin POST that
      updates settings returns the same page the core handler would).
    """

    templates: Jinja2Templates
    get_session: Callable[[Request], Iterator[Session]]
    verify_csrf: Callable[[Request], None]
    render_settings_page: Callable[[Request, Session], Response]
