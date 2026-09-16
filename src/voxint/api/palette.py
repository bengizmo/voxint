"""Command palette data for the shell (issue #162).

Produces the server-driven destination and action lists that the palette
island receives as props. Everything here is pure and cheap (settings reads
only, no DB), matching the :func:`_shell_template_context` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from voxint.config import Settings
    from voxint.db.models import User


@dataclass(frozen=True)
class PaletteCommand:
    """A navigable command surfaced in the palette.

    *label* is the display text, *href* is a GET-navigable URL, *kind*
    discriminates destinations (sidebar entries) from per-page actions.
    *hint* is optional secondary text (e.g. the area name).
    """

    label: str
    href: str
    kind: Literal["destination", "action"]
    hint: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "label": self.label,
            "href": self.href,
            "kind": self.kind,
            "hint": self.hint,
        }


def palette_destinations(
    settings: Settings,
    app_state: Any,
    current_user: User | None,
) -> list[dict[str, str | None]]:
    """Build the static destination list mirroring the sidebar rail.

    Flag logic duplicates :func:`_shell_template_context` so the palette
    never advertises a link the sidebar hides. A unit test asserts parity.
    """
    projects_enabled = settings.console_projects_enabled and getattr(
        app_state, "projects_routed", False
    )
    media_enabled = settings.console_media_enabled and getattr(
        app_state, "media_routed", False
    )
    is_admin = (
        current_user is not None and current_user.role == "admin"
    ) or not settings.voxint_multi_user

    dests: list[PaletteCommand] = [
        PaletteCommand("Home", "/", "destination"),
        PaletteCommand(
            "Media",
            "/media" if media_enabled else "/runs",
            "destination",
        ),
    ]
    if projects_enabled:
        dests.append(PaletteCommand("Projects", "/projects", "destination"))
    dests.append(PaletteCommand("Explore", "/explore", "destination"))
    dests.append(PaletteCommand("Speakers", "/speakers", "destination"))
    dests.append(PaletteCommand("Runs", "/runs", "destination"))
    if is_admin:
        dests.append(PaletteCommand("Settings", "/settings", "destination"))
    if settings.voxint_multi_user and current_user is not None:
        dests.append(
            PaletteCommand("Account", "/account/password", "destination")
        )

    return [d.as_dict() for d in dests]


def palette_actions(
    *commands: PaletteCommand,
) -> list[dict[str, str | None]]:
    """Wrap per-page action commands for the template context."""
    return [c.as_dict() for c in commands]
