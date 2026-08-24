"""Voxint plugin framework (issue #137, epic #136).

Optional features (LLM enrichment, semantic search, translation) live here as
self-contained builtin plugins, flag-gated and baked into the app image. This
package's public surface:

* :class:`~voxint.plugins.base.VoxintPlugin` and the frozen contract dataclasses
  a plugin implements (:mod:`voxint.plugins.base`).
* :data:`~voxint.plugins.discover.BUILTIN` — the one sanctioned core -> plugin
  import site.
* :func:`load_plugins` / :func:`get_plugins` — the per-process registry cache the
  api, worker, and CLI share.

The registry is built once per process from ``BUILTIN`` and the
``VOXINT_PLUGINS_DISABLED`` kill switch, then memoized. It is empty until the
conversions land (#139-#141); the framework is dormant with zero call sites
touched in #137.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voxint.plugins.base import (
    FeatureFlag,
    JobLaneSpec,
    PanelContribution,
    PluginError,
    PluginManifest,
    RunCompletedEvent,
    SettingsSection,
    VoxintPlugin,
)
from voxint.plugins.discover import BUILTIN
from voxint.plugins.registry import PluginRegistry, load_registry

if TYPE_CHECKING:
    from voxint.config import Settings

_registry: PluginRegistry | None = None


def parse_disabled_ids(raw: str) -> tuple[str, ...]:
    """Parse the ``VOXINT_PLUGINS_DISABLED`` CSV into a tuple of ids.

    Whitespace-trimmed, empties dropped, order preserved. Case is left as written
    — the kill switch matches manifest ids exactly, and an id that does not match
    is surfaced by the doctor rather than silently coerced.
    """
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_plugins(settings: Settings) -> PluginRegistry:
    """Build the process registry from ``BUILTIN`` and the kill switch, and cache it.

    Called once at api / worker / CLI startup (#138). Fail-loud: a malformed or
    colliding builtin raises :class:`~voxint.plugins.base.PluginError` here, before
    the process serves anything.
    """
    global _registry
    _registry = load_registry(
        BUILTIN, disabled_ids=parse_disabled_ids(settings.voxint_plugins_disabled)
    )
    return _registry


def get_plugins() -> PluginRegistry:
    """The cached registry, building it from the environment on first access.

    Prefer :func:`load_plugins` at a known startup point; this lazy fallback keeps
    an early accessor correct (memoize-on-first-use) rather than ordering-fragile.
    """
    if _registry is None:
        from voxint.config import get_settings

        return load_plugins(get_settings())
    return _registry


def reset_plugins_cache() -> None:
    """Drop the cached registry (test isolation; a fresh process starts empty)."""
    global _registry
    _registry = None


__all__ = [
    "BUILTIN",
    "FeatureFlag",
    "JobLaneSpec",
    "PanelContribution",
    "PluginError",
    "PluginManifest",
    "PluginRegistry",
    "RunCompletedEvent",
    "SettingsSection",
    "VoxintPlugin",
    "get_plugins",
    "load_plugins",
    "load_registry",
    "parse_disabled_ids",
    "reset_plugins_cache",
]
