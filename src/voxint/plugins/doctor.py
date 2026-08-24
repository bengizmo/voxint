"""``voxint doctor`` surface for the plugin framework (issue #137).

A pure formatter so the CLI stays thin and the output is unit-testable without
running the full diagnostics. Returns ``None`` when there is nothing to say — no
active plugins, no kill switch — so the doctor output is byte-identical to before
the framework existed while the registry is empty (#137 is dormant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxint.plugins.registry import PluginRegistry


def format_plugins_status(registry: PluginRegistry) -> str | None:
    """A short plugin-status block for ``voxint doctor``, or ``None`` if empty.

    Lists the active (registered) plugins, any ids taken out by the kill switch,
    and — as a warning — kill-switch ids that match no builtin (a likely typo
    that would otherwise silently do nothing).
    """
    lines: list[str] = []
    if registry.plugins:
        lines.append(
            "plugins active: " + ", ".join(p.manifest.id for p in registry.plugins)
        )
    known_disabled = sorted(registry.disabled_ids - registry.unknown_disabled_ids)
    if known_disabled:
        lines.append("plugins disabled (kill switch): " + ", ".join(known_disabled))
    if registry.unknown_disabled_ids:
        lines.append(
            "[warn] VOXINT_PLUGINS_DISABLED names unknown plugin ids: "
            + ", ".join(sorted(registry.unknown_disabled_ids))
        )
    return "\n".join(lines) if lines else None
