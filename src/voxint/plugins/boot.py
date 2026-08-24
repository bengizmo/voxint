"""Boot-time aggregation of plugin invariants (issue #138, epic #136).

The registry's :func:`~voxint.plugins.registry.load_registry` fails loud on a
malformed or colliding *builtin* — a defect in shipped code. :func:`validate_boot`
covers the other half: a plugin's own cross-flag invariants, evaluated against the
env-sourced configuration at process start.

Rule 5 (epic #136) splits invariant handling by source:

* **Env-sourced violations fail boot**, matching today's behavior for core flags
  (``config.validate_effective_flags`` raising at startup). :func:`validate_boot`
  calls each active plugin's ``invariant_errors(row=None, settings)`` — ``row=None``
  is the "env only" reading — and raises a single :class:`PluginError` aggregating
  every message, so a misconfigured baked-in plugin stops the api, the worker, and
  beat rather than serving a broken feature.
* **Row-sourced violations** (an override an upgrade turned invalid) are NOT this
  function's concern: they fail closed to feature-off with a doctor warning at the
  settings boundary, never a boot refusal. So the settings POST path calls
  ``invariant_errors`` with the candidate row directly; only the env reading routes
  through here.

With an empty registry this is a no-op (nothing to iterate), which is the #138
dormant state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voxint.plugins.base import PluginError

if TYPE_CHECKING:
    from voxint.config import Settings
    from voxint.plugins.registry import PluginRegistry


def validate_boot(registry: PluginRegistry, *, settings: Settings) -> None:
    """Fail boot loudly if any active plugin's env-sourced invariants are violated.

    Iterates ``registry.plugins`` in registry order (by ``manifest.id``), calling
    each plugin's ``invariant_errors(None, settings)``. Every returned message is
    collected, prefixed with its plugin id, and raised as one
    :class:`~voxint.plugins.base.PluginError`; an unexpected exception from the hook
    itself is wrapped the same way (a hook that raises is as much a shipped defect
    as a returned violation). No active plugin, or every hook returning ``[]``, is
    a clean boot.
    """
    problems: list[str] = []
    for plugin in registry.plugins:
        plugin_id = plugin.manifest.id
        try:
            messages = plugin.invariant_errors(None, settings)
        except Exception as exc:  # a hook that raises at boot is a defect to surface
            raise PluginError(
                f"plugin {plugin_id!r} invariant check failed: {exc}"
            ) from exc
        problems.extend(f"[{plugin_id}] {message}" for message in messages)
    if problems:
        raise PluginError(
            "plugin configuration invalid at boot:\n" + "\n".join(problems)
        )
