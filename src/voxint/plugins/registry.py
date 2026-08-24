"""Load, validate, and order the builtin plugins (issue #137).

:func:`load_registry` turns :data:`voxint.plugins.discover.BUILTIN` into a frozen
:class:`PluginRegistry`. Loading is **fail-loud**: a builtin that cannot be
instantiated, that lacks a valid manifest, or that collides with another on its
id or a Celery task name aborts startup with a :class:`~voxint.plugins.base.PluginError`.
A baked-in plugin is shipped code, so a broken one is a defect to surface, not to
quarantine; the ``VOXINT_PLUGINS_DISABLED`` kill switch is the operator's recovery
lever when a specific plugin must be taken out of the picture (epic #136).

Validation runs over ALL builtins before the kill switch filters any out, so a
duplicate id is caught even when one of the pair is disabled. The active set is
then ordered by ``manifest.id`` for a deterministic route / task / section order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from voxint.plugins.base import (
    JobLaneSpec,
    PanelContribution,
    PluginError,
    PluginManifest,
    SettingsSection,
    VoxintPlugin,
)
from voxint.plugins.discover import BUILTIN


@dataclass(frozen=True)
class PluginRegistry:
    """The active plugins plus the kill-switch bookkeeping the doctor surfaces.

    ``plugins`` is kill-switch-filtered and ordered by ``manifest.id``.
    ``disabled_ids`` is the kill switch verbatim; ``unknown_disabled_ids`` is the
    subset naming no builtin, which the doctor warns about (a typo in the kill
    switch would otherwise silently do nothing).
    """

    plugins: tuple[VoxintPlugin, ...]
    disabled_ids: frozenset[str]
    unknown_disabled_ids: frozenset[str]

    def get(self, plugin_id: str) -> VoxintPlugin | None:
        """The active plugin with this id, or ``None`` (missing or killed)."""
        for plugin in self.plugins:
            if plugin.manifest.id == plugin_id:
                return plugin
        return None

    def task_names(self) -> tuple[str, ...]:
        """Every active plugin's Celery task names, in registry order."""
        names: list[str] = []
        for plugin in self.plugins:
            names.extend(plugin.manifest.task_names)
        return tuple(names)

    def task_routes(self) -> dict[str, Mapping[str, str]]:
        """Merged ``task_routes`` for every active plugin (post queue only)."""
        routes: dict[str, Mapping[str, str]] = {}
        for plugin in self.plugins:
            for name, route in plugin.task_routes().items():
                routes[name] = route
        return routes

    def job_lanes(self) -> tuple[JobLaneSpec, ...]:
        """Every active plugin's job lanes, for the core recovery sweep."""
        lanes: list[JobLaneSpec] = []
        for plugin in self.plugins:
            lanes.extend(plugin.job_lanes())
        return tuple(lanes)

    def settings_sections(self) -> tuple[SettingsSection, ...]:
        """Active plugins' settings sections, sorted by ``(order, section_id)``."""
        sections = [
            section
            for plugin in self.plugins
            if (section := plugin.settings_section()) is not None
        ]
        return tuple(sorted(sections, key=lambda s: (s.order, s.section_id)))

    def run_detail_panels(self) -> tuple[PanelContribution, ...]:
        """Active plugins' run-detail panels, sorted by ``(slot, order)``."""
        panels: list[PanelContribution] = []
        for plugin in self.plugins:
            panels.extend(plugin.run_detail_panels())
        return tuple(sorted(panels, key=lambda p: (p.slot, p.order)))


def find_route_collisions(
    routes_by_plugin: Mapping[str, Sequence[tuple[str, str]]],
) -> list[str]:
    """Report ``(path, method)`` pairs claimed by more than one plugin.

    Route mounting happens in the seam-wiring issue (#138), which builds each
    plugin's router and calls this before mounting — routes need
    :class:`~voxint.plugins.deps.PluginRouteDeps`, which the registry does not
    hold. The check lives here so all collision logic sits beside the id and
    task-name checks; the app-level route-inventory contract test is the backstop
    that additionally catches a plugin colliding with a *core* route. Methods are
    upper-cased so ``get`` and ``GET`` collide.
    """
    seen: dict[tuple[str, str], str] = {}
    collisions: list[str] = []
    for plugin_id, routes in routes_by_plugin.items():
        for path, method in routes:
            key = (path, method.upper())
            if key in seen:
                collisions.append(
                    f"{method.upper()} {path}: claimed by both {seen[key]} and {plugin_id}"
                )
            else:
                seen[key] = plugin_id
    return collisions


def _manifest_of(cls: type[VoxintPlugin]) -> PluginManifest:
    manifest = getattr(cls, "manifest", None)
    if not isinstance(manifest, PluginManifest):
        raise PluginError(
            f"plugin class {cls.__module__}.{cls.__qualname__} has no valid "
            "`manifest` (expected a PluginManifest class attribute)"
        )
    return manifest


def _instantiate(cls: type[VoxintPlugin]) -> VoxintPlugin:
    try:
        instance = cls()
    except Exception as exc:  # a baked plugin that cannot construct is a defect
        raise PluginError(
            f"plugin class {cls.__module__}.{cls.__qualname__} failed to "
            f"instantiate: {exc}"
        ) from exc
    if not isinstance(instance, VoxintPlugin):
        raise PluginError(
            f"{cls.__module__}.{cls.__qualname__} is not a VoxintPlugin subclass"
        )
    return instance


def load_registry(
    builtin: Sequence[type[VoxintPlugin]] = BUILTIN,
    *,
    disabled_ids: Iterable[str] = (),
) -> PluginRegistry:
    """Build the registry from ``builtin``, honoring the kill switch.

    Fail-loud on any malformed or colliding builtin. ``disabled_ids`` (the parsed
    ``VOXINT_PLUGINS_DISABLED`` value) removes a plugin from the active set; an id
    that names no builtin is recorded in ``unknown_disabled_ids`` for the doctor
    to warn about, never an error (an operator taking a not-yet-installed plugin
    out of the picture must not break boot).
    """
    disabled = frozenset(disabled_ids)

    instances: list[VoxintPlugin] = []
    seen_ids: dict[str, str] = {}
    seen_tasks: dict[str, str] = {}
    for cls in builtin:
        manifest = _manifest_of(cls)
        if manifest.id in seen_ids:
            raise PluginError(
                f"duplicate plugin id {manifest.id!r}: {seen_ids[manifest.id]} and "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        seen_ids[manifest.id] = f"{cls.__module__}.{cls.__qualname__}"
        for task_name in manifest.task_names:
            if task_name in seen_tasks:
                raise PluginError(
                    f"duplicate Celery task name {task_name!r}: claimed by both "
                    f"{seen_tasks[task_name]} and plugin {manifest.id!r}"
                )
            seen_tasks[task_name] = f"plugin {manifest.id!r}"
        instances.append(_instantiate(cls))

    active = tuple(
        sorted(
            (p for p in instances if p.manifest.id not in disabled),
            key=lambda p: p.manifest.id,
        )
    )
    _validate_active_contributions(active)
    unknown = frozenset(disabled - set(seen_ids))
    return PluginRegistry(
        plugins=active,
        disabled_ids=disabled,
        unknown_disabled_ids=unknown,
    )


def _validate_active_contributions(active: Sequence[VoxintPlugin]) -> None:
    """Fail loud on active-plugin aggregation collisions (routes, sections).

    Task-name and id collisions are caught at the manifest level over ALL
    builtins; these two checks need the instance hooks, so they run over the
    active set (a killed plugin contributes nothing):

    * ``task_routes`` may only route a task the plugin declares in its manifest.
      Because manifest task names are already globally unique, keeping route keys
      inside the manifest also makes a cross-plugin route overwrite impossible.
    * ``settings_section().section_id`` is the stable settings anchor and dedupe
      key; two active plugins sharing one would both render, so a collision is a
      defect to surface at load, not at render.
    """
    seen_sections: dict[str, str] = {}
    for plugin in active:
        manifest = plugin.manifest
        declared = set(manifest.task_names)
        for name in plugin.task_routes():
            if name not in declared:
                raise PluginError(
                    f"plugin {manifest.id!r} routes task {name!r} it does not "
                    "declare in its manifest task_names"
                )
        section = plugin.settings_section()
        if section is not None:
            if section.section_id in seen_sections:
                raise PluginError(
                    f"duplicate settings section_id {section.section_id!r}: "
                    f"claimed by both {seen_sections[section.section_id]} and "
                    f"plugin {manifest.id!r}"
                )
            seen_sections[section.section_id] = f"plugin {manifest.id!r}"
