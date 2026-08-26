"""Settings-section ↔ plugin-registry parity (Console 2.0 P6, issue #161).

The plugins sub-page and its per-plugin detail pages are driven entirely by the
plugin registry, so a plugin cannot ship a settings section that no page renders,
and the detail route cannot resolve an id that is not an active plugin. These are
pure-function contracts over the registry read models (no DB, no HTTP), exercised
with synthetic fixture plugins because no real plugin ships yet (``BUILTIN`` is
empty). When the first real plugin lands (#143), it inherits these guarantees and
adds an integration assertion that its section renders on ``/settings/plugins``.
"""

from __future__ import annotations

from voxint.api.settings_view import build_plugin_detail, build_plugins_view
from voxint.config import Settings
from voxint.plugins.base import PluginManifest, SettingsSection, VoxintPlugin
from voxint.plugins.registry import load_registry

_SETTINGS = Settings(_env_file=None)  # type: ignore[call-arg]


class _SectionPlugin(VoxintPlugin):
    manifest = PluginManifest(id="alpha", name="Alpha", description="a")

    def enabled(self, row: object, settings: object) -> bool:
        return True

    def settings_section(self) -> SettingsSection:
        return SettingsSection(
            section_id="alpha", title="Alpha", template="alpha/section.html"
        )


class _NoSectionPlugin(VoxintPlugin):
    manifest = PluginManifest(id="beta", name="Beta", description="b")


def test_empty_registry_lists_nothing() -> None:
    view = build_plugins_view(load_registry(builtin=()), None, _SETTINGS)
    assert view.active == ()
    assert view.killed_ids == ()
    assert view.unknown_disabled_ids == ()


def test_every_active_plugin_has_exactly_one_list_row() -> None:
    registry = load_registry(builtin=(_SectionPlugin, _NoSectionPlugin))
    view = build_plugins_view(registry, None, _SETTINGS)
    listed = {row.plugin_id for row in view.active}
    assert listed == {p.manifest.id for p in registry.plugins}
    assert len(view.active) == len(registry.plugins)


def test_detail_resolves_every_active_id_and_only_those() -> None:
    registry = load_registry(builtin=(_SectionPlugin, _NoSectionPlugin))
    for plugin in registry.plugins:
        assert build_plugin_detail(registry, plugin.manifest.id, None, _SETTINGS) is not None
    # An id that names no active plugin is a 404 signal (None).
    assert build_plugin_detail(registry, "missing", None, _SETTINGS) is None


def test_section_contributed_is_the_section_rendered_on_detail() -> None:
    registry = load_registry(builtin=(_SectionPlugin,))
    detail = build_plugin_detail(registry, "alpha", None, _SETTINGS)
    assert detail is not None
    contributed = registry.get("alpha").settings_section()  # type: ignore[union-attr]
    assert detail.section == contributed
    # A no-section plugin's detail page carries no section to render.
    only_beta = load_registry(builtin=(_NoSectionPlugin,))
    beta_detail = build_plugin_detail(only_beta, "beta", None, _SETTINGS)
    assert beta_detail is not None and beta_detail.section is None


def test_list_flags_reflect_the_section_hook() -> None:
    registry = load_registry(builtin=(_SectionPlugin, _NoSectionPlugin))
    view = build_plugins_view(registry, None, _SETTINGS)
    by_id = {row.plugin_id: row for row in view.active}
    assert by_id["alpha"].has_section is True
    assert by_id["beta"].has_section is False


def test_kill_switch_partitions_active_and_disabled() -> None:
    registry = load_registry(
        builtin=(_SectionPlugin, _NoSectionPlugin), disabled_ids=("beta", "ghost")
    )
    view = build_plugins_view(registry, None, _SETTINGS)
    # A killed plugin is absent from the active list and has no detail page.
    assert {row.plugin_id for row in view.active} == {"alpha"}
    assert build_plugin_detail(registry, "beta", None, _SETTINGS) is None
    assert view.killed_ids == ("beta",)
    assert view.unknown_disabled_ids == ("ghost",)
