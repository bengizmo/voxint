"""Contract: the plugin framework's load-bearing invariants (issue #137).

Three guards that make the later conversions safe, all green while the registry
is empty:

* **Import direction** — core imports plugin *classes* only in ``discover.py``,
  and ``pipeline/`` never imports the framework at all (rule 1, epic #136).
* **Kill-switch config parity** — ``VOXINT_PLUGINS_DISABLED`` is documented and
  defaults to off, like every other setting.
* **Manifest-driven parity scaffold** — the pure checks a conversion must satisfy
  (settings documented + mirrored, gate honors the env default, task names
  grandfathered or convention-compliant), exercised here against synthetic inputs
  and run over the real (empty) registry. #139-#141 reuse these helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.plugins import PluginManifest, VoxintPlugin, get_plugins, parse_disabled_ids

_SRC = REPO_ROOT / "src" / "voxint"
_ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text()

# The framework modules of voxint.plugins. Any OTHER submodule of the package is a
# concrete plugin, importable from core only via discover.BUILTIN.
_FRAMEWORK_MODULES = frozenset(
    {"__init__", "base", "discover", "registry", "hooks", "media", "deps", "doctor"}
)


# --------------------------------------------------------------------------- #
# Import direction (rule 1).
# --------------------------------------------------------------------------- #
def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _concrete_plugin_imports(text: str) -> list[str]:
    """Concrete-plugin import targets (``voxint.plugins.<id>``) named in ``text``.

    Framework-module imports (``voxint.plugins`` itself, ``.base``, ``.registry``,
    …) are allowed everywhere; only an import of a specific plugin package is the
    core->plugin coupling this rule forbids outside ``discover.py``.
    """
    targets: list[str] = []
    for match in re.finditer(
        r"(?:from|import)\s+voxint\.plugins\.([a-z_][a-z0-9_]*)", text
    ):
        module = match.group(1)
        if module not in _FRAMEWORK_MODULES:
            targets.append(module)
    return targets


def test_core_imports_plugin_classes_only_in_discover() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _python_files(_SRC):
        # discover.py is the ONE sanctioned core->plugin import site.
        if path.parent.name == "plugins" and path.name == "discover.py":
            continue
        # Files INSIDE a concrete plugin package legitimately import their own
        # siblings; the rule constrains core (everything outside voxint/plugins).
        if _SRC / "plugins" in path.parents or path.parent.name == "plugins":
            continue
        imported = _concrete_plugin_imports(path.read_text())
        if imported:
            offenders[str(path.relative_to(REPO_ROOT))] = imported
    assert not offenders, (
        "core modules may import a concrete plugin package only via "
        f"discover.BUILTIN; offenders: {offenders}"
    )


def test_pipeline_never_imports_the_plugin_framework() -> None:
    offenders: list[str] = []
    for path in _python_files(_SRC / "pipeline"):
        if re.search(r"(?:from|import)\s+voxint\.plugins\b", path.read_text()):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"the pipeline must not import voxint.plugins at all; offenders: {offenders}"
    )


# --------------------------------------------------------------------------- #
# Kill-switch config parity.
# --------------------------------------------------------------------------- #
def test_kill_switch_documented_and_off_by_default() -> None:
    assert re.search(
        r"^#?\s*VOXINT_PLUGINS_DISABLED=", _ENV_EXAMPLE, re.MULTILINE
    ), ".env.example lacks a VOXINT_PLUGINS_DISABLED line"
    assert Settings(_env_file=None).voxint_plugins_disabled == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("translation", ("translation",)),
        (" a, b ,,c ", ("a", "b", "c")),
        ("Foo", ("Foo",)),  # case preserved — the doctor flags a non-match
    ],
)
def test_parse_disabled_ids(raw: str, expected: tuple[str, ...]) -> None:
    assert parse_disabled_ids(raw) == expected


# --------------------------------------------------------------------------- #
# Manifest-driven parity scaffold (the conversion safety net).
# --------------------------------------------------------------------------- #
def settings_fields_for_prefixes(prefixes: tuple[str, ...]) -> list[str]:
    """Every ``Settings`` field whose name starts with one of ``prefixes``."""
    return [
        name
        for name in Settings.model_fields
        if any(name.startswith(prefix) for prefix in prefixes)
    ]


def undocumented_settings_fields(
    prefixes: tuple[str, ...], env_example: str
) -> list[str]:
    """Prefixed ``Settings`` fields with no ``.env.example`` line."""
    return [
        name
        for name in settings_fields_for_prefixes(prefixes)
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]


def unmirrored_settings_fields(prefixes: tuple[str, ...]) -> list[str]:
    """Prefixed ``Settings`` fields with no matching ``AppSettings`` column."""
    return [
        name
        for name in settings_fields_for_prefixes(prefixes)
        if not hasattr(AppSettings, name)
    ]


def task_name_is_valid(task_name: str, plugin_id: str, grandfathered: frozenset[str]) -> bool:
    """A plugin task name is grandfathered or follows the ``<id>`` convention."""
    return task_name in grandfathered or task_name.startswith(
        f"voxint.plugin.{plugin_id}."
    )


def test_parity_helpers_pass_for_a_documented_feature() -> None:
    # translation_* is a fully documented + mirrored feature surface, so a plugin
    # claiming that prefix satisfies both parity checks — proving green.
    prefixes = ("translation_",)
    assert settings_fields_for_prefixes(prefixes), "translation_ fields vanished"
    assert undocumented_settings_fields(prefixes, _ENV_EXAMPLE) == []
    assert unmirrored_settings_fields(prefixes) == []


def test_parity_helpers_catch_a_gap() -> None:
    # Same real fields, but an empty env document: every field is now undocumented,
    # proving the checker is not vacuously green.
    prefixes = ("translation_",)
    missing = undocumented_settings_fields(prefixes, "")
    assert set(missing) == set(settings_fields_for_prefixes(prefixes))


def test_task_name_convention() -> None:
    grandfathered = frozenset({"voxint.translate_run"})
    assert task_name_is_valid("voxint.translate_run", "translation", grandfathered)
    assert task_name_is_valid("voxint.plugin.foo.run", "foo", grandfathered)
    assert not task_name_is_valid("voxint.rogue", "foo", grandfathered)


def test_registered_plugins_satisfy_parity() -> None:
    # Vacuous while BUILTIN is empty; becomes the live conversion guard in #139+.
    grandfathered = frozenset({"voxint.translate_run"})  # extended per conversion
    for plugin in get_plugins().plugins:
        manifest = plugin.manifest
        assert (
            undocumented_settings_fields(manifest.settings_prefixes, _ENV_EXAMPLE) == []
        )
        assert unmirrored_settings_fields(manifest.settings_prefixes) == []
        for task_name in manifest.task_names:
            assert task_name_is_valid(task_name, manifest.id, grandfathered)


def test_base_plugin_gate_is_fail_closed() -> None:
    # enabled() defaults False so an unimplemented gate never turns a feature on.
    class Bare(VoxintPlugin):
        manifest = PluginManifest(id="bare", name="Bare", description="d")

    assert Bare().enabled(None, Settings(_env_file=None)) is False
