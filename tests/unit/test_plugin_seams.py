"""Unit coverage for the #138 plugin seam wiring (epic #136).

The empty-registry no-op path is proved by the byte-identical route/task inventory
guards and the existing render tests. These tests inject a synthetic plugin to
prove each *pure* seam helper actually fires: the Features-section effective-meta
merge, the plugin template-dir derivation + ChoiceLoader install, the CLI
subcommand loop, and the plugin CSRF adapter. The router/section/panel *render*
seams need a live app and are covered in
``tests/integration/test_plugin_seams_render.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jinja2 import TemplateNotFound

from voxint.api.csrf import CSRF_PLUGIN, mint_csrf_token
from voxint.api.routers.deps import (
    _configure_template_loader,
    _plugin_template_dirs,
    _verify_plugin_csrf,
    templates,
)
from voxint.api.routers.settings import (
    _FEATURE_FLAG_META,
    _effective_feature_flag_meta,
)
from voxint.plugins.base import (
    FeatureFlag,
    PluginManifest,
    SettingsSection,
    VoxintPlugin,
)
from voxint.plugins.registry import load_registry


def _load_temp_plugin(
    tmp_path: Path, plugin_id: str, *, templates_files: dict[str, str] | None = None
) -> type[VoxintPlugin]:
    """Write a real plugin module to disk and import it, so ``inspect.getfile`` on
    the class resolves to a genuine path (with an optional sibling ``templates/``)."""
    pkgdir = tmp_path / plugin_id
    pkgdir.mkdir(parents=True, exist_ok=True)
    mod_file = pkgdir / "plugin.py"
    mod_file.write_text(
        "from voxint.plugins.base import PluginManifest, VoxintPlugin\n"
        "class P(VoxintPlugin):\n"
        f"    manifest = PluginManifest(id={plugin_id!r}, name='X', description='d')\n"
    )
    if templates_files:
        tdir = pkgdir / "templates"
        tdir.mkdir()
        for name, content in templates_files.items():
            (tdir / name).write_text(content)
    spec = importlib.util.spec_from_file_location(
        f"_voxint_test_plugin_{plugin_id}", mod_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so inspect.getfile(cls) can resolve the class back to
    # its module file (it looks the module up by __module__).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.P  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Effective feature-flag meta merge (seam #3).
# --------------------------------------------------------------------------- #
def test_effective_meta_is_core_object_when_empty() -> None:
    reg = load_registry([])
    # Empty registry returns the SAME object, not a copy — byte-identical render.
    assert _effective_feature_flag_meta(reg) is _FEATURE_FLAG_META


def test_effective_meta_appends_plugin_flags() -> None:
    class Flagged(VoxintPlugin):
        manifest = PluginManifest(id="flagged", name="Flagged", description="d")

        def settings_section(self) -> SettingsSection:
            return SettingsSection(
                section_id="flagged",
                title="Flagged",
                template="flagged/s.html",
                feature_flags=(
                    FeatureFlag("flagged_enabled", "Flagged feature", "help text"),
                ),
            )

    reg = load_registry([Flagged])
    meta = _effective_feature_flag_meta(reg)
    assert meta[: len(_FEATURE_FLAG_META)] == _FEATURE_FLAG_META
    assert meta[-1] == ("flagged_enabled", "Flagged feature", "help text")


# --------------------------------------------------------------------------- #
# Template-dir derivation + loader install (seam #2, rule 10).
# --------------------------------------------------------------------------- #
def test_plugin_template_dirs_finds_sibling_templates(tmp_path: Path) -> None:
    cls = _load_temp_plugin(tmp_path, "withtmpl", templates_files={"s.html": "<p>x</p>"})
    reg = load_registry([cls])
    dirs = _plugin_template_dirs(reg)
    assert dirs == {"withtmpl": str(tmp_path / "withtmpl" / "templates")}


def test_plugin_template_dirs_skips_plugin_without_templates(tmp_path: Path) -> None:
    cls = _load_temp_plugin(tmp_path, "notmpl")
    reg = load_registry([cls])
    assert _plugin_template_dirs(reg) == {}


def _reset_core_template_loader() -> None:
    """Reset the captured core loader so the next _configure_template_loader
    call re-captures from the current templates.env.loader. Needed when a
    prior test or app setup has already installed a ChoiceLoader (e.g. for
    a builtin plugin with templates)."""
    import voxint.api.routers.deps as _deps_mod

    _deps_mod._CORE_TEMPLATE_LOADER = None


def test_configure_template_loader_installs_and_restores(tmp_path: Path) -> None:
    tdir = tmp_path / "t"
    tdir.mkdir()
    (tdir / "hello.html").write_text("PLUGIN-TEMPLATE-MARKER")
    _reset_core_template_loader()
    _configure_template_loader({})
    original = templates.env.loader
    try:
        _configure_template_loader({"myplug": str(tdir)})
        # The namespaced template now resolves through the ChoiceLoader.
        rendered = templates.get_template("myplug/hello.html").render()
        assert rendered == "PLUGIN-TEMPLATE-MARKER"
        # An empty mapping restores the pristine core loader exactly (no nesting).
        _configure_template_loader({})
        assert templates.env.loader is original
        with pytest.raises(TemplateNotFound):
            templates.get_template("myplug/hello.html")
    finally:
        _configure_template_loader({})


def test_configure_template_loader_is_idempotent(tmp_path: Path) -> None:
    _reset_core_template_loader()
    _configure_template_loader({})
    original = templates.env.loader
    _configure_template_loader({})
    _configure_template_loader({})
    # Repeated empty configuration never wraps or nests the core loader.
    assert templates.env.loader is original


# --------------------------------------------------------------------------- #
# CLI subcommand loop (seam #10).
# --------------------------------------------------------------------------- #
def test_build_parser_registers_plugin_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = {"ran": False}

    class CliPlugin(VoxintPlugin):
        manifest = PluginManifest(id="cliplug", name="Cli", description="d")

        def add_cli_commands(self, subparsers: Any) -> None:
            p = subparsers.add_parser("fakecmd", help="a plugin command")
            p.set_defaults(fn=lambda args: marker.__setitem__("ran", True) or 0)

    # build_parser enumerates BUILTIN directly (not the settings-forcing
    # get_plugins), so inject through the builtin set.
    monkeypatch.setattr("voxint.plugins.BUILTIN", (CliPlugin,))

    from voxint.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["fakecmd"])
    assert args.command == "fakecmd"
    assert args.fn(args) == 0
    assert marker["ran"] is True


def test_build_parser_without_plugins_has_no_extra_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("voxint.plugins.BUILTIN", ())
    from voxint.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["fakecmd"])  # no such command


def test_build_parser_survives_invalid_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_parser must not force full Settings validation (issue #138).

    ``--help``, ``--version``, and the settings-free ``score`` command all
    construct the parser, so a broken deployment configuration must not break
    parser construction. When settings fail to load, plugin-subcommand enumeration
    falls back to the environment kill switch rather than raising.
    """
    from voxint import config
    from voxint.cli import build_parser

    def _raise() -> object:
        raise config.SettingsError("invalid settings")

    monkeypatch.setattr(config, "get_settings", _raise)
    monkeypatch.setattr("voxint.plugins.BUILTIN", ())
    parser = build_parser()  # must not raise
    assert parser is not None


# --------------------------------------------------------------------------- #
# Plugin CSRF adapter (seam #6 support).
# --------------------------------------------------------------------------- #
def _fake_request(secret: str, header_token: str | None) -> Any:
    headers = {} if header_token is None else {"x-csrf-token": header_token}
    return SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(csrf_secret=secret)),
    )


def test_verify_plugin_csrf_accepts_valid_header_token() -> None:
    secret = "plugin-csrf-secret"
    token = mint_csrf_token(secret, CSRF_PLUGIN)
    _verify_plugin_csrf(_fake_request(secret, token))  # no raise


def test_verify_plugin_csrf_rejects_missing_and_bad_token() -> None:
    from fastapi import HTTPException

    secret = "plugin-csrf-secret"
    with pytest.raises(HTTPException) as missing:
        _verify_plugin_csrf(_fake_request(secret, None))
    assert missing.value.status_code == 403
    with pytest.raises(HTTPException) as bad:
        _verify_plugin_csrf(_fake_request(secret, "not-a-real-token"))
    assert bad.value.status_code == 403
