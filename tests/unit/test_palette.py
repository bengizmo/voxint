"""Unit tests for the command palette shell module (issue #162)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from voxint.api.palette import PaletteCommand, palette_actions, palette_destinations
from voxint.config import Settings


def _settings(**overrides: Any) -> Settings:
    """Create a Settings with sensible defaults plus overrides."""
    defaults = {
        "console_projects_enabled": False,
        "console_media_enabled": False,
        "console_speakers_enabled": True,
        "console_activity_enabled": False,
        "console_palette_enabled": True,
        "voxint_multi_user": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _app_state(
    *,
    projects_routed: bool = True,
    media_routed: bool = True,
    activity_routed: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        projects_routed=projects_routed,
        media_routed=media_routed,
        activity_routed=activity_routed,
    )


def _user(role: str = "admin") -> SimpleNamespace:
    return SimpleNamespace(role=role, username="test")


class TestPaletteDestinations:
    """Tests that palette_destinations mirrors the sidebar rail."""

    def test_baseline_single_user(self) -> None:
        """Single-user mode with no area flags: gets the core destinations."""
        dests = palette_destinations(_settings(), _app_state(), None)
        labels = [d["label"] for d in dests]
        assert labels == ["Home", "Media", "Explore", "Speakers", "Runs", "Settings"]

    def test_media_href_depends_on_flag(self) -> None:
        """Media destination href reflects console_media_enabled."""
        dests_off = palette_destinations(
            _settings(console_media_enabled=False), _app_state(), None
        )
        media_off = next(d for d in dests_off if d["label"] == "Media")
        assert media_off["href"] == "/runs"

        dests_on = palette_destinations(
            _settings(console_media_enabled=True), _app_state(), None
        )
        media_on = next(d for d in dests_on if d["label"] == "Media")
        assert media_on["href"] == "/media"

    def test_projects_included_when_enabled_and_routed(self) -> None:
        dests = palette_destinations(
            _settings(console_projects_enabled=True), _app_state(), None
        )
        labels = [d["label"] for d in dests]
        assert "Projects" in labels

    def test_projects_excluded_when_flag_off(self) -> None:
        dests = palette_destinations(
            _settings(console_projects_enabled=False), _app_state(), None
        )
        labels = [d["label"] for d in dests]
        assert "Projects" not in labels

    def test_projects_excluded_when_not_routed(self) -> None:
        dests = palette_destinations(
            _settings(console_projects_enabled=True),
            _app_state(projects_routed=False),
            None,
        )
        labels = [d["label"] for d in dests]
        assert "Projects" not in labels

    def test_settings_included_for_admin(self) -> None:
        dests = palette_destinations(
            _settings(voxint_multi_user=True), _app_state(), _user("admin")
        )
        labels = [d["label"] for d in dests]
        assert "Settings" in labels

    def test_settings_excluded_for_non_admin_multi_user(self) -> None:
        dests = palette_destinations(
            _settings(voxint_multi_user=True), _app_state(), _user("operator")
        )
        labels = [d["label"] for d in dests]
        assert "Settings" not in labels

    def test_account_included_in_multi_user(self) -> None:
        dests = palette_destinations(
            _settings(voxint_multi_user=True), _app_state(), _user("operator")
        )
        labels = [d["label"] for d in dests]
        assert "Account" in labels

    def test_account_excluded_in_single_user(self) -> None:
        dests = palette_destinations(_settings(), _app_state(), None)
        labels = [d["label"] for d in dests]
        assert "Account" not in labels

    def test_all_destinations_are_dicts(self) -> None:
        dests = palette_destinations(_settings(), _app_state(), None)
        for d in dests:
            assert isinstance(d, dict)
            assert "label" in d
            assert "href" in d
            assert "kind" in d
            assert d["kind"] == "destination"


class TestPaletteCommand:
    def test_as_dict(self) -> None:
        cmd = PaletteCommand("Test", "/test", "action", hint="a hint")
        d = cmd.as_dict()
        assert d == {
            "label": "Test",
            "href": "/test",
            "kind": "action",
            "hint": "a hint",
        }

    def test_as_dict_no_hint(self) -> None:
        cmd = PaletteCommand("Test", "/test", "destination")
        assert cmd.as_dict()["hint"] is None


class TestPaletteActions:
    def test_wraps_commands(self) -> None:
        cmd = PaletteCommand("Add media", "/media", "action")
        result = palette_actions(cmd)
        assert len(result) == 1
        assert result[0]["label"] == "Add media"

    def test_empty(self) -> None:
        assert palette_actions() == []
