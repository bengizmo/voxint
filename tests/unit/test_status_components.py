"""Status-page component rows for the two AI lanes (#316).

``_build_components`` maps doctor checks to the R6 component list. The old
single "Local AI model" row probed only the BYO endpoint and painted a healthy
bundled-only install as "rejected (HTTP 401)". These tests pin the two-row
split and the banner policy (the banner keys on warn dots: "not configured"
and "off" are never warn; a deliberately-configured endpoint that rejects is).
"""

from typing import Any

from voxint.api.routers.settings import _build_components
from voxint.config import Settings


def _check(name: str, state: str, detail: str) -> dict[str, Any]:
    return {"name": name, "state": state, "detail": detail, "remediation": ""}


def _settings(**over: object) -> Settings:
    return Settings(voxint_user="u", voxint_password="p", **over)


def _row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return next(r for r in rows if r["label"] == label)


def test_bundled_only_install_shows_green_bundle_and_unconfigured_byo() -> None:
    # The walkthrough scenario: bundle healthy, BYO untouched. Neither row may
    # be warn, so the banner (all dots != warn) stays green.
    rows = _build_components(
        [
            _check("llm bundled", "ready", "reachable (HTTP 200)"),
            _check("llm endpoint", "ready", "not configured"),
        ],
        _settings(llm_enabled=True),
    )
    bundled = _row(rows, "Bundled AI model")
    assert bundled["dot"] == "ok"
    assert bundled["state_text"] == "running · reachable (HTTP 200)"
    byo = _row(rows, "Your own AI endpoint")
    assert byo["dot"] == "off"
    assert byo["state_text"] == "not configured"
    assert byo["action_url"] == "/settings#llm" and byo["action_label"] == "Set up"
    assert all(r["dot"] != "warn" for r in rows)


def test_bundled_unreachable_is_a_warning() -> None:
    rows = _build_components(
        [_check("llm bundled", "unverified", "unreachable (ConnectError)")],
        _settings(),
    )
    bundled = _row(rows, "Bundled AI model")
    assert bundled["dot"] == "warn"
    assert bundled["state_text"] == "unreachable (ConnectError)"


def test_bundle_inactive_renders_an_off_row_not_a_warning() -> None:
    rows = _build_components([], _settings())
    bundled = _row(rows, "Bundled AI model")
    assert bundled["dot"] == "off" and bundled["state_text"] == "off"


def test_llm_disabled_byo_row_keeps_turn_on_action() -> None:
    rows = _build_components([], _settings(llm_enabled=False))
    byo = _row(rows, "Your own AI endpoint")
    assert byo["dot"] == "off"
    assert byo["state_text"] == "off -- used for polish & profiles"
    assert byo["action_url"] == "/settings#llm" and byo["action_label"] == "Turn on"


def test_deliberately_configured_byo_that_rejects_stays_warn() -> None:
    # byo_llm_configured() encodes deliberateness: a configured endpoint that
    # rejects is real information and must keep flipping the banner.
    rows = _build_components(
        [
            _check("llm bundled", "ready", "reachable (HTTP 200)"),
            _check("llm endpoint", "unverified", "rejected (HTTP 401)"),
        ],
        _settings(llm_enabled=True),
    )
    byo = _row(rows, "Your own AI endpoint")
    assert byo["dot"] == "warn"
    assert byo["state_text"] == "rejected (HTTP 401)"
    assert not all(r["dot"] != "warn" for r in rows)  # banner flips


def test_old_single_label_is_gone() -> None:
    rows = _build_components(
        [_check("llm endpoint", "ready", "reachable (HTTP 200)")],
        _settings(llm_enabled=True),
    )
    assert all(r["label"] != "Local AI model" for r in rows)
