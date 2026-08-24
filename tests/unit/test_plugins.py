"""Unit tests for the plugin framework machinery (issue #137).

Registry load / ordering / collisions / kill switch, the hook dispatchers'
failure isolation and broker-down handling, and the doctor formatter — all with
synthetic plugins, no database or broker.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from voxint.plugins import (
    get_plugins,
    load_plugins,
    parse_disabled_ids,
    reset_plugins_cache,
)
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
from voxint.plugins.deps import PluginRouteDeps
from voxint.plugins.doctor import format_plugins_status
from voxint.plugins.hooks import dispatch_run_completed, redispatch_stale_lane_jobs
from voxint.plugins.registry import find_route_collisions, load_registry


def _manifest(plugin_id: str, **kw: Any) -> PluginManifest:
    return PluginManifest(id=plugin_id, name=plugin_id.title(), description="d", **kw)


def _plugin(plugin_id: str, **kw: Any) -> type[VoxintPlugin]:
    manifest = _manifest(plugin_id, **kw)

    class _P(VoxintPlugin):
        pass

    _P.manifest = manifest
    _P.__name__ = f"Plugin_{plugin_id}"
    return _P


# --------------------------------------------------------------------------- #
# PluginManifest validation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "A", "1abc", "has-dash", "way_too_" + "x" * 40])
def test_manifest_rejects_bad_id(bad: str) -> None:
    with pytest.raises(PluginError):
        _manifest(bad)


def test_manifest_rejects_empty_name() -> None:
    with pytest.raises(PluginError):
        PluginManifest(id="ok", name="", description="d")


def test_manifest_is_frozen() -> None:
    manifest = _manifest("ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.id = "other"  # type: ignore[misc]


def test_manifest_rejects_empty_settings_prefix() -> None:
    with pytest.raises(PluginError, match="empty settings prefix"):
        _manifest("ok", settings_prefixes=("valid_", ""))


# --------------------------------------------------------------------------- #
# VoxintPlugin base-class hook defaults (fail-closed / empty contract).
# --------------------------------------------------------------------------- #
def test_base_hook_defaults_are_fail_closed_and_empty() -> None:
    plugin = _plugin("bare")()  # instantiate the bare subclass

    # enabled defaults False (a feature never turns on by accident), and the
    # collection-returning hooks default to empty, None-returning hooks to None.
    assert plugin.enabled(None, None) is False  # type: ignore[arg-type]
    assert plugin.invariant_errors(None, None) == []  # type: ignore[arg-type]
    assert plugin.settings_section() is None
    assert plugin.run_detail_panels() == ()
    assert plugin.build_router(None) is None  # type: ignore[arg-type]
    assert plugin.task_modules() == ()
    assert plugin.task_routes() == {}
    assert plugin.job_lanes() == ()
    # Side-effect-only hooks return None; the defaults must simply not raise.
    plugin.on_run_completed(_event())
    plugin.add_cli_commands(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Registry load.
# --------------------------------------------------------------------------- #
def test_empty_registry() -> None:
    reg = load_registry([])
    assert reg.plugins == ()
    assert reg.unknown_disabled_ids == frozenset()


def test_registry_orders_by_id() -> None:
    reg = load_registry([_plugin("zebra"), _plugin("alpha"), _plugin("mid")])
    assert [p.manifest.id for p in reg.plugins] == ["alpha", "mid", "zebra"]


def test_kill_switch_filters_and_reports_unknown() -> None:
    reg = load_registry(
        [_plugin("keep"), _plugin("drop")], disabled_ids=["drop", "ghost"]
    )
    assert [p.manifest.id for p in reg.plugins] == ["keep"]
    assert reg.disabled_ids == frozenset({"drop", "ghost"})
    assert reg.unknown_disabled_ids == frozenset({"ghost"})


def test_get_returns_active_or_none() -> None:
    reg = load_registry([_plugin("keep"), _plugin("drop")], disabled_ids=["drop"])
    assert reg.get("keep") is not None
    assert reg.get("drop") is None  # killed
    assert reg.get("absent") is None


def test_duplicate_id_fails_loud() -> None:
    with pytest.raises(PluginError, match="duplicate plugin id 'dup'"):
        load_registry([_plugin("dup"), _plugin("dup")])


def test_duplicate_id_caught_even_when_one_is_killed() -> None:
    # Collision validation runs over ALL builtins, before the kill switch filters.
    with pytest.raises(PluginError, match="duplicate plugin id"):
        load_registry([_plugin("dup"), _plugin("dup")], disabled_ids=["dup"])


def test_duplicate_task_name_fails_loud() -> None:
    with pytest.raises(PluginError, match=r"duplicate Celery task name 'voxint\.x'"):
        load_registry(
            [
                _plugin("aa", task_names=("voxint.x",)),
                _plugin("bb", task_names=("voxint.x",)),
            ]
        )


def test_manifest_rejects_empty_task_name() -> None:
    with pytest.raises(PluginError, match="empty task name"):
        _manifest("ok", task_names=("voxint.real", "  "))


def test_task_routes_for_undeclared_task_fails_loud() -> None:
    class Rogue(VoxintPlugin):
        manifest = _manifest("rogue", task_names=("voxint.rogue",))

        def task_routes(self) -> dict[str, dict[str, str]]:
            # Routes a task this plugin never declares — would silently overwrite
            # another plugin's route without the collision check.
            return {"voxint.someone_else": {"queue": "post"}}

    with pytest.raises(PluginError, match="routes task"):
        load_registry([Rogue])


def test_duplicate_settings_section_id_fails_loud() -> None:
    def _sectioned(plugin_id: str) -> type[VoxintPlugin]:
        class _P(VoxintPlugin):
            manifest = _manifest(plugin_id)

            def settings_section(self) -> SettingsSection:
                return SettingsSection(
                    section_id="shared", title="T", template="t.html"
                )

        _P.__name__ = f"Sec_{plugin_id}"
        return _P

    with pytest.raises(PluginError, match="duplicate settings section_id 'shared'"):
        load_registry([_sectioned("aa"), _sectioned("bb")])


def test_disabled_plugin_section_id_does_not_collide() -> None:
    # A killed plugin contributes no section, so its id cannot collide with an
    # active one — the check runs over the active set only.
    class Off(VoxintPlugin):
        manifest = _manifest("off")

        def settings_section(self) -> SettingsSection:
            return SettingsSection(section_id="shared", title="T", template="t.html")

    class On(VoxintPlugin):
        manifest = _manifest("on")

        def settings_section(self) -> SettingsSection:
            return SettingsSection(section_id="shared", title="T", template="t.html")

    reg = load_registry([Off, On], disabled_ids=["off"])
    assert [p.manifest.id for p in reg.plugins] == ["on"]


def test_missing_manifest_fails_loud() -> None:
    class NoManifest(VoxintPlugin):
        pass

    with pytest.raises(PluginError, match="no valid `manifest`"):
        load_registry([NoManifest])


def test_construction_failure_fails_loud() -> None:
    class Boom(VoxintPlugin):
        manifest = _manifest("boom")

        def __init__(self) -> None:
            raise RuntimeError("nope")

    with pytest.raises(PluginError, match="failed to instantiate"):
        load_registry([Boom])


# --------------------------------------------------------------------------- #
# Registry aggregation.
# --------------------------------------------------------------------------- #
def test_registry_aggregates_contributions() -> None:
    class Alpha(VoxintPlugin):
        manifest = _manifest("alpha", task_names=("voxint.alpha",))

        def task_routes(self) -> dict[str, dict[str, str]]:
            return {"voxint.alpha": {"queue": "post"}}

        def settings_section(self) -> SettingsSection:
            return SettingsSection(
                section_id="alpha",
                title="Alpha",
                template="alpha/s.html",
                order=50,
                feature_flags=(FeatureFlag("alpha_enabled", "Alpha", "help"),),
            )

        def run_detail_panels(self) -> Sequence[PanelContribution]:
            return (PanelContribution(slot="main", template="alpha/p.html"),)

    class Beta(VoxintPlugin):
        manifest = _manifest("beta", task_names=("voxint.beta",))

        def settings_section(self) -> SettingsSection:
            return SettingsSection(
                section_id="beta", title="Beta", template="beta/s.html", order=10
            )

    reg = load_registry([Alpha, Beta])
    assert reg.task_names() == ("voxint.alpha", "voxint.beta")
    assert reg.task_routes() == {"voxint.alpha": {"queue": "post"}}
    # Sections sort by (order, section_id): Beta(10) before Alpha(50).
    assert [s.section_id for s in reg.settings_sections()] == ["beta", "alpha"]
    assert [p.slot for p in reg.run_detail_panels()] == ["main"]


def test_find_route_collisions() -> None:
    assert find_route_collisions({}) == []
    assert (
        find_route_collisions(
            {"alpha": [("/x", "GET"), ("/y", "POST")], "beta": [("/z", "GET")]}
        )
        == []
    )
    # Same path+method across two plugins collides; method case is normalized.
    collisions = find_route_collisions(
        {"alpha": [("/dup", "get")], "beta": [("/dup", "GET"), ("/ok", "POST")]}
    )
    assert len(collisions) == 1
    assert "/dup" in collisions[0]


# --------------------------------------------------------------------------- #
# run_completed dispatch isolation.
# --------------------------------------------------------------------------- #
def _event() -> RunCompletedEvent:
    return RunCompletedEvent(
        run_id=uuid.uuid4(),
        session_factory=None,  # type: ignore[arg-type]  # handlers here open no session
        settings=None,  # type: ignore[arg-type]
    )


def test_run_completed_dispatch_isolates_failures() -> None:
    calls: list[str] = []

    class Good(VoxintPlugin):
        manifest = _manifest("good")

        def on_run_completed(self, event: RunCompletedEvent) -> None:
            calls.append("good")

    class Bad(VoxintPlugin):
        manifest = _manifest("bad")

        def on_run_completed(self, event: RunCompletedEvent) -> None:
            raise RuntimeError("handler blew up")

    reg = load_registry([Bad, Good])  # ordered bad, good — good still runs
    dispatch_run_completed(reg.plugins, _event())
    assert calls == ["good"]


# --------------------------------------------------------------------------- #
# Stale-lane redispatch.
# --------------------------------------------------------------------------- #
@contextmanager
def _fake_factory() -> Iterator[object]:
    yield object()


def _lane(task_name: str, ids: Sequence[uuid.UUID], limit: int = 100) -> JobLaneSpec:
    def lookup(session: object, *, cutoff: datetime, limit: int) -> Sequence[uuid.UUID]:
        return list(ids)[:limit]

    return JobLaneSpec(
        stale_queued_job_ids=lookup,  # type: ignore[arg-type]
        redispatch_task_name=task_name,
        limit=limit,
    )


def test_redispatch_sends_each_stale_id_by_name() -> None:
    sent: list[tuple[str, tuple[Any, ...]]] = []

    def send_task(name: str, *, args: tuple[Any, ...], ignore_result: bool) -> None:
        sent.append((name, args))

    ids = [uuid.uuid4(), uuid.uuid4()]
    result = redispatch_stale_lane_jobs(
        [_lane("voxint.lane", ids)],
        session_factory=lambda: _fake_factory(),  # type: ignore[arg-type]
        send_task=send_task,
        cutoff=datetime.now(tz=UTC),
    )
    assert result == {"voxint.lane": 2}
    assert [name for name, _ in sent] == ["voxint.lane", "voxint.lane"]
    assert sent[0][1] == (str(ids[0]),)


def test_redispatch_stops_a_lane_when_broker_is_down() -> None:
    from celery.exceptions import OperationalError

    def send_task(name: str, *, args: tuple[Any, ...], ignore_result: bool) -> None:
        raise OperationalError("broker down")

    result = redispatch_stale_lane_jobs(
        [_lane("voxint.lane", [uuid.uuid4(), uuid.uuid4()])],
        session_factory=lambda: _fake_factory(),  # type: ignore[arg-type]
        send_task=send_task,
        cutoff=datetime.now(tz=UTC),
    )
    assert result == {}  # nothing counted; lane deferred to a later sweep


def test_redispatch_isolates_a_lane_whose_lookup_raises() -> None:
    # A lane whose stale-id lookup blows up must not abort the whole sweep; the
    # healthy lane still redispatches (mirrors dispatch_run_completed isolation).
    def bad_lookup(session: object, *, cutoff: datetime, limit: int) -> Sequence[uuid.UUID]:
        raise RuntimeError("bad SQL")

    bad_lane = JobLaneSpec(
        stale_queued_job_ids=bad_lookup,  # type: ignore[arg-type]
        redispatch_task_name="voxint.bad",
        limit=100,
    )
    good_ids = [uuid.uuid4()]
    sent: list[str] = []

    def send_task(name: str, *, args: tuple[Any, ...], ignore_result: bool) -> None:
        sent.append(name)

    result = redispatch_stale_lane_jobs(
        [bad_lane, _lane("voxint.good", good_ids)],
        session_factory=lambda: _fake_factory(),  # type: ignore[arg-type]
        send_task=send_task,
        cutoff=datetime.now(tz=UTC),
    )
    assert result == {"voxint.good": 1}
    assert sent == ["voxint.good"]


def test_redispatch_isolates_a_lane_whose_send_raises_unexpectedly() -> None:
    # A non-broker send error on lane A is contained; lane B still runs.
    ids_a, ids_b = [uuid.uuid4()], [uuid.uuid4()]

    def send_task(name: str, *, args: tuple[Any, ...], ignore_result: bool) -> None:
        if name == "voxint.a":
            raise ValueError("serialization boom")

    result = redispatch_stale_lane_jobs(
        [_lane("voxint.a", ids_a), _lane("voxint.b", ids_b)],
        session_factory=lambda: _fake_factory(),  # type: ignore[arg-type]
        send_task=send_task,
        cutoff=datetime.now(tz=UTC),
    )
    assert result == {"voxint.b": 1}  # lane A contained, lane B unaffected


# --------------------------------------------------------------------------- #
# Doctor formatter.
# --------------------------------------------------------------------------- #
def test_doctor_status_none_when_empty() -> None:
    assert format_plugins_status(load_registry([])) is None


def test_doctor_status_lists_active_disabled_and_unknown() -> None:
    reg = load_registry(
        [_plugin("keep"), _plugin("off")], disabled_ids=["off", "ghost"]
    )
    status = format_plugins_status(reg)
    assert status is not None
    assert "plugins active: keep" in status
    assert "plugins disabled (kill switch): off" in status
    assert "unknown plugin ids: ghost" in status


# --------------------------------------------------------------------------- #
# PluginRouteDeps bundle (frozen, exact surface).
# --------------------------------------------------------------------------- #
def test_plugin_route_deps_holds_its_fields_and_is_frozen() -> None:
    templates = object()
    get_session = object()
    verify_csrf = object()
    render_settings_page = object()
    deps = PluginRouteDeps(
        templates=templates,  # type: ignore[arg-type]
        get_session=get_session,  # type: ignore[arg-type]
        verify_csrf=verify_csrf,  # type: ignore[arg-type]
        render_settings_page=render_settings_page,  # type: ignore[arg-type]
    )
    assert deps.templates is templates
    assert deps.get_session is get_session
    assert deps.verify_csrf is verify_csrf
    assert deps.render_settings_page is render_settings_page
    with pytest.raises(dataclasses.FrozenInstanceError):
        deps.templates = object()  # type: ignore[misc, assignment]


# --------------------------------------------------------------------------- #
# Process registry cache (parse / load / get / reset).
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_registry_cache() -> Iterator[None]:
    reset_plugins_cache()
    yield
    reset_plugins_cache()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("a,b,c", ("a", "b", "c")),
        ("  a , , b ,", ("a", "b")),  # trim, drop empties, preserve order
        ("Keep,Case", ("Keep", "Case")),  # case is matched exactly, not coerced
    ],
)
def test_parse_disabled_ids(raw: str, expected: tuple[str, ...]) -> None:
    assert parse_disabled_ids(raw) == expected


def test_load_plugins_builds_from_empty_builtin_and_caches() -> None:
    settings = SimpleNamespace(voxint_plugins_disabled="")
    reg = load_plugins(settings)  # type: ignore[arg-type]
    # BUILTIN is empty in #137, so the built registry is empty and dormant.
    assert reg.plugins == ()
    # get_plugins returns the same cached instance without rebuilding.
    assert get_plugins() is reg


def test_load_plugins_threads_the_kill_switch_through() -> None:
    settings = SimpleNamespace(voxint_plugins_disabled="ghost, other")
    reg = load_plugins(settings)  # type: ignore[arg-type]
    # No builtins to filter, but the parsed ids reach the registry and surface
    # as unknown for the doctor.
    assert reg.disabled_ids == frozenset({"ghost", "other"})
    assert reg.unknown_disabled_ids == frozenset({"ghost", "other"})


def test_get_plugins_lazy_fallback_builds_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No prior load_plugins call: get_plugins memoizes on first access via
    # get_settings() rather than being ordering-fragile.
    settings = SimpleNamespace(voxint_plugins_disabled="")
    monkeypatch.setattr("voxint.config.get_settings", lambda: settings)
    reg = get_plugins()
    assert reg.plugins == ()
    assert get_plugins() is reg  # second access is the cached instance


def test_reset_plugins_cache_forces_a_rebuild() -> None:
    first = load_plugins(SimpleNamespace(voxint_plugins_disabled=""))  # type: ignore[arg-type]
    reset_plugins_cache()
    second = load_plugins(SimpleNamespace(voxint_plugins_disabled=""))  # type: ignore[arg-type]
    assert first is not second  # a fresh process / test starts empty


# --------------------------------------------------------------------------- #
# validate_boot (issue #138): env-sourced plugin invariants fail boot loud.
# --------------------------------------------------------------------------- #
def test_validate_boot_no_op_on_empty_registry() -> None:
    from voxint.plugins.boot import validate_boot

    reg = load_registry([])
    # Nothing to iterate: a dormant (#138) registry boots clean.
    validate_boot(reg, settings=SimpleNamespace())  # type: ignore[arg-type]


def test_validate_boot_passes_when_invariants_clean() -> None:
    from voxint.plugins.boot import validate_boot

    class Clean(VoxintPlugin):
        manifest = _manifest("clean")

        def invariant_errors(self, row: Any, settings: Any) -> list[str]:
            return []

    reg = load_registry([Clean])
    validate_boot(reg, settings=SimpleNamespace())  # type: ignore[arg-type]


def test_validate_boot_calls_hook_with_env_reading() -> None:
    from voxint.plugins.boot import validate_boot

    seen: list[Any] = []

    class Spy(VoxintPlugin):
        manifest = _manifest("spy")

        def invariant_errors(self, row: Any, settings: Any) -> list[str]:
            seen.append(row)
            return []

    reg = load_registry([Spy])
    validate_boot(reg, settings=SimpleNamespace())  # type: ignore[arg-type]
    # Boot reads the env-sourced config: row is None (no candidate AppSettings).
    assert seen == [None]


def test_validate_boot_aggregates_violations_with_plugin_ids() -> None:
    from voxint.plugins.boot import validate_boot

    class Bad(VoxintPlugin):
        manifest = _manifest("bad")

        def invariant_errors(self, row: Any, settings: Any) -> list[str]:
            return ["needs a base url", "needs a key"]

    class AlsoBad(VoxintPlugin):
        manifest = _manifest("alsobad")

        def invariant_errors(self, row: Any, settings: Any) -> list[str]:
            return ["something else"]

    reg = load_registry([Bad, AlsoBad])
    with pytest.raises(PluginError) as exc:
        validate_boot(reg, settings=SimpleNamespace())  # type: ignore[arg-type]
    message = str(exc.value)
    assert "[bad] needs a base url" in message
    assert "[bad] needs a key" in message
    assert "[alsobad] something else" in message


def test_validate_boot_wraps_a_raising_hook() -> None:
    from voxint.plugins.boot import validate_boot

    class Explodes(VoxintPlugin):
        manifest = _manifest("explodes")

        def invariant_errors(self, row: Any, settings: Any) -> list[str]:
            raise RuntimeError("kaboom")

    reg = load_registry([Explodes])
    with pytest.raises(PluginError, match="plugin 'explodes' invariant check failed"):
        validate_boot(reg, settings=SimpleNamespace())  # type: ignore[arg-type]
