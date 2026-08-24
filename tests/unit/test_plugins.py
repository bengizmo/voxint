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
from typing import Any

import pytest

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
