"""The plugin interface (issue #137, epic #136).

Voxint's optional features (LLM enrichment, semantic search, translation) become
self-contained builtin plugins under ``voxint.plugins.<id>``. This module is the
contract every plugin implements and every core seam consumes. It imports only
stdlib + typing at runtime (SQLAlchemy / FastAPI / Celery types are referenced
under ``TYPE_CHECKING`` via ``from __future__ import annotations``), so a plugin
package can import the base without dragging the web or task stack into
definition time.

The rules that shape this surface (full decision record in epic #136):

* **Registration is static, enablement is runtime.** Every builtin registers its
  routes, tasks, and CLI unconditionally at process start; the tri-state
  :meth:`VoxintPlugin.enabled` gate is re-checked at execution time and is the
  sole authority. A queued task that finds its feature turned off no-ops; a page
  route renders an honest "off" state rather than 404ing.
* **Import direction is law.** Plugins import core; core imports plugin classes
  ONLY in :mod:`voxint.plugins.discover`. Nothing here imports a plugin.
* **Identity strings are frozen contract.** Celery task names, route paths +
  methods, producer names, env var names, and advisory-lock keys never change
  across a conversion; the inventory guard tests pin them.

Every hook has a no-op default, so a plugin implements only what it uses and the
core dispatch loops can call the full surface on any plugin unconditionally.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from fastapi import APIRouter
    from sqlalchemy.orm import Session, sessionmaker

    from voxint.config import Settings
    from voxint.db.models import AppSettings
    from voxint.plugins.deps import PluginRouteDeps


class PluginError(Exception):
    """A builtin plugin is malformed or two plugins collide.

    Raised during registry load. A baked-in plugin that fails to import,
    validate, or that clashes with another on an identity string is a shipped
    defect, so this fails startup loudly rather than quarantining the plugin
    (the ``VOXINT_PLUGINS_DISABLED`` kill switch is the operator recovery lever).
    """


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


@dataclass(frozen=True)
class PluginManifest:
    """Static identity of a plugin — validated once at construction.

    ``settings_prefixes`` drives the generalized config-parity contract test:
    every :class:`~voxint.config.Settings` field whose name starts with one of
    these prefixes must be documented in ``.env.example`` and mirrored on
    :class:`~voxint.db.models.AppSettings`. ``task_names`` are the plugin's Celery
    task names — exact grandfathered names for a converted feature (frozen so the
    task-inventory guard test never sees drift), or the ``voxint.plugin.<id>.*``
    convention for a greenfield plugin. There is deliberately no ``version`` or
    ``compose_overlay`` field: model-service overlays are documented/doctor copy,
    not manifest data (epic #136).
    """

    id: str
    name: str
    description: str
    settings_prefixes: tuple[str, ...] = ()
    task_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _ID_PATTERN.match(self.id):
            raise PluginError(
                f"invalid plugin id {self.id!r}: must match {_ID_PATTERN.pattern}"
            )
        if not self.name:
            raise PluginError(f"plugin {self.id!r} has an empty name")
        for prefix in self.settings_prefixes:
            if not prefix:
                raise PluginError(f"plugin {self.id!r} has an empty settings prefix")


@dataclass(frozen=True)
class FeatureFlag:
    """One tri-state feature flag a plugin's settings section exposes.

    Mirrors the ``(name, label, help_text)`` shape of the core
    ``_FEATURE_FLAG_META`` table (``api/app.py``); the seam-wiring issue (#138)
    merges these into it so the generic radio POST handler governs plugin flags
    without special-casing. ``name`` is the exact ``config.Settings`` /
    ``AppSettings`` field name.
    """

    name: str
    label: str
    help_text: str


@dataclass(frozen=True)
class SettingsSection:
    """A settings-page section a plugin contributes.

    Rendered into the console settings page by the section loop (#138). The
    section's own template (plugin-namespaced, see rule 10) owns its bespoke
    controls; ``feature_flags`` are the subset of tri-state radios that also feed
    the shared ``_FEATURE_FLAG_META`` table. ``order`` sorts sections ascending;
    ``section_id`` is the stable anchor and the dedupe key.
    """

    section_id: str
    title: str
    template: str
    order: int = 100
    feature_flags: tuple[FeatureFlag, ...] = ()


@dataclass(frozen=True)
class PanelContribution:
    """A fragment a plugin renders into a named slot of ``run_detail.html``.

    The run-detail template exposes named slots (translation, run-assets,
    research today); the run-detail slot loop (#138) renders each contribution's
    plugin-namespaced ``template`` into ``slot``, ascending by ``order``.
    """

    slot: str
    template: str
    order: int = 100


class StaleQueuedLookup(Protocol):
    """The stale-QUEUED id lookup a job lane exposes to the recovery sweep.

    Matches ``enrichment.embedding_jobs.stale_queued_job_ids`` verbatim (#130):
    the ids of jobs stranded in QUEUED since before ``cutoff``, oldest first,
    capped at ``limit``. The generic sweep re-dispatches each by id with no row
    mutation — the guarded claim CAS makes a duplicate delivery a no-op.
    """

    def __call__(
        self, session: Session, *, cutoff: datetime, limit: int
    ) -> Sequence[uuid.UUID]: ...


@dataclass(frozen=True)
class JobLaneSpec:
    """The one piece of a plugin's job lane the core recovery sweep drives.

    Generalizes #130's embedding-only stale-QUEUED redispatch to every lane: the
    sweep calls :attr:`stale_queued_job_ids`, then re-publishes each id to
    :attr:`redispatch_task_name` by name (never importing the plugin's task),
    bounded by :attr:`limit` per sweep. Everything else about a lane — its table,
    its claim CAS, its executor — stays in the plugin; the lanes differ too much
    to share more than this (epic #136).
    """

    stale_queued_job_ids: StaleQueuedLookup
    redispatch_task_name: str
    limit: int


@dataclass(frozen=True)
class RunCompletedEvent:
    """Immutable event handed to :meth:`VoxintPlugin.on_run_completed`.

    Fired after a run's finalize commit. Handlers are enqueue-only and idempotent
    and may fire more than once for a run (a redelivered post task), so they open
    their own short session from :attr:`session_factory`, re-check their gate, and
    create work only when it is missing or stale.
    """

    run_id: uuid.UUID
    session_factory: sessionmaker[Session]
    settings: Settings


class VoxintPlugin:
    """Base class for a builtin plugin — every hook has a no-op default.

    A subclass sets the class-level :attr:`manifest` and overrides only the hooks
    it uses. The registry instantiates each class in
    :data:`voxint.plugins.discover.BUILTIN` once per process. Instances hold no
    per-request state; gate reads take ``(row, settings)`` explicitly so the same
    instance serves the api, the worker, and the CLI.
    """

    manifest: ClassVar[PluginManifest]

    def enabled(self, row: AppSettings | None, settings: Settings) -> bool:
        """Whether this plugin's feature is effectively on (row over env).

        Fail-closed default ``False``: an unimplemented gate never turns a feature
        on by accident. Converted features return their existing
        ``resolve_effective_*`` result; the execution-time re-check of this value
        is the sole authority over api / worker / kill-switch asymmetry.
        """
        return False

    def invariant_errors(
        self, row: AppSettings | None, settings: Settings
    ) -> list[str]:
        """Operator-facing messages for every violated cross-flag invariant.

        Called at boot with ``row=None`` (env-sourced values) and from the
        settings POST with the candidate row. Env-sourced violations fail boot;
        row-sourced violations introduced by an upgrade fail closed to feature-off
        with a doctor warning rather than refusing to boot (rule 5). Empty list ⇒
        valid.
        """
        return []

    def settings_section(self) -> SettingsSection | None:
        """The plugin's settings-page section, or ``None`` for no section."""
        return None

    def run_detail_panels(self) -> Sequence[PanelContribution]:
        """Fragments this plugin contributes to ``run_detail.html`` slots."""
        return ()

    def build_router(self, deps: PluginRouteDeps) -> APIRouter | None:
        """The plugin's routes, mounted on the ``protected`` router (no forced
        prefix; URLs are frozen contract), or ``None`` for no routes."""
        return None

    def task_modules(self) -> Sequence[str]:
        """Import paths of the plugin's Celery task modules (for ``include``)."""
        return ()

    def task_routes(self) -> Mapping[str, Mapping[str, str]]:
        """Celery ``task_routes`` entries for the plugin's tasks.

        Post queue only — a plugin never routes onto the serialized GPU lane.
        """
        return {}

    def on_run_completed(self, event: RunCompletedEvent) -> None:
        """React to a completed run. Enqueue-only, idempotent, may fire twice."""

    def job_lanes(self) -> Sequence[JobLaneSpec]:
        """The plugin's job lanes the core recovery sweep should re-dispatch."""
        return ()

    def add_cli_commands(self, subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
        """Register the plugin's ``voxint`` subcommands (no-op by default)."""


__all__ = [
    "FeatureFlag",
    "JobLaneSpec",
    "PanelContribution",
    "PluginError",
    "PluginManifest",
    "RunCompletedEvent",
    "SettingsSection",
    "StaleQueuedLookup",
    "VoxintPlugin",
]
