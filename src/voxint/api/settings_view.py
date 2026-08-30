"""Read-only view builders for the Console 2.0 settings sub-pages (issue #161).

Track D (settings) splits the single long ``/settings`` page into a hub plus
read-only sub-pages: status, hardware, database, and plugins. The page handlers
in :mod:`voxint.api.routers.settings` stay thin by delegating their context to
the builders here, keeping the already-large settings router from growing
further. Everything in this module is read-only and best-effort: a probe or query
failure degrades to an honest empty/unknown state rather than raising into the
request, so a settings sub-page renders a 200 even with a dependency down.

Nothing here writes configuration or restarts a service. The hardware page is
"read plus guided edit": it shows the effective value, the exact ``.env`` key,
and the restart needed to apply a change, and surfaces a restart-pending signal
only when it is *honestly observable* — see :func:`build_hardware_view`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from voxint.config import Settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from voxint.api.service_identity import ServiceIdentityView
    from voxint.db.models import AppSettings
    from voxint.plugins.base import SettingsSection
    from voxint.plugins.registry import PluginRegistry


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def install_kind(settings: Settings) -> str:
    """The deployment's install kind: ``"docker"``, ``"native"``, or ``"unknown"``.

    Reads the ``VOXINT_INSTALL_KIND`` marker (#317): the app-image Dockerfile
    bakes ``docker``, the native launcher's launchd plists set ``native``.
    Anything else — unset, a typo, a future value this build does not know —
    degrades to ``"unknown"`` rather than guessing, so dev-from-source honestly
    reports unknown and a bad marker cannot break the status page.
    """
    kind = settings.voxint_install_kind
    if kind in ("docker", "native"):
        return kind
    return "unknown"


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardwareField:
    """One operator-visible hardware/config value with its guided-edit copy.

    ``running`` is the value the process booted with; ``pending`` is the value the
    current environment would supply on the next restart, set only when it
    differs from ``running`` (see :func:`build_hardware_view` for why that is the
    honest restart-pending signal). ``env_key`` is the exact ``.env`` variable
    that changes it. Values are strings, credential-redacted where they are URLs.
    """

    label: str
    env_key: str
    running: str
    pending: str | None = None

    @property
    def restart_pending(self) -> bool:
        return self.pending is not None


# (settings attribute, .env key, operator label, is_url). Env keys match the
# uppercased field names Pydantic reads (config.Settings has no env_prefix), and
# each is documented in .env.example under the name shown here.
_HARDWARE_FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    ("compute_tier", "COMPUTE_TIER", "Compute tier", False),
    ("asr_url", "ASR_URL", "Transcription service URL", True),
    ("diarizer_url", "DIARIZER_URL", "Diarization service URL", True),
    ("embedder_url", "EMBEDDER_URL", "Speaker embedding service URL", True),
    (
        "gpu_http_timeout_seconds",
        "GPU_HTTP_TIMEOUT_SECONDS",
        "Model service timeout (seconds)",
        False,
    ),
    (
        "health_probe_timeout_seconds",
        "HEALTH_PROBE_TIMEOUT_SECONDS",
        "Health probe timeout (seconds)",
        False,
    ),
)


@dataclass(frozen=True)
class HardwareView:
    """The hardware sub-page's read-only model."""

    fields: tuple[HardwareField, ...]
    services: tuple[ServiceIdentityView, ...]
    restart_pending: bool
    # True when the current environment could not be re-read to check for pending
    # changes, so the page says "could not check" rather than implying nothing is
    # pending (a broken .env is exactly when the operator most needs the truth).
    restart_check_failed: bool = False


# Query-parameter names that carry a secret, redacted before a URL is displayed.
_SENSITIVE_QUERY_KEYS = frozenset(
    {"api_key", "apikey", "token", "key", "auth", "sig", "signature", "password", "secret"}
)


def _redact_url(value: str) -> str:
    """Strip credentials from a URL for display, failing closed.

    Removes any ``user:pass@`` userinfo and redacts known-secret query parameters
    (a service URL rarely carries either, but a settings page must never echo a
    secret embedded in one). A value that cannot be parsed as a URL is hidden
    rather than echoed verbatim, so a malformed string can never leak an embedded
    credential; a plain non-URL value (no scheme or host) is returned unchanged.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return "<hidden>"
    if not (parts.scheme or parts.netloc):
        return value
    netloc = parts.netloc.rsplit("@", 1)[-1]  # drop any userinfo
    query = parts.query
    if query:
        query = urlencode(
            [
                (k, "<redacted>" if k.lower() in _SENSITIVE_QUERY_KEYS else v)
                for k, v in parse_qsl(query, keep_blank_values=True)
            ]
        )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def build_hardware_view(
    settings: Settings,
    services: tuple[ServiceIdentityView, ...],
    *,
    environ_settings: Settings | None = None,
) -> HardwareView:
    """The hardware sub-page: effective values, guided-edit copy, restart-pending.

    Restart-pending compares the value the process booted with (``settings``,
    held on ``app.state``) against the value the *current* environment would
    supply on restart (``environ_settings``, a freshly read
    :class:`~voxint.config.Settings`). They differ only when the environment
    genuinely changed after boot without the process restarting — which happens
    in a native long-running deployment whose ``.env`` was edited in place, and
    *cannot* happen under Docker Compose (env is fixed for a container's life, so
    changing it recreates the container). Note an install whose launcher injects
    these values (e.g. the native launchd install) manages them outside ``.env``,
    so the badge correctly never fires for a key the launcher pins — the page copy
    points at "your environment" rather than ``.env`` alone for that reason. If
    the environment cannot be re-read at all (a broken edit fails validation), the
    view reports ``restart_check_failed`` so the page says so honestly instead of
    claiming nothing is pending.
    """
    check_failed = False
    if environ_settings is None:
        try:
            environ_settings = Settings()
        except Exception:
            environ_settings = settings
            check_failed = True

    fields: list[HardwareField] = []
    for attr, env_key, label, is_url in _HARDWARE_FIELDS:
        running_raw = getattr(settings, attr)
        pending_raw = getattr(environ_settings, attr)
        running = _redact_url(str(running_raw)) if is_url else str(running_raw)
        pending: str | None = None
        if pending_raw != running_raw:
            pending = _redact_url(str(pending_raw)) if is_url else str(pending_raw)
        fields.append(
            HardwareField(label=label, env_key=env_key, running=running, pending=pending)
        )
    return HardwareView(
        fields=tuple(fields),
        services=services,
        restart_pending=any(f.restart_pending for f in fields),
        restart_check_failed=check_failed,
    )


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TableCount:
    """One table's estimated live-row count (from ``pg_stat_user_tables``)."""

    name: str
    estimated_rows: int


@dataclass(frozen=True)
class RetentionSetting:
    """One read-only retention/GC value for the database sub-page."""

    label: str
    value: str


@dataclass(frozen=True)
class DatabaseView:
    """The database sub-page's read-only model. ``available`` is False when the
    metadata queries failed (DB down/unmigrated); the page then renders an honest
    "could not read database size" state instead of numbers."""

    available: bool
    size_bytes: int | None
    tables: tuple[TableCount, ...]
    retention: tuple[RetentionSetting, ...]


# Bound on the table list so the page never renders an unbounded row set.
_MAX_TABLES = 25


def _retention_settings(settings: Settings) -> tuple[RetentionSetting, ...]:
    return (
        RetentionSetting(
            "Media retention",
            "on" if settings.media_retention_enabled else "off",
        ),
        RetentionSetting(
            "Retention age (seconds)", str(settings.media_retention_seconds)
        ),
        RetentionSetting("GC sweep interval (seconds)", str(settings.gc_sweep_seconds)),
        RetentionSetting("GC batch limit (rows/sweep)", str(settings.gc_batch_limit)),
    )


def build_database_view(session: Session, settings: Settings) -> DatabaseView:
    """The database sub-page: size, per-table estimates, retention settings.

    Size is ``pg_database_size`` and counts are ``pg_stat_user_tables`` estimates
    (``n_live_tup``) — never an unbounded ``COUNT(*)`` — bounded to the largest
    :data:`_MAX_TABLES` and fail-soft: any query error degrades to ``available =
    False`` so a down or unmigrated database renders honestly rather than 500ing.
    """
    from sqlalchemy import text

    retention = _retention_settings(settings)
    try:
        size_bytes = session.execute(
            text("SELECT pg_database_size(current_database())")
        ).scalar_one()
        rows = session.execute(
            text(
                "SELECT relname, n_live_tup FROM pg_stat_user_tables "
                "ORDER BY n_live_tup DESC LIMIT :limit"
            ),
            {"limit": _MAX_TABLES},
        ).all()
    except Exception:
        session.rollback()
        return DatabaseView(
            available=False, size_bytes=None, tables=(), retention=retention
        )
    tables = tuple(
        TableCount(name=str(name), estimated_rows=int(count or 0)) for name, count in rows
    )
    return DatabaseView(
        available=True,
        size_bytes=int(size_bytes),
        tables=tables,
        retention=retention,
    )


# --------------------------------------------------------------------------- #
# Plugins
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PluginRow:
    """One active plugin, summarized for the registry list."""

    plugin_id: str
    name: str
    description: str
    enabled: bool
    has_section: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PluginsView:
    """The plugins sub-page's read-only model.

    ``killed_ids`` are kill-switched ids present in a builtin; the registry
    discards their instances, so only the id is available for display (name and
    description are not). ``unknown_disabled_ids`` are kill-switch entries naming
    no builtin (a likely typo). An empty ``active`` list is the honest default
    today: no plugins ship yet (``BUILTIN`` is empty).
    """

    active: tuple[PluginRow, ...]
    killed_ids: tuple[str, ...]
    unknown_disabled_ids: tuple[str, ...]


def build_plugins_view(
    registry: PluginRegistry, row: AppSettings | None, settings: Settings
) -> PluginsView:
    """Summarize the plugin registry for the list page (empty registry is fine)."""
    active = tuple(
        PluginRow(
            plugin_id=p.manifest.id,
            name=p.manifest.name,
            description=p.manifest.description,
            enabled=p.enabled(row, settings),
            has_section=p.settings_section() is not None,
            warnings=tuple(p.invariant_errors(row, settings)),
        )
        for p in registry.plugins
    )
    known_disabled = sorted(registry.disabled_ids - registry.unknown_disabled_ids)
    return PluginsView(
        active=active,
        killed_ids=tuple(known_disabled),
        unknown_disabled_ids=tuple(sorted(registry.unknown_disabled_ids)),
    )


@dataclass(frozen=True)
class PluginDetailView:
    """One plugin's detail page: its identity, gate state, and settings section."""

    plugin_id: str
    name: str
    description: str
    enabled: bool
    warnings: tuple[str, ...]
    section: SettingsSection | None


def build_plugin_detail(
    registry: PluginRegistry,
    plugin_id: str,
    row: AppSettings | None,
    settings: Settings,
) -> PluginDetailView | None:
    """The per-plugin page model, or ``None`` when the id is unknown or killed.

    A ``None`` return is the handler's 404 signal. A killed plugin is absent from
    ``registry.plugins`` and so resolves to ``None`` here — it has no detail page,
    only a listed id on the registry page.
    """
    plugin = registry.get(plugin_id)
    if plugin is None:
        return None
    return PluginDetailView(
        plugin_id=plugin.manifest.id,
        name=plugin.manifest.name,
        description=plugin.manifest.description,
        enabled=plugin.enabled(row, settings),
        warnings=tuple(plugin.invariant_errors(row, settings)),
        section=plugin.settings_section(),
    )
