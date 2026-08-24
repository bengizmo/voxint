"""Shared console-router plumbing: dependencies, auth gate, templates, assets.

Everything here is cross-area infrastructure the per-area routers (and the
remaining routes in ``app.py``) resolve at request time: the request-scoped DB
session, the media gate, the onboarding gate, the CSRF verifier, the shared
Jinja environment with its display globals, and the Vite asset manifest that
backs ``asset_url``. Area-specific helpers stay with their area module.

This module must stay acyclic: it never imports ``voxint.api.app`` or any
router module.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from voxint.api.auth import require_operator
from voxint.api.csrf import verify_csrf_token
from voxint.api.languages import language_label
from voxint.api.presentation import (
    format_age,
    format_duration,
    friendly_media_label,
    humanize_stage,
    humanize_status,
)
from voxint.api.resource_status import (
    device_state,
    gib,
    short_uuid,
    vram_percent,
)
from voxint.app_settings import is_onboarded
from voxint.config import Settings
from voxint.db.models import PipelineRun, Stage
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import operator_correction_message
from voxint.media.serving import MediaGate

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Prebuilt frontend island bundles (issue #48). The Vite build stage in the
# Dockerfile copies dist/ here; running from source with no build leaves it
# absent, which the manifest helper and the asset route both tolerate (pages
# still render server-side — progressive enhancement holds even without a build).
_APP_ASSETS_DIR = (Path(__file__).parent.parent / "static" / "app").resolve()
_APP_MANIFEST_PATH = _APP_ASSETS_DIR / ".vite" / "manifest.json"
_APP_ASSET_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
}
# Vite fingerprints emitted filenames as `<name>-<hash>.<ext>` (base64url hash);
# only those get long-immutable caching (an unhashed name could change in place).
_HASHED_ASSET_RE = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")


def _looks_hashed(name: str) -> bool:
    """True for Vite content-hashed filenames, which are safe to cache forever."""
    return _HASHED_ASSET_RE.search(name) is not None


def _load_asset_manifest() -> dict[str, str]:
    """Map each Vite entry name to its served ``/static/app/...`` URL.

    Parsed once at import. The Vite manifest is keyed by source path
    (``src/main.ts``); we key by the path stem (``main``) so templates request
    islands by their logical entry name. Returns ``{}`` when no build is present
    so ``asset_url`` yields ``None`` and templates emit nothing.
    """
    try:
        raw = _APP_MANIFEST_PATH.read_text()
    except OSError:
        return {}
    try:
        records: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("frontend asset manifest is not valid JSON; islands disabled")
        return {}
    by_entry: dict[str, str] = {}
    for src, record in records.items():
        file = record.get("file") if isinstance(record, dict) else None
        if isinstance(file, str):
            by_entry[Path(src).stem] = "/static/app/" + file
    return by_entry


_APP_ASSET_URLS = _load_asset_manifest()


def asset_url(entry_name: str) -> str | None:
    """Jinja global: entry name -> served URL, or ``None`` when unbuilt."""
    return _APP_ASSET_URLS.get(entry_name)


def _get_session(request: Request) -> Iterator[Session]:
    # Delegates the commit-on-success / rollback-on-exception body to the single
    # session_scope contextmanager rather than duplicating it: FastAPI resumes a
    # yield-dependency past its `yield` on success and throws the route's
    # exception back in on failure, which is exactly the control flow the `with`
    # needs to drive session_scope's commit/rollback. Mutations that commit
    # before publishing (POST /submit, /runs/{id}/requeue) make the trailing
    # commit here a harmless no-op — nothing is left pending.
    factory = request.app.state.session_factory
    if factory is None:
        factory = build_session_factory(build_engine(request.app.state.settings.database_url))
        request.app.state.session_factory = factory
    with session_scope(factory) as session:
        yield session


def _get_media_gate(request: Request) -> MediaGate:
    gate = cast(MediaGate | None, request.app.state.media_gate)
    if gate is None:
        settings: Settings = request.app.state.settings
        gate = MediaGate(
            settings.media_root,
            ffprobe_bin=settings.ffprobe_bin,
            timeout_seconds=settings.media_probe_timeout_seconds,
        )
        request.app.state.media_gate = gate
    return gate


SessionDep = Annotated[Session, Depends(_get_session)]
OperatorDep = Annotated[str, Depends(require_operator)]


def require_onboarded(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
) -> None:
    """First-run gate: redirect an un-onboarded operator to the setup wizard.

    Wired as a single router-level dependency on the *protected* router that
    carries every non-exempt route (``/healthz``, the htmx asset, and ``/setup``
    stay on ``app`` and so are structurally exempt — no path matching to keep in
    sync). It depends on ``OperatorDep`` so authentication runs first: an
    unauthenticated request gets a 401 challenge, never a redirect that would leak
    onboarding state. It depends on ``SessionDep`` so FastAPI's per-request
    dependency cache hands the gate the same ``Session`` the route handler uses —
    one connection, not two.

    The onboarding read is cached on ``request.state`` for the life of the request
    only. It is deliberately NOT cached on ``app.state``: the Celery worker can
    flip ``onboarding_complete`` in its own process, so a cross-request cache would
    serve a stale answer. Not onboarded ⇒ ``303`` to ``/setup`` for an ordinary
    navigation, or a ``204`` carrying ``HX-Redirect`` for an htmx request (htmx
    performs the client-side redirect; a 303's body would be swapped into the page
    instead of navigating).
    """
    onboarded = getattr(request.state, "onboarded", None)
    if onboarded is None:
        onboarded = is_onboarded(session)
        request.state.onboarded = onboarded
    if onboarded:
        return
    if request.headers.get("HX-Request"):
        raise HTTPException(status_code=204, headers={"HX-Redirect": "/setup"})
    raise HTTPException(status_code=303, headers={"Location": "/setup"})


templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
# Island bundle lookup for base.html: `asset_url('main')` / `asset_url('tailwind')`
# resolve to the hashed built file, or None (guarded in the template) when unbuilt.
templates.env.globals["asset_url"] = asset_url
# Operator-facing display helpers (issue #56), called directly from the console
# templates. `format_age` takes an injected `now` the routes pass in context.
templates.env.globals["friendly_media_label"] = friendly_media_label
templates.env.globals["format_duration"] = format_duration
templates.env.globals["format_age"] = format_age
templates.env.globals["humanize_stage"] = humanize_stage
templates.env.globals["humanize_status"] = humanize_status
templates.env.globals["language_label"] = language_label
# Hardware-telemetry display helpers (W3): bytes -> GiB number, used-VRAM %.
templates.env.globals["gib"] = gib
templates.env.globals["vram_percent"] = vram_percent
templates.env.globals["short_uuid"] = short_uuid
templates.env.globals["device_state"] = device_state


def _run_or_404(session: Session, run_id: uuid.UUID) -> PipelineRun:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


def _reject_if_archived(run: PipelineRun) -> None:
    """409 if the run is soft-archived (issue #5) — archived runs are read-only.

    Filtering hides archived runs from ``/runs`` and the review queue, but a
    stale tab or a hand-crafted POST could still drive one live (requeue) or
    claim it. Refuse those mutations until the operator un-archives it, so
    visibility and mutability stay aligned."""
    if run.archived_at is not None:
        raise HTTPException(
            status_code=409,
            detail="run is archived; un-archive it before requeue/claim",
        )


def _require_csrf(request: Request, action: str, token: str | None) -> None:
    """403 unless ``token`` is a valid CSRF token for ``action`` — call before any
    state change. A missing token and a mis-signed one BOTH 403 (the field is
    Optional, so FastAPI never turns an absent token into a 422), giving a forged
    cross-site POST one uniform refusal before the DB is touched."""
    if not verify_csrf_token(request.app.state.csrf_secret, action, token):
        raise HTTPException(status_code=403, detail="invalid or missing CSRF token")


def _publish_run(run_id: uuid.UUID, *, stage: Stage | None = None) -> None:
    """Enqueue a fresh submission or stage-routed resumption (commit-before-publish).

    The Celery/broker import stays out of the module top level so the read path
    — and any DB-less import — never pulls in the broker.

    ``ignore_result=True``: nothing in Voxint consumes ``run_pipeline``'s return
    value (no ``.get()``/``AsyncResult`` anywhere; the status string is only for
    logs), so registering a pending result is pure waste. It also makes a
    dead-broker publish fail *precisely*: with a Redis result backend, plain
    ``.delay()`` on a down broker raises a vague ``RuntimeError`` from the result
    consumer's reconnect loop, whereas ignoring the result surfaces the broker
    connect failure itself as ``kombu.exceptions.OperationalError`` — the exact
    exception ``_publish_or_defer`` catches."""
    from voxint.worker.tasks import pipeline_task_for_stage

    pipeline_task_for_stage(stage).apply_async((str(run_id),), ignore_result=True)


def _publish_or_defer(run_id: uuid.UUID, *, stage: Stage | None = None) -> bool:
    """Publish the run's task, returning ``False`` (never raising) if the broker
    is unreachable so the request can degrade cleanly.

    Commit-before-publish means the durable QUEUED run already exists, so a Redis
    outage is non-fatal: leaving the run QUEUED lets the beat recovery sweep
    re-enqueue it once the broker returns. Only ``OperationalError`` — kombu's
    wrapper for every transport/connection failure — is swallowed, so a genuine
    bug in the publish path still raises rather than being silently deferred."""
    # celery re-exports kombu's OperationalError as the same class; importing it
    # from celery keeps the broker types under the existing celery.* mypy override
    # and stays lazy so the read path never pulls the broker in.
    from celery.exceptions import OperationalError

    try:
        _publish_run(run_id, stage=stage)
    except OperationalError:
        logger.warning(
            "pipeline enqueue deferred (broker unavailable); run %s stays QUEUED "
            "for the recovery sweep",
            run_id,
            exc_info=True,
        )
        return False
    return True



def _submit_domain_pack_detail(exc: DomainPackError) -> str:
    """Operator-facing text for a freeze-time ``DomainPackError`` at a submit boundary.

    ``_run_domain_pack_snapshot`` raises this when the run's resolved domain pack
    can't be applied — either an unresolvable pack name (issue #11) or an operator
    correction that collides with the folder's pack (issue #84). Neutral wording
    covers both, and the underlying reason is softened out of pack jargon so a
    non-technical operator can act on it instead of seeing a 500."""
    return (
        "The domain pack for this media couldn't be applied: "
        f"{operator_correction_message(str(exc))} "
        "Check Settings → Corrections and the folder's domain-pack assignment."
    )
