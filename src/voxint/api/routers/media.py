"""The media library area (Console 2.0 P2a/P2b, #153/#154): the file listing at
/media plus its upload and URL-fetch surface.

The listing (P2a) is one read-only page over :mod:`voxint.api.media_query`. P2b
moves file upload and URL ingestion onto this page (the legacy `/runs` forms stay
until P5), reusing the same broker-free ingest backends
(:func:`voxint.ingest.submit_upload` / :func:`voxint.ingest.submit_url`); each form
may pick a settings folder whose vocabulary/corrections apply to the run without
moving the bytes (ADR 0002 addendum). Every route is always registered so the
console route inventory is stable across the dark-ship flip; access is gated by
:func:`require_media_enabled`, which 404s until ``console_media_enabled`` is on.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Final

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response

from voxint.api.csrf import CSRF_MEDIA_FETCH, CSRF_MEDIA_SUBMIT, mint_csrf_token
from voxint.api.media_query import (
    DEFAULT_SORT,
    MEDIA_LIBRARY_LIMIT,
    SORT_LABELS,
    folder_options,
    media_library,
    sort_is_known,
)
from voxint.api.routers import deps
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    _require_csrf,
    require_media_enabled,
    require_onboarded,
    templates,
)
from voxint.app_settings import get_app_settings, resolve_effective_ytdlp_enabled
from voxint.config import Settings
from voxint.db.models import MediaFolder
from voxint.domain_packs.base import DomainPackError
from voxint.ingest import (
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    submit_upload,
    submit_url,
)

# require_onboarded first (an un-onboarded operator is sent to setup), then the
# area gate (404 when the flag is off) — the same order the module docstring and
# the projects area will follow.
router = APIRouter(
    dependencies=[Depends(require_onboarded), Depends(require_media_enabled)]
)

# The layout toggle: cards for scanning, a table for dense comparison. A ?view=
# outside this set degrades to the default rather than 422-ing (the Home ?window=
# convention), so a bookmarked value never breaks the page.
_VIEWS: Final[tuple[str, ...]] = ("cards", "table")
_DEFAULT_VIEW: Final[str] = "cards"


def _media_redirect(*, published: bool) -> RedirectResponse:
    """303 back to the library after an ingest, honest about broker deferral.

    ``published`` False means the durable QUEUED run exists but the enqueue was
    deferred (broker unavailable); the page says the recovery sweep will pick it
    up, mirroring the legacy ``/submit`` redirect's ``?enqueue=deferred``.
    """
    marker = "1" if published else "deferred"
    return RedirectResponse(f"/media?submitted={marker}", status_code=303)


def _resolve_picked_folder(
    session: SessionDep, raw: str
) -> uuid.UUID | None:
    """Validate the optional settings-folder pick BEFORE any ingest write.

    Empty (the "(no folder — global settings)" choice) resolves to ``None``. A
    non-empty value must be a real folder id: a malformed id or a folder that no
    longer exists (a stale form) is a 400 here, so it never reaches the ingest
    savepoint as an ambiguous IntegrityError. Never moves or reads bytes.
    """
    if not raw.strip():
        return None
    try:
        folder_id = uuid.UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unknown settings folder") from exc
    exists = session.get(MediaFolder, folder_id)
    if exists is None:
        raise HTTPException(status_code=400, detail="Unknown settings folder")
    return folder_id


@router.get("/media", name="media_library")
def media_library_page(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    sort: str | None = None,
    view: str | None = None,
    submitted: str | None = None,
) -> Response:
    settings: Settings = request.app.state.settings
    selected_sort = sort if sort_is_known(sort) else DEFAULT_SORT
    selected_view = view if view in _VIEWS else _DEFAULT_VIEW
    rows = media_library(session, sort=selected_sort)
    secret = request.app.state.csrf_secret
    context = {
        "request": request,
        "active_nav": "media",
        "now": datetime.now(UTC),
        "rows": rows,
        "sort": selected_sort,
        "sorts": SORT_LABELS,
        "view": selected_view,
        "views": _VIEWS,
        # The listing is capped; say so honestly when it is full rather than
        # implying the library ends here.
        "truncated": len(rows) >= MEDIA_LIBRARY_LIMIT,
        "limit": MEDIA_LIBRARY_LIMIT,
        # Upload / URL-fetch surface (P2b). Per-form server-issued ids namespace
        # each submission's path and make a double-submit idempotent; per-form
        # CSRF tokens are bound to their own media actions.
        "submission_id": uuid.uuid4().hex,
        "fetch_submission_id": uuid.uuid4().hex,
        "csrf_media_submit": mint_csrf_token(secret, CSRF_MEDIA_SUBMIT),
        "csrf_media_fetch": mint_csrf_token(secret, CSRF_MEDIA_FETCH),
        # The settings-folder picker; renders whether or not projects are enabled.
        "folder_options": folder_options(session),
        # Gate the URL-fetch form: off => rendered disabled and POST /media/fetch
        # also refuses 403, matching the legacy /runs form and the CLI.
        "ytdlp_enabled": resolve_effective_ytdlp_enabled(
            get_app_settings(session), settings
        ),
        # Post-redirect notice: "1" queued, "deferred" queued-but-broker-down.
        "submitted": submitted if submitted in ("1", "deferred") else None,
    }
    return templates.TemplateResponse(request, "media/media.html", context)


@router.post("/media/submit")
def media_submit_upload(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    submission_id: Annotated[str, Form()],
    media_folder_id: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    # CSRF before anything: a forged cross-site upload is refused before the DB
    # write / file finalize (mirrors the legacy POST /submit).
    _require_csrf(request, CSRF_MEDIA_SUBMIT, csrf_token)
    settings: Settings = request.app.state.settings
    folder_id = _resolve_picked_folder(session, media_folder_id)
    try:
        run = submit_upload(
            session,
            stream=file.file,
            filename=file.filename or "",
            submission_id=submission_id,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            media_folder_id=folder_id,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DomainPackError as exc:
        raise HTTPException(
            status_code=422, detail=deps._submit_domain_pack_detail(exc)
        ) from exc
    run_id = run.id
    # Commit-before-publish: the durable QUEUED run must exist before the enqueue,
    # so a broker outage is non-fatal (the recovery sweep republishes).
    session.commit()
    return _media_redirect(published=deps._publish_or_defer(run_id))


@router.post("/media/fetch")
def media_fetch_url(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    url: Annotated[str, Form()],
    submission_id: Annotated[str, Form()],
    media_folder_id: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    # CSRF first — reject a forged cross-site fetch before any other check, so a
    # forgery is refused regardless of the ytdlp_enabled flag's state.
    _require_csrf(request, CSRF_MEDIA_FETCH, csrf_token)
    settings: Settings = request.app.state.settings
    if not resolve_effective_ytdlp_enabled(get_app_settings(session), settings):
        # Generic message — never echo the submitted URL into an error body.
        raise HTTPException(status_code=403, detail="URL ingestion is disabled")
    folder_id = _resolve_picked_folder(session, media_folder_id)
    try:
        run = submit_url(
            session,
            url=url,
            submission_id=submission_id,
            media_folder_id=folder_id,
        )
    except (UrlValidationError, UploadValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DomainPackError as exc:
        raise HTTPException(
            status_code=422, detail=deps._submit_domain_pack_detail(exc)
        ) from exc
    run_id = run.id
    session.commit()
    return _media_redirect(published=deps._publish_or_defer(run_id))
