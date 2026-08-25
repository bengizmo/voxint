"""The media library area (Console 2.0 P2a/P2b, #153/#154): the file listing at
/media plus its upload, URL-fetch, and organization surface.

The listing (P2a) is one read-only page over :mod:`voxint.api.media_query`. P2b
moves file upload and URL ingestion onto this page (the legacy `/runs` forms stay
until P5), reusing the same broker-free ingest backends
(:func:`voxint.ingest.submit_upload` / :func:`voxint.ingest.submit_url`); each form
may pick a settings folder whose vocabulary/corrections apply to the run without
moving the bytes (ADR 0002 addendum). P2b also makes the library operable: a
multi-select drives a non-destructive bulk **assign** (set each file's settings
folder, including clearing it), and a folder panel registers/unregisters folders
through the shared write service. Every route is always registered so the console
route inventory is stable across the dark-ship flip; access is gated by
:func:`require_media_enabled`, which 404s until ``console_media_enabled`` is on.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Final
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from voxint.api.csrf import (
    CSRF_MEDIA_ASSIGN,
    CSRF_MEDIA_FETCH,
    CSRF_MEDIA_FOLDERS,
    CSRF_MEDIA_SUBMIT,
    mint_csrf_token,
)
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
from voxint.db.models import MediaFolder, MediaItem
from voxint.domain_packs.base import DomainPackError
from voxint.ingest import (
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    submit_upload,
    submit_url,
)
from voxint.media.registration import register_folder, unregister_folder_by_id

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


def _files(count: int) -> str:
    """"1 file" / "3 files" — a small honest pluralizer for notice copy."""
    return f"{count} file" if count == 1 else f"{count} files"


def _safe_sort(sort: str | None) -> str:
    return sort if sort_is_known(sort) else DEFAULT_SORT


def _safe_view(view: str | None) -> str:
    return view if view in _VIEWS else _DEFAULT_VIEW


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


def _success_notice(
    *, assigned: str | None, folder: str | None, reverted: str | None
) -> dict[str, str] | None:
    """Build the post-redirect success banner from the PRG query params.

    Bulk mutations redirect (POST-redirect-GET) so a refresh never re-submits; the
    outcome rides back as a small query param the GET turns into a banner. Only the
    shapes this router emits are honored, and a malformed/negative count is dropped
    rather than rendered.
    """
    if assigned is not None:
        try:
            count = int(assigned)
        except ValueError:
            return None
        if count < 0:
            return None
        return {
            "kind": "success",
            "text": f"Updated the settings folder for {_files(count)}.",
        }
    if folder == "added":
        return {"kind": "success", "text": "Folder registered."}
    if folder == "removed":
        text = "Folder unregistered."
        count = 0
        if reverted is not None:
            try:
                count = int(reverted)
            except ValueError:
                count = 0
        if count > 0:
            text += f" {_files(count)} reverted to global settings."
        return {"kind": "success", "text": text}
    return None


def _library_context(
    request: Request,
    session: SessionDep,
    settings: Settings,
    *,
    sort: str | None,
    view: str | None,
    submitted: str | None = None,
    notice: dict[str, str] | None = None,
    selected_ids: frozenset[str] = frozenset(),
    attempted_folder_id: str = "",
) -> dict[str, Any]:
    """The full /media render context, shared by the GET page and every error
    re-render.

    One builder so an error branch cannot ship a page missing a CSRF token or the
    folder options (a broken form or a 500). ``sort``/``view`` are coerced through
    the same allowlists the GET uses, so a crafted hidden field cannot skew the
    re-render. ``selected_ids``/``attempted_folder_id`` let a rejected bulk action
    re-check the operator's boxes and target instead of silently dropping them.
    """
    selected_sort = _safe_sort(sort)
    selected_view = _safe_view(view)
    rows = media_library(session, sort=selected_sort)
    secret = request.app.state.csrf_secret
    return {
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
        "csrf_media_assign": mint_csrf_token(secret, CSRF_MEDIA_ASSIGN),
        "csrf_media_folders": mint_csrf_token(secret, CSRF_MEDIA_FOLDERS),
        # The settings-folder picker; renders whether or not projects are enabled.
        "folder_options": folder_options(session),
        # Gate the URL-fetch form: off => rendered disabled and POST /media/fetch
        # also refuses 403, matching the legacy /runs form and the CLI.
        "ytdlp_enabled": resolve_effective_ytdlp_enabled(
            get_app_settings(session), settings
        ),
        # Post-redirect notice: "1" queued, "deferred" queued-but-broker-down.
        "submitted": submitted if submitted in ("1", "deferred") else None,
        # A success/error banner for the bulk-assign and folder-panel flows.
        "notice": notice,
        # Re-check the operator's selection + chosen target when an error
        # re-renders, so a rejected bulk action does not drop their work.
        "selected_ids": selected_ids,
        "attempted_folder_id": attempted_folder_id,
    }


@router.get("/media", name="media_library")
def media_library_page(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    sort: str | None = None,
    view: str | None = None,
    submitted: str | None = None,
    assigned: str | None = None,
    folder: str | None = None,
    reverted: str | None = None,
) -> Response:
    settings: Settings = request.app.state.settings
    notice = _success_notice(assigned=assigned, folder=folder, reverted=reverted)
    context = _library_context(
        request,
        session,
        settings,
        sort=sort,
        view=view,
        submitted=submitted,
        notice=notice,
    )
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


@router.post("/media/assign")
def media_assign(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    # Every field is optional/defaulted so a missing value never triggers
    # FastAPI's own 422 BEFORE _require_csrf runs (CSRF must gate first); the
    # handler validates each one itself.
    media_id: Annotated[list[str] | None, Form()] = None,
    media_folder_id: Annotated[str, Form()] = "",
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Bulk-set the settings folder over a selection (ADR 0002 addendum).

    Prevalidates the WHOLE selection before any write: malformed/empty/oversized
    selections and a stale target folder are rejected with zero writes and the
    page re-rendered (the selection and target preserved). Only
    ``media_folder_id`` is touched — never ``source_path``/``current_path``, the
    bytes, or any frozen run snapshot. An empty target clears membership to the
    global baseline. On success it PRG-redirects with an ``assigned=N`` banner.
    """
    _require_csrf(request, CSRF_MEDIA_ASSIGN, csrf_token)
    settings: Settings = request.app.state.settings
    raw_ids = media_id or []

    def _reject(status_code: int, text: str) -> Response:
        # No write has been issued yet, but roll back defensively so the
        # re-render's SELECTs run on a clean transaction.
        session.rollback()
        context = _library_context(
            request,
            session,
            settings,
            sort=sort,
            view=view,
            notice={"kind": "error", "text": text},
            selected_ids=frozenset(raw_ids),
            attempted_folder_id=media_folder_id,
        )
        return templates.TemplateResponse(
            request, "media/media.html", context, status_code=status_code
        )

    # Parse + dedup the selection (order preserved for a stable count-match).
    parsed: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in raw_ids:
        try:
            media_uuid = uuid.UUID(raw)
        except ValueError:
            return _reject(
                400, "That selection was not valid. Reload the page and try again."
            )
        if media_uuid not in seen:
            seen.add(media_uuid)
            parsed.append(media_uuid)
    if not parsed:
        return _reject(400, "Select at least one file first.")
    if len(parsed) > MEDIA_LIBRARY_LIMIT:
        # Reject rather than silently truncate: assigning a misleading subset of
        # the selection would be dishonest. The page shows at most this many rows,
        # so this only bites a forged/oversized request.
        return _reject(400, f"Select at most {MEDIA_LIBRARY_LIMIT} files at once.")

    # Validate the target settings folder (empty = clear to global settings).
    target: uuid.UUID | None = None
    raw_folder = media_folder_id.strip()
    if raw_folder:
        try:
            target = uuid.UUID(raw_folder)
        except ValueError:
            return _reject(
                400, "That settings folder no longer exists. Nothing was changed."
            )
        if session.get(MediaFolder, target) is None:
            return _reject(
                400, "That settings folder no longer exists. Nothing was changed."
            )

    # Load the whole selection; a count mismatch means a selected file was deleted
    # since the page rendered — reject with zero writes.
    items = list(
        session.execute(select(MediaItem).where(MediaItem.id.in_(parsed))).scalars()
    )
    if len(items) != len(parsed):
        return _reject(
            409,
            "Some selected files no longer exist. Nothing was changed; reload "
            "and try again.",
        )

    for item in items:
        item.media_folder_id = target
    try:
        session.commit()
    except IntegrityError:
        # The target folder was unregistered between validation and commit (a
        # concurrent tab). Surface it cleanly instead of a raw 500.
        return _reject(
            409,
            "That settings folder was removed while you were assigning. Nothing "
            "was changed.",
        )

    query = urlencode(
        {"assigned": len(items), "sort": _safe_sort(sort), "view": _safe_view(view)}
    )
    return RedirectResponse(f"/media?{query}", status_code=303)


@router.post("/media/folders")
def media_folders(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    action: Annotated[str, Form()] = "",
    folder: Annotated[str, Form()] = "",
    folder_id: Annotated[str, Form()] = "",
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Register (``action=add``) or unregister (``action=remove``) a folder.

    One action-field route, mirroring the Settings folder panel. Add takes the
    operator's raw MEDIA_ROOT-relative path through the shared
    :func:`register_folder` (resolve + containment validated there); a validation
    message re-renders the page rather than throwing. Remove names the row by id
    (the panel renders ids from ``folder_options``), so it never depends on
    reconstructing the exact stored path, and reports how many media the delete
    reverts to global settings. Neither touches the filesystem.
    """
    _require_csrf(request, CSRF_MEDIA_FOLDERS, csrf_token)
    settings: Settings = request.app.state.settings

    def _reject(status_code: int, text: str) -> Response:
        session.rollback()
        context = _library_context(
            request,
            session,
            settings,
            sort=sort,
            view=view,
            notice={"kind": "error", "text": text},
        )
        return templates.TemplateResponse(
            request, "media/media.html", context, status_code=status_code
        )

    if action == "add":
        error = register_folder(session, settings, folder)
        if error is not None:
            return _reject(400, error)
        session.commit()
        query = urlencode(
            {"folder": "added", "sort": _safe_sort(sort), "view": _safe_view(view)}
        )
        return RedirectResponse(f"/media?{query}", status_code=303)

    if action == "remove":
        # A malformed/missing id is treated as already-gone (idempotent): the
        # panel simply refreshes rather than erroring.
        fid: uuid.UUID | None
        try:
            fid = uuid.UUID(folder_id)
        except ValueError:
            fid = None
        reverted = 0
        if fid is not None:
            _removed, reverted = unregister_folder_by_id(session, fid)
        session.commit()
        query = urlencode(
            {
                "folder": "removed",
                "reverted": reverted,
                "sort": _safe_sort(sort),
                "view": _safe_view(view),
            }
        )
        return RedirectResponse(f"/media?{query}", status_code=303)

    # An unknown verb is a malformed request (stale form / typo), never a silent
    # no-op — the same posture as the Settings folder route.
    raise HTTPException(status_code=422, detail=f"unknown folder action {action!r}")
