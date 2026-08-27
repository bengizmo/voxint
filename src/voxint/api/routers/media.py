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

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Final
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import StaleDataError

from voxint.api.csrf import (
    CSRF_MEDIA_ARCHIVE,
    CSRF_MEDIA_ASSIGN,
    CSRF_MEDIA_EMPTY_TRASH,
    CSRF_MEDIA_FETCH,
    CSRF_MEDIA_FOLDERS,
    CSRF_MEDIA_RERUN,
    CSRF_MEDIA_RERUN_CONFIRM,
    CSRF_MEDIA_RESTORE,
    CSRF_MEDIA_SUBMIT,
    CSRF_MEDIA_TRASH,
    CSRF_MEDIA_UNARCHIVE,
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
from voxint.api.presentation import friendly_media_label
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
from voxint.db.models import (
    MediaFolder,
    MediaItem,
    MediaOperation,
    OperationState,
    OperationType,
    PipelineRun,
)
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import operator_correction_message
from voxint.domain_packs.registry import resolve_domain_pack_by_name
from voxint.ingest import (
    EffectiveConfigPreview,
    RunNotArchivableError,
    RunNotFoundError,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    archive_run,
    preview_effective_config,
    submit_media_item,
    submit_upload,
    submit_url,
    unarchive_run,
)
from voxint.ingest.sidecar import Sidecar, SidecarError, find_sidecar, read_sidecar
from voxint.media.executor import (
    execute_operation,
    plan_restore,
    plan_trash,
)
from voxint.media.operations import OperationRefused
from voxint.media.purge import build_manifest, execute_purge, plan_purge
from voxint.media.registration import register_folder, unregister_folder_by_id

# require_onboarded first (an un-onboarded operator is sent to setup), then the
# area gate (404 when the flag is off) — the same order the module docstring and
# the projects area will follow.
logger = logging.getLogger(__name__)

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


class _SelectionError(Exception):
    """A bulk selection that must be rejected with zero writes.

    Carries the HTTP ``status_code`` and the operator-facing ``text`` so each
    bulk route can catch it and re-render the page (or the confirm page) with the
    same prevalidation discipline — malformed/empty/oversized selections and
    files that vanished since the page rendered all reject before any mutation.
    """

    def __init__(self, status_code: int, text: str) -> None:
        super().__init__(text)
        self.status_code = status_code
        self.text = text


def _parse_media_selection(raw_ids: list[str]) -> list[uuid.UUID]:
    """Parse + dedup a submitted selection, order preserved, or raise.

    Shared by every bulk route so they reject a crafted selection identically:
    a malformed id is a 400, an empty selection is a 400, and more than
    :data:`MEDIA_LIBRARY_LIMIT` ids is a 400 (reject, never silently truncate —
    acting on a misleading subset would be dishonest, and the page never shows
    more than the cap, so this only bites a forged request). Order is preserved
    so a later count-match against a stable load is deterministic.
    """
    parsed: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in raw_ids:
        try:
            media_uuid = uuid.UUID(raw)
        except ValueError as exc:
            raise _SelectionError(
                400, "That selection was not valid. Reload the page and try again."
            ) from exc
        if media_uuid not in seen:
            seen.add(media_uuid)
            parsed.append(media_uuid)
    if not parsed:
        raise _SelectionError(400, "Select at least one file first.")
    if len(parsed) > MEDIA_LIBRARY_LIMIT:
        raise _SelectionError(
            400, f"Select at most {MEDIA_LIBRARY_LIMIT} files at once."
        )
    return parsed


def _load_selection(
    session: SessionDep, parsed: list[uuid.UUID], *, lock: bool = False
) -> list[MediaItem]:
    """Load every selected :class:`MediaItem`, or raise on a count mismatch.

    A mismatch means a selected file was deleted since the page rendered — a 409
    with zero writes, never a partial action on a stale selection. ``lock`` takes
    a ``FOR UPDATE`` row lock in id-sorted order (the confirm path), so two
    overlapping bulk confirms serialize on the shared rows in a deadlock-free
    order rather than racing to mint duplicate runs.
    """
    stmt = select(MediaItem).where(MediaItem.id.in_(parsed))
    if lock:
        # Sorted, so overlapping confirms acquire the shared rows in one global
        # order (no lock-cycle deadlock); the recheck against the preview baseline
        # then serializes behind whichever confirm commits first.
        stmt = stmt.order_by(MediaItem.id).with_for_update()
    items = list(session.execute(stmt).scalars())
    if len(items) != len(parsed):
        raise _SelectionError(
            409,
            "Some selected files no longer exist. Nothing was changed; reload "
            "and try again.",
        )
    return items


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


def _nonneg(raw: str | None) -> int | None:
    """Parse a non-negative int PRG count, or ``None`` for absent/malformed/negative.

    Every success banner rides back as a query param a crafted URL could forge, so
    a value that is not a plain ``>= 0`` integer is dropped rather than rendered.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _success_notice(
    *,
    assigned: str | None,
    folder: str | None,
    reverted: str | None,
    archive_done: str | None = None,
    unarchive_done: str | None = None,
    trash_done: str | None = None,
    restore_done: str | None = None,
    empty_trash_done: str | None = None,
    skipped: str | None = None,
    live_skipped: str | None = None,
) -> dict[str, str] | None:
    """Build the post-redirect success banner from the PRG query params.

    Bulk mutations redirect (POST-redirect-GET) so a refresh never re-submits; the
    outcome rides back as a small query param the GET turns into a banner. Only the
    shapes this router emits are honored, and a malformed/negative count is dropped
    rather than rendered.
    """
    assigned_n = _nonneg(assigned)
    if assigned_n is not None:
        return {
            "kind": "success",
            "text": f"Updated the settings folder for {_files(assigned_n)}.",
        }
    if folder == "added":
        return {"kind": "success", "text": "Folder registered."}
    if folder == "removed":
        text = "Folder unregistered."
        count = _nonneg(reverted) or 0
        if count > 0:
            text += f" {_files(count)} reverted to global settings."
        return {"kind": "success", "text": text}
    archived_n = _nonneg(archive_done)
    if archived_n is not None:
        text = f"Archived the latest run for {_files(archived_n)}."
        skip = _nonneg(skipped) or 0
        if skip > 0:
            # Skip, not abort: a file whose shown run had already been archived or
            # had changed since the page loaded is reported, never silently dropped.
            text += (
                f" {_files(skip)} skipped (already archived or changed since you "
                "loaded the page)."
            )
        live = _nonneg(live_skipped) or 0
        if live > 0:
            # A live latest run cannot be archived — say the action needed, rather
            # than folding it into the generic skip count (honest UX).
            text += (
                f" {_files(live)} skipped because a run is still in progress; "
                "cancel it first."
            )
        return {"kind": "success", "text": text}
    restored_n = _nonneg(unarchive_done)
    if restored_n is not None:
        text = f"Restored the latest archived run for {_files(restored_n)}."
        skip = _nonneg(skipped) or 0
        if skip > 0:
            text += (
                f" {_files(skip)} skipped (already restored or changed since you "
                "loaded the page)."
            )
        return {"kind": "success", "text": text}
    trash_n = _nonneg(trash_done)
    if trash_n is not None:
        text = f"Moved {_files(trash_n)} to trash."
        skip = _nonneg(skipped) or 0
        if skip > 0:
            text += f" {_files(skip)} skipped."
        return {"kind": "success", "text": text}
    restore_n = _nonneg(restore_done)
    if restore_n is not None:
        text = f"Restored {_files(restore_n)} from trash."
        skip = _nonneg(skipped) or 0
        if skip > 0:
            text += f" {_files(skip)} skipped."
        return {"kind": "success", "text": text}
    purged_n = _nonneg(empty_trash_done)
    if purged_n is not None:
        text = f"Permanently deleted {_files(purged_n)}."
        skip = _nonneg(skipped) or 0
        if skip > 0:
            text += f" {_files(skip)} skipped."
        return {"kind": "success", "text": text}
    return None


def _library_context(
    request: Request,
    session: SessionDep,
    settings: Settings,
    *,
    sort: str | None,
    view: str | None,
    archived: bool = False,
    trashed: bool = False,
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
    re-render. ``archived`` picks the active vs archived view (the latter shows only
    items whose latest run is archived, the target set for bulk unarchive).
    ``trashed`` selects the trash item view and is mutually exclusive with archived
    at the GET route. ``selected_ids``/``attempted_folder_id`` let a rejected bulk
    action re-check the operator's boxes and target instead of silently dropping
    them.
    """
    selected_sort = _safe_sort(sort)
    selected_view = _safe_view(view)
    rows = media_library(
        session, sort=selected_sort, archived=archived, trashed=trashed
    )
    secret = request.app.state.csrf_secret
    # The archive-view toggle target: the same sort/layout in the opposite view.
    # Active -> add ?archived=1; archived -> drop it (mirrors the /runs toggle).
    toggle_params = {"sort": selected_sort, "view": selected_view}
    if not archived:
        toggle_params["archived"] = "1"
    archived_toggle_url = f"/media?{urlencode(toggle_params)}"
    trash_params = {"sort": selected_sort, "view": selected_view}
    if not trashed:
        trash_params["trashed"] = "1"
    trash_toggle_url = f"/media?{urlencode(trash_params)}"
    media_root = settings.media_root
    missing: set[uuid.UUID] = set()
    for row in rows:
        path = row.current_path if row.current_path is not None else row.source_path
        resolved = media_root / path
        try:
            if not resolved.resolve().is_file():
                missing.add(row.id)
        except (OSError, RuntimeError):
            missing.add(row.id)
    missing_file_ids = frozenset(missing)
    return {
        "request": request,
        "active_nav": "media",
        "now": datetime.now(UTC),
        "rows": rows,
        "sort": selected_sort,
        "sorts": SORT_LABELS,
        "view": selected_view,
        "views": _VIEWS,
        # The archived view shows only items with an archived latest run (the bulk
        # unarchive target set); the active view hides archived runs.
        "archived": archived,
        "trashed": trashed,
        "archived_toggle_url": archived_toggle_url,
        "trash_toggle_url": trash_toggle_url,
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
        "csrf_media_rerun": mint_csrf_token(secret, CSRF_MEDIA_RERUN),
        "csrf_media_archive": mint_csrf_token(secret, CSRF_MEDIA_ARCHIVE),
        "csrf_media_unarchive": mint_csrf_token(secret, CSRF_MEDIA_UNARCHIVE),
        "csrf_media_trash": mint_csrf_token(secret, CSRF_MEDIA_TRASH),
        "csrf_media_restore": mint_csrf_token(secret, CSRF_MEDIA_RESTORE),
        "csrf_media_empty_trash": mint_csrf_token(secret, CSRF_MEDIA_EMPTY_TRASH),
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
        "missing_file_ids": missing_file_ids,
    }


@router.get("/media", name="media_library")
def media_library_page(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    sort: str | None = None,
    view: str | None = None,
    archived: str | None = None,
    trashed: str | None = None,
    submitted: str | None = None,
    assigned: str | None = None,
    folder: str | None = None,
    reverted: str | None = None,
    archive_done: str | None = None,
    unarchive_done: str | None = None,
    trash_done: str | None = None,
    restore_done: str | None = None,
    empty_trash_done: str | None = None,
    skipped: str | None = None,
    live_skipped: str | None = None,
) -> Response:
    settings: Settings = request.app.state.settings
    # ?archived=1 flips to the archived-only view (mirrors /runs); anything else
    # (absent / "0" / blank) is the default active listing that hides archived runs.
    show_trashed = trashed == "1"
    show_archived = archived == "1" and not show_trashed
    notice = _success_notice(
        assigned=assigned,
        folder=folder,
        reverted=reverted,
        archive_done=archive_done,
        unarchive_done=unarchive_done,
        trash_done=trash_done,
        restore_done=restore_done,
        empty_trash_done=empty_trash_done,
        skipped=skipped,
        live_skipped=live_skipped,
    )
    context = _library_context(
        request,
        session,
        settings,
        sort=sort,
        view=view,
        archived=show_archived,
        trashed=show_trashed,
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
    except IntegrityError as exc:
        # The picked settings folder was unregistered between the pre-check and the
        # insert (a concurrent tab): fail cleanly rather than a raw 500.
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="The selected settings folder was removed. Reload and try again.",
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
    except IntegrityError as exc:
        # The picked settings folder was unregistered between the pre-check and the
        # insert (a concurrent tab): fail cleanly rather than a raw 500.
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="The selected settings folder was removed. Reload and try again.",
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

    try:
        parsed = _parse_media_selection(raw_ids)
    except _SelectionError as exc:
        return _reject(exc.status_code, exc.text)

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
    try:
        items = _load_selection(session, parsed)
    except _SelectionError as exc:
        return _reject(exc.status_code, exc.text)

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


# The baseline sentinel for a media item that had no run when the operator
# previewed — carried as a hidden field so the confirm step distinguishes "no run
# at preview" from "a run existed", and detects a run that appeared in between.
_NO_RUN_BASELINE: Final[str] = "none"


def _baseline_str(run_id: uuid.UUID | None) -> str:
    """The wire form of a latest-run baseline: the run id, or the no-run sentinel."""
    return _NO_RUN_BASELINE if run_id is None else str(run_id)


def _latest_run_ids(
    session: SessionDep, media_ids: list[uuid.UUID]
) -> dict[uuid.UUID, uuid.UUID]:
    """Each media item's most-recent run id across ALL runs (any archive state).

    Keyed by media id; an item with no run at all is simply absent. This is the
    double-submit baseline for bulk re-run: a re-run mints a fresh non-archived
    run, so a latest-run id that changed between preview and confirm means a run
    appeared in between, and that item is skipped. Archiving a run mutates a
    column, not the row's recency, so this id is stable under a concurrent archive
    and moves only when a NEW run is created — exactly the event we guard against.
    """
    ranked = (
        select(
            PipelineRun.media_item_id.label("media_item_id"),
            PipelineRun.id.label("run_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rn"),
        )
        .where(PipelineRun.media_item_id.in_(media_ids))
        .subquery()
    )
    stmt = select(ranked.c.media_item_id, ranked.c.run_id).where(ranked.c.rn == 1)
    return {row.media_item_id: row.run_id for row in session.execute(stmt)}


def _latest_run_in_view(
    session: SessionDep, media_ids: list[uuid.UUID], *, archived: bool
) -> dict[uuid.UUID, uuid.UUID]:
    """Each media item's latest run WITHIN one archive view, keyed by media id.

    ``archived`` False resolves the latest NON-archived run (the active view's
    archive target); True resolves the latest ARCHIVED run (the archived view's
    unarchive target). This is the exact run the library shows for the item in that
    view — the same window :func:`media_library` uses — so a bulk archive/unarchive
    acts on precisely what the operator sees. An item with no run in that view is
    simply absent (a skip for the caller). Unlike :func:`_latest_run_ids` (which
    spans both archive states for the re-run double-submit baseline), this is
    view-scoped.
    """
    run_view = (
        PipelineRun.archived_at.is_not(None)
        if archived
        else PipelineRun.archived_at.is_(None)
    )
    ranked = (
        select(
            PipelineRun.media_item_id.label("media_item_id"),
            PipelineRun.id.label("run_id"),
            func.row_number()
            .over(
                partition_by=PipelineRun.media_item_id,
                order_by=(PipelineRun.created_at.desc(), PipelineRun.id.desc()),
            )
            .label("rn"),
        )
        .where(PipelineRun.media_item_id.in_(media_ids), run_view)
        .subquery()
    )
    stmt = select(ranked.c.media_item_id, ranked.c.run_id).where(ranked.c.rn == 1)
    return {row.media_item_id: row.run_id for row in session.execute(stmt)}


def _reread_sidecar(source_path: str, settings: Settings) -> Sidecar | None:
    """Re-read a media file's on-disk YAML sidecar for a fresh run, or ``None``.

    A re-run re-resolves current config and re-applies the sidecar sitting next to
    the source today; it deliberately does NOT carry the prior run's frozen
    sidecar, ``operator_notes``, or manual speaker-count hints (those belong to
    the old run). ``None`` means the file has no sidecar (submit plain). Raises
    :class:`SidecarError` when a sidecar exists but cannot be read/parsed or names
    an unresolvable pack, so the caller skips that one item honestly instead of
    minting a run against a half-read or broken sidecar. Never moves or rewrites
    bytes — only the paired ``.yaml`` is read (ADR 0002; AC-3).
    """
    # Defence-in-depth: source_path is a system-generated relative identity, but a
    # corrupted/absolute value would let `/` discard media_root and read outside it.
    # Resolve and confine; a path that escapes is treated as no sidecar (skip).
    root = settings.media_root.resolve()
    media_path = (root / source_path).resolve()
    if not media_path.is_relative_to(root):
        return None
    sidecar_path = find_sidecar(media_path)
    if sidecar_path is None:
        return None
    parsed = read_sidecar(sidecar_path)
    if parsed.domain_pack is not None:
        # Attribute an unknown pack to the SIDECAR (fix that file), matching the
        # watch sweep's classification, so it becomes a per-item skip rather than
        # a generic config failure that would look like a folder/global problem.
        try:
            resolve_domain_pack_by_name(parsed.domain_pack, settings)
        except DomainPackError as exc:
            raise SidecarError(f"{sidecar_path.name}: {exc}") from exc
    return parsed


@router.post("/media/rerun")
def media_rerun(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    # Optional/defaulted so a missing value never 422s before _require_csrf runs.
    media_id: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Advisory preview of a bulk re-run — resolves config, mutates NOTHING.

    Prevalidates the whole selection (the same parse/dedup/cap/count-match
    discipline as :func:`media_assign`), then for each file resolves the config a
    fresh run WOULD freeze right now (:func:`preview_effective_config`) and
    captures its current latest-run id as a double-submit baseline. It renders a
    confirm page carrying that baseline per item; it never creates a run. The
    preview is ADVISORY: config is re-resolved at confirm time (a separate READ
    COMMITTED txn), so the minted run reflects confirm-time config, and the page
    says so. A file whose config or sidecar can't be resolved is shown flagged so
    the operator sees, before confirming, that it will be skipped.
    """
    _require_csrf(request, CSRF_MEDIA_RERUN, csrf_token)
    settings: Settings = request.app.state.settings
    raw_ids = media_id or []

    def _reject(status_code: int, text: str) -> Response:
        session.rollback()
        context = _library_context(
            request,
            session,
            settings,
            sort=sort,
            view=view,
            notice={"kind": "error", "text": text},
            selected_ids=frozenset(raw_ids),
        )
        return templates.TemplateResponse(
            request, "media/media.html", context, status_code=status_code
        )

    try:
        parsed = _parse_media_selection(raw_ids)
        items = _load_selection(session, parsed)
    except _SelectionError as exc:
        return _reject(exc.status_code, exc.text)

    baselines = _latest_run_ids(session, parsed)
    by_id = {item.id: item for item in items}
    previews: list[dict[str, Any]] = []
    for media_uuid in parsed:  # render in the operator's selection order
        item = by_id[media_uuid]
        issue: str | None = None
        preview: EffectiveConfigPreview | None = None
        try:
            preview = preview_effective_config(
                session, item.media_folder_id, settings=settings
            )
        except DomainPackError as exc:
            issue = (
                "This file's settings folder has a configuration problem: "
                f"{operator_correction_message(str(exc))}"
            )
        if issue is None:
            # Read-only sidecar pre-check so a broken sidecar is visible BEFORE the
            # operator confirms (it becomes a skip at confirm, not a surprise).
            try:
                _reread_sidecar(item.source_path, settings)
            except SidecarError as exc:
                issue = f"Its sidecar could not be read: {exc}"
        baseline_run_id = baselines.get(item.id)
        previews.append(
            {
                "id": item.id,
                "label": friendly_media_label(None, item.source_path),
                "source_path": item.source_path,
                # media_id:baseline — one field carries both so the pair can never
                # be split apart or mis-ordered on the way to confirm.
                "pair": f"{item.id}:{_baseline_str(baseline_run_id)}",
                "has_prior_run": baseline_run_id is not None,
                "preview": preview,
                "issue": issue,
            }
        )

    secret = request.app.state.csrf_secret
    context = {
        "request": request,
        "active_nav": "media",
        "previews": previews,
        "runnable": sum(1 for p in previews if p["issue"] is None),
        "sort": _safe_sort(sort),
        "view": _safe_view(view),
        "csrf_media_rerun_confirm": mint_csrf_token(secret, CSRF_MEDIA_RERUN_CONFIRM),
    }
    return templates.TemplateResponse(request, "media/rerun_confirm.html", context)


@router.post("/media/rerun/confirm")
def media_rerun_confirm(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    # media_id:baseline pairs from the confirm page; optional/defaulted so a
    # missing value never 422s before _require_csrf runs.
    item: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Atomically mint a fresh run per selected file, then report per item.

    Row-locks the selected media in id-sorted order, re-verifies each file's
    latest-run baseline from the preview, and SKIPS any file that gained a newer
    run since preview (so a double-confirm mints at most one run per file). Each
    surviving file gets a fresh run via :func:`submit_media_item`, which
    re-resolves config off the file's STORED setting-folder membership (AC-2) and
    re-reads its on-disk sidecar; a broken sidecar or an unresolvable pack skips
    that one file honestly. Every run is created in ONE transaction and committed
    once (any unexpected pre-commit failure rolls back to zero new runs), then
    each is published commit-before-publish. Renders a per-item result summary.
    """
    _require_csrf(request, CSRF_MEDIA_RERUN_CONFIRM, csrf_token)
    settings: Settings = request.app.state.settings
    raw_items = item or []

    def _reject(status_code: int, text: str) -> Response:
        session.rollback()
        selected = frozenset(entry.partition(":")[0] for entry in raw_items)
        context = _library_context(
            request,
            session,
            settings,
            sort=sort,
            view=view,
            notice={"kind": "error", "text": text},
            selected_ids=selected,
        )
        return templates.TemplateResponse(
            request, "media/media.html", context, status_code=status_code
        )

    # Split each "media_id:baseline" pair, deduping by media id (first wins) with
    # the same reject discipline as _parse_media_selection.
    media_raw: list[str] = []
    baseline_by_id: dict[uuid.UUID, str] = {}
    for entry in raw_items:
        media_str, sep, baseline = entry.partition(":")
        if not sep:
            return _reject(
                400, "That selection was not valid. Reload the page and try again."
            )
        try:
            media_uuid = uuid.UUID(media_str)
        except ValueError:
            return _reject(
                400, "That selection was not valid. Reload the page and try again."
            )
        if media_uuid not in baseline_by_id:
            baseline_by_id[media_uuid] = baseline
            media_raw.append(media_str)

    try:
        parsed = _parse_media_selection(media_raw)
        # Lock in sorted order so overlapping confirms serialize deadlock-free.
        items = _load_selection(session, parsed, lock=True)
    except _SelectionError as exc:
        return _reject(exc.status_code, exc.text)

    # Re-read the current latest-run ids under the lock: a file whose latest run
    # changed since preview gained a run in between and is skipped.
    current_latest = _latest_run_ids(session, parsed)
    by_id = {item_row.id: item_row for item_row in items}

    results: list[dict[str, Any]] = []
    minted: list[uuid.UUID] = []
    for media_uuid in parsed:  # process + report in the operator's selection order
        media = by_id[media_uuid]
        label = friendly_media_label(None, media.source_path)
        if _baseline_str(current_latest.get(media_uuid)) != baseline_by_id[media_uuid]:
            results.append(
                {
                    "label": label,
                    "status": "skipped",
                    "reason": "a newer run appeared since you previewed it",
                }
            )
            continue
        try:
            sidecar = _reread_sidecar(media.source_path, settings)
        except SidecarError as exc:
            results.append(
                {
                    "label": label,
                    "status": "skipped",
                    "reason": f"its sidecar could not be read ({exc})",
                }
            )
            continue
        try:
            run = submit_media_item(
                session, media.source_path, settings=settings, sidecar=sidecar
            )
        except DomainPackError as exc:
            # A folder/global config problem — a plain Python error, so the txn is
            # still usable; skip this one file and keep going.
            results.append(
                {
                    "label": label,
                    "status": "skipped",
                    "reason": (
                        "its settings couldn't be applied "
                        f"({operator_correction_message(str(exc))})"
                    ),
                }
            )
            continue
        minted.append(run.id)
        results.append({"label": label, "status": "queued", "run_id": run.id})

    try:
        session.commit()
    except IntegrityError:
        # A concurrent delete/config change tripped a constraint on commit: nothing
        # is created (atomic), so report zero runs rather than a raw 500.
        return _reject(
            409,
            "Something changed while the runs were being queued. Nothing was "
            "started; reload and try again.",
        )

    # Commit-before-publish: the durable QUEUED runs exist, so a broker outage only
    # defers the enqueue (the recovery sweep republishes). Annotate each queued row.
    published_ids = {run_id: deps._publish_or_defer(run_id) for run_id in minted}
    for result in results:
        if result["status"] == "queued":
            result["published"] = published_ids.get(result["run_id"], False)

    context = {
        "request": request,
        "active_nav": "media",
        "results": results,
        "queued": len(minted),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "any_deferred": any(
            r["status"] == "queued" and not r.get("published") for r in results
        ),
        "sort": _safe_sort(sort),
        "view": _safe_view(view),
    }
    return templates.TemplateResponse(request, "media/rerun_result.html", context)


def _bulk_set_archived(
    request: Request,
    session: SessionDep,
    settings: Settings,
    *,
    media_id: list[str] | None,
    run_baseline: list[str] | None,
    sort: str,
    view: str,
    archiving: bool,
) -> Response:
    """Bulk archive (``archiving=True``) or unarchive the selection's latest run.

    Archive acts on each file's latest NON-archived run (the active view, where the
    button lives); unarchive on each file's latest ARCHIVED run (the archived view).
    Prevalidates the whole selection with the shared helpers (malformed/empty/
    oversized/deleted-since-render all reject with zero writes). Each selected row
    carries its render-time latest-run id as a ``media_id:run_id`` baseline (the same
    wire idiom as bulk re-run's confirm), so the action targets exactly the run the
    operator saw. A replay finds that run already archived — the latest-in-view has
    moved on — so the compare drifts and the item is SKIPPED, making a double-submit
    idempotent (archiving is otherwise per-run idempotent but the *target* would
    otherwise slide to the next-older run). Skip-not-abort mirrors re-run: a file
    with no run in the view, a drifted/already-acted baseline, or (archiving) a live
    latest run that is not archivable is a counted skip reported honestly in the
    banner; only an unexpected pre-commit failure rolls back to zero. On success it
    PRG-redirects to the acted-on view with counts.
    """
    raw_ids = media_id or []
    # Unarchive works over the archived view; archive over the active view. The
    # error re-render must show the same view the operator acted from.
    view_archived = not archiving

    def _reject(status_code: int, text: str) -> Response:
        session.rollback()
        context = _library_context(
            request,
            session,
            settings,
            sort=sort,
            view=view,
            archived=view_archived,
            notice={"kind": "error", "text": text},
            selected_ids=frozenset(raw_ids),
        )
        return templates.TemplateResponse(
            request, "media/media.html", context, status_code=status_code
        )

    try:
        parsed = _parse_media_selection(raw_ids)
        # The rows themselves are not needed (we act on runs, resolved by media id),
        # but the count-match guard rejects a selection whose file vanished.
        _load_selection(session, parsed)
    except _SelectionError as exc:
        return _reject(exc.status_code, exc.text)

    # Each selected file must carry its render-time latest-run baseline (the page
    # emits one hidden pair per row); a checked id without one is a stale/forged
    # form and rejects the whole request, matching the confirm route's discipline.
    baseline_by_id: dict[uuid.UUID, str] = {}
    for entry in run_baseline or []:
        media_str, sep, baseline = entry.partition(":")
        if not sep:
            continue
        try:
            mu = uuid.UUID(media_str)
        except ValueError:
            continue
        baseline_by_id.setdefault(mu, baseline)
    if any(mu not in baseline_by_id for mu in parsed):
        return _reject(
            400, "That selection was not valid. Reload the page and try again."
        )

    current = _latest_run_in_view(session, parsed, archived=view_archived)
    done = 0
    skipped = 0
    live_skipped = 0
    for media_uuid in parsed:
        run_id = current.get(media_uuid)
        # Skip on drift: the run the operator saw is no longer the latest in this
        # view (already archived on a replay, restored, or a newer run appeared), or
        # the file has no run in the view. This targets exactly the previewed run.
        if run_id is None or _baseline_str(run_id) != baseline_by_id[media_uuid]:
            skipped += 1
            continue
        try:
            if archiving:
                archive_run(session, run_id)
            else:
                unarchive_run(session, run_id)
        except RunNotArchivableError:
            # A live latest run (only archive hits this): cancel it first. Counted
            # apart from a plain skip so the banner can say what action is needed.
            live_skipped += 1
            continue
        except RunNotFoundError:
            # The run was deleted between resolve and act (a rare concurrent admin
            # delete): honestly skip it rather than 500 the whole action.
            skipped += 1
            continue
        done += 1

    try:
        session.commit()
    except (IntegrityError, StaleDataError):
        # A concurrent delete tripped a constraint (IntegrityError) or the run row
        # vanished between resolve and flush (StaleDataError's 0-row UPDATE): nothing
        # is committed (atomic), so report a clean 409 rather than a raw 500.
        return _reject(
            409,
            "Something changed while updating. Nothing was changed; reload and try "
            "again.",
        )

    params: dict[str, Any] = {"sort": _safe_sort(sort), "view": _safe_view(view)}
    if archiving:
        params["archive_done"] = done
    else:
        # Stay in the archived view so the operator sees the restored items leave it.
        params["archived"] = "1"
        params["unarchive_done"] = done
    if skipped:
        params["skipped"] = skipped
    if live_skipped:
        params["live_skipped"] = live_skipped
    return RedirectResponse(f"/media?{urlencode(params)}", status_code=303)


@router.post("/media/archive")
def media_archive(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    # Optional/defaulted so a missing value never 422s before _require_csrf runs.
    media_id: Annotated[list[str] | None, Form()] = None,
    # Per-row "media_id:run_id" render-time baselines (idempotent replay).
    run_baseline: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Bulk-archive the latest terminal run of each selected file (issue #5 archive,
    reversible). Hides those runs from the active library and the review queue while
    keeping every row intact; a file whose latest run is live or absent is skipped,
    not failed. Re-run's per-run archive routes stay for single runs (ADR 0002)."""
    _require_csrf(request, CSRF_MEDIA_ARCHIVE, csrf_token)
    settings: Settings = request.app.state.settings
    return _bulk_set_archived(
        request,
        session,
        settings,
        media_id=media_id,
        run_baseline=run_baseline,
        sort=sort,
        view=view,
        archiving=True,
    )


@router.post("/media/unarchive")
def media_unarchive(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    media_id: Annotated[list[str] | None, Form()] = None,
    # Per-row "media_id:run_id" render-time baselines (idempotent replay).
    run_baseline: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Bulk-restore the latest archived run of each selected file — it reappears in
    the active library and the review queue. Reached from the archived view, where
    every shown file has an archived run; a file with none is skipped. Idempotent."""
    _require_csrf(request, CSRF_MEDIA_UNARCHIVE, csrf_token)
    settings: Settings = request.app.state.settings
    return _bulk_set_archived(
        request,
        session,
        settings,
        media_id=media_id,
        run_baseline=run_baseline,
        sort=sort,
        view=view,
        archiving=False,
    )


def _reject_media_operation(
    request: Request,
    session: SessionDep,
    settings: Settings,
    *,
    status_code: int,
    text: str,
    sort: str,
    view: str,
    trashed: bool,
    selected_ids: list[str] | None = None,
) -> Response:
    """Roll back and re-render the originating media-operation view."""
    session.rollback()
    context = _library_context(
        request,
        session,
        settings,
        sort=sort,
        view=view,
        trashed=trashed,
        notice={"kind": "error", "text": text},
        selected_ids=frozenset(selected_ids or []),
    )
    return templates.TemplateResponse(
        request, "media/media.html", context, status_code=status_code
    )


@router.post("/media/trash")
def media_trash(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    media_id: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Move selected active media into the durable operation-owned trash tree."""
    _require_csrf(request, CSRF_MEDIA_TRASH, csrf_token)
    settings: Settings = request.app.state.settings
    raw_ids = media_id or []
    try:
        parsed = _parse_media_selection(raw_ids)
        _load_selection(session, parsed)
    except _SelectionError as exc:
        return _reject_media_operation(
            request,
            session,
            settings,
            status_code=exc.status_code,
            text=exc.text,
            sort=sort,
            view=view,
            trashed=False,
            selected_ids=raw_ids,
        )

    done = 0
    skipped = 0
    for media_uuid in parsed:
        claim_token = uuid.uuid4().hex
        try:
            operation = plan_trash(session, media_uuid, claim_token)
            session.commit()
            execute_operation(session, settings.media_root, operation, claim_token)
            session.expire(operation)
            if operation.state == OperationState.COMPLETED.value:
                done += 1
            else:
                skipped += 1
        except OperationRefused:
            session.rollback()
            skipped += 1
        except Exception:
            logger.exception("unexpected error trashing media %s", media_uuid)
            session.rollback()
            skipped += 1

    params: dict[str, Any] = {
        "trash_done": done,
        "sort": _safe_sort(sort),
        "view": _safe_view(view),
    }
    if skipped:
        params["skipped"] = skipped
    return RedirectResponse(f"/media?{urlencode(params)}", status_code=303)


@router.post("/media/restore")
def media_restore(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    media_id: Annotated[list[str] | None, Form()] = None,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Restore selected trash items to their pre-trash paths."""
    _require_csrf(request, CSRF_MEDIA_RESTORE, csrf_token)
    settings: Settings = request.app.state.settings
    raw_ids = media_id or []
    try:
        parsed = _parse_media_selection(raw_ids)
        _load_selection(session, parsed)
    except _SelectionError as exc:
        return _reject_media_operation(
            request,
            session,
            settings,
            status_code=exc.status_code,
            text=exc.text,
            sort=sort,
            view=view,
            trashed=True,
            selected_ids=raw_ids,
        )

    done = 0
    skipped = 0
    for media_uuid in parsed:
        claim_token = uuid.uuid4().hex
        try:
            trash_operation = session.execute(
                select(MediaOperation)
                .where(
                    MediaOperation.media_id == media_uuid,
                    MediaOperation.operation_type == OperationType.TRASH.value,
                    MediaOperation.state == OperationState.COMPLETED.value,
                )
                .order_by(MediaOperation.created_at.desc(), MediaOperation.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if trash_operation is None:
                raise OperationRefused("completed trash operation does not exist")
            operation = plan_restore(
                session, media_uuid, trash_operation.id, claim_token
            )
            session.commit()
            execute_operation(session, settings.media_root, operation, claim_token)
            session.expire(operation)
            if operation.state == OperationState.COMPLETED.value:
                done += 1
            else:
                skipped += 1
        except OperationRefused:
            session.rollback()
            skipped += 1
        except Exception:
            logger.exception("unexpected error restoring media %s", media_uuid)
            session.rollback()
            skipped += 1

    params: dict[str, Any] = {
        "trashed": "1",
        "restore_done": done,
        "sort": _safe_sort(sort),
        "view": _safe_view(view),
    }
    if skipped:
        params["skipped"] = skipped
    return RedirectResponse(f"/media?{urlencode(params)}", status_code=303)


@router.post("/media/empty-trash")
def media_empty_trash(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    sort: Annotated[str, Form()] = DEFAULT_SORT,
    view: Annotated[str, Form()] = _DEFAULT_VIEW,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Permanently delete every trashed, non-purged media item and its artifacts."""
    _require_csrf(request, CSRF_MEDIA_EMPTY_TRASH, csrf_token)
    settings: Settings = request.app.state.settings
    media_ids = list(
        session.execute(
            select(MediaItem.id).where(
                MediaItem.trashed_at.is_not(None),
                MediaItem.purged_at.is_(None),
            )
        ).scalars()
    )

    done = 0
    skipped = 0
    for media_uuid in media_ids:
        claim_token = uuid.uuid4().hex
        try:
            operation = plan_purge(session, media_uuid, claim_token)
            session.commit()
            build_manifest(session, operation)
            session.commit()
            execute_purge(session, settings.media_root, operation, claim_token)
            session.expire(operation)
            if operation.state == OperationState.COMPLETED.value:
                done += 1
            else:
                skipped += 1
        except OperationRefused:
            session.rollback()
            skipped += 1
        except Exception:
            logger.exception("unexpected error purging media %s", media_uuid)
            session.rollback()
            skipped += 1

    params: dict[str, Any] = {
        "trashed": "1",
        "empty_trash_done": done,
        "sort": _safe_sort(sort),
        "view": _safe_view(view),
    }
    if skipped:
        params["skipped"] = skipped
    return RedirectResponse(f"/media?{urlencode(params)}", status_code=303)
