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
from pathlib import Path
from typing import Annotated, Any, Final
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from voxint.api.csrf import (
    CSRF_MEDIA_ASSIGN,
    CSRF_MEDIA_FETCH,
    CSRF_MEDIA_FOLDERS,
    CSRF_MEDIA_RERUN,
    CSRF_MEDIA_RERUN_CONFIRM,
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
from voxint.db.models import MediaFolder, MediaItem, PipelineRun
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import operator_correction_message
from voxint.domain_packs.registry import resolve_domain_pack_by_name
from voxint.ingest import (
    EffectiveConfigPreview,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    preview_effective_config,
    submit_media_item,
    submit_upload,
    submit_url,
)
from voxint.ingest.sidecar import Sidecar, SidecarError, find_sidecar, read_sidecar
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
        "csrf_media_rerun": mint_csrf_token(secret, CSRF_MEDIA_RERUN),
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
    media_path = settings.media_root / Path(source_path)
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
