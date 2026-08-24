"""Legacy runs area: submission, browsing, run actions, dashboards, media.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151).
Four routers, all behind the router-level onboarding gate, because the run
routes interleave with the review family in registration order (which the P0b
order contract pins): ``core_router`` (/ index, /runs browsing + search +
submit/fetch + run detail and transcript), ``actions_router`` (requeue,
cancel, archive, media delete, notes, export), ``dashboards_router``
(/metrics, /dashboard, /resources), and ``tail_router`` (run assets,
translation, media streaming + peaks). Console 2.0 later phases fold these
surfaces into the new home/media/jobs areas, hence "legacy".
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, BinaryIO
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import review_backlog_count
from voxint.adjudication.transcript import (
    TranscriptText,
    attributed_transcript,
    paragraphize_transcript,
    parse_transcript_text,
)
from voxint.api.csrf import (
    CSRF_ASSETS_CANCEL,
    CSRF_ASSETS_GENERATE,
    CSRF_CANCEL,
    CSRF_FETCH,
    CSRF_NOTES,
    CSRF_REQUEUE,
    CSRF_RUN_ARCHIVE,
    CSRF_RUN_MEDIA_DELETE,
    CSRF_RUN_UNARCHIVE,
    CSRF_SUBMIT,
    CSRF_TRANSLATION_CANCEL,
    CSRF_TRANSLATION_GENERATE,
    mint_csrf_token,
)
from voxint.api.languages import LANGUAGE_NAMES, language_label
from voxint.api.meaning_query import search_passages
from voxint.api.model_provenance import select_run_model_identity
from voxint.api.playback import MediaResolutionError, playback_capability, resolve_servable_media
from voxint.api.presentation import title_from_snapshot
from voxint.api.resource_status import (
    ResourceSnapshot,
    build_resource_strip,
    collect_resource_status,
    render_resource_prometheus,
)
from voxint.api.routers import deps
from voxint.api.routers.deps import (
    _TRANSLATION_ACTIVE_STATUSES,
    OperatorDep,
    SessionDep,
    _get_media_gate,
    _reject_if_archived,
    _require_csrf,
    _run_or_404,
    require_onboarded,
    templates,
)
from voxint.api.runs_query import (
    Cursor,
    InvalidCursorError,
    ReviewFilter,
    latest_completed_run,
    list_runs,
    parse_review_filter,
    parse_search_filters,
    parse_status_filter,
    runs_url,
    searchable_languages,
)
from voxint.api.speaker_colors import speaker_palette
from voxint.api.stats_query import DEFAULT_WINDOW, collect_stats, parse_since, render_prometheus
from voxint.api.transcript_view import (
    _run_label_universe,
    _transcript_island_props,
    _wants_island_json,
)
from voxint.api.tutorial_view import _tutorial_banner
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_translation_target_language,
    resolve_effective_ytdlp_enabled,
)
from voxint.config import Settings
from voxint.db.models import (
    PipelineRun,
    RunAssetJob,
    RunAssetJobStatus,
    RunAssetKind,
    RunStatus,
    Stage,
    StageRun,
    TranscriptSegment,
    TranslationJob,
)
from voxint.domain_packs.base import DomainPackError
from voxint.enrichment.asset_jobs import (
    RunAssetJobError,
    active_or_last_jobs,
    create_jobs,
    run_asset_gates_open,
)
from voxint.enrichment.asset_jobs import request_cancel as request_asset_cancel
from voxint.enrichment.run_assets import (
    RunAssetError,
    latest_assets,
    load_source,
    source_content_hash,
)
from voxint.enrichment.translation_jobs import (
    TranslationJobError,
    normalized_language,
    translation_gates_open,
)
from voxint.enrichment.translation_jobs import active_or_last_job as active_or_last_translation_job
from voxint.enrichment.translation_jobs import create_job as create_translation_job
from voxint.enrichment.translation_jobs import request_cancel as request_translation_cancel
from voxint.enrichment.translations import (
    TranslationError,
    current_translations,
    load_translation_source,
    translation_source_hash,
    translation_texts,
)
from voxint.export import MEDIA_TYPES, format_timespan, transcript_payload
from voxint.ingest import (
    MissingStageError,
    RunMediaNotDeletableError,
    RunNotArchivableError,
    RunNotCancellableError,
    RunNotFailedError,
    RunNotFoundError,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    archive_run,
    cancel_run,
    delete_run_derived_media,
    requeue_failed_run,
    submit_upload,
    submit_url,
    unarchive_run,
    unlink_media_paths,
)
from voxint.media.peaks import (
    CachedPeaks,
    PeaksError,
    SourceFingerprint,
    compute_peaks,
    load_cached_peaks,
    store_peaks,
)
from voxint.media.reclaim import run_intermediate_reclaimed_at
from voxint.media.redaction import provenance_host
from voxint.media.serving import RangeNotSatisfiableError, parse_range
from voxint.media.source_metadata import RAW_URL_KEYS
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path
from voxint.pipeline.transitions import InvalidTransitionError, StaleRevisionError
from voxint.speakers.roster import searchable_speakers
from voxint.tutorial.steps import TutorialPage

logger = logging.getLogger(__name__)


core_router = APIRouter(dependencies=[Depends(require_onboarded)])
actions_router = APIRouter(dependencies=[Depends(require_onboarded)])
dashboards_router = APIRouter(dependencies=[Depends(require_onboarded)])
tail_router = APIRouter(dependencies=[Depends(require_onboarded)])


_MEDIA_CHUNK_BYTES = 256 * 1024
# Bound on the per-run operator notes (issue #36) — hygiene for a TEXT column,
# generous enough for real operator prose.
MAX_OPERATOR_NOTES_CHARS = 10_000

def _export_raw_host_only(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce the URL keys in an exported ``raw`` snapshot to host-only (D4).

    ``raw`` is an allowlisted scalar subset — only ``RAW_URL_KEYS`` hold URLs — so
    those are reduced to a bare host (matching the run-detail provenance policy)
    while every other key passes through unchanged."""
    if raw is None:
        return None
    return {
        key: (provenance_host(value) if key in RAW_URL_KEYS else value)
        for key, value in raw.items()
    }



def _publish_translation_job(job_id: uuid.UUID) -> bool:
    """Enqueue a committed translation job, returning False on a broker outage.

    Mirrors ``_publish_run_asset_job``: no recovery sweep (v1), so the console
    shows a deferred job as queued with its age and the operator cancels and
    retries."""
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import translate_run

    try:
        translate_run.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "broker unavailable — translation job %s stays QUEUED",
            job_id,
            exc_info=True,
        )
        return False
    return True


def _publish_run_asset_job(job_id: uuid.UUID) -> bool:
    """Enqueue a committed run-asset job, returning False on a broker outage.

    Mirrors ``_publish_research_job``: no recovery sweep (v1), so the console
    shows a deferred job as queued with its age and the operator cancels and
    retries."""
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import generate_run_asset

    try:
        generate_run_asset.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "run-asset enqueue deferred (broker unavailable); job %s stays QUEUED",
            job_id,
            exc_info=True,
        )
        return False
    return True


def _run_redirect(run_id: uuid.UUID, *, published: bool) -> RedirectResponse:
    """303 back to the run detail; a deferred publish flags the QUEUED banner.

    The ``enqueue=deferred`` marker is read only as a boolean by the detail
    route — never echoed — and worded historically there, since a refresh or
    bookmark can carry the parameter past the point the sweep already published."""
    suffix = "" if published else "?enqueue=deferred"
    return RedirectResponse(f"/runs/{run_id}{suffix}", status_code=303)


def _media_delete_banner(request: Request) -> dict[str, int] | None:
    """Parse the post-delete PRG counters into a banner dict, or None.

    A derived-media deletion (issue #5) redirects back with ``?media=deleted``
    and non-negative ``files``/``missing``/``failed`` counts. They are read as
    bare ints and never echoed as text; a malformed/absent value yields no
    banner (a refresh or bookmark can carry the parameter past its meaning)."""
    if request.query_params.get("media") != "deleted":
        return None

    def _count(name: str) -> int:
        raw = request.query_params.get(name, "0")
        return int(raw) if raw.isdigit() else 0

    return {
        "files": _count("files"),
        "missing": _count("missing"),
        "failed": _count("failed"),
    }


_RUN_ASSET_KINDS = tuple(kind.value for kind in RunAssetKind)
# Asset jobs have their own enum — do not borrow the research tuple just
# because the string values coincide today.
_ASSET_ACTIVE_STATUSES = (
    RunAssetJobStatus.QUEUED.value,
    RunAssetJobStatus.RUNNING.value,
)
_ASSET_KIND_TITLES = {
    "summary": "Summary",
    "topics": "Topics",
    "entity_mentions": "Entity mentions",
}


def _run_assets_state(
    session: Session, settings: Settings, run_id: uuid.UUID, error: str | None = None
) -> dict[str, Any]:
    """The run's asset block: per kind, the current asset (with staleness
    against the freshly recomputed source hash) and the active/last job."""
    assets = latest_assets(session, run_id)
    jobs = active_or_last_jobs(session, run_id)
    current_hash: str | None = None
    source_problem: str | None = None
    try:
        current_hash = source_content_hash(load_source(session, run_id))
    except RunAssetError as exc:
        source_problem = str(exc)
    kinds = []
    any_active = False
    for kind in _RUN_ASSET_KINDS:
        asset = assets.get(kind)
        job = jobs.get(kind)
        active = job is not None and job.status in _ASSET_ACTIVE_STATUSES
        any_active = any_active or active
        kinds.append(
            {
                "kind": kind,
                "title": _ASSET_KIND_TITLES[kind],
                "asset": asset,
                "stale": (
                    asset is not None
                    and current_hash is not None
                    and asset.source_content_hash != current_hash
                ),
                "job": job,
                "job_active": active,
            }
        )
    return {
        "run_id": run_id,
        "kinds": kinds,
        "any_active": any_active,
        "gates_open": run_asset_gates_open(settings, get_app_settings(session)),
        "source_problem": source_problem,
        "error": error,
    }


def _run_assets_response(
    request: Request, session: Session, run_id: uuid.UUID, error: str | None = None
) -> Response:
    """The per-run assets fragment — the polling target and every asset
    mutation's response."""
    secret = request.app.state.csrf_secret
    return templates.TemplateResponse(
        request,
        "fragments/run_assets.html",
        {
            "request": request,
            "assets": _run_assets_state(
                session, request.app.state.settings, run_id, error
            ),
            "csrf_assets_generate": mint_csrf_token(secret, CSRF_ASSETS_GENERATE),
            "csrf_assets_cancel": mint_csrf_token(secret, CSRF_ASSETS_CANCEL),
        },
    )




def _run_translation_state(
    session: Session, settings: Settings, run_id: uuid.UUID, error: str | None = None
) -> dict[str, Any]:
    """The run's translation block: every current generation (with staleness
    against the freshly recomputed source hash) and the active/last job."""
    row = get_app_settings(session)
    heads = current_translations(session, run_id)
    job = active_or_last_translation_job(session, run_id)
    job_active = job is not None and job.status in _TRANSLATION_ACTIVE_STATUSES
    current_hash: str | None = None
    source_problem: str | None = None
    try:
        current_hash = translation_source_hash(load_translation_source(session, run_id))
    except TranslationError as exc:
        source_problem = str(exc)
    run = session.get(PipelineRun, run_id)
    detected = normalized_language(run.detected_language) if run is not None else None
    preferred = normalized_language(
        resolve_effective_translation_target_language(row, settings)
    )
    # The generate select defaults to the preferred language; with none set,
    # the operator picks one per run (blank forces an explicit choice). A
    # preferred language MATCHING the detected one also defaults to blank:
    # the template drops the detected language from the options, so keeping
    # it as the default would leave no option selected and the browser would
    # silently submit the first arbitrary language instead.
    default_target = preferred if preferred and preferred != detected else ""
    return {
        "run_id": run_id,
        "translations": [
            {
                "translation": head,
                # Fail closed like the transcript view and exports: an
                # unreadable source (current_hash None) marks every generation
                # out of date rather than dressing it up as viewable (review
                # finding — the card must never promise a view the transcript
                # page and ?lang= exports would refuse).
                "stale": current_hash is None
                or head.source_content_hash != current_hash,
            }
            for head in heads
        ],
        "job": job,
        "job_active": job_active,
        "any_active": job_active,
        "gates_open": translation_gates_open(settings, row),
        "source_problem": source_problem,
        "error": error,
        "default_target": default_target,
        "detected_language": detected,
        "language_options": sorted(LANGUAGE_NAMES.items(), key=lambda item: item[1]),
    }


def _run_translation_response(
    request: Request, session: Session, run_id: uuid.UUID, error: str | None = None
) -> Response:
    """The per-run translation fragment — the polling target and every
    translation mutation's response."""
    secret = request.app.state.csrf_secret
    return templates.TemplateResponse(
        request,
        "fragments/run_translation.html",
        {
            "request": request,
            "translation_state": _run_translation_state(
                session, request.app.state.settings, run_id, error
            ),
            "csrf_translation_generate": mint_csrf_token(secret, CSRF_TRANSLATION_GENERATE),
            "csrf_translation_cancel": mint_csrf_token(secret, CSRF_TRANSLATION_CANCEL),
        },
    )


def _transcript_translation_context(
    session: Session,
    run_id: uuid.UUID,
    variant: TranscriptText,
    line_count: int,
    requested: str | None,
) -> dict[str, Any]:
    """The read-only transcript page's translation state (issue #133).

    ``views`` lists every current generation with its freshness so the page can
    offer a toggle; ``active`` carries the selected FRESH generation's texts for
    interleaving (island props + server fallback); ``note`` is the honest
    explanation when a requested translation cannot be shown. A stale or
    missing generation never interleaves — the whole generation is stale as one
    unit, matching the export policy. Only the reviewed (corrected) variant
    pairs with a translation; on raw/enhanced the toggle is absent and a
    hand-typed ``?translation=`` gets a note, not silently-mismatched lines.
    """
    heads = current_translations(session, run_id)
    if not heads and requested is None:
        return {"views": [], "active": None, "note": None, "fresh": []}
    current_hash: str | None = None
    # No readable source ⇒ every generation renders as out of date.
    with contextlib.suppress(TranslationError):
        current_hash = translation_source_hash(load_translation_source(session, run_id))
    heads_by_code = {head.target_language: head for head in heads}
    # The hash is the ONLY freshness signal (issue #133 invariant): no readable
    # source fails closed to stale. The displayed variant's line count must NOT
    # feed this — raw/enhanced/read renders could differ from the corrected
    # translation source and would falsely stale a fresh generation (review
    # finding); the count defense lives at interleave time below, where the
    # variant is provably CORRECTED.
    views = [
        {
            "code": head.target_language,
            "label": language_label(head.target_language),
            "stale": current_hash is None
            or head.source_content_hash != current_hash,
        }
        for head in heads
    ]
    active: dict[str, Any] | None = None
    note: str | None = None
    if requested is not None:
        target = normalized_language(requested)
        view = next((v for v in views if v["code"] == target), None)
        if view is None or target is None:
            note = "No such translation for this run — generate one from the run page."
        elif variant is not TranscriptText.CORRECTED:
            note = (
                "Translations pair with the reviewed text only — showing the"
                " original. Switch to the corrected version to see the translation."
            )
        elif view["stale"]:
            note = (
                f"The {view['label']} translation is out of date — the transcript"
                " changed since it was generated. Re-translate from the run page;"
                " the old rendition is not shown against the changed transcript."
            )
        else:
            head = heads_by_code[target]
            texts = translation_texts(head)
            # Defensive count check, HERE only: the variant is CORRECTED on
            # this branch, so line_count is the translation source's own
            # geometry. Hash equality implies equality; a corrupt row must
            # degrade to "out of date", never misalign.
            if len(texts) != line_count:
                note = (
                    f"The {view['label']} translation is out of date — the"
                    " transcript changed since it was generated. Re-translate"
                    " from the run page."
                )
            else:
                active = {
                    "language": head.target_language,
                    "label": language_label(head.target_language),
                    "texts": texts,
                    "model": head.model,
                    "completed_at": head.completed_at,
                    "source_language": head.source_language,
                }
    return {
        "views": views,
        "active": active,
        "note": note,
        # Fresh generations only — the export menu's translated links (a stale
        # generation gets NO link; its export would 409).
        "fresh": [
            {"code": v["code"], "label": v["label"]} for v in views if not v["stale"]
        ],
    }



def _peaks_cache_trusted(
    session: Session, run_id: uuid.UUID, media_root: Path, cached: CachedPeaks
) -> bool:
    """Whether a cached envelope may be served for this run.

    Fail-closed: trusted ONLY when the WAV is formally reclaimed (nothing left
    to verify against — the static waveform is honest derived evidence) OR the
    live WAV's fstat fingerprint still matches what the peaks were computed
    from. Any other live-media state — no resolvable preprocessed row, a stat
    failure, a malformed stored fingerprint — is untrusted, so the route falls
    through to ``resolve_servable_media`` and answers the honest 404/410 (or
    recomputes) rather than serving unverified bytes.
    """
    if run_intermediate_reclaimed_at(session, run_id) is not None:
        return True
    try:
        wav_path = normalized_audio_path(session, run_id, media_root)
    except StageDataError:
        return False
    live = SourceFingerprint.of_path(wav_path)
    return live is not None and live == cached.source_fingerprint


def _peaks_cache_response(request: Request, body: bytes, artifact_id: uuid.UUID) -> Response:
    """Serve the cached envelope with real conditional-GET semantics.

    The artifact row UUID is a valid strong ETag because a row's bytes are
    immutable once published — every refresh (re-prepare, stale-fingerprint
    recompute) mints a new row or rewrites meta+file together before this is
    called. ``no-cache`` forces revalidation, so a requeued run can never show
    a stale waveform from the browser cache; the 304 makes that revalidation
    cost one conditional GET.

    ``If-None-Match`` uses the weak comparison RFC 9110 §13.1.2 mandates, so a
    ``W/"<uuid>"`` validator from an intermediary still matches our strong tag.
    """
    etag = f'"{artifact_id}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None:
        # Strip an optional weak-validator prefix before comparing (weak match).
        candidates = {
            v.strip()[2:] if v.strip().startswith('W/') else v.strip()
            for v in if_none_match.split(",")
        }
        if "*" in candidates or etag in candidates:
            return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


def _stream_file(fh: BinaryIO, start: int, length: int) -> Iterator[bytes]:
    """Stream from the gate-validated descriptor — never reopen by path."""
    remaining = length
    with fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_MEDIA_CHUNK_BYTES, remaining))
            if not chunk:
                return  # file shrank mid-stream; truncate rather than hang
            remaining -= len(chunk)
            yield chunk


@core_router.get("/", include_in_schema=False)
def index(operator: OperatorDep) -> RedirectResponse:
    # On the protected router: when onboarded the gate passes and we land on
    # the review queue; when not, the gate has already redirected to /setup,
    # so this stays an unconditional redirect (no second onboarding read).
    return RedirectResponse("/review", status_code=303)


@core_router.get("/runs")
def runs(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    status: str | None = None,
    review: str | None = None,
    cursor: str | None = None,
    q: str | None = None,
    speaker: str | None = None,
    source: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    language: str | None = None,
    archived: str | None = None,
) -> Response:
    settings: Settings = request.app.state.settings
    # ?archived=1 flips to the archived-only view (issue #5); anything else
    # (absent / "0" / blank) is the default listing that hides archived runs.
    show_archived = archived == "1"
    try:
        status_filter = parse_status_filter(status)
        review_filter = parse_review_filter(review)
        search_filters = parse_search_filters(
            q=q,
            speaker=speaker,
            source=source,
            created_from=created_from,
            created_to=created_to,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    parsed_cursor: Cursor | None = None
    # A blank cursor means "start at page 1", mirroring blank status/review
    # meaning "all"; only a non-empty but malformed token is a 400.
    if cursor:
        try:
            parsed_cursor = Cursor.decode(cursor)
        except InvalidCursorError as exc:
            raise HTTPException(status_code=400, detail="invalid cursor") from exc
    page = list_runs(
        session,
        status=status_filter,
        review=review_filter,
        cursor=parsed_cursor,
        page_size=settings.runs_page_size,
        filters=search_filters,
        archived=show_archived,
    )
    next_url = (
        runs_url(
            status=status_filter,
            review=review_filter,
            filters=search_filters,
            archived=show_archived,
            cursor=page.next_cursor,
        )
        if page.next_cursor
        else None
    )
    return templates.TemplateResponse(
        request,
        "runs.html",
        {
            "request": request,
            "page": page,
            "status": status_filter,
            "review": review_filter,
            "statuses": list(RunStatus),
            "reviews": list(ReviewFilter),
            "filters": search_filters,
            "show_archived": show_archived,
            # Toggle target: the same listing in the opposite archive view,
            # preserving the active status/review/search facets.
            "archived_toggle_url": runs_url(
                status=status_filter,
                review=review_filter,
                filters=search_filters,
                archived=not show_archived,
            ),
            "facet_speakers": searchable_speakers(session),
            "facet_languages": searchable_languages(
                session,
                archived=show_archived,
                include=search_filters.language,
            ),
            "next_url": next_url,
            # Server-issued per-render ids: each namespaces its form's path and
            # makes a double-submit idempotent (see POST /submit, POST /fetch).
            # The upload and fetch forms get INDEPENDENT ids so the two unrelated
            # submissions can never collide on a shared source_path.
            "submission_id": uuid.uuid4().hex,
            "fetch_submission_id": uuid.uuid4().hex,
            # Per-form CSRF tokens, each bound to its own action so one form's
            # token is not valid on the other's route (or on /requeue).
            "csrf_submit": mint_csrf_token(request.app.state.csrf_secret, CSRF_SUBMIT),
            "csrf_fetch": mint_csrf_token(request.app.state.csrf_secret, CSRF_FETCH),
            # Gate the URL-fetch form: when off, it renders disabled (POST /fetch
            # also refuses with 403), matching the CLI's ytdlp_enabled refusal.
            "ytdlp_enabled": resolve_effective_ytdlp_enabled(
                get_app_settings(session), settings
            ),
            # Injected clock for the relative-age render (format_age(now=…)).
            "now": datetime.now(UTC),
            "active_nav": "runs",
        },
    )

@core_router.get("/search")
def search(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    q: str | None = None,
) -> Response:
    # The ranked semantic "Meaning" mode (issue #121): a distinct surface from
    # the chronological /runs browse, reading the embedding index the spine
    # builds. search_passages needs the session FACTORY, not this request's
    # session: it opens its own two short sessions (a settings gate, then one
    # read-only REPEATABLE READ snapshot that wraps all ranking arms so a
    # concurrent publish cannot straddle them). SessionDep still runs first,
    # so the factory is initialized and the basic-auth + onboarding gates on
    # the `protected` router have already passed. The feature/weights/indexing
    # state comes back on the page object, which the template renders honestly.
    settings: Settings = request.app.state.settings
    page = search_passages(
        request.app.state.session_factory,
        settings=settings,
        query=q or "",
    )
    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "request": request,
            "page": page,
            "query": page.query,
            # Meaning search pairs with Runs (reached from its toggle), so the
            # Runs nav item stays lit rather than leaving the nav orphaned.
            "active_nav": "runs",
        },
    )

@core_router.post("/submit")
def submit_media_upload(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    submission_id: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    # CSRF before anything: a forged cross-site upload is refused before the DB
    # write / file finalize. (FastAPI spools the multipart file part before this
    # body runs — the pre-body spool is bounded by _RequestSizeLimitMiddleware +
    # the streaming cap, so a forgery cannot write past those; the DB and the
    # os.replace publish are still gated here.)
    _require_csrf(request, CSRF_SUBMIT, csrf_token)
    settings: Settings = request.app.state.settings
    # An over-cap Content-Length was already rejected before the body was read
    # (_RequestSizeLimitMiddleware); submit_upload enforces the exact per-file
    # cap authoritatively while streaming (covers a lying/absent length).
    try:
        run = submit_upload(
            session,
            stream=file.file,
            filename=file.filename or "",
            submission_id=submission_id,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DomainPackError as exc:
        # Freeze-time snapshot collision (issue #84) / unresolvable pack (issue
        # #11): surface a plain-language 422, not the raw 500 the bare raise gave.
        raise HTTPException(
            status_code=422, detail=deps._submit_domain_pack_detail(exc)
        ) from exc
    run_id = run.id
    # Commit-before-publish: the durable QUEUED run must exist before the
    # enqueue, so commit here rather than leaning on the dependency's
    # post-return commit (which would run after publish). A broker outage is
    # then non-fatal — the run stays QUEUED and the recovery sweep republishes.
    session.commit()
    return _run_redirect(run_id, published=deps._publish_or_defer(run_id))

@core_router.post("/fetch")
def fetch_media_url(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    url: Annotated[str, Form()],
    submission_id: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    # CSRF first — reject a forged cross-site fetch before any other check, so
    # a forgery is refused regardless of the ytdlp_enabled flag's state.
    _require_csrf(request, CSRF_FETCH, csrf_token)
    # URL ingestion is an authenticated egress capability gated at the
    # submission surface: refuse before touching the DB when it is off, so no
    # row is created and nothing is published. (The worker's ACQUIRE stage
    # never consults the flag — an already-queued URL run still completes.)
    settings: Settings = request.app.state.settings
    if not resolve_effective_ytdlp_enabled(get_app_settings(session), settings):
        # Generic message — never echo the submitted URL into an error body.
        raise HTTPException(status_code=403, detail="URL ingestion is disabled")
    # Mirrors POST /submit: the ingest service is broker-free and does its own
    # SSRF/replay handling; map its typed errors to status codes. The error
    # text is URL-free by construction (UrlValidationError never echoes the
    # URL; a conflict names only the internal source_path), so a signed query
    # string can't leak into a 4xx body.
    try:
        run = submit_url(session, url=url, submission_id=submission_id)
    except (UrlValidationError, UploadValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DomainPackError as exc:
        # Freeze-time snapshot collision (issue #84) / unresolvable pack (issue
        # #11): plain-language 422 rather than a raw 500. The snapshot is frozen
        # before any row/source_path write, so nothing is stranded.
        raise HTTPException(
            status_code=422, detail=deps._submit_domain_pack_detail(exc)
        ) from exc
    run_id = run.id
    # Commit-before-publish, exactly as /submit: the durable QUEUED run must
    # exist before the enqueue, so a broker outage leaves it QUEUED for the
    # recovery sweep rather than failing the request.
    session.commit()
    return _run_redirect(run_id, published=deps._publish_or_defer(run_id))

@core_router.get("/runs/{run_id}")
def run_detail(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    run = _run_or_404(session, run_id)
    # The attempt ledger, chronological — matches `voxint status`.
    stage_runs = list(
        session.execute(
            select(StageRun)
            .where(StageRun.pipeline_run_id == run_id)
            .order_by(StageRun.started_at)
        ).scalars()
    )
    settings: Settings = request.app.state.settings
    # Present-only links, decided in Postgres (no filesystem on the read path).
    # Audio reuses the SAME exactly-one-artifact predicate /media serves through
    # (normalized_audio_path resolves iff there is exactly one preprocessed-audio
    # row), so the link never promises a page that would 404; a transcript link
    # needs only that TRANSCRIBE wrote at least one segment.
    try:
        normalized_audio_path(session, run_id, settings.media_root)
        audio_available = True
    except StageDataError:
        audio_available = False
    # GC (issue #15) may have reclaimed the normalized-audio intermediate.
    # The DB row survives, so normalized_audio_path still resolves — but the
    # file is gone. Suppress the dead audio link and show a reclaimed notice.
    media_reclaimed_at = run_intermediate_reclaimed_at(session, run_id)
    if media_reclaimed_at is not None:
        audio_available = False
    transcript_available = bool(
        session.scalar(select(exists().where(TranscriptSegment.pipeline_run_id == run_id)))
    )
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "stage_runs": stage_runs,
            # Which model actually answered each stage, from that stage's
            # latest completed attempt (A1 provenance). "Not recorded" for
            # legacy runs stamped before this existed.
            "model_provenance": select_run_model_identity(stage_runs),
            # The frozen sidecar title (issue #104), shown as the run's
            # display name above the raw path. Tolerant read: None when the
            # run has no sidecar (or a tampered one).
            "sidecar_title": title_from_snapshot(run.sidecar),
            # Provenance for a URL run, reduced to a bare host — the raw
            # source_url (which can carry a signed token in its query) is
            # NEVER passed to the template; None for a local/uploaded run
            # renders as a neutral placeholder.
            "provenance": provenance_host(run.media_item.source_url),
            "audio_available": audio_available,
            "media_reclaimed_at": media_reclaimed_at,
            "transcript_available": transcript_available,
            # Read as a bare boolean (never echoed): a submit/requeue whose
            # enqueue was deferred by a broker outage redirects here with it.
            "enqueue_deferred": request.query_params.get("enqueue") == "deferred",
            # CSRF token for the requeue form (rendered only when FAILED).
            "csrf_requeue": mint_csrf_token(request.app.state.csrf_secret, CSRF_REQUEUE),
            # CSRF token for the cancel form (rendered only for a live run:
            # queued / running / awaiting_adjudication — issue #5).
            "csrf_cancel": mint_csrf_token(request.app.state.csrf_secret, CSRF_CANCEL),
            # Soft-archive + derived-media deletion (issue #5, slice 2). The
            # buttons render only for a terminal run; un-archive replaces
            # archive once the run carries a stamp.
            "run_archived": run.archived_at is not None,
            "run_terminal": run.status
            in (
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            ),
            "csrf_run_archive": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_RUN_ARCHIVE
            ),
            "csrf_run_unarchive": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_RUN_UNARCHIVE
            ),
            "csrf_run_media_delete": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_RUN_MEDIA_DELETE
            ),
            # Post-redirect banner after a derived-media deletion (PRG): the
            # non-negative counts are read as bare ints, never echoed as text.
            "media_delete_result": _media_delete_banner(request),
            # Acquisition context (issue #36): the write-once snapshot, or
            # None for uploads / pre-capture URL runs. Scraped metadata and
            # the operator's own notes render in separate sections.
            "source_metadata": run.media_item.source_metadata,
            "csrf_notes": mint_csrf_token(request.app.state.csrf_secret, CSRF_NOTES),
            # Run-level assets (issue #41): current summary/topics/entity
            # mentions with staleness, plus generation controls.
            "assets": _run_assets_state(session, settings, run_id),
            "csrf_assets_generate": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_ASSETS_GENERATE
            ),
            "csrf_assets_cancel": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_ASSETS_CANCEL
            ),
            # Transcript translation (issue #133): current generation(s)
            # with staleness, plus generation controls.
            "translation_state": _run_translation_state(session, settings, run_id),
            "csrf_translation_generate": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_TRANSLATION_GENERATE
            ),
            "csrf_translation_cancel": mint_csrf_token(
                request.app.state.csrf_secret, CSRF_TRANSLATION_CANCEL
            ),
            "tutorial": _tutorial_banner(
                request, session, page=TutorialPage.RUN_DETAIL, run_id=run_id
            ),
            "active_nav": "runs",
        },
    )

@core_router.get("/runs/{run_id}/transcript")
def run_transcript(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    read: bool = False,
    timestamps: bool = True,
    translation: str | None = None,
) -> Response:
    run = _run_or_404(session, run_id)
    try:
        variant = parse_transcript_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Compute once, reuse for both the server-rendered fallback (`lines`) and
    # the island props (avoids a double query). The transcript-player island
    # (issue #48) reuses the already-auth-gated, Range-capable GET /media
    # for its <audio src>; no new backend route.
    lines = attributed_transcript(session, run_id, text=variant)
    settings: Settings = request.app.state.settings
    if read:
        # Read mode (issue #65): a timestamp-optional, shareable reading view
        # rendered purely server-side from the SAME presentation seam and the
        # SAME paragraph grouping the Markdown export uses — no island, no
        # second transcript truth. Grouping and timestamp formatting stay in
        # Python; the template only lays out the supplied rows. Jinja
        # autoescape (not the markdown-specific `_md_escape`) makes hostile
        # transcript text render literally in the HTML view.
        read_rows = [
            {
                "speaker": para.speaker,
                "lines": para.text.split("\n"),
                "timespan": (
                    format_timespan(para.start_seconds, para.end_seconds)
                    if timestamps
                    else None
                ),
            }
            for para in paragraphize_transcript(lines)
        ]
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "request": request,
                "run": run,
                "lines": lines,
                "read": True,
                "read_rows": read_rows,
                "read_timestamps": timestamps,
                "text": variant,
                "variants": list(TranscriptText),
                # Reading view stays original-language (no interleave), but
                # the export menu still lists fresh translated downloads.
                "translation_ctx": _transcript_translation_context(
                    session, run_id, variant, len(lines), None
                ),
                "active_nav": "runs",
            },
        )
    # Fail-closed seek gating (issue #55): the island only offers per-line
    # playback when GET /media would truly serve and the timeline is sound.
    capability = playback_capability(session, run, settings, _get_media_gate(request))
    # Per-speaker identity color (issue #50): derive the palette from the run's
    # canonical label universe (turns and segments) — the SAME universe the
    # workbench card uses — so a label's color agrees across both surfaces and
    # the JS-off fallback matches the hydrated island. The union also colors a
    # transcript-only label (a segment whose label has no turn). Two cheap
    # DISTINCT queries, kept explicit so the shared universe is provable here.
    palette = speaker_palette(_run_label_universe(session, run_id))
    island_props = _transcript_island_props(
        session, run_id, lines, palette, capability, settings
    )
    # Interleaved translation view (issue #133): a fresh generation the
    # operator selected renders each translated line beneath its original —
    # in the island props AND the JS-off fallback, from the same texts.
    translation_ctx = _transcript_translation_context(
        session, run_id, variant, len(lines), translation
    )
    if translation_ctx["active"] is not None:
        island_props["translation"] = {
            "language": translation_ctx["active"]["language"],
            "label": translation_ctx["active"]["label"],
            "lines": translation_ctx["active"]["texts"],
        }
    return templates.TemplateResponse(
        request,
        "transcript.html",
        {
            "request": request,
            "run": run,
            "lines": lines,
            "read": False,
            "island_props": island_props,
            "palette": palette,
            "low_confidence_threshold": settings.review_low_confidence_threshold,
            "text": variant,
            "variants": list(TranscriptText),
            "translation_ctx": translation_ctx,
            "active_nav": "runs",
        },
    )

# /review/{run_id}/transcript: moved to routers/legacy_review.py
# (transcript_router); included here to keep registration order.

@actions_router.post("/runs/{run_id}/requeue")
def requeue_run(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    revision: Annotated[int, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    _require_csrf(request, CSRF_REQUEUE, csrf_token)
    # An archived run is read-only — refuse before touching pipeline state so
    # a stale tab can't drive a hidden run live (issue #5).
    run = _run_or_404(session, run_id)
    _reject_if_archived(run)
    # Stable across this operation: FAILED -> QUEUED keeps the stage, and
    # QUEUED can leave only for RUNNING at that same stage or CANCELLED.
    failed_stage = Stage(run.current_stage) if run.current_stage else None
    # Exact-revision CAS from the form's hidden field: a stale browser tab
    # holding an older revision 409s rather than requeuing a run that already
    # moved on. FAILED-only, at the failed stage (the service enforces both).
    try:
        requeue_failed_run(session, run_id, expected_revision=revision)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RunNotFailedError, MissingStageError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (StaleRevisionError, InvalidTransitionError) as exc:
        # StaleRevisionError: the tab's revision lost the CAS.
        # InvalidTransitionError: defensive — this FAILED→QUEUED-same-stage
        # path cannot trip it, but the CAS contract permits it.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Commit-before-publish, mirroring /submit: the durable QUEUED run must
    # exist before the enqueue, and a broker outage leaves it QUEUED for the
    # recovery sweep rather than failing the request.
    session.commit()
    return _run_redirect(
        run_id, published=deps._publish_or_defer(run_id, stage=failed_stage)
    )

@actions_router.post("/runs/{run_id}/cancel")
def cancel_run_route(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    revision: Annotated[int, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Cancel a live run (QUEUED / RUNNING / AWAITING_ADJUDICATION) — issue #5.

    Exact-revision CAS from the form's hidden field, mirroring /requeue: a
    stale tab holding an older revision 409s rather than cancelling a run
    that already moved on. Cancellation is pure DB state (the existing
    ``→ CANCELLED`` transition), so unlike /submit and /requeue there is
    NOTHING to publish — a worker mid-run observes the cancel at its next
    stage boundary (the currently executing stage body finishes first), and
    a QUEUED run simply never starts. Cancelling an already-cancelled run is
    an idempotent 303, not a 409 (a double-click is success)."""
    _require_csrf(request, CSRF_CANCEL, csrf_token)
    try:
        cancel_run(session, run_id, expected_revision=revision)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RunNotCancellableError, StaleRevisionError, InvalidTransitionError) as exc:
        # RunNotCancellableError: a COMPLETED/FAILED run can't be cancelled.
        # StaleRevisionError: the tab's revision lost the CAS.
        # InvalidTransitionError: defensive — the cancellable-status guard
        # already excludes the states the CAS would reject, but the CAS
        # contract permits raising it.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Pure DB state, no enqueue — commit and return to the run detail (PRG).
    session.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)

@actions_router.post("/runs/{run_id}/archive")
def archive_run_route(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Soft-archive a terminal run — hide it from /runs and /review while
    keeping every row intact (issue #5). Last-write-wins operator-visibility
    metadata, so — unlike /requeue and /cancel — there is NO revision field
    and no CAS: a stale tab never 409s, and re-archiving is an idempotent
    303. Only terminal runs archive; a live run 409s (cancel it first)."""
    _require_csrf(request, CSRF_RUN_ARCHIVE, csrf_token)
    try:
        archive_run(session, run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunNotArchivableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)

@actions_router.post("/runs/{run_id}/unarchive")
def unarchive_run_route(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Reverse an archive — the run reappears in /runs and /review. Always
    allowed and idempotent (un-archiving a non-archived run is a no-op 303).
    Last-write-wins, no CAS (issue #5)."""
    _require_csrf(request, CSRF_RUN_UNARCHIVE, csrf_token)
    try:
        unarchive_run(session, run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)

@actions_router.post("/runs/{run_id}/media/delete")
def delete_run_media_route(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Delete a terminal run's DERIVED audio files — the preprocessed wav and
    per-chunk files — and their rows (issue #5). Irreversible, but scoped:
    it NEVER touches the shared original source (that would orphan a sibling
    run) or the evidence ledger. Commit-before-side-effect: the row delete
    commits first, THEN the files are unlinked (a rolled-back request must
    not leave the DB pointing at a file already gone). A live run 409s."""
    _require_csrf(request, CSRF_RUN_MEDIA_DELETE, csrf_token)
    settings: Settings = request.app.state.settings
    try:
        plan = delete_run_derived_media(
            session, run_id, media_root=settings.media_root
        )
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunMediaNotDeletableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    # Post-commit unlink (best-effort, idempotent): the rows are already gone,
    # so a partial filesystem failure is swept later, never fails the action.
    result = unlink_media_paths(plan.paths)
    params = urlencode(
        {
            "media": "deleted",
            "files": result.files_deleted,
            "missing": result.files_missing,
            "failed": result.files_failed,
        }
    )
    return RedirectResponse(f"/runs/{run_id}?{params}", status_code=303)

@actions_router.post("/runs/{run_id}/notes")
def save_operator_notes(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    notes: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    """Save the run's operator notes (issue #36) — human context, stored
    apart from scraped source metadata and deliberately OUTSIDE the CAS
    revision: notes are operator prose, not pipeline state, so a plain
    last-write-wins update is the honest single-operator semantics (unlike
    /requeue, where a stale tab acting on pipeline state must 409)."""
    _require_csrf(request, CSRF_NOTES, csrf_token)
    run = _run_or_404(session, run_id)
    cleaned = notes.strip()
    if len(cleaned) > MAX_OPERATOR_NOTES_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"notes exceed {MAX_OPERATOR_NOTES_CHARS} characters",
        )
    run.operator_notes = cleaned or None
    session.commit()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)

@actions_router.get("/runs/{run_id}/export.json")
def export_run_json(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
) -> Response:
    """Run-level JSON export (issue #36): an object envelope carrying the
    run, its acquisition metadata snapshot, the operator notes, and the
    same segment objects as the pinned transcript export. A NEW endpoint on
    purpose — /review/{run_id}/export.json is a frozen bare-array contract
    consumed as-is, so context rides a versioned envelope here instead of a
    breaking shape change there."""
    run = _run_or_404(session, run_id)
    try:
        variant = parse_transcript_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    snapshot = run.media_item.source_metadata
    # Finding D4: the export's URL fields are reduced to host-only, matching
    # the run-detail UI's provenance_host policy — the full page/uploader/
    # channel URLs (and raw's webpage_url) are the acquisition surface the
    # operator should not re-share by handing off an export. Descriptive
    # metadata (title, uploader/channel names, tags, description) is the
    # operator's own data and stays. `raw` is already an allowlisted scalar
    # subset (no signed URLs), so only its known URL keys need reducing.
    source_metadata = (
        {
            "source_kind": snapshot.source_kind,
            "title": snapshot.title,
            "uploader": snapshot.uploader,
            "uploader_url": provenance_host(snapshot.uploader_url),
            "channel": snapshot.channel,
            "channel_url": provenance_host(snapshot.channel_url),
            "description": snapshot.description,
            "upload_date": (snapshot.upload_date.isoformat() if snapshot.upload_date else None),
            "duration_seconds": snapshot.duration_seconds,
            "tags": snapshot.tags,
            "canonical_url": provenance_host(snapshot.canonical_url),
            "extractor": snapshot.extractor,
            "extractor_version": snapshot.extractor_version,
            "raw": _export_raw_host_only(snapshot.raw),
            "raw_schema_version": snapshot.raw_schema_version,
            "acquired_at": snapshot.acquired_at.isoformat(),
        }
        if snapshot is not None
        else None
    )
    # Run-level assets (issue #41): latest successful asset per kind with
    # full provenance + freshness. An ADDITIVE key under schema_version 1
    # (nullable; absent kinds are simply missing) — failed/running attempts
    # are operational state and stay off the portable export.
    assets = latest_assets(session, run_id)
    current_hash: str | None = None
    # No transcript yet → staleness is simply unknown.
    with contextlib.suppress(RunAssetError):
        current_hash = source_content_hash(load_source(session, run_id))
    enrichment_assets = (
        {
            kind: {
                "payload": asset.payload,
                "payload_schema_version": asset.payload_schema_version,
                "producer": asset.producer,
                "producer_version": asset.producer_version,
                "model": asset.model,
                "generation": asset.generation,
                "source_content_hash": asset.source_content_hash,
                "stale": (
                    asset.source_content_hash != current_hash
                    if current_hash is not None
                    else None
                ),
                "machine_generated": True,
                "completed_at": asset.completed_at.isoformat(),
            }
            for kind, asset in sorted(assets.items())
        }
        if assets
        else None
    )
    envelope = {
        # v2 (finding D4): source_metadata URL fields are host-only. v1 emitted
        # full uploader/channel/canonical URLs and raw.webpage_url verbatim.
        "schema_version": 2,
        "run": {
            "id": str(run.id),
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "operator_notes": run.operator_notes,
        },
        "source_metadata": source_metadata,
        "segments": transcript_payload(attributed_transcript(session, run_id, text=variant)),
        "enrichment_assets": enrichment_assets,
    }
    return Response(
        content=json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        media_type=MEDIA_TYPES["json"],
    )


# ---- Prometheus metrics ----------------------------------------------------
# Read-only aggregate exposition on the *protected* router: Prometheus scrapes
# it with basic_auth, so the "everything but /healthz authenticates" invariant
# holds without a new flag or token path. The one windowed series
# (voxint_runs_created_24h) bakes its window into the metric name.

def _resource_snapshot(request: Request) -> ResourceSnapshot:
    """The cached hardware snapshot for a render, guarded to never raise.

    ``/metrics`` and the dashboard/resource pages read this cached value
    (never ``force``): a 15s poll across tabs shares one probe, and a probe
    failure degrades to an empty snapshot rather than breaking the page.
    """
    try:
        return collect_resource_status(request.app.state.settings)
    except Exception:
        # Telemetry is advisory; a probe failure must never break a render.
        return ResourceSnapshot(gpus=(), services=(), collected_age_seconds=0.0)

@dashboards_router.get("/metrics")
def metrics(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    now = datetime.now(UTC)
    stats = collect_stats(session, since=now - DEFAULT_WINDOW, now=now)
    # Append the hardware gauges from the same cached snapshot the dashboard
    # and resource page render, so a scrape and the UI cannot disagree.
    body = render_prometheus(stats) + render_resource_prometheus(
        _resource_snapshot(request)
    )
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

# ---- Operator dashboard (issue #13) ----------------------------------------
# The HTML sibling of /metrics: the same collect_stats() snapshot, rendered
# for a human instead of a scraper. Protected like every non-/healthz page.
# An htmx poll (hx-trigger="every 15s") re-requests this same route with an
# HX-Request header; we answer that with just the numbers fragment so the
# nav/chrome are not re-swapped. A malformed ?since= degrades to the 24h
# default rather than 500-ing a bookmarked or hand-edited URL.

@dashboards_router.get("/dashboard")
def dashboard(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    now = datetime.now(UTC)
    raw_since = request.query_params.get("since")
    since = now - DEFAULT_WINDOW
    since_invalid = False
    if raw_since:
        try:
            since = parse_since(raw_since, now=now)
        except ValueError:
            # A bad/bookmarked value degrades to the default window rather
            # than 500-ing; we still tell the operator we ignored it so the
            # page never silently shows a different window than was asked for.
            since_invalid = True
    stats = collect_stats(session, since=since, now=now)
    # The 15s htmx poll re-requests this route and swaps ONLY the metrics
    # fragment; the task-card queries (issue #117 Phase C) feed the static full
    # page, never that fragment, so skip them on the poll rather than paying for
    # two queries whose results would be discarded.
    is_htmx = bool(request.headers.get("HX-Request"))
    context = {
        "request": request,
        "stats": stats,
        # Curated hardware strip (W3): rendered inside the htmx-swapped
        # fragment so it refreshes on the same 15s poll as the run figures.
        "resource_strip": build_resource_strip(_resource_snapshot(request)),
        "active_nav": "dashboard",
        # Iterate the enum (not the sparse status_counts map) so the status
        # table renders in a stable order and zero-fills empty statuses, the
        # same contract format_stats_text/render_prometheus hold.
        "run_statuses": list(RunStatus),
        # Carry the accepted window through the 15s htmx poll so a custom
        # ?since= isn't lost on the first refresh. Only echo a value we
        # actually honored (an invalid one falls back to the default, so we
        # drop it from the poll URL too).
        "since_param": "" if since_invalid or not raw_since else raw_since,
        "since_invalid": since_invalid,
    }
    if not is_htmx:
        # The count of runs actually eligible for review, sharing the queue's
        # own predicate (issue #117). It powers the static "Continue review (N)"
        # task card; review_backlog_count derives from adjudication_queue and
        # cannot drift from it. It is deliberately the ONLY copy of this number
        # on the page — no live stat-card duplicate that could contradict it.
        context["review_backlog"] = review_backlog_count(session)
        # The "Last finished run" task card: newest run by terminal-stage
        # completion, None when nothing has finished (honest empty state).
        context["last_completed"] = latest_completed_run(session)
    template = "fragments/dashboard_metrics.html" if is_htmx else "dashboard.html"
    return templates.TemplateResponse(request, template, context)

# ---- Hardware resource page (hardware-aware W3) -----------------------------
# The fuller live view behind the dashboard strip: the aggregated GPU card
# (utilization/VRAM/temperature labeled INSTANTANEOUS, peak temp + throttle
# events labeled cumulative-since-restart), per-service admission, and the
# curated warnings. Reads the same cached snapshot as /metrics and the strip,
# so the three cannot disagree. An htmx poll swaps just the fragment, like
# /dashboard. Protected like every non-/healthz page.

@dashboards_router.get("/resources")
def resources(request: Request, operator: OperatorDep) -> Response:
    snapshot = _resource_snapshot(request)
    context = {
        "request": request,
        "snapshot": snapshot,
        "resource_strip": build_resource_strip(snapshot),
        "active_nav": "resources",
    }
    template = (
        "fragments/resource_status.html"
        if request.headers.get("HX-Request")
        else "resources.html"
    )
    return templates.TemplateResponse(request, template, context)


@tail_router.get("/runs/{run_id}/assets")
def run_assets_fragment(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    _run_or_404(session, run_id)
    return _run_assets_response(request, session, run_id)

@tail_router.post("/runs/{run_id}/assets/generate")
def run_assets_generate(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    kind: Annotated[str | None, Form()] = None,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Start generation jobs — one kind, or all three when none is named.

    Kinds with an active job are skipped (create_jobs maps the partial
    unique index to a per-kind skip), so "generate all" degrades per kind.
    Commit-before-publish like every enqueue."""
    _require_csrf(request, CSRF_ASSETS_GENERATE, csrf_token)
    _run_or_404(session, run_id)
    settings: Settings = request.app.state.settings
    if kind is None:
        kinds = tuple(RunAssetKind)
    else:
        try:
            kinds = (RunAssetKind(kind),)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"unknown asset kind {kind!r}"
            ) from exc
    try:
        created, _skipped = create_jobs(
            session, pipeline_run_id=run_id, kinds=kinds, settings=settings
        )
    except RunAssetJobError as exc:
        session.rollback()
        return _run_assets_response(request, session, run_id, error=str(exc))
    job_ids = [job.id for job in created]
    session.commit()
    deferred = sum(1 for job_id in job_ids if not _publish_run_asset_job(job_id))
    # Honest UX: a broker outage leaves real QUEUED rows with no recovery
    # sweep — say so instead of rendering a silently-stuck spinner.
    notice = (
        "worker broker unavailable — the queued job(s) will not start;"
        " cancel and retry once the worker is back"
        if deferred
        else None
    )
    return _run_assets_response(request, session, run_id, error=notice)

@tail_router.post("/runs/{run_id}/assets/{job_id}/cancel")
def run_assets_cancel(
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Cancel: resolves a QUEUED job outright; a RUNNING one is re-checked
    after its LLM call, and a provably-dead one is force-cancelled."""
    _require_csrf(request, CSRF_ASSETS_CANCEL, csrf_token)
    _run_or_404(session, run_id)
    job = session.get(RunAssetJob, job_id)
    if job is None or job.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such asset job")
    request_asset_cancel(session, job_id)
    # Commit now so the executor's post-call check sees it immediately.
    session.commit()
    return _run_assets_response(request, session, run_id)

@tail_router.get("/runs/{run_id}/translation")
def run_translation_fragment(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    _run_or_404(session, run_id)
    return _run_translation_response(request, session, run_id)

@tail_router.post("/runs/{run_id}/translation/generate")
def run_translation_generate(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    target_language: Annotated[str | None, Form(max_length=64)] = None,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Start one translation job for the run (#133).

    The form's language select defaults to the installation's preferred
    language; a per-run choice overrides it for this generation only.
    Same-language, no-transcript, and gates-off requests come back as
    plain-language card errors, not HTTP failures — the card is the
    operator's surface. The review stepper's Translate action (issue #133
    Slice B) asks for JSON instead (the island-json convention) and gets
    ``{started, error}`` — the same truthful outcomes, without parsing the
    card fragment. Commit-before-publish like every enqueue."""
    _require_csrf(request, CSRF_TRANSLATION_GENERATE, csrf_token)
    _run_or_404(session, run_id)
    settings: Settings = request.app.state.settings

    def respond(error: str | None, *, started: bool) -> Response:
        if _wants_island_json(request):
            return JSONResponse({"started": started, "error": error})
        return _run_translation_response(request, session, run_id, error=error)

    target = normalized_language(target_language)
    if target is None:
        target = normalized_language(
            resolve_effective_translation_target_language(
                get_app_settings(session), settings
            )
        )
    if target is None:
        return respond(
            "Pick a language to translate into (or set a preferred"
            " language in Settings → Translation).",
            started=False,
        )
    try:
        job, already_active = create_translation_job(
            session, pipeline_run_id=run_id, target_language=target, settings=settings
        )
    except TranslationJobError as exc:
        session.rollback()
        return respond(str(exc), started=False)
    if already_active or job is None:
        session.rollback()
        return respond("A translation is already in progress.", started=False)
    job_id = job.id
    session.commit()
    # Honest UX: a broker outage leaves a real QUEUED row with no recovery
    # sweep — say so instead of rendering a silently-stuck spinner.
    if _publish_translation_job(job_id):
        return respond(None, started=True)
    return respond(
        "worker broker unavailable — the queued job will not start;"
        " cancel and retry once the worker is back",
        started=False,
    )

@tail_router.post("/runs/{run_id}/translation/{job_id}/cancel")
def run_translation_cancel(
    run_id: uuid.UUID,
    job_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Cancel: resolves a QUEUED job outright; a RUNNING one is re-checked
    between LLM batches, and a provably-dead one is force-cancelled."""
    _require_csrf(request, CSRF_TRANSLATION_CANCEL, csrf_token)
    _run_or_404(session, run_id)
    job = session.get(TranslationJob, job_id)
    if job is None or job.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such translation job")
    request_translation_cancel(session, job_id)
    # Commit now so the executor's post-call check sees it immediately.
    session.commit()
    return _run_translation_response(request, session, run_id)

@tail_router.get("/media/{run_id}")
@tail_router.head("/media/{run_id}")
def media(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    settings: Settings = request.app.state.settings
    _run_or_404(session, run_id)
    gate = _get_media_gate(request)
    # ONE servability seam (issue #55): resolve_servable_media runs the same
    # reclaimed -> artifact -> gate checks the playback-capability predicate
    # uses, and carries the honest HTTP status on each failure (410 reclaimed,
    # 404 missing/unservable). Capability can therefore never advertise
    # seek_enabled while this route would 404/410.
    try:
        fh, size = resolve_servable_media(session, run_id, settings, gate)
    except MediaResolutionError as exc:
        raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
    try:
        byte_range = parse_range(request.headers.get("range"), size)
    except RangeNotSatisfiableError:
        fh.close()
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/wav"}
    if byte_range is None:
        status, start, length = 200, 0, size
    else:
        status, start, length = 206, byte_range.start, byte_range.length
        headers["Content-Range"] = f"bytes {byte_range.start}-{byte_range.end}/{size}"
    headers["Content-Length"] = str(length)
    if request.method == "HEAD":
        fh.close()
        return Response(status_code=status, headers=headers)
    return StreamingResponse(
        _stream_file(fh, start, length), status_code=status, headers=headers
    )

@tail_router.get("/media/{run_id}/peaks")
def media_peaks(
    run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    """Waveform amplitude envelope (issue #57): lazy compute, cached artifact.

    Cache trust: while the WAV is live, the cached row's source fingerprint
    is fstat-verified (prepare replaces the WAV before its DB commit, so a
    crash can strand a stale row) and a mismatch recomputes. A reclaimed WAV
    cannot be verified — the cache is served as-is so a static waveform
    still renders (derived evidence, like the transcript itself survives
    reclamation). No cache + no servable WAV answers with /media's honest
    status (410 reclaimed, 404 missing/unservable); a WAV that cannot yield
    trustworthy peaks answers 404 and caches nothing (fail closed).
    """
    settings = request.app.state.settings
    _run_or_404(session, run_id)
    media_root: Path = settings.media_root

    # Fast path: a trusted cache hit, no lock.
    cached = load_cached_peaks(session, run_id, media_root)
    if cached is not None and _peaks_cache_trusted(
        session, run_id, media_root, cached
    ):
        return _peaks_cache_response(request, cached.body, cached.artifact_id)

    # Slow path: serialize compute+publish per run with a transaction-scoped
    # advisory lock, so two first-paint tabs (possibly straddling a prepare
    # re-run) can never publish divergent bytes under one row/ETag. Held
    # until this request's transaction ends (the commit below). Key: a
    # stable 63-bit digest of the run UUID — same run, same lock, any worker.
    lock_key = run_id.int & 0x7FFFFFFFFFFFFFFF
    session.execute(select(func.pg_advisory_xact_lock(lock_key)))

    # Re-check under the lock: a racing request may have just populated the
    # cache while we waited, and its committed row is the canonical one.
    cached = load_cached_peaks(session, run_id, media_root)
    if cached is not None and _peaks_cache_trusted(
        session, run_id, media_root, cached
    ):
        return _peaks_cache_response(request, cached.body, cached.artifact_id)

    gate = _get_media_gate(request)
    try:
        fh, _size = resolve_servable_media(session, run_id, settings, gate)
    except MediaResolutionError as exc:
        raise HTTPException(
            status_code=exc.http_status, detail=f"waveform unavailable: {exc}"
        ) from exc
    try:
        fingerprint = SourceFingerprint.of_descriptor(fh)
        try:
            payload = compute_peaks(fh)
        except PeaksError as exc:
            raise HTTPException(
                status_code=404, detail=f"waveform unavailable: {exc}"
            ) from exc
    finally:
        fh.close()
    row_id = store_peaks(session, run_id, media_root, payload, fingerprint)
    session.commit()
    return _peaks_cache_response(request, payload.to_json_bytes(), row_id)

# Mount the gated routes last: every @protected route above is now attached to
# `app` behind require_onboarded, while the @app routes stay exempt.

