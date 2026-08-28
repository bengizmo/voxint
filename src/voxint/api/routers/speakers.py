"""Speakers area: roster curation (issue #7) and web research review (#42).

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151).
Every route rides the router-level onboarding gate; each mutation verifies its
per-action CSRF token before any write.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from voxint.api.csrf import (
    CSRF_PROFILE_DECISION,
    CSRF_RESEARCH_CANCEL,
    CSRF_RESEARCH_START,
    CSRF_ROSTER_ARCHIVE,
    CSRF_ROSTER_CREATE,
    CSRF_ROSTER_EMBEDDING_DELETE,
    CSRF_ROSTER_MERGE,
    CSRF_ROSTER_RENAME,
    CSRF_ROSTER_RESTORE,
    CSRF_SPEAKER_PROFILE_EDIT,
    mint_csrf_token,
)
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    _require_csrf,
    require_onboarded,
    require_speakers_enabled,
    templates,
)
from voxint.api.speakers_query import (
    SORT_LABELS,
    VIEWS,
    normalize_sort,
    normalize_view,
    speakers_overview,
)
from voxint.api.triage_view import _triage_for
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_source_authority_domains,
)
from voxint.config import Settings
from voxint.db.models import (
    ClaimField,
    EnrichmentCandidate,
    MediaItem,
    PipelineRun,
    ProfileDecision,
    ResearchJob,
    ResearchJobStatus,
    Speaker,
)
from voxint.enrichment.queries import (
    CandidateState,
    candidates_for_speaker,
)
from voxint.enrichment.research_jobs import (
    ResearchJobError,
    budget_snapshot,
    create_job,
    request_cancel,
    research_gates_open,
)
from voxint.enrichment.review import ConflictingReplayError as EnrichmentReplayError
from voxint.enrichment.review import StaleCandidateError, record_profile_decision
from voxint.enrichment.triage import TriageScore, parse_authority_domains
from voxint.speakers.aggregate import aggregate_for_speaker, enrollment_count
from voxint.speakers.matching import gates_from_settings
from voxint.speakers.profile import (
    ProfileFieldError,
    clear_profile_field,
    profile_for,
    set_profile_field,
)
from voxint.speakers.roster import (
    RosterError,
    RosterNotFoundError,
    alias_ids,
    archive_speaker,
    canonicalize,
    create_speaker,
    delete_embedding,
    merge_map,
    merge_speakers,
    rename_speaker,
    restore_speaker,
    roster_overview,
    voiceprint_bars,
)
from voxint.speakers.tiers import evidence_for, tier_for

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_onboarded)])

def _publish_research_job(job_id: uuid.UUID) -> bool:
    """Enqueue a committed research job, returning False on a broker outage.

    Mirrors ``_publish_or_defer``, minus the recovery sweep: research jobs have
    none (v1 — hidden re-execution of a non-deterministic loop is worse than a
    visible stall), so the console shows a deferred job as queued with its age
    and the operator cancels and retries."""
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import research_speaker

    try:
        research_speaker.apply_async((str(job_id),), ignore_result=True)
    except OperationalError:
        logger.warning(
            "research enqueue deferred (broker unavailable); job %s stays QUEUED",
            job_id,
            exc_info=True,
        )
        return False
    return True


# The claim fields the web-research review surface serves; NAME stays on the
# workbench's dedicated suggestion flow.
_PROFILE_FIELDS = (ClaimField.BIO.value, ClaimField.AFFILIATION.value, ClaimField.LINK.value)
_ACTIVE_JOB_STATUSES = (ResearchJobStatus.QUEUED.value, ResearchJobStatus.RUNNING.value)


def _research_state(
    session: Session,
    settings: Settings,
    speaker: Speaker,
    error: str | None = None,
    *,
    include_aliases: bool = False,
) -> dict[str, Any]:
    """One speaker's research block: latest job, budgets, reviewable drafts.

    ``include_aliases`` (the Console 2.0 surfaces, #159) also gathers drafts
    recorded under ids since merged into this speaker — the accept path
    canonicalizes on write anyway. The legacy roster keeps the single-id read
    so the flag-off page stays exactly as it always rendered.
    """
    job = session.execute(
        select(ResearchJob)
        .where(ResearchJob.speaker_id == speaker.id)
        .order_by(ResearchJob.created_at.desc(), ResearchJob.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    sids = (
        sorted(alias_ids(session, speaker.id), key=str) if include_aliases else [speaker.id]
    )
    views = [
        view
        for sid in sids
        for view in candidates_for_speaker(session, sid)
        if view.candidate.field in _PROFILE_FIELDS
    ]
    row = get_app_settings(session)
    authority = parse_authority_domains(
        resolve_effective_source_authority_domains(row, settings)
    )
    triage: dict[uuid.UUID, TriageScore] = {
        view.candidate.id: _triage_for(view, voice=None, peer_count=1, authority=authority)
        for view in views
        if view.state is CandidateState.PROPOSED
    }
    # The unresolved bucket: proposed (undecided) drafts, highest review priority
    # first (#42). No score floor — a floor is an uncalibrated implicit reject.
    proposed = sorted(
        (v for v in views if v.state is CandidateState.PROPOSED),
        key=lambda v: (
            -triage[v.candidate.id].priority,
            v.candidate.value.casefold(),
            v.candidate.created_at,
        ),
    )
    return {
        "speaker": speaker,
        "job": job,
        "job_active": job is not None and job.status in _ACTIVE_JOB_STATUSES,
        "gates_open": research_gates_open(settings, row),
        "budget": budget_snapshot(settings),
        "proposed": proposed,
        "triage": triage,
        "decided_count": sum(
            1 for v in views if v.state in (CandidateState.ACCEPTED, CandidateState.REJECTED)
        ),
        "error": error,
    }


def _research_csrf(request: Request) -> dict[str, Any]:
    secret = request.app.state.csrf_secret
    return {
        "csrf_research_start": mint_csrf_token(secret, CSRF_RESEARCH_START),
        "csrf_research_cancel": mint_csrf_token(secret, CSRF_RESEARCH_CANCEL),
        "csrf_profile_decision": mint_csrf_token(secret, CSRF_PROFILE_DECISION),
        "nonce": lambda: uuid.uuid4().hex,
    }


def _research_qs(request: Request) -> str:
    """The page-context query string research URLs round-trip (#159): the
    profile page's research forms carry ``?page=profile`` so their responses
    keep rendering profile-aware URLs (and the decision response knows to
    refresh the profile panel). Everywhere else: empty, markup unchanged."""
    return "?page=profile" if request.query_params.get("page") == "profile" else ""


def _research_response(
    request: Request, session: Session, speaker: Speaker, error: str | None = None
) -> Response:
    """The per-speaker research fragment — the polling target and every
    research mutation's response."""
    return templates.TemplateResponse(
        request,
        "speakers/research.html",
        {
            "request": request,
            "research": _research_state(
                session,
                request.app.state.settings,
                speaker,
                error,
                include_aliases=_speakers_flag_on(request),
            ),
            "research_qs": _research_qs(request),
            **_research_csrf(request),
        },
    )


def _roster_context(request: Request, session: Session, error: str | None = None) -> dict[str, Any]:
    """Template context for the roster page and its htmx fragment."""
    overview = roster_overview(session)
    secret = request.app.state.csrf_secret
    settings: Settings = request.app.state.settings
    return {
        "request": request,
        "overview": overview,
        "voiceprints": {
            entry.speaker.id: voiceprint_bars(entry.embeddings) for entry in overview.active
        },
        "roster_error": error,
        "active_nav": "speakers",
        "csrf_rename": mint_csrf_token(secret, CSRF_ROSTER_RENAME),
        "csrf_merge": mint_csrf_token(secret, CSRF_ROSTER_MERGE),
        "csrf_archive": mint_csrf_token(secret, CSRF_ROSTER_ARCHIVE),
        "csrf_restore": mint_csrf_token(secret, CSRF_ROSTER_RESTORE),
        "csrf_embedding_delete": mint_csrf_token(secret, CSRF_ROSTER_EMBEDDING_DELETE),
        "research_by_speaker": {
            entry.speaker.id: _research_state(session, settings, entry.speaker)
            for entry in overview.active
        },
        **_research_csrf(request),
    }


def _speakers_flag_on(request: Request) -> bool:
    settings: Settings = request.app.state.settings
    return bool(settings.console_speakers_enabled)


def _overview_context(
    request: Request, session: Session, error: str | None = None
) -> dict[str, Any]:
    """Template context for the Console 2.0 overview page and its fragment.

    Sort/view come from the query string (POST forms carry them in their
    action URLs so a mutation re-render preserves both); unknown values
    degrade to the defaults, never a 422.
    """
    sort = normalize_sort(request.query_params.get("sort"))
    view = normalize_view(request.query_params.get("view"))
    settings: Settings = request.app.state.settings
    secret = request.app.state.csrf_secret
    overview = speakers_overview(session, gates_from_settings(settings), sort=sort)
    reminder_run_ids = {r.run_id for r in overview.name_suggestions}
    reminder_runs = (
        {
            run.id: run
            for run in session.execute(
                select(PipelineRun)
                .where(PipelineRun.id.in_(reminder_run_ids))
                .options(
                    selectinload(PipelineRun.media_item).selectinload(
                        MediaItem.source_metadata
                    )
                )
            ).scalars()
        }
        if reminder_run_ids
        else {}
    )
    return {
        "request": request,
        "overview": overview,
        "reminder_runs": reminder_runs,
        "voiceprints": {
            row.entry.speaker.id: voiceprint_bars(row.entry.embeddings)
            for row in overview.rows
        },
        "sort": sort,
        "sorts": SORT_LABELS,
        "view": view,
        "views": VIEWS,
        "roster_error": error,
        "active_nav": "speakers",
        "now": datetime.now(UTC),
        "csrf_create": mint_csrf_token(secret, CSRF_ROSTER_CREATE),
        "csrf_rename": mint_csrf_token(secret, CSRF_ROSTER_RENAME),
        "csrf_merge": mint_csrf_token(secret, CSRF_ROSTER_MERGE),
        "csrf_archive": mint_csrf_token(secret, CSRF_ROSTER_ARCHIVE),
        "csrf_restore": mint_csrf_token(secret, CSRF_ROSTER_RESTORE),
    }


def _roster_response(request: Request, session: Session, error: str | None = None) -> Response:
    """Post-mutation response, mirroring ``_labels_response``: htmx gets the
    refreshed roster fragment (operator errors rendered inline), a plain form
    POST gets a 303 back to the page — or the full page when it carries an
    error to show. CSRF/auth failures never come here; they stay real 403s.

    Flag-aware (#159): with ``console_speakers_enabled`` on, every branch
    renders the new overview skin (fragment / page / redirect), preserving the
    normalized ``?sort``/``?view`` the mutating form carried; off, the legacy
    roster renders exactly as it always has.
    """
    if not _speakers_flag_on(request):
        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                request, "speakers/roster.html", _roster_context(request, session, error)
            )
        if error is not None:
            return templates.TemplateResponse(
                request, "speakers/speakers.html", _roster_context(request, session, error)
            )
        return RedirectResponse("/speakers", status_code=303)
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "speakers/overview_roster.html",
            _overview_context(request, session, error),
        )
    if error is not None:
        return templates.TemplateResponse(
            request, "speakers/overview.html", _overview_context(request, session, error)
        )
    sort = normalize_sort(request.query_params.get("sort"))
    view = normalize_view(request.query_params.get("view"))
    return RedirectResponse(f"/speakers?sort={sort}&view={view}", status_code=303)


# ---- Speaker roster curation (issue #7) ------------------------------------
# View, rename, merge, archive/restore, and remove enrollment embeddings.
# The append-only decision ledger is never written here — every mutation goes
# through speakers.roster, which curates only the mutable side. Each POST
# verifies its own per-action CSRF token before any write; operator-level
# refusals (RosterError) re-render the roster with the message inline, while
# missing speakers/embeddings stay real 404s.

@router.get("/speakers")
def speakers_page(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    # Content branch, not an access gate (#159): /speakers shipped long before
    # Console 2.0, so the flag swaps WHICH page renders — off keeps the legacy
    # roster byte-identical, on renders the new overview.
    if _speakers_flag_on(request):
        return templates.TemplateResponse(
            request, "speakers/overview.html", _overview_context(request, session)
        )
    return templates.TemplateResponse(
        request, "speakers/speakers.html", _roster_context(request, session)
    )

@router.post("/speakers")
def speaker_create(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    display_name: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_CREATE, csrf_token)
    try:
        speaker = create_speaker(session, display_name)
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    if request.headers.get("HX-Request"):
        return _roster_response(request, session)
    target = f"/speakers/{speaker.id}" if _speakers_flag_on(request) else "/speakers"
    return RedirectResponse(target, status_code=303)

@router.post("/speakers/{speaker_id}/rename")
def speaker_rename(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    display_name: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_RENAME, csrf_token)
    try:
        rename_speaker(session, speaker_id, display_name)
    except RosterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    return _roster_response(request, session)

@router.post("/speakers/{speaker_id}/merge")
def speaker_merge(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    target_id: Annotated[uuid.UUID, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_MERGE, csrf_token)
    try:
        merge_speakers(session, speaker_id, target_id)
    except RosterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    return _roster_response(request, session)

@router.post("/speakers/{speaker_id}/archive")
def speaker_archive(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_ARCHIVE, csrf_token)
    try:
        archive_speaker(session, speaker_id)
    except RosterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    return _roster_response(request, session)

@router.post("/speakers/{speaker_id}/restore")
def speaker_restore(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_RESTORE, csrf_token)
    try:
        restore_speaker(session, speaker_id)
    except RosterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    return _roster_response(request, session)

@router.post("/speakers/{speaker_id}/embeddings/{embedding_id}/delete")
def speaker_embedding_delete(
    speaker_id: uuid.UUID,
    embedding_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_ROSTER_EMBEDDING_DELETE, csrf_token)
    try:
        delete_embedding(session, speaker_id, embedding_id)
    except RosterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RosterError as exc:
        session.rollback()
        return _roster_response(request, session, error=str(exc))
    return _roster_response(request, session)

# ---- Web-research jobs + profile-draft review (issue #40) -----------------
# All research mutations answer with the per-speaker fragment; the fragment
# re-polls itself (hx-trigger="every 3s") only while its job is active, so
# polling stops the moment a terminal render goes out.

def _speaker_or_404(session: Session, speaker_id: uuid.UUID) -> Speaker:
    speaker = session.get(Speaker, speaker_id)
    if speaker is None:
        raise HTTPException(status_code=404, detail="no such speaker")
    return speaker

@router.get("/speakers/{speaker_id}/research")
def research_fragment(
    speaker_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    return _research_response(request, session, _speaker_or_404(session, speaker_id))

@router.post("/speakers/{speaker_id}/research/start")
def research_start(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    operator_note: Annotated[str | None, Form(max_length=1000)] = None,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Start one budgeted research job for this speaker.

    The rendered form is the budget preview the operator approved; those
    budgets are snapshotted onto the job. Commit-before-publish like every
    enqueue: a broker outage leaves an honest QUEUED job (no hidden
    recovery — the operator cancels and retries)."""
    _require_csrf(request, CSRF_RESEARCH_START, csrf_token)
    speaker = _speaker_or_404(session, speaker_id)
    # Archived speakers are read-only (#159 review): the pages hide the form,
    # and a stale tab's POST must not spend research budget on one either.
    if speaker.deleted_at is not None:
        raise HTTPException(
            status_code=409,
            detail="speaker is archived; restore it before starting research",
        )
    settings: Settings = request.app.state.settings
    if (
        session.execute(
            select(ResearchJob.id)
            .where(
                ResearchJob.speaker_id == speaker_id,
                ResearchJob.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .limit(1)
        ).scalar_one_or_none()
        is not None
    ):
        return _research_response(
            request, session, speaker, error="a research job is already active"
        )
    try:
        job = create_job(
            session,
            speaker_id=speaker_id,
            settings=settings,
            operator_note=operator_note,
        )
    except ResearchJobError as exc:
        session.rollback()
        return _research_response(request, session, speaker, error=str(exc))
    except IntegrityError:
        # The DB's one-active-job-per-speaker partial unique index caught a
        # start the friendly pre-check raced past (double-submit, two tabs).
        session.rollback()
        return _research_response(
            request, session, speaker, error="a research job is already active"
        )
    job_id = job.id
    session.commit()
    _publish_research_job(job_id)
    return _research_response(request, session, speaker)

@router.post("/speakers/{speaker_id}/research/{job_id}/cancel")
def research_cancel(
    speaker_id: uuid.UUID,
    job_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Cooperative cancel: the loop stops before its next round."""
    _require_csrf(request, CSRF_RESEARCH_CANCEL, csrf_token)
    speaker = _speaker_or_404(session, speaker_id)
    job = session.get(ResearchJob, job_id)
    if job is None or job.speaker_id != speaker_id:
        raise HTTPException(status_code=404, detail="no such research job")
    request_cancel(session, job_id)
    # Commit now so the worker's between-rounds check sees it immediately,
    # not after this response finishes rendering.
    session.commit()
    return _research_response(request, session, speaker)

@router.post("/speakers/{speaker_id}/research/candidates/{candidate_id}/decision")
def decide_profile_candidate(
    speaker_id: uuid.UUID,
    candidate_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    verdict: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Accept/reject one profile claim — a review record, never identity.

    Field-by-field: each candidate row carries one field/value and gets
    its own terminal ruling. Writes the profile-review trail only."""
    _require_csrf(request, CSRF_PROFILE_DECISION, csrf_token)
    speaker = _speaker_or_404(session, speaker_id)
    # Archived speakers are read-only (#159 review): rulings that could
    # materialize profile rows wait until the speaker is restored. Cancelling
    # an already-running job stays allowed.
    if speaker.deleted_at is not None:
        raise HTTPException(
            status_code=409,
            detail="speaker is archived; restore it before deciding drafts",
        )
    try:
        decision = ProfileDecision(verdict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown verdict {verdict!r}") from exc
    candidate = session.get(EnrichmentCandidate, candidate_id)
    # This surface serves speaker-scoped profile fields only; NAME stays on
    # the workbench's suggestion flow. Alias-aware (#159): a draft recorded
    # under an id since merged into this speaker is still this speaker's.
    if (
        candidate is None
        or candidate.speaker_id is None
        or candidate.speaker_id not in alias_ids(session, speaker_id)
        or candidate.field not in _PROFILE_FIELDS
    ):
        raise HTTPException(status_code=404, detail="no such candidate for this speaker")
    try:
        record_profile_decision(
            session,
            candidate_id=candidate_id,
            decision=decision,
            operator=operator,
            idempotency_key=nonce,
        )
    except StaleCandidateError as exc:
        raise HTTPException(
            status_code=409,
            detail="superseded by a newer research run — refresh and re-review",
        ) from exc
    except EnrichmentReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if _research_qs(request):
        # Profile-page context (#159): an accept may have materialized a
        # profile field, so the htmx response carries the refreshed profile
        # panel out-of-band alongside the research fragment; without JS the
        # 303 re-renders the whole page.
        if not request.headers.get("HX-Request"):
            return RedirectResponse(f"/speakers/{speaker.id}", status_code=303)
        research_html = templates.env.get_template("speakers/research.html").render(
            research=_research_state(
                session, request.app.state.settings, speaker, include_aliases=True
            ),
            research_qs=_research_qs(request),
            **_research_csrf(request),
        )
        panel_html = templates.env.get_template("speakers/profile_panel.html").render(
            _profile_panel_context(request, session, speaker), profile_oob=True
        )
        return HTMLResponse(research_html + panel_html)
    return _research_response(request, session, speaker)



# ---- Speaker profile page (Console 2.0 P4, issue #159) ---------------------
# New routes, so they ride the area gate (404 while console_speakers_enabled
# is off) — unlike GET /speakers, which is live and branches skins instead.


def _profile_panel_context(
    request: Request, session: Session, speaker: Speaker, error: str | None = None
) -> dict[str, Any]:
    """Context for the profile-panel fragment (page include, htmx re-render,
    and the research-decision out-of-band refresh)."""
    secret = request.app.state.csrf_secret
    return {
        "speaker": speaker,
        "profile": profile_for(session, speaker.id),
        "profile_fields": _PROFILE_FIELDS,
        "archived": speaker.deleted_at is not None,
        "profile_error": error,
        "csrf_profile_edit": mint_csrf_token(secret, CSRF_SPEAKER_PROFILE_EDIT),
    }


def _profile_context(
    request: Request, session: Session, speaker: Speaker, error: str | None = None
) -> dict[str, Any]:
    """The full profile page's context: stats, tier evidence, profile panel,
    research block, and the associated-recordings table."""
    settings: Settings = request.app.state.settings
    aggregate = aggregate_for_speaker(session, speaker.id)
    gates = gates_from_settings(settings)
    evidence = evidence_for(session, aggregate.grounded_keys)
    run_ids = {a.run_id for a in aggregate.appearances}
    runs_by_id = {
        run.id: run
        for run in session.execute(
            select(PipelineRun)
            .where(PipelineRun.id.in_(run_ids))
            .options(
                selectinload(PipelineRun.media_item).selectinload(
                    MediaItem.source_metadata
                ),
                selectinload(PipelineRun.media_item).selectinload(
                    MediaItem.media_folder
                )
            )
        ).scalars()
    } if run_ids else {}
    return {
        "request": request,
        "aggregate": aggregate,
        "tier": tier_for(evidence, gates),
        "gates": gates,
        "enrollments": enrollment_count(session, speaker.id),
        "runs_by_id": runs_by_id,
        "research": _research_state(session, settings, speaker, include_aliases=True),
        "research_qs": "?page=profile",
        "active_nav": "speakers",
        "now": datetime.now(UTC),
        **_profile_panel_context(request, session, speaker, error),
        **_research_csrf(request),
    }


@router.get("/speakers/{speaker_id}", dependencies=[Depends(require_speakers_enabled)])
def speaker_profile_page(
    speaker_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    speaker = _speaker_or_404(session, speaker_id)
    if speaker.merged_into_id is not None:
        # A merged-away id is a tombstone, not a page: follow the merge chain
        # so old bookmarks land on the speaker that absorbed it.
        canonical = canonicalize(speaker.id, merge_map(session))
        return RedirectResponse(f"/speakers/{canonical}", status_code=303)
    return templates.TemplateResponse(
        request, "speakers/profile.html", _profile_context(request, session, speaker)
    )


@router.post(
    "/speakers/{speaker_id}/profile", dependencies=[Depends(require_speakers_enabled)]
)
def speaker_profile_edit(
    speaker_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    field: Annotated[str, Form()],
    value: Annotated[str | None, Form()] = None,
    action: Annotated[str, Form()] = "save",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Manually set or clear one profile field (#159).

    ``action=clear`` removes the field; anything else saves ``value``. Writes
    go through ``speakers/profile.py`` (canonical-speaker lock inside), so a
    concurrent accept, merge, or second edit serializes cleanly. Operator
    mistakes (bad field, empty value) re-render the panel inline; archived
    speakers are read-only and refuse with a 409."""
    _require_csrf(request, CSRF_SPEAKER_PROFILE_EDIT, csrf_token)
    speaker = _speaker_or_404(session, speaker_id)
    if speaker.merged_into_id is not None:
        speaker = _speaker_or_404(session, canonicalize(speaker.id, merge_map(session)))
    if speaker.deleted_at is not None:
        raise HTTPException(
            status_code=409,
            detail="speaker is archived; restore it before editing the profile",
        )
    try:
        if action == "clear":
            clear_profile_field(
                session, speaker_id=speaker.id, field=field, operator=operator
            )
        else:
            set_profile_field(
                session,
                speaker_id=speaker.id,
                field=field,
                value=value or "",
                operator=operator,
            )
    except ProfileFieldError as exc:
        session.rollback()
        return _profile_edit_response(request, session, speaker, error=str(exc))
    return _profile_edit_response(request, session, speaker)


def _profile_edit_response(
    request: Request, session: Session, speaker: Speaker, error: str | None = None
) -> Response:
    """Post-edit response: htmx swaps the refreshed panel (errors inline), a
    plain form POST gets a 303 back to the page — or the full page when it
    carries an error to show. Mirrors ``_roster_response``."""
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "speakers/profile_panel.html",
            _profile_panel_context(request, session, speaker, error),
        )
    if error is not None:
        return templates.TemplateResponse(
            request, "speakers/profile.html", _profile_context(request, session, speaker, error)
        )
    return RedirectResponse(f"/speakers/{speaker.id}", status_code=303)
