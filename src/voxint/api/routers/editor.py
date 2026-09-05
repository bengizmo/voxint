"""Media detail and editor page (Console 2.0 P3a/P3b, issues #156/#157).

``GET /media/{media_id}/editor`` serves the media-editor island: run metadata,
a run chooser, and — for a completed run — the editor with transcript, speaker
rail, annotations, and waveform. The existing ``/review/{run_id}/*`` endpoints
are the mutation surface; the island calls them directly.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from voxint.adjudication.resolver import label_states
from voxint.adjudication.review_state import verified_progress
from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    refresh_run_claim,
    release_run,
    verify_claim,
)
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.api.csrf import (
    CSRF_ANNOTATION_TAGS,
    CSRF_CLAIM,
    CSRF_CLIP_EXTRACT,
    CSRF_RESTART,
    CSRF_TRANSLATION_GENERATE,
    mint_csrf_token,
)
from voxint.api.editor_query import media_detail
from voxint.api.languages import language_label
from voxint.api.playback import playback_capability
from voxint.api.presentation import friendly_media_label
from voxint.api.routers.deps import (
    _TRANSLATION_ACTIVE_STATUSES,
    OperatorDep,
    SessionDep,
    _get_media_gate,
    _reject_if_archived,
    _require_csrf,
    require_media_enabled,
    require_onboarded,
    templates,
)
from voxint.api.annotation_view import annotation_limits, annotations_payload
from voxint.api.speaker_colors import run_label_universe, speaker_palette
from voxint.api.transcript_view import _transcript_island_props
from voxint.app_settings import get_app_settings, resolve_effective_translation_target_language
from voxint.config import Settings
from voxint.db.models import PipelineRun, RunStatus
from voxint.enrichment.translation_jobs import (
    active_or_last_job as active_or_last_translation_job,
)
from voxint.enrichment.translation_jobs import (
    normalized_language,
    translation_gates_open,
)
from voxint.ingest import restart_impact
from voxint.speakers.matching import gates_from_settings
from voxint.speakers.roster import active_speakers

router = APIRouter(
    dependencies=[Depends(require_onboarded), Depends(require_media_enabled)]
)


@router.get("/media/{media_id}/editor", name="media_detail")
def media_detail_page(
    media_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    run: uuid.UUID | None = None,
    token: uuid.UUID | None = None,
) -> Response:
    """The media detail page with run selection and transcript display.

    Defaults to the latest completed run; ``?run=`` overrides. An optional
    ``?token=`` is verified against the selected run for edit capability;
    stale or absent tokens render read-only.
    """
    detail = media_detail(session, media_id, run_override=run)
    if detail is None:
        raise HTTPException(status_code=404, detail="not found")

    settings: Settings = request.app.state.settings
    media_label = friendly_media_label(None, detail.media.source_path)

    island_props: dict[str, object] | None = None
    verified_n = 0
    total = 0
    selected_run_obj: PipelineRun | None = None
    claim_valid = False

    if detail.selected_run is not None:
        run_id = detail.selected_run.id
        selected_run_obj = session.get(PipelineRun, run_id)
        if selected_run_obj is None:
            raise HTTPException(status_code=404, detail="not found")

        if detail.selected_run.status == RunStatus.COMPLETED.value:
            if token is not None:
                try:
                    verify_claim(session, run_id, token)
                    claim_valid = True
                except ClaimMismatchError:
                    token = None

            lines = attributed_transcript(
                session, run_id, text=TranscriptText.CORRECTED
            )
            palette = speaker_palette(run_label_universe(session, run_id))
            verified_n, total = verified_progress(session, run_id)
            capability = playback_capability(
                session, selected_run_obj, settings, _get_media_gate(request)
            )

            island_props = _transcript_island_props(
                session, run_id, lines, palette, capability, settings
            )
            island_props["mediaId"] = str(media_id)
            island_props["reviewToken"] = str(token) if claim_valid else None
            island_props["initialProgress"] = {
                "verified": verified_n,
                "total": total,
            }
            island_props["speakers"] = [
                {"id": str(sp.id), "displayName": sp.display_name}
                for sp in active_speakers(session)
            ]

            ann_payload = annotations_payload(session, run_id, [])
            island_props["annotations"] = ann_payload["annotations"]
            island_props["annotationTags"] = ann_payload["tags"]
            island_props["annotationLimits"] = annotation_limits()
            states = label_states(session, run_id, gates=gates_from_settings(settings))
            island_props["labelStates"] = [
                {
                    "label": s.label,
                    "paletteIndex": palette.get(s.label),
                    "turnCount": s.turn_count,
                    "totalSeconds": s.total_seconds,
                    "resolution": s.resolution.value,
                    "speakerId": str(s.speaker_id) if s.speaker_id else None,
                    "speakerName": s.speaker_name,
                    "cosineConfidence": s.cosine_confidence,
                    "cosineSpeakerId": str(s.cosine_speaker_id) if s.cosine_speaker_id else None,
                    "cosineSpeakerName": s.cosine_speaker_name,
                    "cosineGrounded": s.cosine_grounded,
                    "llmHintName": s.llm_hint_name,
                    "band": s.band.value if s.band else None,
                    "bandReason": s.band_reason,
                    "candidatePromptAllowed": s.candidate_prompt_allowed,
                    "matchDecision": s.match_decision,
                    "matchReason": s.match_reason,
                    "matchMargin": s.match_margin,
                    "matchEligibleSeconds": s.match_eligible_seconds,
                }
                for s in states
            ]

            if claim_valid:
                island_props["tagCsrf"] = mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_ANNOTATION_TAGS
                )
                island_props["clipCsrf"] = mint_csrf_token(
                    request.app.state.csrf_secret, CSRF_CLIP_EXTRACT
                )

            translate_row = get_app_settings(session)
            if translation_gates_open(settings, translate_row):
                detected = normalized_language(selected_run_obj.detected_language)
                preferred = normalized_language(
                    resolve_effective_translation_target_language(translate_row, settings)
                )
                translate_job = active_or_last_translation_job(session, run_id)
                default_target = (
                    preferred if preferred is not None and preferred != detected else None
                )
                island_props["translate"] = {
                    "csrf": mint_csrf_token(
                        request.app.state.csrf_secret, CSRF_TRANSLATION_GENERATE
                    ),
                    "defaultTarget": default_target,
                    "defaultTargetLabel": (
                        language_label(default_target) if default_target else None
                    ),
                    "active": translate_job is not None
                    and translate_job.status in _TRANSLATION_ACTIVE_STATUSES,
                    "runAnchor": f"/runs/{run_id}#run-translation-{run_id}",
                    "transcriptUrl": f"/runs/{run_id}/transcript",
                }
            else:
                island_props["translate"] = None

    if island_props is not None:
        island_props["claimCsrf"] = mint_csrf_token(
            request.app.state.csrf_secret, CSRF_CLAIM
        )
        island_props["multiUser"] = settings.voxint_multi_user

    csrf_restart = (
        mint_csrf_token(request.app.state.csrf_secret, CSRF_RESTART)
        if selected_run_obj is not None
        else None
    )
    _TERMINAL = {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    _run_restartable = (
        selected_run_obj is not None
        and selected_run_obj.status in _TERMINAL
        and selected_run_obj.archived_at is None
    )
    ri = (
        restart_impact(session, selected_run_obj.id)
        if selected_run_obj is not None and _run_restartable
        else None
    )

    return templates.TemplateResponse(
        request,
        "editor/detail.html",
        {
            "request": request,
            "detail": detail,
            "media_label": media_label,
            "selected_run": selected_run_obj,
            "island_props": island_props,
            "token": token if claim_valid else None,
            "progress": {"verified": verified_n, "total": total},
            "csrf_restart": csrf_restart,
            "restart_impact": ri,
            "active_nav": "media",
        },
    )


@router.post("/media/{media_id}/editor/claim")
def editor_claim(
    media_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    run_id: Annotated[uuid.UUID, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Claim a run for editing from the editor island (ADR 0004).

    CSRF-gated (no prior token exists to guard the POST). Returns the
    claim token as JSON so the island can adopt it without a page reload.
    """
    _require_csrf(request, CSRF_CLAIM, csrf_token)
    run = session.get(PipelineRun, run_id)
    if run is None or run.media_item_id != media_id:
        raise HTTPException(status_code=404, detail="not found")
    _reject_if_archived(run)
    settings: Settings = request.app.state.settings
    try:
        token = claim_run(
            session,
            run_id,
            reviewer=operator,
            ttl_seconds=settings.review_claim_ttl_seconds,
        )
    except ClaimUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({
        "token": str(token),
        "tagCsrf": mint_csrf_token(
            request.app.state.csrf_secret, CSRF_ANNOTATION_TAGS
        ),
        "clipCsrf": mint_csrf_token(
            request.app.state.csrf_secret, CSRF_CLIP_EXTRACT
        ),
    })


@router.post("/media/{media_id}/editor/refresh")
def editor_refresh(
    media_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    run_id: Annotated[uuid.UUID, Form()],
    token: Annotated[uuid.UUID, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Extend the active editor claim's TTL."""
    _require_csrf(request, CSRF_CLAIM, csrf_token)
    run = session.get(PipelineRun, run_id)
    if run is None or run.media_item_id != media_id:
        raise HTTPException(status_code=404, detail="not found")
    _reject_if_archived(run)
    settings: Settings = request.app.state.settings
    try:
        refresh_run_claim(
            session,
            run_id,
            token,
            ttl_seconds=settings.review_claim_ttl_seconds,
        )
    except ClaimMismatchError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"X-Voxint-Conflict": "claim"},
        ) from exc
    return JSONResponse({"ok": True})


@router.post("/media/{media_id}/editor/release")
def editor_release(
    media_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    run_id: Annotated[uuid.UUID, Form()],
    token: Annotated[uuid.UUID, Form()],
) -> Response:
    """Release a held claim from the editor island."""
    run = session.get(PipelineRun, run_id)
    if run is None or run.media_item_id != media_id:
        raise HTTPException(status_code=404, detail="not found")
    try:
        release_run(session, run_id, token)
    except ClaimMismatchError:
        return Response(status_code=204)
    except ClaimUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"released": True})
