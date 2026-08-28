"""Media detail and editor page (Console 2.0 P3a, issue #156).

``GET /media/{media_id}`` is the media detail page that will host the editor
island (#157). Until the island ships, it renders a server-side transcript
fallback with run metadata, a run chooser, and progress state. The existing
``/review/{run_id}/*`` endpoints remain the only mutation surface; the editor
island will call them directly.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from voxint.adjudication.resolver import label_states
from voxint.adjudication.review_state import verified_progress
from voxint.adjudication.slots import ClaimMismatchError, verify_claim
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.api.csrf import (
    CSRF_ANNOTATION_TAGS,
    CSRF_CLIP_EXTRACT,
    mint_csrf_token,
)
from voxint.api.editor_query import media_detail
from voxint.api.playback import playback_capability
from voxint.api.presentation import friendly_media_label
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    _get_media_gate,
    require_media_enabled,
    require_onboarded,
    templates,
)
from voxint.api.routers.legacy_review import _annotation_limits, _annotations_payload
from voxint.api.speaker_colors import speaker_palette
from voxint.api.transcript_view import (
    _run_label_universe,
    _transcript_island_props,
)
from voxint.config import Settings
from voxint.db.models import PipelineRun, RunStatus
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
            palette = speaker_palette(_run_label_universe(session, run_id))
            verified_n, total = verified_progress(session, run_id)
            capability = playback_capability(
                session, selected_run_obj, settings, _get_media_gate(request)
            )

            island_props = _transcript_island_props(
                session, run_id, lines, palette, capability, settings
            )
            island_props["reviewToken"] = str(token) if claim_valid else None
            island_props["initialProgress"] = {
                "verified": verified_n,
                "total": total,
            }
            island_props["speakers"] = [
                {"id": str(sp.id), "displayName": sp.display_name}
                for sp in active_speakers(session)
            ]

            annotations_payload = _annotations_payload(session, run_id, [])
            island_props["annotations"] = annotations_payload["annotations"]
            island_props["annotationTags"] = annotations_payload["tags"]
            island_props["annotationLimits"] = _annotation_limits()
            states = label_states(session, run_id)
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
                    "cosineSpeakerName": s.cosine_speaker_name,
                    "cosineGrounded": s.cosine_grounded,
                    "llmHintName": s.llm_hint_name,
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
            "active_nav": "media",
        },
    )
