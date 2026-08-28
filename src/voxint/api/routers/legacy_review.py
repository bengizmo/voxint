"""Legacy review area: the queue, workbench, exports, and annotation layer.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151).
Two routers, both behind the router-level onboarding gate: ``transcript_router``
carries only ``/review/{run_id}/transcript`` (it registers earlier than the
main block, between the run routes, and stays a separate router so the
registration order the P0b contract pins is preserved); ``router`` carries the
review queue, workbench, segment mutations, exports, and annotations. Console
2.0 later phases retire these routes behind redirects (the editor replaces
review), hence "legacy".
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.activity import record_speaker_identified, record_speaker_merge
from voxint.adjudication.annotations import (
    TIMING_WORD,
    AnnotationError,
    AnnotationIdempotencyError,
    AnnotationNotFoundError,
    AnnotationStaleError,
    AnnotationTagConflictError,
    AnnotationValidationError,
    CaptureEndpoint,
    CapturePayload,
    ResolvedAnnotation,
    annotations_for_run,
    capture_annotation,
    clip_lines_for_export,
    create_tag,
    derive_live_anchor,
    list_tags,
    live_annotation_or_404,
    load_covered_segments,
    normalize_note,
    reanchor_annotation,
    refresh_annotation,
    resolve_annotation_spans,
    resolve_tag_names,
    resolved_order_key,
    soft_delete_annotation,
    stored_anchor_from_derived,
    stored_anchor_from_row,
    tags_for_annotations,
    update_annotation,
    update_tag,
)
from voxint.adjudication.corrections_view import run_reconciliation
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import (
    ConflictingReplayError,
    WordRangeError,
    decision_exists,
    record_decision,
)
from voxint.adjudication.merge import MergeConflictError, MergeError, apply_merge, preview_merge
from voxint.adjudication.resolver import (
    QUEUE_SORTS,
    LabelState,
    Resolution,
    adjudication_queue,
    label_states,
    segment_states,
)
from voxint.adjudication.review_state import set_correction, set_verified, verified_progress
from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    release_run,
    verify_claim,
)
from voxint.adjudication.splits import (
    UnsplittableError,
    record_split,
    splittable_words,
    trace_has_entries,
)
from voxint.adjudication.transcript import (
    TranscriptLine,
    TranscriptText,
    attributed_transcript,
    effective_text,
    parse_transcript_text,
)
from voxint.api.clip_service import (
    ClipServiceError,
    clip_download_filename,
    generate_or_adopt_clip,
)
from voxint.api.csrf import (
    CSRF_ANNOTATION_TAGS,
    CSRF_CLAIM,
    CSRF_CLIP_EXTRACT,
    CSRF_TRANSLATION_GENERATE,
    mint_csrf_token,
)
from voxint.api.languages import LANGUAGE_NAMES, language_label
from voxint.api.model_provenance import select_run_model_identity
from voxint.api.playback import playback_capability, representative_turns
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
from voxint.api.routers.deps import run_source_title as _run_source_title
from voxint.api.speaker_colors import speaker_palette
from voxint.api.transcript_view import (
    _load_run_rule_index,
    _run_label_universe,
    _run_reconcile_response,
    _segment_child_ranges,
    _segment_is_corrected,
    _segment_is_split,
    _transcript_island_props,
    _wants_island_json,
)
from voxint.api.triage_view import _name_suggestions
from voxint.api.tutorial_view import _tutorial_banner
from voxint.app_settings import (
    get_app_settings,
    ready_tutorial_run_id,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_translation_target_language,
)
from voxint.config import Settings
from voxint.db.models import (
    HIGHLIGHT_PALETTE_SIZE,
    MAX_ANNOTATION_NOTE_CHARS,
    MAX_ANNOTATION_QUOTE_CHARS,
    MAX_ANNOTATION_SPAN_SEGMENTS,
    MAX_CORRECTED_TEXT_CHARS,
    MAX_TAG_NAME_CHARS,
    MAX_TAGS_PER_ANNOTATION,
    AnnotationTag,
    ArtifactKind,
    AudioArtifact,
    ClaimField,
    Decision,
    DiarizationTurn,
    EnrichmentCandidate,
    PipelineRun,
    ProfileDecision,
    SegmentReviewState,
    SegmentSplitBoundary,
    Speaker,
    StageRun,
    TranscriptAnnotation,
    TranscriptSegment,
)
from voxint.enrichment.drafts import EnrichmentScope
from voxint.enrichment.producers.names import PRODUCER_NAME as NAMES_PRODUCER
from voxint.enrichment.producers.names import NameProducerError, run_offline_name_producer
from voxint.enrichment.queries import CandidateState, latest_producer_run
from voxint.enrichment.review import ConflictingReplayError as EnrichmentReplayError
from voxint.enrichment.review import StaleCandidateError, record_profile_decision
from voxint.enrichment.translation_jobs import active_or_last_job as active_or_last_translation_job
from voxint.enrichment.translation_jobs import normalized_language, translation_gates_open
from voxint.enrichment.translations import (
    TranslationError,
    current_translation,
    load_translation_source,
    translation_source_hash,
    translation_texts,
)
from voxint.export import (
    ANNOTATION_BULK_SEPARATOR,
    ANNOTATION_MEDIA_TYPES,
    MEDIA_TYPES,
    TranscriptFormat,
    annotation_pull_quote,
    render_transcript,
    to_rttm,
)
from voxint.export.manifest import (
    ClipRef,
    QuoteLine,
    StageProvenance,
    StageRole,
    build_quote_bundle,
    build_quote_manifest,
)
from voxint.speakers.matching import gates_from_settings
from voxint.speakers.roster import active_speakers
from voxint.speakers.roster import is_active as roster_is_active
from voxint.tutorial.steps import TutorialPage

logger = logging.getLogger(__name__)


transcript_router = APIRouter(dependencies=[Depends(require_onboarded)])
router = APIRouter(dependencies=[Depends(require_onboarded)])


def _label_previews(
    session: Session, run_id: uuid.UUID, states: list[LabelState], limit: int
) -> dict[str, list[TranscriptSegment]]:
    previews: dict[str, list[TranscriptSegment]] = {}
    for state in states:
        previews[state.label] = list(
            session.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.diarization_label == state.label,
                )
                .order_by(TranscriptSegment.segment_index)
                .limit(limit)
            ).scalars()
        )
    return previews


def _workbench_context(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID | None,
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    states = label_states(session, run.id)
    # Assignable identities only — merged and archived speakers are curated out
    # of the roster and must not attract new decisions.
    speakers = active_speakers(session)
    name_hints_run, name_hints_labels, name_triage = _name_suggestions(session, run.id)
    # Per-turn playback (issue #49) + fail-closed seek gating (issue #55). The
    # workbench-player island (mounted OUTSIDE #labels) reads `capability` to
    # enable/disable the server-rendered, htmx-swapped seek buttons; the buttons
    # themselves carry the representative-turn timings for "preview this speaker".
    capability = playback_capability(session, run, settings, _get_media_gate(request))
    return {
        "name_hints_run": name_hints_run,
        "name_hints_labels": name_hints_labels,
        # Review-priority + component breakdown per representative (#42), keyed by
        # candidate id so labels.html can render it without changing the hint shape.
        "name_triage": name_triage,
        "names_enabled": resolve_effective_enrichment_names_enabled(
            get_app_settings(session), settings
        ),
        "names_last_run": latest_producer_run(session, NAMES_PRODUCER, EnrichmentScope.run(run.id)),
        # An accepted per-label suggestion prefills the Enroll input (editable,
        # never auto-submitted) — the one-click path from hint to enrollment.
        "enroll_prefill": {
            label: next(
                (view.candidate.value for view in views if view.state is CandidateState.ACCEPTED),
                "",
            )
            for label, views in name_hints_labels.items()
        },
        "request": request,
        "run": run,
        "states": states,
        # Per-speaker identity color (issue #50): one canonical map keyed on the
        # raw label, derived from the SAME run-label universe the transcript page
        # uses (turns and segments), so a label's color never drifts between the
        # two surfaces.
        "palette": speaker_palette(_run_label_universe(session, run.id)),
        "previews": _label_previews(session, run.id, states, settings.review_preview_segments),
        # Per-segment overrides (issue #54 Phase B): keyed by segment id, so a
        # preview segment can show its this-segment attribution + a reset control.
        "segment_overrides": segment_states(session, run.id),
        # Island props for the workbench-player (mounted OUTSIDE #labels). It owns
        # the <audio>, the speed control, the visible capability banner, and the
        # document-delegated enabling of the server-rendered seek buttons.
        "workbench_props": {
            "runId": str(run.id),
            "mediaUrl": f"/media/{run.id}",
            "capability": capability.to_props(),
        },
        # Per-label representative turn (start, end) for the "preview this speaker"
        # button — longest clean (non-overlap) DiarizationTurn, fallback longest.
        "representative_turns": representative_turns(session, run.id),
        "speakers": speakers,
        "token": token,
        "resolution": Resolution,
        "nonce": lambda: uuid.uuid4().hex,
        # CSRF token for the (re)claim form shown when the run is not claimed here.
        "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
    }


def _wants_html(request: Request) -> bool:
    """True for a plain browser form navigation (Accept prefers text/html), False
    for the island's fetch (which asks for application/json explicitly). Lets one
    write route serve the JSON island contract AND a JS-off HTML redirect fallback
    without a second route. The default httpx/TestClient Accept (``*/*``) is not
    HTML, so it keeps the JSON path — existing route tests are unaffected."""
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept


# Response header marking a 409 as a lost/again-taken claim (issue #59), so the
# island can distinguish it from a segment-STATE 409 the same route raises (a
# non-child range, an already-split parent). A claim loss must stop the review
# loop and prompt a re-claim; a state conflict shows an inline reason and keeps
# the claim. The value is opaque; only presence + "claim" matters to the client.
_CLAIM_CONFLICT_HEADERS = {"X-Voxint-Conflict": "claim"}

# Annotation-layer 409 markers (issue #86), mirroring _CLAIM_CONFLICT_HEADERS so
# the console can tell a stale-quote/anchor conflict, a replayed-nonce idempotency
# conflict, and a duplicate tag name apart from a lost claim. The taxonomy lives in
# docs/annotations.md ("API surface and error taxonomy").
_ANNOTATION_STALE_HEADERS = {"X-Voxint-Conflict": "stale"}
_ANNOTATION_IDEMPOTENCY_HEADERS = {"X-Voxint-Conflict": "idempotency"}
_ANNOTATION_TAG_CONFLICT_HEADERS = {"X-Voxint-Conflict": "duplicate-tag"}


def _annotation_http_error(exc: AnnotationError) -> HTTPException:
    """Map an annotation-domain error to its HTTP shape (docs/annotations.md error
    taxonomy). 422 validation, 404 not-found (fail closed — forged is not
    distinguished from missing), and three distinctly-marked 409s."""
    if isinstance(exc, AnnotationValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, AnnotationNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AnnotationStaleError):
        return HTTPException(status_code=409, detail=str(exc), headers=_ANNOTATION_STALE_HEADERS)
    if isinstance(exc, AnnotationIdempotencyError):
        return HTTPException(
            status_code=409, detail=str(exc), headers=_ANNOTATION_IDEMPOTENCY_HEADERS
        )
    if isinstance(exc, AnnotationTagConflictError):
        return HTTPException(
            status_code=409, detail=str(exc), headers=_ANNOTATION_TAG_CONFLICT_HEADERS
        )
    # Defensive: an unmapped AnnotationError is a validation-class failure, never a 500.
    return HTTPException(status_code=422, detail=str(exc))


def _annotation_limits() -> dict[str, int]:
    """The server-enforced annotation caps, echoed to the island so the client can
    pre-validate (the server stays the source of truth). Names mirror the constants
    in docs/annotations.md."""
    return {
        "paletteSize": HIGHLIGHT_PALETTE_SIZE,
        "maxSpanSegments": MAX_ANNOTATION_SPAN_SEGMENTS,
        "maxNoteChars": MAX_ANNOTATION_NOTE_CHARS,
        "maxTagsPerAnnotation": MAX_TAGS_PER_ANNOTATION,
        "maxQuoteChars": MAX_ANNOTATION_QUOTE_CHARS,
        "maxTagNameChars": MAX_TAG_NAME_CHARS,
    }


def _tag_shape(tag: AnnotationTag) -> dict[str, Any]:
    """One tag's island/JSON shape (camelCase, matching the frontend islands)."""
    return {
        "id": str(tag.id),
        "name": tag.name,
        "color": tag.color,
        "archived": tag.archived_at is not None,
    }


def _annotation_shapes(
    session: Session, run_id: uuid.UUID, rows: list[TranscriptAnnotation]
) -> list[dict[str, Any]]:
    """Resolve a set of stored annotations against the CURRENT render into the island
    JSON shape (camelCase): highlight spans, staleness, honest timing precision and
    seconds, live speakers, and the row metadata (colour/quote/note/tags). Shared by
    the list GET and the single-row create/patch responses so one row can never
    serialize two ways. Reads render the CORRECTED variant, exactly as the review
    surface does."""
    if not rows:
        return []
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    anchors = [stored_anchor_from_row(row) for row in rows]
    resolved = {r.annotation_id: r for r in resolve_annotation_spans(lines, covered, anchors)}
    tags_by_id = tags_for_annotations(session, [row.id for row in rows])
    # Canonical transcript order is applied AFTER resolution — by rendered line index,
    # not the captured start_segment_index annotations_for_run sorts by — so the panel
    # and the bulk pull-quote export agree even when a split child reorders lines.
    ordered_rows = sorted(rows, key=lambda row: resolved_order_key(resolved[row.id]))
    shapes: list[dict[str, Any]] = []
    for row in ordered_rows:
        res = resolved[row.id]
        shapes.append(
            {
                "id": str(row.id),
                "anchorKind": row.anchor_kind,
                "colorIndex": row.color_index,
                "quote": row.quote_text,
                "note": row.note,
                "operator": row.operator,
                "stale": res.stale,
                "timingPrecision": res.timing_precision,
                "startSeconds": res.start_seconds,
                "endSeconds": res.end_seconds,
                "speakers": list(res.speakers),
                "spans": [
                    {"lineIndex": s.line_index, "start": s.start, "end": s.end} for s in res.spans
                ],
                "locatorLineIndex": res.locator_line_index,
                "startSegmentIndex": row.start_segment_index,
                "endSegmentIndex": row.end_segment_index,
                "tags": [_tag_shape(t) for t in tags_by_id.get(row.id, [])],
            }
        )
    return shapes


def _require_filter_tags_exist(session: Session, tag_ids: list[uuid.UUID]) -> None:
    """Fail closed (404) when a ``?tag=`` filter names a tag that does not exist — a
    mistyped or forged id (docs/annotations.md: an unknown tag is a 404, never
    silently indistinguishable from a valid filter that matched nothing)."""
    if not tag_ids:
        return
    found = set(
        session.execute(select(AnnotationTag.id).where(AnnotationTag.id.in_(tag_ids))).scalars()
    )
    missing = [t for t in tag_ids if t not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown tag id {missing[0]}")


def _annotations_payload(
    session: Session, run_id: uuid.UUID, tag_ids: list[uuid.UUID]
) -> dict[str, Any]:
    """The GET /annotations body: the run's live annotations (optionally OR-filtered
    by tag) as island shapes, plus the full tag universe for the panel/picker."""
    rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
    return {
        "annotations": _annotation_shapes(session, run_id, rows),
        "tags": [_tag_shape(t) for t in list_tags(session)],
    }


def _pull_quote_markdown(
    resolved: ResolvedAnnotation,
    lines: Sequence[TranscriptLine],
    *,
    source_title: str,
    tags: Sequence[str],
    note: str | None,
) -> str:
    """Assemble one highlight's pull-quote Markdown, refusing a stale or otherwise
    unresolvable highlight with a 409 ``X-Voxint-Conflict: stale`` (docs/annotations.md):
    the captured copy alone cannot reconstruct the live speaker attribution and
    per-line geometry a faithful quote needs, so the export never fabricates it from
    ``quote_text``."""
    clipped = clip_lines_for_export(resolved, lines)
    if (
        resolved.stale
        or not clipped
        or resolved.start_seconds is None
        or resolved.end_seconds is None
    ):
        raise HTTPException(
            status_code=409,
            detail="highlight is stale; refresh or re-anchor it before exporting",
            headers=_ANNOTATION_STALE_HEADERS,
        )
    return annotation_pull_quote(
        clipped,
        source_title=source_title,
        start_seconds=resolved.start_seconds,
        end_seconds=resolved.end_seconds,
        timing_precision=resolved.timing_precision,
        tags=tags,
        note=note,
    )


def _capture_payload_from_form(
    start_segment_id: uuid.UUID,
    start_offset: int,
    start_child_word_start: int | None,
    start_child_word_end: int | None,
    end_segment_id: uuid.UUID,
    end_offset: int,
    end_child_word_start: int | None,
    end_child_word_end: int | None,
    client_quote: str,
) -> CapturePayload:
    """Assemble a :class:`CapturePayload` from the flat form sextuple (x2). The service
    normalizes direction and classifies; the route never picks the anchor kind."""
    return CapturePayload(
        start=CaptureEndpoint(
            segment_id=start_segment_id,
            offset=start_offset,
            child_word_start=start_child_word_start,
            child_word_end=start_child_word_end,
        ),
        end=CaptureEndpoint(
            segment_id=end_segment_id,
            offset=end_offset,
            child_word_start=end_child_word_start,
            child_word_end=end_child_word_end,
        ),
        client_quote=client_quote,
    )


def _verify_annotation_claim(session: Session, run_id: uuid.UUID, token: uuid.UUID) -> PipelineRun:
    """Gate a run-scoped annotation write. An unknown run is a 404 (docs/annotations.md
    fail-closed taxonomy, checked before the claim so a missing run never masquerades
    as a claim conflict); a lost claim is a 409 marked ``X-Voxint-Conflict: claim`` so
    the island stops the loop and re-claims. Holds the row lock (``for_update``) so a
    concurrent re-claim serializes against the write."""
    _run_or_404(session, run_id)
    try:
        return verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(
            status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
        ) from exc


def _labels_response(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID,
) -> Response:
    """Post-mutation response: htmx gets the refreshed label list, a plain
    form POST gets a redirect back to the workbench."""
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "legacy_review/labels.html",
            _workbench_context(request, session, run, token),
        )
    return RedirectResponse(f"/review/{run.id}?token={token}", status_code=303)


@transcript_router.get("/review/{run_id}/transcript", name="review_transcript")
def review_transcript(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: uuid.UUID | None = None,
) -> Response:
    """Claim-gated verify-and-advance surface (issue #53) + inline text
    correction (issue #58). Reuses the SAME linear player/props as the
    read-only transcript; the difference is the claim token (carried in
    ``?token=`` from the workbench, never re-acquired here — claims are
    takeover, so a fresh claim would kill the workbench tab) which unlocks the
    review-stepper island. A stale/absent token degrades to read-only with a
    prompt to claim, exactly like the workbench GET."""
    run = _run_or_404(session, run_id)
    if token is not None:
        try:
            verify_claim(session, run_id, token)
        except ClaimMismatchError:
            token = None  # stale tab: render read-only with a claim prompt
    # The review surface always shows operator-effective text (corrected →
    # enhanced → raw); raw evidence stays reachable on the read-only page.
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    settings: Settings = request.app.state.settings
    capability = playback_capability(session, run, settings, _get_media_gate(request))
    palette = speaker_palette(_run_label_universe(session, run_id))
    verified_n, total = verified_progress(session, run_id)
    island_props = _transcript_island_props(
        session, run_id, lines, palette, capability, settings
    )
    # Extras the review loop needs beyond the shared display props.
    island_props["reviewToken"] = str(token) if token is not None else None
    island_props["initialProgress"] = {"verified": verified_n, "total": total}
    # The assignable roster for the per-child reassign picker (issue #59 slice
    # 3): ACTIVE identities only — merged/archived speakers are curated out and
    # must not attract new rulings, mirroring the /relabel route's own
    # roster_is_active guard so the picker never offers a speaker the write
    # would reject. The read-only transcript-player never gets this (it cannot
    # relabel); only the claim-gated review-stepper needs a picker.
    island_props["speakers"] = [
        {"id": str(sp.id), "displayName": sp.display_name}
        for sp in active_speakers(session)
    ]
    # Declared-rule reconciliation (#83): the run-level "declared but never
    # fired" panel. Computed ONCE per page load by replaying the frozen pack
    # over every segment's IMMUTABLE raw_text (one row per segment — never the
    # split-expanded lines, which would double-count a rule's applied segments).
    # [] when the snapshot is unavailable or declares no corrections, so the
    # panel simply does not render. Skip the replay (and the raw-text query)
    # entirely for an unclaimed/read-only view — the panel lives inside the
    # writable block and never renders there — and when the pack declares no
    # rules (run_reconciliation would short-circuit to [] anyway).
    rule_index = _load_run_rule_index(session, run_id) if token is not None else None
    if rule_index is not None and rule_index.rules:
        raw_texts = (
            session.execute(
                select(TranscriptSegment.raw_text).where(
                    TranscriptSegment.pipeline_run_id == run_id
                )
            )
            .scalars()
            .all()
        )
        island_props["reconciliation"] = run_reconciliation(rule_index, raw_texts)
    else:
        island_props["reconciliation"] = []
    # Operator annotation layer (issue #86): the annotations to render + the tag
    # universe + a tag-CRUD CSRF token + the server caps. Hydrated for the
    # review-stepper only (annotation authoring is claim-gated); the read-only
    # transcript-player never gets these. The list mirrors GET /annotations so a
    # write's JSON reconciles against the same shape the page hydrated with.
    annotations_payload = _annotations_payload(session, run_id, [])
    island_props["annotations"] = annotations_payload["annotations"]
    island_props["annotationTags"] = annotations_payload["tags"]
    island_props["annotationLimits"] = _annotation_limits()
    island_props["tagCsrf"] = mint_csrf_token(
        request.app.state.csrf_secret, CSRF_ANNOTATION_TAGS
    )
    # Attributed audio-clip extraction (issue #88): the per-highlight "Extract
    # clip" action POSTs to /review/{run}/annotations/{id}/clips. Minted only for
    # the writable review-stepper (like tagCsrf) — the read-only transcript-player
    # never offers clip extraction.
    island_props["clipCsrf"] = mint_csrf_token(
        request.app.state.csrf_secret, CSRF_CLIP_EXTRACT
    )
    # Terminal Translate action (issue #133): once every line is checked, the
    # stepper offers translation beside "Open the transcript to export". Only
    # gate state + the preferred-language default are needed here — freshness
    # lives on the run page card and the transcript view, not in this launcher.
    # None when the LLM gates are closed (no dead button, matching the card).
    translate_row = get_app_settings(session)
    if translation_gates_open(settings, translate_row):
        detected = normalized_language(run.detected_language)
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
    return templates.TemplateResponse(
        request,
        "legacy_review/review_transcript.html",
        {
            "request": request,
            "run": run,
            "lines": lines,
            "island_props": island_props,
            "palette": palette,
            "low_confidence_threshold": settings.review_low_confidence_threshold,
            "token": token,
            "progress": {"verified": verified_n, "total": total},
            "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
            # Guided-tutorial banner (issue #117 Phase B): CHECK_WORDS and EXPORT
            # both bind this transcript page. The banner lives above the body
            # (base.html), outside the island, so a stepper re-render never
            # clobbers it. The claim token flows through so the export step's
            # onward link stays writable.
            "tutorial": _tutorial_banner(
                request,
                session,
                page=TutorialPage.TRANSCRIPT,
                run_id=run_id,
                token=token,
            ),
            "active_nav": "runs",
        },
    )


@router.get("/review", name="review_queue")
def review_queue(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    # Whitelist the sort so a bookmarked/garbage value degrades to the
    # default FIFO order rather than erroring (issue #56).
    sort = request.query_params.get("sort") or "oldest"
    if sort not in QUEUE_SORTS:
        sort = "oldest"
    return templates.TemplateResponse(
        request,
        "legacy_review/queue.html",
        {
            "request": request,
            "entries": adjudication_queue(session, sort=sort),
            "sort": sort,
            # Injected clock for the relative-age render (format_age(now=…)).
            "now": datetime.now(UTC),
            "operator": operator,
            # CSRF token for the per-row claim forms.
            "csrf_claim": mint_csrf_token(request.app.state.csrf_secret, CSRF_CLAIM),
            "tutorial": _tutorial_banner(request, session, page=TutorialPage.REVIEW_QUEUE),
            "active_nav": "review",
        },
    )

@router.post("/review/{run_id}/claim")
def claim(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    # Claim mints the run's claim token, so — unlike the other workbench
    # mutations — it has no unguessable token of its own to gate a forged POST.
    # A CSRF token closes that (a cross-site claim would otherwise pin a run to
    # the victim for the claim TTL). Verified before claim_run touches the DB.
    _require_csrf(request, CSRF_CLAIM, csrf_token)
    settings: Settings = request.app.state.settings
    # An archived run is hidden from the queue; refuse a stale/forged claim so
    # it can't be pinned to a reviewer while hidden (issue #5).
    _reject_if_archived(_run_or_404(session, run_id))
    try:
        token = claim_run(
            session,
            run_id,
            reviewer=operator,
            ttl_seconds=settings.review_claim_ttl_seconds,
        )
    except ClaimUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Guided-tutorial continuity: claiming the tutorial run — while the
    # walkthrough is still active (not yet completed) — lands on the workbench
    # in `adjudicate` mode. Keyed on RUN IDENTITY, not a hidden form field, so
    # EVERY claim control continues the tutorial identically: the banner's own
    # button, the queue row's ordinary "Review" button, and the workbench claim
    # button. A first-time user therefore cannot silently fall out of the
    # walkthrough by clicking the "wrong" (but identical-looking) button. A
    # non-tutorial run is never rewritten, and once completed the tutorial run
    # claims normally (no banner) — replay clears completion and re-activates it.
    suffix = ""
    if ready_tutorial_run_id(session) == run_id:
        row = get_app_settings(session)
        if row is not None and row.tutorial_completed_at is None:
            suffix = "&tutorial=adjudicate"
    return RedirectResponse(f"/review/{run_id}?token={token}{suffix}", status_code=303)

@router.get("/review/{run_id}", name="workbench")
def workbench(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: uuid.UUID | None = None,
) -> Response:
    run = _run_or_404(session, run_id)
    if token is not None:
        try:
            verify_claim(session, run_id, token)
        except ClaimMismatchError:
            token = None  # stale tab: render read-only with a claim button
    context = _workbench_context(request, session, run, token)
    # Full-page render only: the banner lives above the body (base.html),
    # outside the #labels htmx target, so decision/enroll fragment swaps — which
    # go through _labels_response and never carry `tutorial` — leave it in place.
    context["tutorial"] = _tutorial_banner(
        request, session, page=TutorialPage.WORKBENCH, run_id=run_id, token=token
    )
    return templates.TemplateResponse(request, "legacy_review/run.html", context)

@router.post("/review/{run_id}/release")
def release(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
) -> RedirectResponse:
    try:
        release_run(session, run_id, token)
    except (ClaimMismatchError, ClaimUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RedirectResponse("/review", status_code=303)

@router.post("/review/{run_id}/labels/{label}/decision")
def decide(
    run_id: uuid.UUID,
    label: str,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    action: Annotated[str, Form()],
    speaker_id: Annotated[uuid.UUID | None, Form()] = None,
) -> Response:
    try:
        run = verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        decision = Decision(action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown action {action!r}") from exc
    # `inherit` is a segment-scope reset only — never a whole-label ruling
    # (the DB CHECK would otherwise reject it as a raw 500).
    if decision not in (Decision.ASSIGN, Decision.EXCLUDE, Decision.UNKNOWN):
        raise HTTPException(status_code=422, detail=f"invalid label action {action!r}")
    if (decision is Decision.ASSIGN) != (speaker_id is not None):
        raise HTTPException(
            status_code=422, detail="choose a speaker to assign, or pick a different action"
        )
    speaker: Speaker | None = None
    if speaker_id is not None:
        # FOR SHARE: a concurrent archive/merge takes FOR UPDATE on this
        # row, so the active check and the ledger append below serialize
        # with roster curation instead of racing it.
        speaker = session.execute(
            select(Speaker).where(Speaker.id == speaker_id).with_for_update(read=True)
        ).scalar_one_or_none()
        if speaker is None:
            raise HTTPException(status_code=422, detail=f"no speaker {speaker_id}")
        if not roster_is_active(speaker):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"speaker {speaker.display_name!r} is no longer an active"
                    " roster identity — refresh and pick another"
                ),
            )
    states = label_states(session, run_id)
    if label not in {s.label for s in states}:
        raise HTTPException(status_code=404, detail=f"no label {label!r} in run")
    # The label's effective speaker BEFORE this ruling: an assign that re-asserts
    # it identifies nothing new and must not toast (issue #162 activity).
    prior_speaker_id = next((s.speaker_id for s in states if s.label == label), None)
    # A replay returns the existing row without changing effective attribution, so
    # it must not toast (a stale/superseded identification); only a fresh ruling.
    is_replay = decision_exists(session, nonce)
    try:
        row = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label=label,
            decision=decision,
            operator=operator,
            idempotency_key=nonce,
            speaker_id=speaker_id,
        )
    except ConflictingReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    settings: Settings = request.app.state.settings
    if (
        settings.console_activity_enabled
        and not is_replay
        and decision is Decision.ASSIGN
        and speaker is not None  # ASSIGN <=> speaker_id set (checked above)
        and speaker_id != prior_speaker_id  # effective attribution actually changed
    ):
        record_speaker_identified(
            session, run_id=run_id, decision_id=row.id, speaker_name=speaker.display_name
        )
    return _labels_response(request, session, run, token)

@router.post("/review/{run_id}/merge/preview")
def merge_preview(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    labels: Annotated[list[str], Form()],
    target: Annotated[str, Form()],
    new_name: Annotated[str | None, Form()] = None,
) -> Response:
    """Server-computed impact of merging labels — reads only, writes nothing.

    The confirm step the operator sees before applying: the exact turns and
    segments the merge touches (never an advisory client count), plus the
    optimistic-concurrency token (each label's current effective ruling id)
    echoed into the confirm form so :func:`merge_apply` can reject a stale
    confirm. Claim-gated like every workbench mutation; JS-off never reaches
    here (the panel enhances progressively — see legacy_review/labels.html).

    ``target`` is the single unambiguous survivor chooser from the panel: the
    sentinel ``"new"`` (enroll ``new_name``) or an existing speaker's UUID.
    The confirm form it renders echoes the resolved speaker_id XOR
    display_name, so :func:`merge_apply` never has to disambiguate.
    """
    try:
        run = verify_claim(session, run_id, token)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    target_speaker: Speaker | None = None
    display_name: str | None = None
    if target == "new":
        display_name = (new_name or "").strip() or None
        if display_name is None:
            raise HTTPException(status_code=400, detail="enter a name for the new speaker")
    else:
        try:
            target_speaker = session.get(Speaker, uuid.UUID(target))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="choose a survivor speaker") from exc
        if target_speaker is None:
            raise HTTPException(status_code=400, detail="that speaker no longer exists")
    try:
        preview = preview_merge(session, run_id, labels)
    except MergeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    expected = json.dumps(
        {
            impact.label: (
                str(impact.expected_decision_id)
                if impact.expected_decision_id is not None
                else None
            )
            for impact in preview.labels
        }
    )
    return templates.TemplateResponse(
        request,
        "legacy_review/merge_confirm.html",
        {
            "request": request,
            "run": run,
            "token": token,
            "nonce": lambda: uuid.uuid4().hex,
            "preview": preview,
            "target_speaker": target_speaker,
            "target_name": display_name,
            "expected_json": expected,
            "palette": speaker_palette(_run_label_universe(session, run_id)),
        },
    )

@router.post("/review/{run_id}/merge")
def merge_apply(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    labels: Annotated[list[str], Form()],
    expected: Annotated[str, Form()],
    speaker_id: Annotated[uuid.UUID | None, Form()] = None,
    display_name: Annotated[str | None, Form()] = None,
) -> Response:
    """Rule that several labels are one speaker in this run — atomically.

    Run-local: records one assign ruling per label to a single survivor; it
    never calls the roster-wide merge_speakers. Under the claim lock it
    re-verifies the previewed rulings still hold (409 if they drifted) and
    appends every ruling in one transaction with deterministic child
    idempotency keys, so a replay returns the original outcome and a partial
    apply is impossible.
    """
    try:
        run = verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # An untouched text field posts "" not None; the confirm form omits the
    # unused target entirely, but normalise defensively so the XOR check sees
    # a clean None rather than an empty string.
    display_name = (display_name or "").strip() or None
    try:
        raw = json.loads(expected)
        if not isinstance(raw, dict):
            raise TypeError("expected-state must be a JSON object")
        expected_ids: dict[str, uuid.UUID | None] = {
            str(label): (uuid.UUID(value) if value is not None else None)
            for label, value in raw.items()
        }
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="malformed expected-state") from exc
    settings: Settings = request.app.state.settings
    try:
        result = apply_merge(
            session,
            run_id=run_id,
            labels=labels,
            operator=operator,
            nonce=nonce,
            gates=gates_from_settings(settings),
            target_speaker_id=speaker_id,
            target_name=display_name,
            expected=expected_ids,
        )
    except MergeConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConflictingReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MergeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # One event per merge (server-side coalescing): a merge is one operator
    # consolidation, keyed on a stable per-merge decision id (issue #162 activity).
    # A replayed merge (double-click / retry) never re-announces.
    if settings.console_activity_enabled and not result.is_replay and result.decision_ids:
        record_speaker_merge(
            session,
            run_id=run_id,
            occurrence_decision_id=min(result.decision_ids.values()),
            survivor_name=result.survivor_name,
            label_count=len(result.labels),
        )
    return _labels_response(request, session, run, token)

@router.post("/review/{run_id}/segments/{segment_id}/relabel")
def relabel_segment(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    action: Annotated[str, Form()],
    speaker_id: Annotated[uuid.UUID | None, Form()] = None,
    start_word_index: Annotated[int | None, Form()] = None,
    end_word_index: Annotated[int | None, Form()] = None,
) -> Response:
    """Two-scope relabel, THIS-SEGMENT scope (issue #54 Phase B), optionally
    narrowed to a word-range (issue #59 slice 3).

    Overrides one transcript segment's attribution without touching the rest
    of its label. ``action`` is ``assign`` (a speaker just for this segment)
    or ``inherit`` (append-only reset: the segment follows its label's
    resolution again). The diarization label is derived from the segment row
    server-side, never trusted from the client. Claim-gated and idempotent
    like every workbench ruling; a later whole-label ruling leaves the
    override intact, and inherit tracks the label live rather than freezing.

    With ``start_word_index``/``end_word_index`` the ruling scopes just that
    half-open ``[start, end)`` word-range — reassigning ONE derived split
    child. The range must match a child that currently exists (validated
    against the segment's live cut set), so a ruling can only target a real
    partition, never an arbitrary span the read path would ignore. Both
    indices are set together or both omitted (whole-segment scope).
    """
    try:
        run = verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        # Marked so the island can tell a lost claim from the segment-STATE 409
        # this route also raises (a non-child range): the picker treats a plain
        # 409 as a state conflict, but a claim loss must stop the loop.
        raise HTTPException(
            status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
        ) from exc
    try:
        decision = Decision(action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown action {action!r}") from exc
    if decision not in (Decision.ASSIGN, Decision.INHERIT):
        raise HTTPException(
            status_code=422, detail="segment scope allows only assign or inherit"
        )
    if (decision is Decision.ASSIGN) != (speaker_id is not None):
        raise HTTPException(
            status_code=422,
            detail="choose a speaker to assign this segment to, or use inherit to follow the label",
        )
    if (start_word_index is None) != (end_word_index is None):
        raise HTTPException(
            status_code=422,
            detail=(
                "provide both word-range boundaries together,"
                " or omit both for whole-segment scope"
            ),
        )
    segment = session.get(TranscriptSegment, segment_id)
    if segment is None or segment.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such segment in this run")
    if segment.diarization_label is None:
        raise HTTPException(
            status_code=400, detail="segment has no diarization label to override"
        )
    if start_word_index is not None and end_word_index is not None:
        # A ranged ruling may only target a child that exists right now, so it
        # can never write a row the read path silently drops.
        child_ranges = _segment_child_ranges(session, segment)
        if (start_word_index, end_word_index) not in child_ranges:
            raise HTTPException(
                status_code=409,
                detail=(
                    "that word range does not match a current split — split "
                    "the segment at that boundary first, then try again"
                ),
            )
    speaker: Speaker | None = None
    if speaker_id is not None:
        speaker = session.execute(
            select(Speaker).where(Speaker.id == speaker_id).with_for_update(read=True)
        ).scalar_one_or_none()
        if speaker is None:
            raise HTTPException(status_code=422, detail=f"no speaker {speaker_id}")
        if not roster_is_active(speaker):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"speaker {speaker.display_name!r} is no longer an active"
                    " roster identity — refresh and pick another"
                ),
            )
    # Only a fresh ruling announces; a replay returns the existing row unchanged.
    is_replay = decision_exists(session, nonce)
    try:
        row = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label=segment.diarization_label,
            decision=decision,
            operator=operator,
            idempotency_key=nonce,
            speaker_id=speaker_id,
            transcript_segment_id=segment_id,
            start_word_index=start_word_index,
            end_word_index=end_word_index,
        )
    except ConflictingReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WordRangeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    settings: Settings = request.app.state.settings
    # A per-segment override to a named speaker is a positive identification
    # (INHERIT is a reset, never announced — issue #162 activity).
    if (
        settings.console_activity_enabled
        and not is_replay
        and decision is Decision.ASSIGN
        and speaker is not None
    ):
        record_speaker_identified(
            session, run_id=run_id, decision_id=row.id, speaker_name=speaker.display_name
        )
    # The island (JSON Accept) reassigns a split child and reconciles against a
    # whole-run render — a child's speaker string moved, which the per-segment
    # review shape can't carry. The htmx labels workbench (and any non-island
    # caller) keeps the byte-identical server-rendered fragment.
    if _wants_island_json(request):
        return _run_reconcile_response(session, run_id)
    return _labels_response(request, session, run, token)

def _segment_review_json(
    session: Session, run_id: uuid.UUID, segment: TranscriptSegment
) -> JSONResponse:
    """The state a triage-loop write returns to the island: this segment's
    verified/corrected flags + effective text, and the run's N-of-M counter."""
    row = session.get(SegmentReviewState, segment.id)
    corrected = row.corrected_text if row is not None else None
    verified_n, total = verified_progress(session, run_id)
    return JSONResponse(
        {
            "segmentId": str(segment.id),
            "verified": row is not None and row.verified_at is not None,
            "corrected": corrected is not None,
            "text": effective_text(segment, corrected),
            "progress": {"verified": verified_n, "total": total},
        }
    )

@router.post("/review/{run_id}/segments/{segment_id}/verify")
def verify_segment(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    verified: Annotated[bool, Form()] = True,
) -> Response:
    """Mark (or unmark) a segment verified — the verify-and-advance step
    (issue #53). Claim-gated; a mutable UPSERT, so idempotent without a nonce.
    The island (fetch) gets the updated state + N-of-M progress as JSON; a
    JS-off browser form navigation is redirected back to the review page, so
    the server-rendered fallback verifies for real (degrade-to-plain-HTML)."""
    try:
        verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    segment = session.get(TranscriptSegment, segment_id)
    if segment is None or segment.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such segment in this run")
    set_verified(session, segment=segment, verified=verified)
    if _wants_html(request):
        return RedirectResponse(
            f"/review/{run_id}/transcript?token={token}", status_code=303
        )
    return _segment_review_json(session, run_id, segment)

@router.post("/review/{run_id}/segments/{segment_id}/text")
def correct_segment(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    text: Annotated[str, Form(max_length=MAX_CORRECTED_TEXT_CHARS)] = "",
) -> JSONResponse:
    """Set or clear the operator's corrected text for a segment (issue #58).
    Empty text, or text equal to the pipeline rendering, reverts to no
    correction. Editing clears the segment's verified mark in the same
    transaction. Claim-gated; the corrected text is written beside raw_text,
    never over it (raw stays the immutable ASR evidence)."""
    try:
        verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    segment = session.get(TranscriptSegment, segment_id)
    if segment is None or segment.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such segment in this run")
    # A split parent renders as word-derived children, which free-form
    # corrected text cannot be partitioned across (issue #59, deferred). Refuse
    # correcting a split segment — the mirror of forbidding a split on an
    # already-corrected segment, so the two never coexist.
    if _segment_is_split(session, segment_id):
        raise HTTPException(
            status_code=409,
            detail="cannot correct a split segment; remove the split first",
        )
    set_correction(session, segment=segment, text=text)
    return _segment_review_json(session, run_id, segment)

@router.post("/review/{run_id}/segments/{segment_id}/split")
def split_segment(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    word_index: Annotated[int, Form()],
) -> JSONResponse:
    """Split a segment at a word boundary (issue #59), inserting one cut
    "before word ``word_index``". Claim-gated and structurally idempotent (the
    boundary's UNIQUE key makes a replayed split a no-op — no nonce needed).

    Refuses a corrected segment (mutually exclusive with correction) and an
    unsplittable one (no aligned word timings, or materially-enhanced text) with
    the operator-facing reason. Returns the run's re-rendered island segments so
    the console reconciles against server truth — the same shape as hydration,
    with the parent now expanded into its derived children."""
    try:
        verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        # Marked so the island distinguishes a lost claim from the segment-STATE
        # 409 this route also raises (already-split / corrected): the split
        # handler treats a plain 409 as a state conflict, a claim loss must stop.
        raise HTTPException(
            status_code=409, detail=str(exc), headers=_CLAIM_CONFLICT_HEADERS
        ) from exc
    segment = session.get(TranscriptSegment, segment_id)
    if segment is None or segment.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such segment in this run")
    row = session.get(SegmentReviewState, segment_id)
    if row is not None and row.corrected_text is not None:
        raise HTTPException(
            status_code=409,
            detail="cannot split a corrected segment; clear the correction first",
        )
    # This release supports a SINGLE cut per parent (two children). A second,
    # DISTINCT cut would re-derive the children and orphan any word-range
    # reassignment keyed on the old child coordinates — a written ruling the
    # read path then silently ignores (issue #59 slice 3). The UI already
    # disables further splits, but a second tab sharing the claim could still
    # POST one; refuse it server-side. A replay of the EXISTING cut still falls
    # through to record_split's idempotent no-op (same word_index), so /split
    # stays idempotent.
    existing_cuts = {
        wi
        for (wi,) in session.execute(
            select(SegmentSplitBoundary.word_index).where(
                SegmentSplitBoundary.parent_segment_id == segment_id
            )
        )
    }
    if existing_cuts and word_index not in existing_cuts:
        raise HTTPException(
            status_code=409,
            detail=(
                "this segment already has a split; only one split per segment "
                "is supported — reassign the existing halves, or remove "
                "their word-range rulings and try a different boundary"
            ),
        )
    try:
        record_split(
            session, parent=segment, word_index=word_index, operator=operator
        )
    except UnsplittableError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _run_reconcile_response(session, run_id)

@router.get("/review/{run_id}/segments/{segment_id}/words")
def segment_words(
    run_id: uuid.UUID,
    segment_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
) -> JSONResponse:
    """The active segment's word tokens for the split UI (issue #59), fetched
    LAZILY only when the operator enters split mode — never bloating the shared
    read payload with every run's words. Reports ``splittable`` (+ a reason when
    not) so the console shows an honest disabled affordance rather than a
    split that would fail."""
    segment = session.get(TranscriptSegment, segment_id)
    if segment is None or segment.pipeline_run_id != run_id:
        raise HTTPException(status_code=404, detail="no such segment in this run")
    words = splittable_words(segment)
    if words is None:
        if _segment_is_corrected(session, segment_id):
            reason = "this segment has an operator correction; clear it to split"
        elif trace_has_entries(segment.correction_trace):
            reason = "a domain-pack correction was applied here; splitting is disabled"
        else:
            reason = "no aligned word timings for this segment (or its text was enhanced)"
        return JSONResponse(
            {"segmentId": str(segment_id), "splittable": False, "reason": reason, "words": []}
        )
    if _segment_is_corrected(session, segment_id):
        return JSONResponse(
            {
                "segmentId": str(segment_id),
                "splittable": False,
                "reason": "this segment has an operator correction; clear it to split",
                "words": [],
            }
        )
    return JSONResponse(
        {
            "segmentId": str(segment_id),
            "splittable": True,
            "reason": None,
            "words": [
                {"start": w.start, "end": w.end, "word": w.text} for w in words
            ],
        }
    )

@router.post("/review/{run_id}/labels/{label}/enroll")
def enroll(
    run_id: uuid.UUID,
    label: str,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    display_name: Annotated[str, Form()],
) -> Response:
    try:
        run = verify_claim(session, run_id, token, for_update=True)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    settings: Settings = request.app.state.settings
    # Only a fresh enrollment announces, never a replay of one.
    is_replay = decision_exists(session, nonce)
    try:
        enrollment = enroll_new_speaker(
            session,
            run_id=run_id,
            diarization_label=label,
            display_name=display_name,
            operator=operator,
            idempotency_key=nonce,
            gates=gates_from_settings(settings),
        )
    except EnrollmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConflictingReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if settings.console_activity_enabled and not is_replay:
        # Resolve the authoritative roster name from the speaker row, never the
        # submitted display_name: a replay reuses an existing (maybe renamed)
        # speaker and enroll deliberately ignores the posted name then.
        enrolled = session.get(Speaker, enrollment.speaker_id)
        if enrolled is not None:
            record_speaker_identified(
                session,
                run_id=run_id,
                decision_id=enrollment.decision_id,
                speaker_name=enrolled.display_name,
            )
    return _labels_response(request, session, run, token)

@router.post("/review/{run_id}/enrich/names")
def enrich_names(
    run_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
) -> Response:
    """Operator-triggered offline name sweep (issue #38), synchronous.

    Pure regex + DB over one run's stored rows — sub-second — so it runs
    inline and the htmx response swaps the refreshed suggestion list in.
    Claim-token-gated like every other workbench mutation.
    """
    settings: Settings = request.app.state.settings
    if not resolve_effective_enrichment_names_enabled(get_app_settings(session), settings):
        raise HTTPException(status_code=404, detail="name enrichment is disabled")
    try:
        run = verify_claim(session, run_id, token)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        run_offline_name_producer(session, run_id=run_id, settings=settings)
    except NameProducerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _labels_response(request, session, run, token)

@router.post("/review/{run_id}/candidates/{candidate_id}/decision")
def decide_name_candidate(
    run_id: uuid.UUID,
    candidate_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    verdict: Annotated[str, Form()],
) -> Response:
    """Accept/reject one name suggestion — a review record, never identity.

    Writes the profile-review trail only: no speaker, assignment, or
    adjudication ruling is created (drafts are suggestions about identity).
    """
    settings: Settings = request.app.state.settings
    if not resolve_effective_enrichment_names_enabled(get_app_settings(session), settings):
        raise HTTPException(status_code=404, detail="name enrichment is disabled")
    try:
        run = verify_claim(session, run_id, token)
    except ClaimMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        decision = ProfileDecision(verdict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown verdict {verdict!r}") from exc
    candidate = session.get(EnrichmentCandidate, candidate_id)
    # This route serves the NAME surface only — a candidate from another
    # run, or another claim field never rendered here, is not decidable.
    if (
        candidate is None
        or candidate.pipeline_run_id != run_id
        or candidate.field != ClaimField.NAME.value
    ):
        raise HTTPException(status_code=404, detail="no such candidate in this run")
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
            detail="superseded by a newer sweep — refresh and re-review",
        ) from exc
    except EnrichmentReplayError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _labels_response(request, session, run, token)

# Transcript downloads: one attributed read shaped by a pure formatter (see
# voxint/export). The sibling extensions (.txt/.srt/.vtt/.json) share the CLI's
# exact byte output through render_transcript, so a download and a piped
# `voxint export … --format …` can never disagree. RTTM lives on its own route
# (it reads diarization turns, not attributed lines). All accept
# ?text=corrected|enhanced|raw (default corrected: operator corrections applied
# over enhanced/raw; enhanced = pipeline text, no corrections; raw = immutable
# ASR evidence), except RTTM which is speaker-label-only.
def _export_translated_lines(
    session: Session,
    run_id: uuid.UUID,
    lines: list[TranscriptLine],
    lang: str,
    variant: TranscriptText,
) -> list[TranscriptLine]:
    """The reviewed lines with translated text substituted, or an honest
    HTTP failure — NEVER partial or mixed-language output (issue #133).

    Fail-closed policy: 422 for an unknown code or a raw/enhanced variant
    (a translation is a rendition of the reviewed transcript only), 409
    when no current generation exists or the transcript has changed since
    it was generated. Substitution is by line order within the generation;
    the hash equality is what proves the order still describes this
    transcript. Subtitle cue timing is untouched (no reflow) — translated
    captions may read fast, and the docs say so.
    """
    target = normalized_language(lang)
    if target is None or target not in LANGUAGE_NAMES:
        raise HTTPException(
            status_code=422, detail=f"unknown translation language code {lang!r}"
        )
    if variant is not TranscriptText.CORRECTED:
        raise HTTPException(
            status_code=422,
            detail=(
                "a translation renders the reviewed transcript only — drop"
                " text= (or use text=corrected) with lang="
            ),
        )
    label = language_label(target)
    head = current_translation(session, run_id, target)
    if head is None:
        # Job lookup only on the failure path (review finding): a
        # successful export must not scan the run's job history just to
        # phrase a 409 it will never raise.
        job = active_or_last_translation_job(session, run_id)
        running = (
            job is not None
            and job.status in _TRANSLATION_ACTIVE_STATUSES
            and job.target_language == target
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"a {label} translation is still being generated — retry when"
                " it finishes"
                if running
                else f"no {label} translation exists for this run — generate"
                " one from the run page first"
            ),
        )
    try:
        current_hash = translation_source_hash(load_translation_source(session, run_id))
    except TranslationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    texts = translation_texts(head)
    if head.source_content_hash != current_hash or len(texts) != len(lines):
        raise HTTPException(
            status_code=409,
            detail=(
                f"the {label} translation is out of date — the transcript"
                " changed since it was generated; re-translate from the run"
                " page and retry"
            ),
        )
    return [
        dataclass_replace(ln, text=translated)
        for ln, translated in zip(lines, texts, strict=True)
    ]

def _export_transcript(
    run_id: uuid.UUID,
    session: Session,
    fmt: TranscriptFormat,
    text: str | None,
    *,
    timestamps: bool = True,
    lang: str | None = None,
) -> Response:
    _run_or_404(session, run_id)
    try:
        variant = parse_transcript_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    lines = attributed_transcript(session, run_id, text=variant)
    if lang is not None:
        lines = _export_translated_lines(session, run_id, lines, lang, variant)
    return Response(
        content=render_transcript(lines, fmt, timestamps=timestamps),
        media_type=MEDIA_TYPES[fmt.value],
    )

@router.get("/review/{run_id}/export.txt")
def export_transcript_txt(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    timestamps: bool = True,
    lang: str | None = None,
) -> Response:
    # ?timestamps=false drops the [start end] bracket column for a clean
    # reading copy (issue #52). Only txt and md honor the flag.
    # ?lang=<code> substitutes the current fresh translation (issue #133) —
    # fail closed, see _export_translated_lines. All five formats take it.
    return _export_transcript(
        run_id, session, TranscriptFormat.TXT, text, timestamps=timestamps, lang=lang
    )

@router.get("/review/{run_id}/export.md")
def export_transcript_md(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    timestamps: bool = True,
    lang: str | None = None,
) -> Response:
    # Readable Markdown (issue #65): ## speaker headings + merged blockquotes.
    # ?timestamps=false drops the per-paragraph time range for a clean copy.
    return _export_transcript(
        run_id,
        session,
        TranscriptFormat.MARKDOWN,
        text,
        timestamps=timestamps,
        lang=lang,
    )

@router.get("/review/{run_id}/export.srt")
def export_transcript_srt(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    lang: str | None = None,
) -> Response:
    return _export_transcript(run_id, session, TranscriptFormat.SRT, text, lang=lang)

@router.get("/review/{run_id}/export.vtt")
def export_transcript_vtt(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    lang: str | None = None,
) -> Response:
    return _export_transcript(run_id, session, TranscriptFormat.VTT, text, lang=lang)

@router.get("/review/{run_id}/export.json")
def export_transcript_json(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    text: str | None = None,
    lang: str | None = None,
) -> Response:
    return _export_transcript(run_id, session, TranscriptFormat.JSON, text, lang=lang)

@router.get("/review/{run_id}/export.rttm")
def export_transcript_rttm(
    run_id: uuid.UUID, operator: OperatorDep, session: SessionDep
) -> Response:
    _run_or_404(session, run_id)
    turns = (
        session.execute(
            select(DiarizationTurn)
            .where(DiarizationTurn.pipeline_run_id == run_id)
            .order_by(DiarizationTurn.turn_index)
        )
        .scalars()
        .all()
    )
    return Response(content=to_rttm(turns, str(run_id)), media_type=MEDIA_TYPES["rttm"])

# ---- Operator annotation layer (issue #86) --------------------------------
# Thin handlers over voxint.adjudication.annotations: the service owns all
# coordinate math, classification, idempotency, and staleness; routes only
# parse the wire shape, gate auth/claim/CSRF, and map AnnotationError to HTTP
# (docs/annotations.md). Reads need onboarding only; run-scoped writes need the
# live review claim; global tag writes are CSRF-gated like run notes.

@router.get("/review/{run_id}/annotations")
def list_annotations(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    tag: Annotated[list[uuid.UUID] | None, Query()] = None,
) -> JSONResponse:
    """The run's live annotations (transcript order) resolved against the current
    render, plus the tag universe. Onboarding-auth only — no claim, so a reviewer
    can read annotations without holding the slot. Repeated ``?tag=`` is an
    OR-union filter, identically in the panel and exports; an unknown tag id in
    the filter fails closed (404)."""
    _run_or_404(session, run_id)
    tag_ids = tag or []
    _require_filter_tags_exist(session, tag_ids)
    return JSONResponse(_annotations_payload(session, run_id, tag_ids))

@router.get("/review/{run_id}/annotations/export.md")
def export_annotations_md(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    tag: Annotated[list[uuid.UUID] | None, Query()] = None,
) -> Response:
    """All (filtered) highlights as Markdown pull-quotes in canonical transcript
    order (issue #86), joined by a thematic-break separator. Onboarding-auth only,
    NO claim — Copy is a read, available in a read-only review tab. Repeated
    ``?tag=`` is the same OR-union filter as the panel; an unknown tag id is 404.
    Fails ATOMICALLY with 409 ``X-Voxint-Conflict: stale`` if ANY matched highlight
    is stale (it is never silently omitted). An empty match is an empty body (200)."""
    run = _run_or_404(session, run_id)
    tag_ids = tag or []
    _require_filter_tags_exist(session, tag_ids)
    rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
    if not rows:
        return Response(content="", media_type=ANNOTATION_MEDIA_TYPES["md"])
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    resolved = {
        r.annotation_id: r
        for r in resolve_annotation_spans(
            lines, covered, [stored_anchor_from_row(row) for row in rows]
        )
    }
    tags_by_id = tags_for_annotations(session, [row.id for row in rows])
    source_title = _run_source_title(run)
    ordered = sorted(rows, key=lambda row: resolved_order_key(resolved[row.id]))
    quotes = [
        _pull_quote_markdown(
            resolved[row.id],
            lines,
            source_title=source_title,
            tags=[t.name for t in tags_by_id.get(row.id, [])],
            note=row.note,
        )
        for row in ordered
    ]
    return Response(
        content=ANNOTATION_BULK_SEPARATOR.join(quotes),
        media_type=ANNOTATION_MEDIA_TYPES["md"],
    )

@router.get("/review/{run_id}/annotations/{annotation_id}/export.md")
def export_annotation_md(
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
) -> Response:
    """One highlight as a Markdown pull-quote (issue #86). Onboarding-auth only,
    NO claim. A foreign, forged, or soft-deleted id is 404 (fail closed). A stale
    highlight is refused 409 ``X-Voxint-Conflict: stale`` — the operator refreshes
    or re-anchors it first."""
    run = _run_or_404(session, run_id)
    try:
        row = live_annotation_or_404(session, run_id, annotation_id)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    resolved = resolve_annotation_spans(lines, covered, [stored_anchor_from_row(row)])[0]
    tags = [t.name for t in tags_for_annotations(session, [row.id]).get(row.id, [])]
    markdown = _pull_quote_markdown(
        resolved,
        lines,
        source_title=_run_source_title(run),
        tags=tags,
        note=row.note,
    )
    return Response(content=markdown, media_type=ANNOTATION_MEDIA_TYPES["md"])


# ---------------------------------------------------------------------------
# JSON provenance manifest (issue #122)
# ---------------------------------------------------------------------------

def _stage_provenance_from_run(
    session: Session, run_id: uuid.UUID
) -> dict[str, StageProvenance]:
    """Read per-stage model identity from the latest completed attempt."""
    stage_runs = (
        session.execute(
            select(StageRun).where(StageRun.pipeline_run_id == run_id)
        )
        .scalars()
        .all()
    )
    stages: dict[str, StageProvenance] = {}
    for sm in select_run_model_identity(stage_runs):
        if not sm.recorded:
            stages[sm.stage] = StageProvenance(
                attempt=sm.attempt or 0,
                finished_at=None,
                roles={},
            )
            continue
        best = None
        for sr in stage_runs:
            if sr.stage == sm.stage and sr.attempt == sm.attempt:
                best = sr
                break
        roles: dict[str, StageRole] = {}
        for mr in sm.roles:
            roles[mr.role] = StageRole(
                reachable=mr.reachable,
                model=mr.model,
                revision=mr.revision,
                engine=mr.engine,
            )
        stages[sm.stage] = StageProvenance(
            attempt=sm.attempt or 0,
            finished_at=best.finished_at if best else None,
            roles=roles,
        )
    return stages


def _clip_ref_for_annotation(
    session: Session,
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    settings: Settings,
) -> ClipRef | None:
    """Look up the latest non-reclaimed clip for an annotation."""
    row = (
        session.execute(
            select(AudioArtifact)
            .where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
                AudioArtifact.reclaimed_at.is_(None),
                AudioArtifact.meta["annotation_id"].as_string()
                == str(annotation_id),
            )
            .order_by(AudioArtifact.created_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    if row is None:
        return None
    meta = row.meta or {}
    clip_path = settings.media_root / row.path
    clip_sha256: str | None = None
    if clip_path.is_file():
        import hashlib

        clip_sha256 = hashlib.sha256(clip_path.read_bytes()).hexdigest()
    return ClipRef(
        id=row.id,
        download_url=f"/runs/{run_id}/clips/{row.id}",
        filename=clip_download_filename(run_id, row.id),
        sha256=clip_sha256 or "",
        sample_rate=meta.get("sample_rate", 16000),
        channels=1,
        start_sample=meta.get("start_sample", 0),
        end_sample=meta.get("end_sample", 0),
    )


def _manifest_for_resolved(
    resolved: ResolvedAnnotation,
    row: TranscriptAnnotation,
    lines: Sequence[TranscriptLine],
    *,
    source_title: str,
    tags: Sequence[str],
    clip: ClipRef | None,
    media_id: uuid.UUID,
    run_id: uuid.UUID,
    media_sha256: str | None,
    app_version: str,
    stages: dict[str, StageProvenance],
    exported_at: datetime,
) -> dict[str, Any]:
    """Build a manifest dict for one resolved annotation."""
    clipped = clip_lines_for_export(resolved, lines)
    if (
        resolved.stale
        or not clipped
        or resolved.start_seconds is None
        or resolved.end_seconds is None
    ):
        raise HTTPException(
            status_code=409,
            detail="highlight is stale; refresh or re-anchor it before exporting",
            headers=_ANNOTATION_STALE_HEADERS,
        )
    quote_lines = [
        QuoteLine(
            text=ln.text,
            speaker=ln.speaker,
            start_seconds=ln.start_seconds,
            end_seconds=ln.end_seconds,
        )
        for ln in clipped
    ]
    return build_quote_manifest(
        exported_at=exported_at,
        annotation_id=row.id,
        source_text_hash=row.source_text_hash,
        annotation_updated_at=row.updated_at,
        lines=quote_lines,
        timing_precision=resolved.timing_precision,
        tags=list(tags),
        note=row.note,
        clip=clip,
        media_id=media_id,
        run_id=run_id,
        source_title=source_title,
        media_sha256=media_sha256,
        app_version=app_version,
        stages=stages,
    )


@router.get("/review/{run_id}/annotations/{annotation_id}/export.json")
def export_annotation_json(
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
) -> JSONResponse:
    """One highlight as a JSON provenance manifest (issue #122)."""
    from voxint import __version__

    run = _run_or_404(session, run_id)
    try:
        row = live_annotation_or_404(session, run_id, annotation_id)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    resolved = resolve_annotation_spans(lines, covered, [stored_anchor_from_row(row)])[0]
    tags = [t.name for t in tags_for_annotations(session, [row.id]).get(row.id, [])]
    settings: Settings = request.app.state.settings
    clip = _clip_ref_for_annotation(session, run_id, annotation_id, settings)
    stages = _stage_provenance_from_run(session, run_id)
    media_item = run.media_item
    manifest = _manifest_for_resolved(
        resolved,
        row,
        lines,
        source_title=_run_source_title(run),
        tags=tags,
        clip=clip,
        media_id=media_item.id,
        run_id=run_id,
        media_sha256=media_item.sha256,
        app_version=__version__,
        stages=stages,
        exported_at=datetime.now(UTC),
    )
    fn = f"voxint-{run_id.hex[:8]}-manifest-{annotation_id.hex[:8]}.json"
    return JSONResponse(
        content=manifest,
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )


@router.get("/review/{run_id}/annotations/export.json")
def export_annotations_json(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    request: Request,
    tag: Annotated[list[uuid.UUID] | None, Query()] = None,
) -> JSONResponse:
    """All (filtered) highlights as a JSON provenance bundle (issue #122)."""
    from voxint import __version__

    run = _run_or_404(session, run_id)
    tag_ids = tag or []
    _require_filter_tags_exist(session, tag_ids)
    rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
    if not rows:
        empty = {
            "schema_version": 1,
            "kind": "quote_provenance_bundle",
            "quotes": [],
        }
        return JSONResponse(content=empty)
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    resolved_map = {
        r.annotation_id: r
        for r in resolve_annotation_spans(
            lines, covered, [stored_anchor_from_row(row) for row in rows]
        )
    }
    tags_by_id = tags_for_annotations(session, [row.id for row in rows])
    settings: Settings = request.app.state.settings
    stages = _stage_provenance_from_run(session, run_id)
    media_item = run.media_item
    exported_at = datetime.now(UTC)

    # Batch clip lookup: one query for all annotation clips in this run.
    clip_rows = (
        session.execute(
            select(AudioArtifact)
            .where(
                AudioArtifact.pipeline_run_id == run_id,
                AudioArtifact.kind == ArtifactKind.AUDIO_CLIP.value,
                AudioArtifact.reclaimed_at.is_(None),
            )
        )
        .scalars()
        .all()
    )
    clips_by_ann: dict[str, AudioArtifact] = {}
    for cr in clip_rows:
        ann_id_str = (cr.meta or {}).get("annotation_id")
        if ann_id_str and (
            ann_id_str not in clips_by_ann
            or cr.created_at > clips_by_ann[ann_id_str].created_at
        ):
            clips_by_ann[ann_id_str] = cr

    ordered = sorted(rows, key=lambda row: resolved_order_key(resolved_map[row.id]))
    quote_entries: list[dict[str, Any]] = []
    for row in ordered:
        resolved = resolved_map[row.id]
        ann_clip: ClipRef | None = None
        cr = clips_by_ann.get(str(row.id))
        if cr is not None:
            meta = cr.meta or {}
            clip_path = settings.media_root / cr.path
            clip_sha256 = ""
            if clip_path.is_file():
                import hashlib

                clip_sha256 = hashlib.sha256(clip_path.read_bytes()).hexdigest()
            ann_clip = ClipRef(
                id=cr.id,
                download_url=f"/runs/{run_id}/clips/{cr.id}",
                filename=clip_download_filename(run_id, cr.id),
                sha256=clip_sha256,
                sample_rate=meta.get("sample_rate", 16000),
                channels=1,
                start_sample=meta.get("start_sample", 0),
                end_sample=meta.get("end_sample", 0),
            )
        entry = _manifest_for_resolved(
            resolved,
            row,
            lines,
            source_title=_run_source_title(run),
            tags=[t.name for t in tags_by_id.get(row.id, [])],
            clip=ann_clip,
            media_id=media_item.id,
            run_id=run_id,
            media_sha256=media_item.sha256,
            app_version=__version__,
            stages=stages,
            exported_at=exported_at,
        )
        quote_entries.append({"quote": entry["quote"], "clip": entry["clip"]})

    bundle = build_quote_bundle(
        exported_at=exported_at,
        media_id=media_item.id,
        run_id=run_id,
        source_title=_run_source_title(run),
        media_sha256=media_item.sha256,
        app_version=__version__,
        stages=stages,
        quotes=quote_entries,
    )
    fn = f"voxint-{run_id.hex[:8]}-manifests.json"
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )


@router.post("/review/{run_id}/annotations/{annotation_id}/clips")
def extract_annotation_clip(
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Extract (or adopt the cached) attributed audio clip for one highlight
    (issue #88). CSRF-gated, onboarding-auth, NO claim — extraction is idempotent
    and content-addressed, so a same-span replay adopts the one canonical clip.

    A foreign, forged, or soft-deleted id is 404 (fail closed). A stale highlight
    is 409 ``X-Voxint-Conflict: stale`` (refresh or re-anchor it first). A
    highlight with no precise word timing is 422 (nothing to cut). A clip that
    cannot be generated carries the service's honest status (409 the processed
    audio is gone, 422 the span is unclippable). Returns the clip id + its
    download URL (201)."""
    _require_csrf(request, CSRF_CLIP_EXTRACT, csrf_token)
    _run_or_404(session, run_id)
    try:
        row = live_annotation_or_404(session, run_id, annotation_id)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    covered = load_covered_segments(session, run_id)
    resolved = resolve_annotation_spans(lines, covered, [stored_anchor_from_row(row)])[0]
    if resolved.stale:
        raise HTTPException(
            status_code=409,
            detail="this highlight is stale; refresh or re-anchor it before extracting a clip",
            headers=_ANNOTATION_STALE_HEADERS,
        )
    if (
        resolved.start_seconds is None
        or resolved.end_seconds is None
        or resolved.timing_precision != TIMING_WORD
    ):
        raise HTTPException(
            status_code=422,
            detail="this highlight has no precise word timing to clip",
        )
    try:
        clip_id = generate_or_adopt_clip(
            session,
            run_id,
            annotation_id=row.id,
            annotation_source_text_hash=row.source_text_hash,
            start_seconds=resolved.start_seconds,
            end_seconds=resolved.end_seconds,
            settings=request.app.state.settings,
            gate=_get_media_gate(request),
        )
        session.commit()
    except ClipServiceError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc
    return JSONResponse(
        {
            "clipId": str(clip_id),
            "downloadUrl": f"/runs/{run_id}/clips/{clip_id}",
            "filename": clip_download_filename(run_id, clip_id),
        },
        status_code=201,
    )

@router.post("/review/{run_id}/annotations/export/live.md")
def export_live_pull_quote(
    run_id: uuid.UUID,
    operator: OperatorDep,
    session: SessionDep,
    start_segment_id: Annotated[uuid.UUID, Form()],
    start_offset: Annotated[int, Form()],
    end_segment_id: Annotated[uuid.UUID, Form()],
    end_offset: Annotated[int, Form()],
    client_quote: Annotated[str, Form()],
    start_child_word_start: Annotated[int | None, Form()] = None,
    start_child_word_end: Annotated[int | None, Form()] = None,
    end_child_word_start: Annotated[int | None, Form()] = None,
    end_child_word_end: Annotated[int | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
    tags: Annotated[list[uuid.UUID] | None, Form()] = None,
) -> Response:
    """A live pull-quote for an UNSAVED selection (issue #86, docs/annotations.md):
    classify + validate exactly as create, but persist NOTHING and return the
    Markdown. Onboarding-auth only — no claim, no nonce, no CSRF, because nothing is
    written. Same caps/validation as create (422 on a bad anchor or cap; 409 stale on
    a drifted client quote). The optional ``note``/``tags`` are echoed into the quote
    trailer, never stored; unknown tag ids are 404."""
    run = _run_or_404(session, run_id)
    payload = _capture_payload_from_form(
        start_segment_id,
        start_offset,
        start_child_word_start,
        start_child_word_end,
        end_segment_id,
        end_offset,
        end_child_word_start,
        end_child_word_end,
        client_quote,
    )
    try:
        tag_names = resolve_tag_names(session, list(tags) if tags else [])
        normalized_note = normalize_note(note)
        derived, covered = derive_live_anchor(session, run_id, payload)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    anchor = stored_anchor_from_derived(derived, uuid.uuid4())
    resolved = resolve_annotation_spans(lines, covered, [anchor])[0]
    markdown = _pull_quote_markdown(
        resolved,
        lines,
        source_title=_run_source_title(run),
        tags=tag_names,
        note=normalized_note,
    )
    return Response(content=markdown, media_type=ANNOTATION_MEDIA_TYPES["md"])

@router.post("/review/{run_id}/annotations")
def create_annotation(
    run_id: uuid.UUID,
    session: SessionDep,
    operator: OperatorDep,
    token: Annotated[uuid.UUID, Form()],
    nonce: Annotated[str, Form(min_length=8, max_length=64)],
    start_segment_id: Annotated[uuid.UUID, Form()],
    start_offset: Annotated[int, Form()],
    end_segment_id: Annotated[uuid.UUID, Form()],
    end_offset: Annotated[int, Form()],
    client_quote: Annotated[str, Form()],
    color_index: Annotated[int, Form()],
    start_child_word_start: Annotated[int | None, Form()] = None,
    start_child_word_end: Annotated[int | None, Form()] = None,
    end_child_word_start: Annotated[int | None, Form()] = None,
    end_child_word_end: Annotated[int | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
    tags: Annotated[list[uuid.UUID] | None, Form()] = None,
) -> JSONResponse:
    """Create an annotation (the sole create path). Claim-gated and idempotent by
    the client ``nonce``: a same-payload replay returns the original row, a
    different payload is a 409 idempotency conflict. The server classifies the
    anchor kind and derives the quote/hash/seconds; the client never picks the
    kind. Returns the created annotation's island shape (201)."""
    run = _verify_annotation_claim(session, run_id, token)
    _reject_if_archived(run)
    payload = _capture_payload_from_form(
        start_segment_id,
        start_offset,
        start_child_word_start,
        start_child_word_end,
        end_segment_id,
        end_offset,
        end_child_word_start,
        end_child_word_end,
        client_quote,
    )
    try:
        row = capture_annotation(
            session,
            run_id=run_id,
            payload=payload,
            operator=operator,
            nonce=nonce,
            color_index=color_index,
            note=note,
            tag_ids=list(tags) if tags else None,
        )
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    return JSONResponse(_annotation_shapes(session, run_id, [row])[0], status_code=201)

@router.patch("/review/{run_id}/annotations/{annotation_id}")
def patch_annotation(
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    session: SessionDep,
    operator: OperatorDep,
    token: Annotated[uuid.UUID, Form()],
    op: Annotated[str, Form()],
    color_index: Annotated[int | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
    tags: Annotated[list[uuid.UUID] | None, Form()] = None,
    start_segment_id: Annotated[uuid.UUID | None, Form()] = None,
    start_offset: Annotated[int | None, Form()] = None,
    end_segment_id: Annotated[uuid.UUID | None, Form()] = None,
    end_offset: Annotated[int | None, Form()] = None,
    start_child_word_start: Annotated[int | None, Form()] = None,
    start_child_word_end: Annotated[int | None, Form()] = None,
    end_child_word_start: Annotated[int | None, Form()] = None,
    end_child_word_end: Annotated[int | None, Form()] = None,
    client_quote: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Mutate an existing annotation. ``op`` is one of three mutually-exclusive
    operations (docs/annotations.md): ``edit`` replaces metadata (colour/note/
    tags), ``refresh`` re-derives the quote/hash/seconds when the anchor still
    deterministically identifies its span, and ``reanchor`` atomically replaces
    the anchor from a fresh capture payload. Claim-gated; a deleted or foreign id
    is 404. Returns the updated annotation's island shape."""
    run = _verify_annotation_claim(session, run_id, token)
    _reject_if_archived(run)
    try:
        if op == "edit":
            if color_index is None:
                raise HTTPException(status_code=422, detail="edit requires color_index")
            row = update_annotation(
                session,
                run_id=run_id,
                annotation_id=annotation_id,
                color_index=color_index,
                note=note,
                tag_ids=list(tags) if tags else None,
            )
        elif op == "refresh":
            row = refresh_annotation(session, run_id=run_id, annotation_id=annotation_id)
        elif op == "reanchor":
            if (
                start_segment_id is None
                or start_offset is None
                or end_segment_id is None
                or end_offset is None
                or client_quote is None
            ):
                raise HTTPException(
                    status_code=422,
                    detail="reanchor requires a full capture payload",
                )
            payload = _capture_payload_from_form(
                start_segment_id,
                start_offset,
                start_child_word_start,
                start_child_word_end,
                end_segment_id,
                end_offset,
                end_child_word_start,
                end_child_word_end,
                client_quote,
            )
            row = reanchor_annotation(
                session,
                run_id=run_id,
                annotation_id=annotation_id,
                payload=payload,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"unknown op {op!r}; expected edit, refresh, or reanchor",
            )
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    return JSONResponse(_annotation_shapes(session, run_id, [row])[0])

@router.delete("/review/{run_id}/annotations/{annotation_id}")
def delete_annotation(
    run_id: uuid.UUID,
    annotation_id: uuid.UUID,
    session: SessionDep,
    operator: OperatorDep,
    token: Annotated[uuid.UUID, Form()],
) -> Response:
    """Soft-delete an annotation (idempotent): a repeat DELETE of an already-
    deleted row is a no-op 204, never a 404 — the row still exists and a create
    replay still finds it. An unknown/foreign id is 404. Claim-gated."""
    run = _verify_annotation_claim(session, run_id, token)
    _reject_if_archived(run)
    try:
        soft_delete_annotation(session, run_id=run_id, annotation_id=annotation_id)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    return Response(status_code=204)

@router.get("/annotations/tags")
def list_annotation_tags(operator: OperatorDep, session: SessionDep) -> JSONResponse:
    """The global tag universe (all tags, archived included), in display order.
    Onboarding-auth only; tags are not run-scoped."""
    return JSONResponse({"tags": [_tag_shape(t) for t in list_tags(session)]})

@router.post("/annotations/tags")
def create_annotation_tag(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    name: Annotated[str, Form()],
    color: Annotated[int, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Create a global tag (name + palette colour). CSRF-gated like run notes —
    tag writes have no run or claim context. A normalized-name duplicate is a 409;
    a blank/over-cap name or bad colour is a 422. Returns the created tag (201)."""
    _require_csrf(request, CSRF_ANNOTATION_TAGS, csrf_token)
    try:
        tag = create_tag(session, name=name, color=color)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    return JSONResponse(_tag_shape(tag), status_code=201)

@router.patch("/annotations/tags/{tag_id}")
def update_annotation_tag(
    tag_id: uuid.UUID,
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    name: Annotated[str | None, Form()] = None,
    color: Annotated[int | None, Form()] = None,
    archived: Annotated[bool | None, Form()] = None,
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Rename / recolour / archive / restore a tag. Each field is independently
    optional (absent leaves it untouched; ``archived`` is a tri-state). CSRF-gated.
    A rename colliding with a different tag is a 409; an unknown id is 404."""
    _require_csrf(request, CSRF_ANNOTATION_TAGS, csrf_token)
    try:
        tag = update_tag(session, tag_id=tag_id, name=name, color=color, archived=archived)
    except AnnotationError as exc:
        raise _annotation_http_error(exc) from exc
    return JSONResponse(_tag_shape(tag))

