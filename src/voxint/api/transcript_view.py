"""Transcript-island payload assembly shared by the run and review transcripts.

Builds the props, per-segment shapes, and reconcile payload the React
transcript islands hydrate from, plus the split/correction lookups those
shapes need and the label universe both transcript surfaces render. Used by
the run transcript (routers/legacy_runs.py) and the review transcript and
mutations (routers/legacy_review.py), so it lives in a neutral module none
of them own.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.corrections_view import (
    DeclaredRuleIndex,
    build_declared_rule_index,
    resolve_segment_provenance,
)
from voxint.adjudication.review_state import verified_progress
from voxint.adjudication.splits import derive_children
from voxint.adjudication.transcript import TranscriptLine, TranscriptText, attributed_transcript
from voxint.api.playback import PlaybackCapability
from voxint.api.speaker_colors import speaker_palette
from voxint.config import Settings
from voxint.db.models import (
    DiarizationTurn,
    PipelineRun,
    SegmentReviewState,
    SegmentSplitBoundary,
    TranscriptSegment,
)
from voxint.enrichment.outline import build_outline
from voxint.media.peaks import peaks_artifact_row

logger = logging.getLogger(__name__)

def _run_label_universe(session: Session, run_id: uuid.UUID) -> set[str]:
    """Every diarization label present in a run, from BOTH its diarization turns
    and its transcript segments.

    A transcript segment may carry a label with no turn (the supported degenerate
    case the resolver's turn-derived ``label_states`` does not enumerate), and a
    turn's label may have no segment; the union covers both. This is the ONE
    canonical universe the per-speaker palette (#50) is built from, so the
    transcript page, its JS-off fallback, and the workbench cards color a given
    label identically. Two cheap indexed ``DISTINCT`` queries — deliberately not
    ``label_states`` (which resolves turn stats, proposals, decisions, and merges)."""
    turn_labels = session.execute(
        select(DiarizationTurn.label)
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    segment_labels = session.execute(
        select(TranscriptSegment.diarization_label)
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .distinct()
    ).scalars()
    return {*turn_labels, *(label for label in segment_labels if label is not None)}


def _wants_island_json(request: Request) -> bool:
    """True only when the caller explicitly asks for JSON (the island's ``apiFetch``
    sets ``Accept: application/json``). Unlike ``not _wants_html``, this is a
    POSITIVE signal: the htmx labels workbench and the default ``*/*`` test client
    stay on the server-rendered path, so a route that also serves the island keeps
    its HTML-fragment contract byte-identical for every non-island caller."""
    return "application/json" in request.headers.get("accept", "")


# Capability reason codes meaning GET /media would not serve bytes at all (as
# opposed to a present-but-untrusted timeline). Mirrors MEDIA_UNAVAILABLE_CODES
# in frontend/src/components/PlaybackControls.tsx.
_MEDIA_UNAVAILABLE_CODES = frozenset({"media_missing", "media_reclaimed", "media_unservable"})


def _transcript_island_props(
    session: Session,
    run_id: uuid.UUID,
    lines: list[TranscriptLine],
    palette: dict[str, int],
    capability: PlaybackCapability,
    settings: Settings,
) -> dict[str, Any]:
    """Shared island props for the linear transcript surfaces (issues #48/#50/#53).

    Both the read-only ``transcript-player`` and the claim-gated ``review-stepper``
    read the SAME per-segment shape, so the hydrated island and the JS-off
    fallback flag/color identically and a segment's write id never drifts between
    the display and the review loop.
    """
    # Waveform strip (issue #57): the strip's colored regions come from the
    # DIARIZATION TURNS, not the transcript segments — a segment carries only
    # its dominant-overlap label, which is not an honest who-spoke-when map
    # (it hides overlaps and untranscribed speech). Same palette as the list
    # badges, so the colors can never disagree.
    turn_rows = session.execute(
        select(
            DiarizationTurn.start_seconds,
            DiarizationTurn.end_seconds,
            DiarizationTurn.label,
            DiarizationTurn.overlap,
        )
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .order_by(DiarizationTurn.start_seconds, DiarizationTurn.turn_index)
    ).all()
    # peaksUrl is server-owned truth like mediaUrl: non-null only when the peaks
    # route could actually answer 200 — either the WAV is servable (a first
    # request computes the envelope) OR it was formally RECLAIMED and a cached
    # envelope survives (served unverified, by design). A cached row does NOT
    # rescue media_missing/media_unservable: with no reclamation stamp the route
    # cannot verify the (absent/unopenable) WAV, so it fails closed to 404/410 —
    # emitting the URL there would make the island fetch on a loop. Any capability
    # reason not about media servability (a bad timeline) still leaves the
    # amplitude route answerable, so it does not gate peaksUrl.
    reason_codes = {r.code for r in capability.reasons}
    media_unavailable = bool(reason_codes & _MEDIA_UNAVAILABLE_CODES)
    reclaimed_with_cache = (
        "media_reclaimed" in reason_codes
        and peaks_artifact_row(session, run_id) is not None
    )
    peaks_available = not media_unavailable or reclaimed_with_cache
    # The run's frozen pack, resolved once for every segment's correction
    # provenance (#83) — read-time, from the immutable per-run snapshot.
    rule_index = _load_run_rule_index(session, run_id)
    return {
        "runId": str(run_id),
        "mediaUrl": f"/media/{run_id}",
        "peaksUrl": f"/media/{run_id}/peaks" if peaks_available else None,
        "capability": capability.to_props(),
        "turns": [
            {
                "start": start,
                "end": end,
                "paletteIndex": palette.get(label),
                "overlap": overlap,
            }
            for start, end, label, overlap in turn_rows
        ],
        # Low-confidence triage threshold (issue #53): the island and the JS-off
        # fallback compare against the SAME server setting, so they flag
        # identically. A segment with confidence None is never flagged.
        "lowConfidenceThreshold": settings.review_low_confidence_threshold,
        "segments": [
            _island_segment(ln, palette, rule_index) for ln in lines
        ],
        # Navigable outline (issue #87): grounded entity-mention jump targets plus
        # inert summary/topics context. Read-only navigation, so it rides in the
        # SHARED props both surfaces build; only the review-stepper renders the
        # panel today. The client resolves each target's startSeconds to a current
        # line at click time (segment ordinals can diverge from rendered lines
        # after a split), so no line index is baked here.
        "outline": build_outline(session, run_id, settings),
    }


def _load_run_rule_index(
    session: Session, run_id: uuid.UUID
) -> DeclaredRuleIndex | None:
    """The run's frozen domain-pack snapshot resolved into a declared-rule index
    for read-time provenance (#83), or ``None`` when the run is gone or its
    snapshot is absent/corrupt.

    Reads the ``domain_pack`` snapshot column DIRECTLY and hands it to
    :func:`build_declared_rule_index` — never through ``domain_pack_from_snapshot``,
    which would degrade a NULL/corrupt snapshot to the current default pack and
    fabricate declarations this run never had.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        return None
    return build_declared_rule_index(run.domain_pack)


def _island_segment(
    ln: TranscriptLine,
    palette: dict[str, int],
    rule_index: DeclaredRuleIndex | None = None,
) -> dict[str, Any]:
    """One transcript line as the island's per-segment shape — the ONE builder the
    hydrated props and the split-route response share, so a page reload and a live
    split can never disagree on a segment's fields.

    ``sourceSegmentId`` is the immutable PARENT id (issue #59): the verify / correct
    / split write target, identical to ``segmentId`` for an unsplit line and shared
    across a split parent's derived children. ``reviewTarget`` is true on exactly
    one line per parent — the queue entry — so the N-of-M loop counts one target
    per parent and never double-counts children.

    ``corrections`` / ``rawText`` (#83) carry deterministic domain-pack correction
    provenance and the immutable raw evidence for the compare/reset affordance.
    Both are whole-segment concerns: a split child (``word_start`` set) never
    carries them (the parent's spans address its full enhanced text, not a child
    slice), and ``correction_trace`` is ``None`` there anyway (a corrected segment
    is never split).
    """
    is_split_child = ln.word_start is not None
    # Operator edit supersedes pipeline provenance (#83): once the operator saves
    # their own text (`corrected`), the domain-pack trace's spans address the
    # PIPELINE-enhanced text, not the operator-effective text now shown — so the
    # "corrected by domain pack" marker would be stale and misleading. The client
    # clears it locally on a /text save, but the SERVER must own the rule too, or a
    # page reload (and any whole-run reconcile via /split or /relabel, which reuse
    # this builder) resurrects the stale marker. `rawText` stays exposed — the
    # compare / reset-to-raw affordance remains honest and useful after an edit.
    corrections = (
        None
        if is_split_child or ln.corrected
        else resolve_segment_provenance(
            ln.correction_trace, ln.corrector_version, rule_index
        )
    )
    return {
        "start": ln.start_seconds,
        "end": ln.end_seconds,
        "speaker": ln.speaker,
        "text": ln.text,
        "label": ln.diarization_label,
        "confidence": ln.confidence,
        # None short-circuits (palette is keyed on real labels only); keeps mypy
        # happy without changing the value (get(None) → None).
        "paletteIndex": (
            palette.get(ln.diarization_label) if ln.diarization_label is not None else None
        ),
        # Per-segment review state (issues #53/#58). segmentId is the write target
        # for verify/correct; verified/corrected drive the verify-and-advance loop
        # and the "edited" badge. None segmentId (a synthetic/blank line) is simply
        # never a review target.
        "segmentId": (str(ln.segment_id) if ln.segment_id is not None else None),
        "verified": ln.verified,
        "corrected": ln.corrected,
        # Split provenance (issue #59): the parent write target + the single
        # queue-entry flag. sourceSegmentId == segmentId for an unsplit line.
        "sourceSegmentId": (
            str(ln.source_segment_id) if ln.source_segment_id is not None else None
        ),
        "reviewTarget": ln.review_target,
        # Word-range coordinates of a split child (issue #59 slice 3): what the
        # per-child reassign picker posts to /relabel to scope a ruling to this
        # child. Both None on unsplit and synthetic lines.
        "wordStart": ln.word_start,
        "wordEnd": ln.word_end,
        # The child's OWN range-override speaker id (None ⇒ inheriting): the picker
        # binds its <select> to this so it shows a child-scoped assignment only when
        # one exists, never an inherited speaker mislabeled as a child ruling.
        "wordRangeSpeakerId": (
            str(ln.word_range_speaker_id) if ln.word_range_speaker_id is not None else None
        ),
        # Deterministic domain-pack correction provenance (#83): which pack/rule
        # produced each edit, or an honest unavailable state; None when no rule
        # materially fired (or on a split child). Never a text diff — driven by the
        # persisted trace (trace_has_entries) alone.
        "corrections": corrections,
        # The immutable raw ASR text for the whole segment (#83), for the console's
        # compare / reset-to-raw affordance. None on split children (raw is a
        # whole-segment concern) and synthetic export lines.
        "rawText": None if is_split_child else ln.raw_text,
    }


def _run_island_segments(session: Session, run_id: uuid.UUID) -> list[dict[str, Any]]:
    """The run's island segment payload (issue #59) — CORRECTED variant, split
    parents expanded — for a live write to reconcile the console against server
    truth. Same builder as hydration, so a split response and a page reload agree."""
    palette = speaker_palette(_run_label_universe(session, run_id))
    lines = attributed_transcript(session, run_id, text=TranscriptText.CORRECTED)
    rule_index = _load_run_rule_index(session, run_id)
    return [_island_segment(ln, palette, rule_index) for ln in lines]


def _run_reconcile_response(session: Session, run_id: uuid.UUID) -> JSONResponse:
    """The whole-run island reconcile — every segment (split parents expanded) plus
    the run's N-of-M counter — the shape a STRUCTURAL write returns so the console
    adopts server truth wholesale rather than patching one line. Shared by /split
    and the island /relabel path (a reassignment changes a child's speaker string,
    which a per-segment patch cannot express, so both re-render the whole run)."""
    verified_n, total = verified_progress(session, run_id)
    return JSONResponse(
        {
            "segments": _run_island_segments(session, run_id),
            "progress": {"verified": verified_n, "total": total},
        }
    )


def _segment_is_split(session: Session, segment_id: uuid.UUID) -> bool:
    """Whether a segment carries at least one operator split boundary (issue #59)."""
    return (
        session.execute(
            select(SegmentSplitBoundary.id)
            .where(SegmentSplitBoundary.parent_segment_id == segment_id)
            .limit(1)
        ).first()
        is not None
    )


def _segment_child_ranges(
    session: Session, segment: TranscriptSegment
) -> set[tuple[int, int]]:
    """The half-open ``(word_start, word_end)`` ranges of a segment's current
    derived split children (issue #59 slice 3).

    The reassign route validates a submitted range against this set so a ruling
    can only target a child that actually exists right now — an arbitrary range
    would write a ledger row the read path never applies (it matches children by
    exact coordinates). Empty for an unsplit or unsplittable segment."""
    cuts = list(
        session.execute(
            select(SegmentSplitBoundary.word_index).where(
                SegmentSplitBoundary.parent_segment_id == segment.id
            )
        ).scalars()
    )
    if not cuts:
        return set()
    children = derive_children(segment, cuts)
    if children is None or len(children) < 2:
        return set()
    return {(child.word_start, child.word_end) for child in children}


def _segment_is_corrected(session: Session, segment_id: uuid.UUID) -> bool:
    """Whether a segment has operator-corrected text (issues #58/#59)."""
    row = session.get(SegmentReviewState, segment_id)
    return row is not None and row.corrected_text is not None


