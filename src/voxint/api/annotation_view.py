"""Annotation presentation helpers shared by the editor and adjudication API.

Extracted from the legacy review router (issue #158) so the editor page and
the adjudication mutation surface can both build annotation island payloads
without importing a legacy module.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from voxint.adjudication.annotations import (
    annotations_for_run,
    list_tags,
    load_covered_segments,
    resolve_annotation_spans,
    resolved_order_key,
    stored_anchor_from_row,
    tags_for_annotations,
)
from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.db.models import (
    HIGHLIGHT_PALETTE_SIZE,
    MAX_ANNOTATION_NOTE_CHARS,
    MAX_ANNOTATION_QUOTE_CHARS,
    MAX_ANNOTATION_SPAN_SEGMENTS,
    MAX_TAG_NAME_CHARS,
    MAX_TAGS_PER_ANNOTATION,
    AnnotationTag,
    TranscriptAnnotation,
)


def annotation_limits() -> dict[str, int]:
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


def tag_shape(tag: AnnotationTag) -> dict[str, Any]:
    """One tag's island/JSON shape (camelCase, matching the frontend islands)."""
    return {
        "id": str(tag.id),
        "name": tag.name,
        "color": tag.color,
        "archived": tag.archived_at is not None,
    }


def annotation_shapes(
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
                "tags": [tag_shape(t) for t in tags_by_id.get(row.id, [])],
            }
        )
    return shapes


def annotations_payload(
    session: Session, run_id: uuid.UUID, tag_ids: list[uuid.UUID]
) -> dict[str, Any]:
    """The GET /annotations body: the run's live annotations (optionally OR-filtered
    by tag) as island shapes, plus the full tag universe for the panel/picker."""
    rows = annotations_for_run(session, run_id, tag_ids=tag_ids or None)
    return {
        "annotations": annotation_shapes(session, run_id, rows),
        "tags": [tag_shape(t) for t in list_tags(session)],
    }
