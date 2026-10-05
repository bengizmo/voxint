"""CORRECTED clean-up status from the export's committed filler trace.

An absent emission means "markable, nothing detected". Unfocused emissions
contain only annotated units; focus includes every unit of the parent segment.
Unmarkable emissions always appear, with a reason and no units. Offsets address
the island's shown text in Python str code points, including astral characters.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.turns import (
    WordMarkKey,
    emission_anchors,
    load_turn_inputs,
    project_turns,
    selected_text,
)
from voxint.adjudication.word_marks import effective_marks
from voxint.app_settings import get_app_settings, resolve_effective_filler_list
from voxint.config import Settings
from voxint.db.models import SegmentWordMark, TranscriptSegment
from voxint.export.filler_lists import FillerListError
from voxint.export.fillers import Protected, RemovalCause
from voxint.export.turn_filters import apply_turn_filters


def build_word_marks_payload(
    session: Session, run_id: uuid.UUID, settings: Settings, *,
    focus_segment_id: uuid.UUID | None,
) -> dict[str, Any]:
    """Project and annotate a single materialised walk, excluding stale marks.

    Callers read inside one snapshot: the GET route opens a repeatable-read
    transaction, and writes build this inside their own claim-serialised one.

    A broken saved filler list does not fail the read: detection is empty and
    ``detectionError`` says why, so marks can still be listed and cleared.
    """
    # Read first: a mark committed during the build can only make this older.
    seq = session.scalar(select(func.max(SegmentWordMark.seq)).where(
        SegmentWordMark.pipeline_run_id == run_id,
    )) or 0
    states, turn_rows, emissions = load_turn_inputs(session, run_id)
    projected = project_turns(emissions, turn_rows, states, text=TranscriptText.CORRECTED)
    marks = effective_marks(session, run_id)
    anchors = [emission_anchors(e, text=TranscriptText.CORRECTED) for e in emissions]
    anchored = {
        (a.segment_id, a.token_start, a.token_end) for item in anchors for a in item.anchors
    }
    placeable = {key: action for key, action in marks.items() if key in anchored}
    detection_error: str | None = None
    try:
        filler_list = resolve_effective_filler_list(get_app_settings(session), settings)
    except FillerListError as exc:
        filler_list, detection_error = None, str(exc)
    filtered = apply_turn_filters(
        projected, fillers=filler_list, drop_repeats=False, marks=placeable,
    )
    removed: dict[WordMarkKey, RemovalCause] = {}
    protected: set[WordMarkKey] = set()
    for entry in filtered.trace:
        if isinstance(entry, Protected):
            protected.update(entry.sources)
        else:
            for key in entry.sources:
                removed[key] = entry.cause
    rendered: list[dict[str, Any]] = []
    # The version covers the run's state, never the focus: an unfocused shape.
    state: list[Any] = []
    segment_starts = {e.seg.id: e.seg.start_seconds for e in emissions}
    missing = {key[0] for key in marks} - segment_starts.keys()
    if missing:
        segment_starts.update(session.execute(select(
            TranscriptSegment.id, TranscriptSegment.start_seconds,
        ).where(TranscriptSegment.id.in_(missing))).tuples().all())
    for emission, item in zip(emissions, anchors, strict=True):
        units: list[dict[str, Any]] = []
        for anchor in item.anchors:
            key = (anchor.segment_id, anchor.token_start, anchor.token_end)
            annotated = key in removed or key in protected or key in placeable
            if not annotated and emission.seg.id != focus_segment_id:
                continue
            unit = {
                "start": anchor.token_start, "end": anchor.token_end,
                "from": anchor.lex_start, "to": anchor.lex_end,
                "removed": removed.get(key), "protected": key in protected,
                "mark": placeable.get(key),
            }
            units.append(unit)
            if annotated:
                state.append([str(emission.seg.id), unit])
        if units or not item.anchors or emission.seg.id == focus_segment_id:
            rendered.append({
                "segmentId": str(emission.seg.id),
                "wordStart": emission.child.word_start if emission.child else None,
                "wordEnd": emission.child.word_end if emission.child else None,
                "markable": bool(item.anchors), "reason": item.reason, "units": units,
            })
    stale = [
        {
            "segmentId": str(key[0]), "start": key[1], "end": key[2],
            "action": marks[key], "segmentStart": segment_starts[key[0]],
        }
        for key in sorted(
            (key for key in marks if key not in anchored),
            key=lambda key: (segment_starts[key[0]], key[1], key[2], str(key[0])),
        )
    ]
    payload: dict[str, Any] = {
        "runId": str(run_id),
        "fillerListDefault": filler_list.is_default if filler_list is not None else None,
        "detectionError": detection_error,
        "emissions": rendered, "stale": stale,
    }
    # The shown text is hashed too: an edit can leave every unit's status alone.
    texts = [
        [str(e.seg.id), e.child.word_start if e.child else None, item.reason,
         selected_text(e, TranscriptText.CORRECTED)]
        for e, item in zip(emissions, anchors, strict=True)
    ]
    canonical = json.dumps(
        [state, stale, texts, payload["fillerListDefault"], detection_error],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    payload["version"] = f"{seq}.{digest}"
    return payload
