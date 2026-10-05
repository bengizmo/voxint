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
)
from voxint.adjudication.word_marks import effective_marks
from voxint.app_settings import get_app_settings, resolve_effective_filler_list
from voxint.config import Settings
from voxint.db.models import SegmentWordMark
from voxint.export.fillers import Protected, RemovalCause
from voxint.export.turn_filters import apply_turn_filters


def build_word_marks_payload(
    session: Session, run_id: uuid.UUID, settings: Settings, *,
    focus_segment_id: uuid.UUID | None,
) -> dict[str, Any]:
    """Project and annotate a single materialised walk, excluding stale marks."""
    states, turn_rows, emissions = load_turn_inputs(session, run_id)
    projected = project_turns(emissions, turn_rows, states, text=TranscriptText.CORRECTED)
    marks = effective_marks(session, run_id)
    anchors = [emission_anchors(e, text=TranscriptText.CORRECTED) for e in emissions]
    anchored = {
        (a.segment_id, a.token_start, a.token_end) for item in anchors for a in item.anchors
    }
    placeable = {key: action for key, action in marks.items() if key in anchored}
    filler_list = resolve_effective_filler_list(get_app_settings(session), settings)
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
    segment_starts = {e.seg.id: e.seg.start_seconds for e in emissions}
    for emission, item in zip(emissions, anchors, strict=True):
        units: list[dict[str, Any]] = []
        for anchor in item.anchors:
            key = (anchor.segment_id, anchor.token_start, anchor.token_end)
            if (
                emission.seg.id != focus_segment_id and key not in removed
                and key not in protected and key not in placeable
            ):
                continue
            units.append({
                "start": anchor.token_start, "end": anchor.token_end,
                "from": anchor.lex_start, "to": anchor.lex_end,
                "removed": removed.get(key), "protected": key in protected,
                "mark": placeable.get(key),
            })
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
        "runId": str(run_id), "fillerListDefault": filler_list.is_default,
        "emissions": rendered, "stale": stale,
    }
    seq = session.scalar(select(func.max(SegmentWordMark.seq)).where(
        SegmentWordMark.pipeline_run_id == run_id,
    )) or 0
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    payload["version"] = f"{seq}.{digest}"
    return payload
