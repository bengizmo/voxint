"""Run-level speaker timeline read model (issue #337).

The timeline keeps diarization geometry and attribution resolution separate:
``DiarizationTurn`` supplies the intervals while :func:`label_states` supplies
the canonical speaker identity and resolution for each run-local label.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from math import isclose
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import LabelState, Resolution, label_states
from voxint.api.speaker_colors import run_label_universe, speaker_palette
from voxint.db.models import DiarizationTurn

ADJACENT_GAP_SECONDS = 0.05


class TimelineInterval(TypedDict):
    start_seconds: float
    end_seconds: float
    label: str
    speaker_name: str | None
    speaker_id: str | None
    resolution: str
    overlap: bool


class TimelineLane(TypedDict):
    label: str
    speaker_name: str | None
    speaker_id: str | None
    resolution: str
    palette_index: int
    total_seconds: float
    turn_count: int
    intervals: list[TimelineInterval]


class SpeakerTimeline(TypedDict):
    duration_seconds: float
    lanes: list[TimelineLane]
    speaker_count: int


def _resolution_value(resolution: Resolution | str) -> str:
    """Return the JSON representation for a resolver resolution."""
    return resolution.value if isinstance(resolution, Resolution) else str(resolution)


def _speaker_id_value(speaker_id: uuid.UUID | None) -> str | None:
    return str(speaker_id) if speaker_id is not None else None


def _interval(turn: DiarizationTurn, state: LabelState) -> TimelineInterval:
    return {
        "start_seconds": float(turn.start_seconds),
        "end_seconds": float(turn.end_seconds),
        "label": turn.label,
        "speaker_name": state.speaker_name,
        "speaker_id": _speaker_id_value(state.speaker_id),
        "resolution": _resolution_value(state.resolution),
        "overlap": bool(turn.overlap),
    }


def _merge_adjacent(intervals: list[TimelineInterval]) -> list[TimelineInterval]:
    """Coalesce only chronologically adjacent intervals separated by <50 ms."""
    merged: list[TimelineInterval] = []
    for interval in sorted(
        intervals, key=lambda item: (item["start_seconds"], item["end_seconds"])
    ):
        if not merged:
            merged.append(interval.copy())
            continue
        previous = merged[-1]
        gap = interval["start_seconds"] - previous["end_seconds"]
        if gap < ADJACENT_GAP_SECONDS and not isclose(
            gap, ADJACENT_GAP_SECONDS, rel_tol=0.0, abs_tol=1e-9
        ):
            previous["end_seconds"] = max(previous["end_seconds"], interval["end_seconds"])
            previous["overlap"] = previous["overlap"] or interval["overlap"]
        else:
            merged.append(interval.copy())
    return merged


def build_speaker_timeline(session: Session, run_id: uuid.UUID) -> SpeakerTimeline | None:
    """Build the label-lane timeline for one run, or ``None`` without turns."""
    turns = list(
        session.execute(
            select(DiarizationTurn)
            .where(DiarizationTurn.pipeline_run_id == run_id)
            .order_by(DiarizationTurn.turn_index)
        )
        .scalars()
        .all()
    )
    if not turns:
        return None

    states = {state.label: state for state in label_states(session, run_id)}
    palette = speaker_palette(run_label_universe(session, run_id))
    grouped: defaultdict[str, list[DiarizationTurn]] = defaultdict(list)
    for turn in turns:
        grouped[turn.label].append(turn)

    lanes: list[TimelineLane] = []
    for label, label_turns in grouped.items():
        # label_states derives its universe from the same diarization turns, so
        # a missing entry means the resolver contract and the queried data have
        # diverged. Fail explicitly instead of inventing attribution metadata.
        state = states.get(label)
        if state is None:
            raise RuntimeError(f"label_states omitted diarization label {label!r}")
        intervals = _merge_adjacent([_interval(turn, state) for turn in label_turns])
        lanes.append(
            {
                "label": label,
                "speaker_name": state.speaker_name,
                "speaker_id": _speaker_id_value(state.speaker_id),
                "resolution": _resolution_value(state.resolution),
                "palette_index": palette.get(label, 0),
                "total_seconds": float(state.total_seconds),
                "turn_count": int(state.turn_count),
                "intervals": intervals,
            }
        )

    lanes.sort(
        key=lambda lane: (
            lane["speaker_name"] is None,
            (lane["speaker_name"] or lane["label"]).casefold(),
            lane["label"].casefold(),
        )
    )
    return {
        "duration_seconds": max(float(turn.end_seconds) for turn in turns),
        "lanes": lanes,
        "speaker_count": len(lanes),
    }
