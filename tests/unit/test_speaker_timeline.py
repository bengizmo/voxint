"""Unit tests for the run-level speaker timeline read model."""

from __future__ import annotations

import uuid
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import LabelState, Resolution
from voxint.api import speaker_timeline
from voxint.api.speaker_timeline import build_speaker_timeline
from voxint.db.models import DiarizationTurn


def _turn(
    index: int,
    start: float,
    end: float,
    label: str = "SPEAKER_00",
    *,
    overlap: bool = False,
) -> DiarizationTurn:
    return DiarizationTurn(
        pipeline_run_id=uuid.uuid4(),
        turn_index=index,
        start_seconds=start,
        end_seconds=end,
        label=label,
        overlap=overlap,
        overlap_seconds=0.0,
        snr_db=None,
        skip_reason="too_short",
        embedding=None,
        embedding_space=None,
    )


def _state(
    label: str,
    *,
    name: str | None = None,
    speaker_id: uuid.UUID | None = None,
    resolution: Resolution = Resolution.UNRESOLVED,
    total_seconds: float = 1.0,
    turn_count: int = 1,
) -> LabelState:
    return LabelState(
        label=label,
        turn_count=turn_count,
        total_seconds=total_seconds,
        resolution=resolution,
        speaker_id=speaker_id,
        speaker_name=name,
        cosine_speaker_id=None,
        cosine_speaker_name=None,
        cosine_confidence=None,
        cosine_grounded=False,
        llm_hint_name=None,
        effective_decision=None,
    )


def _session(turns: list[DiarizationTurn]) -> Session:
    session = MagicMock(spec=Session)
    session.execute.return_value.scalars.return_value.all.return_value = turns
    return cast(Session, session)


def _patch_states(monkeypatch: pytest.MonkeyPatch, states: list[LabelState]) -> None:
    monkeypatch.setattr(speaker_timeline, "label_states", lambda _session, _run_id: states)


def test_no_diarization_turns_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def states_not_expected(_session: Session, _run_id: uuid.UUID) -> list[LabelState]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(speaker_timeline, "label_states", states_not_expected)

    assert build_speaker_timeline(_session([]), uuid.uuid4()) is None
    assert called is False


def test_single_speaker_multiple_turns_builds_one_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker_id = uuid.uuid4()
    turns = [_turn(0, 0.0, 1.0), _turn(1, 2.0, 4.0)]
    _patch_states(
        monkeypatch,
        [
            _state(
                "SPEAKER_00",
                name="Alice",
                speaker_id=speaker_id,
                resolution=Resolution.HUMAN_ASSIGN,
                total_seconds=3.0,
                turn_count=2,
            )
        ],
    )

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert timeline["speaker_count"] == 1
    assert timeline["lanes"] == [
        {
            "label": "SPEAKER_00",
            "speaker_name": "Alice",
            "speaker_id": str(speaker_id),
            "resolution": "human_assign",
            "total_seconds": 3.0,
            "turn_count": 2,
            "intervals": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "label": "SPEAKER_00",
                    "speaker_name": "Alice",
                    "speaker_id": str(speaker_id),
                    "resolution": "human_assign",
                    "overlap": False,
                },
                {
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "label": "SPEAKER_00",
                    "speaker_name": "Alice",
                    "speaker_id": str(speaker_id),
                    "resolution": "human_assign",
                    "overlap": False,
                },
            ],
        }
    ]


def test_named_lanes_sort_before_unnamed_then_alphabetically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [
        _turn(0, 0.0, 1.0, "Z_UNKNOWN"),
        _turn(1, 1.0, 2.0, "BOB"),
        _turn(2, 2.0, 3.0, "ALICE"),
        _turn(3, 3.0, 4.0, "A_EXCLUDED"),
    ]
    _patch_states(
        monkeypatch,
        [
            _state("Z_UNKNOWN"),
            _state("BOB", name="Bob", speaker_id=uuid.uuid4()),
            _state("ALICE", name="alice", speaker_id=uuid.uuid4()),
            _state("A_EXCLUDED", resolution=Resolution.HUMAN_EXCLUDE),
        ],
    )

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert [lane["label"] for lane in timeline["lanes"]] == [
        "ALICE",
        "BOB",
        "A_EXCLUDED",
        "Z_UNKNOWN",
    ]
    assert timeline["lanes"][2]["resolution"] == "human_exclude"
    assert timeline["lanes"][3]["resolution"] == "unresolved"


def test_only_sub_50ms_nonnegative_gaps_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Input is intentionally not chronological to prove per-lane ordering.
    turns = [
        _turn(2, 2.049, 3.0),
        _turn(0, 0.0, 1.0),
        _turn(3, 3.05, 4.0),
        _turn(1, 1.049, 2.0),
    ]
    _patch_states(
        monkeypatch,
        [_state("SPEAKER_00", total_seconds=3.902, turn_count=4)],
    )

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert [
        (interval["start_seconds"], interval["end_seconds"])
        for interval in timeline["lanes"][0]["intervals"]
    ] == [(0.0, 3.0), (3.05, 4.0)]


def test_overlapping_same_label_intervals_are_not_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = [_turn(0, 0.0, 2.0), _turn(1, 1.5, 3.0)]
    _patch_states(monkeypatch, [_state("SPEAKER_00", turn_count=2)])

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert len(timeline["lanes"][0]["intervals"]) == 2


def test_overlap_flag_survives_adjacent_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [_turn(0, 0.0, 1.0), _turn(1, 1.01, 2.0, overlap=True)]
    _patch_states(monkeypatch, [_state("SPEAKER_00", turn_count=2)])

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert timeline["lanes"][0]["intervals"] == [
        {
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "label": "SPEAKER_00",
            "speaker_name": None,
            "speaker_id": None,
            "resolution": "unresolved",
            "overlap": True,
        }
    ]


def test_duration_is_maximum_turn_end(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [
        _turn(0, 0.0, 9.5, "SPEAKER_00"),
        _turn(1, 1.0, 3.0, "SPEAKER_01"),
    ]
    _patch_states(
        monkeypatch,
        [_state("SPEAKER_00"), _state("SPEAKER_01")],
    )

    timeline = build_speaker_timeline(_session(turns), uuid.uuid4())

    assert timeline is not None
    assert timeline["duration_seconds"] == 9.5


def test_missing_resolver_label_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_states(monkeypatch, [])

    with pytest.raises(RuntimeError, match="omitted diarization label"):
        build_speaker_timeline(_session([_turn(0, 0.0, 1.0)]), uuid.uuid4())
