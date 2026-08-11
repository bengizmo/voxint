"""The segment-labeling rule: maximum temporal intersection, deterministic ties."""

from voxint.clients.base import DiarizationTurn
from voxint.pipeline.stages.diarize_embed import _dominant_label

TURNS = (
    DiarizationTurn(0.0, 4.0, "SPEAKER_00"),
    DiarizationTurn(4.0, 9.0, "SPEAKER_01"),
)


def test_max_overlap_wins() -> None:
    assert _dominant_label(3.0, 6.0, TURNS) == "SPEAKER_01"  # 1s vs 2s
    assert _dominant_label(0.0, 4.0, TURNS) == "SPEAKER_00"


def test_no_intersection_is_none() -> None:
    assert _dominant_label(10.0, 12.0, TURNS) is None
    assert _dominant_label(0.0, 1.0, ()) is None


def test_boundary_touch_is_not_overlap() -> None:
    # A segment ending exactly where a turn starts shares no time with it.
    assert _dominant_label(9.0, 10.0, TURNS) is None


def test_exact_tie_goes_to_earliest_turn() -> None:
    assert _dominant_label(3.0, 5.0, TURNS) == "SPEAKER_00"  # 1s each


def test_repeated_labels_accumulate_nothing() -> None:
    # Two half-second turns of SPEAKER_00 do NOT outweigh one full second of
    # SPEAKER_01 — the rule is per-turn intersection, not per-label sum.
    turns = (
        DiarizationTurn(0.0, 0.5, "SPEAKER_00"),
        DiarizationTurn(1.0, 2.0, "SPEAKER_01"),
        DiarizationTurn(2.5, 3.0, "SPEAKER_00"),
    )
    assert _dominant_label(0.0, 3.0, turns) == "SPEAKER_01"
