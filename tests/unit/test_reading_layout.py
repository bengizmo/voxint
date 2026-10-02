"""Synthetic reading layout boundaries and monotonic minute markers."""

from dataclasses import replace

import pytest

from voxint.adjudication.turns import PieceRule, SpeakerTurn, TextMapping, TurnPiece
from voxint.export.reading import PARAGRAPH_MIN_WORDS, PARAGRAPH_PAUSE_SECONDS, layout_turns


def piece(text: str, start: float, end: float, *, timed: bool = True) -> TurnPiece:
    return TurnPiece(text, start, end, timed, True, PieceRule.SINGLE_IDENTITY, TextMapping.VERBATIM)


def turn(*pieces: TurnPiece, name: str = "Alex") -> SpeakerTurn:
    return SpeakerTurn(("label", name), name, pieces)


@pytest.mark.parametrize("start", [4, -1])
def test_pause_and_backwards_time(start: float) -> None:
    result = layout_turns([turn(piece("Hello", 0, 1), piece("there", start, start + 1))])
    assert [p.continuation for p in result] == [False, True]
    assert [p.start_seconds for p in result] == [0, start]


@pytest.mark.parametrize("ending", [".", '?"', "!\u201d", ".')]"])
@pytest.mark.parametrize("offset,expected", [(-1, 1), (0, 2), (1, 2)])
def test_sentence_threshold(offset: int, expected: int, ending: str) -> None:
    body = " ".join(["word"] * (PARAGRAPH_MIN_WORDS + offset)) + ending
    result = layout_turns([turn(piece(body, 0, 1), piece("Next", 1, 2))])
    assert len(result) == expected


def test_no_sentence_and_atomic_coarse_piece() -> None:
    body = "word " * 100 + "still speaking"
    result = layout_turns([turn(piece(body, 0, 200, timed=False), piece("Next", 200, 201))])
    assert len(result) == 1
    assert [r.marker_seconds for r in result[0].runs] == [None, 180]
    assert result[0].runs[0].text == body
    ended = piece(body + ".", 0, 200, timed=False)
    assert len(layout_turns([turn(ended, piece("Next", 200, 201))])) == 2


def test_markers_and_paragraph_starts() -> None:
    result = layout_turns([turn(
        piece("First", 59, 60), piece("second", 60, 61), piece("third", 61, 62),
        piece("Later", 120, 121), piece("backwards", 60, 61),
        piece("forward", 61, 180), piece("last", 180, 181),
    ), turn(piece("Sam speaks", 240, 241), name="Sam")])
    assert [p.continuation for p in result] == [False, True, True, False]
    assert [p.start_seconds for p in result] == [59, 120, 60, 240]
    assert [[r.marker_seconds for r in p.runs] for p in result] == [
        [None, 60], [None], [None, 180], [None],
    ]
    assert result[-1].speaker == "Sam"


def test_coarse_pause_and_empty_pieces() -> None:
    first = piece("First", 0, 1, timed=False)
    second = piece("Second", 4, 5, timed=False)
    result = layout_turns([turn(first, replace(first, text="  "), second)])
    assert len(result) == 2
    assert result[1].continuation
    assert layout_turns([turn(replace(first, text="  "))]) == []


def test_paragraph_start_consumes_minute() -> None:
    result = layout_turns([turn(
        piece("Early", 0, 1), piece("New paragraph", 120, 121), piece("same minute", 121, 122),
    )])
    assert len(result) == 2
    assert [r.marker_seconds for r in result[1].runs] == [None]
    assert result[1].runs[0].text == "New paragraph same minute"


def test_pause_below_threshold_does_not_break() -> None:
    just_under = 1 + PARAGRAPH_PAUSE_SECONDS - 0.001
    assert len(layout_turns([turn(piece("Early", 0, 1), piece("Later", just_under, 5))])) == 1


def test_thresholds_are_the_measured_values() -> None:
    # Pinned on purpose: these were set by measurement (see the module
    # docstring), so a change should be a visible, deliberate diff.
    assert (PARAGRAPH_PAUSE_SECONDS, PARAGRAPH_MIN_WORDS) == (3.0, 20)
