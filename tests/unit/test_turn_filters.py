"""Shared filter composition and word-token counts at each removal step."""

import pytest

from voxint.adjudication.turns import PieceRule, SpeakerTurn, TextMapping, TurnPiece, join_pieces
from voxint.export.fillers import drop_fillers_with_seams
from voxint.export.repeats import drop_repeats
from voxint.export.turn_filters import apply_turn_filters


def piece(text: str, index: int = 0, *, coarse: bool = False, boundary: bool = False) -> TurnPiece:
    return TurnPiece(
        text,
        index,
        index + 1,
        not coarse,
        boundary,
        PieceRule.NO_WORDS if coarse else PieceRule.WORD_LEVEL,
        TextMapping.COARSE if coarse else TextMapping.VERBATIM,
    )


def turn(*pieces: TurnPiece, name: str = "Alex") -> SpeakerTurn:
    return SpeakerTurn(("speaker", name), name, pieces)


def test_noop_preserves_turn_identity() -> None:
    original = (turn(piece("um the the cat")), turn(piece("Hello"), name="Sam"))
    result = apply_turn_filters(original, drop_fillers=False, drop_repeats=False)
    assert isinstance(result.turns, list)
    assert len(result.turns) == len(original)
    assert all(actual is expected for actual, expected in zip(result.turns, original, strict=True))
    assert result.fillers_removed == result.repeats_removed == 0


@pytest.mark.parametrize("text", ["um ...", "um , !"])
def test_punctuation_remnants_are_not_words(text: str) -> None:
    result = apply_turn_filters(
        [turn(piece(text, coarse=True))], drop_fillers=True, drop_repeats=False
    )
    assert result.turns == []
    assert result.fillers_removed == 1
    assert result.repeats_removed == 0


def test_f5_dropped_turn_and_seam_counts() -> None:
    original = [turn(piece("Hello")), turn(piece("um", 1), name="Sam"), turn(piece("there", 2))]
    result = apply_turn_filters(original, drop_fillers=True, drop_repeats=False)
    assert len(result.turns) == 1
    assert join_pieces(result.turns[0].pieces) == "Hello there"
    assert result.fillers_removed == 1
    assert result.repeats_removed == 0


@pytest.mark.parametrize(
    "fillers, repeats, expected, filler_count, repeat_count",
    [
        (False, False, "the um the cat and and dog", 0, 0),
        (True, False, "the the cat and and dog", 1, 0),
        (False, True, "the um the cat and dog", 0, 1),
        (True, True, "the cat and dog", 1, 2),
    ],
)
def test_each_step_counts_its_own_input(
    fillers: bool, repeats: bool, expected: str, filler_count: int, repeat_count: int
) -> None:
    original = [turn(piece("the um the cat and and dog", coarse=True))]
    result = apply_turn_filters(original, drop_fillers=fillers, drop_repeats=repeats)
    assert join_pieces(result.turns[0].pieces) == expected
    assert result.fillers_removed == filler_count
    assert result.repeats_removed == repeat_count


def test_repeats_do_not_cross_f5_seam() -> None:
    original = [turn(piece("the")), turn(piece("um", 1), name="Sam"), turn(piece("the cat", 2))]
    result = apply_turn_filters(original, drop_fillers=True, drop_repeats=True)
    assert len(result.turns) == 1
    assert join_pieces(result.turns[0].pieces) == "the the cat"
    assert result.fillers_removed == 1
    assert result.repeats_removed == 0


def test_counts_match_visible_words_with_segment_boundaries() -> None:
    original = turn(piece("um", boundary=True), piece("the the café 123 , !", 1, boundary=True))
    before = join_pieces(original.pieces)
    result = apply_turn_filters([original], drop_fillers=True, drop_repeats=True)
    after = join_pieces(result.turns[0].pieces)
    assert before == "um the the café 123 , !"
    assert after == "The café 123 , !"
    before_words = [token for token in before.split() if any(char.isalnum() for char in token)]
    after_words = [token for token in after.split() if any(char.isalnum() for char in token)]
    assert result.fillers_removed == 1
    assert result.repeats_removed == 1
    assert result.fillers_removed + result.repeats_removed == len(before_words) - len(after_words)


@pytest.mark.parametrize("fillers", [False, True])
@pytest.mark.parametrize("repeats", [False, True])
@pytest.mark.parametrize(
    "original",
    [
        [],
        [turn(piece(""))],
        [turn(piece("... , !"))],
        [turn(piece("um ..."))],
        [turn(piece("um , !"))],
        [turn(piece("the um the cat and and dog", coarse=True))],
        [turn(piece("um, café 123 — !", coarse=True))],
        [turn(piece("we")), turn(piece("um", 1), name="Sam"), turn(piece("we go", 2))],
        [turn(piece("um", boundary=True), piece("hello", 1, boundary=True))],
        [turn(piece("umbrella uh-huh erm..."))],
    ],
)
def test_nonnegative_counts_and_direct_composition(
    original: list[SpeakerTurn], fillers: bool, repeats: bool
) -> None:
    expected = list(original)
    seams: list[frozenset[int]] | None = None
    if fillers:
        cleaned = drop_fillers_with_seams(expected)
        expected = [item for item, _ in cleaned]
        seams = [boundaries for _, boundaries in cleaned]
    if repeats:
        expected = drop_repeats(expected, seams)
    result = apply_turn_filters(original, drop_fillers=fillers, drop_repeats=repeats)
    assert result.turns == expected
    assert result.fillers_removed >= 0
    assert result.repeats_removed >= 0
