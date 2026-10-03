"""Synthetic F1-F6 cases for word-level, coarse and mixed turns."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from voxint.adjudication.turns import PieceRule, SpeakerTurn, TextMapping, TurnPiece, join_pieces
from voxint.export import to_markdown_turns
from voxint.export.fillers import drop_fillers
from voxint.export.reading import layout_turns


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


def cleaned(text: str) -> str:
    return "".join(join_pieces(t.pieces) for t in drop_fillers([turn(piece(text, coarse=True))]))


@pytest.mark.parametrize("word", ["um", "uh", "umm", "uhh", "uhm", "erm"])
@pytest.mark.parametrize("case", [str.lower, str.upper, str.capitalize])
def test_f1_words(word: str, case: Callable[[str], str]) -> None:
    # Case variants are deliberately the same fixtures for each of the six words.
    assert cleaned(f"{case(word)}, so we start") == "So we start"


@pytest.mark.parametrize(
    "text",
    [
        "umbrella",
        "hum",
        "uh-huh",
        "um's",
        "erm...",
        "um..",
        "um?!",
        "um,x",
        "xum",
        "umx",
        "1um",
        "um1",
        "-um",
        "um-",
        "'um's",
        "x'um",
        "um\u2019s",
        "mm-hmm",
        "hmm",
        "like",
        "you know",
        "um_word",
        "um!x",
        "um:word",
    ],
)
def test_f1_not_fillers(text: str) -> None:
    assert cleaned(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Yes um.", "Yes."),
        ("Yes uh?", "Yes?"),
        ("Yes erm!", "Yes!"),
        ("and then, um.", "and then."),
        ("then; uh?", "then?"),
        ("then: uh!", "then!"),
        ("Right? Um.", "Right?"),
        ("Right. Um!", "Right."),
        ("Right! Um?", "Right!"),
        ("Um. So we start", "So we start"),
        ("Uh? So we start", "So we start"),
        ("Erm! So we start", "So we start"),
        ("I think, um, that", "I think, that"),
        ("and um; then", "and then"),
        ("and uh: then", "and then"),
        ('"Yes" um.', '"Yes".'),
        ('"Right?" Um.', '"Right?".'),
        ("item 1 um.", "item 1."),
        ("value / um.", "value /"),
        ("um, so", "So"),
        ("and um, so", "and so"),
        ("um, So", "So"),
        ("Done. um, so", "Done. So"),
        ('Done!" um, so', 'Done!" So'),
        ("um, 3 things", "3 things"),
        ('um, "so we start"', '"So we start"'),
        ("um, [so we start]", "[So we start]"),
        ("um, #word", "#word"),
        ('"um, so', '"So'),
        ("(uh, so", "(So"),
        ("[erm, so", "[So"),
        ("\u201cuh, so", "\u201cSo"),
        ("\u2018um, so", "\u2018So"),
        ('"um," so', '"" So'),
        ("um, uh, so", "So"),
        ("  um,   so", "So"),
        ("and   um,   so", "and so"),
        ("and um  ", "and"),
    ],
)
def test_f2_f3_f4_coarse(text: str, expected: str) -> None:
    assert cleaned(text) == expected


@pytest.mark.parametrize(
    "texts, boundaries, expected",
    [
        (("Yes", " um.", " next"), (False, False, False), "Yes. Next"),
        (("value /", " um.", " next"), (False, False, False), "value / next"),
        (("can't", " um,", " go"), (False, False, False), "can't go"),
        (("hello ", "um, ", "world"), (False, False, False), "hello world"),
        (("hello", "um,", "world"), (True, True, True), "hello world"),
        (("um,", "world"), (True, True), "World"),
        ((" um,", " so"), (False, False), "So"),
        (("Right?", " Um.", " so"), (False, False, False), "Right? So"),
        (("Done.", " um,", ' "so"'), (False, False, False), 'Done. "So"'),
        (("and", " um", " uh,", " so"), (True, False, False, False), "and so"),
    ],
)
def test_f3_word_units(texts: tuple[str, ...], boundaries: tuple[bool, ...], expected: str) -> None:
    original = turn(
        *(piece(t, i, boundary=b) for i, (t, b) in enumerate(zip(texts, boundaries, strict=True)))
    )
    result = drop_fillers([original])
    assert join_pieces(result[0].pieces) == expected
    for survivor in result[0].pieces:
        assert (
            replace(survivor, text=original.pieces[int(survivor.start_seconds)].text)
            == (original.pieces[int(survivor.start_seconds)])
        )
    assert to_markdown_turns(layout_turns(result), header="Synthetic", timestamps=False) == (
        f"# Synthetic\n\n**Alex:** {expected}\n"
    )


def test_f5_empty_turns_merge_by_identity_in_order() -> None:
    first, last = piece("Hello"), piece(" there", 2)
    result = drop_fillers(
        [
            turn(first),
            turn(piece(" um, uh.", 1), name="Sam"),
            turn(last),
            turn(piece(" erm", 3), name="Jo"),
        ]
    )
    assert result == [turn(first, last)]
    assert drop_fillers([turn(piece(" \n")), turn(piece("")), turn(piece("um"))]) == []


def test_f5_merged_turns_keep_a_separator_and_are_cleaned_as_one() -> None:
    # The dropped middle turn owned the only whitespace at the seam.
    glued = drop_fillers(
        [turn(piece("yes")), turn(piece("um ", 1), name="Sam"), turn(piece("right", 2))]
    )
    assert join_pieces(glued[0].pieces) == "yes right"
    assert glued[0].pieces[1] == replace(piece("right", 2), text=" right")
    # A filler at the seam is judged against the merged text, not a turn start.
    mid = drop_fillers([
        turn(piece(" yes")), turn(piece(" uh,", 1), name="Sam"),
        turn(piece(" um", 2), piece(" right", 3)),
    ])
    assert [join_pieces(t.pieces) for t in mid] == [" yes right"]
    start = drop_fillers([
        turn(piece("Yes.")), turn(piece(" uh", 1), name="Sam"),
        turn(piece(" um", 2), piece(" right", 3)),
    ])
    assert [join_pieces(t.pieces) for t in start] == ["Yes. Right"]
    assert start[0].pieces == (piece("Yes."), replace(piece(" right", 3), text=" Right"))


def test_f5_three_speakers_and_same_display_name() -> None:
    alex = turn(piece("Hello"))
    sam = turn(piece(" there", 2), name="Sam")
    jo = turn(piece(" Next", 3), name="Jo")
    assert drop_fillers([alex, turn(piece("um"), name="Jo"), sam, jo]) == [alex, sam, jo]
    same_name = replace(sam, speaker="Alex")
    assert drop_fillers([alex, same_name]) == [alex, same_name]


def test_f6_no_filler_bytes_and_metadata_unchanged() -> None:
    original = [
        turn(
            piece("  can't", 0),
            piece(" re-enter!  ", 1),
            piece("Coarse  text", 2, coarse=True, boundary=True),
        )
    ]
    result = drop_fillers(original)
    assert result == original
    assert join_pieces(result[0].pieces) == join_pieces(original[0].pieces)
    assert to_markdown_turns(layout_turns(result), header="Synthetic") == (
        to_markdown_turns(layout_turns(original), header="Synthetic")
    )
    assert drop_fillers([]) == []


def test_mixed_grain_and_punctuation_move_preserve_metadata() -> None:
    original = turn(
        piece("Yes,", 0),
        piece(" um.", 1),
        piece(" um, so we start", 2, coarse=True, boundary=True),
        piece(" now", 3),
    )
    result = drop_fillers([original])[0]
    assert join_pieces(result.pieces) == "Yes. So we start now"
    assert result.pieces == (
        replace(original.pieces[0], text="Yes."),
        replace(original.pieces[2], text=" So we start"),
        original.pieces[3],
    )
