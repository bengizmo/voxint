"""Synthetic F1-F6 cases for word-level, coarse and mixed turns."""

import re
from collections.abc import Callable
from dataclasses import replace

import pytest

from tests.unit.filter_inputs import seeded_filler_soups
from voxint.adjudication.turns import PieceRule, SpeakerTurn, TextMapping, TurnPiece, join_pieces
from voxint.export import to_markdown_turns
from voxint.export.filler_lists import DEFAULT_FILLER_LIST, TIER_1, effective_filler_list
from voxint.export.fillers import (
    _clean_indexed,
    _word_pattern,
    drop_fillers,
    drop_fillers_with_seams,
)
from voxint.export.reading import layout_turns
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


def cleaned(text: str) -> str:
    return "".join(
        join_pieces(t.pieces)
        for t in drop_fillers([turn(piece(text, coarse=True))], fillers=DEFAULT_FILLER_LIST)
    )


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
        '"um" so',
        '"um," so',
        "\u201cum,\u201d so",
        "'uh' so",
        "the word um\u201d",
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
        ('"Right?" Um.', '"Right?"'),
        ("(yes) um.", "(yes)."),
        ("[yes] um.", "[yes]."),
        ("Done.) um, next", "Done.) Next"),
        ("Done.] um, next", "Done.] Next"),
        ("(Done.) um, next", "(Done.) Next"),
        ('" um. so', '" So'),
        ('Done. " um, so', 'Done. " So'),
        ('"and then," um.', '"and then."'),
        ('"Right?" Um!', '"Right?"'),
        ("um, 3", "3"),
        ("um, \u00df, um.", "\u00df."),
        ('um, "\u00df," um.', '"\u00df."'),
        ("um, \ufb02ow", "\ufb02ow"),
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
        (("and", ' "um,', " so"), (False, False, False), 'and "so'),
        (("and", '"um,', " so"), (True, True, False), 'and "so'),
        (("and", " um,", ' "so'), (True, False, False), 'and "so'),
        (("and", " um", " uh,", " so"), (True, False, False, False), "and so"),
    ],
)
def test_f3_word_units(texts: tuple[str, ...], boundaries: tuple[bool, ...], expected: str) -> None:
    original = turn(
        *(piece(t, i, boundary=b) for i, (t, b) in enumerate(zip(texts, boundaries, strict=True)))
    )
    result = drop_fillers([original], fillers=DEFAULT_FILLER_LIST)
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
        ],
        fillers=DEFAULT_FILLER_LIST,
    )
    assert result == [turn(first, last)]
    assert (
        drop_fillers(
            [turn(piece(" \n")), turn(piece("")), turn(piece("um"))], fillers=DEFAULT_FILLER_LIST
        )
        == []
    )
    # Punctuation around a removed filler does not keep a turn alive.
    for text in ('"um,', "(uh", "um, ...", "Um. Uh?", "\u201cerm, uh!"):
        assert drop_fillers([turn(piece(text, coarse=True))], fillers=DEFAULT_FILLER_LIST) == [], (
            text
        )


def test_f5_merged_turns_keep_a_separator_and_are_cleaned_as_one() -> None:
    # The dropped middle turn owned the only whitespace at the seam.
    glued = drop_fillers(
        [turn(piece("yes")), turn(piece("um ", 1), name="Sam"), turn(piece("right", 2))],
        fillers=DEFAULT_FILLER_LIST,
    )
    assert join_pieces(glued[0].pieces) == "yes right"
    assert glued[0].pieces[1] == replace(piece("right", 2), text=" right")
    # Whatever whitespace either side carries, the seam holds one space, on the right.
    for left, right in (("Hello ", " world"), ("Hello ", "world"), ("Hello  ", "\n\nworld")):
        merged = drop_fillers(
            [turn(piece(left)), turn(piece(" um,", 1), name="Sam"), turn(piece(right, 2))],
            fillers=DEFAULT_FILLER_LIST,
        )
        assert merged[0].pieces == (piece("Hello"), replace(piece(right, 2), text=" world"))
    # A filler at the seam is judged against the merged text, not a turn start.
    mid = drop_fillers(
        [
            turn(piece(" yes")),
            turn(piece(" uh,", 1), name="Sam"),
            turn(piece(" um", 2), piece(" right", 3)),
        ],
        fillers=DEFAULT_FILLER_LIST,
    )
    assert [join_pieces(t.pieces) for t in mid] == [" yes right"]
    start = drop_fillers(
        [
            turn(piece("Yes.")),
            turn(piece(" uh", 1), name="Sam"),
            turn(piece(" um", 2), piece(" right", 3)),
        ],
        fillers=DEFAULT_FILLER_LIST,
    )
    assert [join_pieces(t.pieces) for t in start] == ["Yes. Right"]
    assert start[0].pieces == (piece("Yes."), replace(piece(" right", 3), text=" Right"))


def test_f5_three_speakers_and_same_display_name() -> None:
    alex = turn(piece("Hello"))
    sam = turn(piece(" there", 2), name="Sam")
    jo = turn(piece(" Next", 3), name="Jo")
    assert drop_fillers(
        [alex, turn(piece("um"), name="Jo"), sam, jo], fillers=DEFAULT_FILLER_LIST
    ) == [alex, sam, jo]
    same_name = replace(sam, speaker="Alex")
    assert drop_fillers([alex, same_name], fillers=DEFAULT_FILLER_LIST) == [alex, same_name]


def test_f6_no_filler_bytes_and_metadata_unchanged() -> None:
    original = [
        turn(
            piece("  can't", 0),
            piece(" re-enter!  ", 1),
            piece("Coarse  text", 2, coarse=True, boundary=True),
        )
    ]
    result = drop_fillers(original, fillers=DEFAULT_FILLER_LIST)
    assert result == original
    assert join_pieces(result[0].pieces) == join_pieces(original[0].pieces)
    assert to_markdown_turns(layout_turns(result), header="Synthetic") == (
        to_markdown_turns(layout_turns(original), header="Synthetic")
    )
    assert drop_fillers([], fillers=DEFAULT_FILLER_LIST) == []


def test_mixed_grain_and_punctuation_move_preserve_metadata() -> None:
    original = turn(
        piece("Yes,", 0),
        piece(" um.", 1),
        piece(" um, so we start", 2, coarse=True, boundary=True),
        piece(" now", 3),
    )
    result = drop_fillers([original], fillers=DEFAULT_FILLER_LIST)[0]
    assert join_pieces(result.pieces) == "Yes. So we start now"
    assert result.pieces == (
        replace(original.pieces[0], text="Yes."),
        replace(original.pieces[2], text=" So we start"),
        original.pieces[3],
    )


def test_seams_mark_where_a_dropped_turn_joined_two_turns() -> None:
    result = drop_fillers_with_seams(
        [turn(piece("Hello", 0)), turn(piece(" um", 1), name="Sam"), turn(piece(" there", 2))],
        fillers=DEFAULT_FILLER_LIST,
    )

    assert [(join_pieces(t.pieces), seams) for t, seams in result] == [
        ("Hello there", frozenset({1}))
    ]


def test_seam_moves_to_first_survivor_when_the_joined_head_is_a_filler() -> None:
    result = drop_fillers_with_seams(
        [
            turn(piece("so", 0), piece(" yes", 1)),
            turn(piece(" um", 2), name="Sam"),
            turn(piece(" um", 3), piece(" right", 4)),
        ],
        fillers=DEFAULT_FILLER_LIST,
    )

    assert [(join_pieces(t.pieces), seams) for t, seams in result] == [
        ("so yes right", frozenset({2}))
    ]


def test_no_seams_without_a_merge() -> None:
    result = drop_fillers_with_seams(
        [turn(piece("um so")), turn(piece(" yes", 1), name="Sam")], fillers=DEFAULT_FILLER_LIST
    )

    assert [seams for _, seams in result] == [frozenset(), frozenset()]
    assert [t for t, _ in result] == drop_fillers(
        [turn(piece("um so")), turn(piece(" yes", 1), name="Sam")], fillers=DEFAULT_FILLER_LIST
    )


# The original production literal is retained only as a regression oracle.
_OLD_FILLER = re.compile(
    r"""(?<!\S)(["'(\[\u201c\u2018]*)(?:umm|uhh|uhm|erm|um|uh)"""
    r"""(?:(?P<mark>[,.?!;:])(?=$|\s)|(?=$|\s))\s*""",
    re.IGNORECASE,
)


def test_default_pattern_equality_and_seeded_soups() -> None:
    generated = _word_pattern(DEFAULT_FILLER_LIST.words)
    assert generated.pattern == _OLD_FILLER.pattern
    assert generated.flags == _OLD_FILLER.flags
    for pieces in seeded_filler_soups():
        body = join_pieces(pieces)
        assert [(m.span(), m.groups()) for m in generated.finditer(body)] == [
            (m.span(), m.groups()) for m in _OLD_FILLER.finditer(body)
        ]


@pytest.mark.parametrize(
    "text, expected, count",
    [
        ("It was, you know, big. You know the rules.", "It was, big. You know the rules.", 2),
        ("Do you know, sir?", "Do you know, sir?", 0),
        ("And you know, we left.", "And you know, we left.", 0),
        (
            "He lied, you know? Will you come? I guess.",
            "He lied, you know? Will you come? I guess.",
            0,
        ),
        ("I guess.", "I guess.", 0),
        ("Fine, you know what I mean, we left.", "Fine, we left.", 5),
        ('It was, you know" big.', 'It was, you know" big.', 0),
        ('It was, you know," big.', 'It was, you know," big.', 0),
        ("It was... you know, big.", "It was... you know, big.", 0),
        ("It was\u2026 you know, big.", "It was\u2026 you know, big.", 0),
        ("It was - you know, big.", "It was - you know, big.", 0),
        ("It was \u2014 you know, big.", "It was \u2014 you know, big.", 0),
        ("It was \u2013 you know, big.", "It was \u2013 you know, big.", 0),
        ("It was, you know,, big.", "It was, you know,, big.", 0),
        ("It was,, you know, big.", "It was,, big.", 2),
        ("It was, you know, I mean, big.", "It was, big.", 4),
        ("And you know, I mean, big.", "And you know, big.", 2),
        ("It was, you\t\nknow, big.", "It was, big.", 2),
        ("It was, you know.", "It was.", 2),
        ("You know, it works.", "It works.", 2),
        ("It was, You Know, big.", "It was, big.", 2),
        ("It was, you know", "It was,", 2),
        ("You know", "You know", 0),
        ("It was, you know; big.", "It was, big.", 2),
        ("It was, you know: big.", "It was, big.", 2),
    ],
)
def test_phrase_boundaries_and_counts(text: str, expected: str, count: int) -> None:
    fillers = effective_filler_list(["you know", "I mean", "I guess", "you know what I mean"])
    result = apply_turn_filters([turn(piece(text))], fillers=fillers, drop_repeats=False)
    assert [join_pieces(t.pieces) for t in result.turns] == [expected]
    assert result.fillers_removed == count


@pytest.mark.parametrize("boundary", [False, True])
def test_phrase_piece_ownership_and_pass_composition(boundary: bool) -> None:
    original = turn(
        piece("It was,", 0),
        piece(" you", 1),
        piece("know," if boundary else " know,", 2, boundary=boundary),
        piece(" um,", 3),
        piece(" big.", 4, coarse=True, boundary=True),
    )
    fillers = effective_filler_list(["you know"])
    indexed = _clean_indexed(original.pieces, fillers=fillers)
    assert indexed == ((0, original.pieces[0]), (4, original.pieces[4]))
    result = apply_turn_filters([original], fillers=fillers, drop_repeats=False)
    assert result.turns == [turn(original.pieces[0], original.pieces[4])]
    assert join_pieces(result.turns[0].pieces) == "It was, big."
    assert result.fillers_removed == 3


def test_phrase_never_crosses_f5_source_seam() -> None:
    original = [turn(piece("So, you")), turn(piece("um"), name="Sam"), turn(piece("know, yes."))]
    result = apply_turn_filters(
        original,
        fillers=effective_filler_list(["you know"]),
        drop_repeats=False,
    )
    assert [join_pieces(t.pieces) for t in result.turns] == ["So, you know, yes."]
    assert result.fillers_removed == 1


def test_phrase_removal_preserves_repeat_seam_and_counts() -> None:
    original = [
        turn(piece("the")),
        turn(piece("you know,"), name="Sam"),
        turn(piece("you know,", 2), piece(" the cat", 3)),
    ]
    result = apply_turn_filters(
        original,
        fillers=effective_filler_list(["you know"]),
        drop_repeats=True,
    )
    # The merged left context conservatively keeps the second phrase.
    assert [join_pieces(t.pieces) for t in result.turns] == ["the you know, the cat"]
    assert result.fillers_removed == 2
    assert result.repeats_removed == 0
    original = [
        turn(piece("the")),
        turn(piece("you know,"), name="Sam"),
        turn(piece("the cat", 3)),
    ]
    result = apply_turn_filters(
        original,
        fillers=effective_filler_list(["you know"]),
        drop_repeats=True,
    )
    assert [join_pieces(t.pieces) for t in result.turns] == ["the the cat"]
    assert result.fillers_removed == 2
    assert result.repeats_removed == 0


def test_keep_um_and_empty_list() -> None:
    result = apply_turn_filters(
        [turn(piece("um, uh, yes"))],
        fillers=effective_filler_list(keep=["um"]),
        drop_repeats=False,
    )
    assert join_pieces(result.turns[0].pieces) == "um, yes"
    assert result.fillers_removed == 1
    original = [turn(piece("um, uh, umm uhh uhm erm. you know,"))]
    result = apply_turn_filters(
        original, fillers=effective_filler_list(keep=TIER_1), drop_repeats=False
    )
    assert result.turns == original
    assert result.fillers_removed == 0


@pytest.mark.parametrize("text", ['"um,', "(uh", "um, ...", "Um. Uh?", "\u201cerm, uh!"])
def test_single_word_filler_only_turns_drop_before_backstop(text: str) -> None:
    assert _clean_indexed((piece(text),), fillers=DEFAULT_FILLER_LIST) == ()


def test_phrases_only_skip_empty_word_alternation() -> None:
    result = apply_turn_filters(
        [turn(piece("It was, you know, big."))],
        fillers=effective_filler_list(["you know"], TIER_1),
        drop_repeats=False,
    )
    assert join_pieces(result.turns[0].pieces) == "It was, big."
    assert result.fillers_removed == 2


def test_kept_phrase_takes_pending_capital() -> None:
    result = apply_turn_filters(
        [turn(piece("I mean, you know? Fine."))],
        fillers=effective_filler_list(["you know", "I mean"]),
        drop_repeats=False,
    )
    assert join_pieces(result.turns[0].pieces) == "You know? Fine."


def test_seam_space_does_not_block_a_phrase_inside_one_turn() -> None:
    original = [turn(piece("Fine, you know,")), turn(piece("um"), name="Sam"), turn(piece("yes."))]
    result = apply_turn_filters(
        original, fillers=effective_filler_list(["you know"]), drop_repeats=False
    )
    assert [join_pieces(t.pieces) for t in result.turns] == ["Fine, yes."]
    assert result.fillers_removed == 3


def test_punctuation_only_turns_without_fillers_survive() -> None:
    original = [turn(piece("...")), turn(piece("—"), name="Sam")]
    assert drop_fillers(original, fillers=DEFAULT_FILLER_LIST) == original


def test_phrase_alternation_is_longest_first_for_any_list() -> None:
    fillers = replace(DEFAULT_FILLER_LIST, phrases=("you know", "you know what"))
    result = drop_fillers([turn(piece("Hello, you know what, yes."))], fillers=fillers)
    assert [join_pieces(t.pieces) for t in result] == ["Hello, yes."]


@pytest.mark.parametrize(
    "entry, text",
    [("Weiß", "Well, wEIß, yes."), ("İ", "Well, İ, yes."), ("İ mean", "Well, İ mean, yes.")],
)
def test_entry_with_expanding_case_mapping_matches_its_spelling(entry: str, text: str) -> None:
    result = drop_fillers([turn(piece(text))], fillers=effective_filler_list([entry]))
    assert [join_pieces(t.pieces) for t in result] == ["Well, yes."]
