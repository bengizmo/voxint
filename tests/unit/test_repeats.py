"""Synthetic R1-R9 goldens for coarse, word-level and mixed turns."""

from dataclasses import replace

import pytest

from voxint.adjudication.turns import PieceRule, SpeakerTurn, TextMapping, TurnPiece, join_pieces
from voxint.export import to_markdown_turns
from voxint.export.fillers import drop_fillers_with_seams
from voxint.export.reading import layout_turns
from voxint.export.repeats import FUNCTION_WORDS, ONE_WORD_ELIGIBLE, ONE_WORD_EXCLUDED, drop_repeats


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


def words(text: str) -> SpeakerTurn:
    return turn(*(piece((" " if i else "") + word, i) for i, word in enumerate(text.split())))


def cleaned(text: str) -> str:
    return join_pieces(drop_repeats([turn(piece(text, coarse=True))])[0].pieces)


def rendered(turns: list[SpeakerTurn]) -> str:
    return to_markdown_turns(layout_turns(turns), header="T", timestamps=True)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("go to the the store", "go to the store"),
        ("we were we were going", "we were going"),
        ("The the cat sat.", "The cat sat."),
        ("Hi. The the cat", "Hi. The cat"),
        ("Hi!) The the cat", "Hi!) The cat"),
        ("Hi?] The the cat", "Hi?] The cat"),
        ('"Hi!" The the cat', '"Hi!" The cat'),
        ("we we were were going", "we were going"),
        ("It\u2019s it's fine", "It\u2019s fine"),
        ("I'm I'm here", "I'm here"),
        ("don't don't", "don't"),
        ("the the", "the"),
        ("the the \t", "the"),
        ("I i think", "I think"),
        ("THE THE cat", "THE cat"),
        ("We were we were going", "We were going"),
        ("we Were we Were going", "we Were going"),
        ("it's it\u2019s fine", "it's fine"),
        ("Now it's it\u2019s fine", "Now it's fine"),
        ("I'm I\u2019m here", "I'm here"),
        ("go we're in we\u2019re in town", "go we're in town"),
        ("that is that is fine", "that is fine"),
        ("the the , next", "the , next"),
        ("\u2018use label\u2019 the the cat", "\u2018use label\u2019 the cat"),
        ("\u2018use label\u2019, the the cat", "\u2018use label\u2019, the cat"),
        ("” the the cat", "” the cat"),
        ("\u2019 the the cat", "\u2019 the cat"),
        ("'label' the the cat", "'label' the cat"),
        ("we we were were we we", "we were we"),
        ("  go\tto the\nthe\u2003store  now\n", "  go\tto the store  now\n"),
    ],
)
def test_removes(text: str, expected: str) -> None:
    assert cleaned(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "No, no, no",
        "We were, we were going",
        "the. The",
        "the-the",
        "the the.",
        "the the2",
        "(the the)",
        "the the,",
        "the/the",
        "the_the",
        "the the—",
        "I I I think",
        "no no no no",
        "we were we were we were",
        "we were we were we",
        "were we were we were",
        "that that that that",
        "we were we were were",
        "we we were we were",
        'he said "the the" twice',
        "he said “use the the label”",
        "he said 'use the the label'",
        'the "the" store',
        'he said "the the',
        "he said \u2018use the the label",
        "he said 'use the the label",
        "going going gone",
        "she had had enough",
        "I know that that is true",
        "what it was was a joke",
        "I gave her her keys",
        "the will will be read",
        "no no",
        "so so",
        "you know you know",
        "we went we went",
        "in May may be",
        "US us",
        "THE the cat",
        "the The cat",
        "go The the cat",
        "We Were we were going",
        "we we're we we we",
        "the um the cat",
        "é é",
        "the the'",
        "the 'the",
        "the thé",
    ],
)
def test_declines_reuse_turn(text: str) -> None:
    original = turn(piece(text, coarse=True))
    result = drop_repeats([original])
    assert result[0] is original
    assert join_pieces(result[0].pieces) == text
    assert rendered(result) == rendered([original])


@pytest.mark.parametrize(
    "texts, expected",
    [
        (("we ", "we ", "go"), "we go"),
        ((" we", " we", " go"), " we go"),
        (("we", " we", " go"), "we go"),
        (("we  we go",), "we go"),
        (("we\twe go",), "we go"),
        (("we\nwe go",), "we go"),
        (("we", " ", "we", "\t", "go"), "we go"),
        (("w", "e we", " go"), "we go"),
        (("go  to the", " the", " store\t now"), "go  to the store\t now"),
    ],
)
def test_separator_shapes(texts: tuple[str, ...], expected: str) -> None:
    original = turn(*(piece(text, coarse=True) for text in texts))
    result = drop_repeats([original])
    assert join_pieces(result[0].pieces) == expected
    assert to_markdown_turns(layout_turns(result), header="T", timestamps=False) == (
        f"# T\n\n**Alex:** {expected.strip()}\n"
    )


def test_mixed_grain_metadata_and_piece_identity() -> None:
    original = turn(
        piece("go  to", coarse=True, boundary=True),
        piece(" the", 1),
        piece(" the", 2),
        piece(" store", 3),
        piece(" now", 4),
    )
    result = drop_repeats([original])[0]
    assert result is not original
    assert result.identity_key == original.identity_key
    assert result.speaker == original.speaker
    assert result.pieces == original.pieces[:2] + original.pieces[3:]
    assert all(
        new is old
        for new, old in zip(result.pieces, original.pieces[:2] + original.pieces[3:], strict=True)
    )
    assert original.pieces[2].text == " the"


def test_changed_pieces_replaced_and_untouched_empty_pieces_preserved() -> None:
    original = turn(piece("we "), piece("we ", 1), piece("go", 2), piece("", 3))
    result = drop_repeats([original])[0]
    assert result.pieces == (
        replace(original.pieces[0], text="we"),
        replace(original.pieces[2], text=" go"),
        original.pieces[3],
    )
    assert result.pieces[0] is not original.pieces[0]
    assert result.pieces[1] is not original.pieces[2]
    assert result.pieces[2] is original.pieces[3]


@pytest.mark.parametrize("separator", ["", " "])
def test_segment_boundary_blocks(separator: str) -> None:
    original = turn(piece("we"), piece(separator + "we", 1, boundary=True), piece(" go", 2))
    assert drop_repeats([original])[0] is original


def test_empty_segment_boundary_inside_token_blocks() -> None:
    original = turn(piece("we w"), piece("", boundary=True), piece("e go"))
    assert drop_repeats([original])[0] is original


def test_synthetic_separators_are_never_emitted_into_piece_text() -> None:
    first = piece("hello", coarse=True)
    middle = piece("we we go", 1, coarse=True, boundary=True)
    last = piece("next", 2, coarse=True, boundary=True)
    result = drop_repeats([turn(first, middle, last)])[0]
    assert result.pieces == (first, replace(middle, text="we go"), last)
    assert result.pieces[0] is first
    assert result.pieces[2] is last
    assert join_pieces(result.pieces) == "hello we go next"


def test_removed_separator_before_new_segment_is_owned_by_next_token() -> None:
    original = turn(piece("we we", coarse=True), piece("go", coarse=True, boundary=True))
    result = drop_repeats([original])[0]
    assert [p.text for p in result.pieces] == ["we", " go"]
    assert join_pieces(result.pieces) == "we go"


@pytest.mark.parametrize("removed_start", [3.0, 1.2])
def test_pause_guard_uses_surviving_neighbors(removed_start: float) -> None:
    original = turn(
        piece("we"),
        replace(piece(" we"), start_seconds=removed_start, end_seconds=removed_start + 0.5),
        piece(" go", 4),
    )
    result = drop_repeats([original])
    assert result[0] is original
    assert rendered(result) == rendered([original])


@pytest.mark.parametrize("following_start", [1.6, 3.999])
def test_short_surviving_pause_removes(following_start: float) -> None:
    original = turn(
        piece("we"),
        replace(piece(" we"), start_seconds=1.2, end_seconds=1.5),
        replace(piece(" go"), start_seconds=following_start, end_seconds=5.0),
    )
    result = drop_repeats([original])
    assert join_pieces(result[0].pieces) == "we go"


@pytest.mark.parametrize("untimed_index", [0, 2])
def test_pause_guard_requires_both_survivors_timed(untimed_index: int) -> None:
    pieces = [piece("we"), piece(" we", 3), piece(" go", 10)]
    pieces[untimed_index] = replace(pieces[untimed_index], timed=False)
    assert join_pieces(drop_repeats([turn(*pieces)])[0].pieces) == "we go"


def test_word_pair_uses_end_of_kept_pair_for_pause() -> None:
    original = words("we were we were going")
    assert join_pieces(drop_repeats([original])[0].pieces) == "we were going"
    paused = replace(original, pieces=(*original.pieces[:-1], piece(" going", 5)))
    assert drop_repeats([paused])[0] is paused


def test_layout_preserves_first_timestamp_and_minute_marker() -> None:
    original = turn(piece("we", 58, boundary=True), piece(" we", 59), piece(" go", 60))
    result = drop_repeats([original])
    assert layout_turns(result)[0].start_seconds == layout_turns([original])[0].start_seconds == 58
    assert rendered(result) == rendered([original]).replace("we we", "we")
    assert "[00:01:00]" in rendered(result)


def test_noop_new_list_reuses_turns_and_never_crosses_turns() -> None:
    original = [
        turn(piece("the")),
        turn(piece("the cat")),
        turn(piece("the"), name="Sam"),
        turn(piece("")),
        turn(piece(" \t")),
    ]
    result = drop_repeats(original)
    assert result is not original
    assert all(new is old for new, old in zip(result, original, strict=True))
    assert drop_repeats([]) == []


def test_word_lists() -> None:
    contractions = frozenset(
        [
            "i'm",
            "we're",
            "you're",
            "they're",
            "he's",
            "she's",
            "it's",
            "that's",
            "i've",
            "we've",
            "you've",
            "they've",
            "i'll",
            "we'll",
            "you'll",
            "he'll",
            "she'll",
            "it'll",
            "they'll",
            "that'll",
            "i'd",
            "we'd",
            "you'd",
            "he'd",
            "she'd",
            "it'd",
            "they'd",
            "that'd",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "don't",
            "doesn't",
            "didn't",
            "haven't",
            "hasn't",
            "hadn't",
            "couldn't",
            "shouldn't",
            "wouldn't",
            "can't",
            "won't",
        ]
    )
    assert {word for word in FUNCTION_WORDS if "'" in word} == contractions
    assert isinstance(FUNCTION_WORDS, frozenset)
    assert (
        frozenset(["that", "had", "is", "was", "do", "did", "her", "no", "so"]) == ONE_WORD_EXCLUDED
    )
    assert ONE_WORD_ELIGIBLE == FUNCTION_WORDS - ONE_WORD_EXCLUDED
    assert not FUNCTION_WORDS.intersection(
        [
            "well",
            "like",
            "right",
            "just",
            "there",
            "yes",
            "yeah",
            "okay",
            "how",
            "where",
            "why",
            "can",
            "will",
        ]
    )
    for word in ONE_WORD_ELIGIBLE:
        assert cleaned(f"{word} {word} next") == f"{word} next"
    for word in ONE_WORD_EXCLUDED:
        assert cleaned(f"{word} {word} next") == f"{word} {word} next"


def test_declined_candidate_still_cancels_its_overlap() -> None:
    # The one-word pair "we We" fails the case rule, but it overlaps the
    # two-word repeat; applying the latter alone would leave "we We were".
    assert cleaned("we We were We were") == "we We were We were"


def test_seam_blocks_like_a_segment() -> None:
    joined = turn(piece("the", 0), piece(" the", 1), piece(" cat", 2))

    assert join_pieces(drop_repeats([joined])[0].pieces) == "the cat"
    assert drop_repeats([joined], [frozenset({1})])[0] is joined


def test_seams_must_match_turns() -> None:
    with pytest.raises(ValueError, match="one entry per turn"):
        drop_repeats([words("the the")], [])


def test_filler_only_interruption_is_not_crossed() -> None:
    alex_1 = turn(piece("the", 0))
    sam = turn(piece(" um", 1), name="Sam")
    alex_2 = turn(piece(" the", 2), piece(" cat", 3))
    merged, seams = zip(*drop_fillers_with_seams([alex_1, sam, alex_2]), strict=True)

    assert [join_pieces(t.pieces) for t in merged] == ["the the cat"]
    assert [join_pieces(t.pieces) for t in drop_repeats(merged, seams)] == ["the the cat"]


def test_fillers_then_repeats_inside_one_turn() -> None:
    merged, seams = zip(
        *drop_fillers_with_seams([turn(piece("the um the cat", coarse=True))]), strict=True
    )

    assert [join_pieces(t.pieces) for t in drop_repeats(merged, seams)] == ["the cat"]
