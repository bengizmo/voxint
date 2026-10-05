"""Per-unit keep/omit overlay on the real filler, seam and repeat pipeline."""

import itertools
from dataclasses import asdict, replace
from typing import Any, Literal
from unittest.mock import patch

import pytest

from tests.unit.filter_inputs import seeded_filler_soups
from tests.unit.test_filler_trace import _A, _B, _C, _piece, _source, _turn
from tests.unit.test_filter_characterization import _DATA, _OPTIONS, _decode
from tests.unit.test_turns import STATES, emission
from voxint.adjudication.splits import derive_children
from voxint.adjudication.transcript import TranscriptText
from voxint.adjudication.turns import (
    SpeakerTurn,
    WordAnchor,
    join_pieces,
    project_turns,
)
from voxint.adjudication.word_marks import EffectiveMarks
from voxint.export.filler_lists import DEFAULT_FILLER_LIST, FillerList, effective_filler_list
from voxint.export.fillers import (
    Protected,
    Removal,
    UnplaceableWordMarkError,
    _clean_tracked,
    _flatten,
    drop_fillers_with_trace,
)
from voxint.export.turn_filters import FilteredTurns, apply_turn_filters


def _filter(
    turns: list[SpeakerTurn],
    marks: EffectiveMarks,
    *,
    fillers: FillerList = DEFAULT_FILLER_LIST,
    repeats: bool = False,
) -> FilteredTurns:
    result = apply_turn_filters(turns, fillers=fillers, drop_repeats=repeats, marks=marks)
    # On these anchored fixtures, filler causes and the legacy delta agree.
    assert result.fillers_removed == sum(
        any(c.isalnum() for c in token)
        for entry in result.trace
        if entry.cause in ("filler", "phrase")
        for token in entry.text.split()
    )
    assert result.fillers_removed == len(
        {
            source
            for entry in result.trace
            if entry.cause in ("filler", "phrase")
            for source in entry.sources
        }
    )
    return result


def _texts(result: FilteredTurns) -> list[str]:
    return [join_pieces(turn.pieces) for turn in result.turns]


def test_keep_defeats_list_without_rewriting_input() -> None:
    original = _turn(_piece("Um, we start now."))
    result = _filter([original], {_source(_A, 0): "keep"})
    assert _texts(result) == ["Um, we start now."]
    assert result.trace == (Protected("Um", (_source(_A, 0),)),)
    assert (result.fillers_removed, result.omitted, result.kept) == (0, 0, 1)
    assert result.turns[0].pieces[0] is original.pieces[0]
    assert original.pieces[0].text == "Um, we start now."


@pytest.mark.parametrize("separator", [" ", "\t", "\n", " \t\n "])
def test_keep_immediately_after_unkept_filler_ignores_separator(separator: str) -> None:
    original = _turn(_piece("um" + separator + "uh we start"))
    result = _filter([original], {_source(_A, 1): "keep"})
    assert _texts(result) == ["Uh we start"]
    assert result.trace == (
        Removal("filler", "um", (_source(_A, 0),)),
        Protected("uh", (_source(_A, 1),)),
    )
    assert (result.fillers_removed, result.kept) == (1, 1)


def test_keep_inside_phrase_and_another_occurrence_removed() -> None:
    original = _turn(_piece("Well, you know, it froze; you know, we left."))
    result = _filter(
        [original], {_source(_A, 2): "keep"}, fillers=effective_filler_list(["you know"])
    )
    assert _texts(result) == ["Well, you know, it froze; we left."]
    assert result.trace == (
        Protected("you know", (_source(_A, 1), _source(_A, 2))),
        Removal("phrase", "you know", (_source(_A, 5), _source(_A, 6))),
    )
    assert (result.fillers_removed, result.kept) == (2, 1)


def test_phrase_adjacent_whitespace_does_not_protect_phrase() -> None:
    original = _turn(_piece("Well, you know,\t uh we start"))
    result = _filter(
        [original], {_source(_A, 3): "keep"}, fillers=effective_filler_list(["you know"])
    )
    assert _texts(result) == ["Well, uh we start"]
    assert result.trace == (
        Removal("phrase", "you know", (_source(_A, 1), _source(_A, 2))),
        Protected("uh", (_source(_A, 3),)),
    )
    assert (result.fillers_removed, result.kept) == (2, 1)


@pytest.mark.parametrize("text", ["we you know go", "you know.", "you know?", "Yes. you know!"])
def test_f1p_rejected_phrase_is_not_counted_as_kept(text: str) -> None:
    tokens = text.split()
    token = next(i for i, value in enumerate(tokens) if value == "you")
    result = _filter(
        [_turn(_piece(text))],
        {_source(_A, token): "keep"},
        fillers=effective_filler_list(["you know"]),
    )
    assert _texts(result) == [text]
    assert result.trace == ()
    assert result.kept == 0


def test_keep_count_deduplicates_phrase_and_word_protection() -> None:
    original = _turn(_piece("um uh, we start"))
    result = _filter([original], {_source(_A, 1): "keep"}, fillers=effective_filler_list(["um uh"]))
    assert _texts(result) == ["Uh, we start"]
    assert result.trace == (
        Protected("um uh", (_source(_A, 0), _source(_A, 1))),
        Removal("filler", "um", (_source(_A, 0),)),
        Protected("uh", (_source(_A, 1),)),
    )
    assert result.kept == 1


@pytest.mark.parametrize(
    "text, token, expected, lexical",
    [
        ("So basically, the coil froze.", 1, "So the coil froze.", "basically"),
        ("basically, the coil froze.", 0, "The coil froze.", "basically"),
        ("Yes, basically. next", 1, "Yes. Next", "basically"),
        ("Yes basically? next", 1, "Yes? Next", "basically"),
        ("Yes; basically! next", 1, "Yes! Next", "basically"),
        ("Yes. basically! next", 1, "Yes. Next", "basically"),
        ("So basically; the coil froze.", 1, "So the coil froze.", "basically"),
        ("So basically: the coil froze.", 1, "So the coil froze.", "basically"),
        ("So basically,\t\n the coil froze.", 1, "So the coil froze.", "basically"),
        ('So "basically" the coil froze.', 1, 'So the coil froze.', "basically"),
        ("So (basically) the coil froze.", 1, "So the coil froze.", "basically"),
        ("So [basically] the coil froze.", 1, "So the coil froze.", "basically"),
        (
            "So \u2018basically\u2019 the coil froze.",
            1,
            "So the coil froze.",
            "basically",
        ),
        ('So "basically", the coil froze.', 1, 'So the coil froze.', "basically"),
        ('So "basically." next', 1, 'So. Next', "basically"),
        ("So (basically). next", 1, "So. Next", "basically"),
        ("(basically). next", 0, "Next", "basically"),
        ("So (basically), next", 1, "So next", "basically"),
        ("(basically) the coil froze.", 0, "The coil froze.", "basically"),
        ('"basically" the coil froze.', 0, "The coil froze.", "basically"),
        ("So (basically.) next", 1, "So. Next", "basically"),
        ("So ([basically.]) next", 1, "So. Next", "basically"),
        ("So (basically next", 1, "So (next", "basically"),
        ("So basically) next", 1, "So ) next", "basically"),
        ("So (basically] next", 1, "So (] next", "basically"),
    ],
)
def test_omit_reuses_f2_f4_and_preserves_unbalanced_wrappers(
    text: str,
    token: int,
    expected: str,
    lexical: str,
) -> None:
    original = _turn(_piece(text))
    result = _filter([original], {_source(_A, token): "omit"})
    assert _texts(result) == [expected]
    assert result.trace == (Removal("omit", lexical, (_source(_A, token),)),)
    assert (result.fillers_removed, result.omitted, result.kept) == (0, 1, 0)
    assert original.pieces[0].text == text


@pytest.mark.parametrize("unit", [",", "--", "()", "..."])
def test_omit_punctuation_only_unit_is_removed(unit: str) -> None:
    result = _filter([_turn(_piece(f"So {unit} the coil froze."))], {_source(_A, 1): "omit"})
    assert _texts(result) == ["So the coil froze."]
    assert result.trace == (Removal("omit", unit, (_source(_A, 1),)),)
    assert (result.omitted, result.fillers_removed, result.kept) == (1, 0, 0)


@pytest.mark.parametrize("word", ["like", "so", "right", "well"])
def test_operator_omit_applies_even_with_an_empty_effective_list(word: str) -> None:
    fillers = replace(DEFAULT_FILLER_LIST, words=(), phrases=())
    result = _filter(
        [_turn(_piece(f"{word} the coil froze."))], {_source(_A, 0): "omit"}, fillers=fillers
    )
    assert _texts(result) == ["The coil froze."]
    assert result.trace == (Removal("omit", word, (_source(_A, 0),)),)
    assert (result.omitted, result.fillers_removed) == (1, 0)


def test_omit_spans_follow_text_order_and_do_not_overlap() -> None:
    marks: EffectiveMarks = {_source(_A, 2): "omit", _source(_A, 0): "omit", _source(_A, 1): "omit"}
    result = _filter([_turn(_piece("basically, actually, really. next"))], marks)
    assert _texts(result) == ["Next"]
    assert result.trace == tuple(
        Removal("omit", word, (_source(_A, token),))
        for token, word in enumerate(("basically", "actually", "really"))
    )
    assert result.omitted == 3


def test_omitted_filler_counts_once_as_omit() -> None:
    result = _filter([_turn(_piece("um uh next"))], {_source(_A, 0): "omit"})
    assert _texts(result) == ["Next"]
    assert result.trace == (
        Removal("omit", "um", (_source(_A, 0),)),
        Removal("filler", "Uh", (_source(_A, 1),)),
    )
    assert (result.omitted, result.fillers_removed) == (1, 1)


def test_omit_then_phrase_judges_remaining_comma() -> None:
    original = _turn(_piece("you, basically, know, it froze"))
    result = _filter(
        [original], {_source(_A, 1): "omit"}, fillers=effective_filler_list(["you know"])
    )
    assert _texts(result) == ["you, know, it froze"]
    # The comma on "you," stays under F2; it prevents a phrase match.
    assert result.trace == (Removal("omit", "basically", (_source(_A, 1),)),)
    assert (result.omitted, result.fillers_removed, result.kept) == (1, 0, 0)


def test_omit_creates_a_phrase_and_keep_can_protect_it() -> None:
    original = _turn(_piece("you basically, know, it froze"))
    fillers = effective_filler_list(["you know"])
    # Both passes read the rewritten text; original token coordinates stay fixed.
    omitted: EffectiveMarks = {_source(_A, 1): "omit"}
    result = _filter([original], omitted, fillers=fillers)
    assert _texts(result) == ["It froze"]
    assert result.trace == (
        Removal("omit", "basically", (_source(_A, 1),)),
        Removal("phrase", "you know", (_source(_A, 0), _source(_A, 2))),
    )
    assert (result.omitted, result.fillers_removed) == (1, 2)
    protected = _filter([original], {**omitted, _source(_A, 2): "keep"}, fillers=fillers)
    assert _texts(protected) == ["you know, it froze"]
    assert protected.trace == (
        result.trace[0],
        Protected("you know", (_source(_A, 0), _source(_A, 2))),
    )
    assert (protected.omitted, protected.fillers_removed, protected.kept) == (1, 0, 1)


def test_omit_only_turn_merges_identity_and_repeats_respect_seam() -> None:
    original = [
        _turn(_piece("the")),
        _turn(_piece("basically", segment=_C), speaker="Sam"),
        _turn(_piece("the cat", segment=_B)),
    ]
    marks: EffectiveMarks = {_source(_C, 0): "omit"}
    cleaned, trace = drop_fillers_with_trace(original, fillers=DEFAULT_FILLER_LIST, marks=marks)
    assert [(join_pieces(t.pieces), seams) for t, seams in cleaned] == [
        ("the the cat", frozenset({1}))
    ]
    assert trace == (Removal("omit", "basically", (_source(_C, 0),)),)
    result = _filter(original, marks, repeats=True)
    assert _texts(result) == ["the the cat"]
    assert (result.omitted, result.repeats_removed) == (1, 0)


def test_keep_changes_f5_survival_and_blocks_identity_merge() -> None:
    original = [
        _turn(_piece("Yes")),
        _turn(_piece("uh", segment=_C), speaker="Sam"),
        _turn(_piece("next", segment=_B)),
    ]
    result = _filter(original, {_source(_C, 0): "keep"})
    assert _texts(result) == ["Yes", "uh", "next"]
    assert result.trace == (Protected("uh", (_source(_C, 0),)),)
    assert result.kept == 1


def test_f5_probe_uses_marks_but_final_merge_alone_traces() -> None:
    original = [
        _turn(_piece("the")),
        _turn(_piece("basically", segment=_C), speaker="Sam"),
        _turn(_piece("you know, uh next", segment=_B)),
    ]
    marks: EffectiveMarks = {_source(_C, 0): "omit", _source(_B, 1): "keep", _source(_B, 2): "keep"}
    result = _filter(original, marks, fillers=effective_filler_list(["you know"]))
    assert _texts(result) == ["the you know, uh next"]
    # Probe protects "you know", but the merged F1p rejects it first.
    assert result.trace == (
        Protected("uh", (_source(_B, 2),)),
        Removal("omit", "basically", (_source(_C, 0),)),
    )
    assert (result.omitted, result.kept) == (1, 1)


def test_f5_final_merge_moves_omit_mark_and_preserves_sources() -> None:
    original = [
        _turn(_piece("Yes,")),
        _turn(_piece("um", segment=_C), speaker="Sam"),
        _turn(_piece("basically. uh next", segment=_B)),
    ]
    marks: EffectiveMarks = {_source(_B, 0): "omit", _source(_B, 1): "keep"}
    fillers = DEFAULT_FILLER_LIST
    with patch("voxint.export.fillers._clean_tracked", wraps=_clean_tracked) as clean:
        result = _filter(original, marks, fillers=fillers)
    assert _texts(result) == ["Yes. Uh next"]
    assert result.trace == (
        Removal("omit", "basically", (_source(_B, 0),)),
        Protected("Uh", (_source(_B, 1),)),
        Removal("filler", "um", (_source(_C, 0),)),
    )
    final_call = next(call for call in clean.call_args_list if call.kwargs.get("sources") == (0, 1))
    tracked = _clean_tracked(final_call.args[0], fillers=fillers, sources=(0, 1), marks=marks)
    chars = _flatten(tracked)
    assert next(char for char in chars if char.text == ".").source == _source(_B, 0)
    assert next(char for char in chars if char.text == "U").source == _source(_B, 1)
    assert all(char.source is None for char in chars if char.text.isspace())
    assert (result.omitted, result.kept, result.fillers_removed) == (1, 1, 1)


def test_keep_does_not_protect_neighbor_via_a_moved_terminal_mark() -> None:
    original = _turn(_piece("Yes, basically. um next"))
    result = _filter([original], {_source(_A, 0): "keep", _source(_A, 1): "omit"})
    assert _texts(result) == ["Yes. Next"]
    assert result.trace == (
        Removal("omit", "basically", (_source(_A, 1),)),
        Removal("filler", "Um", (_source(_A, 2),)),
    )
    assert result.kept == 0


@pytest.mark.parametrize("action, expected", [("keep", "left um right"), ("omit", "left right")])
def test_mark_on_real_split_child_uses_parent_unit(
    action: Literal["keep", "omit"],
    expected: str,
) -> None:
    parent = emission((" left", " um", " right"))
    children = derive_children(parent.seg, [1])
    assert children is not None
    turns = project_turns(
        [replace(parent, child=child, child_index=i) for i, child in enumerate(children)],
        [],
        STATES,
        text=TranscriptText.RAW,
    )
    key = (parent.seg.id, 1, 2)
    assert turns[0].pieces[1].anchors[0] == WordAnchor(parent.seg.id, 1, 2, 0, 2)
    result = _filter(turns, {key: action})
    assert _texts(result) == [expected]
    assert (result.kept, result.omitted, result.fillers_removed) == (
        (1, 0, 0) if action == "keep" else (0, 1, 0)
    )


def test_glued_unit_counts_one_word_not_parent_tokens() -> None:
    original = replace(
        _piece("um next", coarse=True),
        anchors=(WordAnchor(_A, 0, 3, 0, 2), WordAnchor(_A, 3, 4, 3, 7)),
    )
    result = _filter([_turn(original)], {(_A, 0, 3): "omit"})
    assert _texts(result) == ["Next"]
    assert result.trace == (Removal("omit", "um", ((_A, 0, 3),)),)
    assert (result.omitted, result.fillers_removed) == (1, 0)


def test_repeats_run_after_omit() -> None:
    result = _filter(
        [_turn(_piece("the basically the cat"))], {_source(_A, 1): "omit"}, repeats=True
    )
    assert _texts(result) == ["the cat"]
    assert (result.omitted, result.fillers_removed, result.repeats_removed) == (1, 0, 1)


def test_keep_does_not_override_repeat_rules() -> None:
    original = _turn(_piece("the the cat"))
    fillers = effective_filler_list(["the"])
    marks: EffectiveMarks = {_source(_A, 0): "keep", _source(_A, 1): "keep"}
    result = _filter([original], marks, fillers=fillers, repeats=True)
    assert _texts(result) == ["the cat"]
    assert (result.kept, result.fillers_removed, result.repeats_removed) == (2, 0, 1)
    assert result.trace == (
        Protected("the", (_source(_A, 0),)),
        Protected("the", (_source(_A, 1),)),
    )
    uh = _filter([_turn(_piece("uh uh next"))], marks)
    assert _texts(uh) == ["uh uh next"]
    # "uh" is outside the repeat allow-list, regardless of keep.
    assert _filter([_turn(_piece("uh uh next"))], marks, repeats=True) == uh


@pytest.mark.parametrize("action", ["keep", "omit"])
@pytest.mark.parametrize(
    "original", [[], [_turn(_piece("um", anchored=False))], [_turn(_piece("um", segment=_B))]]
)
def test_unplaceable_marks_refuse_before_any_clean(
    original: list[SpeakerTurn],
    action: Literal["keep", "omit"],
) -> None:
    marks: EffectiveMarks = {_source(_A, 0): action}
    with (
        patch("voxint.export.fillers._clean_tracked", wraps=_clean_tracked) as clean,
        pytest.raises(UnplaceableWordMarkError) as error,
    ):
        drop_fillers_with_trace(original, fillers=DEFAULT_FILLER_LIST, marks=marks)
    assert error.value.marks == [(_A, 0, 1, action)]
    assert str(_A) in str(error.value)
    clean.assert_not_called()


def test_unplaceable_error_reports_all_missing_marks_only() -> None:
    marks: EffectiveMarks = {_source(_A, 0): "keep", _source(_B, 0): "omit", _source(_C, 0): "keep"}
    with pytest.raises(UnplaceableWordMarkError) as error:
        _filter([_turn(_piece("um"))], marks)
    assert error.value.marks == [(_B, 0, 1, "omit"), (_C, 0, 1, "keep")]


@pytest.mark.parametrize("repeats", [False, True])
def test_marks_ignored_entirely_without_fillers(repeats: bool) -> None:
    original = [_turn(_piece("the the um cat"))]
    marks: EffectiveMarks = {_source(_A, 0): "omit", _source(_A, 2): "keep", _source(_C, 9): "omit"}
    marked = apply_turn_filters(original, fillers=None, drop_repeats=repeats, marks=marks)
    baseline = apply_turn_filters(original, fillers=None, drop_repeats=repeats)
    assert asdict(marked) == asdict(baseline)
    assert marked.trace == ()
    assert marked.omitted == marked.kept == 0
    if not repeats:
        assert marked.turns[0] is original[0]


@pytest.mark.parametrize("case", _DATA["cases"], ids=lambda case: case["name"])
def test_empty_marks_identical_over_characterization_inputs(case: dict[str, Any]) -> None:
    original = _decode(case["input"])
    for fillers, repeats in itertools.product(_OPTIONS, (False, True)):
        baseline = apply_turn_filters(original, fillers=fillers, drop_repeats=repeats, marks=None)
        empty = apply_turn_filters(original, fillers=fillers, drop_repeats=repeats, marks={})
        assert asdict(empty) == asdict(baseline)
        if fillers is not None:
            assert drop_fillers_with_trace(original, fillers=fillers, marks={}) == (
                drop_fillers_with_trace(original, fillers=fillers, marks=None)
            )


def test_empty_marks_identical_on_shared_soups() -> None:
    for pieces in seeded_filler_soups(200):
        original = [_turn(*pieces)]
        for fillers in (DEFAULT_FILLER_LIST, effective_filler_list(["you know"])):
            assert drop_fillers_with_trace(original, fillers=fillers, marks={}) == (
                drop_fillers_with_trace(original, fillers=fillers, marks=None)
            )
