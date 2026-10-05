"""Source identities and committed removals through F1-F5 rewrites."""

import re
import uuid
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import pytest

from voxint.adjudication.turns import (
    PieceRule,
    SpeakerTurn,
    TextMapping,
    TurnPiece,
    WordAnchor,
    join_pieces,
)
from voxint.export.filler_lists import DEFAULT_FILLER_LIST, effective_filler_list
from voxint.export.fillers import (
    Protected,
    Removal,
    SourceIdentity,
    _Char,
    _clean_pass,
    _clean_tracked,
    _flatten,
    _join_at_seam,
    _removal_spans,
    _track,
    _with_chars,
    _word_pattern,
    drop_fillers_with_seams,
    drop_fillers_with_trace,
)
from voxint.export.turn_filters import apply_turn_filters

_A = uuid.UUID(int=1)
_B = uuid.UUID(int=2)
_C = uuid.UUID(int=3)


def _piece(
    text: str, token: int = 0, *, segment: uuid.UUID = _A, coarse: bool = False,
    anchored: bool = True, boundary: bool = False,
) -> TurnPiece:
    anchors = tuple(WordAnchor(segment, token + i, token + i + 1, m.start(), m.end())
                    for i, m in enumerate(re.finditer(r"\S+", text))) if anchored else ()
    return TurnPiece(text, token, token + 0.5, not coarse, boundary,
                     PieceRule.SPLIT_CHILD if coarse else PieceRule.WORD_LEVEL,
                     TextMapping.COARSE if coarse else TextMapping.VERBATIM, anchors)


def _turn(*pieces: TurnPiece, speaker: str = "Alex") -> SpeakerTurn:
    return SpeakerTurn(("speaker", speaker), speaker, pieces)


def _source(segment: uuid.UUID, token: int) -> SourceIdentity:
    return segment, token, token + 1


def test_phrase_then_word_on_anchored_split_child() -> None:
    original = _piece("you know, um, stay", coarse=True)
    fillers = effective_filler_list(["you know"])
    cleaned, trace = drop_fillers_with_trace([_turn(original)], fillers=fillers)
    assert trace == (
        Removal("phrase", "you know", (_source(_A, 0), _source(_A, 1))),
        Removal("filler", "Um", (_source(_A, 2),)),
    )
    assert join_pieces(cleaned[0][0].pieces) == "Stay"
    assert cleaned[0][0].pieces[0].anchors == ()
    assert len(original.anchors) == 4
    tracked = _clean_tracked(_track((original,)), fillers=fillers)
    assert [(c.text, c.source) for c in _flatten(tracked)] == [
        (c, _source(_A, 3)) for c in "Stay"
    ]


@pytest.mark.parametrize("boundary", [False, True])
def test_phrase_spanning_segments_has_two_ordered_sources(boundary: bool) -> None:
    original = _turn(_piece("Fine,"), _piece(" you", 1),
                     _piece("know," if boundary else " know,", segment=_B, boundary=boundary),
                     _piece(" yes.", 1, segment=_B))
    cleaned, trace = drop_fillers_with_trace(
        [original], fillers=effective_filler_list(["you know"]),
    )
    assert trace == (Removal("phrase", "you know", (_source(_A, 1), _source(_B, 0))),)
    assert join_pieces(cleaned[0][0].pieces) == "Fine, yes."
    assert original.pieces[0].anchors == cleaned[0][0].pieces[0].anchors


@pytest.mark.parametrize("left, expected", [("Yes", "Yes. Next"), ("Yes,", "Yes. Next"),
                                          ('"and then,"', '"and then." Next')])
def test_moved_mark_retains_filler_mark_source_through_both_passes(
    left: str, expected: str,
) -> None:
    original = (_piece(left), _piece(" you", 10), _piece(" know.", segment=_B),
                _piece(" um,", 1, segment=_B), _piece(" next", 2, segment=_B))
    fillers = effective_filler_list(["you know"])
    # A comma sets off the phrase; without it the word pass moves its own mark.
    if left == "Yes":
        original = (_piece(left), _piece(" um.", segment=_B),
                    _piece(" next", 2, segment=_B))
    tracked = _clean_tracked(_track(original), fillers=fillers)
    assert join_pieces(p.piece for p in tracked) == expected
    chars = _flatten(tracked)
    assert next(c for c in chars if c.text == ".").source == _source(_B, 0)
    assert next(c for c in chars if c.text == "N").source == _source(_B, 2)
    assert all(c.source is None for c in chars if c.text.isspace())
    assert all(not p.piece.anchors for p in tracked if p.piece.text != original[p.index].text)


def test_existing_terminal_mark_keeps_its_source_and_capitalisation_keeps_next_source() -> None:
    tracked = _clean_tracked(_track((
        _piece("Done."), _piece(" um!", segment=_B), _piece(' "next"', segment=_C),
    )), fillers=DEFAULT_FILLER_LIST)
    assert join_pieces(p.piece for p in tracked) == 'Done. "Next"'
    chars = _flatten(tracked)
    assert next(c for c in chars if c.text == ".").source == _source(_A, 0)
    assert next(c for c in chars if c.text == "N").source == _source(_C, 0)


@pytest.mark.parametrize("initial", ["ß", "ﬂow"])
def test_expanding_uppercase_retains_characters_and_identities(initial: str) -> None:
    tracked = _clean_tracked(_track((_piece("um,"), _piece(" " + initial, segment=_B))),
                             fillers=DEFAULT_FILLER_LIST)
    assert join_pieces(p.piece for p in tracked) == initial
    assert all(c.source == _source(_B, 0) for c in _flatten(tracked))


def test_seam_trims_only_whitespace_and_preserves_head_and_tail_identities() -> None:
    left = _track((_piece("Yes, \t"),))
    right = _track((_piece("\n you", segment=_B), _piece(" know.", 1, segment=_B),
                    _piece(" next", 2, segment=_B)))
    joined = _join_at_seam(left, right)
    assert join_pieces(p.piece for p in joined) == "Yes, you know. next"
    assert joined[0].piece.anchors == joined[1].piece.anchors == ()
    chars = _flatten(joined)
    assert [(c.text, c.source) for c in chars[:6]] == [
        *[(c, _source(_A, 0)) for c in "Yes,"], (" ", None), ("y", _source(_B, 0)),
    ]
    # Owners guard F1p across an F5 seam, independently of anchor segment ids.
    cleaned = _clean_tracked(joined, fillers=effective_filler_list(["you know"]),
                             sources=(0, 1, 1, 1))
    assert join_pieces(p.piece for p in cleaned) == "Yes. Next"
    assert next(c for c in _flatten(cleaned) if c.text == ".").source == _source(_B, 1)


def test_f5_discards_merges_and_traces_each_committed_removal_once() -> None:
    original = [_turn(_piece("Yes, \t")),
                _turn(_piece("um", segment=_C), speaker="Sam"),
                _turn(_piece("\n um.", segment=_B), _piece(" next", 1, segment=_B))]
    # Each original turn is anchored once, even though the clean runs twice.
    with patch("voxint.export.fillers._track", wraps=_track) as track:
        cleaned, trace = drop_fillers_with_trace(original, fillers=DEFAULT_FILLER_LIST)
    assert track.call_count == len(original)
    assert [(join_pieces(t.pieces), s) for t, s in cleaned] == [("Yes. Next", frozenset({1}))]
    assert trace == (Removal("filler", "um", (_source(_B, 0),)),
                     Removal("filler", "um", (_source(_C, 0),)))
    assert all(p.anchors == () for p in cleaned[0][0].pieces)
    assert cleaned == drop_fillers_with_seams(original, fillers=DEFAULT_FILLER_LIST)
    result = apply_turn_filters(original, fillers=DEFAULT_FILLER_LIST, drop_repeats=False)
    assert result.trace == trace
    assert result.fillers_removed == 2


def test_survival_probe_removal_that_is_rejected_after_merge_is_not_traced() -> None:
    original = [_turn(_piece("the")),
                _turn(_piece("um", segment=_C), speaker="Sam"),
                _turn(_piece("you know,", segment=_B), _piece(" the cat", 2, segment=_B))]
    cleaned, trace = drop_fillers_with_trace(original, fillers=effective_filler_list(["you know"]))
    assert join_pieces(cleaned[0][0].pieces) == "the you know, the cat"
    assert trace == (Removal("filler", "um", (_source(_C, 0),)),)


def test_unanchored_coarse_pieces_have_no_sources() -> None:
    original = _turn(_piece("Um, you know, yes.", coarse=True, anchored=False))
    _, trace = drop_fillers_with_trace([original], fillers=effective_filler_list(["you know"]))
    assert trace == (Removal("phrase", "you know", ()), Removal("filler", "Um", ()))


def test_lexical_trace_excludes_wrappers_and_deduplicates_glued_token_ranges() -> None:
    original = replace(_piece('"um, yes'), anchors=(WordAnchor(_A, 4, 7, 0, 4),))
    cleaned, trace = drop_fillers_with_trace([_turn(original)], fillers=DEFAULT_FILLER_LIST)
    assert trace == (Removal("filler", "um", ((_A, 4, 7),)),)
    assert join_pieces(cleaned[0][0].pieces) == '"Yes'
    tracked = _clean_tracked(_track((original,)), fillers=DEFAULT_FILLER_LIST)
    assert tracked[0].chars[0].source == (_A, 4, 7)
    assert tracked[0].chars[1].source is None


def test_unchanged_pieces_keep_anchors_and_changed_repeat_pieces_clear_them() -> None:
    original = _turn(_piece("um,"), _piece(" next", 1), _piece(" steady", 2))
    result = apply_turn_filters([original], fillers=DEFAULT_FILLER_LIST, drop_repeats=False)
    assert result.turns[0].pieces[0].anchors == ()
    assert result.turns[0].pieces[1].anchors == original.pieces[2].anchors
    repeated = _turn(_piece("the the cat", coarse=True))
    result = apply_turn_filters([repeated], fillers=None, drop_repeats=True)
    assert result.trace == ()
    assert result.turns[0].pieces[0].anchors == ()
    assert join_pieces(result.turns[0].pieces) == "the cat"
    assert apply_turn_filters([original], fillers=None, drop_repeats=False).trace == ()


def test_reserved_record_types_are_frozen_and_emit_no_omit_or_kept_entries() -> None:
    omitted = Removal("omit", "word", ())
    kept = Protected("um", (_source(_A, 0),))
    assert kept.cause == "kept"
    with pytest.raises(FrozenInstanceError):
        omitted.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        kept.text = "changed"  # type: ignore[misc]
    _, trace = drop_fillers_with_trace([_turn(_piece("um, yes"))], fillers=DEFAULT_FILLER_LIST)
    assert all(r.cause == "filler" for r in trace)


def test_span_skip_hook_receives_only_lexical_characters() -> None:
    # The hook is dormant in production; wrappers and separators are outside its range.
    pieces = _track((_piece('"um, uh, yes'),))
    chars = _flatten(pieces)
    spans = _removal_spans("".join(c.text for c in chars),
                           _word_pattern(DEFAULT_FILLER_LIST.words), "filler")
    seen: list[str] = []

    def skip(lexical: Sequence[_Char]) -> bool:
        text = "".join(c.text for c in lexical)
        seen.append(text)
        return True

    result = _clean_pass(pieces, chars, spans, skip=skip)
    assert seen == ["um", "uh"]
    assert result[0].piece.text == pieces[0].piece.text



def test_piece_restored_to_original_text_retains_its_original_anchors() -> None:
    # The phrase pass materializes a boundary separator; the word pass removes it.
    original = _turn(_piece("you know,", coarse=True), _piece("um,", 2, boundary=True),
                     _piece("Stay", 3, boundary=True))
    result = apply_turn_filters([original], fillers=effective_filler_list(["you know"]),
                                drop_repeats=False)
    assert join_pieces(result.turns[0].pieces) == "Stay"
    assert result.turns[0].pieces[0].anchors == original.pieces[2].anchors


def test_identity_only_rewrite_does_not_restore_original_anchors() -> None:
    original = _track((_piece("stay"),))[0]
    rewritten = [_Char(c.text, c.owner, _source(_B, 0)) for c in original.chars]
    result = _with_chars(original, rewritten)
    assert result.piece.text == original.original.text
    assert result.piece.anchors == ()
    assert all(c.source == _source(_B, 0) for c in result.chars)


def test_skipped_filler_is_capitalised_after_sentence_start_removal() -> None:
    original = _track((_piece("um, um, next", coarse=True),))
    # Slice 3 keep protection relies on F4 still capitalising a skipped filler.
    result = _clean_tracked(original, fillers=DEFAULT_FILLER_LIST,
                            skip=lambda chars: chars[0].source == _source(_A, 1))
    assert result[0].piece.text == "Um, next"
    assert result[0].chars[0].source == _source(_A, 1)


def test_f5_repeated_reindexing_preserves_owners_seams_and_sources() -> None:
    original = [
        _turn(_piece("First, \t")),
        _turn(_piece("um", segment=_C), speaker="Sam"),
        _turn(_piece("\n um.", segment=_B), _piece(" next", 1, segment=_B)),
        _turn(_piece("uh", 1, segment=_C), speaker="Sam"),
        _turn(_piece("\t uh!", 2, segment=_B), _piece(" last", 3, segment=_B)),
    ]
    joined = _join_at_seam(_track(original[0].pieces), _track(original[2].pieces))
    joined = _join_at_seam(joined, _track(original[4].pieces))
    owners = (0, 1, 1, 2, 2)
    assert [p.index for p in joined] == list(range(5))
    assert [[c.owner for c in p.chars] for p in joined] == [
        [i] * len(p.chars) for i, p in enumerate(joined)
    ]
    # Inspect the actual final F5 clean, including its input-turn owner groups.
    with patch("voxint.export.fillers._clean_tracked", wraps=_clean_tracked) as clean:
        output, trace = drop_fillers_with_trace(original, fillers=DEFAULT_FILLER_LIST)
    final_call = next(call for call in clean.call_args_list
                      if call.kwargs.get("sources") == owners)
    actual_joined = final_call.args[0]
    assert actual_joined == joined
    cleaned = _clean_tracked(actual_joined, fillers=DEFAULT_FILLER_LIST, sources=owners)
    assert [(p.index, p.piece.text) for p in cleaned] == [
        (0, "First."), (2, " Next!"), (4, " Last"),
    ]
    expected = [
        *[(c, 0, _source(_A, 0)) for c in "First"], (".", 0, _source(_B, 0)),
        (" ", 2, None), *[(c, 2, _source(_B, 1)) for c in "Next"],
        ("!", 2, _source(_B, 2)), (" ", 4, None),
        *[(c, 4, _source(_B, 3)) for c in "Last"],
    ]
    assert [(c.text, c.owner, c.source) for c in _flatten(cleaned)] == expected
    assert [(join_pieces(t.pieces), seams) for t, seams in output] == [
        ("First. Next! Last", frozenset({1, 2})),
    ]
    assert trace == (
        Removal("filler", "um", (_source(_B, 0),)),
        Removal("filler", "uh", (_source(_B, 2),)),
        Removal("filler", "um", (_source(_C, 0),)),
        Removal("filler", "uh", (_source(_C, 1),)),
    )
