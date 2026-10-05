"""Synthetic projection fixtures with shared emission and run round trips."""

import html
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from voxint.adjudication.attribution import Emission
from voxint.adjudication.resolver import LabelState, Resolution, SegmentOverride
from voxint.adjudication.splits import DerivedChild, derive_children, validated_words
from voxint.adjudication.transcript import TranscriptText, _join_segment_texts, resolve_body
from voxint.adjudication.turns import (
    PieceRule,
    SpeakerTurn,
    TextMapping,
    TurnPiece,
    WordAnchor,
    coarse_turns,
    join_pieces,
    project_turns,
)
from voxint.db.models import AdjudicationDecision, SegmentReviewState, TranscriptSegment
from voxint.export import to_markdown_turns
from voxint.export.reading import layout_turns


@dataclass(frozen=True)
class Span:
    turn_index: int
    start_seconds: float
    end_seconds: float
    label: str


def state(label: str, name: str, number: int) -> LabelState:
    return LabelState(
        label,
        1,
        10,
        Resolution.HUMAN_ASSIGN,
        uuid.UUID(int=number),
        name,
        None,
        None,
        None,
        False,
        None,
        None,
    )


STATES = {
    s.label: s
    for s in (
        state("SPEAKER_00", "Alex", 1),
        state("SPEAKER_01", "Sam", 2),
        state("SPEAKER_02", "Jo", 3),
    )
}
A, B, C = STATES


def emission(
    tokens: Sequence[str] = (" hello", " world"),
    *,
    start: float = 0,
    label: str | None = C,
    enhanced: str | None = None,
) -> Emission:
    seg = TranscriptSegment(
        id=uuid.uuid4(),
        pipeline_run_id=uuid.uuid4(),
        segment_index=0,
        start_seconds=start,
        end_seconds=start + len(tokens) + 1,
        raw_text="".join(tokens),
        enhanced_text=enhanced,
        diarization_label=label,
        words=[{"word": t, "start": start + i, "end": start + i + 1} for i, t in enumerate(tokens)],
    )
    return Emission(seg, None, None, STATES.get(label or ""), None, None, None)


def override(label: str) -> SegmentOverride:
    s = STATES[label]
    assert s.speaker_id is not None
    return SegmentOverride(s.speaker_id, s.speaker_name, AdjudicationDecision())


def check_anchors(
    emissions: list[Emission], result: list[SpeakerTurn], text: TranscriptText,
) -> None:
    """Check lexical slices and ordered parent coordinates independently of units."""
    selected = [
        (e, e.child.text if e.child else resolve_body(
            e.seg, e.review.corrected_text if e.review else None, text,
        ))
        for e in emissions
    ]
    selected = [(e, body) for e, body in selected if body.strip()]
    emission_index = -1
    anchor_index = 0
    previous_end: dict[uuid.UUID, int] = {}
    for turn in result:
        for piece in turn.pieces:
            if piece.segment_start:
                emission_index += 1
                anchor_index = 0
            e, body = selected[emission_index]
            words = validated_words(e.seg, min_words=1)
            if piece.mapping is not TextMapping.COARSE:
                assert len(piece.anchors) == 1
            lexical_end = 0
            for anchor in piece.anchors:
                assert words is not None
                assert anchor.segment_id == e.seg.id
                assert 0 <= anchor.token_start < anchor.token_end <= len(words)
                assert anchor.token_start >= previous_end.get(anchor.segment_id, 0)
                previous_end[anchor.segment_id] = anchor.token_end
                assert 0 <= anchor.lex_start < anchor.lex_end <= len(piece.text)
                assert anchor.lex_start >= lexical_end
                lexical_end = anchor.lex_end
                lexical = piece.text[anchor.lex_start:anchor.lex_end]
                assert lexical == lexical.strip()
                edited = piece.mapping is TextMapping.TOKEN or (
                    piece.rule is PieceRule.SEGMENT_OVERRIDE
                    and body.strip() != "".join(w.text for w in words).strip()
                )
                expected = (
                    re.findall(r"\S+", body)[anchor_index]
                    if edited
                    else "".join(w.text for w in words[anchor.token_start:anchor.token_end]).strip()
                )
                assert lexical == expected
                anchor_index += 1


def check(
    emissions: list[Emission],
    spans: Sequence[Span],
    *,
    text: TranscriptText = TranscriptText.RAW,
    states: dict[str, LabelState] | None = None,
) -> list[SpeakerTurn]:
    """Every projection fixture verifies each emission and the full run."""
    result = project_turns(iter(emissions), spans, STATES if states is None else states, text=text)
    check_anchors(emissions, result, text)
    pieces = [p for turn in result for p in turn.pieces]
    bodies = [
        e.child.text
        if e.child
        else resolve_body(e.seg, e.review.corrected_text if e.review else None, text)
        for e in emissions
    ]
    # Each non-empty emission starts exactly one piece group, even across turns.
    groups: list[str] = []
    for piece in pieces:
        if piece.segment_start:
            groups.append("")
        assert groups
        groups[-1] += piece.text
    assert [g.strip() for g in groups] == [b.strip() for b in bodies if b.strip()]
    # Exact, not whitespace-normalised: the boundary rule adds one space at most,
    # and only between emissions that do not already carry one.
    assert join_pieces(pieces) == _join_segment_texts(groups)
    assert " ".join(join_pieces(pieces).split()) == " ".join(_join_segment_texts(bodies).split())
    # Exercise rendered bytes for every shared projection fixture as well.
    rendered = to_markdown_turns(layout_turns(result))
    rendered = re.sub(r"\[\d+:\d{2}:\d{2}\] ?", "", rendered)
    rendered = re.sub(r"(?m)^\*\*.*?:\*\* ?", "", rendered)
    rendered = html.unescape(re.sub(r"\\(.)", r"\1", rendered))
    assert " ".join(rendered.split()) == " ".join(_join_segment_texts(bodies).split())
    return result


def test_two_speakers_raw_and_edited() -> None:
    spans = [Span(0, 0, 1, A), Span(1, 1, 2, B)]
    e = emission(enhanced="Hello, WORLD!  ")
    raw = check([e], spans)
    assert [t.speaker for t in raw] == ["Alex", "Sam"]
    assert all(
        p.rule == PieceRule.WORD_LEVEL and p.mapping == TextMapping.VERBATIM and p.timed
        for t in raw
        for p in t.pieces
    )
    edited = check([e], spans, text=TranscriptText.ENHANCED)
    assert join_pieces(p for t in edited for p in t.pieces) == e.seg.enhanced_text
    assert all(p.mapping == TextMapping.TOKEN for t in edited for p in t.pieces)
    corrected = replace(e, review=SegmentReviewState(corrected_text="HELLO! World?"))
    result = check([corrected], spans, text=TranscriptText.CORRECTED)
    assert join_pieces(p for t in result for p in t.pieces) == "HELLO! World?"


@pytest.mark.parametrize("one_identity", [False, True])
def test_rewritten_text(one_identity: bool) -> None:
    result = check(
        [emission(enhanced="A different sentence.")],
        [Span(0, 0, 1, A), Span(1, 1, 2, A if one_identity else B)],
        text=TranscriptText.ENHANCED,
    )
    assert len(result) == 1
    assert result[0].speaker == ("Alex" if one_identity else "Jo")
    piece = result[0].pieces[0]
    assert piece.rule == (PieceRule.SINGLE_IDENTITY if one_identity else PieceRule.UNMAPPED_TEXT)
    assert not piece.timed and piece.mapping == TextMapping.COARSE


@pytest.mark.parametrize(
    "tokens",
    [
        (" can", "'t", " re", "-", "enter", "!"),
        ("can", "'t", "re", "-", "enter"),
        (" ", " hello", " world", " ", "\n"),
    ],
)
def test_glued_tokens_and_whitespace(tokens: tuple[str, ...]) -> None:
    result = check(
        [emission(tokens)], [Span(i, i, i + 1, A if i % 2 == 0 else B) for i in range(len(tokens))]
    )
    pieces = [p for t in result for p in t.pieces]
    if tokens[0] == " can":
        assert [p.text for p in pieces] == [" can't", " re-enter!"]
        assert [(p.start_seconds, p.end_seconds) for p in pieces] == [(0, 2), (2, 6)]
    elif tokens[0] == "can":
        assert len(pieces) == 1
    else:
        assert pieces[0].text == "  hello" and pieces[-1].text == " world \n"


def test_glued_token_keeps_the_longest_end() -> None:
    # Validation orders token starts only, so glued punctuation can END before
    # the word it attaches to. The unit must keep the word's full interval, or
    # the short punctuation span decides the speaker.
    e = emission((" hello", "!"))
    e.seg.words = [
        {"word": " hello", "start": 0, "end": 2},
        {"word": "!", "start": 0.1, "end": 0.2},
    ]
    result = check([e], [Span(0, 0, 0.2, A), Span(1, 0.2, 2, B)])
    assert [t.speaker for t in result] == ["Sam"]
    piece = result[0].pieces[0]
    assert (piece.text, piece.start_seconds, piece.end_seconds) == (" hello!", 0, 2)


@pytest.mark.parametrize("leading", [False, True])
def test_whitespace_token_timing_does_not_pick_the_speaker(leading: bool) -> None:
    # A whitespace-only token keeps its bytes in the unit, but its timing is
    # not evidence of who spoke the word next to it.
    e = emission((" ", " hello") if leading else (" hello", " "))
    blank = {"word": " ", "start": 0, "end": 2} if leading else {"word": " ", "start": 3, "end": 5}
    word = {"word": " hello", "start": 2, "end": 3}
    e.seg.words = [blank, word] if leading else [word, blank]
    e.seg.end_seconds = 5  # the trailing blank must stay inside the segment
    spans = [Span(0, 2, 3, A), Span(1, 0, 2, B), Span(2, 3, 5, B)]
    result = check([e], spans)
    assert [t.speaker for t in result] == ["Alex"]
    piece = result[0].pieces[0]
    assert piece.text == ("  hello" if leading else " hello ")
    assert (piece.start_seconds, piece.end_seconds) == (2, 3)


def test_curly_quotes_hyphens_and_nonempty_keys() -> None:
    e = emission((" \u2018Can\u2019t\u2019", " re\u2010enter"), enhanced='"CAN\'T" RE-ENTER!\n')
    result = check([e], [Span(0, 0, 1, A), Span(1, 1, 2, B)], text=TranscriptText.ENHANCED)
    assert all(p.mapping == TextMapping.TOKEN for t in result for p in t.pieces)
    e = emission((" hello", " !"), enhanced="Hello ?")
    result = check([e], [Span(0, 0, 1, A), Span(1, 1, 2, B)], text=TranscriptText.ENHANCED)
    assert result[0].pieces[0].rule == PieceRule.UNMAPPED_TEXT


def test_zero_length_boundary_and_tie() -> None:
    e = emission(("hello",))
    e.seg.words = [{"word": "hello", "start": 1, "end": 1}]
    result = check([e], [Span(0, 0, 1, C), Span(9, 1, 2, B), Span(2, 1, 2, A)])
    assert result[0].speaker == "Alex"
    assert result[0].pieces[0].start_seconds == result[0].pieces[0].end_seconds == 1


def test_overlap_tie_and_greatest_overlap() -> None:
    e = emission((" hello",))
    assert check([e], [Span(9, 0, 1, B), Span(2, 0, 1, A)])[0].speaker == "Alex"
    assert check([e], [Span(0, 0, 0.2, B), Span(1, 0.2, 1, A)])[0].speaker == "Alex"


@pytest.mark.parametrize("bucket", [None, [], [{"word": "bad"}]])
def test_no_usable_words(bucket: list[dict[str, str]] | None) -> None:
    e = emission()
    e.seg.words = bucket
    result = check([e], [Span(0, 0, 3, A)])
    assert result[0].speaker == "Jo"
    assert result[0].pieces[0].rule == PieceRule.NO_WORDS


def test_reconcatenation_mismatch() -> None:
    e = emission()
    e.seg.raw_text = "Some other text"
    assert check([e], [Span(0, 0, 3, A)])[0].pieces[0].rule == PieceRule.NO_WORDS


@pytest.mark.parametrize("raw", [False, True])
def test_empty_selected_text(raw: bool) -> None:
    e = emission((" ", "\n")) if raw else emission(enhanced="")
    assert check([e], [], text=TranscriptText.RAW if raw else TranscriptText.ENHANCED) == []


def test_three_speakers_and_backward_segments() -> None:
    later = emission((" later",), start=4)
    earlier = emission((" earlier", " again"))
    result = check([later, earlier], [Span(0, 4, 5, C), Span(1, 0, 1, A), Span(2, 1, 2, B)])
    assert [t.speaker for t in result] == ["Jo", "Alex", "Sam"]
    assert [p.start_seconds for t in result for p in t.pieces] == [4, 0, 1]


@pytest.mark.parametrize("across", [False, True])
def test_supported_brackets(across: bool) -> None:
    emissions = (
        [emission((" one", " gap")), emission((" gap", " last"), start=2)]
        if across
        else [emission((" one", " gap", " gap", " last"))]
    )
    result = check(emissions, [Span(0, 0, 1, A), Span(1, 3, 4, A)])
    assert len(result) == 1 and result[0].speaker == "Alex"
    assert len(result[0].pieces) == 4
    assert {p.rule for p in result[0].pieces} == {PieceRule.SINGLE_IDENTITY}


def test_different_brackets_edges_and_no_chaining() -> None:
    e = emission((" edge", " one", " gap", " gap", " last", " edge"))
    result = check([e], [Span(0, 1, 2, A), Span(1, 4, 5, B)])
    assert [t.speaker for t in result] == ["Jo", "Alex", "Jo", "Sam", "Jo"]
    # Both interior gaps infer Alex, but cannot then infer the unbracketed tail.
    result = check([e], [Span(0, 1, 2, A), Span(1, 4, 5, A)])
    assert [t.speaker for t in result] == ["Jo", "Alex", "Jo"]
    assert len(result[1].pieces) == 4


@pytest.mark.parametrize("barrier", ["E1", "E2", "E3"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_coarse_barriers(barrier: str, side: str) -> None:
    middle = emission((" barrier",), start=2)
    if barrier == "E1":
        middle = replace(middle, child=DerivedChild(0, 1, 2, 3, "barrier"), child_index=0)
    elif barrier == "E2":
        middle = replace(middle, seg_override=override(B))
    else:
        middle.seg.words = None
    left = emission((" one", " gap"))
    right = emission((" gap", " last"), start=3)
    emissions = [left, middle, right]
    spans = [Span(0, 0, 1, A), Span(1, 4, 5, A)]
    if side == "left":
        left.seg.words = [{"word": " one gap", "start": 0, "end": 1}]
    else:
        right.seg.words = [{"word": " gap last", "start": 4, "end": 5}]
    result = check(emissions, spans)
    gaps = [t.speaker for t in result for p in t.pieces if p.text == " gap"]
    assert gaps == ["Jo"]
    assert any(p.rule.value == barrier for t in result for p in t.pieces)


def test_override_and_child_precedence() -> None:
    e = emission()
    spans = [Span(0, 0, 1, A), Span(1, 1, 2, B)]
    result = check([replace(e, seg_override=override(A))], spans)
    assert len(result[0].pieces) == 1 and result[0].speaker == "Alex"
    assert result[0].pieces[0].rule == PieceRule.SEGMENT_OVERRIDE
    child = DerivedChild(0, 1, 0.25, 0.75, "hello")
    for segment, ranged, name in [
        (None, None, "Jo"),
        (override(B), None, "Sam"),
        (override(B), override(A), "Alex"),
    ]:
        result = check(
            [replace(e, child=child, child_index=0, seg_override=segment, range_override=ranged)],
            spans,
        )
        assert result[0].speaker == name
        assert result[0].pieces[0].rule == PieceRule.SPLIT_CHILD
        assert not result[0].pieces[0].timed
        assert (result[0].pieces[0].start_seconds, result[0].pieces[0].end_seconds) == (0.25, 0.75)


def test_range_override_without_a_child_stays_coarse() -> None:
    # The walk never builds this (a range override rides on a split child), but
    # the pure function fails closed toward the operator ruling if it is handed one.
    e = replace(emission(), range_override=override(A))
    result = check([e], [Span(0, 0, 1, B), Span(1, 1, 2, C)])
    assert [t.speaker for t in result] == ["Alex"]
    assert [p.rule for p in result[0].pieces] == [PieceRule.SEGMENT_OVERRIDE]


def test_label_rulings_and_canonical_grouping() -> None:
    states = dict(STATES)
    states[B] = replace(states[B], resolution=Resolution.HUMAN_EXCLUDE)
    states[C] = replace(states[C], resolution=Resolution.HUMAN_UNKNOWN)
    result = check(
        [emission((" a", " b", " c", " d", " e"))],
        [Span(i, i, i + 1, label) for i, label in enumerate([A, B, A, C, A])],
        states=states,
    )
    assert [t.speaker for t in result] == [
        "Alex",
        "(excluded) Voice 2",
        "Alex",
        "Unknown (Voice 3)",
        "Alex",
    ]
    states[B] = replace(states[A], label=B, speaker_name="Sam")
    result = check([emission()], [Span(0, 0, 1, A), Span(1, 1, 2, B)], states=states)
    assert len(result) == 1 and result[0].speaker == "Alex"
    assert result[0].identity_key == ("speaker", str(states[A].speaker_id))
    e = emission(label=None)
    result = check([e], [])
    assert result[0].speaker == "(no speaker)"
    assert result[0].identity_key == ("label", "", "unresolved")
    result = check([emission()], [Span(0, 0, 3, "SPEAKER_03")])
    assert result[0].speaker == "Voice 4"


def test_whitespace_does_not_split_and_segment_start_join() -> None:
    first = emission(("can", "'t"), label=A)
    blank = emission((" ",), label=B)
    last = emission(("go",), start=3, label=A)
    result = check([first, blank, last], [])
    assert len(result) == 1
    assert join_pieces(result[0].pieces) == "can't go"
    assert [p.segment_start for p in result[0].pieces] == [True, True]
    result = check([emission(("hello ", "world"), label=A)], [])
    assert join_pieces(result[0].pieces) == "hello world"
    assert [p.segment_start for p in result[0].pieces] == [True, False]


def test_single_token_validation() -> None:
    e = emission(("hello",))
    assert validated_words(e.seg, min_words=1) is not None
    assert validated_words(e.seg, min_words=2) is None
    assert check([e], [Span(0, 0, 1, A)])[0].pieces[0].timed


@pytest.mark.parametrize("selected", ["world hello", "Hello changed", "hello", "hello world extra"])
def test_t2_refuses_substitution_reordering_and_count_changes(selected: str) -> None:
    result = check(
        [emission(enhanced=selected)],
        [Span(0, 0, 1, A), Span(1, 1, 2, B)],
        text=TranscriptText.ENHANCED,
    )
    assert result[0].speaker == "Jo"
    assert result[0].pieces[0].rule == PieceRule.UNMAPPED_TEXT


@pytest.mark.parametrize("resolution", list(Resolution))
@pytest.mark.parametrize("has_id", [False, True])
def test_label_identity_requires_assign_resolution_and_id(
    resolution: Resolution,
    has_id: bool,
) -> None:
    resolved = replace(
        STATES[A], resolution=resolution, speaker_id=STATES[A].speaker_id if has_id else None
    )
    result = check([emission((" hello",))], [Span(0, 0, 1, A)], states={A: resolved})
    assigned = resolution in (
        Resolution.HUMAN_ASSIGN,
        Resolution.GROUNDED_COSINE,
        Resolution.AUTO_ENROLL,
    )
    expected = (
        ("speaker", str(resolved.speaker_id))
        if has_id and assigned
        else ("label", A, resolution.value)
    )
    assert result[0].identity_key == expected


def test_unsupported_fallback_never_becomes_a_bracket() -> None:
    result = check(
        [
            emission((" edge",), label=A),
            emission((" gap",), start=1, label=B),
            emission((" supported",), start=2),
        ],
        [Span(0, 2, 3, A)],
    )
    assert [t.speaker for t in result] == ["Alex", "Sam", "Alex"]


def test_coarse_translation_children_and_counts() -> None:
    parent = emission()
    children = [
        replace(parent, child=DerivedChild(0, 1, 0, 1, "hello"), seg_override=override(A)),
        replace(parent, child=DerivedChild(1, 2, 1, 2, "world"),
                seg_override=override(A), range_override=override(B)),
    ]
    result = coarse_turns(children, ["Hola", "mundo"])
    assert [t.speaker for t in result] == ["Alex", "Sam"]
    assert all(p.rule == PieceRule.TRANSLATION and not p.timed and p.segment_start
               for t in result for p in t.pieces)
    assert [(p.start_seconds, p.end_seconds) for t in result for p in t.pieces] == [(0, 1), (1, 2)]
    for bodies in (["one"], ["one", "two", "three"]):
        with pytest.raises(ValueError):
            coarse_turns(iter(children), bodies)
    assert coarse_turns(children, ["", " "]) == []


@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("multiple_identities", [False, True])
def test_word_piece_anchors(edited: bool, multiple_identities: bool) -> None:
    e = emission(enhanced="\tHELLO, World!  \n")
    result = check(
        [e], [Span(0, 0, 1, A), Span(1, 1, 2, B if multiple_identities else A)],
        text=TranscriptText.ENHANCED if edited else TranscriptText.RAW,
    )
    pieces = [p for t in result for p in t.pieces]
    assert [p.rule for p in pieces] == [
        PieceRule.WORD_LEVEL if multiple_identities else PieceRule.SINGLE_IDENTITY,
    ] * 2
    assert [p.mapping for p in pieces] == [
        TextMapping.TOKEN if edited else TextMapping.VERBATIM,
    ] * 2
    assert [p.anchors for p in pieces] == [
        (WordAnchor(e.seg.id, 0, 1, 1, 7 if edited else 6),),
        (WordAnchor(e.seg.id, 1, 2, 1, 7 if edited else 6),),
    ]
    assert pieces[0] == replace(pieces[0], anchors=())
    assert hash(pieces[0]) == hash(replace(pieces[0], anchors=()))
    assert "anchors" not in repr(pieces[0])
    assert join_pieces(pieces) == join_pieces(replace(p, anchors=()) for p in pieces)
    assert to_markdown_turns(layout_turns(result)) == to_markdown_turns(layout_turns([
        replace(t, pieces=tuple(replace(p, anchors=()) for p in t.pieces)) for t in result
    ]))


def test_glued_and_trailing_whitespace_anchor_ranges() -> None:
    e = emission((" can", "'t", " re", "-", "enter", "!", " ", "\n"))
    result = check([e], [])
    assert [p.anchors for p in result[0].pieces] == [
        (WordAnchor(e.seg.id, 0, 2, 1, 6),),
        (WordAnchor(e.seg.id, 2, 8, 1, 10),),
    ]
    assert [(p.text, p.start_seconds, p.end_seconds) for p in result[0].pieces] == [
        (" can't", 0, 2), (" re-enter! \n", 2, 6),
    ]


def test_split_child_anchors_use_parent_ranges_and_outer_trim() -> None:
    parent = emission((" \t hello", " world  ", " \n again", " friend\t ", "\n"))
    children = derive_children(parent.seg, [2])
    assert children is not None
    emissions = [replace(parent, child=child, child_index=i) for i, child in enumerate(children)]
    result = check(emissions, [Span(0, 0, 5, A)])
    assert result[0].speaker == "Jo"
    pieces = result[0].pieces
    assert [(p.text, p.rule, p.mapping, p.timed) for p in pieces] == [
        ("hello world", PieceRule.SPLIT_CHILD, TextMapping.COARSE, False),
        ("again friend", PieceRule.SPLIT_CHILD, TextMapping.COARSE, False),
    ]
    assert [p.anchors for p in pieces] == [
        (WordAnchor(parent.seg.id, 0, 1, 0, 5), WordAnchor(parent.seg.id, 1, 2, 6, 11)),
        (WordAnchor(parent.seg.id, 2, 3, 0, 5), WordAnchor(parent.seg.id, 3, 5, 6, 12)),
    ]


@pytest.mark.parametrize("body", ["hello  world", "hello WORLD", "hello elsewhere"])
def test_split_child_anchor_mismatch_fails_closed(body: str) -> None:
    e = replace(emission(), child=DerivedChild(0, 2, 0, 2, body), child_index=0)
    piece = check([e], [])[0].pieces[0]
    assert piece.text == body and piece.rule is PieceRule.SPLIT_CHILD
    assert piece.anchors == ()


@pytest.mark.parametrize("edited", [False, True])
def test_segment_override_anchors_preserve_coarse_piece(edited: bool) -> None:
    e = replace(
        emission(enhanced="\n HELLO, World! \t" if edited else "\n hello world \t"),
        seg_override=override(B),
    )
    piece = check([e], [Span(0, 0, 2, A)], text=TranscriptText.ENHANCED)[0].pieces[0]
    assert piece == TurnPiece(
        e.seg.enhanced_text or "", 0, 3, False, True,
        PieceRule.SEGMENT_OVERRIDE, TextMapping.COARSE,
    )
    assert piece.anchors == (
        WordAnchor(e.seg.id, 0, 1, 2, 8 if edited else 7),
        WordAnchor(e.seg.id, 1, 2, 9 if edited else 8, 15 if edited else 13),
    )


@pytest.mark.parametrize("reason", ["unmapped", "invalid", "mismatch", "blank"])
def test_segment_override_without_anchor_evidence(reason: str) -> None:
    e = replace(emission(enhanced="A different sentence."), seg_override=override(A))
    if reason == "invalid":
        e.seg.words = [{"word": "hello"}]
    elif reason == "mismatch":
        # The selected text maps onto the tokens, so only the guard that the
        # tokens reconcatenate to raw_text can refuse the anchors.
        e.seg.enhanced_text = "hello world"
        e.seg.raw_text = "different raw text"
    elif reason == "blank":
        e.seg.words = [{"word": " ", "start": 0, "end": 1}]
        e.seg.raw_text = " "
    piece = check([e], [], text=TranscriptText.ENHANCED)[0].pieces[0]
    assert piece.rule is PieceRule.SEGMENT_OVERRIDE and piece.anchors == ()


@pytest.mark.parametrize(
    "rule", [PieceRule.NO_WORDS, PieceRule.UNMAPPED_TEXT, PieceRule.SINGLE_IDENTITY],
)
def test_unanchored_coarse_rules(rule: PieceRule) -> None:
    e = emission(enhanced="A different sentence.")
    if rule is PieceRule.NO_WORDS:
        e.seg.words = None
    spans = [Span(0, 0, 1, A), Span(1, 1, 2, A if rule is PieceRule.SINGLE_IDENTITY else B)]
    piece = check([e], spans, text=TranscriptText.ENHANCED)[0].pieces[0]
    assert piece.rule is rule and piece.mapping is TextMapping.COARSE
    assert piece.anchors == ()
    translated = coarse_turns([e], ["Hola mundo"])
    assert translated[0].pieces[0].anchors == ()


@pytest.mark.parametrize("barrier", [PieceRule.SPLIT_CHILD, PieceRule.SEGMENT_OVERRIDE])
@pytest.mark.parametrize("side", ["left", "right"])
def test_anchor_units_preserve_coarse_barrier_pieces(barrier: PieceRule, side: str) -> None:
    middle = emission((" barrier",), start=2)
    middle = (
        replace(middle, child=DerivedChild(0, 1, 2, 3, "barrier"), child_index=0)
        if barrier is PieceRule.SPLIT_CHILD
        else replace(middle, seg_override=override(B))
    )
    left = emission((" one", " gap"))
    right = emission((" gap", " last"), start=3)
    if side == "left":
        left.seg.words = [{"word": " one gap", "start": 0, "end": 1}]
        expected_left = [("Alex", TurnPiece(
            " one gap", 0, 1, True, True, PieceRule.SINGLE_IDENTITY, TextMapping.VERBATIM,
        ))]
        expected_right = [
            ("Jo", TurnPiece(" gap", 3, 4, True, True, PieceRule.WORD_LEVEL, TextMapping.VERBATIM)),
            ("Alex", TurnPiece(
                " last", 4, 5, True, False, PieceRule.WORD_LEVEL, TextMapping.VERBATIM,
            )),
        ]
    else:
        right.seg.words = [{"word": " gap last", "start": 4, "end": 5}]
        expected_left = [
            ("Alex", TurnPiece(
                " one", 0, 1, True, True, PieceRule.WORD_LEVEL, TextMapping.VERBATIM,
            )),
            ("Jo", TurnPiece(
                " gap", 1, 2, True, False, PieceRule.WORD_LEVEL, TextMapping.VERBATIM,
            )),
        ]
        expected_right = [("Alex", TurnPiece(
            " gap last", 4, 5, True, True, PieceRule.SINGLE_IDENTITY, TextMapping.VERBATIM,
        ))]
    expected_middle = ("Jo" if barrier is PieceRule.SPLIT_CHILD else "Sam", TurnPiece(
        "barrier" if barrier is PieceRule.SPLIT_CHILD else " barrier",
        2, 3 if barrier is PieceRule.SPLIT_CHILD else 4, False, True, barrier, TextMapping.COARSE,
    ))
    result = check([left, middle, right], [Span(0, 0, 1, A), Span(1, 4, 5, A)])
    assert [(t.speaker, p) for t in result for p in t.pieces] == [
        *expected_left, expected_middle, *expected_right,
    ]
    assert next(p for t in result for p in t.pieces if p.rule is barrier).anchors


def test_split_cut_inside_a_glued_word_anchors_each_child_part() -> None:
    """A cut may fall between glued tokens; each child anchors its own part."""
    parent = emission((" can", "'t", " go"))
    children = derive_children(parent.seg, [1])
    assert children is not None
    emissions = [replace(parent, child=child, child_index=i) for i, child in enumerate(children)]
    pieces = [p for t in check(emissions, [Span(0, 0, 4, A)]) for p in t.pieces]
    assert [(p.text, p.anchors) for p in pieces] == [
        ("can", (WordAnchor(parent.seg.id, 0, 1, 0, 3),)),
        ("'t go", (
            WordAnchor(parent.seg.id, 1, 2, 0, 2), WordAnchor(parent.seg.id, 2, 3, 3, 5),
        )),
    ]


def test_segment_override_anchors_under_a_review_correction() -> None:
    e = replace(
        emission(), seg_override=override(B),
        review=SegmentReviewState(corrected_text="Hello, WORLD."),
    )
    piece = check([e], [Span(0, 0, 2, A)], text=TranscriptText.CORRECTED)[0].pieces[0]
    assert piece.rule is PieceRule.SEGMENT_OVERRIDE and piece.text == "Hello, WORLD."
    assert piece.anchors == (
        WordAnchor(e.seg.id, 0, 1, 0, 6), WordAnchor(e.seg.id, 1, 2, 7, 13),
    )
