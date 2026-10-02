"""Synthetic projection fixtures with shared emission and run round trips."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from voxint.adjudication.attribution import Emission
from voxint.adjudication.resolver import LabelState, Resolution, SegmentOverride
from voxint.adjudication.splits import DerivedChild, validated_words
from voxint.adjudication.transcript import TranscriptText, _join_segment_texts, resolve_body
from voxint.adjudication.turns import (
    PieceRule,
    SpeakerTurn,
    TextMapping,
    join_pieces,
    project_turns,
)
from voxint.db.models import AdjudicationDecision, SegmentReviewState, TranscriptSegment


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


def check(
    emissions: list[Emission],
    spans: Sequence[Span],
    *,
    text: TranscriptText = TranscriptText.RAW,
    states: dict[str, LabelState] | None = None,
) -> list[SpeakerTurn]:
    """Every projection fixture verifies each emission and the full run."""
    result = project_turns(iter(emissions), spans, STATES if states is None else states, text=text)
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
    assert " ".join(join_pieces(pieces).split()) == " ".join(_join_segment_texts(bodies).split())
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
