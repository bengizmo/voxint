"""Speaker turns at word grain: a read-time projection of the attribution walk.

A whisper segment carries ONE diarization label (the turn it overlaps most), so
a 30 second segment that holds two people renders under one name. The per-word
timings and the full turn ledger are both stored; this module joins them at read
time. Nothing stored is mutated, and every operator ruling still wins: the
projection composes with :func:`walk_attributions`, it never re-derives it.

Per emission, first match wins:

* **E1** a derived split child (an operator cut) is one coarse piece under the
  child's winning attribution. It is never subdivided.
* **E2** a whole-segment override is one coarse piece under the override.
* **E3** no usable words (``words`` NULL or empty, structural validation fails,
  the tokens do not reconcatenate to ``raw_text``, or they hold no text) is one
  coarse piece under the existing attribution.
* **E4** every word unit resolves to one identity: that identity, as word pieces
  when the selected text maps onto the units, else as one coarse piece.
* **E5** two or more identities and the text maps: word-level pieces.
* **E6** two or more identities and the text does not map: one coarse piece
  under the existing attribution, which is what the segment-level view shows.

**Word units.** Tokens are grouped so a token with no leading whitespace stays
glued to the one before it; a speaker change can never land inside a word. In
the normal whisper case (every token has a leading space) units equal tokens.

**Unit to identity.** The diarization turn with the greatest positive
intersection, ties to the lowest ``turn_index`` (the ``_dominant_label`` rule
the pipeline uses per segment). A zero-length unit uses point containment in
``[start, end)``. A unit inside no turn takes the emission's existing
attribution, unless the nearest turn-supported units on both sides share one
identity, in which case it takes that identity. That search may cross a segment
boundary but stops at an E1, E2 or E3 emission, and an inferred unit is never
evidence for another. There is no smoothing: a one-word turn is the feature.

**Text to units.** Timings belong to ``raw_text``. **T1**: the selected text
equals the joined tokens apart from outer whitespace, so pieces are the units
verbatim. **T2**: the selected text splits on whitespace into as many tokens as
there are units and each pair agrees on a normalized key (case, quote and dash
variants, edge punctuation), which covers punctuation and casing edits. Anything
else is unmapped. There is no sequence alignment and no positional guess.

Pieces keep walk order (segment, then word): no global sort, no deduplication,
no clamping. Consecutive pieces of one identity form a turn, across segment
boundaries, grouped by identity and never by display string.
"""

import enum
import re
import unicodedata
import uuid
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.attribution import (
    AttributionScope,
    Emission,
    walk_attributions,
    winning_attribution,
)
from voxint.adjudication.resolver import LabelState, Resolution, label_states
from voxint.adjudication.splits import WordToken, reconcatenates, validated_words
from voxint.adjudication.transcript import (
    TranscriptText,
    label_display_name,
    resolve_body,
    segment_speaker,
)
from voxint.db.models import DiarizationTurn


class PieceRule(enum.StrEnum):
    """The first matching per-emission rule that produced a piece."""

    SPLIT_CHILD = "E1"
    SEGMENT_OVERRIDE = "E2"
    NO_WORDS = "E3"
    SINGLE_IDENTITY = "E4"
    WORD_LEVEL = "E5"
    UNMAPPED_TEXT = "E6"
    TRANSLATION = "translation"


class TextMapping(enum.StrEnum):
    """Verbatim tokens, equivalent edited tokens, or an unmapped coarse body."""

    VERBATIM = "T1"
    TOKEN = "T2"
    COARSE = "coarse"


WordMarkKey = tuple[uuid.UUID, int, int]
EffectiveMarks = Mapping[WordMarkKey, Literal["keep", "omit"]]


@dataclass(frozen=True)
class WordAnchor:
    """An input-only lexical coordinate on a piece produced by project_turns.

    Any later rewrite of piece.text invalidates these offsets, including the
    dataclasses.replace calls in fillers and repeats, so a filter that needs
    them must convert them to per-character source identities before it
    rewrites anything. Rendering never reads them.
    """

    segment_id: uuid.UUID
    token_start: int
    token_end: int
    lex_start: int
    lex_end: int


@dataclass(frozen=True)
class EmissionAnchors:
    """Lexical code-point offsets into the emission's shown text."""

    anchors: tuple[WordAnchor, ...]
    reason: str | None


@dataclass(frozen=True)
class TurnPiece:
    """Verbatim text with word timing, or a coarse emission interval.

    ``segment_start`` marks where the segment boundary join rule applies.
    Leading whitespace belongs to the piece and is never stripped.
    """

    text: str
    start_seconds: float
    end_seconds: float
    timed: bool
    segment_start: bool
    rule: PieceRule
    mapping: TextMapping
    anchors: tuple[WordAnchor, ...] = field(default=(), compare=False, repr=False)


IdentityKey = tuple[str, ...]
_Identity = tuple[IdentityKey, str]


@dataclass(frozen=True)
class SpeakerTurn:
    """Consecutive pieces of one identity, named by its first piece."""

    identity_key: IdentityKey
    speaker: str
    pieces: tuple[TurnPiece, ...]


class TurnSpan(Protocol):
    """The read-only timing columns needed from a diarization turn."""

    @property
    def turn_index(self) -> int: ...

    @property
    def start_seconds(self) -> float: ...

    @property
    def end_seconds(self) -> float: ...

    @property
    def label(self) -> str: ...


def join_pieces(pieces: Iterable[TurnPiece]) -> str:
    """Join verbatim, adding a space only at an unspaced emission boundary."""
    out: list[str] = []
    for piece in pieces:
        if not piece.text:
            continue
        if (
            piece.segment_start
            and out
            and not out[-1][-1].isspace()
            and not piece.text[0].isspace()
        ):
            out.append(" ")
        out.append(piece.text)
    return "".join(out)


def _label_identity(state: LabelState | None, label: str | None) -> _Identity:
    """Mirror the label branch of winning_attribution with display annotations."""
    key: IdentityKey
    resolution = state.resolution if state else Resolution.UNRESOLVED
    if (
        state is not None
        and state.speaker_id is not None
        and resolution
        in (Resolution.HUMAN_ASSIGN, Resolution.GROUNDED_COSINE, Resolution.AUTO_ENROLL)
    ):
        key = ("speaker", str(state.speaker_id))
    else:
        key = ("label", label or "", resolution.value)
    return key, label_display_name(state, label)


def _existing_identity(emission: Emission) -> _Identity:
    """Use range > segment > label precedence for an emission's fallback."""
    scope, _, speaker_id, _ = winning_attribution(emission)
    if scope is AttributionScope.LABEL:
        return _label_identity(emission.label_state, emission.seg.diarization_label)
    override = emission.range_override or emission.seg_override
    if override is None:
        raise RuntimeError("an override scope won without an override on the emission")
    return ("speaker", str(speaker_id)), segment_speaker(override, emission.seg)


@dataclass(frozen=True)
class _Unit:
    """Verbatim grouped tokens with a half-open range in the parent's words."""

    start: float
    end: float
    text: str
    token_start: int
    token_end: int


def _word_units(words: list[WordToken], *, token_start: int = 0) -> list[_Unit]:
    """Keep glued tokens together and retain leading and trailing whitespace.

    A unit's text is every constituent token verbatim. Its interval is the
    envelope of the tokens that carry text: validation only orders token starts,
    so a glued token can end before the word it attaches to, and a
    whitespace-only token's timing says nothing about who spoke the word.
    """
    groups: list[list[WordToken]] = []
    has_text = False
    for word in words:
        if not groups or (
            has_text and (word.text[0].isspace() or groups[-1][-1].text[-1].isspace())
        ):
            groups.append([])
            has_text = False
        groups[-1].append(word)
        has_text = has_text or bool(word.text.strip())
    if len(groups) > 1 and not has_text:
        groups[-2].extend(groups.pop())
    units: list[_Unit] = []
    for group in groups:
        spoken = [w for w in group if w.text.strip()] or group
        units.append(
            _Unit(
                spoken[0].start,
                max(w.end for w in spoken),
                "".join(w.text for w in group),
                token_start,
                token_start + len(group),
            )
        )
        token_start += len(group)
    return units


class _TurnLookup:
    """Bisect starts and prefix maximum ends, independent of unit time order."""

    def __init__(self, turns: Sequence[TurnSpan]) -> None:
        self.turns = sorted(turns, key=lambda t: t.start_seconds)
        self.starts = [t.start_seconds for t in self.turns]
        self.ends: list[float] = []
        maximum = float("-inf")
        for turn in self.turns:
            maximum = max(maximum, turn.end_seconds)
            self.ends.append(maximum)

    def label(self, unit: _Unit) -> str | None:
        """Select positive overlap, or half-open containment for a point unit."""
        point = unit.start == unit.end
        lo = bisect_right(self.ends, unit.start)
        hi = bisect_right(self.starts, unit.start) if point else bisect_left(self.starts, unit.end)
        best: TurnSpan | None = None
        score = -1.0
        for i in range(lo, hi):
            turn = self.turns[i]
            overlap = min(unit.end, turn.end_seconds) - max(unit.start, turn.start_seconds)
            if not (turn.start_seconds <= unit.start < turn.end_seconds if point else overlap > 0):
                continue
            if (
                best is None
                or overlap > score
                or (overlap == score and turn.turn_index < best.turn_index)
            ):
                best, score = turn, overlap
        return best.label if best is not None else None


def _token_key(text: str) -> str:
    """Normalize case, quotes and Unicode hyphens, then trim edge punctuation."""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.translate(
        str.maketrans("\u2018\u2019\u201c\u201d", "''\"\"")
    )
    value = "".join("-" if unicodedata.category(c) == "Pd" else c for c in value)
    lo, hi = 0, len(value)
    while lo < hi and not value[lo].isalnum():
        lo += 1
    while hi > lo and not value[hi - 1].isalnum():
        hi -= 1
    return value[lo:hi]


def _map_text(selected: str, units: list[_Unit]) -> tuple[TextMapping, list[str]]:
    """Apply T1 or conservative T2; no positional substitution or alignment."""
    if selected.strip() == "".join(u.text for u in units).strip():
        return TextMapping.VERBATIM, [u.text for u in units]
    matches = list(re.finditer(r"\s*\S+", selected))
    texts = [m.group() for m in matches]
    if texts:
        texts[-1] += selected[matches[-1].end() :]
    if len(texts) == len(units) and all(
        _token_key(t.strip()) and _token_key(t.strip()) == _token_key(u.text.strip())
        for t, u in zip(texts, units, strict=True)
    ):
        return TextMapping.TOKEN, texts
    return TextMapping.COARSE, []


def _word_anchors(
    segment_id: uuid.UUID, body: str, units: list[_Unit], texts: list[str],
) -> tuple[WordAnchor, ...]:
    """Walk mapped text, allowing only the owning piece's outer whitespace delta."""
    joined = "".join(texts)
    if joined.strip() != body.strip():
        return ()
    position = len(body) - len(body.lstrip()) - (len(joined) - len(joined.lstrip()))
    anchors: list[WordAnchor] = []
    for unit, rendered in zip(units, texts, strict=True):
        lexical = rendered.strip()
        lo = position + len(rendered) - len(rendered.lstrip())
        hi = lo + len(lexical)
        if not lexical or not 0 <= lo < hi <= len(body) or body[lo:hi] != lexical:
            return ()
        anchors.append(WordAnchor(segment_id, unit.token_start, unit.token_end, lo, hi))
        position += len(rendered)
    return tuple(anchors)


def _usable_words(words: list[WordToken], raw_text: str) -> bool:
    """The E3 test: tokens that reconcatenate to the raw text and hold text."""
    return reconcatenates(words, raw_text) and any(w.text.strip() for w in words)


def _coarse_anchors(emission: Emission, selected: str) -> tuple[WordAnchor, ...]:
    """Prepare E1/E2 provenance without adding attribution evidence or brackets."""
    words = validated_words(emission.seg, min_words=1)
    if words is None:
        return ()
    child = emission.child
    if child is not None:
        if not 0 <= child.word_start < child.word_end <= len(words):
            return ()
        units = _word_units(words[child.word_start:child.word_end], token_start=child.word_start)
        texts = [unit.text for unit in units]
    else:
        if not _usable_words(words, emission.seg.raw_text):
            return ()
        units = _word_units(words)
        mapping, texts = _map_text(selected, units)
        if mapping is TextMapping.COARSE:
            return ()
    return _word_anchors(emission.seg.id, selected, units, texts)


def _selected_text(emission: Emission, text: TranscriptText) -> str:
    return (
        emission.child.text if emission.child is not None else resolve_body(
            emission.seg, emission.review.corrected_text if emission.review else None, text
        )
    )


def emission_anchors(emission: Emission, *, text: TranscriptText) -> EmissionAnchors:
    """Mirror projection provenance, with offsets in the whole shown emission."""
    selected = _selected_text(emission, text)
    words = validated_words(emission.seg, min_words=1)
    no_timings = "This segment has no recorded word timings, so its words cannot be marked."
    changed = (
        "Clean-up marks need the text to match the recorded words. "
        "This segment's text was changed too much."
    )
    if (
        emission.child is not None or emission.seg_override is not None
        or emission.range_override is not None
    ):
        anchors = _coarse_anchors(emission, selected)
    elif words is None or not _usable_words(words, emission.seg.raw_text):
        return EmissionAnchors((), no_timings)
    else:
        units = _word_units(words)
        mapping, texts = _map_text(selected, units)
        if mapping is TextMapping.COARSE:
            return EmissionAnchors((), changed)
        anchors = _word_anchors(emission.seg.id, selected, units, texts)
    return EmissionAnchors(anchors, None if anchors else changed if words else no_timings)


@dataclass
class _EmissionUnits:
    """Prepared emission; supported evidence remains separate from inference."""

    emission: Emission
    selected: str
    existing: _Identity
    coarse_rule: PieceRule | None
    units: list[_Unit]
    supported: list[_Identity | None]
    identities: list[_Identity]


def project_turns(
    emissions: Iterable[Emission],
    turns: Sequence[TurnSpan],
    states: Mapping[str, LabelState],
    *,
    text: TranscriptText,
) -> list[SpeakerTurn]:
    """Apply E1-E6 and T1-T2 in walk order, grouping by canonical identity.

    Supported brackets may cross emissions but never E1/E2/E3 barriers. Inferred
    units never become evidence. Empty pieces do not split turns. Timings are
    retained as supplied, without sorting, deduplicating or clamping pieces.
    """
    lookup = _TurnLookup(turns)
    prepared: list[_EmissionUnits] = []
    for emission in emissions:
        seg, child = emission.seg, emission.child
        selected = _selected_text(emission, text)
        existing = _existing_identity(emission)
        rule: PieceRule | None = None
        words = validated_words(seg, min_words=1)
        if child is not None:
            rule = PieceRule.SPLIT_CHILD
        elif emission.seg_override is not None or emission.range_override is not None:
            # The walk only attaches a range override to a split child (E1).
            # Should one ever arrive on a whole segment, it is still an
            # operator ruling: stay coarse under it, never subdivide.
            rule = PieceRule.SEGMENT_OVERRIDE
        elif words is None or not _usable_words(words, seg.raw_text):
            rule = PieceRule.NO_WORDS
        units = _word_units(words) if rule is None and words is not None else []
        supported = []
        for unit in units:
            label = lookup.label(unit)
            supported.append(
                _label_identity(states.get(label), label) if label is not None else None
            )
        prepared.append(
            _EmissionUnits(
                emission,
                selected,
                existing,
                rule,
                units,
                supported,
                [s if s is not None else existing for s in supported],
            )
        )

    # Two passes record nearest original evidence only, never inferred brackets.
    previous: _Identity | None = None
    left: dict[tuple[int, int], _Identity | None] = {}
    for i, item in enumerate(prepared):
        if item.coarse_rule is not None:
            previous = None
        for j, supported_identity in enumerate(item.supported):
            left[i, j] = previous
            if supported_identity is not None:
                previous = supported_identity
    following: _Identity | None = None
    for i in range(len(prepared) - 1, -1, -1):
        item = prepared[i]
        if item.coarse_rule is not None:
            following = None
        for j in range(len(item.units) - 1, -1, -1):
            supported_identity = item.supported[j]
            preceding = left[i, j]
            if supported_identity is not None:
                following = supported_identity
            elif preceding is not None and following is not None and preceding[0] == following[0]:
                item.identities[j] = preceding

    all_pieces: list[tuple[_Identity, TurnPiece]] = []
    for item in prepared:
        mapping, texts = (
            _map_text(item.selected, item.units) if item.units else (TextMapping.COARSE, [])
        )
        rule = item.coarse_rule
        identity = item.existing
        if rule is None:
            if len({identity[0] for identity in item.identities}) == 1:
                rule = PieceRule.SINGLE_IDENTITY
                identity = item.identities[0]
            elif mapping is not TextMapping.COARSE:
                rule = PieceRule.WORD_LEVEL
            else:
                rule = PieceRule.UNMAPPED_TEXT
        emitted: list[tuple[_Identity, TurnPiece]] = []
        if mapping is not TextMapping.COARSE:
            for j, (unit, body) in enumerate(zip(item.units, texts, strict=True)):
                emitted.append(
                    (
                        item.identities[j],
                        TurnPiece(
                            body, unit.start, unit.end, True, j == 0, rule, mapping,
                            _word_anchors(item.emission.seg.id, body, [unit], [body]),
                        ),
                    )
                )
        else:
            seg, child = item.emission.seg, item.emission.child
            emitted.append(
                (
                    identity,
                    TurnPiece(
                        item.selected,
                        child.start_seconds if child else seg.start_seconds,
                        child.end_seconds if child else seg.end_seconds,
                        False,
                        True,
                        rule,
                        TextMapping.COARSE,
                        _coarse_anchors(item.emission, item.selected)
                        if rule in (PieceRule.SPLIT_CHILD, PieceRule.SEGMENT_OVERRIDE)
                        else (),
                    ),
                )
            )
        all_pieces.extend(emitted)
    return _group_pieces(all_pieces)


def _group_pieces(emitted: Iterable[tuple[_Identity, TurnPiece]]) -> list[SpeakerTurn]:
    """Skip empty pieces and group adjacent canonical identities."""
    groups: list[tuple[_Identity, list[TurnPiece]]] = []
    for identity, piece in emitted:
        if not piece.text.strip():
            continue
        if not groups or groups[-1][0][0] != identity[0]:
            groups.append((identity, []))
        groups[-1][1].append(piece)
    return [SpeakerTurn(identity[0], identity[1], tuple(pieces)) for identity, pieces in groups]


def coarse_turns(emissions: Iterable[Emission], texts: Sequence[str]) -> list[SpeakerTurn]:
    """Project translated text by emission order, retaining split-child rulings.

    Translation bypasses word evidence. Counts must agree, including empty
    translations; each non-empty body is one atomic coarse piece.
    """
    pieces: list[tuple[_Identity, TurnPiece]] = []
    for emission, body in zip(emissions, texts, strict=True):
        interval = emission.child if emission.child is not None else emission.seg
        pieces.append((_existing_identity(emission), TurnPiece(
            body, interval.start_seconds, interval.end_seconds, False, True,
            PieceRule.TRANSLATION, TextMapping.COARSE,
        )))
    return _group_pieces(pieces)


def translated_turns(
    session: Session, run_id: uuid.UUID, texts: Sequence[str],
) -> list[SpeakerTurn]:
    """Load the attribution walk for a translation without loading word turns."""
    return coarse_turns(walk_attributions(session, run_id), texts)


def attributed_turns(
    session: Session,
    run_id: uuid.UUID,
    *,
    text: TranscriptText,
) -> list[SpeakerTurn]:
    """Load label states once and only turn timing columns, never embeddings."""
    states, turns, emissions = load_turn_inputs(session, run_id)
    return project_turns(emissions, turns, states, text=text)


def load_turn_inputs(
    session: Session, run_id: uuid.UUID,
) -> tuple[dict[str, LabelState], Sequence[TurnSpan], list[Emission]]:
    """Materialise one attribution walk for projection and emission consumers."""
    states = {s.label: s for s in label_states(session, run_id)}
    turns = session.execute(
        select(
            DiarizationTurn.turn_index,
            DiarizationTurn.start_seconds,
            DiarizationTurn.end_seconds,
            DiarizationTurn.label,
        )
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .order_by(DiarizationTurn.turn_index)
    ).all()
    return states, turns, list(walk_attributions(session, run_id, states=states))
