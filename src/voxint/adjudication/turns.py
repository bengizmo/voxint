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
from dataclasses import dataclass
from typing import Protocol

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


class TextMapping(enum.StrEnum):
    """Verbatim tokens, equivalent edited tokens, or an unmapped coarse body."""

    VERBATIM = "T1"
    TOKEN = "T2"
    COARSE = "coarse"


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
    assert override is not None
    return ("speaker", str(speaker_id)), segment_speaker(override, emission.seg)


def _word_units(words: list[WordToken]) -> list[WordToken]:
    """Keep glued tokens together and retain leading and trailing whitespace."""
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
    return [WordToken(g[0].start, g[-1].end, "".join(w.text for w in g)) for g in groups]


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

    def label(self, unit: WordToken) -> str | None:
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


def _map_text(selected: str, units: list[WordToken]) -> tuple[TextMapping, list[str]]:
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


@dataclass
class _EmissionUnits:
    """Prepared emission; supported evidence remains separate from inference."""

    emission: Emission
    selected: str
    existing: _Identity
    coarse_rule: PieceRule | None
    units: list[WordToken]
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
        selected = (
            child.text
            if child
            else resolve_body(
                seg, emission.review.corrected_text if emission.review else None, text
            )
        )
        existing = _existing_identity(emission)
        rule: PieceRule | None = None
        words = validated_words(seg, min_words=1)
        if child is not None:
            rule = PieceRule.SPLIT_CHILD
        elif emission.seg_override is not None:
            rule = PieceRule.SEGMENT_OVERRIDE
        elif (
            words is None
            or not reconcatenates(words, seg.raw_text)
            or not any(w.text.strip() for w in words)
        ):
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

    groups: list[tuple[_Identity, list[TurnPiece]]] = []
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
                        TurnPiece(body, unit.start, unit.end, True, j == 0, rule, mapping),
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
                    ),
                )
            )
        for identity, piece in emitted:
            if not piece.text.strip():
                continue
            if not groups or groups[-1][0][0] != identity[0]:
                groups.append((identity, []))
            groups[-1][1].append(piece)
    return [SpeakerTurn(identity[0], identity[1], tuple(pieces)) for identity, pieces in groups]


def attributed_turns(
    session: Session,
    run_id: uuid.UUID,
    *,
    text: TranscriptText,
) -> list[SpeakerTurn]:
    """Load label states once and only turn timing columns, never embeddings."""
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
    return project_turns(
        walk_attributions(session, run_id, states=states), turns, states, text=text
    )
