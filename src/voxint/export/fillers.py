"""Render-time English filler removal, before the Markdown reading layout.

* F1: standalone entries of the effective filler list (by default um, uh,
  umm, uhh, uhm and erm), ignoring case. Opening quotes and brackets before
  the word are kept; a word directly followed by a closing quote is being
  quoted and stays, as do compounds, possessives and ellipses.
* F1p: multi-word phrases are removed first, and only when clause
  punctuation or a turn edge sets them off on both sides. Tag questions,
  whole sentences and phrases spanning an F5 seam stay.
* F2: remove the word and its following comma, semicolon or colon. Move a
  terminal mark to preceding text, replacing comma/semicolon/colon, or drop
  it at turn start or after another terminal mark. A preceding comma stays.
* F3: collapse only separators beside a removal, to one between survivors
  and none at turn start. Respect the segment boundary rule of join_pieces.
* F4: after a sentence-start filler, capitalise the next lowercase initial,
  skipping leading quotes/brackets. Other casing stays unchanged.
* F5: discard turns made only of fillers (nothing with a letter or digit
  left), then merge adjacent surviving canonical identities with exactly one
  separator at the seam. The merged turn is cleaned as one, so a filler at
  the seam is judged against the text that now precedes it.
* F6: leave no-filler text unchanged, and retain every surviving piece's
  timing and flags. Rewrites clear stale anchors, never stored transcript data.

Removal spans carry lexical and separator ranges. Characters retain their
input anchor identities through rewrites; final F5 cleans emit a removal trace.
Trace order is per output group: merged turns at their first input position,
discarded filler-only turns at their own position, omits then phrases then
words within a group. Consumers must key by source identity, not by position.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from voxint.adjudication.turns import EffectiveMarks, SpeakerTurn, TurnPiece, WordMarkKey
from voxint.export.filler_lists import TIER_1, FillerList

_OPEN = "\"'([\u201c\u2018"
_CLOSE = "\"')]\u201d\u2019"


@lru_cache(maxsize=32)
def _word_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(
        words,
        key=lambda word: (
            -len(word),
            TIER_1.index(word) if word in TIER_1 else len(TIER_1),
            word,
        ),
    )
    return _pattern("|".join(re.escape(word) for word in ordered))


@lru_cache(maxsize=32)
def _phrase_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    # Alternation is first-match: a rejected shorter prefix would hide the longer phrase.
    ordered = sorted(
        phrases, key=lambda phrase: (-len(phrase.split()), -len(phrase), phrase.casefold(), phrase)
    )
    return _pattern(
        "|".join(r"\s+".join(re.escape(word) for word in phrase.split()) for phrase in ordered)
    )


def _pattern(alternation: str) -> re.Pattern[str]:
    return re.compile(
        r"""(?<!\S)(["'(\[\u201c\u2018]*)(?:""" + alternation + ")"
        r"""(?:(?P<mark>[,.?!;:])(?=$|\s)|(?=$|\s))\s*""",
        re.IGNORECASE,
    )


# Applied to text with surrounding quotes and brackets already stripped.
_SENTENCE_END = re.compile(r"[.?!]$")
_WRAPPERS = _OPEN + _CLOSE + " "
SourceIdentity = WordMarkKey
RemovalCause = Literal["phrase", "filler", "omit"]


@dataclass(frozen=True)
class Removal:
    """A committed lexical removal, with distinct input units in text order."""

    cause: RemovalCause
    text: str
    sources: tuple[SourceIdentity, ...]


@dataclass(frozen=True)
class Protected:
    """An otherwise committed match blocked by a kept lexical unit."""

    text: str
    sources: tuple[SourceIdentity, ...]
    cause: Literal["kept"] = "kept"


class UnplaceableWordMarkError(ValueError):
    """Effective marks have no anchored characters in the requested rendering."""

    def __init__(self, marks: EffectiveMarks) -> None:
        self.marks = [(*key, action) for key, action in marks.items()]
        super().__init__(f"cannot place word marks on segments: {self.marks}")


@dataclass(frozen=True, slots=True)
class _Char:
    text: str
    owner: int
    source: SourceIdentity | None = None


@dataclass(frozen=True, slots=True)
class _Piece:
    index: int
    piece: TurnPiece
    chars: tuple[_Char, ...]
    original: TurnPiece
    original_chars: tuple[_Char, ...]


@dataclass(frozen=True, slots=True)
class _RemovalSpan:
    """Opening [start, lex_start), lexical range, then consumed mark/separator."""

    start: int
    end: int
    lex_start: int
    lex_end: int
    mark_start: int | None
    cause: RemovalCause


@dataclass(frozen=True, slots=True)
class _Turn:
    turn: SpeakerTurn
    pieces: tuple[_Piece, ...]
    owners: tuple[int, ...]
    survives: bool


def _track(pieces: tuple[TurnPiece, ...]) -> tuple[_Piece, ...]:
    """Read anchors once, before any seam or filter rewrites text."""
    tracked: list[_Piece] = []
    for index, piece in enumerate(pieces):
        identities: list[SourceIdentity | None] = [None] * len(piece.text)
        for anchor in piece.anchors:
            source = (anchor.segment_id, anchor.token_start, anchor.token_end)
            for offset in range(anchor.lex_start, anchor.lex_end):
                if not piece.text[offset].isspace():
                    identities[offset] = source
        chars = tuple(
            _Char(char, index, source)
            for char, source in zip(piece.text, identities, strict=True)
        )
        tracked.append(_Piece(index, piece, chars, piece, chars))
    return tuple(tracked)


def _with_chars(piece: _Piece, chars: Sequence[_Char]) -> _Piece:
    text = "".join(char.text for char in chars)
    chars = tuple(chars)
    original_sources = (
        text == piece.original.text
        and (chars == piece.original_chars or all(
            char.source == original.source
            for char, original in zip(chars, piece.original_chars, strict=True)
        ))
    )
    anchors = piece.original.anchors if original_sources else ()
    if chars == piece.chars and anchors == piece.piece.anchors:
        return piece
    rendered = replace(piece.piece, text=text, anchors=anchors)
    return _Piece(piece.index, rendered, chars, piece.original, piece.original_chars)


def _flatten(pieces: tuple[_Piece, ...]) -> list[_Char]:
    chars: list[_Char] = []
    for tracked in pieces:
        piece = tracked.piece
        if not piece.text:
            continue
        if (
            piece.segment_start
            and chars
            and not chars[-1].text.isspace()
            and not piece.text[0].isspace()
        ):
            chars.append(_Char(" ", tracked.index))
        chars.extend(tracked.chars)
    return chars


def _removal_spans(body: str, pattern: re.Pattern[str], cause: RemovalCause) -> list[_RemovalSpan]:
    spans: list[_RemovalSpan] = []
    for match in pattern.finditer(body):
        mark_start = match.start("mark") if match.group("mark") is not None else None
        spans.append(_RemovalSpan(
            match.start(), match.end(), match.start() + len(match.group(1)),
            mark_start if mark_start is not None else match.start() + len(match.group(0).rstrip()),
            mark_start, cause,
        ))
    return spans


def _omit_spans(chars: Sequence[_Char], marks: EffectiveMarks) -> list[_RemovalSpan]:
    """Locate omitted units before rewrites, retaining their surrounding wrappers."""
    positions: dict[WordMarkKey, list[int]] = {}
    for index, char in enumerate(chars):
        if char.source is not None and marks.get(char.source) == "omit":
            positions.setdefault(char.source, []).append(index)
    spans: list[_RemovalSpan] = []
    for offsets in positions.values():
        start, stop = offsets[0], offsets[-1] + 1
        lex_start = start
        while lex_start < stop and chars[lex_start].text in _OPEN:
            lex_start += 1
        lex_end = stop
        while lex_end > lex_start and chars[lex_end - 1].text in _CLOSE:
            lex_end -= 1
        mark_start = None
        if (
            lex_end > lex_start
            and chars[lex_end - 1].text in ",.?!;:"
            and (lex_end - 1 == lex_start or chars[lex_end - 2].text not in ",.?!;:")
        ):
            mark_start = lex_end - 1
            lex_end -= 1
        # A closing wrapper between the word and punctuation keeps both outside.
        while lex_end > lex_start and chars[lex_end - 1].text in _CLOSE:
            lex_end -= 1
            mark_start = None
        if lex_end == lex_start:
            continue
        end = mark_start + 1 if mark_start is not None else lex_end
        while end < len(chars) and chars[end].text.isspace():
            end += 1
        spans.append(_RemovalSpan(start, end, lex_start, lex_end, mark_start, "omit"))
    return spans


def _clean_pieces(
    pieces: tuple[TurnPiece, ...],
    *,
    fillers: FillerList,
) -> tuple[TurnPiece, ...]:
    return tuple(piece for _, piece in _clean_indexed(pieces, fillers=fillers))


def _clean_indexed(
    pieces: tuple[TurnPiece, ...],
    *,
    fillers: FillerList,
    sources: Sequence[int] | None = None,
) -> tuple[tuple[int, TurnPiece], ...]:
    """Clean phrases then words, preserving original input indexes."""
    return tuple((p.index, p.piece) for p in _clean_tracked(
        _track(pieces), fillers=fillers, sources=sources,
    ))


def _clean_tracked(
    pieces: tuple[_Piece, ...],
    *,
    fillers: FillerList,
    sources: Sequence[int] | None = None,
    trace: list[Removal | Protected] | None = None,
    skip: Callable[[Sequence[_Char]], bool] | None = None,
    marks: EffectiveMarks | None = None,
) -> tuple[_Piece, ...]:
    if marks:
        chars = _flatten(pieces)
        pieces = _clean_pass(pieces, chars, _omit_spans(chars, marks), trace=trace)
        previous_skip = skip

        def protected(lexical: Sequence[_Char]) -> bool:
            return any(
                marks.get(char.source) == "keep" for char in lexical if char.source is not None
            ) or (previous_skip is not None and previous_skip(lexical))

        skip = protected
    for entries, cause in ((fillers.phrases, "phrase"), (fillers.words, "filler")):
        if not entries:
            continue
        chars = _flatten(pieces)
        body = "".join(char.text for char in chars)
        pattern = _phrase_pattern(entries) if cause == "phrase" else _word_pattern(entries)
        spans = _removal_spans(body, pattern, "phrase" if cause == "phrase" else "filler")
        pieces = _clean_pass(pieces, chars, spans, sources=sources, trace=trace, skip=skip)
    return pieces


def _clean_pass(
    pieces: tuple[_Piece, ...],
    chars: list[_Char],
    spans: Sequence[_RemovalSpan],
    *,
    sources: Sequence[int] | None = None,
    trace: list[Removal | Protected] | None = None,
    skip: Callable[[Sequence[_Char]], bool] | None = None,
) -> tuple[_Piece, ...]:
    """Apply F1-F4 to input spans; an optional predicate sees only lexical chars."""
    if not spans:
        return pieces
    out: list[_Char] = []
    cursor = 0
    capitalise = False

    def append(chunk: Sequence[_Char]) -> None:
        nonlocal capitalise
        for char in chunk:
            if capitalise and not char.text.isspace() and char.text not in _OPEN:
                # Expanding uppercase (ß to SS) would split one owned character.
                upper = char.text.upper()
                if char.text.islower() and len(upper) == 1:
                    char = _Char(upper, char.owner, char.source)
                capitalise = False
            out.append(char)

    # Spans are found on the original pass text, never on partial rewrites.
    for span in spans:
        append(chars[cursor:span.start])
        mark_char = chars[span.mark_start] if span.mark_start is not None else None
        mark = mark_char.text if mark_char is not None else None
        core = "".join(char.text for char in out).rstrip().rstrip(_WRAPPERS)
        lexical = chars[span.lex_start:span.lex_end]
        rejected = False
        if span.cause == "phrase":
            turn_end = mark is None and span.end == len(chars)
            left_clause = not core or (core[-1] in ",;:.?!" and not core.endswith("..."))
            right_clause = mark in (",", ";", ":", ".", "!") or turn_end
            whole_sentence = (not core or core[-1] in ".?!") and (mark in (".", "!") or turn_end)
            # Wrappers and mark count towards F1p owners, trailing whitespace does not.
            stop = span.mark_start + 1 if span.mark_start is not None else span.lex_end
            owners = (
                {sources[char.owner] for char in chars[span.start:stop]}
                if sources is not None else set()
            )
            rejected = rejected or (
                not left_clause or not right_clause or whole_sentence or len(owners) > 1
            )
        if not rejected and skip is not None and skip(lexical):
            rejected = True
            if trace is not None:
                trace.append(
                    Protected(
                        "".join(char.text for char in lexical),
                        tuple(
                            dict.fromkeys(
                                char.source for char in lexical if char.source is not None
                            )
                        ),
                    )
                )
        if rejected:
            append(chars[span.start:span.end])
            cursor = span.end
            continue
        if trace is not None:
            trace.append(Removal(span.cause, "".join(char.text for char in lexical), tuple(
                dict.fromkeys(char.source for char in lexical if char.source is not None)
            )))
        while out and out[-1].text.isspace():
            out.pop()
        if (
            mark_char is not None and mark in (".", "?", "!") and core
            and not _SENTENCE_END.search(core)
        ):
            last = core[-1]
            if last in ",;:":
                previous = out[len(core) - 1]
                out[len(core) - 1] = _Char(mark_char.text, previous.owner, mark_char.source)
            elif last.isalnum():
                out.append(_Char(mark_char.text, out[-1].owner, mark_char.source))
            core = "".join(char.text for char in out).rstrip(_WRAPPERS)
        capitalise = capitalise or not core or bool(_SENTENCE_END.search(core))
        opening = chars[span.start:span.lex_start]
        cursor = span.end
        if out and cursor < len(chars):
            out.append(_Char(" ", (opening or chars[cursor:])[0].owner))
        append(opening)
    append(chars[cursor:])
    if not any(char.text.isalnum() for char in out):
        out = []
    grouped: dict[int, list[_Char]] = {p.index: [] for p in pieces}
    for char in out:
        grouped[char.owner].append(char)
    return tuple(
        _with_chars(piece, grouped[piece.index]) for piece in pieces
        if any(not char.text.isspace() for char in grouped[piece.index])
    )


def _join_at_seam(left: tuple[_Piece, ...], right: tuple[_Piece, ...]) -> tuple[_Piece, ...]:
    """Join with one synthetic space on the right, retaining lexical identities."""
    # Reindex the right pieces before rewriting their head whitespace.
    offset = len(left)
    right = tuple(replace(p, index=p.index + offset, chars=tuple(
        _Char(char.text, char.owner + offset, char.source) for char in p.chars
    )) for p in right)
    tail, head = left[-1], right[0]
    tail_end = len(tail.piece.text.rstrip())
    head_start = len(head.piece.text) - len(head.piece.text.lstrip())
    return (
        *left[:-1], _with_chars(tail, tail.chars[:tail_end]),
        _with_chars(head, (_Char(" ", head.index), *head.chars[head_start:])), *right[1:],
    )


def drop_fillers(turns: Sequence[SpeakerTurn], *, fillers: FillerList) -> list[SpeakerTurn]:
    """Apply F1-F6 without changing attribution, ordering or piece metadata."""
    return [turn for turn, _ in drop_fillers_with_seams(turns, fillers=fillers)]


def drop_fillers_with_seams(
    turns: Sequence[SpeakerTurn],
    *,
    fillers: FillerList,
) -> list[tuple[SpeakerTurn, frozenset[int]]]:
    """Apply F1-F6 and report the piece indexes where F5 joined input turns."""
    cleaned, _ = drop_fillers_with_trace(turns, fillers=fillers)
    return cleaned


def drop_fillers_with_trace(
    turns: Sequence[SpeakerTurn],
    *,
    fillers: FillerList,
    marks: EffectiveMarks | None = None,
) -> tuple[list[tuple[SpeakerTurn, frozenset[int]]], tuple[Removal | Protected, ...]]:
    """Return cleaned turns, F5 seams and removals/protections, excluding probes.

    Removals are listed per output group: merged turns at their first input
    position, discarded filler-only turns at their own position. Omits precede
    phrases, then words within each group. Consumers must key by source identity,
    not position.
    """
    tracked: Iterable[tuple[SpeakerTurn, tuple[_Piece, ...]]] = (
        (turn, _track(turn.pieces)) for turn in turns
    )
    if marks:
        tracked = tuple(tracked)
        placed = {
            char.source
            for _, pieces in tracked
            for piece in pieces
            for char in piece.chars
            if char.source is not None
        }
        missing = {key: action for key, action in marks.items() if key not in placed}
        if missing:
            raise UnplaceableWordMarkError(missing)
    final: list[_Turn] = []
    last_survivor: int | None = None
    for turn, pieces in tracked:
        survives = any(
            p.piece.text.strip() for p in _clean_tracked(pieces, fillers=fillers, marks=marks)
        )
        if survives and last_survivor is not None:
            left = final[last_survivor]
            if left.turn.identity_key == turn.identity_key:
                final[last_survivor] = replace(
                    left, pieces=_join_at_seam(left.pieces, pieces),
                    owners=(*left.owners, *([left.owners[-1] + 1 if left.owners else 0]
                                          * len(pieces))),
                )
                continue
        final.append(_Turn(turn, pieces, (0,) * len(pieces), survives))
        if survives:
            last_survivor = len(final) - 1
    trace: list[Removal | Protected] = []
    out: list[tuple[SpeakerTurn, frozenset[int]]] = []
    for group in final:
        cleaned = _clean_tracked(
            group.pieces, fillers=fillers, sources=group.owners, trace=trace, marks=marks,
        )
        if not group.survives or not cleaned:
            continue
        seams = frozenset(
            position for position in range(1, len(cleaned))
            if group.owners[cleaned[position].index] != group.owners[cleaned[position - 1].index]
        )
        out.append((replace(group.turn, pieces=tuple(p.piece for p in cleaned)), seams))
    return out, tuple(trace)
