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
  timing and flags. Only text is replaced, never stored transcript data.
"""

import re
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache

from voxint.adjudication.turns import SpeakerTurn, TurnPiece
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
# Each character keeps its source piece, including a synthetic segment separator.
_Char = tuple[str, int]


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
    indexed = tuple(enumerate(pieces))
    if fillers.phrases:
        indexed = _clean_pass(
            pieces, _phrase_pattern(fillers.phrases), sources=sources, phrase=True
        )
    if fillers.words:
        cleaned = _clean_pass(tuple(piece for _, piece in indexed), _word_pattern(fillers.words))
        indexed = tuple((indexed[index][0], piece) for index, piece in cleaned)
    return indexed


def _clean_pass(
    pieces: tuple[TurnPiece, ...],
    pattern: re.Pattern[str],
    *,
    sources: Sequence[int] | None = None,
    phrase: bool = False,
) -> tuple[tuple[int, TurnPiece], ...]:
    """Clean F1-F4 and pair each surviving piece with its input index."""
    chars: list[_Char] = []
    for index, piece in enumerate(pieces):
        if not piece.text:
            continue
        if (
            piece.segment_start
            and chars
            and not chars[-1][0].isspace()
            and not piece.text[0].isspace()
        ):
            chars.append((" ", index))
        chars.extend((char, index) for char in piece.text)
    body = "".join(char for char, _ in chars)
    if not pattern.search(body):
        return tuple(enumerate(pieces))

    out: list[_Char] = []
    cursor = 0
    capitalise = False

    def append(chunk: list[_Char]) -> None:
        nonlocal capitalise
        for char, owner in chunk:
            if capitalise and not char.isspace() and char not in _OPEN:
                # One owned entry stays one character: a letter whose upper
                # case expands (ß to SS) is left as it is.
                upper = char.upper()
                if char.islower() and len(upper) == 1:
                    char = upper
                capitalise = False
            out.append((char, owner))

    # Apply regex substitutions to owned characters: the match consumes the
    # filler and its following separator, including inside coarse text. Taking
    # matches from the original text prevents new matches inside compounds.
    for match in pattern.finditer(body):
        append(chars[cursor : match.start()])
        mark = match.group("mark")
        # Quotes and brackets around the preceding text do not decide where a
        # terminal mark goes or whether a sentence ended; the text inside does.
        core = "".join(char for char, _ in out).rstrip().rstrip(_WRAPPERS)
        if phrase:
            turn_end = mark is None and match.end() == len(body)
            left_clause = not core or (core[-1] in ",;:.?!" and not core.endswith("..."))
            right_clause = mark in (",", ";", ":", ".", "!") or turn_end
            whole_sentence = (not core or core[-1] in ".?!") and (mark in (".", "!") or turn_end)
            # The seam space belongs to the right-hand turn, so trailing
            # whitespace does not count towards the phrase's sources.
            stop = match.start() + len(match.group(0).rstrip())
            owners = (
                {sources[owner] for _, owner in chars[match.start() : stop]}
                if sources is not None
                else set()
            )
            if not left_clause or not right_clause or whole_sentence or len(owners) > 1:
                append(chars[match.start() : match.end()])
                cursor = match.end()
                continue
        while out and out[-1][0].isspace():
            out.pop()
        if mark in (".", "?", "!") and core and not _SENTENCE_END.search(core):
            last = core[-1]
            if last in ",;:":
                out[len(core) - 1] = (mark, out[len(core) - 1][1])
            elif last.isalnum():
                out.append((mark, out[-1][1]))
            core = "".join(char for char, _ in out).rstrip(_WRAPPERS)
        # Judge the sentence start on the text as it now stands, after a
        # terminal mark moved onto it.
        capitalise = capitalise or not core or bool(_SENTENCE_END.search(core))
        # Quotes/brackets are not filler words; preserve them verbatim. The
        # separator before them belongs to their piece, so regrouping by owner
        # keeps the output order.
        opening = chars[match.start() : match.start() + len(match.group(1))]
        cursor = match.end()
        if out and cursor < len(chars):
            out.append((" ", (opening or chars[cursor:])[0][1]))
        append(opening)
    append(chars[cursor:])
    if not any(char.isalnum() for char, _ in out):
        # Only fillers and the punctuation around them: the turn is empty.
        out = []
    texts: list[list[str]] = [[] for _ in pieces]
    for char, owner in out:
        texts[owner].append(char)
    return tuple(
        (index, replace(piece, text="".join(text)))
        for index, (piece, text) in enumerate(zip(pieces, texts, strict=True))
        if "".join(text).strip()
    )


def _join_at_seam(
    left: tuple[TurnPiece, ...],
    right: tuple[TurnPiece, ...],
) -> tuple[TurnPiece, ...]:
    """Concatenate two turns with exactly one space at the seam, on the right.

    The dropped turn between them may have owned the only whitespace, and the
    right-hand piece need not start a segment, so join_pieces would otherwise
    glue the two words together; when both sides carry whitespace the seam
    would otherwise hold two or more.
    """
    tail, head = left[-1], right[0]
    return (
        *left[:-1],
        replace(tail, text=tail.text.rstrip()),
        replace(head, text=" " + head.text.lstrip()),
        *right[1:],
    )


def drop_fillers(turns: Sequence[SpeakerTurn], *, fillers: FillerList) -> list[SpeakerTurn]:
    """Apply F1-F6 without changing attribution, ordering or piece metadata."""
    return [turn for turn, _ in drop_fillers_with_seams(turns, fillers=fillers)]


def drop_fillers_with_seams(
    turns: Sequence[SpeakerTurn],
    *,
    fillers: FillerList,
) -> list[tuple[SpeakerTurn, frozenset[int]]]:
    """Apply F1-F6 and report, per output turn, the piece indexes where F5 joined
    in another input turn, so a later filter can refuse to work across them."""
    kept = [
        turn
        for turn in turns
        if any(p.text.strip() for p in _clean_pieces(turn.pieces, fillers=fillers))
    ]
    merged: list[tuple[SpeakerTurn, list[int]]] = []
    for turn in kept:
        if merged and merged[-1][0].identity_key == turn.identity_key:
            left, sources = merged[-1]
            sources.extend([sources[-1] + 1 if sources else 0] * len(turn.pieces))
            merged[-1] = (replace(left, pieces=_join_at_seam(left.pieces, turn.pieces)), sources)
        else:
            merged.append((turn, [0] * len(turn.pieces)))
    out: list[tuple[SpeakerTurn, frozenset[int]]] = []
    for turn, sources in merged:
        cleaned = _clean_indexed(turn.pieces, fillers=fillers, sources=sources)
        if not cleaned:
            # Removals emptied the merged turn; a turn without a match is never empty here.
            continue
        seams = frozenset(
            position
            for position in range(1, len(cleaned))
            if sources[cleaned[position][0]] != sources[cleaned[position - 1][0]]
        )
        out.append((replace(turn, pieces=tuple(piece for _, piece in cleaned)), seams))
    return out
