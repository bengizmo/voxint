"""Render-time English filler removal, before the Markdown reading layout.

* F1: standalone um, uh, umm, uhh, uhm and erm, ignoring case. Quotes and
  opening brackets are allowed; compounds, possessives and ellipses stay.
* F2: remove the word and its following comma, semicolon or colon. Move a
  terminal mark to preceding text, replacing comma/semicolon/colon, or drop
  it at turn start or after another terminal mark. A preceding comma stays.
* F3: collapse only separators beside a removal, to one between survivors
  and none at turn start. Respect the segment boundary rule of join_pieces.
* F4: after a sentence-start filler, capitalise the next lowercase initial,
  skipping leading quotes/brackets. Other casing stays unchanged.
* F5: discard turns made only of fillers, then merge adjacent surviving
  canonical identities. The merged turn is cleaned as one, so a filler at
  the seam is judged against the text that now precedes it.
* F6: leave no-filler text unchanged, and retain every surviving piece's
  timing and flags. Only text is replaced, never stored transcript data.
"""

import re
from collections.abc import Sequence
from dataclasses import replace

from voxint.adjudication.turns import SpeakerTurn, TurnPiece

_OPEN = "\"'([\u201c\u2018"
_CLOSE = "\"'\u201d\u2019"
_FILLER = re.compile(
    r"""(?<!\S)(["'(\[\u201c\u2018]*)(?:umm|uhh|uhm|erm|um|uh)"""
    r"""(?:(?P<mark>[,.?!;:])(?=$|\s|["'\u201d\u2019])|(?=$|\s))\s*""",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"""[.?!]["'\u201d\u2019]*$""")
# Each character keeps its source piece, including a synthetic segment separator.
_Char = tuple[str, int]


def _clean_pieces(pieces: tuple[TurnPiece, ...]) -> tuple[TurnPiece, ...]:
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
    if not _FILLER.search(body):
        return pieces

    out: list[_Char] = []
    cursor = 0
    capitalise = False

    def append(chunk: list[_Char]) -> None:
        nonlocal capitalise
        for char, owner in chunk:
            if capitalise and not char.isspace() and char not in _OPEN:
                if char.islower():
                    char = char.upper()
                capitalise = False
            out.append((char, owner))

    # Apply regex substitutions to owned characters: the match consumes the
    # filler and its following separator, including inside coarse text. Taking
    # matches from the original text prevents new matches inside compounds.
    for match in _FILLER.finditer(body):
        append(chars[cursor : match.start()])
        while out and out[-1][0].isspace():
            out.pop()
        mark = match.group("mark")
        if mark in (".", "?", "!") and out:
            last, owner = out[-1]
            if last in ",;:":
                out[-1] = (mark, owner)
            elif last.isalnum() or last in _CLOSE:
                out.append((mark, owner))
        # Judge the sentence start on the text as it now stands, after a
        # terminal mark moved onto it.
        preceding = "".join(char for char, _ in out)
        capitalise = capitalise or not preceding or bool(_SENTENCE_END.search(preceding))
        # Quotes/brackets are not filler words; preserve them verbatim.
        opening = chars[match.start() : match.start() + len(match.group(1))]
        cursor = match.end()
        if out and cursor < len(chars):
            out.append((" ", chars[cursor][1]))
        append(opening)
    append(chars[cursor:])
    texts: list[list[str]] = [[] for _ in pieces]
    for char, owner in out:
        texts[owner].append(char)
    return tuple(
        replace(piece, text="".join(text))
        for piece, text in zip(pieces, texts, strict=True)
        if "".join(text).strip()
    )


def _join_at_seam(
    left: tuple[TurnPiece, ...], right: tuple[TurnPiece, ...],
) -> tuple[TurnPiece, ...]:
    """Concatenate two turns, keeping one separator where neither side has one.

    The dropped turn between them may have owned the only whitespace, and the
    right-hand piece need not start a segment, so join_pieces would otherwise
    glue the two words together.
    """
    head = right[0]
    if left and not left[-1].text[-1].isspace() and not head.text[0].isspace():
        right = (replace(head, text=" " + head.text), *right[1:])
    return left + right


def drop_fillers(turns: Sequence[SpeakerTurn]) -> list[SpeakerTurn]:
    """Apply F1-F6 without changing attribution, ordering or piece metadata."""
    kept = [turn for turn in turns if any(p.text.strip() for p in _clean_pieces(turn.pieces))]
    merged: list[SpeakerTurn] = []
    for turn in kept:
        if merged and merged[-1].identity_key == turn.identity_key:
            merged[-1] = replace(merged[-1], pieces=_join_at_seam(merged[-1].pieces, turn.pieces))
        else:
            merged.append(turn)
    return [replace(turn, pieces=_clean_pieces(turn.pieces)) for turn in merged]
