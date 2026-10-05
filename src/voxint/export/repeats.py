"""Conservative render-time English repeat removal.

* R1: compare whole letter/apostrophe tokens, with sentence-start case rules.
* R2: remove the second word of an eligible maximal run of exactly two.
* R3: remove the second distinct function-word pair without extensions.
* R4: protect every token in equal-key runs longer than two.
* R5: cancel overlapping candidates, judged before any guard declines one;
  collect once, without rescanning.
* R6: stay within a turn, a segment and a filler seam; protect surviving
  timed pause gaps.
* R7: protect tokens inside single or double quoted spans.
* R8: replace removal separators with one space owned by the next token.
* R9: reuse unchanged turns and pieces, preserving metadata and attribution.

Deliberately absent: well like right just there yes yeah okay how where why
can will.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace

from voxint.adjudication.turns import SpeakerTurn, TurnPiece
from voxint.export.reading import PARAGRAPH_PAUSE_SECONDS

FUNCTION_WORDS = frozenset(
    [
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "who",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "would",
        "shall",
        "should",
        "could",
        "may",
        "might",
        "must",
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "each",
        "every",
        "no",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "from",
        "with",
        "by",
        "about",
        "into",
        "over",
        "under",
        "after",
        "before",
        "between",
        "through",
        "up",
        "down",
        "out",
        "off",
        "and",
        "or",
        "but",
        "so",
        "if",
        "because",
        "as",
        "than",
        "then",
        "when",
        "while",
    ]
) | frozenset(
    f"{stem}'{suffix}"
    for suffix, stems in (
        ("m", "i"),
        ("re", "we you they"),
        ("s", "he she it that"),
        ("ve", "i we you they"),
        ("ll", "i we you he she it they that"),
        ("d", "i we you he she it they that"),
        ("t", "isn aren wasn weren don doesn didn haven hasn hadn couldn shouldn wouldn can won"),
    )
    for stem in stems.split()
)
ONE_WORD_EXCLUDED = frozenset(["that", "had", "is", "was", "do", "did", "her", "no", "so"])
ONE_WORD_ELIGIBLE = FUNCTION_WORDS - ONE_WORD_EXCLUDED
_CLOSE = "\"')]\u201d\u2019"


@dataclass(frozen=True)
class _Char:
    text: str
    owner: int
    segment: int
    synthetic: bool = False


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    raw: str
    key: str | None
    quoted: bool


def _flatten(pieces: tuple[TurnPiece, ...], seams: frozenset[int]) -> list[_Char]:
    chars: list[_Char] = []
    segment = 0
    for index, piece in enumerate(pieces):
        segment += piece.segment_start or index in seams
        if not piece.text:
            continue
        if (
            piece.segment_start
            and chars
            and not chars[-1].text.isspace()
            and not piece.text[0].isspace()
        ):
            chars.append(_Char(" ", index, segment, synthetic=True))
        chars.extend(_Char(char, index, segment) for char in piece.text)
    return chars


def _tokens(body: str) -> list[_Token]:
    quoted: list[bool] = []
    double = single = False
    for index, char in enumerate(body):
        prev = body[index - 1] if index else ""
        following = body[index + 1] if index + 1 < len(body) else ""
        if char == "\u201c":
            double = True
        elif char == "\u201d":
            double = False
        elif char == '"':
            double = not double
        elif char in "'\u2018\u2019" and not (prev.isalpha() and following.isalpha()):
            if char in "'\u2018" and following.isalpha() and (not prev or prev.isspace()):
                single = True
            elif (
                char in "'\u2019"
                and prev.isalpha()
                and (
                    not following
                    or following.isspace()
                    or unicodedata.category(following).startswith("P")
                )
            ):
                single = False
        quoted.append(double or single)
    tokens: list[_Token] = []
    for match in re.finditer(r"\S+", body):
        raw = match.group()
        parts = raw.replace("\u2019", "'").split("'")
        key = raw.casefold().replace("\u2019", "'") if all(p.isalpha() for p in parts) else None
        first_letter = next(
            (i for i in range(match.start(), match.end()) if body[i].isalpha()), None
        )
        tokens.append(
            _Token(
                match.start(),
                match.end(),
                raw,
                key,
                quoted[first_letter] if first_letter is not None else False,
            )
        )
    return tokens


def _case_matches(tokens: list[_Token], first: int, second: int, *, initial: bool) -> bool:
    # Apostrophe style is not a case difference.
    left, right = (tokens[i].raw.replace("\u2019", "'") for i in (first, second))
    if left == right:
        return True
    sentence_start = first == 0 or tokens[first - 1].raw.rstrip(_CLOSE).endswith((".", "?", "!"))
    return (
        initial
        and sentence_start
        and not (sum(char.isalpha() for char in left) > 1 and left.isupper())
        and right.islower()
    )


def _candidates(tokens: list[_Token]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    protected: set[int] = set()
    index = 0
    while index < len(tokens):
        end = index + 1
        key = tokens[index].key
        if key is not None:
            while end < len(tokens) and tokens[end].key == key:
                end += 1
            if end - index > 2:
                protected.update(range(index, end))
            elif end - index == 2 and key in ONE_WORD_ELIGIBLE:
                candidates.append((index, 1))
        index = end
    for index in range(len(tokens) - 3):
        a, b, c, d = (token.key for token in tokens[index : index + 4])
        if (
            a in FUNCTION_WORDS
            and b in FUNCTION_WORDS
            and a != b
            and a == c
            and b == d
            and (index == 0 or tokens[index - 1].key != b)
            and (index + 4 == len(tokens) or tokens[index + 4].key != a)
            and not protected.intersection(range(index, index + 4))
        ):
            candidates.append((index, 2))
    return sorted(candidates)


def _clean_pieces(pieces: tuple[TurnPiece, ...], seams: frozenset[int]) -> tuple[TurnPiece, ...]:
    chars = _flatten(pieces, seams)
    tokens = _tokens("".join(char.text for char in chars))
    found = _candidates(tokens)
    # Overlap is judged on every candidate in the input, so a declined one still
    # cancels its neighbour instead of leaving a half-cleaned repeat behind.
    uses = [0] * len(tokens)
    for start, size in found:
        for index in range(start, start + 2 * size):
            uses[index] += 1
    applied: list[tuple[int, int]] = []
    for start, size in found:
        if any(uses[index] != 1 for index in range(start, start + 2 * size)):
            continue
        end = start + 2 * size
        span = tokens[start:end]
        if any(token.quoted for token in span):
            continue
        if chars[span[0].start].segment != chars[span[-1].end - 1].segment:
            continue
        if not all(
            _case_matches(tokens, start + offset, start + size + offset, initial=offset == 0)
            for offset in range(size)
        ):
            continue
        prev = pieces[chars[tokens[start + size - 1].end - 1].owner]
        if end < len(tokens):
            following = pieces[chars[tokens[end].start].owner]
            if (
                prev is not following
                and prev.timed
                and following.timed
                and following.start_seconds - prev.end_seconds >= PARAGRAPH_PAUSE_SECONDS
            ):
                continue
        applied.append((start, size))
    if not applied:
        return pieces
    out: list[_Char] = []
    cursor = 0
    for start, size in applied:
        out.extend(chars[cursor : tokens[start + size - 1].end])
        end = start + 2 * size
        cursor = tokens[end].start if end < len(tokens) else len(chars)
        if cursor < len(chars):
            out.append(replace(chars[cursor], text=" ", synthetic=False))
    out.extend(chars[cursor:])
    texts: list[list[str]] = [[] for _ in pieces]
    for char in out:
        if not char.synthetic:
            texts[char.owner].append(char.text)
    result: list[TurnPiece] = []
    for piece, text_chars in zip(pieces, texts, strict=True):
        text = "".join(text_chars)
        if text == piece.text:
            result.append(piece)
        elif text.strip():
            result.append(replace(piece, text=text, anchors=()))
    return tuple(result)


def drop_repeats(
    turns: Sequence[SpeakerTurn],
    seams: Sequence[frozenset[int]] | None = None,
) -> list[SpeakerTurn]:
    """Apply R1-R9 without changing attribution, ordering or piece metadata.

    ``seams`` holds, per turn, the piece indexes where filler removal joined two
    input turns (``drop_fillers_with_seams``); no repeat is matched across one.
    """
    if seams is not None and len(seams) != len(turns):
        raise ValueError("seams must hold one entry per turn")
    result: list[SpeakerTurn] = []
    for index, turn in enumerate(turns):
        pieces = _clean_pieces(turn.pieces, seams[index] if seams is not None else frozenset())
        result.append(turn if pieces is turn.pieces else replace(turn, pieces=pieces))
    return result
