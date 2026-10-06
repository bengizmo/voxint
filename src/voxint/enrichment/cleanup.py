"""Pure core of the LLM clean-up text variant (#758); no database, no LLM.

The LLM proposes each line cleaned. Voxint accepts a proposal only as a pure
deletion of source words (D1), never of a protected word (D2), and builds the
cleaned line from the source words alone, so the model never contributes a
character:

* Source words are the line's #757 anchors (D5). A line without anchors is
  unanchorable: any change to it is rejected as ``unanchored``.
* :func:`cleanup_key` is the single normalization authority: ``token_key``
  (NFKC, casefold, straight quotes, Unicode hyphens, edge punctuation
  trimmed), then every remaining non-alphanumeric character dropped. So
  ``don't``, its curly-apostrophe spelling and ``dont`` match, and
  ``twenty-one`` matches ``twentyone``.
* Alignment is protection-preferring and deterministic: among the ways to
  embed the proposal's keys in the source keys, it takes one that deletes no
  protected word, keeping the latest source occurrences on ties (a speaker's
  restart is the occurrence that survives).
* Rejection precedence is ``not_deletion`` > ``protected`` > ``whole_line``.
  Words whose key is empty (punctuation only) are never deleted and never
  required.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from voxint.adjudication.turns import WordAnchor, WordMarkKey, token_key
from voxint.export.fillers import delete_spans

# ``malformed`` is never returned here: the producer assigns it to a line
# whose reply could not be parsed, and it is counted alongside the others.
RejectReason = Literal["not_deletion", "protected", "whole_line", "unanchored", "malformed"]
REJECT_REASONS: tuple[RejectReason, ...] = (
    "not_deletion", "protected", "whole_line", "unanchored", "malformed",
)

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen"
    " fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty"
    " sixty seventy eighty ninety hundred thousand million billion trillion"
)
_ORDINALS = (
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh"
    " twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth"
    " nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth"
    " ninetieth hundredth thousandth millionth billionth trillionth"
)
NUMBER_WORDS = frozenset((_UNITS + " " + _ORDINALS).split())
NEGATIONS = frozenset(
    {"not", "no", "never", "nor", "none", "nobody", "nothing", "nowhere", "neither", "cannot"}
)
_I_FORMS = frozenset({"i", "i'm", "i'll", "i'd", "i've"})
_CLOSERS = "\"')]\u201d\u2019"
# A period after these does not end a sentence, so the next capital stays a name.
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "mt"})


def cleanup_key(token: str) -> str:
    """The matching key: ``token_key`` with every non-alphanumeric dropped."""
    return "".join(char for char in token_key(token) if char.isalnum())


def _is_number(normalized: str) -> bool:
    return any(part in NUMBER_WORDS for part in normalized.split("-"))


def _is_negation(normalized: str) -> bool:
    return normalized in NEGATIONS or normalized.endswith("n't")


def _starts_sentence(previous: str | None) -> bool:
    if previous is None:
        return True
    bare = previous.rstrip(_CLOSERS)
    if bare.endswith(".") and cleanup_key(bare) in _ABBREVIATIONS:
        return False
    return bare.endswith((".", "?", "!"))


def _is_name(lexical: str, normalized: str, previous: str | None) -> bool:
    first = next((char for char in lexical if char.isalpha()), "")
    return first.isupper() and normalized not in _I_FORMS and not _starts_sentence(previous)


def is_protected(lexical: str, previous: str | None) -> bool:
    """D2: a digit, a number word, a negation or a likely name.

    ``previous`` is the preceding source word's text, ``None`` at line start.
    """
    normalized = token_key(lexical)
    return (
        any(char.isdigit() for char in unicodedata.normalize("NFKC", lexical))
        or _is_number(normalized)
        or _is_negation(normalized)
        or _is_name(lexical, normalized, previous)
    )


@dataclass(frozen=True)
class SourceWord:
    """One anchored source unit: its lexical text, keys and protection.

    A unit can hold several whitespace-separated words (``will not``); it is
    kept or deleted whole, matches that many proposal words, and is protected
    when any of its words is. ``keys`` omits empty (punctuation-only) keys.
    """

    anchor: WordAnchor
    text: str
    keys: tuple[str, ...]
    protected: bool

    @property
    def identity(self) -> WordMarkKey:
        return (self.anchor.segment_id, self.anchor.token_start, self.anchor.token_end)


@dataclass(frozen=True)
class SourceLine:
    """A displayed line's text and its anchored words, in text order."""

    text: str
    words: tuple[SourceWord, ...]


def source_line(text: str, anchors: Sequence[WordAnchor]) -> SourceLine:
    """Freeze a line's anchored units; anchors are offsets into ``text``."""
    words: list[SourceWord] = []
    previous: str | None = None
    for anchor in sorted(anchors, key=lambda a: a.lex_start):
        lexical = text[anchor.lex_start:anchor.lex_end]
        keys: list[str] = []
        protected = False
        for part in lexical.split():
            key = cleanup_key(part)
            if key:
                keys.append(key)
                protected = protected or is_protected(part, previous)
            previous = part
        words.append(SourceWord(anchor, lexical, tuple(keys), protected))
    return SourceLine(text, tuple(words))


@dataclass(frozen=True)
class Accepted:
    """Delete these indexes into ``SourceLine.words``; never empty."""

    deleted: frozenset[int]


@dataclass(frozen=True)
class Unchanged:
    """The proposal keeps every source word."""


@dataclass(frozen=True)
class Rejected:
    """The proposal is not an acceptable deletion; the line stays verbatim."""

    reason: RejectReason


Outcome = Accepted | Unchanged | Rejected


def proposal_keys(proposed: str) -> list[str]:
    """Whitespace tokens of a proposal as keys, empty keys ignored."""
    return [key for key in map(cleanup_key, proposed.split()) if key]


def _align(
    words: Sequence[SourceWord], proposal: Sequence[str], *, respect: bool,
) -> list[int] | None:
    """Latest-occurrence embedding of ``proposal`` in the keyed source units.

    Each unit matches its whole run of keys or is skipped (deleted). With
    ``respect``, no protected unit may be skipped. Returns the matched source
    indexes, or ``None`` when no such embedding exists. O(n*m). Skipping is
    preferred whenever the rest still embeds, so a restart keeps its final
    occurrence: ``I mean, I think`` against ``I think`` deletes ``I mean,``.
    """
    keyed = [i for i, word in enumerate(words) if word.keys]
    n, m = len(keyed), len(proposal)

    def matches(i: int, k: int) -> bool:
        keys = words[keyed[i]].keys
        return tuple(proposal[k:k + len(keys)]) == keys

    # feasible[i][k]: proposal[k:] embeds in keyed[i:] under the constraint.
    feasible = [[False] * (m + 1) for _ in range(n + 1)]
    feasible[n][m] = True
    for i in range(n - 1, -1, -1):
        width = len(words[keyed[i]].keys)
        deletable = not (respect and words[keyed[i]].protected)
        for k in range(m, -1, -1):
            match = matches(i, k) and feasible[i + 1][k + width]
            feasible[i][k] = match or (deletable and feasible[i + 1][k])
    if not feasible[0][0]:
        return None
    matched: list[int] = []
    k = 0
    for i in range(n):
        width = len(words[keyed[i]].keys)
        deletable = not (respect and words[keyed[i]].protected)
        if deletable and feasible[i + 1][k]:
            continue
        if matches(i, k) and feasible[i + 1][k + width]:
            matched.append(keyed[i])
            k += width
    return matched


def validate_proposal(line: SourceLine, proposed: str) -> Outcome:
    """Judge one proposal for one line (D1, D2, D5, whole-line rejection)."""
    proposal = proposal_keys(proposed)
    if not line.words:
        source = proposal_keys(line.text)
        return Unchanged() if proposal == source else Rejected("unanchored")
    if _align(line.words, proposal, respect=False) is None:
        return Rejected("not_deletion")
    matched = _align(line.words, proposal, respect=True)
    if matched is None:
        return Rejected("protected")
    keyed = {i for i, word in enumerate(line.words) if word.keys}
    deleted = frozenset(keyed - set(matched))
    if not deleted:
        return Unchanged()
    if deleted == keyed:
        return Rejected("whole_line")
    return Accepted(deleted)


def render_cleaned(line: SourceLine, deleted: frozenset[int]) -> str:
    """Build the cleaned line from source characters under the omit rules."""
    return delete_spans(
        line.text,
        [word.anchor for word in line.words],
        [line.words[i].identity for i in sorted(deleted)],
    )

