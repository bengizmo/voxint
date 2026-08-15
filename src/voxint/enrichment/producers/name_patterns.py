"""Pure name-mention extraction for the offline name-candidate producer (#38).

No I/O, no DB: text in, :class:`RawMention` list out. The inventory is a
bounded, versioned set of regex patterns over media metadata (title,
description, channel/uploader, tags) and transcript segments
(self-introductions, host-introductions). Matching is deliberately
capitalization-independent — ASR ``raw_text`` carries no reliable casing — so
false-positive control lives in explicit guards (stoplists, capture
termination, context windows), all table-tested with lowercase ASR examples.

A mention is a *heard/read* name: never grounded identity, and never an
assignment. Attribution decides how far a mention can reach:

- ``SELF`` (a self-introduction) is the only attribution eligible to become a
  cluster-level (run_label) candidate, and only for the diarization label of
  the segment that contains it.
- ``OTHER`` and ``METADATA`` mentions say a name is probably *in the
  recording* — run-level only. No adjacent-turn inference in v1.

Changing patterns, reliabilities, or guards changes producer output, so bump
``PATTERN_SET_VERSION`` (it feeds the producer's config and idempotency
signature).
"""

import enum
import re
import unicodedata
from dataclasses import dataclass

PATTERN_SET_VERSION = 1

MAX_NAME_TOKENS = 4
MIN_NAME_CHARS = 2
MAX_NAME_CHARS = 120
SNIPPET_CONTEXT_CHARS = 60
CONTEXT_WINDOW_CHARS = 80


class Attribution(enum.StrEnum):
    """Who put the name in the signal: the speaker, another voice, or text."""

    SELF = "self"
    OTHER = "other"
    METADATA = "metadata"


@dataclass(frozen=True)
class MetadataRef:
    """Source pointer for a metadata mention: the snapshot column it came from."""

    field: str
    item_index: int | None = None  # for array fields such as ``tags``


@dataclass(frozen=True)
class SegmentRef:
    """Source pointer for a transcript mention (pure — DB ids mapped later)."""

    segment_index: int
    diarization_label: str | None
    start_seconds: float
    suspect: bool


SourceRef = MetadataRef | SegmentRef


@dataclass(frozen=True)
class RawMention:
    """One extracted name occurrence with its provenance and pattern context."""

    name: str  # normalized display form
    raw_span: str  # exact matched name text
    pattern_id: str
    reliability: float
    attribution: Attribution
    source: SourceRef
    snippet: str
    ambiguous: bool = False  # single-token name that is also a common word


@dataclass(frozen=True)
class NamePattern:
    """A compiled trigger with its reliability weight and attribution class."""

    pattern_id: str
    regex: re.Pattern[str]
    reliability: float
    attribution: Attribution


# A name token: letters (unicode), with internal apostrophes/hyphens/periods.
_LETTER = r"[^\W\d_]"
_TOKEN = rf"{_LETTER}+(?:['’.\-]{_LETTER}+)*"  # noqa: RUF001 — curly apostrophe intended
_NAME_CAPTURE = rf"(?P<name>{_TOKEN}(?:\s+{_TOKEN}){{0,{MAX_NAME_TOKENS - 1}}})"

# Words that terminate a name capture (missing ASR punctuation makes regexes
# run away; a bounded token count plus discourse stopwords reins them in).
CAPTURE_STOPWORDS = frozenset(
    [
        "and",
        "but",
        "or",
        "so",
        "because",
        "today",
        "tonight",
        "tomorrow",
        "yesterday",
        "here",
        "there",
        "from",
        "with",
        "at",
        "on",
        "in",
        "to",
        "for",
        "of",
        "about",
        "welcome",
        "thanks",
        "thank",
        "everybody",
        "everyone",
        "guys",
        "folks",
        "joining",
        "again",
        "back",
        "live",
        "right",
        "now",
        "this",
        "that",
        "these",
        "those",
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "my",
        "your",
        "our",
        "his",
        "her",
        "their",
        "its",
        "it",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "as",
        "if",
        "then",
        "than",
        "gonna",
        "wanna",
        "let's",
        "im",
        "i'm",
        "we're",
        "it's",
        "coming",
        "going",
        "getting",
        "digs",
        "dives",
        "explores",
        "discusses",
        "talks",
        "shares",
        "breaks",
        "joins",
        "brings",
        "sits",
        "asks",
        "answers",
        "covers",
        "explains",
        "interviews",
        "chats",
        "speaks",
        "takes",
        "walks",
        "goes",
        "gets",
        "tells",
        "teaches",
        "learns",
        "shows",
        "returns",
        "visits",
        "reveals",
        "unpacks",
    ]
)

# First-token rejects for transcript captures: predicate complements ("i'm
# happy", "i'm from boston"), states, and fillers that follow the triggers.
PREDICATE_STOPWORDS = frozenset(
    [
        "sure",
        "sorry",
        "glad",
        "happy",
        "excited",
        "thrilled",
        "going",
        "just",
        "really",
        "very",
        "not",
        "only",
        "always",
        "never",
        "actually",
        "honestly",
        "obviously",
        "definitely",
        "probably",
        "also",
        "more",
        "most",
        "less",
        "quite",
        "pretty",
        "super",
        "extremely",
        "kind",
        "sort",
        "like",
        "about",
        "still",
        "ready",
        "trying",
        "looking",
        "thinking",
        "talking",
        "working",
        "hoping",
        "wondering",
        "kidding",
        "serious",
        "afraid",
        "aware",
        "certain",
        "confident",
        "okay",
        "ok",
        "fine",
        "good",
        "great",
        "sad",
        "tired",
        "curious",
        "done",
        "all",
        "out",
        "off",
        "up",
        "down",
        "over",
        "under",
        "sitting",
        "standing",
        "recording",
        "proud",
        "grateful",
        "thankful",
        "blessed",
        "honored",
        "humbled",
        "delighted",
        "pleased",
        "broadcasting",
        "streaming",
        "awkward",
        "weird",
        "crazy",
        "amazing",
        "incredible",
        "interesting",
        "important",
        "different",
        "huge",
        "big",
    ]
)

# Reject the whole capture if any token is one of these (pronouns, greetings
# targets, weekdays — things a person is never actually named in practice).
GENERIC_NONNAMES = frozenset(
    [
        "me",
        "him",
        "us",
        "them",
        "someone",
        "somebody",
        "anyone",
        "anybody",
        "nobody",
        "everyone",
        "everybody",
        "guys",
        "folks",
        "people",
        "person",
        "friends",
        "friend",
        "family",
        "listeners",
        "listener",
        "viewers",
        "viewer",
        "subscribers",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "mom",
        "dad",
        "god",
        "lord",
        "man",
        "dude",
        "bro",
        "sis",
        "y'all",
        "yall",
    ]
)

# Organization-shaped vocabulary: rejects channel-as-host and any capture
# containing one of these tokens (legal suffixes, media org words).
ORG_WORDS = frozenset(
    [
        "podcast",
        "podcasts",
        "media",
        "tv",
        "official",
        "channel",
        "show",
        "network",
        "radio",
        "news",
        "studio",
        "studios",
        "productions",
        "entertainment",
        "music",
        "gaming",
        "games",
        "team",
        "group",
        "company",
        "co",
        "inc",
        "llc",
        "ltd",
        "corp",
        "corporation",
        "university",
        "college",
        "church",
        "institute",
        "academy",
        "school",
        "association",
        "foundation",
        "magazine",
        "journal",
        "daily",
        "weekly",
        "review",
        "report",
        "series",
        "episode",
        "ep",
        "vlog",
        "blog",
        "talk",
        "talks",
        "tips",
        "tricks",
        "guide",
        "tutorial",
        "basics",
        "myths",
        "facts",
        "secrets",
        "hacks",
    ]
)

# Single-token names that double as common words (or months): kept only when
# the pattern is strong or a domain-pack seed vouches for them (scoring rule).
AMBIGUOUS_SINGLE_TOKEN_NAMES = frozenset(
    [
        "will",
        "may",
        "mark",
        "hope",
        "faith",
        "bill",
        "rose",
        "chase",
        "grace",
        "miles",
        "art",
        "dawn",
        "gene",
        "penny",
        "ray",
        "guy",
        "wade",
        "dean",
        "buck",
        "chip",
        "rusty",
        "don",
        "pat",
        "mike",
        "march",
        "april",
        "june",
        "august",
        "summer",
        "autumn",
        "destiny",
        "harmony",
        "melody",
    ]
)

HONORIFICS = frozenset(
    [
        "dr",
        "mr",
        "mrs",
        "ms",
        "miss",
        "prof",
        "professor",
        "doctor",
        "sir",
        "dame",
        "rev",
        "reverend",
        "pastor",
        "coach",
        "captain",
    ]
)

NAME_SUFFIXES = frozenset(["jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"])

# Trigger vocabulary that marks a title segment as a phrase, not a name
# ("Interview with Jane Doe" split by separators is still not person-shaped).
_TITLE_TRIGGER_WORDS = frozenset(
    [
        "interview",
        "featuring",
        "ft",
        "feat",
        "guest",
        "guests",
        "host",
        "hosts",
        "hosted",
        "presents",
        "versus",
        "vs",
    ]
)

# If any of these appear in the surrounding window, the mention is sponsor /
# ad-read boilerplate, not a participant.
_SPONSOR_MARKERS = (
    "sponsor",
    "promo code",
    "use code",
    "brought to you by",
    "discount",
    "coupon",
)

# Quotative markers just before a trigger mean reported speech, not a
# self-introduction ("he said my name is ...").
_QUOTATIVE_MARKERS = ("said", "says", "saying", "told", "asked", "wrote", "quote")


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


TITLE_PATTERNS: tuple[NamePattern, ...] = (
    NamePattern(
        "title_interview_with",
        _compile(rf"\b(?:an?\s+)?interview\s+with\s+{_NAME_CAPTURE}"),
        0.85,
        Attribution.METADATA,
    ),
    NamePattern(
        "title_ft",
        _compile(rf"\b(?:ft\.?|feat\.?|featuring)\s+{_NAME_CAPTURE}"),
        0.8,
        Attribution.METADATA,
    ),
    NamePattern(
        "title_with",
        _compile(rf"\b(?:with|w/)\s+{_NAME_CAPTURE}\s*$"),
        0.65,
        Attribution.METADATA,
    ),
)

DESCRIPTION_PATTERNS: tuple[NamePattern, ...] = (
    NamePattern(
        "desc_hosted_by",
        _compile(rf"\b(?:hosted\s+by|hosts?\s*:|your\s+host)[,:]?\s+{_NAME_CAPTURE}"),
        0.8,
        Attribution.METADATA,
    ),
    NamePattern(
        "desc_guest",
        _compile(
            r"\b(?:guests?\s*:|featuring|joined\s+by|speaks\s+with"
            rf"|in\s+conversation\s+with|interview\s+with)\s+{_NAME_CAPTURE}"
        ),
        0.75,
        Attribution.METADATA,
    ),
)

TRANSCRIPT_PATTERNS: tuple[NamePattern, ...] = (
    NamePattern(
        "self_my_name_is",
        _compile(rf"\bmy\s+name(?:\s+is|['’]s)\s+{_NAME_CAPTURE}"),  # noqa: RUF001
        0.9,
        Attribution.SELF,
    ),
    NamePattern(
        "self_i_am",
        _compile(rf"\b(?:i'm|i\s+am)\s+{_NAME_CAPTURE}"),
        0.7,
        Attribution.SELF,
    ),
    NamePattern(
        "self_this_is",
        _compile(
            r"^\W{0,8}(?:(?:hey|hi|hello|okay|ok|so|well|alright|and)[,!]?\s+)?"
            rf"this\s+is\s+{_NAME_CAPTURE}"
        ),
        0.6,
        Attribution.SELF,
    ),
    NamePattern(
        "self_speaking",
        _compile(rf"^\W{{0,8}}{_NAME_CAPTURE}\s*,?\s+(?:here|speaking)\b"),
        0.55,
        Attribution.SELF,
    ),
    NamePattern(
        "other_welcome",
        _compile(rf"\b(?:please\s+)?welcome\b[,!]?\s+(?:to\s+the\s+show[,!]?\s+)?{_NAME_CAPTURE}"),
        0.7,
        Attribution.OTHER,
    ),
    NamePattern(
        "other_joined_by",
        _compile(
            r"\b(?:we(?:'re|\s+are)\s+joined\s+(?:today\s+)?by|with\s+me\s+today\s+is"
            rf"|i(?:'m|\s+am)\s+here\s+with|joining\s+(?:me|us)\s+(?:today\s+|now\s+)?is)\s+"
            rf"{_NAME_CAPTURE}"
        ),
        0.75,
        Attribution.OTHER,
    ),
    NamePattern(
        "other_thanks_joining",
        _compile(
            r"\bthanks?\s+(?:so\s+much\s+)?for\s+(?:joining|being)\s+(?:us|me|here)"
            rf"[,!]?\s+{_NAME_CAPTURE}"
        ),
        0.7,
        Attribution.OTHER,
    ),
    NamePattern(
        "other_vocative_thanks",
        _compile(rf"\bthank\s+you[,!]\s+{_NAME_CAPTURE}"),
        0.35,
        Attribution.OTHER,
    ),
)

CHANNEL_AS_HOST_RELIABILITY = 0.55
TAG_PERSON_RELIABILITY = 0.4
TITLE_SEGMENT_RELIABILITY = 0.5

_TITLE_SEPARATORS = re.compile(r"\s*[|:—–]\s*|\s+-\s+")  # noqa: RUF001 — en dash intended


def normalize_name(raw: str) -> str | None:
    """Canonicalize a captured span into a display name, or reject it.

    NFKC-normalizes, strips wrapping quotes/brackets, strips edge honorifics
    and suffixes, and applies the token/character guards. Casing: ALL-CAPS or
    all-lowercase input is re-cased per token; mixed case is preserved.
    Returns ``None`` when the span cannot be a person name.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.strip(" \t\r\n\"'“”‘’()[]{}<>,.;!?…-–—")  # noqa: RUF001 — curly quotes/dashes
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    lowered = text.casefold()
    if any(marker in lowered for marker in ("http", "www.", ".com", "@")):
        return None
    if any(ch.isdigit() for ch in text):
        return None

    tokens = text.split(" ")
    while tokens and tokens[0].casefold().rstrip(".") in HONORIFICS:
        tokens = tokens[1:]
    while tokens and tokens[-1].casefold().rstrip(".,") in NAME_SUFFIXES:
        tokens = tokens[:-1]
    if not tokens or len(tokens) > MAX_NAME_TOKENS:
        return None
    folded = [token.casefold().strip(".,") for token in tokens]
    if any(token in GENERIC_NONNAMES for token in folded):
        return None
    if any(token in ORG_WORDS for token in folded):
        return None

    result = " ".join(tokens)
    if not MIN_NAME_CHARS <= len(result) <= MAX_NAME_CHARS:
        return None
    if result.isupper() or result.islower():
        result = " ".join(token.title() for token in result.split(" "))
    return result


def person_shaped(value: str) -> str | None:
    """A whole field that *is* a name (channel-as-host, person tags).

    Requires 2..4 tokens after normalization — a single token ("Veritasium")
    is not person-shaped — and the value must be nothing but the name.
    """
    stripped = value.strip()
    if not stripped or not re.fullmatch(rf"{_TOKEN}(?:\s+{_TOKEN}){{1,3}}", stripped):
        return None
    folded = {token.casefold().strip(".") for token in stripped.split()}
    if folded & (CAPTURE_STOPWORDS | _TITLE_TRIGGER_WORDS):
        return None
    return normalize_name(stripped)


def _is_ambiguous(name: str) -> bool:
    tokens = name.split(" ")
    return len(tokens) == 1 and tokens[0].casefold() in AMBIGUOUS_SINGLE_TOKEN_NAMES


def _snippet(text: str, start: int, end: int) -> str:
    lo = max(0, start - SNIPPET_CONTEXT_CHARS)
    hi = min(len(text), end + SNIPPET_CONTEXT_CHARS)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def _window_before(text: str, position: int) -> str:
    return text[max(0, position - CONTEXT_WINDOW_CHARS) : position].casefold()


def _window_around(text: str, start: int, end: int) -> str:
    lo = max(0, start - CONTEXT_WINDOW_CHARS)
    hi = min(len(text), end + CONTEXT_WINDOW_CHARS)
    return text[lo:hi].casefold()


def _trim_capture(raw_span: str) -> str | None:
    """Cut the captured tokens at the first discourse stopword."""
    kept: list[str] = []
    for token in raw_span.split():
        if token.casefold().strip(".,!?") in CAPTURE_STOPWORDS:
            break
        kept.append(token)
    return " ".join(kept) if kept else None


def _mention_from_match(
    text: str,
    match: re.Match[str],
    pattern: NamePattern,
    source: SourceRef,
) -> RawMention | None:
    name_start = match.start("name")
    if pattern.attribution is not Attribution.METADATA:
        before = _window_before(text, match.start())
        if any(marker in before for marker in _QUOTATIVE_MARKERS):
            return None
    around = _window_around(text, match.start(), match.end())
    if any(marker in around for marker in _SPONSOR_MARKERS):
        return None

    trimmed = _trim_capture(match.group("name"))
    if trimmed is None:
        return None
    first = trimmed.split()[0].casefold().strip(".,!?")
    if pattern.attribution is not Attribution.METADATA and first in PREDICATE_STOPWORDS:
        return None
    name = normalize_name(trimmed)
    if name is None:
        return None
    return RawMention(
        name=name,
        raw_span=trimmed,
        pattern_id=pattern.pattern_id,
        reliability=pattern.reliability,
        attribution=pattern.attribution,
        source=source,
        snippet=_snippet(text, name_start, name_start + len(match.group("name"))),
        ambiguous=_is_ambiguous(name),
    )


def _dedupe(mentions: list[RawMention]) -> list[RawMention]:
    """Overlapping patterns on the same span keep only the strongest mention.

    "Interview with Jane Doe" matches both ``title_interview_with`` and
    ``title_with``; counting both as corroboration would be fake diversity.
    Keyed by (source, normalized name) per field/segment.
    """
    best: dict[tuple[SourceRef, str], RawMention] = {}
    for mention in mentions:
        key = (mention.source, mention.name.casefold())
        current = best.get(key)
        if current is None or mention.reliability > current.reliability:
            best[key] = mention
    return list(best.values())


def _run_patterns(
    text: str, patterns: tuple[NamePattern, ...], source: SourceRef
) -> list[RawMention]:
    found: list[RawMention] = []
    for pattern in patterns:
        for match in pattern.regex.finditer(text):
            mention = _mention_from_match(text, match, pattern, source)
            if mention is not None:
                found.append(mention)
    return found


def _title_segment_mentions(title: str, source: MetadataRef) -> list[RawMention]:
    """Person-shaped standalone title segments: "Jane Doe | Ep 12" → Jane Doe."""
    mentions: list[RawMention] = []
    for part in _TITLE_SEPARATORS.split(title):
        name = person_shaped(part)
        if name is None:
            continue
        mentions.append(
            RawMention(
                name=name,
                raw_span=part.strip(),
                pattern_id="title_guest_sep",
                reliability=TITLE_SEGMENT_RELIABILITY,
                attribution=Attribution.METADATA,
                source=source,
                snippet=re.sub(r"\s+", " ", title).strip()[: SNIPPET_CONTEXT_CHARS * 2],
                ambiguous=_is_ambiguous(name),
            )
        )
    return mentions


def extract_from_metadata(
    *,
    title: str | None,
    description: str | None,
    channel: str | None,
    uploader: str | None,
    tags: tuple[str, ...] = (),
) -> list[RawMention]:
    """Extract METADATA-attributed mentions from a #36 snapshot's text fields."""
    mentions: list[RawMention] = []
    if title:
        source = MetadataRef(field="title")
        mentions.extend(_run_patterns(title, TITLE_PATTERNS, source))
        mentions.extend(_title_segment_mentions(title, source))
    if description:
        mentions.extend(
            _run_patterns(description, DESCRIPTION_PATTERNS, MetadataRef(field="description"))
        )
    for field, value in (("channel", channel), ("uploader", uploader)):
        if not value:
            continue
        name = person_shaped(value)
        if name is not None:
            mentions.append(
                RawMention(
                    name=name,
                    raw_span=value.strip(),
                    pattern_id="channel_as_host",
                    reliability=CHANNEL_AS_HOST_RELIABILITY,
                    attribution=Attribution.METADATA,
                    source=MetadataRef(field=field),
                    snippet=value.strip()[: SNIPPET_CONTEXT_CHARS * 2],
                    ambiguous=_is_ambiguous(name),
                )
            )
    for index, tag in enumerate(tags):
        name = person_shaped(tag)
        if name is None:
            continue
        mentions.append(
            RawMention(
                name=name,
                raw_span=tag.strip(),
                pattern_id="tag_person",
                reliability=TAG_PERSON_RELIABILITY,
                attribution=Attribution.METADATA,
                source=MetadataRef(field="tags", item_index=index),
                snippet=tag.strip()[: SNIPPET_CONTEXT_CHARS * 2],
                ambiguous=_is_ambiguous(name),
            )
        )
    return _dedupe(mentions)


def extract_from_segment(
    text: str,
    *,
    segment_index: int,
    diarization_label: str | None,
    start_seconds: float,
    suspect: bool = False,
) -> list[RawMention]:
    """Extract SELF/OTHER mentions from one transcript segment's text.

    The segment's ``suspect`` flag rides along on the source ref; the scoring
    layer applies its reliability penalty (extraction stays a pure record of
    what matched where).
    """
    if not text.strip():
        return []
    source = SegmentRef(
        segment_index=segment_index,
        diarization_label=diarization_label,
        start_seconds=start_seconds,
        suspect=suspect,
    )
    return _dedupe(_run_patterns(text, TRANSCRIPT_PATTERNS, source))
