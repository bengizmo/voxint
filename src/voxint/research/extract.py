"""Bounded HTML/plain-text extraction for fetched pages (issue #39).

Deliberately stdlib-only (``html.parser``): this is the first module in the
codebase that parses attacker-controlled markup, and a pure-Python tolerant
parser is a smaller supply-chain and memory-safety surface than a C parser
(lxml) or a large transitive tree (trafilatura). Extraction is best-effort and
non-semantic — the consumer is an LLM loop (issue #40) that tolerates messy
text; malformed HTML yields bounded partial text, never an exception.

Retrieved content is data, never instructions. Beyond that stance, the one
transformation applied is stripping characters that can smuggle *invisible*
instructions or forge log/terminal output: Unicode tag block (U+E0000..E007F),
zero-width/joiner formatting, bidi controls, and C0/C1 controls (newline/tab
survive as whitespace). Ordinary text — accents, CJK, emoji — passes through
untouched (no NFKC normalization: silently rewriting evidence would violate
the no-masking doctrine).
"""

import codecs
import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Whitespace runs inside character data (including source newlines — line
# structure comes from BLOCK tags, not source formatting) fold to one space.
_WS_RUN = re.compile(r"\s+")

# Content inside these elements is not prose (or is its own document tree that
# renders invisibly in a text extraction) — skipped wholesale, nesting-counted.
_SKIP_CONTENT_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "iframe"}
)
# Elements that terminate a text run; emitted as a newline so headings/rows
# don't fuse into one line. Everything else joins with a space.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header",
        "footer", "blockquote", "pre", "hr", "form", "dl", "dt", "dd",
    }
)

# Characters with no visible rendering that can carry hidden instructions or
# reorder/forge surrounding text. C0 (minus \t\n\r, handled as whitespace) and
# C1 controls; zero-width + joiner formatting chars; bidi embedding/override/
# isolate controls; BOM/word-joiner; interlinear annotation; Unicode tag block.
_HOSTILE_RANGES: tuple[tuple[int, int], ...] = (
    (0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), (0x7F, 0x9F),
    (0x200B, 0x200F),  # zero-width space/non-joiner/joiner, LRM/RLM
    (0x202A, 0x202E),  # bidi embedding/override
    (0x2060, 0x2064),  # word joiner, invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
    (0xFFF9, 0xFFFB),  # interlinear annotation controls
    (0xE0000, 0xE007F),  # Unicode tag block (invisible "tag" instructions)
)
_HOSTILE_TABLE = {
    cp: None for start, end in _HOSTILE_RANGES for cp in range(start, end + 1)
}


@dataclass(frozen=True)
class ExtractedText:
    text: str
    title: str
    truncated: bool


def sanitize_text(text: str) -> str:
    """Strip invisible/control characters that could smuggle instructions."""
    return text.translate(_HOSTILE_TABLE)


def decode_bytes(data: bytes, *, charset: str | None) -> str:
    """Decode a response body using the declared charset, else UTF-8.

    An unknown/undecodable charset degrades to UTF-8 with replacement — the
    output is best-effort prose for an LLM, and refusing a page over a charset
    label would refuse real content the operator asked for. Decode errors are
    replaced, never raised.
    """
    codec = "utf-8"
    if charset:
        try:
            codec = codecs.lookup(charset).name
        except LookupError:
            codec = "utf-8"
    return data.decode(codec, errors="replace")


class _TextExtractor(HTMLParser):
    """Tolerant single-pass text extractor with an emission cap.

    ``max_chars`` bounds the ACCUMULATED text — once reached, handlers become
    no-ops (the parser still consumes the rest of its input cheaply). The skip
    stack is a nesting counter, so `<div>` inside `<svg>` stays skipped and a
    stray `</script>` cannot underflow it.
    """

    def __init__(self, *, max_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self._max_chars = max_chars
        self._pieces: list[str] = []
        self._length = 0
        self._skip_depth = 0
        self._title_parts: list[str] = []
        self._title_length = 0
        self._in_title = False
        self.truncated = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        # Re-sanitize here: convert_charrefs decodes entity references AFTER
        # the caller's pre-parse sanitize pass, so `&#xE0069;`/`&#x202E;`-style
        # refs would otherwise resurrect the very characters that pass strips
        # (found in review — the pre-parse pass alone is NOT sufficient).
        data = sanitize_text(data)
        if self._in_title and self._title_length < 500:
            self._title_parts.append(data[: 500 - self._title_length])
            self._title_length += min(len(data), 500 - self._title_length)
        if self._skip_depth == 0:
            self._append(_WS_RUN.sub(" ", data))

    def _append(self, piece: str) -> None:
        remaining = self._max_chars - self._length
        if remaining <= 0:
            if piece.strip():
                self.truncated = True
            return
        if len(piece) > remaining:
            if piece[remaining:].strip():
                self.truncated = True
            piece = piece[:remaining]
        self._pieces.append(piece)
        self._length += len(piece)

    @property
    def title(self) -> str:
        return _collapse_whitespace("".join(self._title_parts))

    @property
    def text(self) -> str:
        return _collapse_text("".join(self._pieces))


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _collapse_text(text: str) -> str:
    """Collapse intra-line whitespace runs and blank-line runs."""
    lines = [" ".join(line.split()) for line in text.split("\n")]
    collapsed: list[str] = []
    for line in lines:
        if line:
            collapsed.append(line)
        elif collapsed and collapsed[-1] != "":
            collapsed.append("")
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def extract_html_text(data: bytes, *, charset: str | None, max_chars: int) -> ExtractedText:
    """Decode + parse HTML bytes into bounded, sanitized plain text.

    Sanitization runs BEFORE parsing so invisible characters cannot split a tag
    or entity in a way the cap then hides; the emission cap bounds accumulation
    inside the parser, so a huge page costs one linear pass, never an unbounded
    buffer. Malformed HTML degrades to whatever text the tolerant parser finds.
    """
    parser = _TextExtractor(max_chars=max_chars)
    parser.feed(sanitize_text(decode_bytes(data, charset=charset)))
    parser.close()
    text = parser.text[:max_chars]
    return ExtractedText(
        text=text,
        title=parser.title[:500],
        truncated=parser.truncated or len(parser.text) > len(text),
    )


def extract_plain_text(data: bytes, *, charset: str | None, max_chars: int) -> ExtractedText:
    """Decode + sanitize a text/plain body (no markup interpretation)."""
    text = _collapse_text(sanitize_text(decode_bytes(data, charset=charset)))
    clipped = text[:max_chars]
    return ExtractedText(
        text=clipped, title="", truncated=len(text) > len(clipped)
    )
