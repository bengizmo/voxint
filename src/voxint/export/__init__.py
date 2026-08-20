"""Pure transcript/diarization formatters — the one place output bytes are shaped.

Both transports (the API export routes and the ``voxint export`` CLI) render
through :func:`render_transcript` (and :func:`to_rttm`), so a downloaded ``.srt``
and a piped ``voxint export … --format srt`` are byte-identical by construction —
there is no second formatting path to drift.

Everything here is pure: ``Sequence[TranscriptLine] -> str`` (or turns -> str for
RTTM), no DB, no HTTP, no I/O — so the formats unit-test directly against golden
strings. Callers own the DB read and the transport's error/response mapping.
"""

import enum
import json
import re
from collections.abc import Sequence
from typing import Protocol, assert_never

from voxint.adjudication.transcript import TranscriptLine, paragraphize_transcript


class TranscriptFormat(enum.StrEnum):
    """The transcript-line output formats (RTTM is diarization, handled apart)."""

    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"
    MARKDOWN = "md"


# Content types for HTTP responses, keyed by every CLI/route format including
# rttm. Kept beside the formatters so the routes never invent a media type that
# disagrees with what the formatter actually produces (e.g. JSON as text/plain).
MEDIA_TYPES: dict[str, str] = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "rttm": "text/plain; charset=utf-8",
}


class RttmTurn(Protocol):
    """A diarization turn as RTTM needs it: interval + raw local label.

    Structural, so a DB ``DiarizationTurn`` row and a test stub both satisfy it
    without importing the ORM here (keeps this module pure/db-free).
    """

    @property
    def start_seconds(self) -> float: ...
    @property
    def end_seconds(self) -> float: ...
    @property
    def label(self) -> str: ...


def _hms_parts(seconds: float) -> tuple[int, int, int, int]:
    """Split a time offset into (h, m, s, ms), rounding to the nearest ms.

    Rounding happens once, on the total millisecond count, so the carry
    propagates cleanly: 3599.9999 s rounds to 3600000 ms → 01:00:00,000, never
    00:59:60,000. Negative offsets are clamped to zero — a subtitle timestamp is
    never before the start of media.
    """
    total_ms = round(max(seconds, 0.0) * 1000.0)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return h, m, s, ms


def _timestamp(seconds: float, *, sep: str) -> str:
    """``HH:MM:SS<sep>mmm`` — ``sep`` is ',' for SRT and '.' for WebVTT."""
    h, m, s, ms = _hms_parts(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _cue_text(line: TranscriptLine) -> str:
    """Subtitle cue payload: the speaker attribution then the segment text.

    Speaker on its own line (rather than an inline ``name: text`` prefix) keeps
    multi-line segment text readable and matches how players render a speaker
    tag above the line.

    ``-->`` is the SRT/VTT cue-timing delimiter and is outright forbidden inside a
    WebVTT cue payload; if it appears in speaker or transcript text it corrupts the
    cue. Neutralize it to ``->`` — the only content mutation these subtitle formats
    make, and only in this pathological case (JSON/TXT keep the text verbatim).
    """
    speaker = line.speaker.replace("-->", "->")
    text = line.text.replace("-->", "->")
    return f"{speaker}:\n{text}"


def to_txt(lines: Sequence[TranscriptLine], *, timestamps: bool = True) -> str:
    """The plain-text transcript (the original ``export.txt`` format).

    ``[   start     end] speaker: text`` per line; a trailing newline only when
    there is at least one line, so an empty run yields an empty file.

    ``timestamps=False`` drops the ``[start end]`` bracket column, yielding
    ``speaker: text`` — a clean reading copy for quoting into a document (issue
    #52). The default keeps the byte-for-byte original output, so every existing
    caller and the CLI/route parity are unchanged.
    """
    if timestamps:
        body = "\n".join(
            f"[{line.start_seconds:9.2f} {line.end_seconds:9.2f}] {line.speaker}: {line.text}"
            for line in lines
        )
    else:
        body = "\n".join(f"{line.speaker}: {line.text}" for line in lines)
    return body + ("\n" if lines else "")


def to_srt(lines: Sequence[TranscriptLine]) -> str:
    """SubRip: 1-based index, ``HH:MM:SS,mmm --> …`` timing, cue, blank line."""
    blocks = []
    for i, line in enumerate(lines, start=1):
        start = _timestamp(line.start_seconds, sep=",")
        end = _timestamp(line.end_seconds, sep=",")
        blocks.append(f"{i}\n{start} --> {end}\n{_cue_text(line)}\n")
    return "\n".join(blocks)


def to_vtt(lines: Sequence[TranscriptLine]) -> str:
    """WebVTT: the ``WEBVTT`` header then dot-separated cues (no cue numbers).

    The header is always followed by the mandatory blank line (``WEBVTT\\n\\n``) —
    the file stays spec-valid even for an empty transcript.
    """
    cues = []
    for line in lines:
        start = _timestamp(line.start_seconds, sep=".")
        end = _timestamp(line.end_seconds, sep=".")
        cues.append(f"{start} --> {end}\n{_cue_text(line)}\n")
    return "WEBVTT\n\n" + "\n".join(cues)


def transcript_payload(lines: Sequence[TranscriptLine]) -> "list[dict[str, object]]":
    """The one segment-object shape every JSON transport emits.

    Shared by :func:`to_json` (the pinned bare-array transcript export) and the
    run-level export envelope (issue #36), so the two can never drift on what a
    segment looks like.
    """
    return [
        {
            "start_seconds": line.start_seconds,
            "end_seconds": line.end_seconds,
            "speaker": line.speaker,
            "text": line.text,
        }
        for line in lines
    ]


def to_json(lines: Sequence[TranscriptLine]) -> str:
    """A stable UTF-8 array of ``{start_seconds, end_seconds, speaker, text}``.

    ``ensure_ascii=False`` keeps Unicode text intact; the fixed key order and
    2-space indent make the output diff-stable and human-readable.
    """
    return json.dumps(transcript_payload(lines), ensure_ascii=False, indent=2) + "\n"


# Inline Markdown control characters backslash-escaped so transcript text renders
# literally: no injected emphasis, code spans, strikethrough/tilde fences, link
# syntax, or table pipes. HTML-significant characters (&, <, >) are entity-encoded
# separately in _md_escape so raw HTML in the text can never activate. Line-leading
# block markers (#, =, -, +, ordered lists) are neutralized per physical line by
# _md_defuse_block_start, since those are position-sensitive and _md_escape cannot
# see them. A bare URL is left as text; a GFM renderer may still autolink it, which
# is cosmetic and safe. TXT/JSON stay verbatim; only Markdown (like SRT/VTT's -->
# neutralization) mutates content, and only to defuse it.
_MD_INLINE_ESCAPES = {
    "\\": "\\\\",
    "`": "\\`",
    "*": "\\*",
    "_": "\\_",
    "[": "\\[",
    "]": "\\]",
    "~": "\\~",
    "|": "\\|",
}

# A line-leading ordered-list marker (``1.`` / ``12)``) followed by whitespace or
# end of line; CommonMark starts a list on it. Matched against an already
# inline-escaped line so it can be backslash-defused per physical line.
_MD_ORDERED_LIST = re.compile(r"^(\s*)(\d+)([.)])(?=\s|$)")

# Line-leading block markers CommonMark reads at the start of a physical line:
# ``#`` (ATX heading), ``-`` / ``+`` (bullet, and ``-`` also a thematic break or
# setext underline), and ``=`` (the setext H1 underline). ``*``/``_`` are already
# inline-escaped; ``>`` is entity-encoded; so those need no line-position handling
# here.
_MD_BLOCK_LEADERS = frozenset({"#", "=", "-", "+"})


# EN DASH for the Markdown time range, matching the on-screen transcript view's
# ``[start-end]``. Written as an escape so it is unambiguous in source (ruff
# RUF001) while emitting the same byte the reading view uses.
_TIME_RANGE_DASH = "\u2013"


def format_timespan(start_seconds: float, end_seconds: float) -> str:
    """``[HH:MM:SS.mmm\u2013HH:MM:SS.mmm]`` reading-timestamp range.

    The single formatting truth shared by the Markdown export and the on-screen
    read mode, so the two can never drift on how a paragraph's time span reads.
    """
    start = _timestamp(start_seconds, sep=".")
    end = _timestamp(end_seconds, sep=".")
    return f"[{start}{_TIME_RANGE_DASH}{end}]"


def _md_escape(text: str) -> str:
    """Neutralize Markdown and raw-HTML control syntax in ``text``.

    Backslash-escapes the inline Markdown specials (\\ ` * _ [ ]) and
    entity-encodes ``& < >`` so a hostile or incidental transcript string cannot
    inject emphasis, a code span, link syntax, or raw HTML into the rendered
    document. Line-leading block markers are handled separately, per physical
    line, by :func:`_md_defuse_block_start`. Applied to both the speaker heading
    and the blockquote body.
    """
    out: list[str] = []
    for ch in text:
        escaped = _MD_INLINE_ESCAPES.get(ch)
        if escaped is not None:
            out.append(escaped)
        elif ch == "&":
            out.append("&amp;")
        elif ch == "<":
            out.append("&lt;")
        elif ch == ">":
            out.append("&gt;")
        else:
            out.append(ch)
    return "".join(out)


def _normalize_line_breaks(text: str) -> str:
    """Fold every character a Markdown renderer may treat as a line ending to LF.

    CommonMark ends lines on LF / CR / CRLF only, so ``\\r\\n``/``\\r`` are the
    load-bearing case (a lone ``\\r`` left in the text would let ``foo\\r# x``
    break out of the blockquote to a top-level heading). U+2028 (line separator),
    U+2029 (paragraph separator), and NEL (U+0085) are not CommonMark line endings,
    but assorted non-compliant preview pipelines do break on them; folding them to
    LF here is cheap defense in depth, so each becomes a real, ``> ``-prefixed,
    block-defused physical line on those renderers too rather than an escape.
    """
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\x85", "\n")
    )


def _md_defuse_block_start(line: str) -> str:
    """Backslash-escape a line-leading block marker so one physical line renders
    as literal prose inside the blockquote, never as a heading, list item, or
    thematic break.

    Runs on an already inline-escaped line (see :func:`_md_escape`), closing the
    position-sensitive markers that escaper cannot see: leading ``#`` / ``=`` /
    ``-`` / ``+`` and an ordered-list ``1.`` / ``1)``. ``*``/``_`` are already
    escaped and ``>`` is entity-encoded, so a nested blockquote or emphasis break
    cannot form. Leading indent is measured over spaces AND tabs, since a tab
    before a marker (``\\t#``) is otherwise missed and the marker stays live; the
    caller also strips leading horizontal whitespace, so this is belt-and-braces.
    """
    stripped = line.lstrip(" \t")
    indent = line[: len(line) - len(stripped)]
    if stripped[:1] in _MD_BLOCK_LEADERS:
        return f"{indent}\\{stripped}"
    ordered = _MD_ORDERED_LIST.match(line)
    if ordered is not None:
        return f"{ordered.group(1)}{ordered.group(2)}\\{ordered.group(3)}{line[ordered.end():]}"
    return line


def to_markdown(lines: Sequence[TranscriptLine], *, timestamps: bool = True) -> str:
    """Readable Markdown: an ``## Speaker`` heading per contiguous same-speaker run
    followed by one blockquote paragraph.

    Adjacent same-speaker lines are merged (via :func:`paragraphize_transcript`),
    so the export reads as prose rather than reproducing per-segment noise. Every
    physical line of the paragraph is prefixed ``> ``; with ``timestamps=True``
    the paragraph body opens with ``[HH:MM:SS.mmm-HH:MM:SS.mmm]`` spanning the run.
    Speaker and text are escaped for inline specials, raw HTML, and line-leading
    block markers (:func:`_md_escape` + :func:`_md_defuse_block_start`), and the
    speaker heading is collapsed to one line, so content stays literal and cannot
    forge document structure. A bare URL stays as text (a GFM renderer may still
    autolink it, which is harmless). Empty input yields ``""``; non-empty output
    ends with exactly one newline.
    """
    blocks: list[str] = []
    for para in paragraphize_transcript(lines):
        # A speaker name is a single heading line: fold every line-break-ish char
        # (see _normalize_line_breaks) to a space so a crafted name cannot break out
        # of the `##` and forge its own headings.
        speaker = _normalize_line_breaks(para.speaker).replace("\n", " ")
        heading = f"## {_md_escape(speaker)}"
        # Normalize line breaks before splitting so a bare CR (or a Unicode
        # separator) becomes a real, block-defused physical line rather than
        # smuggling structure past ``.split("\n")`` (``foo\r# x`` would otherwise
        # break ``# x`` out of the quote). Each physical line is then stripped of
        # leading/trailing horizontal whitespace before defusing: four leading
        # spaces or a tab would otherwise open an indented code block inside the
        # quote (forged block structure, and our own backslash escapes would render
        # verbatim inside it), and a trailing double-space would force a ``<br>``.
        # Leading/trailing whitespace is insignificant to a prose paragraph, so the
        # strip is render-equivalent for real transcript text.
        body = _normalize_line_breaks(para.text)
        body_lines = [
            _md_defuse_block_start(_md_escape(line.strip(" \t")))
            for line in body.split("\n")
        ]
        if timestamps:
            body_lines[0] = (
                f"{format_timespan(para.start_seconds, para.end_seconds)} {body_lines[0]}"
            )
        quoted = "\n".join(f"> {line}" for line in body_lines)
        blocks.append(f"{heading}\n\n{quoted}\n")
    return "\n".join(blocks)


def to_rttm(turns: Sequence[RttmTurn], file_id: str) -> str:
    """NIST RTTM diarization output — one ``SPEAKER`` line per turn.

    ``SPEAKER <file_id> 1 <start> <dur> <NA> <NA> <label> <NA> <NA>``. RTTM is
    the diarization interchange format: it carries the **raw local labels**
    (``SPEAKER_00`` …), never adjudicated speaker names, so it round-trips
    against diarization scoring tools. ``file_id`` is the run UUID.
    """
    # Duration is clamped non-negative so a stray inverted interval can never emit
    # a malformed record (the DB's diarization_turns_interval_check already forbids
    # end <= start, so this only defends the pure function against bad callers).
    lines = [
        f"SPEAKER {file_id} 1 {t.start_seconds:.3f}"
        f" {max(t.end_seconds - t.start_seconds, 0.0):.3f}"
        f" <NA> <NA> {t.label} <NA> <NA>"
        for t in turns
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_transcript(
    lines: Sequence[TranscriptLine], fmt: TranscriptFormat, *, timestamps: bool = True
) -> str:
    """Render transcript lines in ``fmt`` — the shared API/CLI dispatch point.

    RTTM is deliberately not here: it consumes diarization turns, not transcript
    lines, so callers reach :func:`to_rttm` directly.

    ``timestamps`` is meaningful for TXT (its bracket column is optional) and
    Markdown (its per-paragraph time range is optional); SRT/VTT cue timing is
    structural and JSON keys are a frozen contract, so the flag is intentionally
    inert for those formats rather than corrupting them.
    """
    match fmt:
        case TranscriptFormat.TXT:
            return to_txt(lines, timestamps=timestamps)
        case TranscriptFormat.SRT:
            return to_srt(lines)
        case TranscriptFormat.VTT:
            return to_vtt(lines)
        case TranscriptFormat.JSON:
            return to_json(lines)
        case TranscriptFormat.MARKDOWN:
            return to_markdown(lines, timestamps=timestamps)
        case _:  # a new TranscriptFormat member without a case is a bug, not a None body
            assert_never(fmt)
