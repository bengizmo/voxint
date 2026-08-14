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
from collections.abc import Sequence
from typing import Protocol, assert_never

from voxint.adjudication.transcript import TranscriptLine


class TranscriptFormat(enum.StrEnum):
    """The transcript-line output formats (RTTM is diarization, handled apart)."""

    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


# Content types for HTTP responses, keyed by every CLI/route format including
# rttm. Kept beside the formatters so the routes never invent a media type that
# disagrees with what the formatter actually produces (e.g. JSON as text/plain).
MEDIA_TYPES: dict[str, str] = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
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


def to_txt(lines: Sequence[TranscriptLine]) -> str:
    """The bracketed plain-text transcript (the original ``export.txt`` format).

    ``[   start     end] speaker: text`` per line; a trailing newline only when
    there is at least one line, so an empty run yields an empty file.
    """
    body = "\n".join(
        f"[{line.start_seconds:9.2f} {line.end_seconds:9.2f}] {line.speaker}: {line.text}"
        for line in lines
    )
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


def to_json(lines: Sequence[TranscriptLine]) -> str:
    """A stable UTF-8 array of ``{start_seconds, end_seconds, speaker, text}``.

    ``ensure_ascii=False`` keeps Unicode text intact; the fixed key order and
    2-space indent make the output diff-stable and human-readable.
    """
    payload = [
        {
            "start_seconds": line.start_seconds,
            "end_seconds": line.end_seconds,
            "speaker": line.speaker,
            "text": line.text,
        }
        for line in lines
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


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


def render_transcript(lines: Sequence[TranscriptLine], fmt: TranscriptFormat) -> str:
    """Render transcript lines in ``fmt`` — the shared API/CLI dispatch point.

    RTTM is deliberately not here: it consumes diarization turns, not transcript
    lines, so callers reach :func:`to_rttm` directly.
    """
    match fmt:
        case TranscriptFormat.TXT:
            return to_txt(lines)
        case TranscriptFormat.SRT:
            return to_srt(lines)
        case TranscriptFormat.VTT:
            return to_vtt(lines)
        case TranscriptFormat.JSON:
            return to_json(lines)
        case _:  # a new TranscriptFormat member without a case is a bug, not a None body
            assert_never(fmt)
