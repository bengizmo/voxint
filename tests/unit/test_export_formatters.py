"""Golden tests for the pure export formatters (no DB, no HTTP).

These pin the byte-level output both transports share, so a formatting change is
a visible, deliberate diff — not a silent drift between the API and the CLI.
"""

import json

import pytest

from voxint.adjudication.transcript import TranscriptLine
from voxint.export import (
    MEDIA_TYPES,
    TranscriptFormat,
    render_transcript,
    to_json,
    to_rttm,
    to_srt,
    to_txt,
    to_vtt,
)


class _Turn:
    """Minimal RttmTurn stub (structural: start/end/label)."""

    def __init__(self, start: float, end: float, label: str) -> None:
        self.start_seconds = start
        self.end_seconds = end
        self.label = label


LINES = [
    TranscriptLine(start_seconds=0.0, end_seconds=2.5, speaker="Alice", text="Hello there."),
    TranscriptLine(
        start_seconds=2.5, end_seconds=5.0, speaker="(no speaker)", text="multi\nline"
    ),
]


def test_txt_matches_original_bracket_format() -> None:
    assert to_txt(LINES) == (
        "[     0.00      2.50] Alice: Hello there.\n"
        "[     2.50      5.00] (no speaker): multi\nline\n"
    )


def test_txt_empty_is_empty_string() -> None:
    # An empty run yields an empty file — no stray trailing newline.
    assert to_txt([]) == ""


def test_srt_cue_numbering_and_comma_timestamps() -> None:
    assert to_srt(LINES) == (
        "1\n00:00:00,000 --> 00:00:02,500\nAlice:\nHello there.\n"
        "\n"
        "2\n00:00:02,500 --> 00:00:05,000\n(no speaker):\nmulti\nline\n"
    )


def test_vtt_header_and_dot_timestamps() -> None:
    out = to_vtt(LINES)
    # Header + mandatory blank line before the first cue.
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.500\nAlice:\nHello there.\n" in out
    # WebVTT cues carry no 1-based index line (unlike SRT).
    assert "\n1\n" not in out


def test_vtt_empty_is_valid_header() -> None:
    # An empty transcript still yields a spec-valid WebVTT file (header + blank).
    assert to_vtt([]) == "WEBVTT\n\n"


def test_subtitle_cue_neutralizes_arrow_sequence() -> None:
    # "-->" is the cue-timing delimiter and is forbidden inside a WebVTT payload;
    # it is neutralized to "->" in SRT and VTT (but NOT in JSON/TXT).
    line = [TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="go --> stop")]
    assert "go -> stop" in to_srt(line) and "-->" not in to_srt(line).split("\n", 2)[2]
    assert "go -> stop" in to_vtt(line)
    assert "go --> stop" in to_json(line)  # structured formats keep text verbatim
    assert "go --> stop" in to_txt(line)


@pytest.mark.parametrize(
    ("seconds", "srt_ts"),
    [
        (0.0, "00:00:00,000"),
        (3599.9999, "01:00:00,000"),  # rounds up, carries m/h — never 00:59:60
        (3661.5, "01:01:01,500"),
        (0.0004, "00:00:00,000"),  # sub-ms rounds down
        (0.0006, "00:00:00,001"),  # sub-ms rounds up
        (-1.0, "00:00:00,000"),  # clamped, never negative
    ],
)
def test_timestamp_rounding_and_carry(seconds: float, srt_ts: str) -> None:
    line = [TranscriptLine(start_seconds=seconds, end_seconds=seconds, speaker="X", text="t")]
    assert to_srt(line).splitlines()[1].split(" --> ")[0] == srt_ts


def test_json_is_stable_utf8_array() -> None:
    out = to_json(
        [TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="Zöe", text="café")]
    )
    assert out.endswith("\n")
    assert "café" in out and "Zöe" in out  # not \u-escaped
    parsed = json.loads(out)
    assert parsed == [
        {"start_seconds": 0.0, "end_seconds": 1.0, "speaker": "Zöe", "text": "café"}
    ]


def test_json_empty_is_empty_array() -> None:
    assert json.loads(to_json([])) == []


def test_rttm_raw_labels_and_duration() -> None:
    turns = [_Turn(1.25, 4.5, "SPEAKER_00"), _Turn(4.5, 6.0, "SPEAKER_01")]
    assert to_rttm(turns, "run-1") == (
        "SPEAKER run-1 1 1.250 3.250 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "SPEAKER run-1 1 4.500 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
    )


def test_rttm_empty_is_empty_string() -> None:
    assert to_rttm([], "run-1") == ""


@pytest.mark.parametrize("fmt", list(TranscriptFormat))
def test_dispatcher_matches_direct_formatters(fmt: TranscriptFormat) -> None:
    direct = {
        TranscriptFormat.TXT: to_txt,
        TranscriptFormat.SRT: to_srt,
        TranscriptFormat.VTT: to_vtt,
        TranscriptFormat.JSON: to_json,
    }[fmt]
    assert render_transcript(LINES, fmt) == direct(LINES)


def test_media_types_cover_every_cli_format() -> None:
    # Every CLI/route format (incl. rttm) must have a declared content type.
    assert set(MEDIA_TYPES) == {"txt", "srt", "vtt", "json", "rttm"}
    assert MEDIA_TYPES["json"].startswith("application/json")
