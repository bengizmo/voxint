"""Golden tests for the pure export formatters (no DB, no HTTP).

These pin the byte-level output both transports share, so a formatting change is
a visible, deliberate diff — not a silent drift between the API and the CLI.
"""

import json

import pytest

from voxint.adjudication.transcript import (
    TranscriptLine,
    TranscriptParagraph,
    paragraphize_transcript,
)
from voxint.export import (
    MEDIA_TYPES,
    TranscriptFormat,
    render_transcript,
    to_json,
    to_markdown,
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


def test_txt_without_timestamps_drops_bracket_column() -> None:
    # issue #52: a clean reading copy — just "speaker: text" per line.
    assert to_txt(LINES, timestamps=False) == (
        "Alice: Hello there.\n(no speaker): multi\nline\n"
    )


def test_txt_without_timestamps_empty_is_empty_string() -> None:
    assert to_txt([], timestamps=False) == ""


def test_txt_timestamps_default_is_unchanged() -> None:
    # The default keeps the original bytes — CLI/route parity and every existing
    # caller stay byte-identical.
    assert to_txt(LINES, timestamps=True) == to_txt(LINES)


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


MD_LINES = [
    TranscriptLine(start_seconds=0.0, end_seconds=2.5, speaker="Alice", text="Hello *there*."),
    TranscriptLine(start_seconds=2.5, end_seconds=5.0, speaker="Alice", text="Still here."),
    TranscriptLine(start_seconds=5.0, end_seconds=8.25, speaker="Bob", text="Yes & noted."),
]


def test_markdown_headings_and_merged_blockquote() -> None:
    # An H2 per contiguous same-speaker run; adjacent Alice lines merge into one
    # blockquote whose time range spans the whole run (0.00 → 5.00).
    assert to_markdown(MD_LINES) == (
        "## Alice\n\n"
        "> [00:00:00.000\u201300:00:05.000] Hello \\*there\\*. Still here.\n"
        "\n"
        "## Bob\n\n"
        "> [00:00:05.000\u201300:00:08.250] Yes &amp; noted.\n"
    )


def test_markdown_without_timestamps_drops_time_range() -> None:
    assert to_markdown(MD_LINES, timestamps=False) == (
        "## Alice\n\n"
        "> Hello \\*there\\*. Still here.\n"
        "\n"
        "## Bob\n\n"
        "> Yes &amp; noted.\n"
    )


def test_markdown_empty_is_empty_string() -> None:
    assert to_markdown([]) == ""


def test_markdown_speaker_return_starts_new_paragraph() -> None:
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="one"),
        TranscriptLine(start_seconds=1.0, end_seconds=2.0, speaker="B", text="two"),
        TranscriptLine(start_seconds=2.0, end_seconds=3.0, speaker="A", text="three"),
    ]
    assert to_markdown(lines, timestamps=False) == (
        "## A\n\n> one\n\n## B\n\n> two\n\n## A\n\n> three\n"
    )


def test_markdown_quotes_every_physical_line() -> None:
    # Embedded newlines in a segment stay as separate blockquote lines.
    lines = [TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="one\ntwo")]
    assert to_markdown(lines, timestamps=False) == "## A\n\n> one\n> two\n"


def test_markdown_escapes_markdown_and_html_control_chars() -> None:
    # Hostile/incidental text must render literally — no injected emphasis, code,
    # links, or raw HTML — in both the heading and the blockquote body.
    lines = [
        TranscriptLine(
            start_seconds=0.0,
            end_seconds=1.0,
            speaker="A <b>",
            text="see `code`, _em_, [link](x) and <script>&",
        )
    ]
    assert to_markdown(lines, timestamps=False) == (
        "## A &lt;b&gt;\n\n"
        "> see \\`code\\`, \\_em\\_, \\[link\\](x) and &lt;script&gt;&amp;\n"
    )


def test_markdown_defuses_line_leading_block_markers() -> None:
    # A physical line that opens with a block marker must render as literal prose
    # inside the blockquote, never as a heading, list item, or thematic break.
    lines = [
        TranscriptLine(
            start_seconds=0.0,
            end_seconds=1.0,
            speaker="A",
            text=(
                "# not a heading\n- not a bullet\n+ not a bullet\n"
                "1. not a list\n2) also not\n--- not a rule"
            ),
        )
    ]
    assert to_markdown(lines, timestamps=False) == (
        "## A\n\n"
        "> \\# not a heading\n"
        "> \\- not a bullet\n"
        "> \\+ not a bullet\n"
        "> 1\\. not a list\n"
        "> 2\\) also not\n"
        "> \\--- not a rule\n"
    )


def test_markdown_timestamp_prefix_keeps_first_line_defused() -> None:
    # With timestamps on, the first body line opens with the [range]; the marker
    # is still defused so turning timestamps off can never re-expose a heading.
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="# hi"),
    ]
    assert to_markdown(lines) == (
        "## A\n\n> [00:00:00.000\u201300:00:01.000] \\# hi\n"
    )


def test_markdown_defuses_setext_equals_underline() -> None:
    # A body line of ``===`` under a text line is a CommonMark setext H1 underline;
    # it must be defused so transcript content cannot forge a heading (the ``-``
    # setext form was already covered by the thematic-break case above).
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="Foo\n==="),
    ]
    assert to_markdown(lines, timestamps=False) == "## A\n\n> Foo\n> \\===\n"


def test_markdown_normalizes_bare_carriage_return_breakout() -> None:
    # A lone ``\r`` is a CommonMark line ending; without normalization
    # ``foo\r# Owned`` would emit ``> foo\r# Owned`` and break ``# Owned`` out of
    # the blockquote to a top-level heading. It must become a defused physical line.
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="foo\r# Owned"),
    ]
    assert to_markdown(lines, timestamps=False) == "## A\n\n> foo\n> \\# Owned\n"


def test_markdown_crlf_collapses_to_one_line_break() -> None:
    # CRLF must not double up into a blank blockquote line; it is one line ending.
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="foo\r\nbar"),
    ]
    assert to_markdown(lines, timestamps=False) == "## A\n\n> foo\n> bar\n"


def test_markdown_escapes_tilde_fence_and_table_pipe() -> None:
    # Tilde code fences (``~~~``) and GFM table pipes (``|``) are structure a
    # blockquote body could otherwise forge; both are inline-escaped literal.
    lines = [
        TranscriptLine(
            start_seconds=0.0,
            end_seconds=1.0,
            speaker="A",
            text="~~~\nh1 | h2\n--- | ---",
        )
    ]
    assert to_markdown(lines, timestamps=False) == (
        "## A\n\n> \\~\\~\\~\n> h1 \\| h2\n> \\--- \\| ---\n"
    )


def test_markdown_folds_speaker_newline_into_one_heading() -> None:
    # A crafted speaker name with an embedded newline cannot break out of the
    # `##` heading to forge its own structure; the name is folded to one line.
    lines = [
        TranscriptLine(
            start_seconds=0.0,
            end_seconds=1.0,
            speaker="Alice\n## forged",
            text="hi",
        )
    ]
    assert to_markdown(lines, timestamps=False) == "## Alice ## forged\n\n> hi\n"


def test_paragraphize_merges_adjacent_same_speaker() -> None:
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=2.5, speaker="Alice", text="Hello."),
        TranscriptLine(start_seconds=2.5, end_seconds=5.0, speaker="Alice", text="Still here."),
    ]
    assert paragraphize_transcript(lines) == [
        TranscriptParagraph(
            speaker="Alice", start_seconds=0.0, end_seconds=5.0, text="Hello. Still here."
        )
    ]


def test_paragraphize_preserves_chronology_on_speaker_return() -> None:
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="one"),
        TranscriptLine(start_seconds=1.0, end_seconds=2.0, speaker="B", text="two"),
        TranscriptLine(start_seconds=2.0, end_seconds=3.0, speaker="A", text="three"),
    ]
    assert [p.speaker for p in paragraphize_transcript(lines)] == ["A", "B", "A"]


def test_paragraphize_empty_is_empty_list() -> None:
    assert paragraphize_transcript([]) == []


def test_paragraphize_join_respects_existing_whitespace() -> None:
    # A segment already ending in a newline gets no extra boundary space.
    lines = [
        TranscriptLine(start_seconds=0.0, end_seconds=1.0, speaker="A", text="line one\n"),
        TranscriptLine(start_seconds=1.0, end_seconds=2.0, speaker="A", text="line two"),
    ]
    assert paragraphize_transcript(lines)[0].text == "line one\nline two"


@pytest.mark.parametrize("fmt", list(TranscriptFormat))
def test_dispatcher_matches_direct_formatters(fmt: TranscriptFormat) -> None:
    direct = {
        TranscriptFormat.TXT: to_txt,
        TranscriptFormat.SRT: to_srt,
        TranscriptFormat.VTT: to_vtt,
        TranscriptFormat.JSON: to_json,
        TranscriptFormat.MARKDOWN: to_markdown,
    }[fmt]
    assert render_transcript(LINES, fmt) == direct(LINES)


def test_dispatcher_timestamps_flag_affects_only_txt_and_markdown() -> None:
    # TXT and Markdown honor timestamps=False; the flag is inert for the other
    # formats (subtitle timing is structural, JSON keys are a frozen contract).
    assert render_transcript(LINES, TranscriptFormat.TXT, timestamps=False) == to_txt(
        LINES, timestamps=False
    )
    assert render_transcript(
        LINES, TranscriptFormat.MARKDOWN, timestamps=False
    ) == to_markdown(LINES, timestamps=False)
    for fmt, direct in (
        (TranscriptFormat.SRT, to_srt),
        (TranscriptFormat.VTT, to_vtt),
        (TranscriptFormat.JSON, to_json),
    ):
        assert render_transcript(LINES, fmt, timestamps=False) == direct(LINES)


def test_media_types_cover_every_cli_format() -> None:
    # Every CLI/route format (incl. rttm) must have a declared content type.
    assert set(MEDIA_TYPES) == {"txt", "srt", "vtt", "json", "md", "rttm"}
    assert MEDIA_TYPES["json"].startswith("application/json")
    assert MEDIA_TYPES["md"].startswith("text/markdown")
