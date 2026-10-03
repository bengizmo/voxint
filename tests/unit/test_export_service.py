"""Option validation and shared export dispatch contracts."""

import uuid
from unittest.mock import Mock

import pytest

from voxint.adjudication.transcript import TranscriptText
from voxint.export import TranscriptFormat
from voxint.export.service import (
    DEFAULT_MARKDOWN_STYLE,
    ExportOptionError,
    MarkdownStyle,
    parse_fillers,
    parse_style,
    render_run_transcript,
)


@pytest.mark.parametrize("raw", [None, ""])
def test_style_defaults_to_turns(raw: str | None) -> None:
    assert DEFAULT_MARKDOWN_STYLE is MarkdownStyle.TURNS
    assert parse_style(raw, TranscriptFormat.MARKDOWN) is MarkdownStyle.TURNS
    assert parse_style(raw, TranscriptFormat.TXT) is None
    assert parse_style(raw, None) is None


@pytest.mark.parametrize("style", ["blocks", "turns"])
def test_style_values(style: str) -> None:
    assert parse_style(style, TranscriptFormat.MARKDOWN) == style
    for fmt in (None, TranscriptFormat.TXT, TranscriptFormat.JSON,
                TranscriptFormat.SRT, TranscriptFormat.VTT):
        with pytest.raises(ExportOptionError, match="style applies to the md format only"):
            parse_style(style, fmt)


def test_unknown_style() -> None:
    with pytest.raises(ExportOptionError, match="valid: turns, blocks"):
        parse_style("bogus", TranscriptFormat.MARKDOWN)


@pytest.mark.parametrize(
    "raw, expected", [(None, False), ("", False), ("keep", False), ("drop", True)]
)
def test_fillers_values(raw: str | None, expected: bool) -> None:
    assert parse_fillers(raw, TranscriptFormat.MARKDOWN, MarkdownStyle.TURNS) is expected


@pytest.mark.parametrize("raw", [None, ""])
def test_fillers_absent_for_other_formats(raw: str | None) -> None:
    assert not parse_fillers(raw, TranscriptFormat.TXT, None)


@pytest.mark.parametrize("raw", ["keep", "drop"])
@pytest.mark.parametrize(
    "fmt, style",
    [
        (None, None),
        (TranscriptFormat.TXT, None),
        (TranscriptFormat.JSON, None),
        (TranscriptFormat.SRT, None),
        (TranscriptFormat.VTT, None),
        (TranscriptFormat.MARKDOWN, MarkdownStyle.BLOCKS),
        (TranscriptFormat.MARKDOWN, None),
    ],
)
def test_fillers_wrong_layout(
    raw: str, fmt: TranscriptFormat | None, style: MarkdownStyle | None
) -> None:
    with pytest.raises(ExportOptionError, match="fillers applies to the md turns style only"):
        parse_fillers(raw, fmt, style)


@pytest.mark.parametrize("raw", ["bogus", "DROP", " keep"])
def test_fillers_unknown(raw: str) -> None:
    with pytest.raises(
        ExportOptionError, match=f"unknown fillers value {raw!r}; valid: keep, drop"
    ):
        parse_fillers(raw, TranscriptFormat.MARKDOWN, MarkdownStyle.TURNS)


def test_renderer_rejects_translation_before_loading() -> None:
    session = Mock()
    with pytest.raises(ExportOptionError, match="fillers cannot be combined with a translation"):
        render_run_transcript(
            session,
            uuid.uuid4(),
            TranscriptFormat.MARKDOWN,
            text=TranscriptText.RAW,
            translated_texts=[],
            drop_fillers=True,
        )
    assert not session.mock_calls
