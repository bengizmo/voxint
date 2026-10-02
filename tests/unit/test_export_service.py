"""Option validation and shared export dispatch contracts."""

import pytest

from voxint.export import TranscriptFormat
from voxint.export.service import ExportOptionError, MarkdownStyle, parse_style


@pytest.mark.parametrize("raw", [None, ""])
def test_style_defaults_to_blocks(raw: str | None) -> None:
    assert parse_style(raw, TranscriptFormat.MARKDOWN) is MarkdownStyle.BLOCKS
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
