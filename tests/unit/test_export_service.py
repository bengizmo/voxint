"""Option validation and shared export dispatch contracts."""

import uuid
from unittest.mock import Mock

import pytest

from voxint.adjudication.transcript import TranscriptText
from voxint.export import TranscriptFormat
from voxint.export.filler_lists import DEFAULT_FILLER_LIST
from voxint.export.service import (
    DEFAULT_MARKDOWN_STYLE,
    ExportOptionError,
    MarkdownStyle,
    parse_fillers,
    parse_filter_value,
    parse_repeats,
    parse_style,
    render_run_transcript,
)


@pytest.mark.parametrize("name", ["fillers", "repeats"])
@pytest.mark.parametrize(
    "raw, expected", [(None, False), ("", False), ("keep", False), ("drop", True)]
)
def test_filter_value(name: str, raw: str | None, expected: bool) -> None:
    assert parse_filter_value(name, raw) is expected


@pytest.mark.parametrize("name", ["fillers", "repeats"])
@pytest.mark.parametrize("raw", ["bogus", "DROP", " keep"])
def test_filter_value_unknown(name: str, raw: str) -> None:
    with pytest.raises(ExportOptionError) as exc:
        parse_filter_value(name, raw)
    assert str(exc.value) == f"unknown {name} value {raw!r}; valid: keep, drop"


@pytest.mark.parametrize("name", ["fillers", "repeats"])
def test_filter_layout_error_precedes_value_error(name: str) -> None:
    parser = parse_fillers if name == "fillers" else parse_repeats
    with pytest.raises(ExportOptionError) as exc:
        parser("bogus", TranscriptFormat.TXT, None)
    assert str(exc.value) == f"{name} applies to md turns and txt turns only"


@pytest.mark.parametrize("raw", [None, ""])
def test_style_defaults_to_turns(raw: str | None) -> None:
    assert DEFAULT_MARKDOWN_STYLE is MarkdownStyle.TURNS
    assert parse_style(raw, TranscriptFormat.MARKDOWN) is MarkdownStyle.TURNS
    assert parse_style(raw, TranscriptFormat.TXT) is None
    assert parse_style(raw, None) is None


@pytest.mark.parametrize("style", ["blocks", "turns"])
def test_style_values(style: str) -> None:
    assert parse_style(style, TranscriptFormat.MARKDOWN) == style
    for fmt in (None, TranscriptFormat.JSON,
                TranscriptFormat.SRT, TranscriptFormat.VTT):
        with pytest.raises(ExportOptionError, match="style applies to the md and txt formats only"):
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
    with pytest.raises(ExportOptionError, match="fillers applies to md turns and txt turns only"):
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
            fillers=DEFAULT_FILLER_LIST,
        )
    assert not session.mock_calls


@pytest.mark.parametrize(
    "raw, expected", [(None, False), ("", False), ("keep", False), ("drop", True)]
)
def test_repeats_values(raw: str | None, expected: bool) -> None:
    assert parse_repeats(raw, TranscriptFormat.MARKDOWN, MarkdownStyle.TURNS) is expected
    assert not parse_repeats(None, TranscriptFormat.TXT, None)


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
def test_repeats_wrong_layout(
    raw: str, fmt: TranscriptFormat | None, style: MarkdownStyle | None
) -> None:
    with pytest.raises(ExportOptionError, match="repeats applies to md turns and txt turns only"):
        parse_repeats(raw, fmt, style)


def test_repeats_unknown() -> None:
    with pytest.raises(ExportOptionError, match="unknown repeats value 'bogus'; valid: keep, drop"):
        parse_repeats("bogus", TranscriptFormat.MARKDOWN, MarkdownStyle.TURNS)


@pytest.mark.parametrize(
    "fillers, detail",
    [
        (False, "repeats cannot be combined with a translation"),
        (True, "fillers cannot be combined with a translation"),
    ],
)
def test_renderer_rejects_repeats_with_translation_before_loading(
    fillers: bool, detail: str
) -> None:
    session = Mock()
    with pytest.raises(ExportOptionError, match=detail):
        render_run_transcript(
            session,
            uuid.uuid4(),
            TranscriptFormat.MARKDOWN,
            text=TranscriptText.RAW,
            translated_texts=[],
            fillers=DEFAULT_FILLER_LIST if fillers else None,
            drop_repeats=True,
        )
    assert not session.mock_calls


def test_renderer_rejects_repeats_outside_md_turns_before_loading() -> None:
    session = Mock()
    with pytest.raises(ExportOptionError, match="repeats applies to md turns and txt turns only"):
        render_run_transcript(
            session, uuid.uuid4(), TranscriptFormat.TXT, text=TranscriptText.RAW, drop_repeats=True
        )
    assert not session.mock_calls


def test_txt_turns_style() -> None:
    assert parse_style("turns", TranscriptFormat.TXT) is MarkdownStyle.TURNS


@pytest.mark.parametrize("raw", ["blocks", "bogus", "TURNS"])
def test_txt_rejects_other_styles(raw: str) -> None:
    with pytest.raises(ExportOptionError) as exc:
        parse_style(raw, TranscriptFormat.TXT)
    assert str(exc.value) == f"unknown style {raw!r} for txt; valid: turns"


@pytest.mark.parametrize("name", ["fillers", "repeats"])
@pytest.mark.parametrize(
    "raw, expected", [(None, False), ("", False), ("keep", False), ("drop", True)]
)
def test_txt_turn_filters(name: str, raw: str | None, expected: bool) -> None:
    parser = parse_fillers if name == "fillers" else parse_repeats
    assert parser(raw, TranscriptFormat.TXT, MarkdownStyle.TURNS) is expected


@pytest.mark.parametrize("name", ["fillers", "repeats"])
def test_txt_turn_filter_unknown(name: str) -> None:
    parser = parse_fillers if name == "fillers" else parse_repeats
    with pytest.raises(ExportOptionError) as exc:
        parser("bogus", TranscriptFormat.TXT, MarkdownStyle.TURNS)
    assert str(exc.value) == f"unknown {name} value 'bogus'; valid: keep, drop"


@pytest.mark.parametrize(
    "add, keep, expected",
    [
        (("you know", "I mean"), ("um",),
         "<!-- Filler words left out: 12. Preset en-1; also removed: you know, I mean; "
         "kept: um. The saved transcript is unchanged. -->"),
        (("you know", "I mean"), (),
         "<!-- Filler words left out: 12. Preset en-1; also removed: you know, I mean. "
         "The saved transcript is unchanged. -->"),
        ((), ("um", "hmm"),
         "<!-- Filler words left out: 12. Preset en-1; kept: um. "
         "The saved transcript is unchanged. -->"),
    ],
)
def test_filler_report_golden(
    add: tuple[str, ...], keep: tuple[str, ...], expected: str,
) -> None:
    from voxint.export.filler_lists import effective_filler_list
    from voxint.export.service import filler_report_comment

    assert filler_report_comment(effective_filler_list(add, keep), 12) == expected


@pytest.mark.parametrize("header", ["Empty transcript", None])
def test_empty_markdown_report(monkeypatch: pytest.MonkeyPatch, header: str | None) -> None:
    from voxint.export.filler_lists import effective_filler_list
    from voxint.export.service import render_run_transcript_report

    attribution = Mock(return_value=[])
    monkeypatch.setattr("voxint.export.service.attributed_turns", attribution)
    monkeypatch.setattr("voxint.export.service.export_title", lambda *a: header)
    result = render_run_transcript_report(
        Mock(), uuid.uuid4(), TranscriptFormat.MARKDOWN, text=TranscriptText.RAW,
        fillers=effective_filler_list(keep=["um"]),
    )
    prefix = "# Empty transcript\n\n" if header else ""
    assert result.content == (
        prefix + "<!-- Filler words left out: 0. Preset en-1; kept: um. "
        "The saved transcript is unchanged. -->\n"
    )
    assert result.fillers_removed == result.repeats_removed == 0
    attribution.assert_called_once()


@pytest.mark.parametrize(
    "fmt, style",
    [(TranscriptFormat.MARKDOWN, MarkdownStyle.BLOCKS), (TranscriptFormat.TXT, None),
     (TranscriptFormat.SRT, None), (TranscriptFormat.VTT, None), (TranscriptFormat.JSON, None)],
)
def test_unfiltered_report_counts(
    monkeypatch: pytest.MonkeyPatch, fmt: TranscriptFormat, style: MarkdownStyle | None,
) -> None:
    from voxint.export.service import render_run_transcript_report

    attribution = Mock(return_value=[])
    filters = Mock(side_effect=AssertionError("unexpected filtering"))
    monkeypatch.setattr("voxint.export.service.attributed_transcript", attribution)
    monkeypatch.setattr("voxint.export.service.apply_turn_filters", filters)
    result = render_run_transcript_report(
        Mock(), uuid.uuid4(), fmt, text=TranscriptText.RAW, style=style,
    )
    assert result.fillers_removed == result.repeats_removed == 0
    assert "Filler words left out" not in result.content
    attribution.assert_called_once()
    filters.assert_not_called()


def test_report_counts_and_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.test_turn_filters import piece, turn
    from voxint.export.service import render_run_transcript_report
    from voxint.export.turn_filters import apply_turn_filters

    attribution = Mock(return_value=[turn(piece("the um the cat and and dog", coarse=True))])
    filtering = Mock(wraps=apply_turn_filters)
    monkeypatch.setattr("voxint.export.service.attributed_turns", attribution)
    monkeypatch.setattr("voxint.export.service.apply_turn_filters", filtering)
    result = render_run_transcript_report(
        Mock(), uuid.uuid4(), TranscriptFormat.TXT, text=TranscriptText.RAW,
        style=MarkdownStyle.TURNS, timestamps=False,
        fillers=DEFAULT_FILLER_LIST, drop_repeats=True,
    )
    assert result.content == "Alex: the cat and dog\n"
    assert (result.fillers_removed, result.repeats_removed) == (1, 2)
    attribution.assert_called_once()
    filtering.assert_called_once()
    attribution.reset_mock()
    filtering.reset_mock()
    assert render_run_transcript(
        Mock(), uuid.uuid4(), TranscriptFormat.TXT, text=TranscriptText.RAW,
        style=MarkdownStyle.TURNS, timestamps=False,
        fillers=DEFAULT_FILLER_LIST, drop_repeats=True,
    ) == result.content
    attribution.assert_called_once()
    filtering.assert_called_once()
