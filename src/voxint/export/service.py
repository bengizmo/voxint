"""Shared run export loading and rendering for CLI and HTTP transports."""

import enum
import uuid
from collections.abc import Sequence
from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.adjudication.turns import attributed_turns, translated_turns
from voxint.api.presentation import (
    format_recorded_date,
    friendly_media_label,
    recorded_from_snapshot,
    title_from_snapshot,
)
from voxint.db.models import DiarizationTurn, PipelineRun
from voxint.export import TranscriptFormat, render_transcript, to_markdown_turns, to_rttm
from voxint.export.fillers import drop_fillers_with_seams
from voxint.export.reading import layout_turns
from voxint.export.repeats import drop_repeats as remove_repeats


class MarkdownStyle(enum.StrEnum):
    """Available Markdown layouts."""

    TURNS = "turns"
    BLOCKS = "blocks"


# Transcript Markdown defaults to the shared reading layout.
DEFAULT_MARKDOWN_STYLE = MarkdownStyle.TURNS


class ExportOptionError(ValueError):
    """An invalid export option combination, safe to show to the operator."""


class TranslationMismatchError(Exception):
    """The translation no longer has one text per transcript emission."""


def parse_style(raw: str | None, fmt: TranscriptFormat | None) -> MarkdownStyle | None:
    """Resolve defaults and reject styles for non-Markdown formats (including RTTM)."""
    if raw in (None, ""):
        return DEFAULT_MARKDOWN_STYLE if fmt is TranscriptFormat.MARKDOWN else None
    if fmt is not TranscriptFormat.MARKDOWN:
        raise ExportOptionError("style applies to the md format only")
    try:
        return MarkdownStyle(raw)
    except ValueError as exc:
        raise ExportOptionError(f"unknown style {raw!r}; valid: turns, blocks") from exc


def _parse_turn_filter(
    name: str,
    raw: str | None,
    fmt: TranscriptFormat | None,
    style: MarkdownStyle | None,
) -> bool:
    if raw in (None, ""):
        return False
    if fmt is not TranscriptFormat.MARKDOWN or style is not MarkdownStyle.TURNS:
        raise ExportOptionError(f"{name} applies to the md turns style only")
    if raw not in ("keep", "drop"):
        raise ExportOptionError(f"unknown {name} value {raw!r}; valid: keep, drop")
    return raw == "drop"


def parse_fillers(
    raw: str | None,
    fmt: TranscriptFormat | None,
    style: MarkdownStyle | None,
) -> bool:
    """Validate the opt-in filler filter before loading any transcript data."""
    return _parse_turn_filter("fillers", raw, fmt, style)


def parse_repeats(
    raw: str | None,
    fmt: TranscriptFormat | None,
    style: MarkdownStyle | None,
) -> bool:
    """Validate the opt-in repeated-word filter before loading any transcript data."""
    return _parse_turn_filter("repeats", raw, fmt, style)


def export_title(session: Session, run_id: uuid.UUID) -> str:
    """Use the frozen title or friendly source basename for the export header.

    A known recording date follows as ``<title> | <D Mon YYYY>``: the frozen
    sidecar ``recorded`` value first, else the source file's creation date. The
    Markdown renderer escapes the whole header, so the separator reads ``\\|``.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    title = friendly_media_label(title_from_snapshot(run.sidecar), run.media_item.source_path)
    recorded = recorded_from_snapshot(run.sidecar) or run.media_item.recorded_on
    return f"{title} | {format_recorded_date(recorded)}" if recorded is not None else title


def render_run_transcript(
    session: Session,
    run_id: uuid.UUID,
    fmt: TranscriptFormat,
    *,
    text: TranscriptText,
    timestamps: bool = True,
    style: MarkdownStyle | None = None,
    translated_texts: Sequence[str] | None = None,
    drop_fillers: bool = False,
    drop_repeats: bool = False,
) -> str:
    """Load one attributed view and render it without transport-specific behavior."""
    resolved = parse_style(style, fmt)
    if drop_fillers:
        if translated_texts is not None:
            raise ExportOptionError("fillers cannot be combined with a translation")
        parse_fillers("drop", fmt, resolved)
    if drop_repeats:
        if translated_texts is not None:
            raise ExportOptionError("repeats cannot be combined with a translation")
        parse_repeats("drop", fmt, resolved)
    if fmt is TranscriptFormat.MARKDOWN and resolved is MarkdownStyle.TURNS:
        if translated_texts is None:
            turns = attributed_turns(session, run_id, text=text)
        else:
            try:
                turns = translated_turns(session, run_id, translated_texts)
            except ValueError as exc:
                raise TranslationMismatchError from exc
        seams: list[frozenset[int]] | None = None
        if drop_fillers:
            cleaned = drop_fillers_with_seams(turns)
            turns = [turn for turn, _ in cleaned]
            seams = [turn_seams for _, turn_seams in cleaned]
        if drop_repeats:
            turns = remove_repeats(turns, seams)
        return to_markdown_turns(
            layout_turns(turns),
            header=export_title(session, run_id),
            timestamps=timestamps,
        )
    lines = attributed_transcript(session, run_id, text=text)
    if translated_texts is not None:
        if len(lines) != len(translated_texts):
            raise TranslationMismatchError
        lines = [
            replace(line, text=body) for line, body in zip(lines, translated_texts, strict=True)
        ]
    return render_transcript(lines, fmt, timestamps=timestamps)


def render_run_rttm(session: Session, run_id: uuid.UUID) -> str:
    """Render raw diarization labels, loading only the required timing columns."""
    turns = session.execute(select(
        DiarizationTurn.start_seconds, DiarizationTurn.end_seconds, DiarizationTurn.label,
    ).where(DiarizationTurn.pipeline_run_id == run_id).order_by(DiarizationTurn.turn_index)).all()
    return to_rttm(turns, str(run_id))
