"""Shared run export loading and rendering for CLI and HTTP transports."""

import enum
import uuid
from collections.abc import Sequence
from dataclasses import replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.adjudication.turns import attributed_turns, translated_turns
from voxint.api.presentation import friendly_media_label, title_from_snapshot
from voxint.db.models import DiarizationTurn, PipelineRun
from voxint.export import TranscriptFormat, render_transcript, to_markdown_turns, to_rttm
from voxint.export.reading import layout_turns


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


def export_title(session: Session, run_id: uuid.UUID) -> str:
    """Use the frozen title or friendly source basename for the export header."""
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    return friendly_media_label(title_from_snapshot(run.sidecar), run.media_item.source_path)


def render_run_transcript(
    session: Session, run_id: uuid.UUID, fmt: TranscriptFormat, *, text: TranscriptText,
    timestamps: bool = True, style: MarkdownStyle | None = None,
    translated_texts: Sequence[str] | None = None,
) -> str:
    """Load one attributed view and render it without transport-specific behavior."""
    resolved = parse_style(style, fmt)
    if fmt is TranscriptFormat.MARKDOWN and resolved is MarkdownStyle.TURNS:
        if translated_texts is None:
            turns = attributed_turns(session, run_id, text=text)
        else:
            try:
                turns = translated_turns(session, run_id, translated_texts)
            except ValueError as exc:
                raise TranslationMismatchError from exc
        return to_markdown_turns(
            layout_turns(turns), header=export_title(session, run_id), timestamps=timestamps,
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
