"""Shared run export loading and rendering for CLI and HTTP transports."""

import enum
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.adjudication.transcript import TranscriptText, attributed_transcript
from voxint.adjudication.turns import attributed_turns, translated_turns
from voxint.adjudication.word_marks import effective_marks
from voxint.api.presentation import (
    format_recorded_date,
    friendly_media_label,
    recorded_from_snapshot,
    title_from_snapshot,
)
from voxint.db.models import DiarizationTurn, PipelineRun, TranscriptSegment
from voxint.export import (
    TranscriptFormat,
    format_clock,
    render_transcript,
    to_markdown_turns,
    to_rttm,
    to_txt_turns,
)
from voxint.export.filler_lists import FillerList
from voxint.export.fillers import UnplaceableWordMarkError
from voxint.export.reading import layout_turns
from voxint.export.turn_filters import apply_turn_filters


class MarkdownStyle(enum.StrEnum):
    """Available Markdown layouts; txt uses TURNS only."""

    TURNS = "turns"
    BLOCKS = "blocks"


# Transcript Markdown defaults to the shared reading layout.
DEFAULT_MARKDOWN_STYLE = MarkdownStyle.TURNS


class ExportOptionError(ValueError):
    """An invalid export option combination, safe to show to the operator."""


class TranslationMismatchError(Exception):
    """The translation no longer has one text per transcript emission."""


class WordMarkPlacementError(Exception):
    """The requested text cannot place effective marks, safe to show to the operator."""

    def __init__(self, session: Session, error: UnplaceableWordMarkError) -> None:
        self.segment_ids = {mark[0] for mark in error.marks}
        starts = session.scalars(select(TranscriptSegment.start_seconds).where(
            TranscriptSegment.id.in_(self.segment_ids),
        ).order_by(TranscriptSegment.start_seconds)).all()

        def clock(start: float) -> str:
            hours, minutes, seconds = format_clock(start).strip("[]").split(":")
            return f"{int(hours)}:{minutes}:{seconds}" if int(hours) else f"{minutes}:{seconds}"

        clocks = ", ".join(dict.fromkeys(clock(start) for start in starts))
        super().__init__(
            f"The requested text cannot place the clean-up marks on segments at {clocks}."
            " Clear the marks in the review console, or export with fillers kept."
        )


def parse_style(raw: str | None, fmt: TranscriptFormat | None) -> MarkdownStyle | None:
    """Resolve md/txt layouts and reject styles for other formats, including RTTM."""
    if raw in (None, ""):
        return DEFAULT_MARKDOWN_STYLE if fmt is TranscriptFormat.MARKDOWN else None
    if fmt is TranscriptFormat.TXT:
        if raw == "turns":
            return MarkdownStyle.TURNS
        raise ExportOptionError(f"unknown style {raw!r} for txt; valid: turns")
    if fmt is not TranscriptFormat.MARKDOWN:
        raise ExportOptionError("style applies to the md and txt formats only")
    try:
        return MarkdownStyle(raw)
    except ValueError as exc:
        raise ExportOptionError(f"unknown style {raw!r}; valid: turns, blocks") from exc


def parse_filter_value(name: str, raw: str | None) -> bool:
    """Parse a keep/drop filter value independently of the export layout."""
    if raw in (None, "", "keep"):
        return False
    if raw == "drop":
        return True
    raise ExportOptionError(f"unknown {name} value {raw!r}; valid: keep, drop")


def _parse_turn_filter(
    name: str,
    raw: str | None,
    fmt: TranscriptFormat | None,
    style: MarkdownStyle | None,
) -> bool:
    if raw in (None, ""):
        return False
    if (
        fmt not in (TranscriptFormat.MARKDOWN, TranscriptFormat.TXT)
        or style is not MarkdownStyle.TURNS
    ):
        raise ExportOptionError(f"{name} applies to md turns and txt turns only")
    return parse_filter_value(name, raw)


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


@dataclass(frozen=True)
class RenderedTranscript:
    content: str
    fillers_removed: int
    repeats_removed: int
    omitted: int = 0
    kept: int = 0


def filler_report_comment(
    fillers: FillerList, removed: int, *, omitted: int = 0, kept: int = 0,
) -> str:
    """Describe the effective custom list and this export's removal count."""
    detail = f"<!-- Filler words left out: {removed}. Preset {fillers.preset_version}"
    if fillers.extra_entries:
        detail += f"; also removed: {', '.join(fillers.extra_entries)}"
    if fillers.effective_keeps:
        detail += f"; kept: {', '.join(fillers.effective_keeps)}"
    if omitted:
        detail += f"; words you left out: {omitted}"
    if kept:
        detail += f"; fillers you kept: {kept}"
    return detail + ". The saved transcript is unchanged. -->"


def render_run_transcript(
    session: Session,
    run_id: uuid.UUID,
    fmt: TranscriptFormat,
    *,
    text: TranscriptText,
    timestamps: bool = True,
    style: MarkdownStyle | None = None,
    translated_texts: Sequence[str] | None = None,
    fillers: FillerList | None = None,
    drop_repeats: bool = False,
) -> str:
    """Render through the shared report without repeating attribution or filtering."""
    return render_run_transcript_report(
        session, run_id, fmt, text=text, timestamps=timestamps, style=style,
        translated_texts=translated_texts, fillers=fillers, drop_repeats=drop_repeats,
    ).content


def render_run_transcript_report(
    session: Session,
    run_id: uuid.UUID,
    fmt: TranscriptFormat,
    *,
    text: TranscriptText,
    timestamps: bool = True,
    style: MarkdownStyle | None = None,
    translated_texts: Sequence[str] | None = None,
    fillers: FillerList | None = None,
    drop_repeats: bool = False,
) -> RenderedTranscript:
    """Load one attributed view and report its render-time filter counts."""
    resolved = parse_style(style, fmt)
    if fillers is not None:
        if translated_texts is not None:
            raise ExportOptionError("fillers cannot be combined with a translation")
        parse_fillers("drop", fmt, resolved)
    if drop_repeats:
        if translated_texts is not None:
            raise ExportOptionError("repeats cannot be combined with a translation")
        parse_repeats("drop", fmt, resolved)
    if fmt in (TranscriptFormat.MARKDOWN, TranscriptFormat.TXT) and resolved is MarkdownStyle.TURNS:
        if translated_texts is None:
            turns = attributed_turns(session, run_id, text=text)
        else:
            try:
                turns = translated_turns(session, run_id, translated_texts)
            except ValueError as exc:
                raise TranslationMismatchError from exc
        try:
            filtered = apply_turn_filters(
                turns, fillers=fillers, drop_repeats=drop_repeats,
                marks=effective_marks(session, run_id) if fillers is not None else None,
            )
        except UnplaceableWordMarkError as exc:
            raise WordMarkPlacementError(session, exc) from exc
        paragraphs = layout_turns(filtered.turns)
        if fmt is TranscriptFormat.TXT:
            content = to_txt_turns(paragraphs, timestamps=timestamps)
        else:
            content = to_markdown_turns(
                paragraphs,
                header=export_title(session, run_id),
                timestamps=timestamps,
            )
            if fillers is not None and (
                not fillers.is_default or filtered.omitted or filtered.kept
            ):
                comment = filler_report_comment(
                    fillers, filtered.fillers_removed, omitted=filtered.omitted, kept=filtered.kept,
                )
                content += ("\n" if content else "") + comment + "\n"
        return RenderedTranscript(
            content, filtered.fillers_removed, filtered.repeats_removed,
            filtered.omitted, filtered.kept,
        )
    lines = attributed_transcript(session, run_id, text=text)
    if translated_texts is not None:
        if len(lines) != len(translated_texts):
            raise TranslationMismatchError
        lines = [
            replace(line, text=body) for line, body in zip(lines, translated_texts, strict=True)
        ]
    return RenderedTranscript(render_transcript(lines, fmt, timestamps=timestamps), 0, 0)


def render_run_rttm(session: Session, run_id: uuid.UUID) -> str:
    """Render raw diarization labels, loading only the required timing columns."""
    turns = session.execute(select(
        DiarizationTurn.start_seconds, DiarizationTurn.end_seconds, DiarizationTurn.label,
    ).where(DiarizationTurn.pipeline_run_id == run_id).order_by(DiarizationTurn.turn_index)).all()
    return to_rttm(turns, str(run_id))
