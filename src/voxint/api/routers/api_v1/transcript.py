"""Public API: transcript export in all formats."""

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from voxint.adjudication.transcript import (
    parse_transcript_text,
)
from voxint.api.api_app import ApiKeyDep, ApiSessionDep
from voxint.db.models import PipelineRun
from voxint.export import MEDIA_TYPES, TranscriptFormat
from voxint.export.filler_lists import DEFAULT_FILLER_LIST
from voxint.export.service import (
    ExportOptionError,
    parse_fillers,
    parse_repeats,
    parse_style,
    render_run_rttm,
    render_run_transcript,
)

router = APIRouter(prefix="/runs", tags=["transcript"])

_VALID_FORMATS = {"txt", "srt", "vtt", "json", "md", "rttm"}


@router.get("/{run_id}/transcript")
def export_transcript(
    run_id: uuid.UUID,
    identity: ApiKeyDep,
    session: ApiSessionDep,
    format: str = "json",
    text: str | None = None,
    timestamps: bool = True,
    style: str | None = None,
    fillers: str | None = None,
    repeats: str | None = None,
) -> Response:
    if format not in _VALID_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown format {format!r}; valid: {', '.join(sorted(_VALID_FORMATS))}",
        )

    fmt = None if format == "rttm" else TranscriptFormat(format)
    try:
        selected_style = parse_style(style, fmt)
        drop_fillers = parse_fillers(fillers, fmt, selected_style)
        drop_repeats = parse_repeats(repeats, fmt, selected_style)
    except ExportOptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    if fmt is None:
        return Response(content=render_run_rttm(session, run_id), media_type=MEDIA_TYPES["rttm"])

    try:
        variant = parse_transcript_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    content = render_run_transcript(
        session,
        run_id,
        fmt,
        text=variant,
        timestamps=timestamps,
        style=selected_style,
        fillers=DEFAULT_FILLER_LIST if drop_fillers else None,
        drop_repeats=drop_repeats,
    )
    return Response(content=content, media_type=MEDIA_TYPES[format])
