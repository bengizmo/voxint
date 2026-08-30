"""Public API: transcript export in all formats."""

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from voxint.adjudication.transcript import (
    attributed_transcript,
    parse_transcript_text,
)
from voxint.api.api_app import ApiKeyDep, ApiSessionDep
from voxint.db.models import DiarizationTurn, PipelineRun
from voxint.export import MEDIA_TYPES, TranscriptFormat, render_transcript, to_rttm

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
) -> Response:
    if format not in _VALID_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown format {format!r}; valid: {', '.join(sorted(_VALID_FORMATS))}",
        )

    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    if format == "rttm":
        from sqlalchemy import select

        turns = list(
            session.scalars(
                select(DiarizationTurn)
                .where(DiarizationTurn.pipeline_run_id == run_id)
                .order_by(DiarizationTurn.turn_index)
            )
        )
        content = to_rttm(turns, file_id=str(run_id))
        return Response(content=content, media_type=MEDIA_TYPES["rttm"])

    try:
        variant = parse_transcript_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    lines = attributed_transcript(session, run_id, text=variant)
    fmt = TranscriptFormat(format)
    content = render_transcript(lines, fmt, timestamps=timestamps)
    return Response(content=content, media_type=MEDIA_TYPES[format])
