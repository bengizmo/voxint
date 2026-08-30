"""Quote board API: save, manage, and export KWIC evidence (issue #338)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

from voxint.api.csrf import CSRF_QUOTE_MANAGE, CSRF_QUOTE_SAVE
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    _require_csrf,
    require_onboarded,
)
from voxint.api.saved_quotes import (
    QuoteDuplicateError,
    delete_quote,
    export_quotes_csv,
    save_quote,
    update_quote_note,
)
from voxint.db.models import MAX_QUOTE_NOTE_CHARS, Project

quotes_router = APIRouter(dependencies=[Depends(require_onboarded)])
router = quotes_router


@router.post("/quotes")
def save_quote_endpoint(
    request: Request,
    session: SessionDep,
    operator: OperatorDep,
    segment_id: Annotated[uuid.UUID, Form()],
    run_id: Annotated[uuid.UUID, Form()],
    search_query: Annotated[str, Form(min_length=1)],
    left_context: Annotated[str, Form()],
    hit: Annotated[str, Form()],
    right_context: Annotated[str, Form()],
    media_title: Annotated[str, Form()],
    start_seconds: Annotated[float, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
    speaker_name: Annotated[str | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Save a KWIC concordance row as project-scoped evidence."""
    _require_csrf(request, CSRF_QUOTE_SAVE, csrf_token)
    if note and len(note) > MAX_QUOTE_NOTE_CHARS:
        return JSONResponse(
            {"ok": False, "error": f"Note must be {MAX_QUOTE_NOTE_CHARS} characters or fewer."},
            status_code=422,
        )
    try:
        quote = save_quote(
            session,
            run_id=run_id,
            segment_id=segment_id,
            search_query=search_query,
            left_context=left_context,
            hit=hit,
            right_context=right_context,
            speaker_name=speaker_name,
            media_title=media_title,
            start_seconds=start_seconds,
            operator=operator,
            note=note,
        )
    except ValueError:
        return JSONResponse(
            {"ok": False, "error": "This recording is not assigned to a project."},
            status_code=422,
        )
    except QuoteDuplicateError:
        return JSONResponse(
            {"ok": False, "error": "This quote is already saved."},
            status_code=409,
        )
    session.commit()
    return JSONResponse(
        {
            "ok": True,
            "id": str(quote.id),
            "project_id": str(quote.project_id),
        },
        status_code=201,
    )


@router.delete("/quotes/{quote_id}")
def delete_quote_endpoint(
    request: Request,
    session: SessionDep,
    operator: OperatorDep,
    quote_id: uuid.UUID,
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Hard-delete a saved quote."""
    _require_csrf(request, CSRF_QUOTE_MANAGE, csrf_token)
    found = delete_quote(session, quote_id)
    if not found:
        return JSONResponse({"ok": False, "error": "Quote not found."}, status_code=404)
    session.commit()
    return JSONResponse({"ok": True})


@router.patch("/quotes/{quote_id}")
def update_quote_endpoint(
    request: Request,
    session: SessionDep,
    operator: OperatorDep,
    quote_id: uuid.UUID,
    note: Annotated[str | None, Form()] = None,
    csrf_token: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Update the note on a saved quote."""
    _require_csrf(request, CSRF_QUOTE_MANAGE, csrf_token)
    if note and len(note) > MAX_QUOTE_NOTE_CHARS:
        return JSONResponse(
            {"ok": False, "error": f"Note must be {MAX_QUOTE_NOTE_CHARS} characters or fewer."},
            status_code=422,
        )
    quote = update_quote_note(session, quote_id, note)
    if quote is None:
        return JSONResponse({"ok": False, "error": "Quote not found."}, status_code=404)
    session.commit()
    return JSONResponse({"ok": True, "note": quote.note})


@router.get("/projects/{project_id}/quotes/csv", response_model=None)
def export_quotes_csv_endpoint(
    request: Request,
    session: SessionDep,
    operator: OperatorDep,
    project_id: uuid.UUID,
) -> StreamingResponse | JSONResponse:
    """Export all saved quotes for a project as CSV."""
    project = session.get(Project, project_id)
    if project is None:
        return JSONResponse({"ok": False, "error": "Project not found."}, status_code=404)
    content, filename = export_quotes_csv(session, project_id, project.name)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        iter((content,)),
        media_type="text/csv",
        headers=headers,
    )


def quote_to_dict(q: Any) -> dict[str, Any]:
    """Serialize a SavedQuote to a dict for island props."""
    return {
        "id": str(q.id),
        "search_query": q.search_query,
        "left_context": q.left_context,
        "hit": q.hit,
        "right_context": q.right_context,
        "speaker_name": q.speaker_name,
        "media_title": q.media_title,
        "run_id": str(q.run_id),
        "start_seconds": q.start_seconds,
        "note": q.note,
        "created_at": q.created_at.isoformat() if q.created_at else None,
    }
