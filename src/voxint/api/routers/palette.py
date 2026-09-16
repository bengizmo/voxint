"""Command palette search routes (issue #162).

GET-only JSON endpoints for the palette island. No CSRF required (read-only).
Auth via OperatorDep (same as all console routes).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from markupsafe import Markup

from voxint.api.meaning_query import search_passages
from voxint.api.palette_query import search_entities
from voxint.api.routers.deps import OperatorDep, SessionDep, get_session_factory

router = APIRouter(prefix="/palette", tags=["palette"])


@router.get("/entities")
def palette_entities(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    q: str = "",
) -> JSONResponse:
    """Search media, speakers, and projects by substring match."""
    result = search_entities(
        session,
        query=q,
        settings=request.app.state.settings,
        app_state=request.app.state,
    )
    return JSONResponse(result.as_dict())


@router.get("/passages")
def palette_passages(
    request: Request,
    operator: OperatorDep,
    q: str = "",
) -> JSONResponse:
    """Semantic + lexical transcript passage search.

    Uses the same ``search_passages`` as the Explore page, with a smaller
    result cap suited to the palette's jump-list UX. The semantic index's
    honest states (off/unavailable/indexing) pass through to the island
    unchanged.
    """
    page = search_passages(
        get_session_factory(request),
        settings=request.app.state.settings,
        query=q,
        top_k=8,
        per_run_cap=2,
    )
    return JSONResponse(
        {
            "state": page.state.value,
            "query": page.query,
            "items": [
                {
                    "run_id": str(item.run_id),
                    "title": item.title,
                    "speaker_label": item.speaker_label,
                    "start_seconds": item.start_seconds,
                    "snippet": (
                        Markup(item.snippet).striptags()
                        if isinstance(item.snippet, (str, Markup))
                        else str(item.snippet)
                    ),
                    "href": item.jump_url,
                }
                for item in page.items
            ],
        }
    )
