"""Command palette search routes (issue #162).

GET-only JSON endpoints for the palette island. No CSRF required (read-only).
Auth via OperatorDep (same as all console routes).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from voxint.api.palette_query import search_entities
from voxint.api.routers.deps import OperatorDep, SessionDep

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
