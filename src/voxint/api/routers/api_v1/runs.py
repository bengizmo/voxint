"""Public API: pipeline run listing, detail, and lifecycle actions."""

import base64
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from voxint.api.api_app import ApiKeyDep, ApiSessionDep
from voxint.db.models import PipelineRun
from voxint.ingest.service import (
    IngestError,
    RunNotFoundError,
    cancel_run,
    pause_run,
    resume_run,
)
from voxint.pipeline.transitions import StaleRevisionError

router = APIRouter(prefix="/runs", tags=["runs"])

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200


# ---- Response schemas ----

class RunResponse(BaseModel):
    id: str
    media_item_id: str
    status: str
    current_stage: str | None
    revision: int
    error: str | None
    created_at: str
    updated_at: str
    detected_language: str | None


class RunsListResponse(BaseModel):
    items: list[RunResponse]
    next_cursor: str | None
    has_more: bool


class LifecycleRequest(BaseModel):
    expected_revision: int | None = None


def _run_to_response(run: PipelineRun) -> RunResponse:
    return RunResponse(
        id=str(run.id),
        media_item_id=str(run.media_item_id),
        status=run.status,
        current_stage=run.current_stage,
        revision=run.revision,
        error=run.error,
        created_at=run.created_at.isoformat() if run.created_at else "",
        updated_at=run.updated_at.isoformat() if run.updated_at else "",
        detected_language=run.detected_language,
    )


# ---- Cursor helpers ----

def _encode_cursor(created_at: datetime, run_id: uuid.UUID) -> str:
    payload = json.dumps(
        {"t": created_at.isoformat(), "id": str(run_id)}, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(raw: str) -> tuple[datetime, uuid.UUID]:
    padded = raw + "=" * (-len(raw) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded))
        return datetime.fromisoformat(data["t"]), uuid.UUID(data["id"])
    except Exception:
        raise HTTPException(status_code=422, detail="invalid cursor") from None


# ---- Routes ----

@router.get("")
def list_runs(
    identity: ApiKeyDep,
    session: ApiSessionDep,
    status: str | None = None,
    media_id: uuid.UUID | None = None,
    cursor: str | None = None,
    limit: int = _DEFAULT_PAGE_SIZE,
) -> RunsListResponse:
    limit = max(1, min(limit, _MAX_PAGE_SIZE))
    stmt = (
        select(PipelineRun)
        .where(PipelineRun.archived_at.is_(None))
        .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
    )
    if status:
        stmt = stmt.where(PipelineRun.status == status)
    if media_id:
        stmt = stmt.where(PipelineRun.media_item_id == media_id)
    if cursor:
        ts, cid = _decode_cursor(cursor)
        stmt = stmt.where(
            (PipelineRun.created_at < ts)
            | ((PipelineRun.created_at == ts) & (PipelineRun.id < cid))
        )

    rows = list(session.scalars(stmt.limit(limit + 1)))
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = (
        _encode_cursor(items[-1].created_at, items[-1].id) if has_more else None
    )
    return RunsListResponse(
        items=[_run_to_response(r) for r in items],
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get("/{run_id}")
def get_run(
    run_id: uuid.UUID,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> RunResponse:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_to_response(run)


def _lifecycle_action(
    session: ApiSessionDep,
    run_id: uuid.UUID,
    body: LifecycleRequest,
    action_fn: Callable[..., Any],
) -> RunResponse:
    try:
        action_fn(session, run_id, expected_revision=body.expected_revision)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except StaleRevisionError as exc:
        raise HTTPException(status_code=409, detail="stale revision") from exc
    except IngestError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_to_response(run)


@router.post("/{run_id}/cancel")
def cancel(
    run_id: uuid.UUID,
    body: LifecycleRequest,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> RunResponse:
    return _lifecycle_action(session, run_id, body, cancel_run)


@router.post("/{run_id}/pause")
def pause(
    run_id: uuid.UUID,
    body: LifecycleRequest,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> RunResponse:
    return _lifecycle_action(session, run_id, body, pause_run)


@router.post("/{run_id}/resume")
def do_resume(
    run_id: uuid.UUID,
    body: LifecycleRequest,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> RunResponse:
    return _lifecycle_action(session, run_id, body, resume_run)
