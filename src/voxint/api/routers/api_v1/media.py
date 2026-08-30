"""Public API: media ingest (upload + URL fetch)."""

import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from voxint.api.api_app import ApiKeyDep, ApiSessionDep
from voxint.config import Settings
from voxint.db.models import PipelineRun
from voxint.ingest.service import (
    SubmissionResult,
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    UrlValidationError,
    submit_upload,
    submit_url,
)

router = APIRouter(prefix="/media", tags=["media"])


class FetchRequest(BaseModel):
    url: str = Field(..., min_length=1)


class SubmissionResponse(BaseModel):
    media_id: str
    run_id: str
    status: str


def _publish_result(
    result: SubmissionResult, db: Session
) -> SubmissionResponse:
    run = db.get(PipelineRun, result.run_id)
    media_id = str(run.media_item_id) if run else ""
    result.publish()
    return SubmissionResponse(
        media_id=media_id,
        run_id=str(result.run_id),
        status="queued",
    )


@router.post("/upload", status_code=201)
def upload_media(
    request: Request,
    file: UploadFile,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> SubmissionResponse:
    idempotency_key = request.headers.get("idempotency-key", "")
    if not idempotency_key:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key header is required",
        )
    try:
        uuid.UUID(idempotency_key)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key must be a valid UUID",
        ) from None

    settings: Settings = request.app.state.settings
    try:
        result = submit_upload(
            session,
            stream=file.file,
            filename=file.filename or "upload",
            submission_id=idempotency_key,
            media_root=settings.media_root,
            max_bytes=settings.upload_max_bytes,
            settings=settings,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session.commit()
    return _publish_result(result, session)


@router.post("/fetch", status_code=201)
def fetch_media(
    request: Request,
    body: FetchRequest,
    identity: ApiKeyDep,
    session: ApiSessionDep,
) -> SubmissionResponse:
    idempotency_key = request.headers.get("idempotency-key", "")
    if not idempotency_key:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key header is required",
        )
    try:
        uuid.UUID(idempotency_key)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key must be a valid UUID",
        ) from None

    settings: Settings = request.app.state.settings
    try:
        result = submit_url(
            session,
            url=body.url,
            submission_id=idempotency_key,
            settings=settings,
        )
    except UploadConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (UrlValidationError, UploadValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session.commit()
    return _publish_result(result, session)
