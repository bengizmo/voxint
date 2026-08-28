"""Synthdetect GPU service -- /v1/score + /healthz. See docs/gpu-contracts.md."""

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.errors import (
    file_not_found,
    inference_failed,
    invalid_media,
    model_unavailable,
    path_violation,
    saturated,
)
from app.paths import PathNotFound, PathViolation, resolve_media_path
from app.resource_probe import ResourceSampler, build_admission, build_resources
from app.schemas import (
    CONTRACT_VERSION,
    INFERENCE_SPACE,
    SERVICE_NAME,
    HealthResponse,
    IntervalResult,
    ScoreRequest,
    ScoreResponse,
)
from app.scoring import DecodeError, create_scorer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/media"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

scorer = create_scorer()

_pending = 0
_rejected = 0
_pending_lock = threading.Lock()

sampler = ResourceSampler()


def _admit() -> bool:
    global _pending, _rejected
    with _pending_lock:
        if _pending >= MAX_PENDING_REQUESTS:
            _rejected += 1
            return False
        _pending += 1
        return True


def _release() -> None:
    global _pending
    with _pending_lock:
        _pending -= 1


def _admission_snapshot() -> tuple[int, int]:
    with _pending_lock:
        return _pending, _rejected


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading synthdetect model at startup")
    await run_in_threadpool(scorer.load_model)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()


app = FastAPI(
    title="Voxint synthdetect service",
    description="Synthetic speech detection (w2v2-AASIST, calibrated risk scoring)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse | JSONResponse:
    loaded = scorer.model_loaded
    _pending_now, _rejected_now = _admission_snapshot()
    admission = build_admission(_pending_now, MAX_PENDING_REQUESTS, _rejected_now)
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        contract_version=CONTRACT_VERSION,
        inference_space=INFERENCE_SPACE,
        model=scorer.model_name if loaded else None,
        device=scorer.device_name,
        engine=scorer.engine,
        engine_version=scorer.engine_version,
        runtime=scorer.runtime,
        runtime_version=scorer.runtime_version,
        model_loaded=loaded,
        resources=build_resources(sampler, admission),
    )
    if not loaded:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    if not scorer.model_loaded:
        raise model_unavailable()
    try:
        audio_path = resolve_media_path(MEDIA_ROOT, request.path)
    except PathNotFound as exc:
        raise file_not_found(str(exc)) from exc
    except PathViolation as exc:
        raise path_violation(str(exc)) from exc

    if not _admit():
        raise saturated()
    start = time.time()
    try:
        outcomes = await run_in_threadpool(
            scorer.score_intervals,
            str(audio_path),
            [(iv.start_seconds, iv.end_seconds) for iv in request.intervals],
        )
    except DecodeError as exc:
        raise invalid_media(str(exc)) from exc
    except Exception as exc:
        logger.exception("Synthdetect scoring failed for %s", request.path)
        raise inference_failed(f"Scoring failed: {exc}") from exc
    finally:
        _release()
        try:
            scorer.cleanup_memory()
        except Exception:
            logger.exception("cleanup_memory failed (ignored)")

    scored = sum(1 for o in outcomes if o.raw_score is not None)
    logger.info(
        "Scored %s: %d/%d intervals in %.2fs",
        request.path,
        scored,
        len(outcomes),
        time.time() - start,
    )
    return ScoreResponse(
        inference_space=INFERENCE_SPACE,
        results=[
            IntervalResult(
                raw_score=o.raw_score,
                window_count=o.window_count,
                skip_reason=o.skip_reason,
            )
            for o in outcomes
        ],
    )
