"""Pyannote GPU service — /v1/diarize + /healthz. See docs/gpu-contracts.md."""

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

from app.diarizer import DecodeError, Diarizer
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
    SERVICE_NAME,
    DiarizeRequest,
    DiarizeResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/media"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

diarizer = Diarizer()

# ``_rejected`` counts admission refusals so /healthz can source contention
# honestly instead of the app inferring it from opaque 503s.
_pending = 0
_rejected = 0
_pending_lock = threading.Lock()

# Background hardware sampler; /healthz serves its cache (never probes NVML per
# request). Constructed at import (import-time torch/NVML-free), started/stopped
# in the lifespan.
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
    logger.info("Loading pyannote pipeline at startup")
    await run_in_threadpool(diarizer.load_model)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()


app = FastAPI(
    title="Voxint pyannote service",
    description="Speaker diarization (pyannote/speaker-diarization-3.1)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse | JSONResponse:
    loaded = diarizer.model_loaded
    # Telemetry is most useful when a service is degraded, so the resources
    # block is built on both paths from the cached snapshot (never a live probe).
    _pending_now, _rejected_now = _admission_snapshot()
    admission = build_admission(_pending_now, MAX_PENDING_REQUESTS, _rejected_now)
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        contract_version=CONTRACT_VERSION,
        model=diarizer.model_name if loaded else None,
        device=diarizer.device_name,
        engine=diarizer.engine,
        engine_version=diarizer.engine_version,
        runtime=diarizer.runtime,
        runtime_version=diarizer.runtime_version,
        model_revision=diarizer.model_revision,
        checkpoint_fingerprint=diarizer.checkpoint_fingerprint if loaded else None,
        model_loaded=loaded,
        resources=build_resources(sampler, admission),
    )
    if not loaded:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.post("/v1/diarize", response_model=DiarizeResponse)
async def diarize(request: DiarizeRequest) -> DiarizeResponse:
    if not diarizer.model_loaded:
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
        result = await run_in_threadpool(
            diarizer.diarize,
            str(audio_path),
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            min_turn_seconds=request.min_turn_seconds,
        )
    except DecodeError as exc:
        raise invalid_media(str(exc)) from exc
    except Exception as exc:
        logger.exception("Diarization failed for %s", request.path)
        raise inference_failed(f"Diarization failed: {exc}") from exc
    finally:
        _release()

    elapsed = time.time() - start
    rtf = result["duration_seconds"] / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Diarized %s: media=%.1fs wall=%.2fs (%.1fx real-time), %d speakers, %d turns",
        request.path,
        result["duration_seconds"],
        elapsed,
        rtf,
        result["num_speakers"],
        len(result["turns"]),
    )
    return DiarizeResponse(**result)
