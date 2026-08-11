"""Whisper GPU service — /v1/transcribe + /healthz. See docs/gpu-contracts.md."""

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
from app.schemas import (
    CONTRACT_VERSION,
    SERVICE_NAME,
    HealthResponse,
    Segment,
    TranscribeRequest,
    TranscribeResponse,
    Word,
)
from app.transcription import DecodeError, WhisperTranscriber

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/media"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

transcriber = WhisperTranscriber(
    model_name=os.getenv("WHISPER_MODEL", "large-v2"),
    device=os.getenv("DEVICE", "cuda"),
    compute_type=os.getenv("COMPUTE_TYPE", "int8"),
    batch_size=int(os.getenv("BATCH_SIZE", "16")),
)

# Bounded admission: single-flight inference means requests queue; past this
# depth we refuse with a retryable 503 rather than outliving caller leases.
_pending = 0
_pending_lock = threading.Lock()


def _admit() -> bool:
    global _pending
    with _pending_lock:
        if _pending >= MAX_PENDING_REQUESTS:
            return False
        _pending += 1
        return True


def _release() -> None:
    global _pending
    with _pending_lock:
        _pending -= 1


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading Whisper model at startup")
    await run_in_threadpool(transcriber.load_model)
    yield


app = FastAPI(
    title="Voxint whisper service",
    description="faster-whisper ASR with hallucination soft-tagging",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse | JSONResponse:
    loaded = transcriber.is_initialized
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        contract_version=CONTRACT_VERSION,
        model=transcriber.model_name if loaded else None,
        device=transcriber.device,
        model_loaded=loaded,
    )
    if not loaded:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest) -> TranscribeResponse:
    if not transcriber.is_initialized:
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
            transcriber.transcribe,
            str(audio_path),
            request.language,
            request.initial_prompt,
            request.vad_filter,
        )
    except DecodeError as exc:
        raise invalid_media(str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed for %s", request.path)
        raise inference_failed(f"Transcription failed: {exc}") from exc
    finally:
        _release()
        transcriber.cleanup_memory()

    elapsed = time.time() - start
    rtf = result.duration_seconds / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Transcribed %s: media=%.1fs wall=%.2fs (%.1fx real-time), %d segments (%d suspect)",
        request.path,
        result.duration_seconds,
        elapsed,
        rtf,
        len(result.segments),
        result.suspect_segment_count,
    )

    return TranscribeResponse(
        language=result.language,
        duration_seconds=result.duration_seconds,
        transcript=result.transcript,
        confidence=result.confidence,
        segments=[Segment(**s) for s in result.segments],
        words=[Word(**w) for w in result.words],
        suspect_segment_count=result.suspect_segment_count,
    )
