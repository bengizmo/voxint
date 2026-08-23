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

from app.backends import create_transcriber
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
    HealthResponse,
    Segment,
    TranscribeRequest,
    TranscribeResponse,
    Word,
)
from app.transcription import DecodeError
from app.whisper_startup import apply_whisper_startup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/media"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

# Fail-closed model selection: the default (large-v2) keeps the baked, offline
# path untouched; an alternate model must be explicitly gated
# (WHISPER_ALLOW_DOWNLOAD=1 + a full-SHA WHISPER_REVISION) or the service refuses
# to start. Applied before create_transcriber so the download-root and offline
# overrides are in os.environ before the model libraries load.
_startup = apply_whisper_startup()

# Fail-closed engine selection via WHISPER_ENGINE (default ct2-legacy); an
# unknown engine raises here at import rather than degrading silently.
transcriber = create_transcriber(
    model_name=_startup.model_name,
    device=os.getenv("DEVICE", "cuda"),
    compute_type=os.getenv("COMPUTE_TYPE", "int8"),
    batch_size=int(os.getenv("BATCH_SIZE", "16")),
)

# Bounded admission: single-flight inference means requests queue; past this
# depth we refuse with a retryable 503 rather than outliving caller leases.
# ``_rejected`` counts those refusals so /healthz can source contention
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
    logger.info("Loading Whisper model at startup")
    await run_in_threadpool(transcriber.load_model)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()


app = FastAPI(
    title="Voxint whisper service",
    description="faster-whisper ASR with hallucination soft-tagging",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse | JSONResponse:
    loaded = transcriber.is_initialized
    # Decode identity is null until load (versions unresolved); computed once
    # and cached by the facade, so healthz never recomputes it per request.
    identity = transcriber.decode_identity() if loaded else {}
    # Telemetry is most useful when a service is degraded, so the resources
    # block is built on both paths from the cached snapshot (never a live probe).
    _pending_now, _rejected_now = _admission_snapshot()
    admission = build_admission(_pending_now, MAX_PENDING_REQUESTS, _rejected_now)
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        contract_version=CONTRACT_VERSION,
        model=transcriber.model_name if loaded else None,
        device=transcriber.device,
        engine=transcriber.engine,
        engine_version=transcriber.engine_version,
        runtime=transcriber.runtime,
        runtime_version=transcriber.runtime_version,
        model_loaded=loaded,
        vad_plan_version=identity.get("vad_plan_version"),
        vad_params=identity.get("vad_params"),
        decode_config_hash=identity.get("decode_config_hash"),
        model_revision=identity.get("model_revision"),
        resources=build_resources(sampler, admission),
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
        language_probability=result.language_probability,
        duration_seconds=result.duration_seconds,
        transcript=result.transcript,
        confidence=result.confidence,
        segments=[Segment(**s) for s in result.segments],
        words=[Word(**w) for w in result.words],
        suspect_segment_count=result.suspect_segment_count,
    )
