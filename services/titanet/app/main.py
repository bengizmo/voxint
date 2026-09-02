"""TitaNet GPU service — /v1/embed + /healthz. See docs/gpu-contracts.md."""

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

from app.embedding import DecodeError, create_embedder
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
    EMBEDDING_SPACE,
    SERVICE_NAME,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    WindowResult,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data/media"))
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))

embedder = create_embedder()

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
    logger.info("Loading TitaNet model at startup")
    await run_in_threadpool(embedder.load_model)
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()


app = FastAPI(
    title="Voxint titanet service",
    description="TitaNet-Large speaker embeddings (192-dim, titanet-large-v2)",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse | JSONResponse:
    loaded = embedder.model_loaded
    # Telemetry is most useful when a service is degraded, so the resources
    # block is built on both paths from the cached snapshot (never a live probe).
    _pending_now, _rejected_now = _admission_snapshot()
    admission = build_admission(_pending_now, MAX_PENDING_REQUESTS, _rejected_now)
    body = HealthResponse(
        status="ok" if loaded else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        contract_version=CONTRACT_VERSION,
        model=embedder.model_name if loaded else None,
        device=embedder.device_name,
        engine=embedder.engine,
        engine_version=embedder.engine_version,
        runtime=embedder.runtime,
        runtime_version=embedder.runtime_version,
        embedding_space=EMBEDDING_SPACE,
        window_cap_seconds=embedder.window_cap_seconds,
        model_loaded=loaded,
        resources=build_resources(sampler, admission),
    )
    if not loaded:
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    if not embedder.model_loaded:
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
            embedder.embed_windows,
            str(audio_path),
            [(w.start_seconds, w.end_seconds) for w in request.windows],
        )
    except DecodeError as exc:
        raise invalid_media(str(exc)) from exc
    except Exception as exc:
        logger.exception("Embedding extraction failed for %s", request.path)
        raise inference_failed(f"Embedding extraction failed: {exc}") from exc
    finally:
        _release()
        try:
            embedder.cleanup_memory()
        except Exception:
            # Cleanup is best-effort: a failure here must never replace the
            # response (or the intended structured error) with a 500.
            logger.exception("cleanup_memory failed (ignored)")

    embedded = sum(1 for o in outcomes if o.embedding is not None)
    logger.info(
        "Embedded %s: %d/%d windows in %.2fs",
        request.path,
        embedded,
        len(outcomes),
        time.time() - start,
    )
    return EmbedResponse(
        embedding_space=EMBEDDING_SPACE,
        results=[
            WindowResult(embedding=o.embedding, snr_db=o.snr_db, skip_reason=o.skip_reason)
            for o in outcomes
        ],
    )
