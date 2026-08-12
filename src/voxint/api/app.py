"""FastAPI application: the review console (queue, workbench, media), health.

Adjudication is post-hoc: only COMPLETED runs appear in the queue, and nothing
here touches the pipeline state machine. Every route except ``/healthz`` sits
behind single-operator basic auth; mutations additionally require the live
claim token, and each rendered form carries a fresh server-issued nonce that
becomes the ledger idempotency key — an htmx retry of the same form is a
harmless replay, while a new submission is a new decision (corrections are
appends; the newest ruling per label wins at read time).
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, cast

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from sqlalchemy import exists, select
from sqlalchemy.orm import Session
from starlette.types import ASGIApp, Receive, Scope, Send

from voxint import __version__
from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.adjudication.resolver import (
    LabelState,
    Resolution,
    adjudication_queue,
    label_states,
)
from voxint.adjudication.slots import (
    ClaimMismatchError,
    ClaimUnavailableError,
    claim_run,
    release_run,
    verify_claim,
)
from voxint.adjudication.transcript import (
    TranscriptText,
    attributed_transcript,
    parse_transcript_text,
)
from voxint.api.auth import require_operator
from voxint.api.runs_query import (
    Cursor,
    InvalidCursorError,
    ReviewFilter,
    list_runs,
    parse_review_filter,
    parse_status_filter,
    runs_url,
)
from voxint.config import Settings, get_settings
from voxint.db.models import (
    Decision,
    PipelineRun,
    RunStatus,
    Speaker,
    StageRun,
    TranscriptSegment,
)
from voxint.db.session import build_engine, build_session_factory
from voxint.ingest import (
    UploadConflictError,
    UploadTooLargeError,
    UploadValidationError,
    submit_upload,
)
from voxint.media.serving import (
    MediaGate,
    MediaNotServableError,
    RangeNotSatisfiableError,
    parse_range,
)
from voxint.pipeline.stages.context import StageDataError, normalized_audio_path
from voxint.speakers.matching import gates_from_settings

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_MEDIA_CHUNK_BYTES = 256 * 1024
# Slack over the per-file upload cap for multipart framing (boundaries,
# Content-Disposition headers, the submission_id field) so the coarse
# Content-Length gate never rejects a legitimately max-sized file; the exact
# per-file cap is enforced while streaming in submit_upload.
_UPLOAD_ENVELOPE_ALLOWANCE = 1024 * 1024


class _RequestSizeLimitMiddleware:
    """Reject an over-cap ``Content-Length`` before the request body is read.

    FastAPI parses a multipart body (spooling file parts to a temp) *before* a
    route's dependencies run, so a per-route check cannot gate body reception —
    by the time the handler executes, the whole body is already spooled. This
    ASGI middleware inspects only the ``Content-Length`` header and returns 413
    before Starlette consumes the body, so an *honestly-declared* oversized upload
    is rejected early ("reject oversized Content-Length early" is real). The
    authoritative per-file cap is still enforced while streaming in
    ``submit_upload``.

    Residual (NOT covered here): a chunked request with no ``Content-Length``, or
    a transport that permits an understated one, is still fully multipart-spooled
    by Starlette before the streaming cap runs — so pre-body spooling is bounded
    only for honest declared lengths, not universally. A truly-bounded streaming
    multipart parse (and moving Basic auth ahead of body parsing, which a per-route
    ``OperatorDep`` cannot, given FastAPI's dispatch order) is deferred to the
    security slice. For single-operator home-IP hosting that residual is low-risk.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            for name, value in scope["headers"]:
                if name != b"content-length":
                    continue
                try:
                    length = int(value)
                except ValueError:
                    break  # unparseable → let the streaming cap be authoritative
                if length > self._max_bytes:
                    await self._reject(send)
                    return
                break
        await self._app(scope, receive, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"request entity too large"})


def create_app(
    settings: Settings | None = None,
    session_factory: Any = None,
) -> FastAPI:
    # No docs/OpenAPI surfaces: the UI is server-rendered, and generated docs
    # would be the only unauthenticated routes besides /healthz.
    app = FastAPI(
        title="Voxint",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    resolved = settings or get_settings()
    app.state.settings = resolved
    # Lazy: building the engine at import time would make `/healthz` (and any
    # DB-less test import) depend on a reachable database.
    app.state.session_factory = session_factory
    app.state.media_gate = None
    # Coarse, header-only body-size gate that runs before any route parses the
    # body (see _RequestSizeLimitMiddleware); the streaming per-file cap stays
    # authoritative. Envelope allowance keeps a max-sized file from tripping it.
    app.add_middleware(
        _RequestSizeLimitMiddleware,
        max_bytes=resolved.upload_max_bytes + _UPLOAD_ENVELOPE_ALLOWANCE,
    )
    _register_routes(app)
    return app


def _get_session(request: Request) -> Iterator[Session]:
    factory = request.app.state.session_factory
    if factory is None:
        factory = build_session_factory(build_engine(request.app.state.settings.database_url))
        request.app.state.session_factory = factory
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _publish_run(run_id: uuid.UUID) -> None:
    """Enqueue the pipeline for a freshly-committed run (commit-before-publish).

    The Celery/broker import stays out of the module top level so the read path
    — and any DB-less import — never pulls in the broker. This is the
    straightforward publish; Slice 5 hardens the broker-down case (here a failed
    enqueue surfaces as a 500, leaving the durable QUEUED run behind)."""
    from voxint.worker.tasks import run_pipeline

    run_pipeline.delay(str(run_id))


def _get_media_gate(request: Request) -> MediaGate:
    gate = cast(MediaGate | None, request.app.state.media_gate)
    if gate is None:
        settings: Settings = request.app.state.settings
        gate = MediaGate(
            settings.media_root,
            ffprobe_bin=settings.ffprobe_bin,
            timeout_seconds=settings.media_probe_timeout_seconds,
        )
        request.app.state.media_gate = gate
    return gate


SessionDep = Annotated[Session, Depends(_get_session)]
OperatorDep = Annotated[str, Depends(require_operator)]

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _run_or_404(session: Session, run_id: uuid.UUID) -> PipelineRun:
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


def _label_previews(
    session: Session, run_id: uuid.UUID, states: list[LabelState], limit: int
) -> dict[str, list[TranscriptSegment]]:
    previews: dict[str, list[TranscriptSegment]] = {}
    for state in states:
        previews[state.label] = list(
            session.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.diarization_label == state.label,
                )
                .order_by(TranscriptSegment.segment_index)
                .limit(limit)
            ).scalars()
        )
    return previews


def _workbench_context(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID | None,
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    states = label_states(session, run.id)
    speakers = list(
        session.execute(select(Speaker).order_by(Speaker.display_name)).scalars()
    )
    return {
        "request": request,
        "run": run,
        "states": states,
        "previews": _label_previews(
            session, run.id, states, settings.review_preview_segments
        ),
        "speakers": speakers,
        "token": token,
        "resolution": Resolution,
        "nonce": lambda: uuid.uuid4().hex,
    }


def _labels_response(
    request: Request,
    session: Session,
    run: PipelineRun,
    token: uuid.UUID,
) -> Response:
    """Post-mutation response: htmx gets the refreshed label list, a plain
    form POST gets a redirect back to the workbench."""
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "fragments/labels.html",
            _workbench_context(request, session, run, token),
        )
    return RedirectResponse(f"/review/{run.id}?token={token}", status_code=303)


def _register_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/", include_in_schema=False)
    def index(operator: OperatorDep) -> RedirectResponse:
        return RedirectResponse("/review", status_code=303)

    @app.get("/static/htmx.min.js")
    def htmx_asset(operator: OperatorDep) -> FileResponse:
        # Served as a route, not a StaticFiles mount: mounts bypass the auth
        # dependency, and "everything but /healthz authenticates" is a stated
        # invariant worth keeping absolute.
        return FileResponse(
            Path(__file__).parent / "static" / "htmx.min.js",
            media_type="text/javascript",
        )

    @app.get("/runs")
    def runs(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        status: str | None = None,
        review: str | None = None,
        cursor: str | None = None,
    ) -> Response:
        settings: Settings = request.app.state.settings
        try:
            status_filter = parse_status_filter(status)
            review_filter = parse_review_filter(review)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        parsed_cursor: Cursor | None = None
        # A blank cursor means "start at page 1", mirroring blank status/review
        # meaning "all"; only a non-empty but malformed token is a 400.
        if cursor:
            try:
                parsed_cursor = Cursor.decode(cursor)
            except InvalidCursorError as exc:
                raise HTTPException(status_code=400, detail="invalid cursor") from exc
        page = list_runs(
            session,
            status=status_filter,
            review=review_filter,
            cursor=parsed_cursor,
            page_size=settings.runs_page_size,
        )
        next_url = (
            runs_url(status=status_filter, review=review_filter, cursor=page.next_cursor)
            if page.next_cursor
            else None
        )
        return templates.TemplateResponse(
            request,
            "runs.html",
            {
                "request": request,
                "page": page,
                "status": status_filter,
                "review": review_filter,
                "statuses": list(RunStatus),
                "reviews": list(ReviewFilter),
                "next_url": next_url,
                # Server-issued per-render id: namespaces the upload path and makes
                # a double-submit of this form idempotent (see POST /submit).
                "submission_id": uuid.uuid4().hex,
                "active_nav": "runs",
            },
        )

    @app.post("/submit")
    def submit_media_upload(
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        file: Annotated[UploadFile, File()],
        submission_id: Annotated[str, Form()],
    ) -> RedirectResponse:
        settings: Settings = request.app.state.settings
        # An over-cap Content-Length was already rejected before the body was read
        # (_RequestSizeLimitMiddleware); submit_upload enforces the exact per-file
        # cap authoritatively while streaming (covers a lying/absent length).
        try:
            run = submit_upload(
                session,
                stream=file.file,
                filename=file.filename or "",
                submission_id=submission_id,
                media_root=settings.media_root,
                max_bytes=settings.upload_max_bytes,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UploadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        run_id = run.id
        # Commit-before-publish: the durable QUEUED run must exist before the
        # enqueue, so commit here rather than leaning on the dependency's
        # post-return commit (which would run after publish).
        session.commit()
        _publish_run(run_id)
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}")
    def run_detail(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        run = _run_or_404(session, run_id)
        # The attempt ledger, chronological — matches `voxint status`.
        stage_runs = list(
            session.execute(
                select(StageRun)
                .where(StageRun.pipeline_run_id == run_id)
                .order_by(StageRun.started_at)
            ).scalars()
        )
        settings: Settings = request.app.state.settings
        # Present-only links, decided in Postgres (no filesystem on the read path).
        # Audio reuses the SAME exactly-one-artifact predicate /media serves through
        # (normalized_audio_path resolves iff there is exactly one preprocessed-audio
        # row), so the link never promises a page that would 404; a transcript link
        # needs only that TRANSCRIBE wrote at least one segment.
        try:
            normalized_audio_path(session, run_id, settings.media_root)
            audio_available = True
        except StageDataError:
            audio_available = False
        transcript_available = bool(
            session.scalar(
                select(exists().where(TranscriptSegment.pipeline_run_id == run_id))
            )
        )
        return templates.TemplateResponse(
            request,
            "run_detail.html",
            {
                "request": request,
                "run": run,
                "stage_runs": stage_runs,
                "audio_available": audio_available,
                "transcript_available": transcript_available,
                "active_nav": "runs",
            },
        )

    @app.get("/runs/{run_id}/transcript")
    def run_transcript(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        text: str | None = None,
    ) -> Response:
        run = _run_or_404(session, run_id)
        try:
            variant = parse_transcript_text(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "request": request,
                "run": run,
                "lines": attributed_transcript(session, run_id, text=variant),
                "text": variant,
                "variants": list(TranscriptText),
                "active_nav": "runs",
            },
        )

    @app.get("/review")
    def review_queue(
        request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "queue.html",
            {
                "request": request,
                "entries": adjudication_queue(session),
                "operator": operator,
                "active_nav": "review",
            },
        )

    @app.post("/review/{run_id}/claim")
    def claim(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> RedirectResponse:
        settings: Settings = request.app.state.settings
        _run_or_404(session, run_id)
        try:
            token = claim_run(
                session,
                run_id,
                reviewer=operator,
                ttl_seconds=settings.review_claim_ttl_seconds,
            )
        except ClaimUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse(f"/review/{run_id}?token={token}", status_code=303)

    @app.get("/review/{run_id}")
    def workbench(
        run_id: uuid.UUID,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: uuid.UUID | None = None,
    ) -> Response:
        run = _run_or_404(session, run_id)
        if token is not None:
            try:
                verify_claim(session, run_id, token)
            except ClaimMismatchError:
                token = None  # stale tab: render read-only with a claim button
        return templates.TemplateResponse(
            request, "run.html", _workbench_context(request, session, run, token)
        )

    @app.post("/review/{run_id}/release")
    def release(
        run_id: uuid.UUID,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
    ) -> RedirectResponse:
        try:
            release_run(session, run_id, token)
        except (ClaimMismatchError, ClaimUnavailableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RedirectResponse("/review", status_code=303)

    @app.post("/review/{run_id}/labels/{label}/decision")
    def decide(
        run_id: uuid.UUID,
        label: str,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        action: Annotated[str, Form()],
        speaker_id: Annotated[uuid.UUID | None, Form()] = None,
    ) -> Response:
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            decision = Decision(action)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown action {action!r}") from exc
        if (decision is Decision.ASSIGN) != (speaker_id is not None):
            raise HTTPException(
                status_code=422, detail="assign requires speaker_id; others forbid it"
            )
        if speaker_id is not None and session.get(Speaker, speaker_id) is None:
            raise HTTPException(status_code=422, detail=f"no speaker {speaker_id}")
        if label not in {s.label for s in label_states(session, run_id)}:
            raise HTTPException(status_code=404, detail=f"no label {label!r} in run")
        try:
            record_decision(
                session,
                pipeline_run_id=run_id,
                diarization_label=label,
                decision=decision,
                operator=operator,
                idempotency_key=nonce,
                speaker_id=speaker_id,
            )
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @app.post("/review/{run_id}/labels/{label}/enroll")
    def enroll(
        run_id: uuid.UUID,
        label: str,
        request: Request,
        operator: OperatorDep,
        session: SessionDep,
        token: Annotated[uuid.UUID, Form()],
        nonce: Annotated[str, Form(min_length=8, max_length=64)],
        display_name: Annotated[str, Form()],
    ) -> Response:
        try:
            run = verify_claim(session, run_id, token, for_update=True)
        except ClaimMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        settings: Settings = request.app.state.settings
        try:
            enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label=label,
                display_name=display_name,
                operator=operator,
                idempotency_key=nonce,
                gates=gates_from_settings(settings),
            )
        except EnrollmentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ConflictingReplayError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _labels_response(request, session, run, token)

    @app.get("/review/{run_id}/export.txt")
    def export_transcript(
        run_id: uuid.UUID, operator: OperatorDep, session: SessionDep
    ) -> PlainTextResponse:
        _run_or_404(session, run_id)
        lines = attributed_transcript(session, run_id, text=TranscriptText.ENHANCED)
        body = "\n".join(
            f"[{line.start_seconds:9.2f} {line.end_seconds:9.2f}]"
            f" {line.speaker}: {line.text}"
            for line in lines
        )
        return PlainTextResponse(body + ("\n" if lines else ""))

    @app.get("/media/{run_id}")
    @app.head("/media/{run_id}")
    def media(
        run_id: uuid.UUID, request: Request, operator: OperatorDep, session: SessionDep
    ) -> Response:
        settings: Settings = request.app.state.settings
        _run_or_404(session, run_id)
        try:
            path = normalized_audio_path(session, run_id, settings.media_root)
        except StageDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        gate = _get_media_gate(request)
        try:
            fh, size = gate.open_for_serving(path)
        except MediaNotServableError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            byte_range = parse_range(request.headers.get("range"), size)
        except RangeNotSatisfiableError:
            fh.close()
            return Response(
                status_code=416, headers={"Content-Range": f"bytes */{size}"}
            )
        headers = {"Accept-Ranges": "bytes", "Content-Type": "audio/wav"}
        if byte_range is None:
            status, start, length = 200, 0, size
        else:
            status, start, length = 206, byte_range.start, byte_range.length
            headers["Content-Range"] = (
                f"bytes {byte_range.start}-{byte_range.end}/{size}"
            )
        headers["Content-Length"] = str(length)
        if request.method == "HEAD":
            fh.close()
            return Response(status_code=status, headers=headers)
        return StreamingResponse(
            _stream_file(fh, start, length), status_code=status, headers=headers
        )


def _stream_file(fh: BinaryIO, start: int, length: int) -> Iterator[bytes]:
    """Stream from the gate-validated descriptor — never reopen by path."""
    remaining = length
    with fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_MEDIA_CHUNK_BYTES, remaining))
            if not chunk:
                return  # file shrank mid-stream; truncate rather than hang
            remaining -= len(chunk)
            yield chunk


app = create_app()
