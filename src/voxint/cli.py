"""Voxint CLI: submit media, inspect runs, requeue failures.

Commands talk to the database directly and hand execution to the Celery
worker by enqueueing ``voxint.run_pipeline`` — the CLI never runs a stage
itself. The ``score`` group (:mod:`voxint.harness.score_cli`) is the
exception: file-based offline scoring that never touches settings or the DB.
"""

import argparse
import math
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from voxint import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session, sessionmaker

    from voxint.app_settings import EffectiveWebResearch
    from voxint.config import Settings
    from voxint.db.models import Stage

# Operational imports (settings, SQLAlchemy, DB models) live inside the
# handlers: `voxint score …` is file-only and must not pay for — or be able to
# touch — any of them.


def _engine_or_report(*, connect_timeout: int | None = None) -> "tuple[Engine | None, int]":
    """Build the DB engine, turning a config/URL failure into a sanitized exit 2.

    A malformed ``DATABASE_URL`` (bad driver, unparseable DSN) or an invalid
    ``Settings`` is a CLI/configuration error, not a down dependency — so it maps
    to exit 2, and the message never echoes the connection string (a SQLAlchemy
    URL error can embed the password). ``connect_timeout`` bounds the TCP connect
    (``voxint doctor`` uses it so a firewalled DB fails fast instead of hanging on
    the multi-minute OS default). Returns ``(engine, 0)`` or ``(None, 2)``.
    """
    from voxint.config import SettingsError, get_settings
    from voxint.db.session import build_engine

    try:
        if connect_timeout is None:
            return build_engine(), 0
        from sqlalchemy import create_engine

        return (
            create_engine(
                get_settings().database_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": connect_timeout},
            ),
            0,
        )
    except SettingsError as exc:
        print(f"error: {exc}")  # already credential-sanitized by get_settings
        return None, 2
    except Exception:
        # Never interpolate the exception — a SQLAlchemy URL error can carry the DSN.
        print("error: could not initialize the database engine (check DATABASE_URL)")
        return None, 2


def _effective_web_or_report(
    settings: "Settings",
) -> "tuple[EffectiveWebResearch | None, int]":
    """Resolve the effective (row-over-env) web-research config from the DB (#74).

    Returns ``(effective_web, 0)`` or ``(None, 2)`` on any DB failure — decision 6:
    the CLI participates in effective settings and FAILS HONESTLY on an unavailable
    DB rather than silently falling back to env (which could bypass a UI disable).
    ``_engine_or_report`` covers engine *construction*; the lazy connect first fails
    inside ``session_scope``, so a well-formed-but-unreachable DB is caught here.
    The message is DSN-free (a SQLAlchemy error can embed the connection string).
    """
    from sqlalchemy.exc import SQLAlchemyError

    from voxint.app_settings import get_app_settings, resolve_effective_web_research
    from voxint.db.session import build_session_factory, session_scope

    engine, code = _engine_or_report()
    if engine is None:
        return None, code
    try:
        with session_scope(build_session_factory(engine)) as session:
            return resolve_effective_web_research(get_app_settings(session), settings), 0
    except SQLAlchemyError:
        print("error: database unavailable — cannot resolve effective settings")
        return None, 2
    finally:
        engine.dispose()


def _publish_or_defer(run_id: uuid.UUID, *, stage: "Stage | None" = None) -> bool:
    """Enqueue the run's pipeline task, degrading cleanly on a broker outage.

    Mirrors the HTTP API's contract (``voxint.api.app._publish_or_defer``):
    commit-before-publish means the durable QUEUED run already exists, so a
    broker (Redis) outage is non-fatal — the run stays QUEUED for the beat
    recovery sweep to re-enqueue once the broker returns. Only kombu's
    ``OperationalError`` (its wrapper for every transport/connection failure)
    is swallowed, with a warning to stderr; a genuine bug in the publish path
    still raises. Returns ``True`` if published, ``False`` if deferred.
    """
    # apply_async(ignore_result=True), not .delay(): with a Redis result
    # backend a dead broker makes .delay() raise a vague RuntimeError from the
    # result consumer's reconnect loop, whereas ignoring the (unused) result
    # surfaces the broker connect failure itself as OperationalError — the exact
    # exception this guard catches. See voxint.api.app._publish_run.
    from celery.exceptions import OperationalError

    from voxint.worker.tasks import pipeline_task_for_stage

    try:
        pipeline_task_for_stage(stage).apply_async((str(run_id),), ignore_result=True)
    except OperationalError:
        print(
            f"warning: broker unavailable; run {run_id} stays QUEUED for the "
            "recovery sweep to re-enqueue",
            file=sys.stderr,
        )
        return False
    return True


def _submit(args: argparse.Namespace) -> int:
    from voxint.config import get_settings
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.domain_packs.base import DomainPackError
    from voxint.domain_packs.corrections import operator_correction_message
    from voxint.ingest import submit_media_item

    settings = get_settings()
    media_root = settings.media_root.resolve()
    source = Path(args.path)
    resolved = (media_root / source).resolve()
    try:
        relative = resolved.relative_to(media_root)
    except ValueError:
        print(f"error: {source} is outside media root {media_root}")
        return 2
    if not resolved.is_file():
        print(f"error: {resolved} is not a regular file")
        return 2

    if args.wait:
        code = _validate_poll_args(args)
        if code is not None:
            return code

    engine = build_engine()
    try:
        factory = build_session_factory(engine)
        try:
            with session_scope(factory) as session:
                run_id = submit_media_item(session, str(relative)).id
        except DomainPackError as exc:
            # Freeze-time domain-pack collision (issue #84) / unresolvable pack
            # (issue #11): fail honestly (exit 2) with a plain-language message
            # rather than a raw traceback (matches the URL-submit posture).
            print(
                "error: the domain pack for this media couldn't be applied: "
                f"{operator_correction_message(str(exc))}"
            )
            return 2
        # Print the id BEFORE publishing: the durable QUEUED run already exists,
        # so a broker outage must never cost the operator the run id. The id
        # stays alone on stdout; --wait progress and warnings go to stderr.
        print(run_id)
        # Publish AFTER the durable commit and OUTSIDE its transaction; a broker
        # outage degrades cleanly (run stays QUEUED for the recovery sweep).
        published = _publish_or_defer(run_id)
        if args.wait:
            if not published:
                print(
                    "note: enqueue deferred (broker unavailable); polling will "
                    "wait until the recovery sweep re-enqueues the run",
                    file=sys.stderr,
                )
            try:
                return _poll_until_stop(
                    factory, run_id, interval=args.interval, timeout=args.timeout
                )
            except KeyboardInterrupt:
                print("interrupted", file=sys.stderr)
                return 130
        return 0
    finally:
        engine.dispose()


def _status(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from voxint.db.models import PipelineRun, StageRun
    from voxint.db.session import build_engine, build_session_factory

    factory = build_session_factory(build_engine())
    run_id = uuid.UUID(args.run_id)
    with factory() as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            print(f"error: no run {run_id}")
            return 2
        print(f"status: {run.status}")
        print(f"stage: {run.current_stage or '-'}")
        if run.error:
            print(f"error: {run.error}")
        attempts = (
            session.execute(
                select(StageRun)
                .where(StageRun.pipeline_run_id == run_id)
                .order_by(StageRun.started_at)
            )
            .scalars()
            .all()
        )
        for attempt in attempts:
            line = f"  {attempt.stage} #{attempt.attempt}: {attempt.status}"
            if attempt.error:
                line += f" — {attempt.error}"
            print(line)
    return 0


def _validate_poll_args(args: argparse.Namespace) -> int | None:
    """Range-check ``--interval``/``--timeout`` before any DB work. None = ok.

    ``argparse(type=float)`` accepts ``nan``/``inf``; both would corrupt the loop
    (``--timeout nan`` makes ``remaining <= 0`` always False so the 124 deadline
    never fires; ``--interval nan`` reaches ``time.sleep(nan)`` and raises), reject
    non-finite values here rather than mid-poll.
    """
    if not math.isfinite(args.interval) or args.interval <= 0:
        print("error: --interval must be a finite number greater than 0")
        return 2
    if not math.isfinite(args.timeout) or args.timeout < 0:
        print("error: --timeout must be a finite number 0 or greater")
        return 2
    return None


def _poll_until_stop(
    factory: "sessionmaker[Session]",
    run_id: uuid.UUID,
    *,
    interval: float,
    timeout: float,
    monotonic: "Callable[[], float]" = time.monotonic,
    sleep: "Callable[[float], None]" = time.sleep,
    write: "Callable[[str], object] | None" = None,
    isatty: bool | None = None,
) -> int:
    """Poll a run until it reaches a non-advancing state, or time out.

    Exit contract: 0 completed · 1 failed/cancelled · 2 missing run · 3
    awaiting_adjudication (paused, needs a human ruling — the state machine can
    still resume it, so it is NOT reported as success) · 124 timeout. Progress is
    written to ``write`` (stderr by default) so ``submit --wait`` keeps the run id
    alone on stdout; on a TTY the line is redrawn with ``\\r``, otherwise only
    transitions are printed. Each poll opens a fresh short session: the shared
    factory sets ``expire_on_commit=False``, so a reused session's identity map
    could hand back a stale run without re-querying.
    """
    from voxint.db.models import PipelineRun, RunStatus

    # Resolve stderr at call time, not import time: capsys (and any stderr
    # redirect) swaps sys.stderr per-call, and a default-bound method would keep
    # writing to the original stream.
    if write is None:
        write = sys.stderr.write

    stop_exit = {
        RunStatus.COMPLETED: 0,
        RunStatus.FAILED: 1,
        RunStatus.CANCELLED: 1,
        RunStatus.AWAITING_ADJUDICATION: 3,
    }
    if isatty is None:
        isatty = sys.stderr.isatty()

    deadline = monotonic() + timeout
    last_line: str | None = None
    warned_unknown: set[str] = set()

    def render(text: str, *, final: bool) -> None:
        nonlocal last_line
        if isatty:
            write("\r\033[K" + text)
            if final:
                write("\n")
        elif text != last_line:
            write(text + "\n")
        last_line = text

    while True:
        with factory() as session:
            run = session.get(PipelineRun, run_id)
        if run is None:
            if isatty and last_line is not None:
                write("\n")
            write(f"error: no run {run_id}\n")
            return 2

        status = run.status
        stage = run.current_stage or "-"
        line = f"{status} — stage {stage}"

        try:
            resolved = RunStatus(status)
        except ValueError:
            resolved = None
        if resolved in stop_exit:
            render(line, final=True)
            return stop_exit[resolved]

        # An unknown status is treated as still-advancing (forward-compatible with
        # a newer worker writing a status this CLI predates) rather than a stop —
        # but say so once, so a genuinely corrupt row isn't a silent wait to 124.
        if resolved is None and status not in warned_unknown:
            warned_unknown.add(status)
            if isatty and last_line is not None:
                write("\n")
                last_line = None
            write(f"note: unrecognized run status {status!r}; polling until timeout\n")

        render(line, final=False)
        remaining = deadline - monotonic()
        if remaining <= 0:
            if isatty and last_line is not None:
                write("\n")
            write(f"timeout after {timeout:g}s (last status: {status})\n")
            return 124
        sleep(min(interval, remaining))


def _watch(args: argparse.Namespace) -> int:
    """Follow a run until it stops advancing (see ``_poll_until_stop``)."""
    from voxint.db.session import build_session_factory

    code = _validate_poll_args(args)
    if code is not None:
        return code

    engine, code = _engine_or_report()
    if engine is None:
        return code
    try:
        factory = build_session_factory(engine)
        return _poll_until_stop(factory, args.run_id, interval=args.interval, timeout=args.timeout)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        engine.dispose()


def _requeue(args: argparse.Namespace) -> int:
    from voxint.db.models import PipelineRun, Stage
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.ingest import IngestError, requeue_failed_run
    from voxint.pipeline.transitions import InvalidTransitionError, StaleRevisionError

    factory = build_session_factory(build_engine())
    run_id = uuid.UUID(args.run_id)
    try:
        with session_scope(factory) as session:
            run = session.get(PipelineRun, run_id)
            failed_stage = Stage(run.current_stage) if run and run.current_stage else None
            requeue_failed_run(session, run_id)
    # InvalidTransitionError can't arise on this FAILED->QUEUED-same-stage path,
    # but cas_update_run's contract permits it — caught as defense-in-depth so a
    # future transition-map change surfaces as exit 2, never an uncaught traceback.
    except (IngestError, StaleRevisionError, InvalidTransitionError) as exc:
        print(f"error: {exc}")
        return 2
    # Confirm the durable requeue BEFORE publishing so a broker outage never
    # hides the fact that the run was re-queued; publish degrades cleanly.
    print(f"requeued {run_id}")
    _publish_or_defer(run_id, stage=failed_stage)
    return 0


def _fetch(args: argparse.Namespace) -> int:
    """Register a URL for acquisition and enqueue its run (mirrors ``_submit``).

    Refuses up front when ``ytdlp_enabled`` is off — URL ingestion is an
    authenticated egress capability gated at the submission surface (the worker's
    ACQUIRE stage never consults the flag, so an already-queued URL run still
    completes). A validation/conflict error prints a message and exits 2; the
    error text is URL-free by construction, so a signed query string can't leak
    to the terminal. The download itself happens later in the worker's ACQUIRE
    stage — this only creates the durable QUEUED run.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from voxint.app_settings import get_app_settings, resolve_effective_ytdlp_enabled
    from voxint.config import get_settings
    from voxint.db.session import build_session_factory, session_scope
    from voxint.domain_packs.base import DomainPackError
    from voxint.domain_packs.corrections import operator_correction_message
    from voxint.ingest import (
        UploadConflictError,
        UploadValidationError,
        UrlValidationError,
        submit_url,
    )

    # Resolve the URL from argv or, when the positional is omitted, one line of
    # stdin. A URL passed positionally is exposed in ps / /proc/<pid>/cmdline and
    # persisted in shell history; a signed URL therefore leaks the same way the
    # rest of this arc works to prevent. Piping it (`… | voxint fetch`) keeps it
    # off both. submit_url still validates it either way.
    url = args.url if args.url is not None else sys.stdin.readline().strip()
    if not url:
        print("error: no URL provided (pass it as an argument or on stdin)")
        return 2

    engine, code = _engine_or_report()
    if engine is None:
        return code
    factory = build_session_factory(engine)
    settings = get_settings()
    # No form here, so mint the idempotency id per invocation; it namespaces the
    # pre-assigned source_path the worker's ACQUIRE stage will download into.
    submission_id = uuid.uuid4().hex
    try:
        with session_scope(factory) as session:
            # Effective (row-over-env) gate resolved from the DB so a UI disable
            # governs the CLI too (issue #74) — no silent env fallback. The DB is
            # already required to create the run, so an unavailable DB fails
            # honestly above rather than bypassing a UI-disabled capability.
            if not resolve_effective_ytdlp_enabled(get_app_settings(session), settings):
                print("error: URL ingestion is disabled (ytdlp_enabled is off)")
                return 2
            run_id = submit_url(session, url=url, submission_id=submission_id).id
    except (UrlValidationError, UploadValidationError, UploadConflictError) as exc:
        print(f"error: {exc}")
        return 2
    except DomainPackError as exc:
        # Freeze-time domain-pack collision (issue #84) / unresolvable pack (issue
        # #11): fail honestly (exit 2) with a plain-language message, not a traceback.
        print(
            "error: the domain pack for this media couldn't be applied: "
            f"{operator_correction_message(str(exc))}"
        )
        return 2
    except SQLAlchemyError:
        # A well-formed but unreachable DB fails here (lazy connect), not in
        # _engine_or_report — fail honestly (exit 2) with a DSN-free message
        # rather than a raw traceback (issue #74, decision 6).
        print("error: database unavailable — cannot resolve effective settings")
        return 2
    finally:
        engine.dispose()
    # Print the id BEFORE publishing so a broker outage never costs the operator
    # the run id (the durable QUEUED run already exists); publish degrades cleanly.
    print(run_id)
    _publish_or_defer(run_id)
    return 0


def _research_speaker(args: argparse.Namespace) -> int:
    """Run one web-research job for a speaker inline (issue #40).

    The same code path the worker runs — a durable job row, the budgeted tool
    loop, drafts via the #37 writer — executed synchronously so headless
    operation and integration tests need no broker. Requires all three gates
    (retrieval + LLM + the producer flag)."""
    import uuid as _uuid

    from voxint.config import SettingsError, get_settings
    from voxint.db.models import ResearchJob
    from voxint.db.session import build_engine, build_session_factory
    from voxint.enrichment.research_jobs import (
        ResearchJobError,
        create_job,
        execute_job,
    )

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")
        return 2
    try:
        speaker_id = _uuid.UUID(args.speaker_id)
    except ValueError:
        print(f"error: {args.speaker_id!r} is not a speaker id")
        return 2
    from sqlalchemy.exc import IntegrityError

    factory = build_session_factory(build_engine(settings.database_url))
    with factory() as session:
        try:
            job = create_job(
                session,
                speaker_id=speaker_id,
                settings=settings,
                operator_note=args.note,
            )
            job_id = job.id
            session.commit()
        except ResearchJobError as exc:
            print(f"error: {exc}")
            return 2
        except IntegrityError:
            print("error: a research job is already active for this speaker")
            return 2
    print(f"job {job_id}: researching speaker {speaker_id} ...")
    execute_job(factory, job_id, settings=settings)
    with factory() as session:
        finished = session.get(ResearchJob, job_id)
        assert finished is not None
        print(
            f"job {finished.status}: {finished.searches_used} searches,"
            f" {finished.reads_used} reads, {finished.rounds_used} rounds"
        )
        if finished.error:
            print(f"error: {finished.error}")
            return 1
        if finished.producer_run_id is not None:
            print(
                "drafts recorded — review them on the speaker's card in the"
                " console (accept/reject per field)"
            )
    return 0


def _enrich_assets(args: argparse.Namespace) -> int:
    """Generate run-level assets inline (issue #41).

    The same code path the worker runs — durable job rows, one strict-JSON
    LLM call per kind, immutable asset rows — executed synchronously so
    headless operation and integration tests need no broker. Kinds run
    independently: one failing is reported and the rest still run (exit 1 if
    any failed)."""
    from voxint.config import SettingsError, get_settings
    from voxint.db.models import RunAssetJob, RunAssetKind
    from voxint.db.session import build_engine, build_session_factory
    from voxint.enrichment.asset_jobs import RunAssetJobError, create_jobs, execute_job

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")
        return 2
    kinds = tuple(
        RunAssetKind(value)
        for value in (args.kind or ["summary", "topics", "entity_mentions"])
    )
    factory = build_session_factory(build_engine(settings.database_url))
    with factory() as session:
        try:
            created, already_active = create_jobs(
                session, pipeline_run_id=args.run_id, kinds=kinds, settings=settings
            )
            session.commit()
            job_ids = [(job.id, job.asset_kind) for job in created]
        except RunAssetJobError as exc:
            print(f"error: {exc}")
            return 2
    for kind in already_active:
        print(f"{kind.value}: skipped — a job is already active for this run")
    failures = 0
    for job_id, kind_value in job_ids:
        print(f"job {job_id}: generating {kind_value} ...")
        execute_job(factory, job_id, settings=settings)
        with factory() as session:
            finished = session.get(RunAssetJob, job_id)
            assert finished is not None
            print(f"{kind_value}: {finished.status}")
            if finished.error:
                print(f"error: {finished.error}")
            if finished.status != "succeeded":
                failures += 1
    return 1 if failures else 0


def _research_search(args: argparse.Namespace) -> int:
    """One provider query via the controlled retrieval capability (issue #39).

    Refuses up front when ``voxint_web_research`` is off — before any DNS or
    network I/O, mirroring the ``_fetch``/``ytdlp_enabled`` precedent. Exists
    so an operator can verify the provider config and egress policy by hand
    before anything (issue #40) consumes the tools programmatically.
    """
    from voxint.config import SettingsError, get_settings
    from voxint.research import Attribution, ResearchBudget, web_search

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")
        return 2
    # Resolve the effective (row-over-env) web-research config from the DB before
    # any network I/O (issue #74) — a UI enable/endpoint must govern the CLI too,
    # and a UI disable must not be bypassed. Fails honestly (exit 2) if the DB is
    # unavailable rather than falling back to env.
    effective_web, code = _effective_web_or_report(settings)
    if effective_web is None:
        return code
    if not effective_web.enabled:
        print("error: web research is disabled (voxint_web_research is off)")
        return 2
    outcome = web_search(
        args.query,
        settings=settings,
        effective_web=effective_web,
        budget=ResearchBudget(max_searches=1, max_reads=0),
        attribution=Attribution(feature="cli", reason="operator-search"),
    )
    if not outcome.ok:
        print(f"error: {outcome.error}: {outcome.error_detail}")
        return 2
    if outcome.dropped_results:
        print(f"# {outcome.dropped_results} result(s) dropped by the egress URL policy")
    for result in outcome.results:
        print(result.url)
        if result.title:
            print(f"  {result.title}")
        if result.snippet:
            print(f"  {result.snippet}")
    if not outcome.results:
        print("# no results")
    return 0


def _research_read(args: argparse.Namespace) -> int:
    """Fetch one page as extracted text under the full egress policy (#39).

    The URL can be piped on stdin instead of passed positionally (the same
    rationale as ``voxint fetch``: argv leaks a signed URL to ps and shell
    history). Output shows the final URL WITHOUT its query — a redirect can
    mint a token the operator's terminal/scrollback must not retain.
    """
    from urllib.parse import urlsplit, urlunsplit

    from voxint.config import SettingsError, get_settings
    from voxint.research import Attribution, ResearchBudget, read_url

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")
        return 2
    # Effective (row-over-env) web-research config resolved from the DB before any
    # network I/O (issue #74) — a UI enable/endpoint governs the CLI, a UI disable
    # is honored, and an unavailable DB fails honestly (exit 2) rather than
    # env-fallback.
    effective_web, code = _effective_web_or_report(settings)
    if effective_web is None:
        return code
    if not effective_web.enabled:
        print("error: web research is disabled (voxint_web_research is off)")
        return 2
    url = args.url if args.url is not None else sys.stdin.readline().strip()
    if not url:
        print("error: no URL provided (pass it as an argument or on stdin)")
        return 2
    outcome = read_url(
        url,
        settings=settings,
        effective_web=effective_web,
        budget=ResearchBudget(max_searches=0, max_reads=1),
        attribution=Attribution(feature="cli", reason="operator-read"),
    )
    if not outcome.ok:
        print(f"error: {outcome.error}: {outcome.error_detail}")
        return 2
    if outcome.title:
        print(f"# {outcome.title}")
    parts = urlsplit(outcome.final_url)
    shown = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    suffix = " [query omitted]" if parts.query else ""
    print(f"# {shown}{suffix} ({outcome.bytes_fetched} bytes, {outcome.hops} hop(s))")
    if outcome.truncated:
        print("# [truncated]")
    print(outcome.text)
    return 0


def _tutorial_seed(args: argparse.Namespace) -> int:
    """Idempotently seed the bundled guided-tutorial run and print its id.

    Safe to run repeatedly: an existing tutorial run is returned untouched (its
    WAV repaired if media_root was wiped), a deleted one is rebuilt.
    """
    del args
    from voxint.config import get_settings
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.tutorial.seed import seed_tutorial_run

    settings = get_settings()
    factory = build_session_factory(build_engine())
    with session_scope(factory) as session:
        run_id = seed_tutorial_run(session, media_root=settings.media_root, settings=settings)
    print(run_id)
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Run the review console. The bind host/port come from Settings, so the
    default-credentials-off-loopback refusal inspects the REAL bind address —
    containers set API_HOST=0.0.0.0 and must therefore set a password."""
    del args
    import uvicorn

    from voxint.config import get_settings

    settings = get_settings()  # validators run here, before any socket opens
    uvicorn.run("voxint.api.app:app", host=settings.api_host, port=settings.api_port)
    return 0


def _export(args: argparse.Namespace) -> int:
    """Write a run's transcript in a structured/subtitle format (or RTTM).

    Renders through the same ``voxint.export`` formatters the API export routes
    use, so a piped export and a downloaded file are byte-identical. RTTM reads
    diarization turns (raw labels); every other format reads attributed lines.
    """
    from sqlalchemy import select

    from voxint.adjudication.transcript import attributed_transcript, parse_transcript_text
    from voxint.db.models import DiarizationTurn, PipelineRun
    from voxint.db.session import build_session_factory
    from voxint.export import TranscriptFormat, render_transcript, to_rttm

    try:
        variant = parse_transcript_text(args.text)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    # Refuse to clobber before touching the DB — a surprised operator keeps their
    # file (the write below re-checks atomically to close the check/write race).
    out_path = Path(args.output) if args.output else None
    if out_path is not None and out_path.exists() and not args.force:
        print(f"error: {out_path} exists (use --force to overwrite)")
        return 2

    engine, code = _engine_or_report()
    if engine is None:
        return code
    try:
        factory = build_session_factory(engine)
        with factory() as session:
            if session.get(PipelineRun, args.run_id) is None:
                print(f"error: no run {args.run_id}")
                return 2
            if args.format == "rttm":
                turns = (
                    session.execute(
                        select(DiarizationTurn)
                        .where(DiarizationTurn.pipeline_run_id == args.run_id)
                        .order_by(DiarizationTurn.turn_index)
                    )
                    .scalars()
                    .all()
                )
                output = to_rttm(turns, str(args.run_id))
            else:
                lines = attributed_transcript(session, args.run_id, text=variant)
                output = render_transcript(
                    lines, TranscriptFormat(args.format), timestamps=args.timestamps
                )
    finally:
        engine.dispose()

    # Write the formatter's exact UTF-8 bytes with LF — never a platform text
    # stream, whose newline translation would turn LF into CRLF on Windows and
    # break the byte-identity contract with the HTTP export routes (which always
    # emit LF). Both transports must produce the same file on every OS.
    if out_path is None:
        sys.stdout.buffer.write(output.encode("utf-8"))
        return 0
    # Exclusive create unless --force, so a file that appeared after the pre-DB
    # check is not silently overwritten (nor a symlink followed). newline="" keeps
    # text mode (utf-8 + x/w semantics) while disabling newline translation.
    try:
        with open(
            out_path, "w" if args.force else "x", encoding="utf-8", newline=""
        ) as fh:
            fh.write(output)
    except FileExistsError:
        print(f"error: {out_path} exists (use --force to overwrite)")
        return 2
    print(f"wrote {out_path}")
    return 0


def _list(args: argparse.Namespace) -> int:
    """Browse the newest runs from the CLI (mirrors the /runs page's query)."""
    import json

    from voxint.api.runs_query import list_runs, parse_status_filter
    from voxint.config import SettingsError, get_settings
    from voxint.db.session import build_session_factory

    try:
        status = parse_status_filter(args.status)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")  # credential-sanitized config error → exit 2
        return 2
    limit = args.limit if args.limit is not None else settings.runs_page_size
    if not 1 <= limit <= 500:
        print("error: --limit must be between 1 and 500")
        return 2

    engine, code = _engine_or_report()
    if engine is None:
        return code
    try:
        factory = build_session_factory(engine)
        with factory() as session:
            page = list_runs(session, status=status, review=None, cursor=None, page_size=limit)
    finally:
        engine.dispose()

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "run_id": str(item.run_id),
                        "status": item.status,
                        "source_path": item.source_path,
                        "created_at": item.created_at.isoformat(),
                        "unresolved_count": item.unresolved_count,
                        "label_count": item.label_count,
                        "claim_live": item.claim_live,
                        "claimed_by": item.claimed_by,
                    }
                    for item in page.items
                ],
                indent=2,
                ensure_ascii=False,  # keep Unicode source paths readable (matches to_json)
            )
        )
        return 0

    if not page.items:
        print("(no runs)")
        return 0
    for item in page.items:
        created = item.created_at.strftime("%Y-%m-%d %H:%M")
        print(
            f"{item.run_id}  {item.status:<10}  {created}"
            f"  {item.unresolved_count}/{item.label_count} unresolved  {item.source_path}"
        )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Preflight: check every dependency and report a single pass/fail verdict."""
    del args
    import httpx

    from voxint import diagnostics
    from voxint.config import SettingsError, get_settings

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")  # already sanitized of credentials by get_settings
        return 2

    # A bounded connect timeout so a firewalled DB fails fast (the command promises
    # a quick verdict); construction errors map to a sanitized exit 2.
    engine, code = _engine_or_report(connect_timeout=5)
    if engine is None:
        return code
    try:
        with httpx.Client(timeout=httpx.Timeout(settings.health_probe_timeout_seconds)) as client:
            results = diagnostics.run_diagnostics(settings, engine, http_client=client)
    finally:
        engine.dispose()

    for result in results:
        tag = "ok  " if result.ok else ("FAIL" if result.hard else "warn")
        print(f"[{tag}] {result.name}: {result.detail}")
    code = diagnostics.exit_code(results)
    verdict = "all hard dependencies OK" if code == 0 else "a hard dependency is down"
    print(f"\ndoctor: {verdict}")
    return code


def _stats(args: argparse.Namespace) -> int:
    """Aggregate, read-only system summary (run/stage health + throughput)."""
    import json
    from datetime import UTC, datetime

    from voxint.api.stats_query import (
        collect_stats,
        format_stats_text,
        parse_since,
        stats_to_json,
    )
    from voxint.db.session import build_session_factory

    now = datetime.now(UTC)
    try:
        since = parse_since(args.since, now=now)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    engine, code = _engine_or_report()
    if engine is None:
        return code
    try:
        factory = build_session_factory(engine)
        with factory() as session:
            stats = collect_stats(session, since=since, now=now)
    finally:
        engine.dispose()

    if args.json:
        print(json.dumps(stats_to_json(stats), indent=2, ensure_ascii=False))
    else:
        sys.stdout.write(format_stats_text(stats))
    return 0


def _enrich_names(args: argparse.Namespace) -> int:
    """Operator-triggered offline name-suggestion sweep (issue #38).

    Runs the ``names.offline`` producer for one run (or every completed run)
    and reports per-run outcomes. Batch mode isolates failures per run and
    exits 1 if any run failed; single-run mode maps a producer error to 2.
    """
    if (args.run_id is not None) == args.all_completed:
        print("error: provide exactly one of RUN_ID or --all-completed", file=sys.stderr)
        return 2

    engine, code = _engine_or_report()
    if engine is None:
        return code

    from sqlalchemy import func, select

    from voxint.app_settings import (
        get_app_settings,
        resolve_effective_enrichment_names_enabled,
        resolve_effective_enrichment_names_llm_enabled,
        resolve_effective_llm_enabled,
    )
    from voxint.config import get_settings
    from voxint.db.models import EnrichmentCandidate, PipelineRun, RunStatus
    from voxint.db.session import build_session_factory, session_scope
    from voxint.enrichment.producers.names import (
        NameProducerError,
        run_offline_name_producer,
    )

    settings = get_settings()
    factory = build_session_factory(engine)
    # Effective (row-over-env) gates resolved from the DB so a UI toggle governs
    # the CLI too (issue #74) — no silent env fallback (the DB is already required
    # for the sweep, so an unavailable DB fails honestly rather than bypassing a
    # UI-disabled capability).
    with session_scope(factory) as gate_session:
        gate_row = get_app_settings(gate_session)
        names_enabled = resolve_effective_enrichment_names_enabled(gate_row, settings)
        names_llm_enabled = resolve_effective_enrichment_names_llm_enabled(gate_row, settings)
        llm_enabled = resolve_effective_llm_enabled(gate_row, settings)
    if not names_enabled:
        # Field-style copy (not an env-var name): the gate is now the effective
        # row-over-env value, so a console disable can be the cause too (#74).
        print(
            "error: name enrichment is disabled"
            " (enrichment_names_enabled is off — env or console override)",
            file=sys.stderr,
        )
        return 2
    if args.llm and not (names_llm_enabled and llm_enabled):
        print(
            "error: the LLM name pass requires enrichment_names_llm_enabled and"
            " llm_enabled (env or console override)",
            file=sys.stderr,
        )
        return 2

    try:
        if args.run_id is not None:
            run_ids = [args.run_id]
        else:
            with session_scope(factory) as session:
                run_ids = list(
                    session.execute(
                        select(PipelineRun.id)
                        .where(PipelineRun.status == RunStatus.COMPLETED.value)
                        .order_by(PipelineRun.created_at)
                    ).scalars()
                )
            if not run_ids:
                print("no completed runs")
                return 0

        from sqlalchemy.exc import SQLAlchemyError

        from voxint.enrichment.drafts import EnrichmentDraftError

        # Per-run failure isolation covers the families a single bad run can
        # raise (producer refusals, draft validation, per-run SQL errors) —
        # a programming error still propagates rather than being masked.
        per_run_errors = (NameProducerError, EnrichmentDraftError, SQLAlchemyError)

        failures = 0
        for run_id in run_ids:
            try:
                with session_scope(factory) as session:
                    producer_run = run_offline_name_producer(
                        session, run_id=run_id, settings=settings
                    )
                    count = session.execute(
                        select(func.count())
                        .select_from(EnrichmentCandidate)
                        .where(EnrichmentCandidate.producer_run_id == producer_run.id)
                    ).scalar_one()
                # Printed only after session_scope committed — a commit
                # failure must not leave a success line behind.
                print(
                    f"{run_id}: outcome={producer_run.outcome}"
                    f" generation={producer_run.generation} candidates={count}"
                )
                if args.llm:
                    # Separate session scope: an LLM failure must not roll
                    # back the already-successful offline sweep.
                    from voxint.enrichment.producers.names_llm import (
                        run_llm_name_producer,
                    )

                    with session_scope(factory) as session:
                        llm_run = run_llm_name_producer(session, run_id=run_id, settings=settings)
                        llm_count = session.execute(
                            select(func.count())
                            .select_from(EnrichmentCandidate)
                            .where(EnrichmentCandidate.producer_run_id == llm_run.id)
                        ).scalar_one()
                    print(
                        f"{run_id}: llm outcome={llm_run.outcome}"
                        f" generation={llm_run.generation} candidates={llm_count}"
                    )
            except per_run_errors as exc:
                failures += 1
                print(f"{run_id}: error: {exc}", file=sys.stderr)
        if failures:
            return 1 if args.all_completed else 2
        return 0
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxint", description="Voxint audio pipeline")
    parser.add_argument("--version", action="version", version=f"voxint {__version__}")
    sub = parser.add_subparsers(dest="command")

    submit_p = sub.add_parser("submit", help="submit a media file (path relative to MEDIA_ROOT)")
    submit_p.add_argument("path")
    submit_p.add_argument(
        "--wait", action="store_true", help="after enqueue, follow the run until it stops"
    )
    submit_p.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="--wait: give up after N seconds (default 3600)",
    )
    submit_p.add_argument(
        "--interval", type=float, default=2.0, help="--wait: poll every N seconds (default 2)"
    )
    submit_p.set_defaults(fn=_submit)

    status_p = sub.add_parser("status", help="show a run's state and stage ledger")
    status_p.add_argument("run_id")
    status_p.set_defaults(fn=_status)

    watch_p = sub.add_parser("watch", help="follow a run until it stops advancing")
    watch_p.add_argument("run_id", type=uuid.UUID)
    watch_p.add_argument(
        "--timeout", type=float, default=3600.0, help="give up after N seconds (default 3600)"
    )
    watch_p.add_argument(
        "--interval", type=float, default=2.0, help="poll every N seconds (default 2)"
    )
    watch_p.set_defaults(fn=_watch)

    requeue_p = sub.add_parser("requeue", help="requeue a failed run at its failed stage")
    requeue_p.add_argument("run_id")
    requeue_p.set_defaults(fn=_requeue)

    fetch_p = sub.add_parser("fetch", help="submit a URL for yt-dlp acquisition + transcription")
    fetch_p.add_argument(
        "url",
        nargs="?",
        help="URL to fetch; omit to read one line from stdin (keeps a signed "
        "URL out of argv and shell history)",
    )
    fetch_p.set_defaults(fn=_fetch)

    export_p = sub.add_parser(
        "export", help="export a run's transcript (txt/md/srt/vtt/json/rttm)"
    )
    export_p.add_argument("run_id", type=uuid.UUID)
    export_p.add_argument(
        "--format",
        choices=["txt", "md", "srt", "vtt", "json", "rttm"],
        default="txt",
        help="output format (default: txt)",
    )
    export_p.add_argument(
        "--text",
        choices=["corrected", "enhanced", "raw"],
        help=(
            "which transcript text to render (default: corrected — operator"
            " corrections applied over enhanced/raw; 'enhanced' is the pipeline"
            " text without corrections; 'raw' is the immutable ASR output;"
            " ignored for rttm)"
        ),
    )
    export_p.add_argument(
        "--no-timestamps",
        dest="timestamps",
        action="store_false",
        default=True,
        help="omit timestamps (txt bracket column / md time range; ignored for other formats)",
    )
    export_p.add_argument("-o", "--output", help="write to PATH instead of stdout")
    export_p.add_argument(
        "--force", action="store_true", help="overwrite an existing --output file"
    )
    export_p.set_defaults(fn=_export)

    list_p = sub.add_parser("list", help="list the most recent runs, newest first")
    list_p.add_argument("--status", help="filter by run status (e.g. completed, failed)")
    list_p.add_argument("--limit", type=int, help="max rows (1..500, default runs_page_size)")
    list_p.add_argument("--json", action="store_true", help="emit a JSON array instead of a table")
    list_p.set_defaults(fn=_list)

    doctor_p = sub.add_parser("doctor", help="preflight diagnostics for every dependency")
    doctor_p.set_defaults(fn=_doctor)

    stats_p = sub.add_parser("stats", help="aggregate system health, throughput, and durations")
    stats_p.add_argument(
        "--since",
        default="24h",
        help="throughput window: '<n>h', '<n>d', or an ISO-8601 datetime (default: 24h)",
    )
    stats_p.add_argument("--json", action="store_true", help="emit a JSON object instead of text")
    stats_p.set_defaults(fn=_stats)

    serve_p = sub.add_parser("serve", help="run the API + review console (binds from settings)")
    serve_p.set_defaults(fn=_serve)

    enrich_p = sub.add_parser("enrich", help="offline enrichment producers (draft suggestions)")
    enrich_sub = enrich_p.add_subparsers(dest="enrich_command", required=True)
    names_p = enrich_sub.add_parser(
        "names",
        help="mine speaker-name suggestions from stored metadata + transcript (issue #38)",
    )
    names_p.add_argument(
        "run_id", nargs="?", type=uuid.UUID, help="run to sweep (omit with --all-completed)"
    )
    names_p.add_argument(
        "--all-completed",
        action="store_true",
        help="sweep every completed run (per-run failures reported, exit 1 if any)",
    )
    names_p.add_argument(
        "--llm",
        action="store_true",
        help="also run the additive LLM name pass (requires"
        " ENRICHMENT_NAMES_LLM_ENABLED and LLM_ENABLED; uses the env LLM config)",
    )
    names_p.set_defaults(fn=_enrich_names)
    assets_p = enrich_sub.add_parser(
        "assets",
        help="generate run-level assets — summary, topics, entity mentions"
        " (issue #41; requires ENRICHMENT_RUN_ASSETS_ENABLED + LLM_ENABLED)",
    )
    assets_p.add_argument("run_id", type=uuid.UUID, help="pipeline run UUID")
    assets_p.add_argument(
        "--kind",
        action="append",
        choices=["summary", "topics", "entity_mentions"],
        help="asset kind to generate (repeatable; default: all three)",
    )
    assets_p.set_defaults(fn=_enrich_assets)

    research_p = sub.add_parser(
        "research",
        help="controlled web retrieval (issue #39; requires VOXINT_WEB_RESEARCH=true)",
    )
    research_sub = research_p.add_subparsers(dest="research_command", required=True)
    rsearch_p = research_sub.add_parser(
        "search", help="query the configured search provider (normalized results)"
    )
    rsearch_p.add_argument("query", help="search query (bounded; never logged)")
    rsearch_p.set_defaults(fn=_research_search)
    rread_p = research_sub.add_parser(
        "read", help="fetch one page as extracted text under the egress policy"
    )
    rread_p.add_argument(
        "url",
        nargs="?",
        default=None,
        help="absolute http(s) URL (public hosts only); omit to read one"
        " line from stdin (keeps a signed URL out of ps/shell history)",
    )
    rread_p.set_defaults(fn=_research_read)
    rspeaker_p = research_sub.add_parser(
        "speaker",
        help="run one budgeted research job for a speaker and record drafts"
        " (issue #40; requires ENRICHMENT_WEB_RESEARCH_ENABLED + LLM_ENABLED)",
    )
    rspeaker_p.add_argument("speaker_id", help="roster speaker UUID")
    rspeaker_p.add_argument(
        "--note",
        default=None,
        help="optional bounded note handed to the researcher as seed context",
    )
    rspeaker_p.set_defaults(fn=_research_speaker)

    tutorial_p = sub.add_parser("tutorial", help="bundled guided-tutorial fixtures")
    tutorial_sub = tutorial_p.add_subparsers(dest="tutorial_command", required=True)
    seed_p = tutorial_sub.add_parser(
        "seed", help="idempotently seed the bundled 3-speaker tutorial run"
    )
    seed_p.set_defaults(fn=_tutorial_seed)

    # File-based scoring harness: no settings, no DB, no worker (docs/harness.md).
    from voxint.harness import score_cli

    score_cli.register(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "fn"):
        parser.print_help()
        return 0
    result: int = args.fn(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
