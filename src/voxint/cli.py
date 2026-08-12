"""Voxint CLI: submit media, inspect runs, requeue failures.

Commands talk to the database directly and hand execution to the Celery
worker by enqueueing ``voxint.run_pipeline`` — the CLI never runs a stage
itself. The ``score`` group (:mod:`voxint.harness.score_cli`) is the
exception: file-based offline scoring that never touches settings or the DB.
"""

import argparse
import uuid
from pathlib import Path

from voxint import __version__

# Operational imports (settings, SQLAlchemy, DB models) live inside the
# handlers: `voxint score …` is file-only and must not pay for — or be able to
# touch — any of them.


def _submit(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from voxint.config import get_settings
    from voxint.db.models import MediaItem
    from voxint.db.session import build_engine, build_session_factory, session_scope

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

    from voxint.pipeline.engine import submit
    from voxint.worker.tasks import run_pipeline

    factory = build_session_factory(build_engine())
    with session_scope(factory) as session:
        existing = session.execute(
            select(MediaItem).where(MediaItem.source_path == str(relative))
        ).scalar_one_or_none()
        media = existing or MediaItem(source_path=str(relative))
        if existing is None:
            session.add(media)
            session.flush()
        run_id = submit(session, media.id).id
    run_pipeline.delay(str(run_id))
    print(run_id)
    return 0


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


def _requeue(args: argparse.Namespace) -> int:
    from voxint.db.models import PipelineRun, RunStatus
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.pipeline.transitions import (
        InvalidTransitionError,
        StaleRevisionError,
        cas_update_run,
        snapshot,
    )
    from voxint.worker.tasks import run_pipeline

    factory = build_session_factory(build_engine())
    run_id = uuid.UUID(args.run_id)
    with session_scope(factory) as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            print(f"error: no run {run_id}")
            return 2
        held = snapshot(run)
        if held.status is not RunStatus.FAILED:
            print(f"error: run is {held.status.value}, only failed runs can be requeued")
            return 2
        if held.current_stage is None:
            # A FAILED run always carries its failed stage; None means the
            # state machine was violated — surface it, never guess a stage.
            print(f"error: run {run_id} is FAILED with no current_stage; refusing to guess")
            return 2
        try:
            cas_update_run(
                session,
                held,
                status=RunStatus.QUEUED,
                current_stage=held.current_stage,
            )
        except (StaleRevisionError, InvalidTransitionError) as exc:
            print(f"error: {exc}")
            return 2
    run_pipeline.delay(str(run_id))
    print(f"requeued {run_id}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxint", description="Voxint audio pipeline")
    parser.add_argument("--version", action="version", version=f"voxint {__version__}")
    sub = parser.add_subparsers(dest="command")

    submit_p = sub.add_parser("submit", help="submit a media file (path relative to MEDIA_ROOT)")
    submit_p.add_argument("path")
    submit_p.set_defaults(fn=_submit)

    status_p = sub.add_parser("status", help="show a run's state and stage ledger")
    status_p.add_argument("run_id")
    status_p.set_defaults(fn=_status)

    requeue_p = sub.add_parser("requeue", help="requeue a failed run at its failed stage")
    requeue_p.add_argument("run_id")
    requeue_p.set_defaults(fn=_requeue)

    serve_p = sub.add_parser("serve", help="run the API + review console (binds from settings)")
    serve_p.set_defaults(fn=_serve)

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
