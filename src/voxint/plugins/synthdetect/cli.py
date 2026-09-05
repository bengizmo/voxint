"""CLI commands for the synthdetect plugin."""

from __future__ import annotations

import argparse
import uuid


def register_commands(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``voxint synthdetect`` commands."""
    synthdetect_p = subparsers.add_parser(
        "synthdetect",
        help="synthetic speech detection commands",
    )
    synthdetect_sub = synthdetect_p.add_subparsers(
        dest="synthdetect_command", required=True
    )
    backfill_p = synthdetect_sub.add_parser(
        "backfill",
        help="score completed runs (stale-only by default; --force scores all)",
    )
    backfill_p.add_argument(
        "run_id",
        nargs="?",
        type=uuid.UUID,
        help="score a single run by ID",
    )
    backfill_p.add_argument(
        "--force",
        action="store_true",
        help="score all completed runs, not just stale ones",
    )
    backfill_p.set_defaults(fn=_synthdetect_backfill)


def _synthdetect_backfill(args: argparse.Namespace) -> int:
    """Score completed runs that lack or have stale synthdetect results."""
    from sqlalchemy import select

    from voxint.app_settings import get_app_settings
    from voxint.config import SettingsError, get_settings
    from voxint.db.models import (
        PipelineRun,
        RunStatus,
        SynthdetectJob,
        SynthdetectJobStatus,
    )
    from voxint.db.session import build_engine, build_session_factory
    from voxint.plugins.synthdetect.client import HttpSynthdetectClient
    from voxint.plugins.synthdetect.jobs import (
        SynthdetectHashError,
        active_synthdetect_job,
        create_job,
        execute_job,
        runs_needing_synthdetect,
        synthdetect_gates_open,
    )

    try:
        settings = get_settings()
    except SettingsError as exc:
        print(f"error: {exc}")
        return 2

    factory = build_session_factory(build_engine(settings.database_url))
    with factory() as session:
        if not synthdetect_gates_open(settings, get_app_settings(session)):
            print("error: synthdetect is disabled")
            return 2

    client = HttpSynthdetectClient(
        settings.synthdetect_url,
        timeout=settings.synthdetect_http_timeout_seconds,
    )
    try:
        client.healthz()
    except Exception as exc:
        print(f"error: synthdetect service unreachable — {exc}")
        return 2

    with factory() as session:
        if args.run_id is not None:
            run_ids = [args.run_id]
        elif args.force:
            run_ids = list(
                session.execute(
                    select(PipelineRun.id)
                    .where(PipelineRun.status == RunStatus.COMPLETED.value)
                    .order_by(PipelineRun.created_at)
                ).scalars()
            )
        else:
            run_ids = runs_needing_synthdetect(session, settings=settings)

    if not run_ids:
        print("nothing to score — every completed run is up to date")
        return 0

    failures = 0
    for run_id in run_ids:
        with factory() as session:
            try:
                job, already_active = create_job(
                    session, pipeline_run_id=run_id, settings=settings
                )
                session.commit()
            except SynthdetectHashError as exc:
                print(f"{run_id}: skipped — {exc}")
                continue
        if already_active:
            with factory() as session:
                stranded = active_synthdetect_job(session, run_id)
                stranded_id = stranded.id if stranded is not None else None
                stranded_status = stranded.status if stranded is not None else None
            if stranded_id is None:
                print(f"{run_id}: active job resolved — rerun backfill to score it")
                continue
            if stranded_status != SynthdetectJobStatus.QUEUED.value:
                print(
                    f"{run_id}: skipped — a job is already running"
                    " (cancel it from the run page if it is stuck)"
                )
                continue
            print(f"job {stranded_id}: recovering stranded run {run_id} ...")
            execute_job(factory, stranded_id, settings=settings)
            with factory() as session:
                recovered = session.get(SynthdetectJob, stranded_id)
                assert recovered is not None
                if recovered.status == SynthdetectJobStatus.RUNNING.value:
                    print(f"{run_id}: claimed by another worker — skipped")
                elif recovered.status == SynthdetectJobStatus.QUEUED.value:
                    print(f"{run_id}: not claimed — rerun backfill to score it")
                else:
                    print(f"{run_id}: {recovered.status}")
                    if recovered.error:
                        print(f"error: {recovered.error}")
                    if recovered.status == SynthdetectJobStatus.FAILED.value:
                        failures += 1
            continue
        assert job is not None
        job_id = job.id
        print(f"job {job_id}: scoring run {run_id} ...")
        execute_job(factory, job_id, settings=settings)
        with factory() as session:
            finished = session.get(SynthdetectJob, job_id)
            assert finished is not None
            print(f"{run_id}: {finished.status}")
            if finished.error:
                print(f"error: {finished.error}")
            if finished.status != "succeeded":
                failures += 1
    return 1 if failures else 0
