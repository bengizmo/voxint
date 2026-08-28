"""Benchmark runner: submit corpus files, poll, score, record results."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import platform
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from voxint import __version__
from voxint.benchmark.resources import corpus_wav_bytes, load_manifest
from voxint.benchmark.scorer import (
    BenchmarkSummary,
    WERCounts,
    compute_wer,
    pool_wer,
    protocol_hash,
)
from voxint.db.models import (
    BenchmarkItem,
    BenchmarkItemStatus,
    BenchmarkRun,
    BenchmarkRunStatus,
    PipelineRun,
    RunStatus,
    StageRun,
    StageStatus,
    TranscriptSegment,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def _collect_system_info() -> dict[str, object]:
    """Gather CPU/RAM/OS info for the benchmark run snapshot."""
    info: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["memory_total_bytes"] = int(line.split()[1]) * 1024
                    break
    except Exception:
        pass
    return info


def _extract_hypothesis(session: Session, pipeline_run_id: uuid.UUID) -> str:
    """Extract raw_text from TranscriptSegments in chronological order."""
    segments = session.execute(
        select(TranscriptSegment.raw_text)
        .where(TranscriptSegment.pipeline_run_id == pipeline_run_id)
        .order_by(TranscriptSegment.segment_index)
    ).scalars().all()
    return " ".join(s for s in segments if s)


def _cancel_pipeline_run(
    factory: sessionmaker[Session], run_id: uuid.UUID,
) -> None:
    """Best-effort cancel of a live pipeline run."""
    from voxint.ingest.service import cancel_run

    try:
        with factory() as session:
            cancel_run(session, run_id)
            session.commit()
    except Exception:
        logger.debug("cancel_run(%s) failed (may already be terminal)", run_id)


def _poll_run(
    factory: sessionmaker[Session],
    run_id: uuid.UUID,
    *,
    timeout: float,
    interval: float = 2.0,
) -> str:
    """Poll a pipeline run until it reaches a terminal status or times out.

    Returns the terminal status string, or 'timeout'.
    """
    deadline = time.monotonic() + timeout
    while True:
        with factory() as session:
            run = session.get(PipelineRun, run_id)
            if run is None:
                return "failed"
            status = run.status

        if status in (
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        ):
            return status

        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def _copy_assets_to_media_root(
    media_root: Path, manifest: dict[str, Any],
) -> dict[str, str]:
    """Copy benchmark WAVs into MEDIA_ROOT/benchmark/, verifying sha256."""
    benchmark_dir = media_root / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for entry in manifest["files"]:
        file_id: str = entry["id"]
        filename: str = entry["filename"]
        expected_sha: str = entry["sha256"]
        dest = benchmark_dir / filename
        if dest.exists():
            actual_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                logger.warning(
                    "benchmark/%s sha256 mismatch, replacing", filename,
                )
                dest.unlink()
                dest.write_bytes(corpus_wav_bytes(file_id))
        else:
            dest.write_bytes(corpus_wav_bytes(file_id))
        paths[file_id] = f"benchmark/{filename}"
    return paths


def _finalize_run(
    factory: sessionmaker[Session],
    benchmark_run_id: uuid.UUID,
    *,
    status: str,
    summary: dict[str, object],
) -> None:
    """Mark the benchmark run terminal with summary and finished_at."""
    with factory() as session:
        run = session.get(BenchmarkRun, benchmark_run_id)
        if run is None:
            return
        run.status = status
        run.summary = summary
        run.finished_at = datetime.now(tz=UTC)
        session.commit()


def _skip_pending_items(
    factory: sessionmaker[Session], benchmark_run_id: uuid.UUID,
) -> None:
    """Mark all pending/submitted items as skipped."""
    with factory() as session:
        pending = session.execute(
            select(BenchmarkItem).where(
                BenchmarkItem.benchmark_run_id == benchmark_run_id,
                BenchmarkItem.status.in_([
                    BenchmarkItemStatus.PENDING.value,
                    BenchmarkItemStatus.SUBMITTED.value,
                ]),
            )
        ).scalars().all()
        for item in pending:
            item.status = BenchmarkItemStatus.SKIPPED.value
        session.commit()


def validate_timeout(timeout: float) -> str | None:
    """Return an error message if timeout is invalid, else None."""
    if not math.isfinite(timeout) or timeout <= 0:
        return "timeout must be a finite number greater than 0"
    return None


def run_benchmark(
    factory: sessionmaker[Session],
    media_root: Path,
    *,
    tag: str | None = None,
    timeout_per_file: float = 300.0,
    quick: bool = False,
    config_snapshot: dict[str, Any] | None = None,
    write: Callable[[str], object] | None = None,
) -> tuple[uuid.UUID, bool]:
    """Execute a full benchmark run: submit, poll, score, record.

    Returns (benchmark_run_id, all_succeeded). Handles Ctrl+C and
    unexpected exceptions by marking in-flight items as skipped and
    the run as failed.
    """
    import sys

    from voxint.ingest import submit_media_item

    _write: Callable[[str], object] = sys.stderr.write if write is None else write

    manifest = load_manifest()
    corpus_version: int = manifest["corpus_version"]
    proto_hash = protocol_hash()

    files = manifest["files"]
    if quick:
        files = [f for f in files if f["category"] == "speech"]

    file_paths = _copy_assets_to_media_root(media_root, manifest)

    system_info = _collect_system_info()
    if config_snapshot is None:
        config_snapshot = {}

    now = datetime.now(tz=UTC)
    benchmark_run = BenchmarkRun(
        id=uuid.uuid4(),
        tag=tag[:60] if tag else None,
        status=BenchmarkRunStatus.RUNNING.value,
        corpus_version=corpus_version,
        protocol_hash=proto_hash,
        voxint_version=__version__,
        config_snapshot=config_snapshot,
        system_info=system_info,
        started_at=now,
    )

    items: list[BenchmarkItem] = []
    for entry in files:
        item = BenchmarkItem(
            id=uuid.uuid4(),
            benchmark_run_id=benchmark_run.id,
            corpus_file_id=entry["id"],
            status=BenchmarkItemStatus.PENDING.value,
        )
        items.append(item)

    with factory() as session:
        session.add(benchmark_run)
        session.add_all(items)
        session.commit()

    run_start = time.monotonic()
    speech_wer_counts: list[WERCounts] = []
    hallucination_total = 0
    hallucination_nonempty = 0
    hallucination_file_count = 0
    speech_count = 0
    failed_count = 0
    interrupted = False
    current_pipeline_run_id: uuid.UUID | None = None

    try:
        for i, entry in enumerate(files, 1):
            file_id: str = entry["id"]
            category: str = entry["category"]
            source_path = file_paths[file_id]

            _write(f"  [{i}/{len(files)}] {file_id} ... ")

            item_start = time.monotonic()
            with factory() as session:
                item = session.execute(
                    select(BenchmarkItem).where(
                        BenchmarkItem.benchmark_run_id == benchmark_run.id,
                        BenchmarkItem.corpus_file_id == file_id,
                    )
                ).scalar_one()

                item.status = BenchmarkItemStatus.SUBMITTED.value
                item.started_at = datetime.now(tz=UTC)

                result = submit_media_item(session, source_path)
                run_id = result.run_id
                current_pipeline_run_id = run_id
                item.pipeline_run_id = run_id
                session.commit()

                published = result.publish()
                if not published:
                    _write("warning: broker unavailable, waiting for recovery sweep\n")

            terminal = _poll_run(factory, run_id, timeout=timeout_per_file)

            if terminal == "timeout":
                _cancel_pipeline_run(factory, run_id)

            current_pipeline_run_id = None

            with factory() as session:
                item = session.execute(
                    select(BenchmarkItem).where(
                        BenchmarkItem.benchmark_run_id == benchmark_run.id,
                        BenchmarkItem.corpus_file_id == file_id,
                    )
                ).scalar_one()

                item.finished_at = datetime.now(tz=UTC)
                elapsed = time.monotonic() - item_start

                if terminal == RunStatus.COMPLETED.value:
                    item.status = BenchmarkItemStatus.COMPLETED.value

                    stage_runs = session.execute(
                        select(StageRun)
                        .where(
                            StageRun.pipeline_run_id == run_id,
                            StageRun.status == StageStatus.COMPLETED.value,
                        )
                        .order_by(StageRun.stage, StageRun.attempt.desc())
                    ).scalars().all()
                    timings: dict[str, float] = {}
                    for sr in stage_runs:
                        if sr.stage not in timings and sr.started_at and sr.finished_at:
                            delta = (sr.finished_at - sr.started_at).total_seconds()
                            timings[sr.stage] = delta
                    item.stage_timings = timings

                    hypothesis = _extract_hypothesis(session, run_id)
                    ref_transcript = entry.get("reference_transcript")

                    if category == "speech" and ref_transcript:
                        wer = compute_wer(ref_transcript, hypothesis)
                        item.wer_counts = wer.to_dict()
                        speech_wer_counts.append(wer)
                        speech_count += 1
                        _write(f"WER {wer.wer:.1%} ({elapsed:.1f}s)\n")
                    else:
                        words = len(hypothesis.split()) if hypothesis.strip() else 0
                        item.hallucination_words = words
                        hallucination_total += words
                        hallucination_file_count += 1
                        if words > 0:
                            hallucination_nonempty += 1
                        label = f"{words} hallucinated words" if words else "clean"
                        _write(f"{label} ({elapsed:.1f}s)\n")
                else:
                    item.status = BenchmarkItemStatus.FAILED.value
                    item.error = f"pipeline {terminal}"
                    failed_count += 1
                    _write(f"FAILED ({terminal})\n")

                session.commit()

    except KeyboardInterrupt:
        interrupted = True
        _write("\ninterrupted; saving partial results...\n")
        if current_pipeline_run_id is not None:
            _cancel_pipeline_run(factory, current_pipeline_run_id)
        _skip_pending_items(factory, benchmark_run.id)

    except Exception:
        logger.exception("benchmark run failed with unexpected error")
        if current_pipeline_run_id is not None:
            _cancel_pipeline_run(factory, current_pipeline_run_id)
        _skip_pending_items(factory, benchmark_run.id)
        failed_count += 1

    total_time = time.monotonic() - run_start
    pooled = pool_wer(speech_wer_counts) if speech_wer_counts else 0.0

    summary = BenchmarkSummary(
        pooled_wer=pooled,
        total_substitutions=sum(c.substitutions for c in speech_wer_counts),
        total_insertions=sum(c.insertions for c in speech_wer_counts),
        total_deletions=sum(c.deletions for c in speech_wer_counts),
        total_reference_words=sum(c.reference_words for c in speech_wer_counts),
        hallucination_total_words=hallucination_total,
        hallucination_nonempty_count=hallucination_nonempty,
        hallucination_file_count=hallucination_file_count,
        speech_file_count=speech_count,
        total_time_s=round(total_time, 2),
    )

    all_succeeded = not interrupted and failed_count == 0
    final_status = (
        BenchmarkRunStatus.COMPLETED.value
        if all_succeeded
        else BenchmarkRunStatus.FAILED.value
    )

    _finalize_run(
        factory, benchmark_run.id,
        status=final_status, summary=summary.to_dict(),
    )

    _write(f"\n{'=' * 50}\n")
    if interrupted:
        _write("Benchmark INTERRUPTED\n")
    elif failed_count:
        _write(f"Benchmark completed with {failed_count} failures\n")
    else:
        _write("Benchmark complete\n")
    if speech_wer_counts:
        _write(f"  Pooled WER: {pooled:.1%}\n")
        _write(f"  Speech files: {speech_count}\n")
    if hallucination_file_count:
        _write(
            f"  Hallucination: {hallucination_total} words "
            f"across {hallucination_nonempty}/{hallucination_file_count} files\n"
        )
    _write(f"  Total time: {total_time:.1f}s\n")
    _write(f"  Run ID: {benchmark_run.id}\n")

    return benchmark_run.id, all_succeeded
