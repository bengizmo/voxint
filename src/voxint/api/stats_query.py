"""Aggregate, read-only system statistics: the data behind ``voxint stats`` and
``GET /metrics``.

A pure query module in the same shape as :mod:`voxint.api.runs_query` — frozen
dataclasses and functions that take a :class:`~sqlalchemy.orm.Session` and issue
one ``SELECT`` each, with no HTTP and no side effects. The rendering half
(``format_stats_text`` / ``stats_to_json`` / ``render_prometheus``) is pure
Python over a :class:`SystemStats`, so it unit-tests without a database; only the
query functions need a live Postgres.

Semantics worth stating plainly (they are contract, not incidental):

- ``stage_failure_counts`` and ``stage_duration_stats`` count **stage attempts**,
  not distinct runs — a retried stage contributes multiple ``stage_runs`` rows.
- A stage duration is ``finished_at - started_at`` over attempts that have a
  ``finished_at`` (completed, failed, or skipped), guarded to non-negative.
- ``runs_created_*`` counts runs *created* since a cutoff — there is no run-level
  completion timestamp, so this is submission throughput, not completion rate.
- :func:`collect_stats` issues several statements under READ COMMITTED; it is a
  fast snapshot, not an atomic report. That is fine for single-operator
  observability.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import Float, func
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.db.models import PipelineRun, RunStatus, Speaker, Stage, StageRun, StageStatus

_RELATIVE_SINCE = re.compile(r"^(\d+)([hd])$")

# The default throughput window, shared by every surface that shows "runs created
# since" so /dashboard, /metrics, and `voxint stats` cannot silently disagree.
# Two string-encoded siblings encode the same 24h and MUST move with it: the
# Prometheus metric name ``voxint_runs_created_24h`` (render_prometheus, below)
# and the CLI ``--since`` default ("24h", cli.py).
DEFAULT_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class StageDurationStat:
    """Average wall-clock duration of a stage, over finished attempts."""

    stage: str
    attempt_count: int
    avg_seconds: float


@dataclass(frozen=True)
class StageFailureCount:
    """Failed stage attempts for one stage (attempts, not distinct runs)."""

    stage: str
    attempt_count: int


@dataclass(frozen=True)
class SystemStats:
    """One aggregate snapshot of the system, ready to render."""

    status_counts: Mapping[str, int]
    stage_failure_counts: tuple[StageFailureCount, ...]
    stage_durations: tuple[StageDurationStat, ...]
    roster_size: int
    runs_created_since: datetime
    runs_created_count: int
    generated_at: datetime
    since: datetime


def parse_since(raw: str, *, now: datetime) -> datetime:
    """Resolve a ``--since`` value to an aware UTC cutoff.

    Accepts a relative ``<int>h`` / ``<int>d`` span (computed back from ``now``)
    or an ISO-8601 datetime. A naive ISO value is rejected: compared against the
    ``TIMESTAMPTZ`` ``created_at`` column Postgres would coerce it using the
    session timezone, silently moving the boundary between machines. ``now`` is
    injected so callers (and tests) control the clock; it must be aware.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    value = raw.strip()
    match = _RELATIVE_SINCE.match(value)
    if match is not None:
        amount = int(match.group(1))
        unit = match.group(2)
        try:
            # OverflowError can fire either building the timedelta (C-int limits)
            # or subtracting below datetime.min — a span so large it's a bad value,
            # not a usable window. Map it to the same sanitized ValueError as any
            # other unparseable --since (the CLI turns that into exit 2, not a
            # traceback).
            delta = timedelta(hours=amount) if unit == "h" else timedelta(days=amount)
            return (now - delta).astimezone(UTC)
        except OverflowError as exc:
            raise ValueError(f"invalid --since {raw!r}: window is too large") from exc
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid --since {raw!r}: expected '<n>h', '<n>d', or an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(f"invalid --since {raw!r}: ISO datetime must carry a timezone offset")
    return parsed.astimezone(UTC)


def run_status_counts(session: Session) -> dict[str, int]:
    """Count runs grouped by status (all time). Empty statuses are absent here."""
    rows = session.execute(
        sa_select(PipelineRun.status, func.count()).group_by(PipelineRun.status)
    ).all()
    return {status: count for status, count in rows}


def stage_failure_counts(session: Session) -> list[StageFailureCount]:
    """Failed stage attempts per stage (all time), ordered by pipeline order."""
    rows = session.execute(
        sa_select(StageRun.stage, func.count())
        .where(StageRun.status == StageStatus.FAILED.value)
        .group_by(StageRun.stage)
    ).all()
    counts = {stage: count for stage, count in rows}
    return [
        StageFailureCount(stage=stage.value, attempt_count=counts[stage.value])
        for stage in Stage
        if stage.value in counts
    ]


def stage_duration_stats(session: Session) -> list[StageDurationStat]:
    """Average finished-attempt duration per stage (all time), in pipeline order.

    ``AVG(EXTRACT(EPOCH ...))`` is cast to float in SQL so the result is a Python
    ``float`` rather than ``Decimal``. Only attempts with a ``finished_at`` at or
    after ``started_at`` count — a guard against clock-skew negatives.
    """
    seconds = func.extract("epoch", StageRun.finished_at - StageRun.started_at)
    rows = session.execute(
        sa_select(
            StageRun.stage,
            func.count(),
            func.avg(seconds).cast(Float),
        )
        .where(
            StageRun.finished_at.isnot(None),
            StageRun.finished_at >= StageRun.started_at,
        )
        .group_by(StageRun.stage)
    ).all()
    stats = {stage: (count, avg) for stage, count, avg in rows}
    return [
        StageDurationStat(
            stage=stage.value,
            attempt_count=stats[stage.value][0],
            avg_seconds=float(stats[stage.value][1]),
        )
        for stage in Stage
        if stage.value in stats
    ]


def roster_size(session: Session) -> int:
    """How many speakers are enrolled in the roster."""
    return session.execute(sa_select(func.count()).select_from(Speaker)).scalar_one()


def runs_created_since(session: Session, *, since: datetime) -> int:
    """Count runs created at or after ``since`` (inclusive)."""
    return session.execute(
        sa_select(func.count()).select_from(PipelineRun).where(PipelineRun.created_at >= since)
    ).scalar_one()


def collect_stats(session: Session, *, since: datetime, now: datetime) -> SystemStats:
    """Assemble a :class:`SystemStats` snapshot from the individual queries."""
    return SystemStats(
        status_counts=run_status_counts(session),
        stage_failure_counts=tuple(stage_failure_counts(session)),
        stage_durations=tuple(stage_duration_stats(session)),
        roster_size=roster_size(session),
        runs_created_since=since,
        runs_created_count=runs_created_since(session, since=since),
        generated_at=now,
        since=since,
    )


# ---- Rendering (pure; no DB) ------------------------------------------------


def format_stats_text(stats: SystemStats) -> str:
    """A compact human table of the snapshot for the terminal."""
    lines: list[str] = []
    lines.append("Runs by status:")
    if stats.status_counts:
        for status in RunStatus:
            count = stats.status_counts.get(status.value)
            if count:
                lines.append(f"  {status.value:<22} {count}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"Roster size: {stats.roster_size}")
    lines.append(
        f"Runs created since {stats.runs_created_since.isoformat()}: {stats.runs_created_count}"
    )

    lines.append("")
    lines.append("Stage failures (attempts):")
    if stats.stage_failure_counts:
        for failure in stats.stage_failure_counts:
            lines.append(f"  {failure.stage:<16} {failure.attempt_count}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Avg stage duration (finished attempts):")
    if stats.stage_durations:
        for duration in stats.stage_durations:
            lines.append(
                f"  {duration.stage:<16} {duration.avg_seconds:8.2f}s"
                f"  (n={duration.attempt_count})"
            )
    else:
        lines.append("  (none)")

    return "\n".join(lines) + "\n"


def stats_to_json(stats: SystemStats) -> dict[str, object]:
    """The snapshot as JSON-serialisable primitives (stable key set)."""
    return {
        "generated_at": stats.generated_at.isoformat(),
        "status_counts": {
            status.value: stats.status_counts.get(status.value, 0) for status in RunStatus
        },
        "roster_size": stats.roster_size,
        "runs_created_since": stats.runs_created_since.isoformat(),
        "runs_created_count": stats.runs_created_count,
        "stage_failure_counts": {f.stage: f.attempt_count for f in stats.stage_failure_counts},
        "stage_durations": [
            {
                "stage": d.stage,
                "attempt_count": d.attempt_count,
                "avg_seconds": d.avg_seconds,
            }
            for d in stats.stage_durations
        ],
    }


def _escape_label(value: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(stats: SystemStats) -> str:
    """Prometheus text exposition (format 0.0.4) for the snapshot.

    Every known ``RunStatus`` and ``Stage`` series is emitted, zero-filled when
    absent, so a series never disappears between scrapes. The one windowed value
    (``voxint_runs_created_24h``) bakes its window into the metric name so a
    scrape's meaning cannot silently drift.
    """
    out: list[str] = []

    # Gauges, not counters: these are recomputed from current DB contents each
    # scrape and can decrease (rows deleted, a requeue clearing a failure), so
    # they carry no `_total` suffix — that convention is reserved for monotonic
    # counters and `promtool check metrics` flags it on a gauge.
    out.append("# HELP voxint_runs Pipeline runs grouped by status.")
    out.append("# TYPE voxint_runs gauge")
    for status in RunStatus:
        count = stats.status_counts.get(status.value, 0)
        out.append(f'voxint_runs{{status="{_escape_label(status.value)}"}} {count}')

    out.append("# HELP voxint_stage_failures Failed stage attempts grouped by stage.")
    out.append("# TYPE voxint_stage_failures gauge")
    failures = {f.stage: f.attempt_count for f in stats.stage_failure_counts}
    for stage in Stage:
        count = failures.get(stage.value, 0)
        out.append(f'voxint_stage_failures{{stage="{_escape_label(stage.value)}"}} {count}')

    out.append(
        "# HELP voxint_stage_duration_seconds Average finished-attempt duration per stage."
    )
    out.append("# TYPE voxint_stage_duration_seconds gauge")
    durations = {d.stage: d.avg_seconds for d in stats.stage_durations}
    for stage in Stage:
        avg = durations.get(stage.value, 0.0)
        out.append(
            f'voxint_stage_duration_seconds{{stage="{_escape_label(stage.value)}"}} {avg}'
        )

    # A companion count so a scrape can tell "no finished attempts" (0 samples,
    # avg zero-filled above) from "a genuinely instantaneous stage" — the average
    # alone conflates them.
    out.append(
        "# HELP voxint_stage_duration_attempts Finished attempts behind each duration average."
    )
    out.append("# TYPE voxint_stage_duration_attempts gauge")
    attempts = {d.stage: d.attempt_count for d in stats.stage_durations}
    for stage in Stage:
        n = attempts.get(stage.value, 0)
        out.append(
            f'voxint_stage_duration_attempts{{stage="{_escape_label(stage.value)}"}} {n}'
        )

    out.append("# HELP voxint_roster_speakers Enrolled speakers in the roster.")
    out.append("# TYPE voxint_roster_speakers gauge")
    out.append(f"voxint_roster_speakers {stats.roster_size}")

    out.append("# HELP voxint_runs_created_24h Runs created in the last 24 hours.")
    out.append("# TYPE voxint_runs_created_24h gauge")
    out.append(f"voxint_runs_created_24h {stats.runs_created_count}")

    return "\n".join(out) + "\n"
