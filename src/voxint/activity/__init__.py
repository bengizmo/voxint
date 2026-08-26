"""Console activity outbox — emit + read (issue #162, Console 2.0 P7).

The write side of the ``activity_events`` table behind the console activity
indicator (completion toasts + a Jobs badge). Deliberately shaped like
:mod:`voxint.notify`: persistence-only, in the CALLER's transaction, idempotent
via ``ON CONFLICT DO NOTHING`` on the occurrence key, no HTTP / Celery / clock
beyond the DB default. The browser polls the table directly (see
``routers/activity.py``); there is no delivery lifecycle.

An event carries a frozen ``title``/``href`` snapshot resolved at emission so the
poll is a pure ``id``-cursor read that never re-scans the source tables. Two kinds
exist: ``run_completed`` (a run reaching COMPLETED, emitted from
``cas_update_run``) and ``speaker_identified`` (an operator naming a diarization
label — assign / enroll / merge, emitted from the adjudication routes). Both are
run-scoped and snapshot-only; neither carries per-kind provenance columns.

Emission lives at the ORCHESTRATION layer (the transition / the routes) where the
``console_activity_enabled`` flag is in hand, never inside the low-level ledger
writer ``adjudication.ledger.record_decision`` (kept policy-free). Any future
``record_decision`` caller that should announce must emit here as well.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult

from voxint.api.presentation import friendly_media_label, title_from_snapshot
from voxint.db.models import ActivityEvent, ActivityKind, PipelineRun

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Newest-N kept by the ``voxint.activity_prune`` sweep. A small fixed cap: the
# outbox is a recent-activity feed for a single operator, not an audit log, so it
# earns no tuning knob (anti-bloat).
ACTIVITY_RETENTION_MAX = 500

# Upper bound on one poll page. The client drains ascending pages via the
# returned cursor, so this only caps a single request; 50 is generous for a
# 15s poll on a single-operator tool.
ACTIVITY_POLL_LIMIT = 50

# The frozen snapshot columns are CHECK-bounded at 500 chars; a media label is
# never truncated by ``friendly_media_label`` (the template does that in CSS), so
# clamp here rather than let an unusually long title roll back a run completion.
_TITLE_MAX = 500


def record_activity_event(
    session: Session,
    *,
    kind: ActivityKind,
    occurrence_key: str,
    pipeline_run_id: uuid.UUID,
    title: str,
    href: str,
) -> None:
    """Insert one activity row in the caller's transaction, idempotently.

    ``ON CONFLICT DO NOTHING`` on ``occurrence_key`` so a retried emission of the
    same occurrence never doubles a row. Runs in the SAME transaction as the
    change it announces: if the caller rolls back, the event rolls back with it
    (never announce something that did not commit).
    """
    stmt = (
        pg_insert(ActivityEvent)
        .values(
            kind=kind.value,
            occurrence_key=occurrence_key,
            pipeline_run_id=pipeline_run_id,
            # Both snapshot columns are CHECK-bounded at 500; clamp both so an
            # unusually long value can never turn a run completion into a failed
            # (rolled-back) transaction.
            title=title[:_TITLE_MAX],
            href=href[:_TITLE_MAX],
        )
        .on_conflict_do_nothing(constraint="uq_activity_events_occurrence_key")
    )
    session.execute(stmt)


def resolve_run_completed_snapshot(session: Session, run_id: uuid.UUID) -> tuple[str, str]:
    """Freeze the (title, href) presentation snapshot for a completed run.

    Resolves the media label through the same precedence the console uses
    (frozen sidecar title, else scraped source-metadata title, else a cleaned
    filename — mirrors ``run_source_title``), inside the caller transaction. The
    href points at the routed ``/jobs/{run_id}`` detail page: the Media editor
    (``/media/{id}``) does not exist yet, and a frozen href is never repointed.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:  # pragma: no cover - the caller just updated this run
        return (f"run {run_id}", f"/jobs/{run_id}")
    title = title_from_snapshot(run.sidecar)
    if title is None and run.media_item.source_metadata is not None:
        title = run.media_item.source_metadata.title
    label = friendly_media_label(title, run.media_item.source_path)
    return (label, f"/jobs/{run_id}")


def record_run_completed(session: Session, run_id: uuid.UUID) -> None:
    """Emit the run-completion activity event for ``run_id`` (caller's tx)."""
    title, href = resolve_run_completed_snapshot(session, run_id)
    record_activity_event(
        session,
        kind=ActivityKind.RUN_COMPLETED,
        occurrence_key=f"run:{run_id}:completed",
        pipeline_run_id=run_id,
        title=title,
        href=href,
    )


def record_speaker_identified(
    session: Session,
    *,
    run_id: uuid.UUID,
    decision_id: uuid.UUID,
    speaker_name: str,
) -> None:
    """Emit one speaker-identification event (caller's tx).

    For a single positive identification: an ``ASSIGN`` naming a speaker (label or
    segment scope) or a new-speaker enrollment. Keyed on the ledger ``decision_id``
    (globally unique, stable under idempotent replay), so replaying the decision
    re-emits the same occurrence and ``ON CONFLICT DO NOTHING`` keeps it one row.
    ``speaker_name`` must be the authoritative roster name resolved in this tx, not
    submitted form input (a replay may name a since-renamed speaker).
    """
    record_activity_event(
        session,
        kind=ActivityKind.SPEAKER_IDENTIFIED,
        occurrence_key=f"decision:{decision_id}:identified",
        pipeline_run_id=run_id,
        title=speaker_name,
        href=f"/jobs/{run_id}",
    )


def record_speaker_merge(
    session: Session,
    *,
    run_id: uuid.UUID,
    occurrence_decision_id: uuid.UUID,
    survivor_name: str,
    label_count: int,
) -> None:
    """Emit ONE speaker-identification event for a merge (caller's tx).

    A merge assigns N labels to a single survivor, but the operator performed one
    consolidation, so the feed carries one event (server-side coalescing) rather
    than N near-identical toasts. Keyed on a stable per-merge ledger decision id
    (the smallest of the merge's decision ids — a nonce is not collision-safe
    across runs / label sets), so a replayed merge re-emits the same occurrence.
    ``label_count`` is the resolved (deduplicated) label count, not raw input.
    """
    record_activity_event(
        session,
        kind=ActivityKind.SPEAKER_IDENTIFIED,
        occurrence_key=f"merge:{occurrence_decision_id}",
        pipeline_run_id=run_id,
        title=f"{survivor_name} ({label_count} labels)",
        href=f"/jobs/{run_id}",
    )


def events_since(session: Session, *, after_id: int, limit: int) -> list[ActivityEvent]:
    """Activity rows with ``id`` greater than ``after_id``, oldest first, capped.

    Ascending so the client renders toasts in occurrence order and advances its
    cursor to the last id it saw; the caller signals ``has_more`` when the page
    fills, and re-polls to drain the rest without skipping any row.

    Delivery is best-effort by design: ``id`` is a sequence, assigned at insert,
    not at commit, so two run completions that commit out of insert order can let
    a poll landing between the two commits advance past the earlier id and miss
    its toast. That is an accepted limitation for a cosmetic notification on a
    single-operator tool: the Jobs badge (a live count, not cursor-based) and the
    Jobs page remain authoritative, and serializing the pipeline's terminal
    completion transaction for a popup is not a trade this tool makes.
    """
    rows = session.execute(
        select(ActivityEvent)
        .where(ActivityEvent.id > after_id)
        .order_by(ActivityEvent.id)
        .limit(limit)
    ).scalars()
    return list(rows)


def high_water(session: Session) -> int:
    """The largest activity ``id`` (0 when the table is empty).

    The bootstrap cursor: a fresh browser baselines here so it does not toast the
    retained backlog, and a stale/pruned cursor rebaselines to it.
    """
    return int(session.execute(select(func.coalesce(func.max(ActivityEvent.id), 0))).scalar_one())


def retained_floor(session: Session) -> int:
    """The smallest retained activity ``id`` (0 when the table is empty).

    Lets the poll endpoint report the retention floor so a client whose stored
    cursor sits below it (events pruned while its tab was closed) can tell it has
    missed events and resync to the high-water mark rather than replay the
    retained backlog as if those completions were new.
    """
    return int(session.execute(select(func.coalesce(func.min(ActivityEvent.id), 0))).scalar_one())


def prune_activity_events(session: Session, *, keep: int = ACTIVITY_RETENTION_MAX) -> int:
    """Delete all but the newest ``keep`` rows; return the number removed.

    Gap-safe: the threshold is the id of the Nth-newest row (``OFFSET keep-1``),
    not ``max(id) - keep`` — identity gaps from rolled-back inserts would make the
    arithmetic form delete too much. A no-op below ``keep`` rows and on an empty
    table (the subquery yields NULL, so the ``<`` predicate is never true).
    """
    if keep < 1:  # OFFSET keep-1 would be negative; there is no legitimate keep<1
        return 0
    threshold = (
        select(ActivityEvent.id)
        .order_by(ActivityEvent.id.desc())
        .offset(keep - 1)
        .limit(1)
        .scalar_subquery()
    )
    result = cast(
        "CursorResult[Any]",
        session.execute(delete(ActivityEvent).where(ActivityEvent.id < threshold)),
    )
    return result.rowcount or 0
