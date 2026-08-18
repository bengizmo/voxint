"""Watch-folder ingest helpers (issue #60).

The orchestration — scan the registered folders, settle-filter, submit each
net-new file, publish, persist the summary — lives in
:func:`voxint.worker.tasks.watch_sweep`. This module holds the two parts worth
unit-testing without a broker or a database: the per-file settle classification
and the sweep-summary shape persisted for the Settings status line.

The feature auto-submits new media dropped into the operator's registered folders
(``app_settings.media_folders``) and SKIPS files already known — a ``MediaItem``
already claims the ``source_path``. "Already known" is exactly that: it includes a
file whose prior run failed (the operator requeues those from the run detail), so
user-facing copy says "already known", never "already transcribed".
"""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings
from voxint.api.setup_wizard import scan_media_folders
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.ingest.service import submit_media_item_if_new


class SettleState(StrEnum):
    """Whether a scanned candidate is ready to submit this sweep."""

    SETTLED = "settled"  # quiescent long enough — safe to ingest
    TOO_FRESH = "too_fresh"  # still within the settle window — retry next sweep
    SKIP = "skip"  # vanished, unreadable, or not a regular file — count and skip


def classify_settle(path: Path, *, now: float, settle_seconds: float) -> SettleState:
    """Classify one candidate file's readiness for ingest.

    A file is :data:`SettleState.SETTLED` when it is a regular (non-symlink) file
    whose most-recent metadata change is at least ``settle_seconds`` old — the
    heuristic that a copy into the watched folder has finished, so the sweep does
    not ingest a file mid-write. ``max(st_mtime, st_ctime)`` is used because a copy
    tool may preserve the source mtime while ctime still moves on the final
    rename/permission set; ``settle_seconds == 0`` accepts immediately.

    Never raises: a file that vanished between the scan and this stat, is
    unreadable, or is not a regular file (a directory/socket that replaced it, a
    dangling symlink) is :data:`SettleState.SKIP` — the caller counts it and moves
    on rather than failing the whole sweep.

    Clock-skew guard: a file whose newest timestamp is further in the FUTURE than
    the whole settle window cannot be an in-progress copy — a NAS/SMB mount with a
    fast server clock, or a recorder stamping a wrong date. Waiting can never settle
    it (wall-clock would have to reach a bogus future), so it is accepted now rather
    than stranded forever and mislabelled "still copying". A small skew within the
    window still waits normally.
    """
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return SettleState.SKIP
    if not stat.S_ISREG(st.st_mode):
        return SettleState.SKIP
    newest = max(st.st_mtime, st.st_ctime)
    if newest - now > settle_seconds:
        return SettleState.SETTLED  # implausibly future timestamp — untrustworthy
    age = now - newest
    return SettleState.SETTLED if age >= settle_seconds else SettleState.TOO_FRESH


@dataclass(frozen=True)
class WatchSweepSummary:
    """The latest watch-sweep outcome, persisted to ``app_settings`` and rendered
    as one plain-language Settings line. The ONLY sweep state kept — no history,
    no per-file ledger."""

    picked_up: int = 0  # net-new files submitted as fresh QUEUED runs
    already_known: int = 0  # files skipped — a MediaItem already claims them
    settling: int = 0  # files still within the settle window this sweep
    deferred: int = 0  # runs committed but not published (broker down)
    stat_errors: int = 0  # candidates that vanished/were unreadable mid-sweep
    hit_entry_cap: bool = False  # the walk stopped at setup_scan_max_entries
    hit_file_cap: bool = False  # net-new stopped at setup_scan_max_files
    root_missing: bool = False  # the configured media root was absent/unmounted this sweep
    completed_at: str | None = None  # ISO-8601; None ⇒ the sweep never ran

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sweep_watch_folders(
    factory: sessionmaker[Session],
    settings: Settings,
    *,
    publish: Callable[[uuid.UUID], bool],
) -> WatchSweepSummary:
    """Run one watch-folder ingest pass and return its summary (issue #60).

    Off unless the EFFECTIVE gate (env default overridden by the runtime
    ``app_settings.watch_folder_enabled`` column) is on — re-checked here, not just
    at beat registration, so an always-present schedule entry no-ops (one DB read,
    no walk) when disabled. Otherwise: walk ``app_settings.media_folders`` via the
    wizard's bounded, containment-safe :func:`scan_media_folders` (which already
    skips files a ``MediaItem`` claims), settle-filter each net-new candidate so a
    file still being copied in is not ingested mid-write, submit the rest with the
    race-safe :func:`submit_media_item_if_new`, COMMIT the whole batch once, then
    ``publish`` each run (``publish`` returns ``False`` on a broker outage, counted
    "deferred" — the durable QUEUED rows are left for the recovery sweep).

    The latest summary is persisted to ``app_settings.watch_folder_last_sweep``,
    newest-wins, for the plain-language Settings status line.
    """
    with factory() as session:
        row = app_settings.get_app_settings(session)
        if not app_settings.resolve_effective_watch_folder_enabled(row, settings):
            return WatchSweepSummary()
        folders = list(row.media_folders) if row and row.media_folders else []
        result = scan_media_folders(session, settings.media_root, folders, settings)
        media_root = settings.media_root.resolve()
        now = datetime.now(tz=UTC).timestamp()
        settling = 0
        stat_errors = 0
        settled: list[str] = []
        for rel in result.candidates:
            state = classify_settle(
                media_root / rel, now=now, settle_seconds=settings.watch_folder_settle_seconds
            )
            if state is SettleState.SETTLED:
                settled.append(rel)
            elif state is SettleState.TOO_FRESH:
                settling += 1
            else:  # SettleState.SKIP — vanished / unreadable / not a regular file
                stat_errors += 1
        run_ids: list[uuid.UUID] = []
        raced_known = 0
        for rel in settled:
            submitted = submit_media_item_if_new(session, rel, settings=settings)
            if submitted is None:
                # A concurrent sweep/submission claimed the path since the scan — it
                # is now known, not newly picked up.
                raced_known += 1
            else:
                run_ids.append(submitted.id)
        # Commit-before-publish: the durable QUEUED rows exist before any enqueue, so
        # a broker outage only defers publishing, never loses a submission.
        session.commit()
    # Publish each committed run. On the FIRST failure the broker is down, so stop
    # retrying (each further apply_async could pay a connect timeout and stall the
    # sweep past its interval) and count the rest deferred — the recovery sweep owns
    # every durable QUEUED row regardless.
    deferred = 0
    broker_down = False
    for rid in run_ids:
        if broker_down:
            deferred += 1
            continue
        if not publish(rid):
            broker_down = True
            deferred += 1
    summary = WatchSweepSummary(
        picked_up=len(run_ids),
        already_known=result.already_known + raced_known,
        settling=settling,
        deferred=deferred,
        stat_errors=stat_errors,
        hit_entry_cap=result.hit_entry_cap,
        hit_file_cap=result.hit_file_cap,
        root_missing=result.root_missing,
        completed_at=datetime.now(tz=UTC).isoformat(),
    )
    _store_summary(factory, settings, summary)
    return summary


def _store_summary(
    factory: sessionmaker[Session], settings: Settings, summary: WatchSweepSummary
) -> None:
    """Persist ``summary`` as the latest watch-sweep status, newest-wins.

    Overwrites ``app_settings.watch_folder_last_sweep`` only when this summary's
    ``completed_at`` is at least as new as the stored one (ISO-8601 UTC strings order
    lexicographically), so an out-of-order overlapping sweep cannot clobber a fresher
    result with a staler one. The read-compare-write runs under a ``FOR UPDATE`` lock
    on the singleton so two overlapping sweeps SERIALIZE on the write: without it both
    could read the same prior and the slower (staler) one commit last, defeating the
    newest-wins guard. ``populate_existing`` forces the post-lock columns over
    ``get_or_create``'s identity-mapped copy (the ``_locked_app_settings`` precedent in
    ``api/app.py``). On SQLite SQLAlchemy renders no locking clause (single-writer);
    the serialization is exercised against the production database.
    """
    with factory() as session:
        app_settings.get_or_create(session, llm_enabled_default=settings.llm_enabled)
        row = session.execute(
            select(AppSettings)
            .where(AppSettings.id == app_settings.SINGLETON_ID)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one()
        prior = row.watch_folder_last_sweep
        prior_at = prior.get("completed_at") if isinstance(prior, dict) else None
        # A summary with no completed_at (never produced by sweep_watch_folders, which
        # always stamps one) must NOT clobber a real prior — so it is not a write.
        if prior_at is None or (
            summary.completed_at is not None and summary.completed_at >= prior_at
        ):
            row.watch_folder_last_sweep = summary.as_dict()
            session.commit()
