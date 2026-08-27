"""Watch-folder ingest helpers (issue #60).

The orchestration — scan the registered folders, settle-filter, submit each
net-new file, publish, persist the summary — lives in
:func:`voxint.worker.tasks.watch_sweep`. This module holds the two parts worth
unit-testing without a broker or a database: the per-file settle classification
and the sweep-summary shape persisted for the Settings status line.

The feature auto-submits new media dropped into the operator's registered folders
(the ``media_folders`` relation, ``watch=true``) and SKIPS files already known — a ``MediaItem``
already claims the ``source_path``. "Already known" is exactly that: it includes a
file whose prior run failed (the operator requeues those from the run detail), so
user-facing copy says "already known", never "already transcribed".

Sidecars (issue #104): a media file may arrive with a companion YAML sidecar
(``clip.wav.yaml`` or ``clip.yaml``; see :mod:`voxint.ingest.sidecar`). The pair
settles together — a settled media file whose sidecar is still being written
waits for the next sweep — and a sidecar with a problem (bad YAML, a bad value
on a known key, an ambiguous stem name, an unknown ``domain_pack``) HOLDS its
media file un-submitted, counted ``sidecar_errors`` and retried every sweep
until the operator fixes the file. The sidecar is frozen at submit: one that
first appears (or is edited) after its media file is already known does
nothing, by design — the run keeps the snapshot it was submitted with.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint import app_settings
from voxint.api.setup_wizard import scan_media_folders
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.registry import resolve_domain_pack_by_name
from voxint.ingest.service import SubmissionResult, submit_media_item_if_new
from voxint.ingest.sidecar import Sidecar, SidecarError, find_sidecar, read_sidecar
from voxint.media.registration import watched_folder_paths

logger = logging.getLogger(__name__)


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
    stat_errors: int = 0  # candidates that couldn't be picked up (vanished/unreadable,
    # or a domain-pack collision that made the run un-submittable — see the sweep log)
    sidecar_errors: int = 0  # media held because its companion .yaml sidecar has a
    # problem (malformed, ambiguous stem, unknown pack) — retried every sweep
    hit_entry_cap: bool = False  # the walk stopped at setup_scan_max_entries
    hit_file_cap: bool = False  # net-new stopped at setup_scan_max_files
    root_missing: bool = False  # the configured media root was absent/unmounted this sweep
    completed_at: str | None = None  # ISO-8601; None ⇒ the sweep never ran

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sweep_watch_folders(
    factory: sessionmaker[Session],
    settings: Settings,
) -> WatchSweepSummary:
    """Run one watch-folder ingest pass and return its summary (issue #60).

    Off unless the EFFECTIVE gate (env default overridden by the runtime
    ``app_settings.watch_folder_enabled`` column) is on — re-checked here, not just
    at beat registration, so an always-present schedule entry no-ops (one DB read,
    no walk) when disabled. Otherwise: walk the ``watch=true`` ``media_folders``
    rows via the wizard's bounded, containment-safe :func:`scan_media_folders` (which already
    skips files a ``MediaItem`` claims), settle-filter each net-new candidate so a
    file still being copied in is not ingested mid-write, submit the rest with the
    race-safe :func:`submit_media_item_if_new`, COMMIT the whole batch once, then
    call :meth:`~SubmissionResult.publish` on each result (returns ``False`` on a
    broker outage, counted "deferred" — the durable QUEUED rows are left for the
    recovery sweep).

    The latest summary is persisted to ``app_settings.watch_folder_last_sweep``,
    newest-wins, for the plain-language Settings status line.
    """
    with factory() as session:
        row = app_settings.get_app_settings(session)
        if not app_settings.resolve_effective_watch_folder_enabled(row, settings):
            return WatchSweepSummary()
        folders = watched_folder_paths(session)
        # The net-new file cap is applied to SUBMISSIONS below, not to the scan:
        # capped at the scan, a few permanently held files (a malformed sidecar
        # is held every sweep until fixed) at the front of the walk order would
        # occupy every cap slot and starve the files behind them forever.
        result = scan_media_folders(
            session, settings.media_root, folders, settings, apply_file_cap=False
        )
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
        pending: list[SubmissionResult] = []
        raced_known = 0
        sidecar_errors = 0
        hit_file_cap = result.hit_file_cap
        for rel in settled:
            if len(pending) >= settings.setup_scan_max_files:
                # The submission cap: everything beyond it simply waits for the
                # next sweep (same operator-facing meaning hit_file_cap always had).
                hit_file_cap = True
                break
            outcome = _paired_sidecar(
                media_root / rel,
                now=now,
                settle_seconds=settings.watch_folder_settle_seconds,
                settings=settings,
            )
            if outcome is _SidecarHold.SETTLING:
                settling += 1
                continue
            if outcome is _SidecarHold.HELD:
                sidecar_errors += 1
                continue
            try:
                submitted = submit_media_item_if_new(
                    session, rel, settings=settings, sidecar=outcome
                )
            except DomainPackError:
                # A freeze-time domain-pack collision (issue #84) / unresolvable pack
                # (issue #11) is a PERSISTENT operator config error, not a transient
                # per-file fault. Log and skip so one bad folder→pack mapping can't
                # crash the recurring beat sweep (which would otherwise silently stop
                # ingesting every other folder). The operator sees the same collision,
                # with a plain-language fix, the moment they submit via the console/CLI.
                # (A sidecar naming an UNKNOWN pack never reaches here — it is
                # pre-validated in _paired_sidecar and counted sidecar_errors.)
                logger.warning(
                    "watch sweep skipped %s: domain pack could not be applied "
                    "(check Settings → Corrections and this folder's pack)",
                    rel,
                )
                stat_errors += 1
                continue
            if submitted is None:
                # A concurrent sweep/submission claimed the path since the scan — it
                # is now known, not newly picked up.
                raced_known += 1
            else:
                pending.append(submitted)
        # Commit-before-publish: the durable QUEUED rows exist before any enqueue, so
        # a broker outage only defers publishing, never loses a submission.
        session.commit()
    # Publish each committed run. On the FIRST failure the broker is down, so stop
    # retrying (each further apply_async could pay a connect timeout and stall the
    # sweep past its interval) and count the rest deferred — the recovery sweep owns
    # every durable QUEUED row regardless.
    deferred = 0
    broker_down = False
    for sub in pending:
        if broker_down:
            deferred += 1
            continue
        if not sub.publish():
            broker_down = True
            deferred += 1
    summary = WatchSweepSummary(
        picked_up=len(pending),
        already_known=result.already_known + raced_known,
        settling=settling,
        deferred=deferred,
        stat_errors=stat_errors,
        sidecar_errors=sidecar_errors,
        hit_entry_cap=result.hit_entry_cap,
        hit_file_cap=hit_file_cap,
        root_missing=result.root_missing,
        completed_at=datetime.now(tz=UTC).isoformat(),
    )
    _store_summary(factory, settings, summary)
    return summary


class _SidecarHold(Enum):
    """_paired_sidecar outcomes that are not a Sidecar: the pair is not ready
    yet (retry next sweep as "settling"), or the sidecar has a problem (hold,
    counted ``sidecar_errors``). A distinct type keeps the helper's return
    honest without overloading None (which means "no sidecar — submit plain")."""

    SETTLING = "settling"
    HELD = "held"


def _paired_sidecar(
    media_path: Path, *, now: float, settle_seconds: float, settings: Settings
) -> Sidecar | _SidecarHold | None:
    """Locate, settle-gate, read, and pre-validate ``media_path``'s sidecar.

    Returns the parsed :class:`Sidecar`, ``None`` when the media file has no
    sidecar (submit plain — a sidecar arriving later is deliberately too late),
    :data:`_SidecarHold.SETTLING` when the sidecar is still within its settle
    window (never ingest a half-written sidecar; the media waits with it), or
    :data:`_SidecarHold.HELD` after logging why the pair must wait for the
    operator.

    A sidecar-named ``domain_pack`` is pre-validated here so an unknown name is
    attributed to the SIDECAR (fix that file), while a resolution failure inside
    the submit path itself (a corrections collision, a broken folder mapping)
    keeps its existing config-error classification.
    """
    try:
        sidecar_path = find_sidecar(media_path)
        if sidecar_path is None:
            return None
        state = classify_settle(sidecar_path, now=now, settle_seconds=settle_seconds)
        if state is SettleState.TOO_FRESH:
            return _SidecarHold.SETTLING
        if state is SettleState.SKIP:
            raise SidecarError(
                f"{sidecar_path.name} disappeared or could not be read during the check"
            )
        parsed = read_sidecar(sidecar_path)
        if parsed.domain_pack is not None:
            try:
                resolve_domain_pack_by_name(parsed.domain_pack, settings)
            except DomainPackError as exc:
                raise SidecarError(f"{sidecar_path.name}: {exc}") from exc
    except SidecarError as exc:
        # Plain-language hold: the reason names the file and the fix; the media
        # file stays un-submitted and is retried every sweep until it's fixed.
        # Only key NAMES/reasons are logged, never sidecar values.
        logger.warning("watch sweep held %s: %s", media_path.name, exc)
        return _SidecarHold.HELD
    if parsed.ignored_keys:
        logger.info(
            "watch sweep: %s has keys Voxint doesn't recognize (kept for "
            "reference, not applied): %s",
            sidecar_path.name,
            # repr-escaped: unknown key NAMES are untrusted text, and a raw
            # newline/control char in one could forge extra log lines.
            ", ".join(repr(key) for key in parsed.ignored_keys),
        )
    return parsed


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
