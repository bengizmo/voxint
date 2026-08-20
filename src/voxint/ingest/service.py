"""Shared submission + requeue service used by the CLI and the API.

Every function here operates on a live SQLAlchemy ``Session`` and **never**
imports Celery: the caller owns the commit boundary and lazily publishes
``voxint.run_pipeline`` *after* the transaction commits (commit-before-publish).
Keeping the broker out of this module is what lets the API's read path stay
Postgres-only and guarantees a broker outage can never leave a half-written
run — the durable QUEUED row exists before anything is enqueued.

:func:`submit_upload` additionally finalizes an uploaded file onto disk (bounded
stream → atomic ``os.replace``) before creating its row; that is filesystem I/O,
not the broker, so the module's broker-free contract is intact.

Failure modes surface as typed exceptions so each caller maps them to its own
surface (the CLI prints a message + exit 2; the API returns 404/409/413/422).
The two ``cas_update_run`` errors (:class:`StaleRevisionError`,
:class:`InvalidTransitionError`) propagate unchanged.
"""

import contextlib
import hashlib
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.app_settings import get_app_settings
from voxint.config import Settings, get_settings
from voxint.db.models import (
    AudioArtifact,
    AudioChunk,
    MediaItem,
    PipelineRun,
    RunStatus,
)
from voxint.domain_packs.base import union_pack_name_seeds
from voxint.domain_packs.corrections import union_pack_corrections
from voxint.domain_packs.registry import resolve_run_domain_pack
from voxint.ingest.sidecar import Sidecar
from voxint.media.netcheck import UrlPolicyError, parse_http_url
from voxint.pipeline.engine import submit
from voxint.pipeline.transitions import (
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
    snapshot,
)

logger = logging.getLogger(__name__)

# Copy the upload in bounded chunks so a huge body never lands in one buffer;
# the authoritative size cap is checked against the running total, not read().
_UPLOAD_CHUNK_BYTES = 1024 * 1024
# Control characters (incl. NUL) are never valid in a stored/served filename.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# ext4/xfs cap a single path component at 255 bytes; stay under it with margin
# for the ``.upload-XXXX.part`` sibling temp name mkstemp writes alongside it.
_MAX_FILENAME_BYTES = 200



class IngestError(Exception):
    """Base for submission/requeue failures a caller maps to a UI/exit code."""


class RunNotFoundError(IngestError):
    def __init__(self, run_id: uuid.UUID) -> None:
        super().__init__(f"no run {run_id}")
        self.run_id = run_id


class RunNotFailedError(IngestError):
    """Requeue attempted on a run that is not FAILED (only FAILED may requeue)."""

    def __init__(self, run_id: uuid.UUID, status: RunStatus) -> None:
        super().__init__(f"run is {status.value}, only failed runs can be requeued")
        self.run_id = run_id
        self.status = status


class MissingStageError(IngestError):
    """A FAILED run carrying no current_stage — the state machine was violated.

    A FAILED run always carries its failed stage; ``None`` means corruption, so
    we refuse to guess a stage rather than requeue into an arbitrary one.
    """

    def __init__(self, run_id: uuid.UUID) -> None:
        super().__init__(f"run {run_id} is FAILED with no current_stage; refusing to guess")
        self.run_id = run_id


class RunNotCancellableError(IngestError):
    """Cancel attempted on a run whose status forbids it (COMPLETED / FAILED).

    Only QUEUED, RUNNING, and AWAITING_ADJUDICATION are cancellable — the same
    states ``ALLOWED_TRANSITIONS`` permits ``→ CANCELLED`` from. FAILED is not
    terminal (it is requeueable), it is simply not *cancellable*; COMPLETED is
    done. An already-CANCELLED run is not an error — :func:`cancel_run` treats a
    repeat cancel as an idempotent no-op, not this exception.
    """

    def __init__(self, run_id: uuid.UUID, status: RunStatus) -> None:
        super().__init__(
            f"run is {status.value}, only queued/running/awaiting_adjudication "
            "runs can be cancelled"
        )
        self.run_id = run_id
        self.status = status


class RunNotArchivableError(IngestError):
    """Archive attempted on a run whose status forbids it (issue #5).

    Only terminal runs — COMPLETED, FAILED, CANCELLED — may be soft-archived: a
    QUEUED/RUNNING/AWAITING_ADJUDICATION run is still live, so archiving (hiding)
    it is ambiguous; the operator must cancel it first. Un-archiving is always
    allowed and never raises this. An already-archived run is not an error either
    — :func:`archive_run` treats a repeat archive as an idempotent no-op.
    """

    def __init__(self, run_id: uuid.UUID, status: RunStatus) -> None:
        super().__init__(
            f"run is {status.value}, only completed/failed/cancelled runs can be "
            "archived; cancel a live run first"
        )
        self.run_id = run_id
        self.status = status


class RunMediaNotDeletableError(IngestError):
    """Derived-media deletion attempted on a non-terminal run (issue #5).

    Deleting a run's derived audio files is destructive; it is refused while the
    run is live (QUEUED/RUNNING/AWAITING_ADJUDICATION) so files are never
    unlinked out from under a worker mid-pipeline. Same terminal set as archive.
    """

    def __init__(self, run_id: uuid.UUID, status: RunStatus) -> None:
        super().__init__(
            f"run is {status.value}, derived media can only be deleted for a "
            "completed/failed/cancelled run"
        )
        self.run_id = run_id
        self.status = status


class UploadValidationError(IngestError):
    """The upload's filename or submission id is unusable (maps to HTTP 422)."""


class UploadTooLargeError(IngestError):
    """The upload exceeded ``upload_max_bytes`` while streaming (maps to HTTP 413)."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"upload exceeds the maximum of {max_bytes} bytes")
        self.max_bytes = max_bytes


class UploadConflictError(IngestError):
    """A replayed submission id carries different bytes than the stored one (HTTP 409).

    The ``submission_id`` namespaces the upload path, so re-POSTing the same form
    is meant to be idempotent — but only if the bytes match. A same-id upload of
    *different* content would silently diverge from the run already created for
    that id, so it is refused rather than guessed at.
    """

    def __init__(self, source_path: str) -> None:
        super().__init__(f"submission already exists with different content: {source_path}")
        self.source_path = source_path


class UrlValidationError(IngestError):
    """A submitted ingest URL failed string-level validation (maps to HTTP 422).

    Raised by :func:`validate_ingest_url` for a URL that is not an absolute
    http/https URL with a plain hostname — bad scheme, missing host, embedded
    credentials, whitespace/control characters, over-length, or a non-public IP
    literal. The message is deliberately generic (never echoes the offending URL)
    so a validation error can't leak a signed/secret query string into logs.
    """


def _run_domain_pack_snapshot(
    session: Session,
    source_path: str | None,
    *,
    settings: Settings | None,
    domain_pack_name: str | None,
    extra_name_seeds: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Freeze the domain-pack snapshot for a NEW run (issue #11).

    Reads the per-folder mapping off the ``app_settings`` singleton and resolves
    it (or an explicit name, or the default pack) against the configured packs.
    ``source_path`` is ``None`` for uploads/URLs, which never sit under a watched
    folder and so take the default unless an explicit name is supplied. Raises
    :class:`~voxint.domain_packs.base.DomainPackError` on an unresolvable name —
    never a silent fallback.

    ``extra_name_seeds`` (issue #104) are sidecar-supplied speaker names, unioned
    onto the resolved pack's ``name_seeds`` after the corrections union — inside
    THIS freeze point, so every downstream reader of ``pipeline_runs.domain_pack``
    sees them with no changes.
    """
    resolved = settings or get_settings()
    row = get_app_settings(session)
    folder_map = dict(row.folder_domain_packs) if row is not None else {}
    pack_snapshot = resolve_run_domain_pack(
        source_path,
        settings=resolved,
        folder_domain_packs=folder_map,
        explicit_name=domain_pack_name,
    )
    # Union the operator's console-authored corrections (#84) onto the resolved
    # pack BEFORE the snapshot is frozen, so #82 compose and #83 provenance read
    # them off pipeline_runs.domain_pack unchanged. A collision (operator/pack
    # duplicate id or a non-idempotent union) raises DomainPackError here — the
    # same visible, never-silently-fall-back posture as an unresolvable pack name.
    if row is not None and row.corrections:
        pack_snapshot = union_pack_corrections(pack_snapshot, row.corrections)
    if extra_name_seeds:
        pack_snapshot = union_pack_name_seeds(pack_snapshot, extra_name_seeds)
    return pack_snapshot


def _effective_pack_name(
    domain_pack_name: str | None, sidecar: Sidecar | None
) -> str | None:
    """The explicit pack name for a submission (issue #104 precedence).

    A caller-supplied name wins over the sidecar's, which wins over the folder
    mapping and the default (the sidecar is the more specific intent than the
    folder it sits in). Explicit ``is not None`` tests — an explicit empty
    string still reaches the resolver and fails loudly, exactly as today.
    """
    if domain_pack_name is not None:
        return domain_pack_name
    if sidecar is not None and sidecar.domain_pack is not None:
        return sidecar.domain_pack
    return None


def submit_media_item(
    session: Session,
    source_path: str,
    *,
    settings: Settings | None = None,
    domain_pack_name: str | None = None,
    sidecar: Sidecar | None = None,
) -> PipelineRun:
    """Create-or-reuse the MediaItem for ``source_path`` and queue a fresh run.

    DB-only: the caller owns the commit and, once it commits, lazily publishes
    ``voxint.run_pipeline`` (commit-before-publish). ``source_path`` is UNIQUE,
    so a repeated local path reuses its MediaItem while every submission still
    mints a distinct run. The run's domain pack is frozen from the per-folder
    mapping (issue #11); ``domain_pack_name`` overrides it explicitly.

    ``sidecar`` (issue #104) applies the file's parsed YAML sidecar at THIS
    freeze point: speakers union into the pack snapshot's ``name_seeds``, notes
    seed ``operator_notes``, the whole mapping is stamped write-once onto the
    run, and the sidecar's ``domain_pack`` acts as the explicit name unless the
    caller passed one. Frozen at submit — editing the sidecar file afterwards
    changes nothing (the #84 posture).
    """
    domain_pack = _run_domain_pack_snapshot(
        session,
        source_path,
        settings=settings,
        domain_pack_name=_effective_pack_name(domain_pack_name, sidecar),
        extra_name_seeds=sidecar.speakers if sidecar is not None else (),
    )
    media = _get_or_create_media(session, source_path)
    return submit(
        session,
        media.id,
        domain_pack=domain_pack,
        sidecar=sidecar.raw if sidecar is not None else None,
        operator_notes=sidecar.notes if sidecar is not None else None,
    )


def submit_media_item_if_new(
    session: Session,
    source_path: str,
    *,
    settings: Settings | None = None,
    domain_pack_name: str | None = None,
    sidecar: Sidecar | None = None,
) -> PipelineRun | None:
    """Queue a run for ``source_path`` ONLY if no MediaItem claims it yet.

    Unlike :func:`submit_media_item` — which reuses an existing MediaItem and always
    mints a *fresh* run — this inserts the MediaItem and mints its run atomically,
    and returns ``None`` when ``source_path`` already exists. That makes it the safe
    primitive for the first-run wizard's "scan for existing media" confirm step: a
    double-clicked confirm, a re-scan, or two concurrent confirms cannot each queue
    another run for a file already ingested. The INSERT is contained in a SAVEPOINT
    so the losing racer's UNIQUE(source_path) conflict rolls back only that insert —
    not the caller's batch transaction — and is reported as ``None`` (skip), never an
    error (mirrors :func:`_get_or_create_media`).

    DB-only, like the rest of this module: the caller commits the whole batch once,
    then lazily publishes ``voxint.run_pipeline`` for each returned run
    (commit-before-publish).

    ``sidecar`` (issue #104) applies exactly as in :func:`submit_media_item`:
    frozen at THIS submit — a sidecar edited (or first appearing) after the
    media file is already known does nothing, by design.
    """
    domain_pack = _run_domain_pack_snapshot(
        session,
        source_path,
        settings=settings,
        domain_pack_name=_effective_pack_name(domain_pack_name, sidecar),
        extra_name_seeds=sidecar.speakers if sidecar is not None else (),
    )
    media = MediaItem(source_path=source_path)
    try:
        with session.begin_nested():
            session.add(media)
            session.flush()
    except IntegrityError:
        # Only a UNIQUE(source_path) conflict is the expected "already ingested"
        # skip (a prior submission, an earlier scan, or a concurrent confirm won the
        # race). Re-read to confirm the row now exists; re-raise anything else rather
        # than masking an unrelated integrity failure as a silent skip (mirrors
        # _get_or_create_media).
        existing = session.execute(
            select(MediaItem).where(MediaItem.source_path == source_path)
        ).scalar_one_or_none()
        if existing is None:
            raise
        return None
    return submit(
        session,
        media.id,
        domain_pack=domain_pack,
        sidecar=sidecar.raw if sidecar is not None else None,
        operator_notes=sidecar.notes if sidecar is not None else None,
    )


def _get_or_create_media(session: Session, source_path: str) -> MediaItem:
    """Return the MediaItem for ``source_path``, inserting it if absent.

    ``source_path`` is UNIQUE. Two callers can both observe no row and both try
    to insert; the loser's INSERT violates the constraint. Containing that INSERT
    in a SAVEPOINT means the conflict rolls back only the insert — not the
    caller's outer transaction — so we can re-read and adopt the row the winner
    committed. A concurrent submission thus still gets a MediaItem to mint its
    own run against, honouring "each submission mints a distinct run". (The API's
    uploads/URLs use uuid-namespaced paths that never collide; this guards the
    CLI's reuse-by-path and any future shared caller.)
    """
    existing = session.execute(
        select(MediaItem).where(MediaItem.source_path == source_path)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    media = MediaItem(source_path=source_path)
    try:
        # add() inside the savepoint so a rolled-back attempt is expunged and
        # cannot be re-INSERTed at the caller's later flush/commit.
        with session.begin_nested():
            session.add(media)
            session.flush()
    except IntegrityError:
        return session.execute(
            select(MediaItem).where(MediaItem.source_path == source_path)
        ).scalar_one()
    return media


def requeue_failed_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    expected_revision: int | None = None,
) -> RunSnapshot:
    """CAS-requeue a FAILED run at its failed stage, guarded by exact revision.

    Pass ``expected_revision`` to enforce exact-revision CAS from a caller that
    already knows the revision it means to act on (e.g. the API's requeue form):
    a mismatch raises :class:`StaleRevisionError` before any write, so a stale
    browser tab can never requeue a run that moved on. The CLI reads fresh and
    omits it — there is no gap to race within a single transaction.

    DB-only: the caller commits then lazily publishes ``voxint.run_pipeline``.
    Raises :class:`RunNotFoundError`, :class:`RunNotFailedError`,
    :class:`MissingStageError`, or — from the CAS — ``StaleRevisionError`` /
    ``InvalidTransitionError``, which callers map to their own responses.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    held = snapshot(run)
    if held.status is not RunStatus.FAILED:
        raise RunNotFailedError(run_id, held.status)
    if held.current_stage is None:
        raise MissingStageError(run_id)
    if expected_revision is not None and held.revision != expected_revision:
        raise StaleRevisionError(run_id, expected_revision)
    return cas_update_run(
        session,
        held,
        status=RunStatus.QUEUED,
        current_stage=held.current_stage,
    )


_CANCELLABLE_STATUSES = frozenset(
    {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.AWAITING_ADJUDICATION}
)


def cancel_run(
    session: Session,
    run_id: uuid.UUID,
    *,
    expected_revision: int | None = None,
) -> RunSnapshot:
    """CAS-cancel a live run (QUEUED / RUNNING / AWAITING_ADJUDICATION), guarded
    by exact revision.

    Cancellation is *cooperative* and pure DB state: it drives the existing
    ``→ CANCELLED`` transition and the caller commits — there is nothing to
    publish (unlike submit/requeue). A worker mid-run observes the cancel only at
    its next stage boundary or when its post-stage CAS loses to this one; the
    currently executing stage body still runs to completion first, so cancel is
    not an immediate process kill. A QUEUED run cancelled before dispatch simply
    never starts. The run keeps its ``current_stage`` so the console shows where
    it stopped (a fresh QUEUED run naturally carries ``None``).

    Idempotent: cancelling an already-CANCELLED run is a no-op that returns the
    current snapshot (a double-click / stale tab gets success, not a 409). Pass
    ``expected_revision`` for exact-revision CAS from a form: a mismatch on a
    still-live run raises :class:`StaleRevisionError` before any write.

    Raises :class:`RunNotFoundError`, :class:`RunNotCancellableError` (COMPLETED /
    FAILED), or — from the CAS — ``StaleRevisionError`` / ``InvalidTransitionError``.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    held = snapshot(run)
    if held.status is RunStatus.CANCELLED:
        return held  # idempotent: already cancelled, nothing to do
    if held.status not in _CANCELLABLE_STATUSES:
        raise RunNotCancellableError(run_id, held.status)
    if expected_revision is not None and held.revision != expected_revision:
        raise StaleRevisionError(run_id, expected_revision)
    try:
        return cas_update_run(
            session,
            held,
            status=RunStatus.CANCELLED,
            current_stage=held.current_stage,
        )
    except StaleRevisionError:
        # A CONCURRENT cancel won the CAS between our read and write (two racing
        # POSTs that both passed the revision guard on the same live revision).
        # Re-read: if the run is now CANCELLED, this is still an idempotent
        # success — a genuine double-click must not leave one operator with a
        # spurious 409. Any other stale outcome (the run advanced under us) is a
        # real stale-view conflict and re-raises. This is the concurrent twin of
        # the already-CANCELLED short-circuit above.
        session.expire(run)
        fresh = snapshot(run)
        if fresh.status is RunStatus.CANCELLED:
            return fresh
        raise


# Terminal statuses eligible for archive and derived-media deletion. FAILED is
# included: it is not cancellable, but it *is* a settled outcome an operator may
# want to hide or clear (it stays requeueable only until archived — see the
# archived-run guard on requeue).
_ARCHIVABLE_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


def archive_run(session: Session, run_id: uuid.UUID) -> PipelineRun:
    """Soft-archive a terminal run: hide it from ``/runs`` and the review queue
    while keeping every row (incl. the append-only ledger) intact.

    Archive is operator-visibility metadata — a single ``archived_at`` stamp,
    deliberately OUTSIDE the CAS revision and orthogonal to ``status`` (mirrors
    :func:`save_operator_notes`' last-write-wins handling). No revision bump, so a
    stale tab never 409s on it. **Idempotent**: archiving an already-archived run
    returns it unchanged. Only terminal runs (COMPLETED / FAILED / CANCELLED) may
    be archived — a live run raises :class:`RunNotArchivableError` (cancel first).

    DB-only: the caller owns the commit. Raises :class:`RunNotFoundError`.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    if run.archived_at is not None:
        return run  # idempotent: already archived, nothing to do
    status = RunStatus(run.status)
    if status not in _ARCHIVABLE_STATUSES:
        raise RunNotArchivableError(run_id, status)
    run.archived_at = datetime.now(UTC)
    return run


def unarchive_run(session: Session, run_id: uuid.UUID) -> PipelineRun:
    """Reverse :func:`archive_run` — clear ``archived_at`` so the run reappears.

    Always allowed (reversibility is the whole point of soft-archive) and
    **idempotent**: un-archiving a run that is not archived is a no-op success.
    Last-write-wins, no revision bump, DB-only. Raises :class:`RunNotFoundError`.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    run.archived_at = None  # idempotent even when already NULL
    return run


@dataclass(frozen=True)
class MediaDeletePlan:
    """Outcome of the DB half of :func:`delete_run_derived_media`.

    ``paths`` are the confined absolute filesystem paths the CALLER must unlink
    *after committing* (commit-before-side-effect: a rolled-back transaction must
    never leave the DB pointing at a file already deleted). Pass them to
    :func:`unlink_media_paths` once the session has committed.
    """

    rows_deleted: int
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class MediaUnlinkResult:
    """Result of the post-commit unlink pass. All counts are non-fatal — a
    partial filesystem failure never fails the operator's action."""

    files_deleted: int
    files_missing: int
    files_failed: int


def _confined_media_path(root: Path, rel_path: str) -> Path | None:
    """Resolve ``rel_path`` under ``root`` and return it only if it stays inside.

    Defense in depth: the stored ``AudioArtifact``/``AudioChunk`` paths are
    relative and trusted, but a malformed or absolute value must never let a
    delete escape ``MEDIA_ROOT``. ``resolve()`` also collapses any ``..`` and
    follows symlinks, so a link pointing outside the root is skipped, not
    followed. Returns ``None`` (skip) for anything that escapes.
    """
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        logger.warning("skipping media path outside MEDIA_ROOT: %r", rel_path)
        return None
    return candidate


def delete_run_derived_media(
    session: Session, run_id: uuid.UUID, *, media_root: Path
) -> MediaDeletePlan:
    """Delete a run's DERIVED audio (``AudioArtifact`` + ``AudioChunk``) rows and
    return their files for the caller to unlink after commit.

    Scope is deliberately narrow (issue #5, v1): only this run's own derived
    audio — the preprocessed 16 kHz wav and the per-chunk files. It **never**
    touches ``MediaItem``/``source_path`` (the original source is a single file
    shared by every run of that media item; deleting it would orphan a sibling
    run — that is a separate, refcount-guarded v2 action) and never touches the
    evidence ledger (adjudication / transcript / diarization rows stay intact).

    DB-only, mirroring the module's commit-before-side-effect contract: it
    deletes the rows in the session and returns the confined absolute paths in a
    :class:`MediaDeletePlan`; the caller commits, then calls
    :func:`unlink_media_paths`. Terminal-only — a live run raises
    :class:`RunMediaNotDeletableError`. Raises :class:`RunNotFoundError`.
    """
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise RunNotFoundError(run_id)
    status = RunStatus(run.status)
    if status not in _ARCHIVABLE_STATUSES:
        raise RunMediaNotDeletableError(run_id, status)

    root = media_root.resolve()
    rows: list[AudioArtifact | AudioChunk] = [
        *session.execute(
            select(AudioArtifact).where(AudioArtifact.pipeline_run_id == run_id)
        )
        .scalars()
        .all(),
        *session.execute(
            select(AudioChunk).where(AudioChunk.pipeline_run_id == run_id)
        )
        .scalars()
        .all(),
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for row in rows:
        confined = _confined_media_path(root, row.path)
        if confined is not None and confined not in seen:
            seen.add(confined)
            paths.append(confined)
        session.delete(row)
    return MediaDeletePlan(rows_deleted=len(rows), paths=tuple(paths))


def unlink_media_paths(paths: tuple[Path, ...]) -> MediaUnlinkResult:
    """Best-effort unlink of derived-media files AFTER the DB delete committed.

    Idempotent and non-fatal: an already-missing file is counted, not raised (a
    retry after a partial crash is clean); any other filesystem error is logged
    and counted as ``files_failed`` so the operator's action still succeeds — the
    DB row is already gone, and the orphaned file can be swept later. Never
    unlinks a directory (only file paths are stored); a directory or other
    ``OSError`` lands in ``files_failed``.
    """
    deleted = missing = failed = 0
    for path in paths:
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            missing += 1
        except OSError as exc:
            failed += 1
            logger.warning("failed to unlink derived-media file %s: %s", path, exc)
    return MediaUnlinkResult(
        files_deleted=deleted, files_missing=missing, files_failed=failed
    )


def sanitize_upload_filename(filename: str) -> str:
    """Reduce a client-supplied filename to a safe bare basename, or reject it.

    Rejects empty, ``.``/``..``, either slash, control characters (incl. NUL),
    and names too long for one path component. Anything surviving is a plain
    basename with no traversal potential — the uuid-namespaced parent dir plus a
    final containment check are the second and third lines of defence.
    """
    name = (filename or "").strip()
    if not name:
        raise UploadValidationError("upload filename is empty")
    if "/" in name or "\\" in name:
        raise UploadValidationError("upload filename must not contain path separators")
    if name in (".", ".."):
        raise UploadValidationError("upload filename must not be '.' or '..'")
    if _CONTROL_CHARS.search(name):
        raise UploadValidationError("upload filename contains control characters")
    if len(name.encode("utf-8")) > _MAX_FILENAME_BYTES:
        raise UploadValidationError(
            f"upload filename exceeds {_MAX_FILENAME_BYTES} bytes"
        )
    # os.path.basename is a no-op once slashes are rejected; keep it as a belt so
    # a future relaxation of the slash rule can never let a directory through.
    if os.path.basename(name) != name:
        raise UploadValidationError("upload filename must be a bare basename")
    return name


def validate_ingest_url(url: str) -> str:
    """Validate an ingest URL at the string level, returning it whitespace-trimmed.

    A thin wrapper over the shared string-level gate
    :func:`voxint.media.netcheck.parse_http_url` — the SINGLE policy for every
    outbound-fetch capability (URL ingestion and web research), so the rules can
    never diverge between consumers. This is the FIRST SSRF guard and gates row
    creation; the authoritative "resolves to a public address" check is re-done
    worker-side at download time (slice 6g). Errors are re-raised as
    :class:`UrlValidationError` with the historical ``"ingest "`` message prefix
    preserved byte-for-byte; messages never echo the URL, so a signed/secret
    query string cannot leak into a 422 body or logs.
    """
    try:
        return parse_http_url(url).url
    except UrlPolicyError as exc:
        raise UrlValidationError(f"ingest {exc}") from None


def _submission_dir(submission_id: str) -> str:
    """Validate the client's hidden ``submission_id`` and return its hex form.

    It namespaces the upload path, so it must be a real UUID: a client-chosen
    string could otherwise carry separators or collide with an unrelated path.
    """
    try:
        return uuid.UUID(submission_id).hex
    except (ValueError, AttributeError, TypeError) as exc:
        raise UploadValidationError("submission_id is not a valid UUID") from exc


def _stream_to_temp(dest_dir: Path, stream: BinaryIO, max_bytes: int) -> tuple[Path, int, str]:
    """Copy ``stream`` into a temp file beside its final home, bounded + hashed.

    Returns ``(temp_path, size_bytes, sha256_hex)``. The temp lives in
    ``dest_dir`` (the eventual parent) so the later publish is an atomic
    same-filesystem ``os.replace``. The size cap is enforced against the running
    total — the moment it is crossed we stop, unlink, and raise, so a lying
    Content-Length cannot write past the bound. Any failure cleans the temp.
    """
    fd, temp_name = tempfile.mkstemp(dir=dest_dir, prefix=".upload-", suffix=".part")
    temp_path = Path(temp_name)
    size = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as temp:
            while True:
                chunk = stream.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError(max_bytes)
                digest.update(chunk)
                temp.write(chunk)
            temp.flush()
            os.fsync(temp.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise
    return temp_path, size, digest.hexdigest()


def _replay_run(
    session: Session, media: MediaItem, *, size: int, sha256: str, domain_pack: dict[str, Any]
) -> PipelineRun:
    """Resolve a same-``submission_id`` re-POST to its original run, or 409.

    The uuid-namespaced ``source_path`` already exists, so this is a form replay:
    identical bytes return the run created the first time (no duplicate run, no
    file rewrite); different bytes are a conflict we refuse. A stored MediaItem
    with no run is a partially-completed first attempt — heal it by submitting
    with ``domain_pack`` (the default snapshot resolved by the caller).
    """
    if media.size_bytes != size or media.sha256 != sha256:
        raise UploadConflictError(media.source_path)
    run = session.execute(
        select(PipelineRun)
        .where(PipelineRun.media_item_id == media.id)
        .order_by(PipelineRun.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return run if run is not None else submit(session, media.id, domain_pack=domain_pack)


def submit_upload(
    session: Session,
    *,
    stream: BinaryIO,
    filename: str,
    submission_id: str,
    media_root: Path,
    max_bytes: int,
    settings: Settings | None = None,
    domain_pack_name: str | None = None,
) -> PipelineRun:
    """Finalize a browser upload into an immutable MediaItem and queue a run.

    The bytes land at ``incoming/{submission_id}/{safe_name}`` — the server-issued
    ``submission_id`` namespaces the path so re-uploading a name never overwrites
    history, and a re-POST of the *same* form (same id, same bytes) is idempotent
    (see :func:`_replay_run`). The file is streamed to a temp in the final dir,
    size-capped and hashed, then atomically ``os.replace``\\d into place; the
    MediaItem records the first ``sha256``/``size_bytes`` the schema ever stores.

    Broker-free: the caller commits, then lazily publishes ``voxint.run_pipeline``
    (commit-before-publish). Raises :class:`UploadValidationError` (bad name/id),
    :class:`UploadTooLargeError` (over the cap), or :class:`UploadConflictError`
    (replayed id, different bytes).
    """
    # Uploads never sit under a watched folder (uuid-namespaced path), so the pack
    # is the default unless an explicit name is supplied (source_path=None).
    domain_pack = _run_domain_pack_snapshot(
        session, None, settings=settings, domain_pack_name=domain_pack_name
    )
    safe_name = sanitize_upload_filename(filename)
    sub = _submission_dir(submission_id)
    # The idempotency key is the full source_path (submission_id + safe_name), per
    # the plan's incoming/{uuid}/{safe_name} scheme: a true form replay (same id,
    # same file) is idempotent. Reusing a stale submission_id with a DIFFERENT
    # filename mints a separate run — harmless, but looser than keying on the id
    # alone, which would need a UNIQUE submission_id column (the 0005 migration in
    # Slice 6). Deliberately deferred.
    rel = str(PurePosixPath("incoming") / sub / safe_name)

    root = media_root.resolve()
    dest = (root / "incoming" / sub / safe_name).resolve()
    if not dest.is_relative_to(root):  # defence-in-depth; validated inputs can't escape
        raise UploadValidationError("upload path escapes the media root")
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_path, size, sha256 = _stream_to_temp(dest.parent, stream, max_bytes)
    published = False
    try:
        # Claim the source_path row BEFORE publishing the file. The filesystem
        # write is gated on winning source_path's UNIQUE insert, so a competing
        # upload for the same submission_id — a sequential re-POST or a genuinely
        # concurrent one — can never os.replace over the winner's bytes and leave
        # the committed sha256/size describing a file that is no longer there. A
        # pre-existing row and a lost race BOTH surface here as IntegrityError and
        # route through _replay_run without ever touching dest. (SAVEPOINT so the
        # conflict rolls back only the insert, not the caller's transaction —
        # mirrors _get_or_create_media.)
        try:
            with session.begin_nested():
                media = MediaItem(source_path=rel, size_bytes=size, sha256=sha256)
                session.add(media)
                session.flush()
        except IntegrityError:
            winner = session.execute(
                select(MediaItem).where(MediaItem.source_path == rel)
            ).scalar_one()
            return _replay_run(
                session, winner, size=size, sha256=sha256, domain_pack=domain_pack
            )
        os.replace(temp_path, dest)  # atomic publish; only the insert winner is here
        published = True
        # Orphan-on-crash (deferred to Slice 5 recovery): if submit() or the
        # caller's commit fails after this replace, the row rolls back but the file
        # stays — a self-healing orphan, since a same-bytes retry re-inserts and
        # re-replaces identically. Durable staging state is Slice 5's job.
        return submit(session, media.id, domain_pack=domain_pack)
    finally:
        if not published:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def _replay_url_run(
    session: Session, media: MediaItem, *, source_url: str, domain_pack: dict[str, Any]
) -> PipelineRun:
    """Resolve a same-``submission_id`` URL re-POST to its original run, or 409.

    The uuid-namespaced ``source_path`` (``incoming/{uuid}/source``) already
    exists, so this is a form replay. The first submission's URL wins: a replay
    carrying the SAME url returns the run created the first time (idempotent, no
    duplicate run); a replay carrying a DIFFERENT url is refused, because
    silently acquiring a URL the operator never pasted is the same divergence the
    upload path refuses on mismatched bytes (hence the shared
    :class:`UploadConflictError`, keyed on the stable ``source_path``). A stored
    MediaItem with no run is a partially-completed first attempt — heal it by
    submitting.
    """
    if media.source_url != source_url:
        raise UploadConflictError(media.source_path)
    run = session.execute(
        select(PipelineRun)
        .where(PipelineRun.media_item_id == media.id)
        .order_by(PipelineRun.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return run if run is not None else submit(session, media.id, domain_pack=domain_pack)


def submit_url(
    session: Session,
    *,
    url: str,
    submission_id: str,
    settings: Settings | None = None,
    domain_pack_name: str | None = None,
) -> PipelineRun:
    """Register a URL for acquisition as an immutable MediaItem and queue a run.

    The origin ``url`` is validated (:func:`validate_ingest_url`) and stored as
    ``MediaItem.source_url``; ``source_path`` is PRE-ASSIGNED to the uuid-
    namespaced ``incoming/{submission_id}/source`` — unique by construction, so
    no schema relaxation is needed and the file the worker's ACQUIRE stage
    downloads (slice 6c) lands there. No filesystem write happens here: unlike an
    upload, the bytes do not exist yet, so ``source_path`` is an *intended*
    location until ACQUIRE materializes it.

    The server-issued ``submission_id`` makes a form re-POST idempotent: the same
    id + same url returns the run created the first time; the same id + a
    DIFFERENT url is a conflict (see :func:`_replay_url_run`). **This module never
    invokes yt-dlp — only the worker's ACQUIRE stage does.**

    Broker-free: the caller commits, then lazily publishes ``voxint.run_pipeline``
    (commit-before-publish). Raises :class:`UrlValidationError` (bad url) or
    :class:`UploadValidationError` (bad submission id) — both HTTP 422 — or
    :class:`UploadConflictError` (replayed id, different url) — HTTP 409.
    """
    # URLs never sit under a watched folder (uuid-namespaced path), so the pack is
    # the default unless an explicit name is supplied (source_path=None).
    domain_pack = _run_domain_pack_snapshot(
        session, None, settings=settings, domain_pack_name=domain_pack_name
    )
    validated_url = validate_ingest_url(url)
    sub = _submission_dir(submission_id)
    rel = str(PurePosixPath("incoming") / sub / "source")

    # Claim the uuid-namespaced source_path row; a re-POST of the same
    # submission_id collides on the UNIQUE constraint and routes through
    # _replay_url_run. (SAVEPOINT so the conflict rolls back only the insert, not
    # the caller's transaction — mirrors _get_or_create_media / submit_upload.)
    try:
        with session.begin_nested():
            media = MediaItem(source_path=rel, source_url=validated_url)
            session.add(media)
            session.flush()
    except IntegrityError:
        winner = session.execute(
            select(MediaItem).where(MediaItem.source_path == rel)
        ).scalar_one()
        return _replay_url_run(
            session, winner, source_url=validated_url, domain_pack=domain_pack
        )
    return submit(session, media.id, domain_pack=domain_pack)
