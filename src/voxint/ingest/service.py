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
import os
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.pipeline.engine import submit
from voxint.pipeline.transitions import (
    RunSnapshot,
    StaleRevisionError,
    cas_update_run,
    snapshot,
)

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


def submit_media_item(session: Session, source_path: str) -> PipelineRun:
    """Create-or-reuse the MediaItem for ``source_path`` and queue a fresh run.

    DB-only: the caller owns the commit and, once it commits, lazily publishes
    ``voxint.run_pipeline`` (commit-before-publish). ``source_path`` is UNIQUE,
    so a repeated local path reuses its MediaItem while every submission still
    mints a distinct run.
    """
    media = _get_or_create_media(session, source_path)
    return submit(session, media.id)


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
    session: Session, media: MediaItem, *, size: int, sha256: str
) -> PipelineRun:
    """Resolve a same-``submission_id`` re-POST to its original run, or 409.

    The uuid-namespaced ``source_path`` already exists, so this is a form replay:
    identical bytes return the run created the first time (no duplicate run, no
    file rewrite); different bytes are a conflict we refuse. A stored MediaItem
    with no run is a partially-completed first attempt — heal it by submitting.
    """
    if media.size_bytes != size or media.sha256 != sha256:
        raise UploadConflictError(media.source_path)
    run = session.execute(
        select(PipelineRun)
        .where(PipelineRun.media_item_id == media.id)
        .order_by(PipelineRun.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return run if run is not None else submit(session, media.id)


def submit_upload(
    session: Session,
    *,
    stream: BinaryIO,
    filename: str,
    submission_id: str,
    media_root: Path,
    max_bytes: int,
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
            return _replay_run(session, winner, size=size, sha256=sha256)
        os.replace(temp_path, dest)  # atomic publish; only the insert winner is here
        published = True
        # Orphan-on-crash (deferred to Slice 5 recovery): if submit() or the
        # caller's commit fails after this replace, the row rolls back but the file
        # stays — a self-healing orphan, since a same-bytes retry re-inserts and
        # re-replaces identically. Durable staging state is Slice 5's job.
        return submit(session, media.id)
    finally:
        if not published:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()
