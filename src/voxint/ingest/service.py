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
import ipaddress
import os
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun, RunStatus
from voxint.media.netcheck import ip_is_public
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

# yt-dlp URL ingestion is an http(s)-only capability; anything else (file:, data:,
# ftp:, a bare scheme-relative //host) is rejected at the string level.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
# A pasted URL that runs to kilobytes is almost certainly hostile or malformed;
# 2048 is the de-facto interoperable URL length ceiling.
_MAX_URL_BYTES = 2048
# A well-formed URL carries no raw whitespace (spaces/tabs/newlines must be
# percent-encoded); an unencoded whitespace char is a splitting/smuggling smell.
_URL_WHITESPACE = re.compile(r"\s")


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


class UrlValidationError(IngestError):
    """A submitted ingest URL failed string-level validation (maps to HTTP 422).

    Raised by :func:`validate_ingest_url` for a URL that is not an absolute
    http/https URL with a plain hostname — bad scheme, missing host, embedded
    credentials, whitespace/control characters, over-length, or a non-public IP
    literal. The message is deliberately generic (never echoes the offending URL)
    so a validation error can't leak a signed/secret query string into logs.
    """


def submit_media_item(session: Session, source_path: str) -> PipelineRun:
    """Create-or-reuse the MediaItem for ``source_path`` and queue a fresh run.

    DB-only: the caller owns the commit and, once it commits, lazily publishes
    ``voxint.run_pipeline`` (commit-before-publish). ``source_path`` is UNIQUE,
    so a repeated local path reuses its MediaItem while every submission still
    mints a distinct run.
    """
    media = _get_or_create_media(session, source_path)
    return submit(session, media.id)


def submit_media_item_if_new(session: Session, source_path: str) -> PipelineRun | None:
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
    """
    media = MediaItem(source_path=source_path)
    try:
        with session.begin_nested():
            session.add(media)
            session.flush()
    except IntegrityError:
        # Already ingested (a prior submission, an earlier scan, or a concurrent
        # confirm won the race) — skip rather than mint a duplicate run.
        return None
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


def validate_ingest_url(url: str) -> str:
    """Validate an ingest URL at the string level, returning it whitespace-trimmed.

    Enforces the shape yt-dlp is permitted to fetch: an absolute http/https URL
    with a plain hostname, no embedded credentials, no whitespace/control
    characters, under the length ceiling, and — when the host is an IP literal —
    a globally routable address (loopback/private/link-local/reserved/multicast
    literals are refused). This is the FIRST SSRF guard and gates row creation.

    It deliberately does **not** resolve DNS: a name that looks public now can
    rebind before the worker fetches it, so the authoritative "resolves to a
    public address" check is re-done worker-side at download time (slice 6g).
    A DNS *name* therefore passes here (only ``localhost`` is refused by name);
    an IP *literal* is checked now because it needs no resolution — including the
    IPv4-in-IPv6 embeddings (deprecated ``::a.b.c.d``, NAT64) and site-local that
    ``is_global`` alone mis-classifies, which the shared :func:`ip_is_public`
    unwraps/rejects here just as the worker gate does. Error messages never echo
    the URL, so a signed/secret query string cannot leak into a 422 body or logs.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise UrlValidationError("ingest URL is empty")
    try:
        url_bytes = len(candidate.encode("utf-8"))
    except UnicodeEncodeError as exc:  # e.g. an unpaired surrogate — uphold the typed contract
        raise UrlValidationError("ingest URL is not valid UTF-8") from exc
    if url_bytes > _MAX_URL_BYTES:
        raise UrlValidationError(f"ingest URL exceeds {_MAX_URL_BYTES} bytes")
    if _URL_WHITESPACE.search(candidate) or _CONTROL_CHARS.search(candidate):
        raise UrlValidationError("ingest URL contains whitespace or control characters")
    if "\\" in candidate:
        # urlsplit keeps backslashes in the authority, but browsers/yt-dlp may
        # treat "\" as "/" — a parser split we refuse rather than try to model.
        raise UrlValidationError("ingest URL must not contain a backslash")
    try:
        parts = urlsplit(candidate)
        # .port is lazily parsed; touch it so a bad port (":abc"/out-of-range)
        # surfaces here rather than as an obscure failure deeper in the worker.
        _ = parts.port
    except ValueError as exc:  # malformed IPv6 literal, un-castable/out-of-range port
        raise UrlValidationError("ingest URL is malformed") from exc
    if parts.scheme not in _ALLOWED_URL_SCHEMES:
        raise UrlValidationError("ingest URL must be an absolute http/https URL")
    if parts.username is not None or parts.password is not None:
        raise UrlValidationError("ingest URL must not embed credentials")
    host = parts.hostname
    if not host:
        raise UrlValidationError("ingest URL has no host")
    # A trailing DNS root dot ("localhost.", "127.0.0.1.") resolves identically to
    # the un-dotted form, so strip it before the policy checks — otherwise a lone
    # dot side-steps both the localhost denylist and the IP-literal parse.
    host = host.rstrip(".")
    if not host:
        raise UrlValidationError("ingest URL has no host")
    bracketed = "[" in parts.netloc  # the authority was an IPv6/IPvFuture literal
    if host == "localhost" or host.endswith(".localhost"):
        # urlsplit lowercases .hostname, so a plain case check is exhaustive.
        raise UrlValidationError("ingest URL host is not permitted")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if bracketed:
            # A bracketed authority that is not a valid IPv6 literal (e.g. an
            # IPvFuture "[v1.foo]") is ambiguous across parsers — refuse it rather
            # than let it fall through to the DNS-name branch.
            raise UrlValidationError("ingest URL has an invalid IPv6 host") from None
        return candidate  # a DNS name — public-address check deferred to the worker (6g)
    if not ip_is_public(ip):
        # Loopback/private/link-local/reserved/unspecified/multicast literals, plus
        # site-local and the IPv4-in-IPv6 embeddings (::a.b.c.d, NAT64) that
        # is_global alone mis-judges. ip_is_public (media.netcheck) is the SINGLE
        # per-address policy shared with the worker's resolved-host gate, so the
        # literal check here and the DNS re-resolution there can never diverge on
        # what "public" means.
        raise UrlValidationError("ingest URL host is not permitted")
    return candidate


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


def _replay_url_run(
    session: Session, media: MediaItem, *, source_url: str
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
    return run if run is not None else submit(session, media.id)


def submit_url(
    session: Session,
    *,
    url: str,
    submission_id: str,
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
        return _replay_url_run(session, winner, source_url=validated_url)
    return submit(session, media.id)
