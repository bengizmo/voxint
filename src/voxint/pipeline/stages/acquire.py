"""Acquire stage: the universal first stage that materializes the source file.

For local or uploaded media (``source_url IS NULL``) the file already sits at
``source_path``, so this stage is a **no-op success** — it exists only so every
run starts at ``STAGE_ORDER[0]`` and the transition machine stays uniform.

For URL runs (``source_url`` set) it downloads the media with yt-dlp into an
attempt-unique temp dir, requires **exactly one non-empty file**, enforces the
authoritative size cap, hashes it, and **atomically ``os.replace``\\s** it onto
``source_path`` — after which it records ``sha256``/``size_bytes`` on the
MediaItem. PREPARE (the next stage) stays the sole containment/decodability
gate, so URL and local inputs validate identically; ACQUIRE never probes.

Idempotency (the engine runs stages at-least-once): if ``source_path`` already
holds a non-empty file, a prior attempt already published it — a crash between
the atomic rename and the DB commit is recovered here by (re)populating
``sha256``/``size_bytes`` from the finalized file WITHOUT re-downloading. The
uuid-namespaced ``source_path`` is written only by this stage's atomic rename,
so its mere presence is proof of a complete prior download.

All terminal download failures raise :class:`AcquisitionError` (deterministic,
not a ``ServiceError``), so the run parks FAILED @ acquire for a manual Requeue
instead of being auto-retried — a datacenter-IP bot-block is expected and must
fail cleanly, not loop.
"""

import contextlib
import hashlib
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, MediaSourceMetadata, PipelineRun, SourceKind
from voxint.media.netcheck import HostNotPublicError, assert_host_resolves_public
from voxint.media.source_metadata import (
    SourceMetadataError,
    extract,
    load_sidecar,
    sidecar_filename,
    to_sidecar_bytes,
)
from voxint.media.ytdlp import INFO_JSON_FILENAME, AcquisitionError
from voxint.pipeline.stages.context import StageContext, StageDataError

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1024 * 1024


def run(ctx: StageContext, session: Session, run_id: uuid.UUID) -> None:
    pipeline_run = session.get(PipelineRun, run_id)
    if pipeline_run is None:
        raise StageDataError(f"no pipeline run {run_id}")
    media = session.get(MediaItem, pipeline_run.media_item_id)
    if media is None:
        raise StageDataError(f"run {run_id}: media item missing")

    if media.source_url is None:
        # Local/uploaded media: the bytes are already at source_path. PREPARE is
        # the sole containment/decodability gate, so there is nothing to do here.
        return

    media_root = ctx.media_root.resolve()
    dest = (ctx.media_root / media.source_path).resolve()
    if not dest.is_relative_to(media_root):
        # Defence-in-depth: source_path is server-assigned (incoming/{uuid}/source)
        # and cannot escape, but never write outside the media root regardless.
        raise StageDataError(
            f"source {media.source_path!r} escapes media root {media_root}"
        )

    # Idempotent replay: a non-empty file at source_path is a completed prior
    # download (only the atomic rename below ever writes there). Re-populate the
    # row if a crash landed between rename and commit, then succeed without
    # re-downloading. Metadata is repaired the same way: the hash-addressed
    # sidecar published beside the media (sidecar-before-media ordering below)
    # lets the replay re-insert a row the crash lost; absent both row and
    # sidecar, metadata is simply absent — capture is best-effort by design.
    if dest.is_file() and dest.stat().st_size > 0:
        if media.sha256 is None or media.size_bytes is None:
            media.size_bytes = dest.stat().st_size
            media.sha256 = _sha256(dest)
        _ensure_source_metadata_row(session, media, dest.parent)
        return

    if ctx.downloader is None:
        raise StageDataError(
            f"run {run_id}: no downloader configured for URL acquisition"
        )

    # Authoritative SSRF gate (slice 6g): re-resolve the host NOW and refuse it if
    # any address is non-public. validate_ingest_url gated row creation at submit
    # but did NOT resolve DNS — a name that looked public then can rebind before
    # this download. Re-parse source_url here rather than trusting a stored parse.
    # On refusal raise AcquisitionError (deterministic → FAILED @ acquire → manual
    # Requeue) with a host-only message; `from None` keeps the URL out of any
    # chained traceback (the HostNotPublicError names only the host, but stay
    # uniform with the module's born-clean discipline).
    host = urlsplit(media.source_url).hostname
    if not host:  # defensive: a stored source_url always has a host (validated)
        raise AcquisitionError("URL acquisition source has no host")
    try:
        assert_host_resolves_public(host.rstrip("."), resolver=ctx.resolver)
    except HostNotPublicError as exc:
        raise AcquisitionError(str(exc)) from None

    canonical_dir = dest.parent
    canonical_dir.mkdir(parents=True, exist_ok=True)
    # Attempt-unique so two overlapping attempts (a lease-expiry edge) never share
    # a temp dir; a crashed attempt's litter is bounded and inert.
    tmp_dir = canonical_dir / f".acquire-{uuid.uuid4().hex}.tmp"
    tmp_dir.mkdir()
    try:
        ctx.downloader(media.source_url, tmp_dir, ctx.ytdlp_max_bytes)
        produced = _single_output(tmp_dir)
        size = produced.stat().st_size
        if size > ctx.ytdlp_max_bytes:
            # Authoritative check: --max-filesize is an early hint yt-dlp does not
            # always honour (e.g. size unknown until fully fetched).
            raise AcquisitionError(
                f"downloaded {size} bytes exceeds the {ctx.ytdlp_max_bytes}-byte limit"
            )
        # The produced file is complete, so its hash is fixed now — computed
        # before the publish because the metadata sidecar's name embeds it.
        produced_sha256 = _sha256(produced)
        # Metadata sidecar (issue #36): sanitize the raw info-JSON into a
        # hash-addressed sidecar and publish it BEFORE the media file. The
        # media file's presence is the replay marker, so sidecar-before-media
        # guarantees: media published ⇒ its sidecar (if metadata was captured)
        # is already beside it, closing the crash-between-publish-and-commit
        # window without a re-download. The raw info-JSON itself never leaves
        # tmp_dir — it carries signed URLs and dies with the attempt dir.
        _publish_metadata_sidecar(ctx, tmp_dir, canonical_dir, produced_sha256)
        # Publish atomically and ONLY if source_path is still absent. os.link raises
        # FileExistsError rather than overwriting, so a superseded ("zombie")
        # attempt whose lease already expired can never clobber the bytes a live
        # attempt published — the published file is immutable once written. The
        # committing attempt then records the hash of the ACTUAL bytes on disk, so
        # the row can never describe different bytes than source_path. (os.replace
        # would unconditionally overwrite, reopening that race — see the dual-review
        # zombie-overwrite finding.)
        try:
            os.link(produced, dest)
        except FileExistsError:
            # Another attempt won the publish; adopt its file as authoritative.
            # The hash-addressed sidecar lookup below then only ever loads
            # metadata describing the WINNER's bytes — this attempt's sidecar
            # (if its bytes differed) is inert under a different name.
            media.size_bytes = dest.stat().st_size
            media.sha256 = _sha256(dest)
        else:
            # produced and dest are the same inode ⇒ this hash matches disk exactly.
            media.size_bytes = size
            media.sha256 = produced_sha256
        _ensure_source_metadata_row(session, media, canonical_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _single_output(tmp_dir: Path) -> Path:
    """Return the one produced media file, or reject missing/multiple/empty output.

    ACQUIRE must publish exactly one source file. A playlist or format-merge that
    left several files, an extractor that wrote none, or a zero-byte artifact are
    all terminal acquisition failures — never a silent pick-the-first. Symlinks are
    excluded (``is_file()`` follows them): the download comes from an untrusted
    remote, so ACQUIRE only ever publishes a genuine regular file, never a link
    whose target could point outside the temp dir (and be deleted by cleanup).
    The pinned metadata sidecar (``source.info.json``, issue #36) is the one
    expected non-media output, matched by exact name and excluded from the
    count — the invariant on *media* files is unchanged.
    """
    files = [
        entry
        for entry in tmp_dir.iterdir()
        if entry.is_file()
        and not entry.is_symlink()
        and entry.name != INFO_JSON_FILENAME
    ]
    if len(files) != 1:
        raise AcquisitionError(
            f"expected exactly one downloaded file, found {len(files)}"
        )
    produced = files[0]
    if produced.stat().st_size == 0:
        raise AcquisitionError("downloaded file is empty")
    return produced


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_metadata_sidecar(
    ctx: StageContext, tmp_dir: Path, canonical_dir: Path, media_sha256: str
) -> None:
    """Sanitize ``tmp_dir``'s info-JSON into a published hash-addressed sidecar.

    Best-effort by design (issue #36 decision): metadata is context, not
    identity, so a missing, oversized, or unparseable info-JSON logs a warning
    and captures nothing — it never fails an otherwise-valid acquisition. The
    sidecar is linked with the same no-clobber ``os.link`` discipline as the
    media file; a FileExistsError means an attempt with identical bytes
    already published identical metadata, which is adopted as-is.
    """
    info_path = tmp_dir / INFO_JSON_FILENAME
    if not info_path.is_file() or info_path.is_symlink():
        logger.warning(
            "acquisition produced no %s; source metadata not captured",
            INFO_JSON_FILENAME,
        )
        return
    try:
        meta = extract(info_path.read_bytes(), extra_secrets=ctx.metadata_secrets)
    except (SourceMetadataError, OSError) as exc:
        logger.warning("source metadata not captured: %s", exc)
        return
    sidecar_tmp = tmp_dir / sidecar_filename(media_sha256)
    sidecar_tmp.write_bytes(
        to_sidecar_bytes(
            meta, media_sha256=media_sha256, acquired_at=datetime.now(UTC)
        )
    )
    # Same bytes ⇒ same sidecar name: FileExistsError means an earlier attempt
    # already published metadata for exactly these bytes; it is authoritative.
    with contextlib.suppress(FileExistsError):
        os.link(sidecar_tmp, canonical_dir / sidecar_tmp.name)


def _ensure_source_metadata_row(
    session: Session, media: MediaItem, canonical_dir: Path
) -> None:
    """Insert the write-once ``media_source_metadata`` row from the sidecar.

    Idempotent and race-safe: nothing to do when a row already exists (rows
    are never updated — a snapshot must not rewrite the context an
    adjudication was made against); the unique ``media_item_id`` constraint
    arbitrates concurrent inserts via a SAVEPOINT, mirroring ``submit_url``'s
    IntegrityError pattern. Only the sidecar matching the authoritative file's
    hash is ever loaded, so metadata can never describe different bytes than
    ``source_path``. Absent row + absent/corrupt sidecar ⇒ metadata stays
    absent (legacy media, or capture legitimately failed) — never an error.
    """
    if media.sha256 is None:
        return
    existing = session.execute(
        select(MediaSourceMetadata.id).where(
            MediaSourceMetadata.media_item_id == media.id
        )
    ).scalar_one_or_none()
    if existing is not None:
        return
    sidecar_path = canonical_dir / sidecar_filename(media.sha256)
    if not sidecar_path.is_file():
        return
    try:
        meta, acquired_at = load_sidecar(
            sidecar_path.read_bytes(), expected_media_sha256=media.sha256
        )
    except (SourceMetadataError, OSError) as exc:
        logger.warning("source metadata sidecar unusable: %s", exc)
        return
    row = MediaSourceMetadata(
        media_item_id=media.id,
        source_kind=SourceKind.YTDLP.value,
        title=meta.title,
        uploader=meta.uploader,
        uploader_url=meta.uploader_url,
        channel=meta.channel,
        channel_url=meta.channel_url,
        description=meta.description,
        upload_date=meta.upload_date,
        duration_seconds=meta.duration_seconds,
        tags=list(meta.tags),
        canonical_url=meta.canonical_url,
        extractor=meta.extractor,
        extractor_version=meta.extractor_version,
        raw=meta.raw or None,
        raw_schema_version=meta.raw_schema_version,
        acquired_at=acquired_at,
    )
    try:
        with session.begin_nested():
            session.add(row)
    except IntegrityError:
        # A concurrent attempt inserted first; its identical-bytes row stands.
        pass
