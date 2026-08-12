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

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from voxint.db.models import MediaItem, PipelineRun
from voxint.media.ytdlp import AcquisitionError
from voxint.pipeline.stages.context import StageContext, StageDataError

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
    # re-downloading.
    if dest.is_file() and dest.stat().st_size > 0:
        if media.sha256 is None or media.size_bytes is None:
            media.size_bytes = dest.stat().st_size
            media.sha256 = _sha256(dest)
        return

    if ctx.downloader is None:
        raise StageDataError(
            f"run {run_id}: no downloader configured for URL acquisition"
        )

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
        # Publish atomically and ONLY if source_path is still absent. os.link raises
        # FileExistsError rather than overwriting, so a superseded ("zombie")
        # attempt whose lease already expired can never clobber the bytes a live
        # attempt published — the published file is immutable once written. The
        # committing attempt then records the hash of the ACTUAL bytes on disk, so
        # the row can never describe different bytes than source_path. (os.replace
        # would unconditionally overwrite, reopening that race — see the dual-review
        # zombie-overwrite finding.) The hash is taken AFTER the link: dest is now
        # fixed, so a crash before commit is repaired identically by the replay path.
        try:
            os.link(produced, dest)
        except FileExistsError:
            # Another attempt won the publish; adopt its file as authoritative.
            media.size_bytes = dest.stat().st_size
            media.sha256 = _sha256(dest)
        else:
            # produced and dest are the same inode ⇒ this hash matches disk exactly.
            media.size_bytes = size
            media.sha256 = _sha256(produced)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _single_output(tmp_dir: Path) -> Path:
    """Return the one produced file, or reject missing/multiple/empty output.

    ACQUIRE must publish exactly one source file. A playlist or format-merge that
    left several files, an extractor that wrote none, or a zero-byte artifact are
    all terminal acquisition failures — never a silent pick-the-first. Symlinks are
    excluded (``is_file()`` follows them): the download comes from an untrusted
    remote, so ACQUIRE only ever publishes a genuine regular file, never a link
    whose target could point outside the temp dir (and be deleted by cleanup).
    """
    files = [
        entry
        for entry in tmp_dir.iterdir()
        if entry.is_file() and not entry.is_symlink()
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
