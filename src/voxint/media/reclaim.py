"""Media retention / garbage collection core (issue #15).

File reclamation only: unlink the large normalized-audio intermediate
(``artifacts/{run_id}/normalized.wav``) for OLD TERMINAL runs and stamp the
``audio_artifacts`` row with ``reclaimed_at`` / ``reclaimed_bytes``. The row is
never deleted (audit trail); the source media, transcript, diarization, and the
immutable adjudication ledger are ALWAYS kept, so a reclaimed run stays
re-processable from its source. Never touches run rows or append-only tables.

Pure and Celery-free so it is testable without a broker. Correctness notes:

- **Concurrency**: eligible rows are claimed with ``FOR UPDATE ... SKIP
  LOCKED`` and reclaimed inside one transaction, so overlapping sweeps (or a
  duplicate Celery delivery) never double-count or clobber a byte measurement —
  a second sweep skips the locked rows and, after commit, the ``reclaimed_at IS
  NULL`` predicate hides them.
- **Crash window**: a crash after ``unlink`` but before ``commit`` rolls back
  the stamp, leaving an absent file + unreclaimed row; the next sweep re-selects
  it, hits ``FileNotFoundError``, and stamps ``reclaimed_bytes = 0``. So
  ``reclaimed_bytes`` is "bytes measured at a clean reclaim, 0 if the file was
  already absent" — an audit counter, not an accounting guarantee.
- **Path safety**: the leaf is confined under ``media_root`` and ``lstat``-ed
  without following symlinks; a symlink, directory, or path escaping the root
  fails closed (logged, left unreclaimed) rather than being unlinked.
"""

import logging
import os
import stat as stat_module
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    AppSettings,
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = (RunStatus.COMPLETED.value, RunStatus.CANCELLED.value)


@dataclass(frozen=True)
class ReclaimSummary:
    """Per-sweep tally, returned and structured-logged."""

    selected: int = 0
    reclaimed: int = 0
    missing: int = 0
    failed: int = 0
    bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "selected": self.selected,
            "reclaimed": self.reclaimed,
            "missing": self.missing,
            "failed": self.failed,
            "bytes": self.bytes,
        }


class _UnsafePathError(Exception):
    """The stored artifact path escapes the media root or is not a plain file."""


class MediaRootUnavailableError(Exception):
    """``media_root`` is not a mounted directory — abort rather than mass-stamp.

    An unmounted/absent root would make every leaf ``lstat`` raise
    ``FileNotFoundError``, which the per-row path would otherwise misread as
    "file already gone" and stamp the whole batch reclaimed with 0 bytes while
    the real files sit on the unmounted volume. Fail the sweep loudly instead;
    the next beat tick retries once the volume is back.
    """


def _confined_parent_and_name(media_root: Path, rel: str) -> tuple[Path, str]:
    """Confine ``media_root / rel`` to ``(resolved_parent_dir, leaf_name)``.

    Rejects (fail closed) any absolute path or one containing a ``..`` segment —
    the alias vector by which a malformed row like
    ``artifacts/run/../../incoming/source.wav`` could otherwise normalize onto a
    retained source file. The parent is resolved (catching a symlinked
    intermediate directory that escapes the root) but the leaf name is kept
    un-followed so the caller's no-follow ``lstat``/``unlink`` can detect a
    symlink AT the artifact location.
    """
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or rel_path.name in ("", ".", ".."):
        raise _UnsafePathError(f"{rel!r} is not a plain relative artifact path")
    root = media_root.resolve()
    parent = (media_root / rel_path).parent.resolve()
    if not parent.is_relative_to(root):
        raise _UnsafePathError(f"{rel!r} escapes media root {root}")
    return parent, rel_path.name


def _reclaim_one(session: Session, artifact: AudioArtifact, media_root: Path) -> tuple[str, int]:
    """Unlink one artifact's file (if present) and stamp the row.

    Returns ``(outcome, bytes)`` where outcome is ``"reclaimed"`` (file unlinked,
    real bytes), ``"missing"`` (file already gone, bytes 0), or ``"failed"`` (an
    unsafe path or an OS error other than not-found — the row is left
    UNRECLAIMED so the next sweep retries it).

    Deletion goes through a directory file descriptor opened ``O_NOFOLLOW`` and
    then ``lstat``/``unlink`` by leaf name relative to it, so a parent-directory
    symlink swap between the check and the unlink cannot redirect the delete
    outside ``media_root`` (TOCTOU). ``media_root`` itself is verified present
    once per batch by the caller, so a missing parent here means the file is
    genuinely gone.
    """
    try:
        parent, name = _confined_parent_and_name(media_root, artifact.path)
    except _UnsafePathError as exc:
        logger.warning("gc: leaving artifact %s unreclaimed: %s", artifact.id, exc)
        return "failed", 0

    try:
        dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        # Parent dir gone under a present media_root ⇒ the file is gone too.
        _stamp(artifact, 0)
        return "missing", 0
    except OSError as exc:
        logger.warning("gc: cannot open artifact %s parent dir: %s", artifact.id, exc)
        return "failed", 0

    try:
        try:
            lst = os.lstat(name, dir_fd=dir_fd)  # never follows the leaf
        except FileNotFoundError:
            _stamp(artifact, 0)
            return "missing", 0
        if stat_module.S_ISLNK(lst.st_mode) or not stat_module.S_ISREG(lst.st_mode):
            logger.warning(
                "gc: artifact %s path %r is not a regular file — leaving unreclaimed",
                artifact.id,
                artifact.path,
            )
            return "failed", 0
        size = lst.st_size
        try:
            os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            _stamp(artifact, 0)
            return "missing", 0
        except OSError as exc:
            logger.warning("gc: unlink failed for artifact %s: %s", artifact.id, exc)
            return "failed", 0
        _stamp(artifact, size)
        return "reclaimed", size
    finally:
        os.close(dir_fd)


def _stamp(artifact: AudioArtifact, size: int) -> None:
    artifact.reclaimed_at = datetime.now(tz=UTC)
    artifact.reclaimed_bytes = size


def _select_eligible(
    session: Session,
    *,
    cutoff: datetime,
    batch_limit: int,
    tutorial_run_id: uuid.UUID | None,
) -> list[AudioArtifact]:
    """Claim up to ``batch_limit`` reclaimable artifacts, oldest run first.

    Eligible = an unreclaimed ``preprocessed_audio`` row under the ``artifacts/``
    subtree (never a source path — prepare writes ``artifacts/{run_id}/…``)
    whose run is terminal (``completed``/``cancelled`` — FAILED is deliberately
    excluded: a requeued FAILED run resumes at its failed stage and still needs
    the intermediate) and untouched since ``cutoff``, that is neither the
    tutorial run nor aliased by any ``media_items.source_path``. The
    ``artifacts/`` prefix plus the ``..``-rejecting confinement in
    :func:`_confined_parent_and_name` make it structurally impossible to reclaim
    a retained source file even via a malformed row.
    """
    source_alias = exists().where(MediaItem.source_path == AudioArtifact.path)
    stmt = (
        select(AudioArtifact)
        .join(PipelineRun, PipelineRun.id == AudioArtifact.pipeline_run_id)
        .where(
            AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            AudioArtifact.path.like("artifacts/%"),
            AudioArtifact.reclaimed_at.is_(None),
            PipelineRun.status.in_(_TERMINAL_STATUSES),
            PipelineRun.updated_at < cutoff,
            ~source_alias,
        )
        .order_by(PipelineRun.updated_at.asc(), AudioArtifact.id.asc())
        .limit(batch_limit)
        .with_for_update(of=AudioArtifact, skip_locked=True)
    )
    if tutorial_run_id is not None:
        stmt = stmt.where(PipelineRun.id != tutorial_run_id)
    return list(session.execute(stmt).scalars().all())


def reclaim_expired_intermediates(
    session: Session,
    *,
    media_root: Path,
    cutoff: datetime,
    batch_limit: int,
    tutorial_run_id: uuid.UUID | None,
) -> ReclaimSummary:
    """Reclaim one batch of expired normalized-audio intermediates.

    Runs the whole batch in the caller's transaction and commits once at the end
    so the ``FOR UPDATE`` claim is held across every unlink (see module docs).
    Per-row failures are isolated — one unsafe path or IO error never aborts the
    batch; that row is simply left unreclaimed for the next sweep.

    Aborts with :class:`MediaRootUnavailableError` if ``media_root`` is not a
    mounted directory, so a transient unmount can never mass-stamp live files as
    reclaimed (a per-row ``lstat`` cannot tell an absent leaf from an absent
    volume — the mount check can).
    """
    if not media_root.is_dir():
        raise MediaRootUnavailableError(
            f"gc: media_root {media_root} is not a directory — aborting sweep"
        )
    artifacts = _select_eligible(
        session, cutoff=cutoff, batch_limit=batch_limit, tutorial_run_id=tutorial_run_id
    )
    reclaimed = missing = failed = total_bytes = 0
    for artifact in artifacts:
        outcome, size = _reclaim_one(session, artifact, media_root)
        if outcome == "reclaimed":
            reclaimed += 1
            total_bytes += size
        elif outcome == "missing":
            missing += 1
        else:
            failed += 1
    session.commit()
    return ReclaimSummary(
        selected=len(artifacts),
        reclaimed=reclaimed,
        missing=missing,
        failed=failed,
        bytes=total_bytes,
    )


def run_intermediate_reclaimed_at(session: Session, run_id: uuid.UUID) -> datetime | None:
    """``reclaimed_at`` of the run's normalized-audio intermediate, or ``None``
    if the run has no such artifact or it has not been reclaimed.

    The console read-path uses this to render "media reclaimed" instead of a
    dead audio link, and ``/media`` uses it to answer 410 rather than the
    incidental 404 the missing file would otherwise produce.
    """
    return session.execute(
        select(AudioArtifact.reclaimed_at)
        .where(
            AudioArtifact.pipeline_run_id == run_id,
            AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
            AudioArtifact.reclaimed_at.is_not(None),
        )
        .limit(1)
    ).scalar_one_or_none()


def configured_tutorial_run_id(session: Session) -> uuid.UUID | None:
    """The raw configured tutorial run id (independent of whether its run still
    exists), used to exclude the tutorial run from reclamation."""
    return session.execute(select(AppSettings.tutorial_run_id).limit(1)).scalar_one_or_none()
