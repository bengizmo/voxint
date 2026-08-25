"""The single write path for the media_folders relation (issue #153).

Console 2.0 P2a moves registered folders out of the ``app_settings.media_folders``
/ ``folder_domain_packs`` columns into first-class :class:`~voxint.db.models.MediaFolder`
rows (ADR 0002). Every registration mutation - the setup wizard folder step, the
Settings folder panel, and any future project-assignment flow - goes through this
module so three invariants hold uniformly and in one place:

- **No overlap.** A new registration is refused when it is an ancestor or a
  descendant of an already-registered folder (:func:`overlapping_registration`),
  because a file under a nested pair would belong to two folders and membership
  would be ambiguous. An exact duplicate is an idempotent no-op.
- **A folder cap** (``MAX_MEDIA_FOLDERS``), the same shape bound the legacy list
  enforced.
- **Serialized writes.** Each mutation takes a transaction-scoped Postgres
  advisory lock. The legacy list serialized on the singleton ``app_settings``
  row's ``FOR UPDATE``; ``media_folders`` has no single row to lock, so two
  overlapping requests (double-clicks, multiple tabs) could otherwise both pass
  the overlap check and insert nested registrations. The lock serializes the
  whole registration space instead. No-op on any non-Postgres harness (SQLite
  test seams are single-writer); the production/test app is Postgres-only.

Reads used by the cutover live here too (:func:`registered_folder_paths`,
:func:`folder_pack_map`, :func:`resolve_media_folder_id`) so the folder panel,
the scan, the watch sweep, and the submit-time pack snapshot all resolve folders
from the same relation the writes target - no dual-write window against the
retained (audit-only) ``app_settings`` columns.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.api.setup_wizard import (
    MAX_MEDIA_FOLDERS,
    SetupValidationError,
    normalize_media_folders,
)
from voxint.config import Settings
from voxint.db.models import MediaFolder, MediaItem
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.registry import available_domain_packs
from voxint.media.folders import deepest_ancestor, overlapping_registration

# "voxintmf" - a distinct advisory-lock namespace from the tutorial seed's key,
# so folder registration and tutorial seeding never contend for the same lock.
_REGISTRATION_ADVISORY_LOCK_KEY = 0x766F78696E746D66

# The Settings/wizard folder panel submits this explicit sentinel for the
# "Default" pack <option> so a disabled/absent select is never misread as "clear
# the mapping" (see :func:`set_folder_pack`). Public because the template context
# and the settings router both reference it.
PACK_DEFAULT_SENTINEL = "__default__"


def _lock(session: Session) -> None:
    """Take the transaction-scoped registration advisory lock (Postgres only)."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(select(func.pg_advisory_xact_lock(_REGISTRATION_ADVISORY_LOCK_KEY)))


def registered_folder_paths(session: Session) -> list[str]:
    """Every registered folder path, oldest registration first.

    Insertion order (``created_at``, then ``path`` as a stable tiebreak) mirrors
    the append-order the legacy ``app_settings.media_folders`` list preserved, so
    the folder panel renders folders in the same order after the cutover.
    """
    return list(
        session.execute(
            select(MediaFolder.path).order_by(MediaFolder.created_at, MediaFolder.path)
        ).scalars()
    )


def watched_folder_paths(session: Session) -> list[str]:
    """Registered folder paths flagged for the watch-folder sweep (``watch=true``).

    The pre-#153 behavior swept every registered folder under the installation-wide
    ``app_settings`` watch gate; the backfilled rows all carry ``watch=true`` so
    that stays byte-identical, while the per-row flag lets a folder opt out later
    without a schema change. Same order as :func:`registered_folder_paths`.
    """
    return list(
        session.execute(
            select(MediaFolder.path)
            .where(MediaFolder.watch.is_(True))
            .order_by(MediaFolder.created_at, MediaFolder.path)
        ).scalars()
    )


def folder_pack_map(session: Session) -> dict[str, str]:
    """The ``{path: domain_pack}`` mapping for pack-assigned folders.

    Only folders whose ``domain_pack`` is set appear, exactly the shape the old
    ``app_settings.folder_domain_packs`` column held and that
    ``domain_packs.registry.resolve_folder_pack_name`` consumes - so submit-time
    pack resolution is byte-identical, just sourced from the relation.
    """
    rows = session.execute(
        select(MediaFolder.path, MediaFolder.domain_pack).where(
            MediaFolder.domain_pack.is_not(None)
        )
    ).all()
    return {path: pack for path, pack in rows if pack is not None}


def resolve_media_folder_id(session: Session, source_path: str) -> uuid.UUID | None:
    """The id of the deepest registered folder that contains ``source_path``.

    ``None`` when the media sits under no registered folder (uploads, URLs,
    tutorial media, or a file outside every registration). Overlap refusal at
    registration time makes the deepest containing folder unique, so this is an
    unambiguous membership assignment (ADR 0002) for every local-path submission.
    """
    rows = session.execute(select(MediaFolder.id, MediaFolder.path)).all()
    id_by_path: dict[str, uuid.UUID] = {path: fid for fid, path in rows}
    folder = deepest_ancestor(source_path, list(id_by_path))
    return id_by_path.get(folder) if folder is not None else None


def register_folder(session: Session, settings: Settings, raw_path: str) -> str | None:
    """Register one folder under MEDIA_ROOT. Returns an error message, else ``None``.

    Only the SUBMITTED path is validated (via ``normalize_media_folders`` - resolve
    + containment + reserved-tree + existing-dir), so a stale/removed *existing*
    folder can never block a new add. Re-adding an already-registered folder is an
    idempotent no-op; an overlapping (nested) registration is refused; the folder
    cap is enforced. All checks run after the advisory lock so a concurrent add
    cannot slip a nesting pair past the overlap check.
    """
    try:
        normalized = normalize_media_folders([raw_path], settings.media_root)
    except SetupValidationError as exc:
        return str(exc)
    if not normalized:
        return "Choose a folder to add."
    folder = normalized[0]
    _lock(session)
    existing = registered_folder_paths(session)
    if folder in existing:
        return None  # idempotent
    overlap = overlapping_registration(folder, existing)
    if overlap is not None:
        return (
            f"That folder overlaps an already-registered folder ({overlap}). "
            "Register only the parent or only the child so each file belongs to "
            "exactly one folder."
        )
    if len(existing) + 1 > MAX_MEDIA_FOLDERS:
        return f"You can register at most {MAX_MEDIA_FOLDERS} folders."
    session.add(MediaFolder(path=folder))
    session.flush()
    return None


def unregister_folder(session: Session, settings: Settings, path: str) -> str | None:
    """Unregister a folder by path. Idempotent; never touches the filesystem.

    A ``path`` not currently registered is a no-op. Deleting the row nulls the
    membership of any media that pointed at it (the ``media_items.media_folder_id``
    FK is ``ON DELETE SET NULL``); their frozen run snapshots are unaffected.
    """
    _lock(session)
    row = session.execute(
        select(MediaFolder).where(MediaFolder.path == path)
    ).scalar_one_or_none()
    if row is None:
        return None  # already gone - idempotent
    session.delete(row)
    session.flush()
    return None


def unregister_folder_by_id(
    session: Session, folder_id: uuid.UUID
) -> tuple[bool, int]:
    """Unregister a folder BY ID, reporting how many media it reverts to global.

    Keyed on the primary key rather than a path string: the /media folder panel
    renders each registered folder from :func:`voxint.api.media_query.folder_options`
    (which carries the row id), so removal names the exact row and never depends on
    reconstructing the stored ``path`` (:func:`unregister_folder` matches ``path``
    exactly and would silently no-op on any mismatch). Returns ``(removed,
    reverted_count)``: ``removed`` is False for an id that is already gone
    (idempotent), and ``reverted_count`` is how many :class:`~voxint.db.models.MediaItem`
    rows pointed at the folder and are therefore nulled by the ``ON DELETE SET NULL``
    FK. The count is taken under the same advisory lock and in the same transaction
    as the delete, immediately before it, so the reported number is what this
    transaction removes, not a value read at an earlier, racy moment (the panel
    shows it honestly: "N files reverted to global settings"). Never touches the
    filesystem; the reverted media keep their frozen run snapshots.
    """
    _lock(session)
    row = session.get(MediaFolder, folder_id)
    if row is None:
        return False, 0  # already gone - idempotent
    reverted = session.execute(
        select(func.count())
        .select_from(MediaItem)
        .where(MediaItem.media_folder_id == row.id)
    ).scalar_one()
    session.delete(row)
    session.flush()
    return True, reverted


def set_folder_pack(
    session: Session, settings: Settings, path: str, pack: str | None
) -> str | None:
    """Assign (or clear) a registered folder's domain pack. Last write wins.

    ``pack is None`` means the ``pack`` field was *absent* from the submission - a
    no-op, never a clear (the panel disables the ``<select>`` when the registry is
    down, and a disabled control submits nothing). ``pack == PACK_DEFAULT_SENTINEL``
    is the explicit "Default" choice and clears the pack (NULL = inherit the
    default). Any other non-empty pack must resolve in ``available_domain_packs``.
    The folder must already be registered.
    """
    if pack is None:
        return None  # field absent (e.g. disabled select) - never touch the pack
    _lock(session)
    row = session.execute(
        select(MediaFolder).where(MediaFolder.path == path)
    ).scalar_one_or_none()
    if row is None:
        return "That folder is not registered."
    if pack == PACK_DEFAULT_SENTINEL:
        row.domain_pack = None
        return None
    try:
        available = available_domain_packs(settings)
    except DomainPackError:
        return (
            "Domain packs can't be listed right now - check your DOMAIN_PACKS_DIR / "
            "DOMAIN_PACK_PATH configuration."
        )
    if pack not in available:
        return f"Unknown domain pack: {pack}."
    row.domain_pack = pack
    return None
