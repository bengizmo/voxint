"""projects, media_folders, and the media location split

Issue #153 (Console 2.0 P2a). Introduces the relational organization layer the
console library and projects build on, and splits media location from identity
(ADR 0001, ADR 0002):

- ``projects`` — a named grouping of media folders with project-scoped
  ``vocabulary`` / ``corrections`` (nullable: NULL inherits, ``[]`` is
  explicitly empty and wins).
- ``media_folders`` — the relational successor to ``app_settings.media_folders``
  + ``folder_domain_packs``: one row per registered folder, ``path`` unique and
  non-empty (MEDIA_ROOT-relative POSIX), optional ``project_id`` FK, per-folder
  ``domain_pack``, and a ``watch`` flag.
- ``media_items.current_path`` — the mutable live location, backfilled to
  ``source_path`` (identity stays on ``source_path``). Nullable in P2a; the P2c
  move slice tightens it to NOT NULL once every byte-opener reads it.
- ``media_items.media_folder_id`` — folder membership (FK), backfilled by the
  deepest registered-folder ancestor of ``source_path``.

The backfill copies the legacy registrations verbatim (``domain_pack`` from
``folder_domain_packs[path]``) and then asserts, for every media row, that the
new folder-relation pack equals the old longest-ancestor resolution. A mismatch
aborts the upgrade: it means the install has nested/overlapping registrations
whose effective pack cannot be preserved by a flat folder relation, and the
operator must reconcile them first (``voxint media folders preflight``). The
``app_settings.media_folders`` / ``folder_domain_packs`` columns are LEFT INTACT
as a one-release rollback/audit input; a later migration drops them.

Revision ID: 0040
Revises: 0039
Create Date: 2026-08-24 19:20:00.000000
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

import sqlalchemy as sa
from alembic import op

revision: str = "0040"
down_revision: str | None = "0039"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _deepest_ancestor(source_path: str, folders: Sequence[str]) -> str | None:
    """The deepest folder in ``folders`` that contains ``source_path``.

    Component-boundary matching (via ``PurePosixPath``) so ``audio/pod`` never
    matches a file under ``audio/podcasts``; a folder equal to the file's own
    directory or any ancestor matches. Mirrors
    ``domain_packs.registry.resolve_folder_pack_name`` but over ALL registered
    folders, not only the pack-mapped ones. ``None`` when no folder contains it.
    """
    src = PurePosixPath(source_path)
    best: str | None = None
    best_depth = -1
    for folder in folders:
        folder_path = PurePosixPath(folder)
        if src == folder_path or folder_path in src.parents:
            depth = len(folder_path.parts)
            if depth > best_depth:
                best_depth = depth
                best = folder
    return best


def _folder_pack_name(
    source_path: str, folder_domain_packs: Mapping[str, str]
) -> str | None:
    """Old resolution: the pack of the deepest pack-MAPPED ancestor folder."""
    mapped = _deepest_ancestor(source_path, list(folder_domain_packs))
    return folder_domain_packs.get(mapped) if mapped is not None else None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("vocabulary", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("corrections", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("name", name="projects_name_key"),
        sa.CheckConstraint("length(btrim(name)) > 0", name="projects_name_nonempty_check"),
        sa.CheckConstraint(
            "corrections IS NULL OR jsonb_typeof(corrections) = 'array'",
            name="projects_corrections_array_check",
        ),
    )
    op.create_table(
        "media_folders",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("domain_pack", sa.Text(), nullable=True),
        sa.Column(
            "watch",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name="media_folders_project_id_fkey",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("path", name="media_folders_path_key"),
        sa.CheckConstraint("length(path) > 0", name="media_folders_path_nonempty_check"),
    )
    op.create_index("ix_media_folders_project_id", "media_folders", ["project_id"])

    op.add_column("media_items", sa.Column("current_path", sa.Text(), nullable=True))
    op.add_column(
        "media_items", sa.Column("media_folder_id", sa.Uuid(), nullable=True)
    )
    op.create_foreign_key(
        "media_items_media_folder_id_fkey",
        "media_items",
        "media_folders",
        ["media_folder_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_media_items_media_folder_id", "media_items", ["media_folder_id"]
    )

    _backfill(op.get_bind())


def _backfill(bind: sa.Connection) -> None:
    # current_path := source_path for every existing row (identity == location at
    # rest; ADR 0001). Idempotent (re-running sets the same value).
    bind.execute(
        sa.text(
            "UPDATE media_items SET current_path = source_path"
            " WHERE current_path IS NULL"
        )
    )

    row = bind.execute(
        sa.text(
            "SELECT media_folders, folder_domain_packs FROM app_settings WHERE id = 1"
        )
    ).one_or_none()
    folders: list[str] = list(row[0]) if row is not None and row[0] else []
    packs: dict[str, str] = dict(row[1]) if row is not None and row[1] else {}
    if not folders:
        return  # fresh/un-onboarded install: nothing registered to migrate.

    # One media_folders row per registered folder, pack copied verbatim from the
    # legacy mapping. Deterministic id keyed on the path so re-running the backfill
    # is a no-op rather than a duplicate.
    folder_ids: dict[str, uuid.UUID] = {}
    for path in folders:
        folder_id = uuid.uuid5(uuid.NAMESPACE_URL, f"voxint:media_folder:{path}")
        folder_ids[path] = folder_id
        bind.execute(
            sa.text(
                "INSERT INTO media_folders (id, path, domain_pack, watch)"
                " VALUES (:id, :path, :pack, true)"
                " ON CONFLICT (path) DO NOTHING"
            ),
            {"id": folder_id, "path": path, "pack": packs.get(path)},
        )

    # Assign each media item to its deepest registered-folder ancestor, and verify
    # the new folder-relation pack reproduces the old longest-ancestor resolution.
    media_rows = bind.execute(
        sa.text("SELECT id, source_path FROM media_items")
    ).all()
    mismatches: list[str] = []
    for media_id, source_path in media_rows:
        folder = _deepest_ancestor(source_path, folders)
        old_pack = _folder_pack_name(source_path, packs)
        new_pack = packs.get(folder) if folder is not None else None
        if old_pack != new_pack:
            mismatches.append(
                f"{source_path!r}: old pack {old_pack!r} vs folder pack {new_pack!r}"
            )
            continue
        if folder is not None:
            bind.execute(
                sa.text(
                    "UPDATE media_items SET media_folder_id = :fid WHERE id = :mid"
                ),
                {"fid": folder_ids[folder], "mid": media_id},
            )
    if mismatches:
        raise RuntimeError(
            "media folder migration would change the effective domain pack for "
            f"{len(mismatches)} media item(s) because of nested/overlapping folder "
            "registrations. Reconcile them (register only the parent or only the "
            "child folder) before upgrading; run `voxint media folders preflight` "
            "for the full report. Mismatches: " + "; ".join(mismatches[:10])
        )


def downgrade() -> None:
    op.drop_index("ix_media_items_media_folder_id", table_name="media_items")
    op.drop_constraint(
        "media_items_media_folder_id_fkey", "media_items", type_="foreignkey"
    )
    op.drop_column("media_items", "media_folder_id")
    op.drop_column("media_items", "current_path")
    op.drop_index("ix_media_folders_project_id", table_name="media_folders")
    op.drop_table("media_folders")
    op.drop_table("projects")
