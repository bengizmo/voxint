"""speaker roster lifecycle: merge tombstones and reversible archive

Adds the curation lifecycle to ``speakers`` (issue #7) without ever touching
the append-only ``adjudication_decisions`` ledger:

- ``merged_into_id`` (self-FK) + ``merged_at``: a merged speaker is retained
  as a tombstone so historical ledger FKs stay valid; embeddings and machine
  assignments are repointed to the target and readers canonicalize at read
  time. Writes keep chains collapsed to depth 1 (service-enforced; a CHECK
  cannot express it).
- ``deleted_at``: reversible archive. Archived speakers keep their name,
  embeddings, and human decisions but leave the matching roster.

CHECKs pinned here: no self-merge; ``merged_into_id``/``merged_at`` set
together; a row is never both merged and deleted. The global
``UNIQUE(display_name)`` from 0001 is deliberately kept — names are never
reusable; operators restore or merge instead of re-creating identities.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-14 18:30:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("speakers", sa.Column("merged_into_id", sa.Uuid(), nullable=True))
    op.add_column(
        "speakers", sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "speakers", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_foreign_key(
        "speakers_merged_into_id_fkey",
        "speakers",
        "speakers",
        ["merged_into_id"],
        ["id"],
    )
    op.create_index("ix_speakers_merged_into_id", "speakers", ["merged_into_id"])
    op.create_check_constraint(
        "speakers_no_self_merge_check",
        "speakers",
        "merged_into_id IS NULL OR merged_into_id != id",
    )
    op.create_check_constraint(
        "speakers_merge_fields_together_check",
        "speakers",
        "(merged_into_id IS NULL) = (merged_at IS NULL)",
    )
    op.create_check_constraint(
        "speakers_not_merged_and_deleted_check",
        "speakers",
        "merged_into_id IS NULL OR deleted_at IS NULL",
    )


def downgrade() -> None:
    op.drop_constraint("speakers_not_merged_and_deleted_check", "speakers", type_="check")
    op.drop_constraint("speakers_merge_fields_together_check", "speakers", type_="check")
    op.drop_constraint("speakers_no_self_merge_check", "speakers", type_="check")
    op.drop_index("ix_speakers_merged_into_id", table_name="speakers")
    op.drop_constraint("speakers_merged_into_id_fkey", "speakers", type_="foreignkey")
    op.drop_column("speakers", "deleted_at")
    op.drop_column("speakers", "merged_at")
    op.drop_column("speakers", "merged_into_id")
