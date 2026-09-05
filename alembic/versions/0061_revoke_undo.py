"""Add compensating REVOKE decisions for adjudication undo (#158).

Revision ID: 0061
Revises: 0060
Create Date: 2026-09-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0061"
down_revision: str | None = "0060"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("adjudication_decisions") as batch_op:
        batch_op.add_column(sa.Column("voids_decision_id", sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            "adjudication_decisions_voids_decision_id_fkey",
            "adjudication_decisions",
            ["voids_decision_id"],
            ["id"],
        )
        batch_op.drop_constraint("adjudication_decisions_check", type_="check")
        batch_op.create_check_constraint(
            "adjudication_decisions_check",
            "decision IN ('assign', 'exclude', 'unknown', 'inherit', 'auto_enroll', 'revoke')",
        )
        batch_op.drop_constraint(
            "adjudication_decisions_assign_speaker_check", type_="check"
        )
        batch_op.create_check_constraint(
            "adjudication_decisions_assign_speaker_check",
            "(decision IN ('assign', 'auto_enroll')) = (speaker_id IS NOT NULL)"
            " OR decision = 'revoke'",
        )
        batch_op.create_check_constraint(
            "adjudication_decisions_revoke_check",
            "decision != 'revoke' OR (speaker_id IS NULL"
            " AND voids_decision_id IS NOT NULL AND transcript_segment_id IS NULL)",
        )
        batch_op.create_check_constraint(
            "adjudication_decisions_voids_only_revoke_check",
            "decision = 'revoke' OR voids_decision_id IS NULL",
        )
        batch_op.create_index(
            "ix_adjudication_decisions_voids",
            ["voids_decision_id"],
            unique=True,
            postgresql_where=sa.text("voids_decision_id IS NOT NULL"),
            sqlite_where=sa.text("voids_decision_id IS NOT NULL"),
        )


def downgrade() -> None:
    with op.batch_alter_table("adjudication_decisions") as batch_op:
        batch_op.drop_index("ix_adjudication_decisions_voids")
        batch_op.drop_constraint(
            "adjudication_decisions_voids_only_revoke_check", type_="check"
        )
        batch_op.drop_constraint("adjudication_decisions_revoke_check", type_="check")
        batch_op.drop_constraint(
            "adjudication_decisions_assign_speaker_check", type_="check"
        )
        batch_op.create_check_constraint(
            "adjudication_decisions_assign_speaker_check",
            "(decision IN ('assign', 'auto_enroll')) = (speaker_id IS NOT NULL)",
        )
        batch_op.drop_constraint("adjudication_decisions_check", type_="check")
        batch_op.create_check_constraint(
            "adjudication_decisions_check",
            "decision IN ('assign', 'exclude', 'unknown', 'inherit', 'auto_enroll')",
        )
        batch_op.drop_constraint(
            "adjudication_decisions_voids_decision_id_fkey", type_="foreignkey"
        )
        batch_op.drop_column("voids_decision_id")
