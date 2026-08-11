"""speaker_assignments.proposed_name + method-shape constraints

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'speaker_assignments', sa.Column('proposed_name', sa.Text(), nullable=True)
    )
    # Remediate pre-0003 rows that would violate the new shape constraints.
    # speaker_assignments are regenerable machine proposals (the human ledger
    # lives in adjudication_decisions), so deleting nonconforming rows is safe;
    # re-running the runs' enhance_match stage recreates them in the new shape.
    op.execute(
        "DELETE FROM speaker_assignments WHERE"
        " (method = 'cosine' AND speaker_id IS NULL)"
        " OR (method = 'llm_hint')"
    )
    op.create_check_constraint(
        'speaker_assignments_cosine_shape_check',
        'speaker_assignments',
        "method != 'cosine' OR (speaker_id IS NOT NULL AND proposed_name IS NULL)",
    )
    op.create_check_constraint(
        'speaker_assignments_llm_hint_shape_check',
        'speaker_assignments',
        "method != 'llm_hint' OR (speaker_id IS NULL AND NOT grounded"
        " AND confidence IS NULL"
        " AND proposed_name IS NOT NULL AND length(trim(proposed_name)) > 0)",
    )


def downgrade() -> None:
    op.drop_constraint(
        'speaker_assignments_llm_hint_shape_check', 'speaker_assignments', type_='check'
    )
    op.drop_constraint(
        'speaker_assignments_cosine_shape_check', 'speaker_assignments', type_='check'
    )
    op.drop_column('speaker_assignments', 'proposed_name')
