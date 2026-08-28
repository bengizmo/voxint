"""Extend adjudication_decisions CHECK constraints to accept 'auto_enroll'.

Auto-enrollment (#275) adds a system-initiated decision type: label-scope only,
always carries a speaker_id (like 'assign'). Two constraints change:

- adjudication_decisions_check: widen the decision enum to include 'auto_enroll'
- adjudication_decisions_assign_speaker_check: speaker_id IS NOT NULL for both
  'assign' and 'auto_enroll'

No new tables or columns.

Revision ID: 0049
Revises: 0048
"""

from alembic import op

revision: str = "0049"
down_revision: str = "0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "adjudication_decisions_check", "adjudication_decisions", type_="check"
    )
    op.create_check_constraint(
        "adjudication_decisions_check",
        "adjudication_decisions",
        "decision IN ('assign', 'exclude', 'unknown', 'inherit', 'auto_enroll')",
    )

    op.drop_constraint(
        "adjudication_decisions_assign_speaker_check",
        "adjudication_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "adjudication_decisions_assign_speaker_check",
        "adjudication_decisions",
        "(decision IN ('assign', 'auto_enroll')) = (speaker_id IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "adjudication_decisions_assign_speaker_check",
        "adjudication_decisions",
        type_="check",
    )
    op.create_check_constraint(
        "adjudication_decisions_assign_speaker_check",
        "adjudication_decisions",
        "(decision = 'assign') = (speaker_id IS NOT NULL)",
    )

    op.drop_constraint(
        "adjudication_decisions_check", "adjudication_decisions", type_="check"
    )
    op.create_check_constraint(
        "adjudication_decisions_check",
        "adjudication_decisions",
        "decision IN ('assign', 'exclude', 'unknown', 'inherit')",
    )
