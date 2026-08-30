"""Widen pipeline_runs status CHECK for operator-initiated pause.

Admits ``paused`` alongside the existing six values. A paused run is
non-terminal and resumable — distinct from ``awaiting_adjudication``
(pipeline-initiated) and from ``cancelled`` (terminal).

Downgrade is destructive to paused runs: the narrow CHECK cannot be
restored while ``paused`` rows exist, so they are cancelled first.

Revision ID: 0053
Revises: 0052
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0053"
down_revision: str | None = "0052"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHECK = "pipeline_runs_status_check"
_WIDE = (
    "'queued', 'running', 'awaiting_adjudication', 'paused',"
    " 'completed', 'failed', 'cancelled'"
)
_NARROW = (
    "'queued', 'running', 'awaiting_adjudication',"
    " 'completed', 'failed', 'cancelled'"
)


def upgrade() -> None:
    op.drop_constraint(_CHECK, "pipeline_runs", type_="check")
    op.create_check_constraint(_CHECK, "pipeline_runs", f"status IN ({_WIDE})")


def downgrade() -> None:
    op.execute(
        "UPDATE pipeline_runs SET status = 'cancelled'"
        " WHERE status = 'paused'"
    )
    op.drop_constraint(_CHECK, "pipeline_runs", type_="check")
    op.create_check_constraint(_CHECK, "pipeline_runs", f"status IN ({_NARROW})")
