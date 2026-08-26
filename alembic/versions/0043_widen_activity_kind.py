"""widen activity_events.kind for speaker identifications (issue #162, P7)

Console 2.0 P7 fast-follow to 0042: admits a second activity kind,
``speaker_identified`` (an operator naming a diarization label — assign / enroll
/ merge), alongside ``run_completed``. Only the ``kind`` CHECK widens; the table
shape is unchanged (every kind stays run-scoped, so ``pipeline_run_id`` remains
NOT NULL and the frozen ``title``/``href`` snapshot is the whole payload — no
per-kind provenance columns).

Downgrade is destructive to speaker events: the narrow CHECK cannot be restored
while ``speaker_identified`` rows exist, so they are deleted first.
``run_completed`` rows are preserved.

Revision ID: 0043
Revises: 0042
Create Date: 2026-08-26 15:35:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0043"
down_revision: str | None = "0042"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHECK = "activity_events_kind_check"
_WIDE = "'run_completed', 'speaker_identified'"
_NARROW = "'run_completed'"


def upgrade() -> None:
    op.drop_constraint(_CHECK, "activity_events", type_="check")
    op.create_check_constraint(_CHECK, "activity_events", f"kind IN ({_WIDE})")


def downgrade() -> None:
    # The narrow CHECK cannot be re-added while speaker rows exist; drop only
    # those (run_completed survives), then restore the single-value CHECK.
    op.execute("DELETE FROM activity_events WHERE kind = 'speaker_identified'")
    op.drop_constraint(_CHECK, "activity_events", type_="check")
    op.create_check_constraint(_CHECK, "activity_events", f"kind IN ({_NARROW})")
