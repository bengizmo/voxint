"""speaker_profiles: current per-field speaker profile with provenance

Issue #159 (Console 2.0 P4 Speakers). One row per (speaker, field) holding the
CURRENT value of a profile field (bio / affiliation / link) and where it came
from: ``provenance = 'manual'`` (operator-typed) or ``'enrichment'`` (an
accepted enrichment candidate materialized it, referenced by
``accepted_candidate_id`` — the CHECK ties the two together). History is NOT
kept here: the immutable enrichment candidate/evidence tables and the
append-only ``profile_review_decisions`` trail already record every draft and
verdict, so a later manual edit overwrites the row without losing anything.
``name`` and ``notes`` are deliberately excluded (they live on
``speakers.display_name`` / ``speakers.notes``).

The backfill materializes rows from decisions accepted BEFORE this migration
(otherwise previously accepted claims would silently vanish from the new
profile page): per (canonical speaker, field), the accepted decision with the
newest ``(created_at, id)`` wins, speaker ids are canonicalized through merge
tombstones at backfill time, and the row's operator/timestamps come from the
winning decision. Deterministic row ids (uuid5) + ON CONFLICT DO NOTHING keep
the backfill idempotent.

Downgrade drops the table. Rows with enrichment provenance are re-derivable
(re-upgrade re-backfills them); MANUAL edits made while on 0041 are lost on
downgrade — acceptable for a pre-1.0 single-operator rollback, and the
decision trail itself is never touched.

Revision ID: 0041
Revises: 0040
Create Date: 2026-08-25 15:30:00.000000
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0041"
down_revision: str | None = "0040"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROFILE_FIELDS = ("bio", "affiliation", "link")
_BACKFILL_NAMESPACE = uuid.UUID("7d1f8a52-9b7c-4f2e-b1a3-159159159159")


def _canonicalize(
    speaker_id: uuid.UUID, tombstones: dict[uuid.UUID, uuid.UUID]
) -> uuid.UUID:
    """Follow merge tombstones (chain-safe, cycle = abort) — mirrors
    ``speakers.roster.canonicalize`` without importing app code into a
    migration."""
    current = speaker_id
    visited = {current}
    while current in tombstones:
        current = tombstones[current]
        if current in visited:
            raise RuntimeError(
                f"speaker merge chain contains a cycle at {current}; "
                "repair the speakers table before migrating"
            )
        visited.add(current)
    return current


def _backfill(bind: sa.engine.Connection) -> None:
    tombstones = {
        row.id: row.merged_into_id
        for row in bind.execute(
            sa.text("SELECT id, merged_into_id FROM speakers WHERE merged_into_id IS NOT NULL")
        )
    }
    rows = bind.execute(
        sa.text(
            """
            SELECT d.id AS decision_id, d.created_at, d.operator,
                   c.id AS candidate_id, c.speaker_id, c.field, c.value
            FROM profile_review_decisions d
            JOIN enrichment_candidates c ON c.id = d.candidate_id
            WHERE d.decision = 'accept'
              AND c.field IN ('bio', 'affiliation', 'link')
              AND c.speaker_id IS NOT NULL
            ORDER BY d.created_at DESC, d.id DESC
            """
        )
    ).fetchall()
    # Newest-first iteration + first-wins per (canonical speaker, field).
    winners: dict[tuple[uuid.UUID, str], sa.Row] = {}
    for row in rows:
        canonical = _canonicalize(row.speaker_id, tombstones)
        winners.setdefault((canonical, row.field), row)
    for (canonical, field), row in winners.items():
        bind.execute(
            sa.text(
                """
                INSERT INTO speaker_profiles
                    (id, speaker_id, field, value, provenance,
                     accepted_candidate_id, operator, created_at, updated_at)
                VALUES
                    (:id, :speaker_id, :field, :value, 'enrichment',
                     :candidate_id, :operator, :created_at, :created_at)
                ON CONFLICT (speaker_id, field) DO NOTHING
                """
            ),
            {
                "id": uuid.uuid5(_BACKFILL_NAMESPACE, f"{canonical}:{field}"),
                "speaker_id": canonical,
                "field": field,
                "value": row.value,
                "candidate_id": row.candidate_id,
                "operator": row.operator,
                "created_at": row.created_at,
            },
        )


def upgrade() -> None:
    op.create_table(
        "speaker_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("speaker_id", sa.Uuid(), nullable=False),
        sa.Column("field", sa.Text(), nullable=False),
        # NULL value = a manual CLEAR tombstone (#159 review): the durable
        # marker that the operator removed the field, so a replayed accept or
        # a reconcile pass can never resurrect a cleared value.
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("provenance", sa.Text(), nullable=False),
        sa.Column("accepted_candidate_id", sa.Uuid(), nullable=True),
        sa.Column("operator", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["accepted_candidate_id"], ["enrichment_candidates.id"]),
        sa.UniqueConstraint("speaker_id", "field", name="speaker_profiles_speaker_field_key"),
        sa.CheckConstraint(
            "field IN ('bio', 'affiliation', 'link')",
            name="speaker_profiles_field_check",
        ),
        sa.CheckConstraint(
            "value IS NULL OR (length(trim(value)) > 0 AND char_length(value) <= 4000)",
            name="speaker_profiles_value_check",
        ),
        sa.CheckConstraint(
            "value IS NOT NULL OR provenance = 'manual'",
            name="speaker_profiles_cleared_shape_check",
        ),
        sa.CheckConstraint(
            "provenance IN ('manual', 'enrichment')",
            name="speaker_profiles_provenance_check",
        ),
        sa.CheckConstraint(
            "(provenance = 'enrichment') = (accepted_candidate_id IS NOT NULL)",
            name="speaker_profiles_provenance_candidate_check",
        ),
        sa.CheckConstraint(
            "length(trim(operator)) > 0 AND char_length(operator) <= 200",
            name="speaker_profiles_operator_check",
        ),
    )
    op.create_index(
        op.f("ix_speaker_profiles_speaker_id"), "speaker_profiles", ["speaker_id"]
    )
    op.create_index(
        op.f("ix_speaker_profiles_accepted_candidate_id"),
        "speaker_profiles",
        ["accepted_candidate_id"],
    )
    _backfill(op.get_bind())


def downgrade() -> None:
    op.drop_index(op.f("ix_speaker_profiles_accepted_candidate_id"), table_name="speaker_profiles")
    op.drop_index(op.f("ix_speaker_profiles_speaker_id"), table_name="speaker_profiles")
    op.drop_table("speaker_profiles")
