"""audio_artifacts: 'waveform_peaks' kind + one-per-run partial unique index

Issue #57 (waveform with color-coded speaker regions): the review console draws
an amplitude envelope of the normalized WAV. Peaks are computed lazily on the
first ``GET /media/{run_id}/peaks`` request and cached as a derived artifact —
``artifacts/{run_id}/peaks.json`` tracked by an ``audio_artifacts`` row with
``kind = 'waveform_peaks'`` — so the existing media-delete path cleans it up and
the GC reclamation sweep (which targets only ``preprocessed_audio``) leaves it
alone; a static waveform can still render after the WAV itself is reclaimed.

Cache-validity invariant: the row's ``meta.source_fingerprint`` records the
``{size, mtime_ns}`` of the WAV the peaks were computed FROM. Row presence alone
is NOT trusted while the WAV is live — prepare atomically replaces
``normalized.wav`` before its DB transaction commits, so a crash in that window
can strand a row describing the previous bytes; the peaks route fstat-verifies
the fingerprint on every cache hit and recomputes on mismatch. Prepare re-runs
also delete the row outright (same statement that clears the stale
``preprocessed_audio`` row).

The partial unique index (one ``waveform_peaks`` row per run) is the backstop
for concurrent first requests: both may compute, but ``INSERT … ON CONFLICT DO
NOTHING`` plus a reselect of the canonical row keeps exactly one.

Downgrade deletes the ``waveform_peaks`` rows (the narrowed CHECK would reject
them) — this ORPHANS any ``peaks.json`` files already on disk (~14 KB each),
because alembic has no media-root access to unlink them. Accepted and
documented rather than hidden; a re-upgrade simply recomputes over them.

Revision ID: 0021
Revises: 0020
Create Date: 2026-08-17 14:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Keep in lockstep with models.ArtifactKind (_enum_values).
    op.drop_constraint("audio_artifacts_kind_check", "audio_artifacts", type_="check")
    op.create_check_constraint(
        "audio_artifacts_kind_check",
        "audio_artifacts",
        "kind IN ('preprocessed_audio', 'chunk', 'transcript_export', 'waveform_peaks')",
    )
    op.create_index(
        "uq_audio_artifacts_waveform_peaks",
        "audio_artifacts",
        ["pipeline_run_id"],
        unique=True,
        postgresql_where=sa.text("kind = 'waveform_peaks'"),
    )


def downgrade() -> None:
    # The narrowed CHECK would reject surviving rows; delete them first. Their
    # peaks.json files are deliberately left on disk (documented above).
    op.execute("DELETE FROM audio_artifacts WHERE kind = 'waveform_peaks'")
    op.drop_index("uq_audio_artifacts_waveform_peaks", table_name="audio_artifacts")
    op.drop_constraint("audio_artifacts_kind_check", "audio_artifacts", type_="check")
    op.create_check_constraint(
        "audio_artifacts_kind_check",
        "audio_artifacts",
        "kind IN ('preprocessed_audio', 'chunk', 'transcript_export')",
    )
