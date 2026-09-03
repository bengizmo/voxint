"""Integration tests for the speakers dedup CLI merge flow.

Covers the atomic rollback guarantee: if any merge in a batch fails,
all preceding merges in the same transaction are also rolled back.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.db.models import Speaker, SpeakerEmbedding

SPACE = "test-dedup-v1"


def _unit(dim: int, index: int) -> list[float]:
    v = np.zeros(dim)
    v[index % dim] = 1.0
    return v.tolist()


def _add_speaker(
    session: Session, name: str, embedding: list[float]
) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    session.add(
        SpeakerEmbedding(
            speaker_id=speaker.id, embedding_space=SPACE, embedding=embedding
        )
    )
    session.flush()
    return speaker.id


E0 = _unit(192, 0)
E0_CLOSE = (np.array(E0) * 0.99 + np.array(_unit(192, 1)) * 0.01).tolist()


def test_batch_rollback_on_mid_merge_error(
    session_factory: sessionmaker[Session],
) -> None:
    """When the second merge raises RosterError, the first merge must
    also be rolled back (no partial batch)."""
    from voxint.cli import main
    from voxint.speakers.roster import MergeResult, RosterError

    with session_factory() as session:
        voice1 = _add_speaker(session, "Voice 1", E0)
        alice = _add_speaker(session, "Alice", E0_CLOSE)
        voice2 = _add_speaker(session, "Voice 2", E0)
        bob = _add_speaker(session, "Bob", E0_CLOSE)
        session.commit()

    call_count = 0
    original_merge = None

    def _merge_with_second_failure(session, source_id, target_id):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RosterError("simulated conflict on second merge")
        return original_merge(session, source_id, target_id)

    with patch("voxint.cli._speakers_dedup.__module__", "voxint.cli"):
        import voxint.speakers.roster as roster_mod

        original_merge = roster_mod.merge_speakers
        with patch.object(roster_mod, "merge_speakers", side_effect=_merge_with_second_failure):
            exit_code = main([
                "speakers", "dedup",
                "--threshold", "0.50",
                "--merge", "--merge-threshold", "0.50", "--yes",
            ])

    assert exit_code == 1

    with session_factory() as session:
        for sid in (voice1, alice, voice2, bob):
            speaker = session.get(Speaker, sid)
            assert speaker is not None
            assert speaker.merged_into_id is None and speaker.deleted_at is None, (
                f"Speaker {speaker.display_name} should still be active after rollback"
            )
        for sid in (voice1, voice2):
            count = (
                session.query(SpeakerEmbedding)
                .filter(SpeakerEmbedding.speaker_id == sid)
                .count()
            )
            assert count >= 1, (
                f"Embeddings for {sid} should not have been repointed after rollback"
            )
