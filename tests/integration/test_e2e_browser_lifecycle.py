"""DB-backed tests for the browser E2E lifecycle tool's seed + reconcile.

These exercise the two pieces that must be correct for the browser lane to mean
anything: the seed builds a COMPLETED run shaped for the review loop (audio
artifact + duration + varied-confidence segments including sub-threshold ones),
and the fail-closed reconciler agrees with durable state ONLY when the browser's
writes match the expectation exactly — over- and under-verification both fail.
The reconciler is fed real ``set_verified`` / ``set_correction`` writes (the
same writers the review routes call), so a drift between the tool and the
production write path is caught here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tools.e2e_browser_lifecycle import (
    _SEED_SEGMENTS,
    Expectation,
    reconcile_run,
    seed_browser_run,
)

from voxint.adjudication.review_state import set_correction, set_verified
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    MediaItem,
    PipelineRun,
    RunStatus,
    TranscriptSegment,
)


def _segments(session: Session, run_id) -> list[TranscriptSegment]:
    return (
        session.query(TranscriptSegment)
        .filter(TranscriptSegment.pipeline_run_id == run_id)
        .order_by(TranscriptSegment.segment_index)
        .all()
    )


def test_seed_builds_a_completed_review_run(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value

        media = session.get(MediaItem, run.media_item_id)
        assert media is not None
        # duration_seconds MUST be set or playback_capability gates seeking off.
        assert media.duration_seconds == pytest.approx(5.0 * len(_SEED_SEGMENTS))

        artifact = (
            session.query(AudioArtifact)
            .filter(AudioArtifact.pipeline_run_id == run_id)
            .one()
        )
        assert artifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
        assert (tmp_path / artifact.path).is_file()  # the playback source exists

        segs = _segments(session, run_id)
        assert len(segs) == len(_SEED_SEGMENTS)
        confidences = [s.confidence for s in segs]
        assert any(c is not None and c < 0.6 for c in confidences)  # uncertain chips
        assert any(c is None for c in confidences)  # a NULL (never flagged)
        assert any(c is not None and c >= 0.6 for c in confidences)  # a certain one


def test_reconcile_passes_on_matching_durable_state(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        set_verified(session, segment=segs[0], verified=True)
        set_verified(session, segment=segs[2], verified=True)
        set_correction(session, segment=segs[1], text="a genuine operator correction")
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [0, 2],
            "corrections": {"1": "a genuine operator correction"},
            "progress": {"verified": 2, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        assert reconcile_run(session, run_id, expect) == []


def test_reconcile_flags_under_and_over_verification(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        set_verified(session, segment=segs[0], verified=True)  # only one verified
        session.commit()

    # Expect TWO verified (0 and 2) → both a missing verify AND a wrong progress.
    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [0, 2],
            "corrections": {},
            "progress": {"verified": 2, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("segment 2" in p and "verified=False" in p for p in problems)
    assert any("progress=(1," in p for p in problems)


def test_reconcile_flags_wrong_correction_text(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        set_correction(session, segment=segs[1], text="what the browser actually wrote")
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {"1": "what the test expected instead"},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("segment 1" in p and "corrected_text" in p for p in problems)


def test_reconcile_reports_missing_run(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    import uuid

    expect = Expectation.from_dict({"progress": {"verified": 0, "total": 0}})
    with session_factory() as session:
        problems = reconcile_run(session, uuid.uuid4(), expect)
    assert any("no transcript segments" in p for p in problems)
