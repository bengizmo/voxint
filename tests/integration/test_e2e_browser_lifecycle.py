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

import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tools.e2e_browser_lifecycle import (
    _EDITOR_SEGMENTS,
    _EDITOR_SPLIT_ELIGIBLE,
    _RAIL_SEGMENTS,
    _SEED_SEGMENTS,
    Expectation,
    reconcile_run,
    seed_browser_run,
)

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import (
    Resolution,
    adjudication_queue,
    label_states,
)
from voxint.adjudication.review_state import set_correction, set_verified
from voxint.adjudication.splits import splittable_words
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    Decision,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentSplitBoundary,
    Speaker,
    TranscriptSegment,
)
from voxint.speakers.matching import MatchingGates
from voxint.speakers.policy import MatchBand


def _segments(session: Session, run_id: uuid.UUID) -> list[TranscriptSegment]:
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
        run_id, _media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value

        media = session.get(MediaItem, run.media_item_id)
        assert media is not None
        # duration_seconds MUST be set or playback_capability gates seeking off.
        assert media.duration_seconds == pytest.approx(5.0 * len(_SEED_SEGMENTS))

        artifact = (
            session.query(AudioArtifact).filter(AudioArtifact.pipeline_run_id == run_id).one()
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
        run_id, _media_id = seed_browser_run(session, tmp_path)

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
        run_id, _media_id = seed_browser_run(session, tmp_path)

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


def test_reconcile_flags_over_verification(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # The browser verified a segment the expectation did NOT list — the fail-closed
    # claim is "over- and under-verification both fail" (codex+kimi review).
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        for seg in (segs[0], segs[1], segs[2]):  # verified one (index 1) it shouldn't have
            set_verified(session, segment=seg, verified=True)
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [0, 2],
            "corrections": {},
            "progress": {"verified": 2, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("segment 1" in p and "verified=True" in p for p in problems)
    assert any("progress=(3," in p for p in problems)


def test_reconcile_flags_unexpected_correction(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        set_correction(session, segment=segs[3], text="browser wrote this unprompted")
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},  # expected NO corrections
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("segment 3" in p and "corrected_text" in p for p in problems)


def test_reconcile_flags_wrong_correction_text(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

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


# --- Editor fixture tests (Phase 6a, #157) ---


def test_seed_editor_fixture_builds_30_segments(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, media_id = seed_browser_run(session, tmp_path, fixture="editor")

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value

        media = session.get(MediaItem, media_id)
        assert media is not None
        assert media.duration_seconds == pytest.approx(5.0 * len(_EDITOR_SEGMENTS))

        segs = _segments(session, run_id)
        assert len(segs) == len(_EDITOR_SEGMENTS)
        labels = {s.diarization_label for s in segs}
        assert labels == {"S0", "S1", "S2", "S3"}


def test_seed_returns_media_id(session_factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_factory() as session:
        run_id, media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        media = session.get(MediaItem, media_id)
        assert media is not None
        assert run.media_item_id == media_id


def test_seed_rail_fixture_renders_every_band(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path, fixture="rail")

    with session_factory() as session:
        states = {
            state.label: state for state in label_states(session, run_id, gates=MatchingGates())
        }
        assert states["S0"].resolution is Resolution.GROUNDED_COSINE
        assert states["S0"].band is MatchBand.AUTO_ATTRIBUTE
        assert states["S0"].speaker_name == "Ada Roster"

        assert states["S1"].resolution is Resolution.UNRESOLVED
        assert states["S1"].band is MatchBand.REVIEW
        assert states["S1"].candidate_prompt_allowed is True
        assert states["S1"].candidate_speaker_name == "Blair Roster"

        assert states["S2"].resolution is Resolution.UNRESOLVED
        assert states["S2"].band is MatchBand.REVIEW
        assert states["S2"].candidate_prompt_allowed is False

        assert states["S3"].resolution is Resolution.UNRESOLVED
        assert states["S3"].band is MatchBand.ABSTAIN
        assert states["S3"].candidate_prompt_allowed is False
        assert states["S3"].match_reason == "below_cosine"

        assert states["S4"].resolution is Resolution.UNRESOLVED
        assert states["S4"].band is MatchBand.ABSTAIN
        assert states["S4"].match_reason == "too_few_turns"

        assert states["S5"].resolution is Resolution.AUTO_ENROLL
        assert states["S5"].speaker_name == "Voice 1"
        assert states["S5"].band is MatchBand.REVIEW
        assert states["S5"].candidate_speaker_name == "Blair Roster"
        assert states["S5"].candidate_prompt_allowed is True

        assert any(
            entry.run_id == run_id
            for entry in adjudication_queue(session, gates=MatchingGates())
        )


def test_reconcile_checks_label_rulings(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path, fixture="rail")

    with session_factory() as session:
        speakers = {speaker.display_name: speaker.id for speaker in session.query(Speaker).all()}
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S1",
            decision=Decision.ASSIGN,
            operator="e2e",
            idempotency_key=f"e2e-assign:{run_id}:S1",
            speaker_id=speakers["Blair Roster"],
        )
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S3",
            decision=Decision.UNKNOWN,
            operator="e2e",
            idempotency_key=f"e2e-unknown:{run_id}:S3",
        )
        session.commit()

    def expectation(label_rulings: dict[str, dict[str, str | None]]) -> Expectation:
        return Expectation.from_dict(
            {
                "progress": {"verified": 0, "total": len(_RAIL_SEGMENTS)},
                "label_rulings": label_rulings,
            }
        )

    matching = expectation(
        {
            "S1": {"decision": "assign", "speaker": "Blair Roster"},
            "S3": {"decision": "unknown", "speaker": None},
        }
    )
    with session_factory() as session:
        assert reconcile_run(session, run_id, matching) == []

    wrong_speaker = expectation(
        {
            "S1": {"decision": "assign", "speaker": "Ada Roster"},
            "S3": {"decision": "unknown", "speaker": None},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, wrong_speaker)
    assert len(problems) == 1
    assert "S1" in problems[0]

    unlisted = expectation({"S1": {"decision": "assign", "speaker": "Blair Roster"}})
    with session_factory() as session:
        problems = reconcile_run(session, run_id, unlisted)
    assert any("S3" in problem and "unlisted" in problem for problem in problems)

    # An explicit empty mapping means "the browser ruled on nothing": both
    # human rulings are unlisted; the seed's auto_enroll row on S5 is not.
    none_expected = expectation({})
    with session_factory() as session:
        problems = reconcile_run(session, run_id, none_expected)
    assert sorted(p.split(":")[0] for p in problems) == ["label S1", "label S3"]

    # An absent key skips the ledger check entirely.
    absent = Expectation.from_dict({"progress": {"verified": 0, "total": len(_RAIL_SEGMENTS)}})
    assert absent.label_rulings is None
    with session_factory() as session:
        assert reconcile_run(session, run_id, absent) == []


@pytest.mark.parametrize(
    "label_rulings",
    [
        {"S1": {"decision": "assign", "speaker": None}},
        {"S3": {"decision": "unknown", "speaker": "Blair Roster"}},
        {"S3": {"decision": "maybe", "speaker": None}},
    ],
)
def test_expectation_rejects_malformed_label_rulings(
    label_rulings: dict[str, dict[str, str | None]],
) -> None:
    with pytest.raises(ValueError):
        Expectation.from_dict(
            {
                "progress": {"verified": 0, "total": len(_RAIL_SEGMENTS)},
                "label_rulings": label_rulings,
            }
        )


def test_editor_fixture_split_eligible_segments_are_splittable(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path, fixture="editor")

    with session_factory() as session:
        segs = _segments(session, run_id)
        for seg in segs:
            words = splittable_words(seg)
            if seg.segment_index in _EDITOR_SPLIT_ELIGIBLE:
                assert words is not None, (
                    f"segment {seg.segment_index} marked split-eligible "
                    f"but splittable_words returned None"
                )
                assert len(words) >= 2
            else:
                if seg.segment_index == 0:
                    assert words is None


def test_reconcile_flags_unexpected_splits(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    import uuid as uuid_mod

    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        session.add(
            SegmentSplitBoundary(
                id=uuid_mod.uuid4(),
                pipeline_run_id=run_id,
                parent_segment_id=segs[2].id,
                word_index=2,
                operator="e2e-test",
            )
        )
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("unexpected splits" in p and "2" in p for p in problems)


def test_reconcile_flags_missing_splits(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
            "split_parent_indexes": [2],
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("expected splits" in p and "2" in p for p in problems)


def test_reconcile_passes_with_expected_splits(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    import uuid as uuid_mod

    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    with session_factory() as session:
        segs = _segments(session, run_id)
        session.add(
            SegmentSplitBoundary(
                id=uuid_mod.uuid4(),
                pipeline_run_id=run_id,
                parent_segment_id=segs[2].id,
                word_index=2,
                operator="e2e-test",
            )
        )
        session.commit()

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
            "split_parent_indexes": [2],
        }
    )
    with session_factory() as session:
        assert reconcile_run(session, run_id, expect) == []


def test_reconcile_flags_wrong_annotation_count(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
            "expected_annotations": 3,
        }
    )
    with session_factory() as session:
        problems = reconcile_run(session, run_id, expect)
    assert any("annotation count=0" in p and "expected 3" in p for p in problems)


def test_reconcile_passes_with_zero_annotations_expected(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id, _media_id = seed_browser_run(session, tmp_path)

    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": len(_SEED_SEGMENTS)},
            "expected_annotations": 0,
        }
    )
    with session_factory() as session:
        assert reconcile_run(session, run_id, expect) == []
