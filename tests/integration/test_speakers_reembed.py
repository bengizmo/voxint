"""DB-backed coverage for the in-place voice-vector migration."""

import uuid
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakeEmbedder
from voxint.db.models import (
    EMBEDDING_DIM,
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MatchCandidate,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.diagnostics import check_voice_embedding_spaces
from voxint.speakers.matching import MatchingGates
from voxint.speakers.reembed import (
    EmbeddingSpaceDriftError,
    build_plan,
    reembed_run,
    refresh_run_matches,
)

OLD = "titanet-large-v1"
NEW = "titanet-large-v2"


def _unit(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


def _run(session: Session, media_root: Path, *, space: str = OLD) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    path = f"work/{run.id}.wav"
    (media_root / "work").mkdir(exist_ok=True)
    (media_root / path).touch()
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=path,
        )
    )
    for index, (start, end) in enumerate(((0.0, 4.0), (5.0, 9.0), (10.0, 10.5))):
        skipped = end - start < 1.0
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=start,
                end_seconds=end,
                label="S0",
                overlap=False,
                overlap_seconds=0.0,
                embedding=None if skipped else _unit(0),
                embedding_space=None if skipped else space,
                skip_reason="too_short" if skipped else None,
            )
        )
    session.add(
        TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=5.0,
            raw_text="unchanged",
            diarization_label="S0",
            suspect=False,
        )
    )
    session.flush()
    return run.id


def _enrollment(
    session: Session,
    run_id: uuid.UUID | None,
    *,
    label: str | None = "S0",
) -> uuid.UUID:
    speaker = Speaker(display_name=f"Speaker {uuid.uuid4()}")
    session.add(speaker)
    session.flush()
    row = SpeakerEmbedding(
        speaker_id=speaker.id,
        embedding_space=OLD,
        embedding=_unit(1),
        source_pipeline_run_id=run_id,
        source_diarization_label=label,
    )
    session.add(row)
    session.flush()
    return row.id


def test_dry_run_plan_selects_only_stale_completed_runs_and_writes_nothing(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        stale = _run(session, tmp_path)
        _run(session, tmp_path, space=NEW)
        session.commit()
    with session_factory() as session:
        plan = build_plan(session, NEW)
        session.rollback()
    assert plan.run_ids == (stale,)
    assert plan.turn_count == 3
    with session_factory() as session:
        assert set(session.scalars(select(DiarizationTurn.embedding_space))) == {OLD, NEW, None}


def test_migration_updates_in_place_rederives_enrollment_and_refreshes_proposals(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _run(session, tmp_path)
        enrollment_id = _enrollment(session, run_id)
        unmigratable_id = _enrollment(session, None)
        turn_ids = tuple(session.scalars(select(DiarizationTurn.id).order_by(DiarizationTurn.id)))
        segment_id = session.scalar(select(TranscriptSegment.id))
        session.commit()
    with session_factory() as session:
        plan = build_plan(session, NEW)
        assert plan.enrollment_count == 1
        assert plan.unmigratable_count == 1
    with session_factory() as session:
        result = reembed_run(
            session, run_id, NEW, FakeEmbedder(NEW), tmp_path, MatchingGates()
        )
        session.commit()
    with session_factory() as session:
        refresh_run_matches(session, run_id, MatchingGates())
        session.commit()
    assert result.turns == 3
    assert result.enrollments == 1
    with session_factory() as session:
        turns = list(
            session.scalars(
                select(DiarizationTurn)
                .where(DiarizationTurn.pipeline_run_id == run_id)
                .order_by(DiarizationTurn.id)
            )
        )
        assert tuple(turn.id for turn in turns) == turn_ids
        assert sum(turn.embedding is not None for turn in turns) == 2
        skipped = next(turn for turn in turns if turn.embedding is None)
        assert skipped.embedding_space is None
        assert skipped.skip_reason == "too_short"
        assert session.scalar(select(func.count(TranscriptSegment.id))) == 1
        segment = session.get(TranscriptSegment, segment_id)
        assert segment is not None and segment.raw_text == "unchanged"
        enrollment = session.get(SpeakerEmbedding, enrollment_id)
        assert enrollment is not None and enrollment.embedding_space == NEW
        unmigratable = session.get(SpeakerEmbedding, unmigratable_id)
        assert unmigratable is not None and unmigratable.embedding_space == OLD
        candidate = session.scalar(
            select(MatchCandidate).where(MatchCandidate.pipeline_run_id == run_id)
        )
        assert candidate is not None and candidate.embedding_space == NEW
        assert session.scalar(
            select(func.count(SpeakerAssignment.id)).where(
                SpeakerAssignment.pipeline_run_id == run_id
            )
        ) == 1
        assert build_plan(session, NEW).run_ids == ()


def test_stale_enrollment_is_reported_and_not_deleted(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _run(session, tmp_path)
        stale_id = _enrollment(session, run_id, label="missing")
        session.commit()
    with session_factory() as session:
        result = reembed_run(
            session, run_id, NEW, FakeEmbedder(NEW), tmp_path, MatchingGates()
        )
        session.commit()
    assert result.stale_enrollment_ids == (stale_id,)
    with session_factory() as session:
        row = session.get(SpeakerEmbedding, stale_id)
        assert row is not None and row.embedding_space == OLD


def test_run_filter_restricts_plan(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        first = _run(session, tmp_path)
        second = _run(session, tmp_path)
        session.commit()
    with session_factory() as session:
        assert build_plan(session, NEW, second).run_ids == (second,)
        assert first not in build_plan(session, NEW, second).run_ids


def test_space_drift_rolls_back_run_transaction(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        run_id = _run(session, tmp_path)
        session.commit()
    with session_factory() as session:
        with pytest.raises(EmbeddingSpaceDriftError):
            reembed_run(
                session,
                run_id,
                NEW,
                FakeEmbedder("unexpected-space"),
                tmp_path,
                MatchingGates(),
            )
        session.rollback()
    with session_factory() as session:
        spaces = set(
            session.scalars(
                select(DiarizationTurn.embedding_space).where(
                    DiarizationTurn.pipeline_run_id == run_id
                )
            )
        )
        assert spaces == {OLD, None}


def test_plan_discovers_enrollment_only_staleness(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """A run whose turns are already in the target space but whose enrollment
    is still in the old space must be discovered by build_plan."""
    with session_factory() as session:
        run_id = _run(session, tmp_path, space=NEW)
        _enrollment(session, run_id)
        session.commit()
    with session_factory() as session:
        plan = build_plan(session, NEW)
    assert plan.run_ids == (run_id,)
    assert plan.enrollment_count == 1


def test_doctor_warns_for_mixed_spaces_and_names_migration(
    engine: Engine, session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_factory() as session:
        _run(session, tmp_path, space=OLD)
        _run(session, tmp_path, space=NEW)
        session.commit()
    result = check_voice_embedding_spaces(engine, NEW)
    assert result.ok is False
    assert result.hard is False
    assert "voxint speakers re-embed" in result.detail
