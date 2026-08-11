"""Fake-provider end-to-end: a run walks all five stages against real Postgres."""

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FAKE_EMBEDDING_SPACE, FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
    Stage,
    StageRun,
    StageStatus,
    TranscriptSegment,
)
from voxint.pipeline.engine import StageFn, execute_run, submit

AUDIO = Path("/nonexistent/fixture.wav")  # fakes never touch the filesystem


def build_stage_fns() -> dict[Stage, StageFn]:
    asr, diarizer, embedder, llm = FakeASR(), FakeDiarizer(), FakeEmbedder(), FakeLLM()

    def prepare(session: Session, run_id: uuid.UUID) -> None:
        pass  # real preprocessing lands in P3

    def transcribe(session: Session, run_id: uuid.UUID) -> None:
        result = asr.transcribe(AUDIO)
        for i, seg in enumerate(result.segments):
            session.add(
                TranscriptSegment(
                    pipeline_run_id=run_id,
                    segment_index=i,
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    raw_text=seg.text,
                )
            )

    def diarize_embed(session: Session, run_id: uuid.UUID) -> None:
        turns = diarizer.diarize(AUDIO).turns
        segments = (
            session.execute(
                select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        for seg in segments:
            for turn in turns:
                if turn.start_seconds <= seg.start_seconds < turn.end_seconds:
                    seg.diarization_label = turn.label
        windows = tuple((t.start_seconds, t.end_seconds) for t in turns)
        embedding = embedder.embed(AUDIO, windows)
        assert len(embedding.entries) == len(windows)
        for turn, entry in zip(turns, embedding.entries, strict=True):
            if entry.embedding is None:
                continue  # skipped window (too_short / low_snr) — no vector persisted
            speaker = Speaker(display_name=f"unmatched-{run_id}-{turn.label}")
            session.add(speaker)
            session.flush()
            session.add(
                SpeakerEmbedding(
                    speaker_id=speaker.id,
                    embedding_space=embedding.embedding_space,
                    embedding=list(entry.embedding),
                    source_pipeline_run_id=run_id,
                )
            )

    def enhance_match(session: Session, run_id: uuid.UUID) -> None:
        segments = (
            session.execute(
                select(TranscriptSegment).where(TranscriptSegment.pipeline_run_id == run_id)
            )
            .scalars()
            .all()
        )
        for seg in segments:
            seg.enhanced_text = llm.enhance(seg.raw_text, context="")

    def finalize(session: Session, run_id: uuid.UUID) -> None:
        pass

    return {
        Stage.PREPARE: prepare,
        Stage.TRANSCRIBE: transcribe,
        Stage.DIARIZE_EMBED: diarize_embed,
        Stage.ENHANCE_MATCH: enhance_match,
        Stage.FINALIZE: finalize,
    }


def test_fake_provider_run_completes(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        media = MediaItem(source_path="/data/media/e2e.wav")
        session.add(media)
        session.flush()
        run = submit(session, media.id)
        run_id = run.id
        session.commit()

    final = execute_run(session_factory, run_id, build_stage_fns())
    assert final.status is RunStatus.COMPLETED
    assert final.current_stage is None

    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value
        # one completed stage_run per stage
        stage_rows = (
            session.execute(select(StageRun).where(StageRun.pipeline_run_id == run_id))
            .scalars()
            .all()
        )
        assert sorted(s.stage for s in stage_rows) == sorted(s.value for s in Stage)
        assert all(s.status == StageStatus.COMPLETED.value for s in stage_rows)

        segments = (
            session.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.pipeline_run_id == run_id)
                .order_by(TranscriptSegment.segment_index)
            )
            .scalars()
            .all()
        )
        assert len(segments) == 3
        # raw preserved, enhancement written beside it
        assert segments[0].raw_text == "hello and welcome to the show"
        assert segments[0].enhanced_text == "Hello and welcome to the show"
        assert {s.diarization_label for s in segments} == {"SPEAKER_00", "SPEAKER_01"}

        embeddings = session.execute(select(SpeakerEmbedding)).scalars().all()
        assert len(embeddings) == 2
        assert {e.embedding_space for e in embeddings} == {FAKE_EMBEDDING_SPACE}
        assert len(embeddings[0].embedding) == 192
