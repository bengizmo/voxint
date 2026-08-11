"""The real stage implementations end-to-end: fake providers, real Postgres,
real ffmpeg normalization, real persistence — everything but a GPU."""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FAKE_EMBEDDING_SPACE, FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    RunStatus,
    TranscriptSegment,
)
from voxint.media.normalize import TARGET_CHANNELS, TARGET_SAMPLE_RATE, probe_audio
from voxint.pipeline.engine import execute_run, submit
from voxint.pipeline.stages import diarize_embed
from voxint.pipeline.stages.context import StageContext, build_stage_fns

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    (tmp_path / "incoming").mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(tmp_path / "incoming" / "e2e.wav"),
        ],
        capture_output=True,
        check=True,
    )
    return tmp_path


@pytest.fixture()
def ctx(media_root: Path) -> StageContext:
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=media_root,
    )


def submit_media(
    session_factory: sessionmaker[Session], source_path: str
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(source_path=source_path)
        session.add(media)
        session.flush()
        run = submit(session, media.id)
        run_id = run.id
        session.commit()
    return run_id


def test_real_stages_run_completes(
    session_factory: sessionmaker[Session], ctx: StageContext, media_root: Path
) -> None:
    run_id = submit_media(session_factory, "incoming/e2e.wav")
    final = execute_run(session_factory, run_id, build_stage_fns(ctx))
    assert final.status is RunStatus.COMPLETED

    with session_factory() as session:
        # prepare: conforming artifact + canonical duration
        artifact = session.execute(select(AudioArtifact)).scalar_one()
        assert artifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
        normalized = media_root / artifact.path
        assert normalized.is_file()
        info = probe_audio(normalized)
        assert (info.sample_rate, info.channels) == (TARGET_SAMPLE_RATE, TARGET_CHANNELS)
        media = session.execute(select(MediaItem)).scalar_one()
        assert media.duration_seconds == pytest.approx(2.0, abs=0.1)

        # transcribe: raw text + suspect flag persisted
        segments = (
            session.execute(
                select(TranscriptSegment).order_by(TranscriptSegment.segment_index)
            )
            .scalars()
            .all()
        )
        assert [s.raw_text for s in segments] == [
            "hello and welcome to the show",
            "thanks for having me",
            "mm",
        ]
        assert [s.suspect for s in segments] == [False, False, True]

        # diarize_embed: full turn ledger, skips auditable, labels by max overlap
        turns = (
            session.execute(select(DiarizationTurn).order_by(DiarizationTurn.turn_index))
            .scalars()
            .all()
        )
        assert [t.label for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        assert turns[0].embedding is not None
        assert turns[0].embedding_space == FAKE_EMBEDDING_SPACE
        assert turns[0].snr_db == 20.0
        assert turns[2].embedding is None
        assert turns[2].skip_reason == "too_short"
        assert turns[2].embedding_space is None
        assert turns[2].overlap_seconds == 0.5
        # "mm" (8.0-9.0) overlaps SPEAKER_01's 4-9 turn more than the 8.5-9 one
        assert [s.diarization_label for s in segments] == [
            "SPEAKER_00",
            "SPEAKER_01",
            "SPEAKER_01",
        ]

        # enhance: written beside raw, never over it
        assert segments[0].enhanced_text == "Hello and welcome to the show"


def test_diarize_embed_retry_is_idempotent(
    session_factory: sessionmaker[Session], ctx: StageContext
) -> None:
    """At-least-once execution with repeated labels: no uniqueness errors, no
    duplicate ledger rows on a second pass."""
    run_id = submit_media(session_factory, "incoming/e2e.wav")
    fns = build_stage_fns(ctx)
    execute_run(session_factory, run_id, fns)

    def turn_rows() -> list[tuple[int, str, str | None, str | None, list[float] | None]]:
        with session_factory() as session:
            turns = (
                session.execute(
                    select(DiarizationTurn).order_by(DiarizationTurn.turn_index)
                )
                .scalars()
                .all()
            )
            return [
                (
                    t.turn_index,
                    t.label,
                    t.skip_reason,
                    t.embedding_space,
                    list(t.embedding) if t.embedding is not None else None,
                )
                for t in turns
            ]

    first_pass = turn_rows()
    with session_factory() as session:
        diarize_embed.run(ctx, session, run_id)
        session.commit()

    # Byte-identical ledger on the second pass — no reorder, no relabel.
    assert turn_rows() == first_pass
    assert len(first_pass) == 3
    with session_factory() as session:
        segments = session.execute(select(TranscriptSegment)).scalars().all()
        assert all(s.diarization_label is not None for s in segments)


def test_enhance_without_llm_leaves_null(
    session_factory: sessionmaker[Session], ctx: StageContext
) -> None:
    run_id = submit_media(session_factory, "incoming/e2e.wav")
    no_llm = StageContext(
        asr=ctx.asr,
        diarizer=ctx.diarizer,
        embedder=ctx.embedder,
        llm=None,
        media_root=ctx.media_root,
    )
    execute_run(session_factory, run_id, build_stage_fns(no_llm))
    with session_factory() as session:
        segments = session.execute(select(TranscriptSegment)).scalars().all()
        assert segments
        assert all(s.enhanced_text is None for s in segments)


def test_prepare_rejects_source_outside_media_root(
    session_factory: sessionmaker[Session], ctx: StageContext
) -> None:
    from voxint.pipeline.engine import StageFailedError

    run_id = submit_media(session_factory, "/etc/passwd")
    with pytest.raises(StageFailedError):
        execute_run(session_factory, run_id, build_stage_fns(ctx))
