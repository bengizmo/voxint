"""Real-pipeline E2E: submit the tutorial clip, run the actual model services,
assert the persistence invariants the pipeline is contracted to leave behind.

These assert *shape and ranges*, never exact transcript text: real ASR output
varies run to run (temperature-0 does not make local inference bit-deterministic),
so an exact-string assertion would be a flake, not a gate. The ranges below were
measured in the Phase 0 canary (``sample-3speaker.wav``, 68.4 s, 3 speakers) and
are deliberately loose — a change that trips them is a real numerics/plumbing
regression, not sampling noise.

Measured quirk worth stating: whisper emits a *small* number of long segments for
this clip, all labelled ``SPEAKER_00``, even though diarization finds several
turns across the three real speakers. So we assert segment/turn/embedding
*counts and identities*, not a per-speaker transcript split.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.config import Settings
from voxint.db.models import (
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    RunStatus,
    TranscriptSegment,
)
from voxint.domain_packs.base import load_default
from voxint.media.normalize import TARGET_CHANNELS, TARGET_SAMPLE_RATE, probe_audio
from voxint.pipeline.engine import execute_run, submit
from voxint.pipeline.stages.context import build_stage_context, build_stage_fns

# Tutorial clip ground truth (Phase 0 canary, 0.16.0 model images).
EXPECTED_DURATION_SECONDS = 68.4
DURATION_TOLERANCE = 0.5
EMBEDDING_SPACE = "titanet-large-v1"

pytestmark = pytest.mark.usefixtures("model_services")


@dataclass(frozen=True)
class PipelineResult:
    run_id: uuid.UUID
    status: RunStatus
    preprocessed: list[AudioArtifact]
    segments: list[TranscriptSegment]
    turns: list[DiarizationTurn]
    duration_seconds: float | None


@pytest.fixture()
def artifact_gc(settings: Settings) -> Iterator[Callable[[uuid.UUID], None]]:
    """Register run ids whose on-disk ``artifacts/<run_id>/`` tree to remove.

    ``session_factory`` truncates DB rows between tests, but the preprocessed WAV
    the PREPARE stage writes under the shared media root is a file, not a row.
    Clean it so the maintainer's media directory does not accrue E2E debris.
    """
    media_root = Path(settings.media_root)
    run_ids: list[uuid.UUID] = []

    def _register(run_id: uuid.UUID) -> None:
        run_ids.append(run_id)

    yield _register

    for run_id in run_ids:
        shutil.rmtree(media_root / "artifacts" / str(run_id), ignore_errors=True)


def _drive_pipeline(
    session_factory: sessionmaker[Session],
    settings: Settings,
    source_path: str,
    register_gc: Callable[[uuid.UUID], None],
) -> PipelineResult:
    """Submit ``source_path`` and run the full pipeline in-process against the
    real services (not the live Celery worker), then read back what persisted."""
    with session_factory() as s:
        media = MediaItem(source_path=source_path)
        s.add(media)
        s.flush()
        media_id = media.id
        run = submit(s, media_id, domain_pack=load_default().to_mapping())
        run_id = run.id
        s.commit()
    register_gc(run_id)

    ctx = build_stage_context(settings)  # real HTTP clients; llm=None
    final = execute_run(session_factory, run_id, build_stage_fns(ctx))

    with session_factory() as s:
        preprocessed = list(
            s.execute(
                select(AudioArtifact).where(
                    AudioArtifact.pipeline_run_id == run_id,
                    AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value,
                )
            ).scalars()
        )
        segments = list(
            s.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.pipeline_run_id == run_id)
                .order_by(TranscriptSegment.segment_index)
            ).scalars()
        )
        turns = list(
            s.execute(
                select(DiarizationTurn).where(DiarizationTurn.pipeline_run_id == run_id)
            ).scalars()
        )
        # Duration is written onto the source MediaItem by the real PREPARE stage.
        duration = s.execute(
            select(MediaItem.duration_seconds).where(MediaItem.id == media_id)
        ).scalar_one()

    return PipelineResult(
        run_id=run_id,
        status=final.status,
        preprocessed=preprocessed,
        segments=segments,
        turns=turns,
        duration_seconds=duration,
    )


def _assert_invariants(result: PipelineResult, settings: Settings) -> None:
    assert result.status is RunStatus.COMPLETED, f"run did not complete: {result.status}"

    assert len(result.preprocessed) == 1, "expected exactly one preprocessed_audio artifact"
    norm = Path(settings.media_root) / result.preprocessed[0].path
    assert norm.is_file(), f"normalized audio missing on disk: {norm}"
    info = probe_audio(norm)
    assert (info.sample_rate, info.channels) == (TARGET_SAMPLE_RATE, TARGET_CHANNELS)

    assert len(result.segments) > 0, "no transcript segments persisted"
    assert all(seg.raw_text for seg in result.segments), "empty raw_text in a segment"

    assert len(result.turns) > 0, "no diarization turns persisted"
    embedded = [t for t in result.turns if t.embedding is not None]
    assert len(embedded) == len(result.turns), "some diarization turns are not embedded"
    spaces = {t.embedding_space for t in embedded}
    assert spaces == {EMBEDDING_SPACE}, f"unexpected embedding space(s): {spaces}"

    assert result.duration_seconds is not None, "media duration not populated by PREPARE"
    assert abs(result.duration_seconds - EXPECTED_DURATION_SECONDS) < DURATION_TOLERANCE, (
        f"duration {result.duration_seconds} outside "
        f"{EXPECTED_DURATION_SECONDS}±{DURATION_TOLERANCE}"
    )


def test_real_pipeline_persists_invariants(
    session_factory: sessionmaker[Session],
    settings: Settings,
    stage_media: Callable[[], str],
    artifact_gc: Callable[[uuid.UUID], None],
) -> None:
    """One full-pipeline pass leaves every persistence invariant intact."""
    result = _drive_pipeline(session_factory, settings, stage_media(), artifact_gc)
    _assert_invariants(result, settings)


def test_real_pipeline_repeats_cleanly(
    session_factory: sessionmaker[Session],
    settings: Settings,
    stage_media: Callable[[], str],
    artifact_gc: Callable[[uuid.UUID], None],
) -> None:
    """Two serial runs both complete with no cross-run leakage.

    Each submission gets a unique ``source_path`` and its own run id; the
    invariant read-backs are scoped by ``pipeline_run_id``, so a leak (rows
    bleeding between runs, a shared-artifact collision) would surface as a
    count/identity mismatch. Kept serial: the maintainer host is only cleared for
    low-concurrency E2E (issue #23).
    """
    first = _drive_pipeline(session_factory, settings, stage_media(), artifact_gc)
    second = _drive_pipeline(session_factory, settings, stage_media(), artifact_gc)

    assert first.run_id != second.run_id
    _assert_invariants(first, settings)
    _assert_invariants(second, settings)
