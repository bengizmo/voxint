"""Real-pipeline E2E: submit the tutorial clip, run the actual model services,
assert the persistence invariants the pipeline is contracted to leave behind.

What is asserted: run status, artifact *presence* and normalization, the
embedding *space identity*, that every diarization turn is embedded, that
duration is populated by the real PREPARE stage, and that segment/turn counts sit
inside *generous* bounds anchored on the Phase 0 canary. Never exact transcript
text and never exact counts: real ASR output varies run to run (temperature-0
does not make local inference bit-deterministic), so a tight assertion would be a
flake, not a gate. The bounds are wide enough to absorb sampling noise but catch
a degenerate service (e.g. an empty or single-full-clip response) — a trip is a
real numerics/plumbing regression.

Measured quirk worth stating: whisper emits a *small* number of long segments for
this clip, all labelled ``SPEAKER_00``, even though diarization finds several
turns across the three real speakers — so the bounds are on *counts*, not on a
per-speaker transcript split.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import func, select
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
from voxint.pipeline.stages.context import StageContext, build_stage_fns

# Tutorial clip ground truth (Phase 0 canary, 0.16.0 model images).
EXPECTED_DURATION_SECONDS = 68.4
DURATION_TOLERANCE = 0.5
EMBEDDING_SPACE = "titanet-large-v2"
# Generous bounds around the canary (3 whisper segments, 7 diarization turns for
# this 68 s / 3-speaker clip). Wide enough that ASR sampling noise never trips
# them; tight enough to catch a degenerate service (0 output, or one full-clip
# blob) or a runaway (hundreds of tiny segments from a broken VAD).
MIN_SEGMENTS, MAX_SEGMENTS = 1, 40
MIN_TURNS, MAX_TURNS = 2, 40

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
    stage_context: StageContext,
    source_path: str,
    register_gc: Callable[[uuid.UUID], None],
) -> PipelineResult:
    """Submit ``source_path`` and run the transcription/diarization/embedding
    stages in-process against the real services, then read back what persisted.

    Scope, stated honestly: this drives the pipeline stage functions directly, so
    it proves the stage + persistence contract against the real models — NOT the
    API→enqueue→Celery-worker path (a broken worker wiring would not surface
    here), and with ``llm=None`` the LLM-gated enrichment stages no-op (their
    lane is separate; see the plan). The stage context is the shared,
    session-scoped one (built once, closed at teardown) — the worker likewise
    builds the transport clients once and reuses them across runs."""
    with session_factory() as s:
        media = MediaItem(source_path=source_path)
        s.add(media)
        s.flush()
        media_id = media.id
        run = submit(s, media_id, domain_pack=load_default().to_mapping())
        run_id = run.id
        s.commit()
    register_gc(run_id)

    final = execute_run(session_factory, run_id, build_stage_fns(stage_context))

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
    artifact_path = result.preprocessed[0].path
    # Layout the artifact_gc fixture depends on to clean up; if PREPARE's path
    # convention drifts, fail loudly here instead of silently leaking files under
    # the operator's media root.
    assert artifact_path.startswith(f"artifacts/{result.run_id}/"), (
        f"preprocessed artifact path {artifact_path!r} is not under "
        f"artifacts/{result.run_id}/ — layout drifted, artifact_gc would leak"
    )
    norm = Path(settings.media_root) / artifact_path
    assert norm.is_file(), f"normalized audio missing on disk: {norm}"
    info = probe_audio(norm)
    assert (info.sample_rate, info.channels) == (TARGET_SAMPLE_RATE, TARGET_CHANNELS)

    n_segments = len(result.segments)
    assert MIN_SEGMENTS <= n_segments <= MAX_SEGMENTS, (
        f"segment count {n_segments} outside [{MIN_SEGMENTS}, {MAX_SEGMENTS}] "
        "(degenerate or runaway ASR?)"
    )
    assert all(seg.raw_text and seg.raw_text.strip() for seg in result.segments), (
        "blank or whitespace-only raw_text in a segment"
    )

    n_turns = len(result.turns)
    assert MIN_TURNS <= n_turns <= MAX_TURNS, (
        f"diarization turn count {n_turns} outside [{MIN_TURNS}, {MAX_TURNS}] "
        "(degenerate or runaway diarization?)"
    )
    embedded = [t for t in result.turns if t.embedding is not None]
    assert len(embedded) == n_turns, "some diarization turns are not embedded"
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
    stage_context: StageContext,
    stage_media: Callable[[], str],
    artifact_gc: Callable[[uuid.UUID], None],
) -> None:
    """One full-pipeline pass leaves every persistence invariant intact."""
    result = _drive_pipeline(session_factory, stage_context, stage_media(), artifact_gc)
    _assert_invariants(result, settings)


def test_real_pipeline_repeats_cleanly(
    session_factory: sessionmaker[Session],
    settings: Settings,
    stage_context: StageContext,
    stage_media: Callable[[], str],
    artifact_gc: Callable[[uuid.UUID], None],
) -> None:
    """Two serial runs both complete with no cross-run leakage.

    Each submission gets a unique ``source_path`` and its own run id, and both
    runs satisfy the full invariant set independently. The leak check is explicit:
    the total rows in the DB equal the sum of the two runs' per-run rows, so a run
    that mis-attributed or duplicated another's segments/turns/artifacts would
    fail here even though the per-run reads (scoped by ``pipeline_run_id``) look
    fine. Kept serial: the maintainer host is only cleared for low-concurrency
    E2E (issue #23).
    """
    first = _drive_pipeline(session_factory, stage_context, stage_media(), artifact_gc)
    second = _drive_pipeline(session_factory, stage_context, stage_media(), artifact_gc)

    assert first.run_id != second.run_id
    _assert_invariants(first, settings)
    _assert_invariants(second, settings)

    with session_factory() as s:
        total_segments = s.execute(select(func.count()).select_from(TranscriptSegment)).scalar_one()
        total_turns = s.execute(select(func.count()).select_from(DiarizationTurn)).scalar_one()
        total_artifacts = s.execute(
            select(func.count()).select_from(AudioArtifact).where(
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
            )
        ).scalar_one()
    assert total_segments == len(first.segments) + len(second.segments), "segment rows leaked"
    assert total_turns == len(first.turns) + len(second.turns), "diarization turn rows leaked"
    assert total_artifacts == 2, "expected exactly two preprocessed_audio artifacts across runs"
