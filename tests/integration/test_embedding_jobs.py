"""Embedding-index job lane against a real Postgres + pgvector (#121).

Exercises the whole PR1 spine — create → claim → resolve → embed → atomic
publish → staleness — with a deterministic fake embedder (no ONNX weights, so
these run in CI). The measured ONNX-vs-sentence-transformers equivalence lives in
``tests/parity/test_text_embedding.py``; here we prove the DB lifecycle:
generation allocation, whole-run atomic supersession, staleness detection, the
disabled/no-transcript refusals, the one-active-per-(run, space) slot, and that a
force-cancel fences the claim.
"""

import hashlib
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import set_correction
from voxint.config import Settings
from voxint.db.models import (
    TEXT_EMBEDDING_DIM,
    EmbeddingJob,
    EmbeddingJobStatus,
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentEmbedding,
    TranscriptSegment,
)
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE, TextEmbedder
from voxint.enrichment.embedding_jobs import (
    EmbeddingJobError,
    create_jobs,
    execute_job,
    request_cancel,
    runs_needing_embeddings,
)

from .conftest import seed_onboarded


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class FakeEmbedder(TextEmbedder):
    """Deterministic, weightless stand-in — one unit vector per distinct text.

    Bypasses the ONNX load entirely (no ``super().__init__``); ``count_tokens`` is
    a word count (fixtures stay well under the 128-token budget, so no split is
    triggered), and ``embed_texts`` seeds a per-text RNG so equal text always maps
    to the same finite unit vector."""

    def __init__(self) -> None:
        pass

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), TEXT_EMBEDDING_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(
                TEXT_EMBEDDING_DIM
            ).astype(np.float32)
            norm = float(np.linalg.norm(vector)) or 1.0
            out[i] = vector / norm
        return out


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    seed_onboarded(session_factory)
    with session_factory() as sess:
        yield sess


def _seed_run(session: Session, texts: list[tuple[str, str]]) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/embed/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    for index, (label, text) in enumerate(texts):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index) * 10.0,
                end_seconds=float(index) * 10.0 + 5.0,
                raw_text=text,
                diarization_label=label,
            )
        )
    session.commit()
    return run.id


def _rows(session: Session, run_id: uuid.UUID) -> list[SegmentEmbedding]:
    return list(
        session.execute(
            select(SegmentEmbedding)
            .where(SegmentEmbedding.pipeline_run_id == run_id)
            .order_by(SegmentEmbedding.generation, SegmentEmbedding.chunk_index)
        ).scalars()
    )


def test_execute_job_stores_embeddings(
    session_factory: sessionmaker[Session], session: Session
) -> None:
    settings = make_settings()
    run_id = _seed_run(
        session,
        [("S0", "we should discuss the merger next quarter"), ("S1", "agreed, let us plan")],
    )
    job, already = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert already is False and job is not None
    job_id = job.id

    execute_job(session_factory, job_id, settings=settings, embedder=FakeEmbedder())
    session.expire_all()  # the worker committed in its own session

    rows = _rows(session, run_id)
    assert len(rows) == 2  # two speakers → two reading paragraphs → two chunks
    assert {r.generation for r in rows} == {1}
    assert [r.chunk_index for r in rows] == [0, 1]
    assert rows[0].chunk_text.startswith("we should discuss the merger")
    assert all(r.embedding_space == EMBEDDING_SPACE for r in rows)
    assert all(len(list(r.embedding)) == TEXT_EMBEDDING_DIM for r in rows)
    assert all(r.text_rendering == "raw" for r in rows)

    finished = session.get(EmbeddingJob, job_id)
    assert finished is not None
    assert finished.status == EmbeddingJobStatus.SUCCEEDED.value
    assert finished.generation == 1
    assert finished.finished_at is not None
    # A freshly-indexed run is not stale.
    assert run_id not in runs_needing_embeddings(session)


def test_correction_restamps_a_new_generation_atomically(
    session_factory: sessionmaker[Session], session: Session
) -> None:
    settings = make_settings()
    run_id = _seed_run(session, [("S0", "the original wording here"), ("S1", "second speaker")])
    job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert job is not None
    execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())
    session.expire_all()
    assert {r.generation for r in _rows(session, run_id)} == {1}

    # An operator correction changes the resolved transcript → the run is stale.
    segment = session.execute(
        select(TranscriptSegment).where(
            TranscriptSegment.pipeline_run_id == run_id,
            TranscriptSegment.segment_index == 0,
        )
    ).scalar_one()
    set_correction(session, segment=segment, text="a corrected wording entirely")
    session.commit()
    assert run_id in runs_needing_embeddings(session)

    job2, already = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert already is False and job2 is not None
    execute_job(session_factory, job2.id, settings=settings, embedder=FakeEmbedder())

    session.expire_all()
    rows = _rows(session, run_id)
    # Whole-run atomic replace: only generation 2 survives, gen 1 is gone.
    assert {r.generation for r in rows} == {2}
    assert any("corrected wording" in r.chunk_text for r in rows)
    assert rows[0].text_rendering == "corrected"
    finished = session.get(EmbeddingJob, job2.id)
    assert finished is not None and finished.generation == 2
    assert run_id not in runs_needing_embeddings(session)


def test_create_jobs_refused_when_disabled(session: Session) -> None:
    # Row override closes the gate even though the env default is on.
    from voxint.app_settings import get_app_settings

    row = get_app_settings(session)
    assert row is not None
    row.semantic_index_enabled = False
    session.commit()
    run_id = _seed_run(session, [("S0", "anything at all")])
    with pytest.raises(EmbeddingJobError, match="semantic search is disabled"):
        create_jobs(session, pipeline_run_id=run_id, settings=make_settings())


def test_create_jobs_refused_without_transcript(session: Session) -> None:
    media = MediaItem(source_path=f"incoming/embed/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.commit()
    with pytest.raises(EmbeddingJobError, match="no transcript"):
        create_jobs(session, pipeline_run_id=run.id, settings=make_settings())


def test_one_active_job_per_run_space(session: Session) -> None:
    settings = make_settings()
    run_id = _seed_run(session, [("S0", "first"), ("S1", "second")])
    first, already = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert first is not None and already is False
    second, already2 = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert second is None and already2 is True


def test_force_cancel_fences_queued_job(
    session_factory: sessionmaker[Session], session: Session
) -> None:
    settings = make_settings()
    run_id = _seed_run(session, [("S0", "will be cancelled")])
    job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
    session.commit()
    assert job is not None
    assert request_cancel(session, job.id) is True
    session.commit()

    # The claim refuses a cancel-flagged row, so execution is a no-op.
    execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())
    session.expire_all()
    finished = session.get(EmbeddingJob, job.id)
    assert finished is not None
    assert finished.status == EmbeddingJobStatus.CANCELLED.value
    assert _rows(session, run_id) == []
