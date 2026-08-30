""""More like this passage" + the semantic meaning map end to end (#357).

Both features read the PR1 embedding index, so both are exercised over real
seeded ``segment_embeddings`` rows produced by the real job flow with the
deterministic weightless ``FakeEmbedder`` — CI needs no ONNX weights.

Similar: the source paragraph (and its overlap span) is excluded, other
passages from the same recording survive, archived runs are hidden, and the
honest not-found/indexing states render. The short-segment fixtures also pin
the covering-chunk fallback: below the token floor the stored paragraph
vector stands in and the endpoint works with no embedder at all.

Map: compute-on-read writes one ``semantic_layout`` artifact, a re-read is
served from cache, archiving a run invalidates the fingerprint, and the
insufficient-corpus state is honest rather than a five-dot decoration.
"""

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_embedding_jobs import FakeEmbedder, make_settings
from tests.integration.test_runs_api import make_run
from voxint.api.app import create_app
from voxint.api.semantic_layout import MIN_POINTS, semantic_layout
from voxint.api.similar_query import SimilarSearchState, similar_passages
from voxint.config import Settings
from voxint.db.models import (
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    PipelineRun,
    TranscriptSegment,
)
from voxint.enrichment.embedding_jobs import create_jobs, execute_job

CREDS = ("reviewer", "s3cret")

# ≥ MIN_QUERY_TOKENS words, so the re-embed path runs (not the chunk fallback).
_LONG_A = (
    "the council spent the whole evening arguing about the proposed rezoning of"
    " the waterfront industrial district and its parking implications"
)
_LONG_B = (
    "residents kept returning to the waterfront rezoning debate and what the"
    " industrial district change would mean for parking in the area"
)
_LONG_OTHER = (
    "an entirely different conversation about the school music program funding"
    " gap and the spring concert scheduling conflict this year"
)


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: object) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,  # type: ignore[arg-type]
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _seed_indexed_run(
    session_factory: sessionmaker[Session],
    segments: Iterable[tuple[str | None, str, str | None]],
    **run_kwargs: object,
) -> uuid.UUID:
    settings = make_settings()
    with session_factory() as session:
        run_id = make_run(session, segments=list(segments), **run_kwargs)  # type: ignore[arg-type]
        job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
        session.commit()
    assert job is not None
    execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())
    return run_id


def _segment_id(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, index: int = 0
) -> uuid.UUID:
    with session_factory() as session:
        seg = session.execute(
            select(TranscriptSegment.id)
            .where(
                TranscriptSegment.pipeline_run_id == run_id,
                TranscriptSegment.segment_index == index,
            )
        ).scalar_one()
    return seg


def _similar(
    session_factory: sessionmaker[Session], segment_id: uuid.UUID, **kwargs: object
):
    return similar_passages(
        session_factory,
        settings=make_settings(),
        segment_id=segment_id,
        embedder=FakeEmbedder(),
        **kwargs,  # type: ignore[arg-type]
    )


def _archive(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> None:
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.archived_at = datetime.now(tz=UTC)
        session.commit()


class TestSimilarPassages:
    def test_identical_passage_in_another_run_is_the_top_hit(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        source = _seed_indexed_run(session_factory, [("S0", _LONG_A, None)])
        twin = _seed_indexed_run(session_factory, [("S0", _LONG_A, None)])
        _seed_indexed_run(session_factory, [("S0", _LONG_OTHER, None)])
        page = _similar(session_factory, _segment_id(session_factory, source))
        assert page.state is SimilarSearchState.OK
        assert page.items, "the identical passage must surface"
        top = page.items[0]
        assert top.run_id == twin
        assert top.jump_url == f"/runs/{twin}/transcript?t=0"
        assert "council" in top.preview or "rezoning" in top.preview

    def test_source_paragraph_is_never_a_result(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        source = _seed_indexed_run(session_factory, [("S0", _LONG_A, None)])
        page = _similar(session_factory, _segment_id(session_factory, source))
        assert page.state is SimilarSearchState.OK
        assert all(item.run_id != source for item in page.items)

    def test_same_run_non_overlapping_passage_survives(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # Two speakers -> two paragraphs at different times in ONE recording.
        # The second paragraph is a legitimate "elsewhere in this interview" hit.
        source = _seed_indexed_run(
            session_factory, [("S0", _LONG_A, None), ("S1", _LONG_B, None)]
        )
        page = _similar(session_factory, _segment_id(session_factory, source, index=0))
        assert page.state is SimilarSearchState.OK
        assert any(item.run_id == source for item in page.items)

    def test_archived_run_is_excluded(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        source = _seed_indexed_run(session_factory, [("S0", _LONG_A, None)])
        twin = _seed_indexed_run(session_factory, [("S0", _LONG_A, None)])
        _archive(session_factory, twin)
        # Keep the index non-empty so the state stays OK, not INDEXING.
        _seed_indexed_run(session_factory, [("S0", _LONG_OTHER, None)])
        page = _similar(session_factory, _segment_id(session_factory, source))
        assert page.state is SimilarSearchState.OK
        assert all(item.run_id != twin for item in page.items)

    def test_unknown_segment_reports_not_found(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        page = _similar(session_factory, uuid.uuid4())
        assert page.state is SimilarSearchState.NOT_FOUND

    def test_empty_index_reports_indexing(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run_id = make_run(session, segments=[("S0", _LONG_A, None)])
            session.commit()
        seed_onboarded(session_factory)
        page = _similar(session_factory, _segment_id(session_factory, run_id))
        assert page.state is SimilarSearchState.INDEXING

    def test_short_segment_uses_covering_chunk_without_an_embedder(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # Below the token floor the stored paragraph vector stands in; passing
        # embedder=None proves the fallback never touches ONNX loading.
        source = _seed_indexed_run(session_factory, [("S0", "a very short remark", None)])
        twin = _seed_indexed_run(session_factory, [("S0", "a very short remark", None)])
        page = similar_passages(
            session_factory,
            settings=make_settings(),
            segment_id=_segment_id(session_factory, source),
        )
        assert page.state is SimilarSearchState.OK
        assert any(item.run_id == twin for item in page.items)


class TestSimilarEndpoint:
    def test_endpoint_returns_shaped_json(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        # Short segments ride the covering-chunk fallback, so the full HTTP
        # path works in CI with no MiniLM weights present.
        source = _seed_indexed_run(session_factory, [("S0", "a very short remark", None)])
        twin = _seed_indexed_run(session_factory, [("S0", "a very short remark", None)])
        seg = _segment_id(session_factory, source)
        resp = client.get(f"/explore/segments/{seg}/similar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "ok"
        assert body["items"], "the twin passage must surface"
        item = body["items"][0]
        assert item["run_id"] == str(twin)
        assert set(item) == {
            "run_id",
            "title",
            "speaker_label",
            "start_seconds",
            "end_seconds",
            "preview",
            "jump_url",
        }

    def test_endpoint_unknown_segment(self, client: TestClient) -> None:
        resp = client.get(f"/explore/segments/{uuid.uuid4()}/similar")
        assert resp.status_code == 200
        assert resp.json() == {"state": "not_found", "items": []}

    def test_endpoint_requires_auth(
        self, client: TestClient
    ) -> None:
        resp = client.get(
            f"/explore/segments/{uuid.uuid4()}/similar", auth=("wrong", "creds")
        )
        assert resp.status_code == 401


def _map_artifacts(session: Session) -> list[CorpusAnalysisArtifact]:
    return list(
        session.execute(
            select(CorpusAnalysisArtifact).where(
                CorpusAnalysisArtifact.artifact_kind
                == CorpusAnalysisArtifactKind.SEMANTIC_LAYOUT.value
            )
        ).scalars()
    )


def _seed_map_corpus(session_factory: sessionmaker[Session]) -> list[uuid.UUID]:
    """Four runs / eight paragraphs — past MIN_POINTS even minus one run.

    Texts are distinct per run: FakeEmbedder maps identical text to one vector,
    and a corpus of two distinct vectors is rank-one — which pca_2d correctly
    refuses. The map fixture must span more than a line.
    """
    return [
        _seed_indexed_run(
            session_factory,
            [
                ("S0", f"discussion number {i} of the harbor budget line", None),
                ("S1", f"commentary number {i} on the harbor budget vote", None),
            ],
        )
        for i in range(4)
    ]


class TestSemanticLayout:
    def test_insufficient_corpus_is_honest(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        assert MIN_POINTS > 1
        _seed_indexed_run(session_factory, [("S0", "one lonely paragraph", None)])
        with session_factory() as session:
            result = semantic_layout(session, make_settings())
            session.commit()
        assert result.state == "insufficient"
        with session_factory() as session:
            assert _map_artifacts(session) == []

    def test_compute_on_read_writes_one_artifact_and_caches(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_ids = _seed_map_corpus(session_factory)
        with session_factory() as session:
            result = semantic_layout(session, make_settings())
            session.commit()
        assert result.state == "ok"
        payload = result.payload
        assert payload["method"] == "pca"
        assert payload["shown_n"] == len(payload["points"]) == 8
        assert payload["total_n"] == 8
        assert payload["sampled"] is False
        point = payload["points"][0]
        assert set(point) == {
            "x",
            "y",
            "run_id",
            "media_title",
            "speaker_label",
            "start_seconds",
            "end_seconds",
            "preview",
            "jump_url",
        }
        assert {p["run_id"] for p in payload["points"]} == {str(r) for r in run_ids}

        with session_factory() as session:
            artifacts = _map_artifacts(session)
            assert len(artifacts) == 1
            cached = semantic_layout(session, make_settings())
            session.commit()
        assert cached.state == "ok"
        assert cached.payload == payload
        with session_factory() as session:
            assert len(_map_artifacts(session)) == 1

    def test_archiving_a_run_invalidates_the_layout(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_ids = _seed_map_corpus(session_factory)
        with session_factory() as session:
            first = semantic_layout(session, make_settings())
            session.commit()
        assert first.payload["shown_n"] == 8

        _archive(session_factory, run_ids[0])
        with session_factory() as session:
            second = semantic_layout(session, make_settings())
            session.commit()
        assert second.state == "ok"
        assert second.payload["shown_n"] == 6
        assert str(run_ids[0]) not in {p["run_id"] for p in second.payload["points"]}

    def test_endpoint_serves_the_map(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        _seed_map_corpus(session_factory)
        resp = client.get("/explore/meaning-map")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "ok"
        assert len(body["points"]) == 8

    def test_endpoint_insufficient_state(self, client: TestClient) -> None:
        resp = client.get("/explore/meaning-map")
        assert resp.status_code == 200
        assert resp.json() == {"state": "insufficient"}
