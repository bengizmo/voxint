"""The ``/search`` "Meaning" mode end to end against real Postgres + pgvector (#121).

Exercises the ranked passage search over the PR1 embedding index: the three-arm
fusion (vector + ``simple`` FTS + exact-quote), the jump-url shape, the escaped
marked snippet, the honest empty/off/unavailable/indexing states, archived-run
exclusion, a punctuation query that yields no lexical candidates while the vector
arm still returns, that a re-embed leaves only the new generation searchable
(Decision 3: the table holds exactly one generation per run), and that a read
straddling a concurrent publish never mixes two generations (the REPEATABLE READ
wrapper). The ranking arms run against real seeded ``segment_embeddings`` rows
produced by the real job flow with a deterministic weightless embedder, so these
run in CI without the ONNX weights.

The final golden test is the only one that needs the vendored MiniLM weights: it
seeds real embeddings and proves a non-English query retrieves the multilingual
merger passages end to end. It SKIPS in dev when the weights are absent and hard-
FAILS under ``VOXINT_PARITY_REQUIRED=1`` so a fully-skipped release can never
green-board.
"""

import hashlib
import os
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_embedding_jobs import FakeEmbedder, make_settings
from tests.integration.test_runs_api import make_run
from voxint.adjudication.review_state import set_correction
from voxint.api.app import create_app
from voxint.api.meaning_query import MeaningSearchState, search_passages
from voxint.app_settings import get_app_settings
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunStatus,
    SegmentEmbedding,
    TranscriptSegment,
)
from voxint.embeddings.onnx_embedder import EMBEDDING_SPACE
from voxint.enrichment.embedding_jobs import create_jobs, execute_job

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
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
    """Seed a run + transcript and drive the REAL job flow to store embeddings.

    Uses ``create_jobs`` + ``execute_job`` with the deterministic ``FakeEmbedder``
    (weightless), so the rows are genuine ``segment_embeddings`` written by the
    production producer, not hand-built fixtures.
    """
    settings = make_settings()
    with session_factory() as session:
        run_id = make_run(session, segments=list(segments), **run_kwargs)  # type: ignore[arg-type]
        job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
        session.commit()
    assert job is not None
    execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())
    return run_id


def _run_search(
    session_factory: sessionmaker[Session], query: str, **kwargs: object
):
    """Run a Meaning search with the deterministic embedder and default settings."""
    return search_passages(
        session_factory,
        settings=make_settings(),
        query=query,
        embedder=FakeEmbedder(),
        **kwargs,  # type: ignore[arg-type]
    )


class TestRankedPassages:
    def test_exact_text_query_ranks_its_passage_and_shapes_the_hit(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # FakeEmbedder maps identical text to the same unit vector, so querying a
        # chunk's exact text puts it at cosine distance 0 — the vector arm's top.
        run_id = _seed_indexed_run(
            session_factory, [("S0", "the quarterly merger discussion", None)]
        )
        page = _run_search(session_factory, "the quarterly merger discussion")
        assert page.state is MeaningSearchState.OK
        assert page.items, "the seeded passage must surface"
        top = page.items[0]
        assert top.run_id == run_id
        assert top.jump_url == f"/runs/{run_id}/transcript?t=0"
        assert "merger" in str(top.snippet)

    def test_second_speaker_passage_carries_its_start_time(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _seed_indexed_run(
            session_factory,
            [("S0", "opening remarks about nothing", None), ("S1", "closing budget vote", None)],
        )
        page = _run_search(session_factory, "closing budget vote")
        assert page.state is MeaningSearchState.OK
        top = page.items[0]
        assert top.run_id == run_id
        # The second speaker's paragraph begins at index 1 -> start 10s.
        assert top.jump_url == f"/runs/{run_id}/transcript?t=10"

    def test_exact_quote_floats_above_a_closer_vector_hit(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # One run literally contains the quoted phrase; another is the exact
        # vector match for the surrounding words. The literal must float to the top.
        quoted = _seed_indexed_run(
            session_factory, [("S0", "before the pivotal handshake moment arrived", None)]
        )
        _seed_indexed_run(session_factory, [("S0", "an unrelated closing summary", None)])
        page = _run_search(session_factory, '"pivotal handshake"')
        assert page.state is MeaningSearchState.OK
        assert page.items[0].run_id == quoted
        assert page.items[0].exact_quote is True
        assert "<mark>pivotal handshake</mark>" in str(page.items[0].snippet)

    def test_hostile_transcript_snippet_is_escaped(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        _seed_indexed_run(
            session_factory,
            [("S0", "<script>alert('xss')</script> capacitor discharge", None)],
        )
        page = _run_search(session_factory, '"capacitor discharge"')
        assert page.state is MeaningSearchState.OK
        assert page.items
        assert "<script>alert" not in str(page.items[0].snippet)


class TestHonestStates:
    def test_empty_query_via_route(self, client: TestClient) -> None:
        resp = client.get("/search")
        assert resp.status_code == 200
        assert "Type what you are looking for" in resp.text
        # The Exact/Meaning toggle renders on the search page too.
        assert 'aria-label="Search mode"' in resp.text

    def test_off_state_when_flag_disabled_via_route(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            row = get_app_settings(session)
            assert row is not None
            row.semantic_index_enabled = False
            session.commit()
        resp = client.get("/search", params={"q": "anything"})
        assert resp.status_code == 200
        assert "Semantic search is turned off" in resp.text

    def test_unavailable_state_when_weights_absent_via_route(
        self, client: TestClient
    ) -> None:
        # The route uses the real in-process embedder; this env has no weights, so
        # a query it cannot embed reports UNAVAILABLE honestly rather than 500ing.
        resp = client.get("/search", params={"q": "anything"})
        assert resp.status_code == 200
        assert "Semantic search is unavailable" in resp.text

    def test_indexing_state_when_enabled_but_nothing_indexed(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        seed_onboarded(session_factory)
        page = _run_search(session_factory, "anything at all")
        assert page.state is MeaningSearchState.INDEXING
        assert page.items == []


class TestRunsToggle:
    def test_runs_shows_the_meaning_toggle_carrying_the_query(
        self, client: TestClient
    ) -> None:
        # PR2 only ADDS the Exact/Meaning strip to /runs; the chronological browse
        # is otherwise untouched. Exact is the current tab; Meaning links to
        # /search and both carry the operator's query across.
        resp = client.get("/runs", params={"q": "compressor"})
        assert resp.status_code == 200
        assert 'aria-label="Search mode"' in resp.text
        assert 'href="/runs?q=compressor" aria-current="page"' in resp.text
        assert 'href="/search?q=compressor"' in resp.text

    def test_runs_toggle_omits_query_when_absent(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 200
        # No stray "?q=" when the operator has not searched.
        assert 'href="/search">Meaning</a>' in resp.text


class TestCorpusScoping:
    def test_archived_run_is_excluded(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        live = _seed_indexed_run(session_factory, [("S0", "shared subcooling reading", None)])
        archived = _seed_indexed_run(
            session_factory, [("S0", "shared subcooling reading", None)]
        )
        with session_factory() as session:
            run = session.get(PipelineRun, archived)
            assert run is not None
            run.archived_at = datetime.now(tz=UTC)
            session.commit()
        page = _run_search(session_factory, "shared subcooling reading")
        run_ids = {it.run_id for it in page.items}
        assert live in run_ids
        assert archived not in run_ids

    def test_punctuation_query_has_no_lexical_hits_but_vector_still_returns(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # "!!! ??? ..." makes an empty simple-config tsquery, so the mandatory @@
        # predicate drops the lexical arm entirely; the vector arm must still carry
        # the query so search never goes dark on non-word input.
        run_id = _seed_indexed_run(session_factory, [("S0", "a perfectly ordinary line", None)])
        page = _run_search(session_factory, "!!! ??? ...")
        assert page.state is MeaningSearchState.OK
        assert {it.run_id for it in page.items} == {run_id}


class TestGenerationConsistency:
    def test_reembed_leaves_only_the_new_generation_searchable(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # Decision 3: publish deletes every prior generation in the same committed
        # txn, so search (reading segment_embeddings directly) sees only the newest.
        settings = make_settings()
        with session_factory() as session:
            run_id = make_run(session, segments=[("S0", "the original wording here", None)])
            job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
            session.commit()
        assert job is not None
        execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())

        # Correct the transcript and re-embed -> a new generation supersedes gen 1.
        with session_factory() as session:
            segment = session.execute(
                select(TranscriptSegment).where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
            ).scalar_one()
            set_correction(session, segment=segment, text="a corrected wording entirely")
            session.commit()
            job2, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
            session.commit()
        assert job2 is not None
        execute_job(session_factory, job2.id, settings=settings, embedder=FakeEmbedder())

        # The old wording is gone from the index; only the corrected passage answers.
        stale = _run_search(session_factory, "the original wording here")
        assert all("original wording" not in str(it.snippet) for it in stale.items)
        fresh = _run_search(session_factory, "a corrected wording entirely")
        assert fresh.state is MeaningSearchState.OK
        assert fresh.items
        assert "corrected wording" in str(fresh.items[0].snippet)

    def test_read_straddling_a_publish_never_mixes_generations(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # The REPEATABLE READ wrapper's job: a publish that commits between two
        # arms must not let one arm see gen-1 rows and another gen-2. We run a
        # concurrent re-embed against a tight search loop and assert every single
        # result set is internally from ONE generation (never an alpha/beta mix).
        settings = make_settings()
        with session_factory() as session:
            run_id = make_run(
                session,
                segments=[("S0", "alpha one passage", None), ("S1", "alpha two passage", None)],
            )
            job, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
            session.commit()
        assert job is not None
        execute_job(session_factory, job.id, settings=settings, embedder=FakeEmbedder())

        # Stage a gen-2 that replaces every chunk's wording with "beta".
        with session_factory() as session:
            for index in (0, 1):
                segment = session.execute(
                    select(TranscriptSegment).where(
                        TranscriptSegment.pipeline_run_id == run_id,
                        TranscriptSegment.segment_index == index,
                    )
                ).scalar_one()
                set_correction(session, segment=segment, text=f"beta {index} passage")
            session.commit()
            job2, _ = create_jobs(session, pipeline_run_id=run_id, settings=settings)
            session.commit()
        assert job2 is not None

        barrier = threading.Barrier(2)
        mixed: list[str] = []

        def publisher() -> None:
            barrier.wait()
            execute_job(session_factory, job2.id, settings=settings, embedder=FakeEmbedder())

        def searcher() -> None:
            barrier.wait()
            for _ in range(40):
                page = _run_search(session_factory, "passage")
                texts = " ".join(str(it.snippet) for it in page.items)
                if "alpha" in texts and "beta" in texts:
                    mixed.append(texts)

        threads = [threading.Thread(target=publisher), threading.Thread(target=searcher)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert mixed == [], f"a read mixed two generations: {mixed[:1]}"

        # And the end state is a clean gen-2-only index.
        final = _run_search(session_factory, "passage")
        assert final.state is MeaningSearchState.OK
        assert all("alpha" not in str(it.snippet) for it in final.items)


# --- Golden retrieval: real weights, gated like the parity suite ---------------

_REPO = Path(__file__).resolve().parents[2]
_ONNX_PATH = Path(
    os.getenv("VOXINT_MINILM_ONNX_PATH") or (_REPO / "vendor" / "minilm" / "model.onnx")
)
_TOKENIZER_PATH = Path(
    os.getenv("VOXINT_MINILM_TOKENIZER_PATH")
    or (_REPO / "vendor" / "minilm" / "tokenizer.json")
)
_REQUIRED = os.getenv("VOXINT_PARITY_REQUIRED") == "1"
_GOLDEN_PREREQS = [
    (_ONNX_PATH.exists(), f"{_ONNX_PATH} missing — fetch the minilm-onnx-v1 asset"),
    (_TOKENIZER_PATH.exists(), f"{_TOKENIZER_PATH} missing — fetch the minilm-onnx-v1 asset"),
]


def _real_embedder():
    from voxint.embeddings.onnx_embedder import TextEmbedder

    return TextEmbedder(str(_ONNX_PATH), str(_TOKENIZER_PATH))


def _seed_real_passage(
    session_factory: sessionmaker[Session], text: str, vector: list[float]
) -> uuid.UUID:
    """Insert one run + a single real-vector ``segment_embeddings`` row directly.

    Bypasses paragraph chunking so the golden test controls the exact indexed
    string and its embedding, mirroring the parity fixtures one passage per run.
    """
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/golden/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        session.add(
            SegmentEmbedding(
                pipeline_run_id=run.id,
                embedding_space=EMBEDDING_SPACE,
                generation=1,
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                speaker_label="S0",
                text_rendering="raw",
                chunk_text=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                embedding=vector,
            )
        )
        session.commit()
        return run.id


if _REQUIRED:
    _missing = [reason for ok, reason in _GOLDEN_PREREQS if not ok]
    if _missing:
        pytest.fail(f"VOXINT_PARITY_REQUIRED=1 but golden prerequisites missing: {_missing}")


@pytest.mark.skipif(
    not all(ok for ok, _ in _GOLDEN_PREREQS),
    reason="MiniLM weights absent (fetch the minilm-onnx-v1 asset)",
)
def test_golden_non_english_query_retrieves_multilingual_merger_passage(
    session_factory: sessionmaker[Session],
) -> None:
    """A non-English merger query ranks the multilingual merger passages above lunch.

    Proves the whole retrieval stack with the real MiniLM embeddings: the query is
    embedded by the vendored ONNX graph, the vector arm ranks by real cosine, and
    the cross-lingual merger cluster beats the unrelated lunch passage. TEXT below
    spans English/Spanish/French/German/Chinese; the query is Spanish.
    """
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    seed_onboarded(session_factory)
    embedder = _real_embedder()

    corpus = {
        "en-merger": "where do they discuss the merger",
        "es-fusion": "¿dónde hablan de la fusión de empresas?",
        "fr-fusion": "où discutent-ils de la fusion des sociétés",
        "de-uebernahme": "die Firmenübernahme wurde letztes Quartal abgeschlossen",
        "zh-merger": "会议讨论了公司合并的问题",
        "en-lunch": "let's talk about lunch tomorrow",
    }
    vectors = embedder.embed_texts(list(corpus.values()))
    run_by_text = {
        text: _seed_real_passage(session_factory, text, vectors[i].tolist())
        for i, text in enumerate(corpus.values())
    }
    lunch_run = run_by_text[corpus["en-lunch"]]

    page = search_passages(
        session_factory,
        settings=make_settings(),
        query="¿dónde discuten la fusión de las dos empresas?",
        embedder=embedder,
    )
    assert page.state is MeaningSearchState.OK
    assert page.items
    ordered = [it.run_id for it in page.items]
    # The top hit is a merger passage, and lunch ranks strictly below every
    # merger passage that surfaced (cross-lingual retrieval, not word overlap).
    assert ordered[0] != lunch_run
    merger_runs = {run_by_text[t] for k, t in corpus.items() if k != "en-lunch"}
    top_merger = next(rid for rid in ordered if rid in merger_runs)
    assert ordered.index(top_merger) < ordered.index(lunch_run)
