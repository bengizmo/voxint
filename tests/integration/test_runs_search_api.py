"""The /runs search facets end to end against real Postgres.

Covers the FTS predicate semantics (both text variants searchable — the
anti-coalesce guarantees), websearch syntax, the speaker facet through
list_runs, facet composition with status/review, keyset pagination stability
under an active search, snippet selection/rendering, route-level parsing and
escaping, and an EXPLAIN assertion that the compiled predicate can use the
0008 expression indexes.
"""

import html
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_runs_api import make_run
from voxint.api.app import create_app
from voxint.api.runs_query import SearchFilters, list_runs
from voxint.config import Settings
from voxint.db.models import RunStatus, Speaker, TranscriptSegment
from voxint.db.search import ts_query, ts_vector

CREDS = ("reviewer", "s3cret")


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        runs_page_size=2,
    )
    test_client = TestClient(
        create_app(settings=settings, session_factory=session_factory)
    )
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def search(
    session: Session,
    *,
    q: str | None = None,
    speaker_id: uuid.UUID | None = None,
    source: str | None = None,
    status: RunStatus | None = None,
    page_size: int = 50,
) -> list[uuid.UUID]:
    page = list_runs(
        session,
        status=status,
        review=None,
        cursor=None,
        page_size=page_size,
        filters=SearchFilters(q=q, speaker_id=speaker_id, source=source),
    )
    return [item.run_id for item in page.items]


class TestTranscriptSearch:
    def test_both_variants_stay_searchable(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(
                session,
                segments=[(None, "the compresser hums", "the compressor hums")],
            )
            # Raw-only segment on another run: never enhanced, still findable.
            raw_only = make_run(session, segments=[(None, "manifold gauge", None)])

            # The raw misrendering survives enhancement...
            assert search(session, q="compresser") == [run]
            # ...and the enhanced correction is findable too.
            assert search(session, q="compressor") == [run]
            assert search(session, q="manifold") == [raw_only]

    def test_partial_enhancement_within_one_run(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(
                session,
                segments=[
                    (None, "first batch enhanced", "first batch polished"),
                    (None, "second batch skipped by breaker", None),
                ],
            )
            assert search(session, q="polished") == [run]
            assert search(session, q="breaker") == [run]

    def test_websearch_syntax(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            heat = make_run(session, segments=[(None, "heat pump defrost cycle", None)])
            furnace = make_run(session, segments=[(None, "furnace ignition", None)])

            assert search(session, q='"heat pump"') == [heat]
            assert search(session, q="ignition -defrost") == [furnace]
            assert set(search(session, q="defrost OR ignition")) == {heat, furnace}
            # case-insensitive, and 'english' stemming: plural finds singular
            assert search(session, q="FURNACES") == [furnace]

    def test_stopword_only_query_matches_nothing(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            make_run(session, segments=[(None, "the and of", None)])
            assert search(session, q="the and of") == []

    def test_terms_split_across_segments_do_not_match(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # The search document is one segment, documented and pinned.
        with session_factory() as session:
            make_run(
                session,
                segments=[(None, "txv only here", None), (None, "orifice only here", None)],
            )
            assert search(session, q="txv orifice") == []


class TestSpeakerAndCompositionFacets:
    def test_speaker_facet_via_list_runs(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            attributed = make_run(session, labels=["SPEAKER_00"], grounded=["SPEAKER_00"])
            make_run(session, labels=["SPEAKER_00"])
            speaker_id = session.execute(
                text(
                    "SELECT speaker_id FROM speaker_assignments"
                    " WHERE pipeline_run_id = :r"
                ),
                {"r": attributed},
            ).scalar_one()
            assert search(session, speaker_id=speaker_id) == [attributed]

    def test_facets_compose_with_status_and_each_other(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            match = make_run(
                session,
                labels=["SPEAKER_00"],
                grounded=["SPEAKER_00"],
                segments=[(None, "subcooling reading", None)],
            )
            # Same text, different (failed) status.
            make_run(
                session,
                status=RunStatus.FAILED,
                segments=[(None, "subcooling reading", None)],
            )
            # Same speaker evidence, different text.
            make_run(session, labels=["SPEAKER_00"], grounded=["SPEAKER_00"])
            speaker_id = session.execute(
                text(
                    "SELECT speaker_id FROM speaker_assignments"
                    " WHERE pipeline_run_id = :r"
                ),
                {"r": match},
            ).scalar_one()
            assert search(
                session,
                q="subcooling",
                speaker_id=speaker_id,
                status=RunStatus.COMPLETED,
            ) == [match]

    def test_source_facet_is_literal_substring(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            a = make_run(session)
            make_run(session)
            source_path = session.execute(
                text(
                    "SELECT source_path FROM media_items m"
                    " JOIN pipeline_runs r ON r.media_item_id = m.id WHERE r.id = :r"
                ),
                {"r": a},
            ).scalar_one()
            fragment = source_path.removeprefix("incoming/")[:12]
            assert search(session, source=fragment) == [a]
            # LIKE metacharacters are literal: a '%' matches nothing rather
            # than everything.
            assert search(session, source="%") == []


class TestKeysetUnderSearch:
    def test_walk_with_identical_timestamps(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            stamp = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)
            expected = []
            for index in range(5):
                expected.append(
                    make_run(
                        session,
                        created_at=stamp if index < 3 else stamp - timedelta(hours=1),
                        segments=[(None, "walkable superheat", None)],
                    )
                )
            # Non-matching neighbors sharing the timestamps must not appear.
            make_run(session, created_at=stamp)

            filters = SearchFilters(q="superheat")
            seen: list[uuid.UUID] = []
            cursor = None
            while True:
                page = list_runs(
                    session,
                    status=None,
                    review=None,
                    cursor=cursor,
                    page_size=2,
                    filters=filters,
                )
                seen.extend(item.run_id for item in page.items)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            assert len(seen) == len(set(seen)) == 5
            assert set(seen) == set(expected)


class TestSnippets:
    def test_first_matching_segment_and_variant_choice(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(
                session,
                segments=[
                    (None, "no match here", None),
                    (None, "raw txv wording", "enhanced txv wording"),
                    (None, "txv appears again later", None),
                ],
            )
            page = list_runs(
                session,
                status=None,
                review=None,
                cursor=None,
                page_size=10,
                filters=SearchFilters(q="txv"),
            )
            (item,) = [i for i in page.items if i.run_id == run]
            assert item.snippet is not None
            # First matching segment (index 1) → its start_seconds, and the
            # enhanced variant is preferred when it matches too.
            assert item.snippet.start_seconds == 10.0
            assert "enhanced" in str(item.snippet.html)
            assert "<mark>txv</mark>" in str(item.snippet.html)

    def test_no_snippet_without_q(self, session_factory: sessionmaker[Session]) -> None:
        with session_factory() as session:
            make_run(session, segments=[(None, "anything", None)])
            page = list_runs(
                session,
                status=None,
                review=None,
                cursor=None,
                page_size=10,
                filters=SearchFilters(),
            )
            assert all(item.snippet is None for item in page.items)


class TestRoute:
    def test_search_renders_snippet_and_preserves_facets_in_older_link(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            for index in range(3):
                make_run(
                    session,
                    segments=[(None, f"reversing valve number {index}", None)],
                )
        response = client.get("/runs", params={"q": "reversing"})
        assert response.status_code == 200
        assert "<mark>reversing</mark>" in response.text
        # page_size=2 → an Older link that must carry the query.
        assert "q=reversing" in response.text

    def test_hostile_transcript_never_reaches_page_unescaped(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            make_run(
                session,
                segments=[(None, "<script>alert('xss')</script> capacitor", None)],
            )
        response = client.get("/runs", params={"q": "capacitor"})
        assert response.status_code == 200
        assert "<script>alert" not in response.text

    def test_invalid_speaker_and_dates_are_422(self, client: TestClient) -> None:
        assert client.get("/runs", params={"speaker": "nope"}).status_code == 422
        assert client.get("/runs", params={"created_from": "13/01"}).status_code == 422
        assert client.get("/runs", params={"created_to": "nope"}).status_code == 422

    def test_facet_dropdown_lists_archived_marked_and_hides_tombstones(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            now = datetime.now(tz=UTC)
            active = Speaker(display_name="Active Annie")
            archived = Speaker(display_name="Archived Arch", deleted_at=now)
            session.add_all([active, archived])
            session.flush()
            tombstone = Speaker(
                display_name="Merged Away", merged_into_id=active.id, merged_at=now
            )
            session.add(tombstone)
            session.commit()
        page = client.get("/runs").text
        assert "Active Annie" in page
        assert "Archived Arch (archived)" in html.unescape(page)
        assert "Merged Away" not in page


def test_fts_predicate_uses_expression_indexes(
    session_factory: sessionmaker[Session],
) -> None:
    """Guard expression drift: the compiled predicate must be index-eligible."""
    with session_factory() as session:
        make_run(session, segments=[(None, "explainable condenser", "polished text")])
        session.execute(text("SET enable_seqscan = off"))
        tsq = ts_query("condenser")
        predicate = ts_vector(TranscriptSegment.raw_text).bool_op("@@")(tsq)
        compiled = predicate.compile(
            dialect=session.bind.dialect,  # type: ignore[union-attr]
            compile_kwargs={"literal_binds": True},
        )
        plan = session.execute(
            text(
                "EXPLAIN (FORMAT JSON) SELECT 1 FROM transcript_segments "
                f"WHERE {compiled}"
            )
        ).scalar_one()
        assert "transcript_segments_raw_fts_idx" in json.dumps(plan)
