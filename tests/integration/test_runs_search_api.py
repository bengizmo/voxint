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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import and_, select, text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_runs_api import make_run
from voxint.adjudication.review_state import set_correction
from voxint.api.app import create_app
from voxint.api.runs_query import ReviewFilter, SearchFilters, list_runs
from voxint.config import Settings
from voxint.db.models import (
    RunStatus,
    SegmentReviewState,
    Speaker,
    TranscriptSegment,
)
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
    created_from: date | None = None,
    created_to: date | None = None,
    language: str | None = None,
    status: RunStatus | None = None,
    review: ReviewFilter | None = None,
    page_size: int = 50,
) -> list[uuid.UUID]:
    page = list_runs(
        session,
        status=status,
        review=review,
        cursor=None,
        page_size=page_size,
        filters=SearchFilters(
            q=q,
            speaker_id=speaker_id,
            source=source,
            created_from=created_from,
            created_to=created_to,
            language=language,
        ),
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


def _correct_segment(session: Session, run_id: uuid.UUID, index: int, text_: str) -> None:
    """Write an operator correction on one segment via the sanctioned writer."""
    segment = session.execute(
        select(TranscriptSegment).where(
            TranscriptSegment.pipeline_run_id == run_id,
            TranscriptSegment.segment_index == index,
        )
    ).scalar_one()
    set_correction(session, segment=segment, text=text_)
    session.flush()


class TestCorrectedTextSearch:
    """Operator corrections (#58, D3) are a third searchable rendering: never
    coalesced, so a term is findable in raw, enhanced, OR corrected."""

    def test_corrected_only_term_is_findable(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(
                session,
                segments=[(None, "the jessca hums", "the jessca hums")],
            )
            # ASR + enhancement both got the name wrong; only the operator's
            # correction contains "Jessica". A silent search miss here would be
            # the completeness lie the design note forbids.
            assert search(session, q="Jessica") == []
            _correct_segment(session, run, 0, "the Jessica hums")
            assert search(session, q="Jessica") == [run]
            # raw/enhanced renderings stay independently searchable.
            assert search(session, q="jessca") == [run]

    def test_reverting_a_correction_drops_it_from_search(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(session, segments=[(None, "manifld gauge", None)])
            _correct_segment(session, run, 0, "manifold gauge")
            assert search(session, q="manifold") == [run]
            # Clearing the correction (empty → NULL) removes it from the index.
            _correct_segment(session, run, 0, "")
            assert search(session, q="manifold") == []
            assert search(session, q="manifld") == [run]

    def test_snippet_prefers_corrected_rendering(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            run = make_run(
                session,
                segments=[(None, "raw krypton line", "enhanced krypton line")],
            )
            _correct_segment(session, run, 0, "corrected krypton line")
            page = list_runs(
                session,
                status=None,
                review=None,
                cursor=None,
                page_size=10,
                filters=SearchFilters(q="krypton"),
            )
            (item,) = [i for i in page.items if i.run_id == run]
            assert item.snippet is not None
            # Corrected takes top precedence among matched renderings.
            assert "corrected" in str(item.snippet.html)
            assert "<mark>krypton</mark>" in str(item.snippet.html)


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

    def test_date_bounds_are_utc_day_inclusive_exclusive(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            day = date(2026, 8, 10)
            at_from_midnight = make_run(
                session, created_at=datetime(2026, 8, 10, 0, 0, 0, tzinfo=UTC)
            )
            late_on_to_day = make_run(
                session, created_at=datetime(2026, 8, 12, 23, 59, 59, tzinfo=UTC)
            )
            before = make_run(
                session, created_at=datetime(2026, 8, 9, 23, 59, 59, tzinfo=UTC)
            )
            after = make_run(
                session, created_at=datetime(2026, 8, 13, 0, 0, 0, tzinfo=UTC)
            )
            hits = search(
                session, created_from=day, created_to=date(2026, 8, 12)
            )
            assert at_from_midnight in hits
            assert late_on_to_day in hits
            assert before not in hits
            assert after not in hits

    def test_review_composes_with_speaker_facet(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # COMPLETED run with unresolved label A + grounded assign on label B:
        # review=needed AND speaker=X must return it; review=resolved must not.
        with session_factory() as session:
            run = make_run(
                session,
                labels=["SPEAKER_00", "SPEAKER_01"],
                grounded=["SPEAKER_01"],
            )
            speaker_id = session.execute(
                text(
                    "SELECT speaker_id FROM speaker_assignments"
                    " WHERE pipeline_run_id = :r"
                ),
                {"r": run},
            ).scalar_one()
            assert search(
                session, speaker_id=speaker_id, review=ReviewFilter.NEEDED
            ) == [run]
            assert search(
                session, speaker_id=speaker_id, review=ReviewFilter.RESOLVED
            ) == []

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


class TestLanguageFacet:
    def test_filters_by_detected_language_and_null_rows(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            spanish = make_run(session, language="es", language_probability=0.92)
            english = make_run(session, language="en")
            # A NULL-language row: queued/failed/legacy — never transcribed.
            unrecorded = make_run(session, status=RunStatus.QUEUED)

            assert search(session, language="es") == [spanish]
            assert search(session, language="en") == [english]
            # Unfiltered, NULL rows still appear...
            assert set(search(session)) == {spanish, english, unrecorded}
            # ...and a specific code never matches them.
            assert unrecorded not in search(session, language="es")
            # A code no run carries yields an empty page, not an error.
            assert search(session, language="fr") == []

    def test_items_carry_language_for_the_column(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            tagged = make_run(session, language="es")
            untagged = make_run(session)
            page = list_runs(
                session,
                status=None,
                review=None,
                cursor=None,
                page_size=10,
                filters=None,
            )
            by_id = {item.run_id: item.language for item in page.items}
            assert by_id[tagged] == "es"
            assert by_id[untagged] is None

    def test_composes_with_status_and_search(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            match = make_run(
                session,
                language="es",
                status=RunStatus.COMPLETED,
                segments=[(None, "subcooling target", None)],
            )
            make_run(session, language="es", status=RunStatus.FAILED)
            make_run(
                session,
                language="en",
                status=RunStatus.COMPLETED,
                segments=[(None, "subcooling target", None)],
            )
            assert search(
                session, language="es", status=RunStatus.COMPLETED, q="subcooling"
            ) == [match]

    def test_keyset_pagination_with_active_language_filter(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            stamp = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)
            expected = [
                make_run(session, language="es", created_at=stamp)
                for _ in range(5)
            ]
            make_run(session, language="en", created_at=stamp)  # must not appear
            filters = SearchFilters(language="es")
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

    def test_searchable_languages_distinct_labeled_and_ordered(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from voxint.api.runs_query import searchable_languages

        with session_factory() as session:
            # Duplicates collapse; codes order by display label (Spanish < Ukrainian
            # < zz — an unmapped code labels as itself); NULLs are excluded.
            make_run(session, language="uk")
            make_run(session, language="es")
            make_run(session, language="es")
            make_run(session, language="zz")
            make_run(session)
            facets = searchable_languages(session)
            assert [(f.code, f.label) for f in facets] == [
                ("es", "Spanish (es)"),
                ("uk", "Ukrainian (uk)"),
                ("zz", "zz"),
            ]

    def test_searchable_languages_scoped_to_archive_view(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from datetime import datetime as dt

        from voxint.api.runs_query import searchable_languages
        from voxint.db.models import PipelineRun

        with session_factory() as session:
            make_run(session, language="es")
            archived_run = make_run(session, language="fr")
            run = session.get(PipelineRun, archived_run)
            assert run is not None
            run.archived_at = dt.now(tz=UTC)
            session.commit()
            # The dropdown offers exactly the languages the view can show: a
            # language living only on archived runs must not appear in the
            # active facet (it would filter to an empty page), and vice versa.
            active = [f.code for f in searchable_languages(session)]
            archived = [f.code for f in searchable_languages(session, archived=True)]
            assert active == ["es"]
            assert archived == ["fr"]

    def test_searchable_languages_include_keeps_active_filter_visible(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from voxint.api.runs_query import searchable_languages

        with session_factory() as session:
            make_run(session, language="es")
            # A stale or hand-typed ?language= the view lacks still renders as
            # the selected option, never as a silently active "all".
            # Label order: "French (fr)" sorts before "Spanish (es)".
            codes = [f.code for f in searchable_languages(session, include="fr")]
            assert codes == ["fr", "es"]
            # An include already present does not duplicate.
            codes = [f.code for f in searchable_languages(session, include="es")]
            assert codes == ["es"]


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

    def test_language_column_facet_and_null_render(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            make_run(session, language="es", language_probability=0.92)
            make_run(session)  # NULL language → em-dash cell
        page = client.get("/runs").text
        # Column header + labeled value, and the facet dropdown offers only
        # languages some run carries.
        # V3 grid-table: column headers are uppercase in gt-header spans.
        assert "LANGUAGE" in page
        assert "Spanish (es)" in page
        assert '<option value="es"' in page

    def test_language_filter_selected_and_preserved_in_links(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            for _ in range(3):  # page_size=2 → an Older link
                make_run(session, language="es")
            make_run(session, language="en")
        response = client.get("/runs", params={"language": "es"})
        assert response.status_code == 200
        page = response.text
        assert '<option value="es" selected>' in page
        # The Older link and the archive toggle both keep the facet.
        assert "language=es" in page

    def test_stale_language_filter_still_renders_selected(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            make_run(session, language="es")
        # No run carries "fr": the page is honestly empty AND the select shows
        # the active filter rather than a misleading "all".
        page = client.get("/runs", params={"language": "fr"}).text
        assert '<option value="fr" selected>' in page
        assert "No runs match these filters" in page

    def test_language_snippet_colspan_tracks_column_count(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with session_factory() as session:
            make_run(
                session,
                language="es",
                segments=[(None, "colspan probe text", None)],
            )
        page = client.get("/runs", params={"q": "probe"}).text
        # V3 grid-table: snippet rows span all columns via grid-column on the
        # child span, not a colspan attribute.
        assert "grid-column: 1 / -1" in page
        assert "probe" in page

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


@pytest.mark.parametrize(
    ("column_name", "index_name"),
    [
        ("raw_text", "transcript_segments_raw_fts_idx"),
        ("enhanced_text", "transcript_segments_enhanced_fts_idx"),
    ],
)
def test_fts_predicate_uses_expression_indexes(
    session_factory: sessionmaker[Session], column_name: str, index_name: str
) -> None:
    """Guard expression drift: the compiled predicates must be index-eligible."""
    with session_factory() as session:
        make_run(session, segments=[(None, "explainable condenser", "polished text")])
        session.execute(text("SET enable_seqscan = off"))
        tsq = ts_query("condenser")
        column = getattr(TranscriptSegment, column_name)
        predicate = ts_vector(column).bool_op("@@")(tsq)
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
        assert index_name in json.dumps(plan)


def test_corrected_fts_predicate_uses_partial_index(
    session_factory: sessionmaker[Session],
) -> None:
    """The corrected rendering (#58, D3) must reach its PARTIAL GIN index, not a
    seqscan — the whole point of the LEFT-JOIN-vs-EXISTS structuring in the run
    filter. The predicate's IS NOT NULL guard must mirror the index's partial
    WHERE, or the planner cannot use it (design note's flagged risk)."""
    with session_factory() as session:
        run = make_run(session, segments=[(None, "raw wording", None)])
        _correct_segment(session, run, 0, "explainable krypton condenser")
        session.execute(text("SET enable_seqscan = off"))
        tsq = ts_query("condenser")
        predicate = and_(
            SegmentReviewState.corrected_text.is_not(None),
            ts_vector(SegmentReviewState.corrected_text).bool_op("@@")(tsq),
        )
        compiled = predicate.compile(
            dialect=session.bind.dialect,  # type: ignore[union-attr]
            compile_kwargs={"literal_binds": True},
        )
        plan = session.execute(
            text(
                "EXPLAIN (FORMAT JSON) SELECT 1 FROM segment_review_states "
                f"WHERE {compiled}"
            )
        ).scalar_one()
        assert "segment_review_states_corrected_fts_idx" in json.dumps(plan)
