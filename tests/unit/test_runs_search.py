"""Pure logic of the /runs search facets: parsing, URLs, headline rendering."""

import uuid
from datetime import UTC, date, datetime
from urllib.parse import urlencode

import pytest
from markupsafe import Markup

from voxint.api.runs_query import (
    Cursor,
    ReviewFilter,
    SearchFilters,
    _escape_like,
    _render_headline,
    parse_search_filters,
    runs_url,
)
from voxint.db.models import RunStatus


def parse(**overrides: str | None) -> SearchFilters:
    params: dict[str, str | None] = {
        "q": None,
        "speaker": None,
        "source": None,
        "created_from": None,
        "created_to": None,
    }
    params.update(overrides)
    return parse_search_filters(**params)  # type: ignore[arg-type]


class TestParseSearchFilters:
    def test_blank_and_absent_mean_off(self) -> None:
        assert parse() == SearchFilters()
        assert parse(
            q="", speaker="", source="", created_from="", created_to="", language=""
        ) == (SearchFilters())
        assert not parse().active()

    def test_values_parse(self) -> None:
        speaker = uuid.uuid4()
        filters = parse(
            q="compressor -brand",
            speaker=str(speaker),
            source="incoming/",
            created_from="2026-08-01",
            created_to="2026-08-14",
            language="es",
        )
        assert filters == SearchFilters(
            q="compressor -brand",
            speaker_id=speaker,
            source="incoming/",
            created_from=date(2026, 8, 1),
            created_to=date(2026, 8, 14),
            language="es",
        )
        assert filters.active()

    def test_language_alone_is_active(self) -> None:
        filters = parse(language="es")
        assert filters == SearchFilters(language="es")
        assert filters.active()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"speaker": "not-a-uuid"},
            {"created_from": "2026-13-01"},
            {"created_to": "yesterday"},
            # date.max would overflow the exclusive +1-day upper bound with an
            # ArithmeticError the route's ValueError→422 mapping misses.
            {"created_to": "9999-12-31"},
            {"created_from": "9999-12-31"},
        ],
    )
    def test_invalid_values_raise(self, overrides: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="invalid"):
            parse(**overrides)

    def test_whitespace_only_q_means_off(self) -> None:
        assert parse(q="   ") == SearchFilters()
        assert parse(q="  compressor  ").q == "compressor"


class TestEscapeLike:
    def test_metacharacters_become_literal(self) -> None:
        assert _escape_like(r"100%_a\b") == r"100\%\_a\\b"

    def test_plain_text_untouched(self) -> None:
        assert _escape_like("incoming/run.wav") == "incoming/run.wav"


class TestRunsUrl:
    def test_all_params_round_trip(self) -> None:
        speaker = uuid.uuid4()
        filters = SearchFilters(
            q='a "b c" -d',
            speaker_id=speaker,
            source="incoming/",
            created_from=date(2026, 8, 1),
            created_to=date(2026, 8, 14),
            language="es",
        )
        url = runs_url(
            status=RunStatus.COMPLETED, review=ReviewFilter.RESOLVED, filters=filters
        )
        assert url.startswith("/runs?")
        assert "status=completed" in url
        assert "review=resolved" in url
        assert f"speaker={speaker}" in url
        assert "created_from=2026-08-01" in url
        assert "created_to=2026-08-14" in url
        assert "language=es" in url
        # querystring-encoded, not raw
        assert "q=a+%22b+c%22+-d" in url

    def test_empty_filters_add_nothing(self) -> None:
        assert runs_url(filters=SearchFilters()) == "/runs"

    def test_cursor_appended_last_with_filters(self) -> None:
        cursor = Cursor(created_at=datetime(2026, 8, 14, tzinfo=UTC), run_id=uuid.uuid4())
        url = runs_url(filters=SearchFilters(q="x"), cursor=cursor)
        assert "q=x" in url
        assert url.endswith(urlencode([("cursor", cursor.encode())]))


class TestRenderHeadline:
    def test_sentinels_become_mark(self) -> None:
        html = _render_headline("before [[voxint-hit[[term]]voxint-hit]] after")
        assert html == Markup("before <mark>term</mark> after")

    def test_hostile_transcript_text_is_escaped(self) -> None:
        html = _render_headline(
            '<script>alert(1)</script> [[voxint-hit[[<b>hit</b>]]voxint-hit]]'
        )
        assert "<script>" not in str(html)
        assert "&lt;script&gt;" in str(html)
        # The hit content itself is escaped too — only our <mark> is live.
        assert "<mark>&lt;b&gt;hit&lt;/b&gt;</mark>" in str(html)

    def test_result_is_markup_safe_against_double_escape(self) -> None:
        html = _render_headline("[[voxint-hit[[x]]voxint-hit]]")
        # Jinja must not re-escape the <mark> when rendering.
        assert isinstance(html, Markup)
        assert html.__html__() == "<mark>x</mark>"
