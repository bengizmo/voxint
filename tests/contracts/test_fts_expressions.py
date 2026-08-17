"""Pin the FTS expression agreement: db/search.py ↔ ORM indexes ↔ migration 0008.

Postgres only uses an expression index when the query's expression matches the
indexed one, so the dictionary and ``to_tsvector`` shapes must agree across
three places that would otherwise drift silently: the compiled query
expressions (``voxint.db.search``), the ORM ``Index`` declarations on
``TranscriptSegment``, and the DDL literals in migration 0008. Changing the
dictionary means changing all three together — in a new migration.
"""

from pathlib import Path

from sqlalchemy.dialects import postgresql

from voxint.db.models import SegmentReviewState, TranscriptSegment
from voxint.db.search import (
    CORRECTED_FTS_INDEX_NAME,
    ENHANCED_FTS_INDEX_NAME,
    RAW_FTS_INDEX_NAME,
    TS_CONFIG,
    ts_query,
    ts_vector,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = REPO_ROOT / "alembic" / "versions" / "0008_transcript_fts_indexes.py"
# The corrected-text partial GIN index (issue #58, D3) rides its own table's
# migration, not 0008.
MIGRATION_CORRECTED = (
    REPO_ROOT / "alembic" / "versions" / "0020_segment_review_states.py"
)


def _compiled(expr: object) -> str:
    return str(expr.compile(dialect=postgresql.dialect()))  # type: ignore[attr-defined]


def test_query_expressions_use_the_pinned_config() -> None:
    raw = _compiled(ts_vector(TranscriptSegment.raw_text))
    enhanced = _compiled(ts_vector(TranscriptSegment.enhanced_text))
    corrected = _compiled(ts_vector(SegmentReviewState.corrected_text))
    assert raw == f"to_tsvector('{TS_CONFIG}', transcript_segments.raw_text)"
    assert enhanced == f"to_tsvector('{TS_CONFIG}', transcript_segments.enhanced_text)"
    assert (
        corrected
        == f"to_tsvector('{TS_CONFIG}', segment_review_states.corrected_text)"
    )
    # The config must be inlined, never a bound parameter — a parameterized
    # config can never match the constant-folded index expression.
    assert "%(" not in raw and "%(" not in enhanced and "%(" not in corrected
    assert _compiled(ts_query("x")).startswith(f"websearch_to_tsquery('{TS_CONFIG}'")


def test_orm_indexes_match_search_module() -> None:
    indexes = {i.name: i for i in TranscriptSegment.__table__.indexes}
    assert RAW_FTS_INDEX_NAME in indexes
    assert ENHANCED_FTS_INDEX_NAME in indexes
    raw_ddl = _compiled(next(iter(indexes[RAW_FTS_INDEX_NAME].expressions)))
    enhanced_ddl = _compiled(next(iter(indexes[ENHANCED_FTS_INDEX_NAME].expressions)))
    assert raw_ddl == f"to_tsvector('{TS_CONFIG}', raw_text)"
    assert enhanced_ddl == f"to_tsvector('{TS_CONFIG}', enhanced_text)"
    assert indexes[RAW_FTS_INDEX_NAME].dialect_options["postgresql"]["using"] == "gin"
    assert (
        indexes[ENHANCED_FTS_INDEX_NAME].dialect_options["postgresql"]["using"] == "gin"
    )


def test_migration_ddl_matches_search_module() -> None:
    # Normalize away Python string-literal wrapping (quotes + line breaks) so
    # the assertion pins the DDL content, not the source formatting.
    normalized = " ".join(MIGRATION.read_text().replace('"', " ").split())
    for name, column in (
        (RAW_FTS_INDEX_NAME, "raw_text"),
        (ENHANCED_FTS_INDEX_NAME, "enhanced_text"),
    ):
        assert (
            f"CREATE INDEX {name} ON transcript_segments "
            f"USING gin (to_tsvector('{TS_CONFIG}', {column}))" in normalized
        ), f"migration DDL drifted from db/search.py for {name}"
        assert f"DROP INDEX {name}" in normalized


def test_corrected_orm_index_matches_search_module() -> None:
    indexes = {i.name: i for i in SegmentReviewState.__table__.indexes}
    assert CORRECTED_FTS_INDEX_NAME in indexes
    index = indexes[CORRECTED_FTS_INDEX_NAME]
    ddl = _compiled(next(iter(index.expressions)))
    assert ddl == f"to_tsvector('{TS_CONFIG}', corrected_text)"
    assert index.dialect_options["postgresql"]["using"] == "gin"
    # PARTIAL — sparse corrections, so the index only covers non-NULL rows. The
    # query's IS NOT NULL guard must mirror this WHERE for the planner to use it.
    where = _compiled(index.dialect_options["postgresql"]["where"])
    assert where == "corrected_text IS NOT NULL"


def test_corrected_migration_ddl_matches_search_module() -> None:
    normalized = " ".join(MIGRATION_CORRECTED.read_text().replace('"', " ").split())
    assert (
        f"CREATE INDEX {CORRECTED_FTS_INDEX_NAME} ON segment_review_states "
        f"USING gin (to_tsvector('{TS_CONFIG}', corrected_text)) "
        "WHERE corrected_text IS NOT NULL" in normalized
    ), "migration DDL drifted from db/search.py for the corrected FTS index"
    assert f"DROP INDEX {CORRECTED_FTS_INDEX_NAME}" in normalized
