"""Full-text search expressions — the one definition the index and queries share.

Postgres only uses an expression index when the query's expression matches the
indexed one, so the ``to_tsvector`` calls built here must stay in lockstep with
the DDL in migration 0008 (a contract test pins the agreement, and an EXPLAIN
integration test proves the planner actually matches them). Two deliberate
choices, recorded once here:

- **Dictionary** ``english``: stemming recall fits the dominant "find the run
  where we discussed X" query. Stopword-only queries yield an empty tsquery
  and match nothing — accepted, pinned by test. Changing dictionary is a new
  migration rebuilding the indexes plus this constant, together.
- **Both text variants indexed separately** (never ``coalesce``): enhancement
  rewrites words, so a coalesced document would lose the raw rendering of a
  term the moment its batch is enhanced. Queries OR the two ``@@`` predicates;
  the enhanced vector is NULL for unenhanced rows, and NULL ``@@`` is falsy.

The config is inlined via ``literal_column`` — a bound parameter would compile
to ``to_tsvector($1, …)``, which can never match the constant-folded index
expression.
"""

from typing import Any

from sqlalchemy import ColumnElement, ColumnExpressionArgument, func, literal_column

TS_CONFIG = "english"

RAW_FTS_INDEX_NAME = "transcript_segments_raw_fts_idx"
ENHANCED_FTS_INDEX_NAME = "transcript_segments_enhanced_fts_idx"

_CONFIG_LITERAL: ColumnElement[str] = literal_column(f"'{TS_CONFIG}'")


def ts_vector(
    text_column: ColumnExpressionArgument[str] | ColumnExpressionArgument[str | None],
) -> ColumnElement[Any]:
    """``to_tsvector('<config>', column)`` matching the 0008 index expressions."""
    return func.to_tsvector(_CONFIG_LITERAL, text_column)


def ts_query(user_query: str) -> ColumnElement[Any]:
    """Parse operator input with ``websearch_to_tsquery`` — never raises on syntax."""
    return func.websearch_to_tsquery(_CONFIG_LITERAL, user_query)


def ts_headline(
    document: ColumnExpressionArgument[str] | ColumnExpressionArgument[str | None],
    query: ColumnElement[Any],
    options: str,
) -> ColumnElement[str]:
    """``ts_headline('<config>', document, query, options)`` for hit snippets."""
    return func.ts_headline(_CONFIG_LITERAL, document, query, options)
