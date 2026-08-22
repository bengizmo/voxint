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

A second, language-neutral config lives here too. The transcript-segment search
above is ``english`` (stemmed). Semantic search (issue #121) fuses a vector arm
with a lexical arm over multilingual chunk text, so its lexical arm uses the
``simple`` dictionary: no stemming, no stopword list, every token kept as a
lexeme. That keeps a Spanish or Chinese chunk searchable by its own words rather
than through an English stemmer. The ``simple`` helpers are used only by the
semantic-search query path; the legacy segment search is untouched.
"""

from typing import Any

from sqlalchemy import ColumnElement, ColumnExpressionArgument, func, literal_column

TS_CONFIG = "english"
SIMPLE_TS_CONFIG = "simple"

RAW_FTS_INDEX_NAME = "transcript_segments_raw_fts_idx"
ENHANCED_FTS_INDEX_NAME = "transcript_segments_enhanced_fts_idx"
# Operator-corrected text (issue #58) is a third independently-searchable
# rendering (never coalesced, same reasoning as raw/enhanced). It lives on
# segment_review_states and is NULL for most rows, so the index is PARTIAL
# (WHERE corrected_text IS NOT NULL) — sparse and cheap. Declared in migration
# 0020 alongside the table; an EXPLAIN test proves the planner uses it.
CORRECTED_FTS_INDEX_NAME = "segment_review_states_corrected_fts_idx"

_CONFIG_LITERAL: ColumnElement[str] = literal_column(f"'{TS_CONFIG}'")
_SIMPLE_CONFIG_LITERAL: ColumnElement[str] = literal_column(f"'{SIMPLE_TS_CONFIG}'")


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


def simple_ts_vector(
    text_column: ColumnExpressionArgument[str] | ColumnExpressionArgument[str | None],
) -> ColumnElement[Any]:
    """``to_tsvector('simple', column)`` — the language-neutral chunk arm (#121)."""
    return func.to_tsvector(_SIMPLE_CONFIG_LITERAL, text_column)


def simple_ts_query(user_query: str) -> ColumnElement[Any]:
    """``websearch_to_tsquery('simple', …)`` for the multilingual chunk arm (#121).

    ``simple`` normalizes case and splits on token boundaries but never stems and
    keeps every word (no stopword list), so non-English chunk text stays
    searchable. A blank or punctuation-only query yields an empty tsquery that
    matches nothing — the caller keeps the vector arm regardless. Quoted phrases
    here mean adjacent lexemes, NOT a literal substring; the exact-quote arm in
    the query path handles literal matching separately.
    """
    return func.websearch_to_tsquery(_SIMPLE_CONFIG_LITERAL, user_query)


def simple_ts_rank(
    vector: ColumnElement[Any], query: ColumnElement[Any]
) -> ColumnElement[float]:
    """``ts_rank_cd(vector, query)`` — cover-density lexical rank for the arm (#121)."""
    return func.ts_rank_cd(vector, query)


def simple_ts_headline(
    document: ColumnExpressionArgument[str] | ColumnExpressionArgument[str | None],
    query: ColumnElement[Any],
    options: str,
) -> ColumnElement[str]:
    """``ts_headline('simple', document, query, options)`` for passage snippets (#121)."""
    return func.ts_headline(_SIMPLE_CONFIG_LITERAL, document, query, options)
