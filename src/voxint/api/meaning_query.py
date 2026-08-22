"""The ``/search`` "Meaning" mode: hybrid semantic + lexical passage search (#121).

The chronological ``/runs`` browser (:mod:`voxint.api.runs_query`) finds a run by
exact words. This module finds a *passage* by meaning, across the whole corpus,
reading the embedding index the spine (PR1) builds. It is deliberately a distinct
mode, not folded into the keyset browse: results are a finite ranked top-k of
paragraph passages, each with a jump into the transcript at the passage time, and
there is no "older" cursor.

Ranking fuses three arms over ``segment_embeddings``:

- a **vector** arm — exact cosine nearest-neighbour on the query embedding;
- a **lexical** arm — a language-neutral ``simple``-config full-text match over
  the stored chunk text (so a non-English chunk stays searchable by its own
  words); and
- an **exact-quote** arm — literal substring hits for any ``"quoted phrase"`` in
  the query, which get hard top priority.

The vector and lexical arms are fused with Reciprocal Rank Fusion; the quote arm
floats its hits above everything. All three read inside ONE short read-only
REPEATABLE READ transaction: the publish routine replaces a run's whole
generation atomically, but two independent READ COMMITTED statements could still
straddle a concurrent publish and mix an old generation's rows from one arm with
a new generation's from another. One snapshot removes that hazard. Because a
publish deletes every prior generation in the same transaction it commits the new
one, the table holds exactly one generation per (run, space) at any committed
read point, so search reads ``segment_embeddings`` directly and never has to
resolve the current generation through ``embedding_jobs``.

No ANN index in v1: an exact cosine scan bounded by a candidate LIMIT is
sub-second at single-operator scale (tens of thousands of chunks). The pure
ranking is factored out (:func:`rank_candidates`, :func:`parse_positive_quotes`)
so it unit-tests without a database.
"""

import enum
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from markupsafe import Markup, escape
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.presentation import title_from_snapshot
from voxint.app_settings import (
    get_app_settings,
    resolve_effective_semantic_index_enabled,
)
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    SegmentEmbedding,
)
from voxint.db.search import simple_ts_query, simple_ts_rank, simple_ts_vector
from voxint.embeddings.onnx_embedder import (
    EMBEDDING_SPACE,
    TextEmbedder,
    get_text_embedder,
    minilm_artifacts_available,
)

# How many rows each bounded arm returns before fusion. Generous relative to the
# top-k so a passage that ranks modestly in both arms can still be fused up. At
# single-operator scale the exact scans that back these stay sub-second.
CANDIDATE_LIMIT = 200
# Reciprocal Rank Fusion constant. 60 is the conventional value; it damps the
# advantage of a rank-1 hit just enough that agreement across arms wins.
RRF_K = 60
# The final ranked page size, and the per-run cap so one long recording cannot
# flood the top with adjacent passages from a single transcript.
DEFAULT_TOP_K = 50
DEFAULT_PER_RUN_CAP = 3
# The passage preview length, in characters, before eliding.
_PREVIEW_CHARS = 320

# The same escape-first-then-promote sentinels the exact search uses, so a hostile
# transcript can never inject markup: escaping happens before the sentinels become
# <mark>, so the only live HTML is ours.
_MARK_START = "[[voxint-hit[["
_MARK_STOP = "]]voxint-hit]]"

_QUOTE_RE = re.compile(r'(?P<neg>-?)"(?P<phrase>[^"]+)"')


class MeaningSearchState(enum.StrEnum):
    """Why a search rendered what it did — drives an honest UI state."""

    OK = "ok"  # a query ran; ``items`` may still be empty (nothing matched)
    EMPTY_QUERY = "empty_query"  # no query typed yet
    OFF = "off"  # the semantic-index feature is disabled
    UNAVAILABLE = "unavailable"  # enabled, but the model weights are absent
    INDEXING = "indexing"  # enabled, but no passages are indexed yet


@dataclass(frozen=True)
class Candidate:
    """One chunk hit gathered from the arms, before and after fusion.

    Carries both the ranking inputs (arm ranks, cosine distance, exact-quote
    flag) and everything the template renders, so the pure ranker needs no DB.
    """

    id: uuid.UUID
    run_id: uuid.UUID
    chunk_index: int
    run_created_at: datetime
    title: str | None
    source_path: str
    speaker_label: str | None
    start_seconds: float
    end_seconds: float
    chunk_text: str
    vector_rank: int | None = None
    lexical_rank: int | None = None
    distance: float | None = None
    exact_quote: bool = False

    def rrf_score(self, *, k: int = RRF_K) -> float:
        """The fused score: 1/(k+rank) summed over the arms this hit appears in."""
        score = 0.0
        if self.vector_rank is not None:
            score += 1.0 / (k + self.vector_rank)
        if self.lexical_rank is not None:
            score += 1.0 / (k + self.lexical_rank)
        return score


@dataclass(frozen=True)
class PassageResult:
    """One ranked passage the template renders."""

    run_id: uuid.UUID
    title: str | None
    source_path: str
    speaker_label: str | None
    start_seconds: float
    end_seconds: float
    snippet: Markup
    jump_url: str
    score: float
    exact_quote: bool


@dataclass(frozen=True)
class MeaningResultsPage:
    """The whole result of one search: a state plus (when OK) the ranked page."""

    state: MeaningSearchState
    query: str
    items: list[PassageResult] = field(default_factory=list)


def parse_positive_quotes(query: str) -> list[str]:
    """Extract the POSITIVE ``"quoted"`` phrases from an operator query.

    A phrase written ``-"like this"`` is an exclusion in websearch syntax and is
    NOT promoted (it must not float to the top). Phrases are returned in order,
    de-duplicated case-insensitively, stripped, and non-empty. Matching later is
    a literal, case-insensitive substring test, so the raw phrase text is kept.
    """
    seen: set[str] = set()
    phrases: list[str] = []
    for match in _QUOTE_RE.finditer(query):
        if match.group("neg"):
            continue
        phrase = match.group("phrase").strip()
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def rank_candidates(
    candidates: list[Candidate],
    *,
    k: int = RRF_K,
    per_run_cap: int = DEFAULT_PER_RUN_CAP,
    top_k: int = DEFAULT_TOP_K,
) -> list[Candidate]:
    """Fuse, order, cap per run, and truncate — the pure ranking, DB-free.

    Order: exact-quote hits first (hard priority), then by RRF score. Ties break
    deterministically on cosine distance (closer first), then newer run, then a
    stable ``(run_id, chunk_index)``. The per-run cap is applied while WALKING the
    final order, so an exact-quote hit consumes a run's allowance before an
    ordinary passage from the same run.
    """

    def sort_key(c: Candidate) -> tuple[int, float, float, float, str, int]:
        return (
            0 if c.exact_quote else 1,
            -c.rrf_score(k=k),
            c.distance if c.distance is not None else float("inf"),
            -c.run_created_at.timestamp(),
            str(c.run_id),
            c.chunk_index,
        )

    ordered = sorted(candidates, key=sort_key)
    per_run: dict[uuid.UUID, int] = {}
    kept: list[Candidate] = []
    for candidate in ordered:
        if per_run.get(candidate.run_id, 0) >= per_run_cap:
            continue
        per_run[candidate.run_id] = per_run.get(candidate.run_id, 0) + 1
        kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept


def _jump_url(run_id: uuid.UUID, start_seconds: float) -> str:
    """The transcript deep link at the passage start (the island scrolls to ?t=).

    Preserves sub-second precision: truncating to whole seconds can push the
    target into the PREVIOUS contiguous segment, because the island resolves the
    line by half-open ``[start, end)`` containment. A passage starting at 10.9s
    truncated to ``t=10`` lands inside a prior ``[9.2, 10.9)`` line and scrolls
    the operator to the wrong place. Format with a stable decimal and trim
    trailing zeros so an integer start still reads ``t=10``.
    """
    t = f"{start_seconds:.3f}".rstrip("0").rstrip(".")
    return f"/runs/{run_id}/transcript?t={t}"


def _preview(chunk_text: str, phrases: list[str]) -> Markup:
    """A bounded, escaped passage preview with any quoted phrases marked.

    Escapes the whole passage FIRST, then promotes sentinel markers to ``<mark>``
    — the exact-search safety contract, so a hostile transcript cannot inject
    markup. The window is centred on the earliest quoted-phrase hit when there is
    one (so the operator sees why it matched), else it is the passage head.
    """
    text = chunk_text.strip()
    # Find the earliest positive-phrase occurrence to anchor the window.
    lowered = text.casefold()
    hits = [pos for pos in (lowered.find(p.casefold()) for p in phrases) if pos != -1]
    anchor = min(hits) if hits else 0
    start = 0
    if anchor > _PREVIEW_CHARS // 2:
        start = anchor - _PREVIEW_CHARS // 2
    window = text[start : start + _PREVIEW_CHARS]
    prefix = "… " if start > 0 else ""
    suffix = " …" if start + _PREVIEW_CHARS < len(text) else ""
    # Wrap phrase occurrences in the window with sentinels (case-insensitive),
    # longest phrase first so a phrase containing another is marked as a whole.
    marked = window
    for phrase in sorted(phrases, key=len, reverse=True):
        if not phrase:
            continue
        marked = re.sub(
            re.escape(phrase),
            lambda m: f"{_MARK_START}{m.group(0)}{_MARK_STOP}",
            marked,
            flags=re.IGNORECASE,
        )
    escaped = str(escape(prefix + marked + suffix))
    return Markup(
        escaped.replace(str(escape(_MARK_START)), "<mark>").replace(
            str(escape(_MARK_STOP)), "</mark>"
        )
    )


def _base_columns() -> list[Any]:
    return [
        SegmentEmbedding.id,
        SegmentEmbedding.pipeline_run_id,
        SegmentEmbedding.chunk_index,
        SegmentEmbedding.speaker_label,
        SegmentEmbedding.start_seconds,
        SegmentEmbedding.end_seconds,
        SegmentEmbedding.chunk_text,
        MediaItem.source_path,
        PipelineRun.sidecar,
        PipelineRun.created_at.label("run_created_at"),
        MediaSourceMetadata.title.label("source_title"),
    ]


def _with_run_joins(stmt: Select[Any]) -> Select[Any]:
    """Join a ``segment_embeddings`` select to its run + media, hiding archived."""
    return (
        stmt.join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .where(
            SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
            PipelineRun.archived_at.is_(None),
        )
    )


def _make_candidate(row: Any, *, exact_quote: bool = False) -> Candidate:
    return Candidate(
        id=row.id,
        run_id=row.pipeline_run_id,
        chunk_index=row.chunk_index,
        run_created_at=row.run_created_at,
        title=title_from_snapshot(row.sidecar) or row.source_title,
        source_path=row.source_path,
        speaker_label=row.speaker_label,
        start_seconds=row.start_seconds,
        end_seconds=row.end_seconds,
        chunk_text=row.chunk_text,
        exact_quote=exact_quote,
    )


def search_passages(
    session_factory: sessionmaker[Session],
    *,
    settings: Settings,
    query: str,
    embedder: TextEmbedder | None = None,
    top_k: int = DEFAULT_TOP_K,
    per_run_cap: int = DEFAULT_PER_RUN_CAP,
    candidate_limit: int = CANDIDATE_LIMIT,
) -> MeaningResultsPage:
    """Run a Meaning search and return a ranked page or an honest empty state.

    The query is embedded BEFORE the read transaction opens (the embed is pure
    CPU and must not hold a DB snapshot). ``embedder`` is injectable for tests;
    in production the in-process singleton is used — never a Celery round-trip.
    """
    stripped = query.strip()
    if not stripped:
        return MeaningResultsPage(MeaningSearchState.EMPTY_QUERY, query)

    with session_factory() as gate_session:
        if not resolve_effective_semantic_index_enabled(get_app_settings(gate_session), settings):
            return MeaningResultsPage(MeaningSearchState.OFF, query)

    resolved_embedder = embedder
    if resolved_embedder is None:
        if not minilm_artifacts_available():
            return MeaningResultsPage(MeaningSearchState.UNAVAILABLE, query)
        resolved_embedder = get_text_embedder()

    query_vector = resolved_embedder.embed_texts([stripped])[0].tolist()
    phrases = parse_positive_quotes(stripped)

    with session_factory() as session:
        # One stable snapshot for all arms: a publish cannot straddle them.
        session.connection(
            execution_options={
                "isolation_level": "REPEATABLE READ",
                "postgresql_readonly": True,
            }
        )
        # Probe for SEARCHABLE rows with the same visibility the arms use
        # (non-archived runs in this space); an index of only archived runs is
        # not something a query can surface, so it reports INDEXING honestly
        # rather than a misleading "no passages match".
        indexed = session.execute(
            select(SegmentEmbedding.id)
            .join(PipelineRun, PipelineRun.id == SegmentEmbedding.pipeline_run_id)
            .where(
                SegmentEmbedding.embedding_space == EMBEDDING_SPACE,
                PipelineRun.archived_at.is_(None),
            )
            .limit(1)
        ).first()
        if indexed is None:
            return MeaningResultsPage(MeaningSearchState.INDEXING, query)

        candidates: dict[uuid.UUID, Candidate] = {}

        # Vector arm: exact cosine nearest-neighbour, deterministic on ties.
        distance = SegmentEmbedding.embedding.cosine_distance(query_vector)
        vector_stmt = (
            _with_run_joins(select(*_base_columns(), distance.label("distance")))
            .order_by(distance.asc(), SegmentEmbedding.id.asc())
            .limit(candidate_limit)
        )
        for rank, row in enumerate(session.execute(vector_stmt), start=1):
            candidates[row.id] = replace(
                _make_candidate(row), vector_rank=rank, distance=row.distance
            )

        # Lexical arm: simple-config FTS, with the mandatory @@ predicate so the
        # tail is not filled with zero-rank non-matches.
        tsq = simple_ts_query(stripped)
        chunk_vector = simple_ts_vector(SegmentEmbedding.chunk_text)
        lexical_stmt = (
            _with_run_joins(select(*_base_columns()))
            .where(chunk_vector.bool_op("@@")(tsq))
            .order_by(simple_ts_rank(chunk_vector, tsq).desc(), SegmentEmbedding.id.asc())
            .limit(candidate_limit)
        )
        for rank, row in enumerate(session.execute(lexical_stmt), start=1):
            existing = candidates.get(row.id) or _make_candidate(row)
            candidates[row.id] = replace(existing, lexical_rank=rank)

        # Exact-quote arm: literal substring hits, ALL of them (a literal can
        # rank outside both bounded arms), match-ANY across phrases. strpos, not
        # ILIKE, so a % or _ in the phrase is a literal, not a wildcard.
        if phrases:
            conditions = [
                func.strpos(func.lower(SegmentEmbedding.chunk_text), phrase.lower()) > 0
                for phrase in phrases
            ]
            quote_stmt = _with_run_joins(select(*_base_columns())).where(or_(*conditions))
            for row in session.execute(quote_stmt):
                existing = candidates.get(row.id) or _make_candidate(row)
                candidates[row.id] = replace(existing, exact_quote=True)

    ranked = rank_candidates(
        list(candidates.values()),
        per_run_cap=per_run_cap,
        top_k=top_k,
    )
    items = [
        PassageResult(
            run_id=c.run_id,
            title=c.title,
            source_path=c.source_path,
            speaker_label=c.speaker_label,
            start_seconds=c.start_seconds,
            end_seconds=c.end_seconds,
            snippet=_preview(c.chunk_text, phrases),
            jump_url=_jump_url(c.run_id, c.start_seconds),
            score=c.rrf_score(),
            exact_quote=c.exact_quote,
        )
        for c in ranked
    ]
    return MeaningResultsPage(MeaningSearchState.OK, query, items)
