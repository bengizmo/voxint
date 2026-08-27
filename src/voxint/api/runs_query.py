"""The `/runs` execution-history browser: filters, keyset pagination, listing.

A read-only view over ``pipeline_runs`` for the operator. Two orthogonal
filters — ``status`` (the raw run status) and ``review`` (a post-hoc
adjudication classification) — combine with AND. The review classification is
derived in SQL from the same resolver definitions the workbench uses, so a run
is classified here exactly as it would be in the ``/review`` queue.

Pagination is keyset (seek) on ``(created_at, id)`` descending — newest first,
stable across concurrent inserts, and bounded to one page per request. The
cursor carries the full-precision sort key of the last row shown; a strict
tuple comparison walks strictly older rows, so identical ``created_at`` values
never drop or duplicate a row (``id`` breaks the tie).
"""

import base64
import enum
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from markupsafe import Markup, escape
from sqlalchemy import ColumnElement, Float, and_, case, cast, func, or_
from sqlalchemy import select as sa_select
from sqlalchemy.orm import Session

from voxint.adjudication.resolver import (
    label_count,
    speaker_attributed_exists,
    unresolved_label_count,
    unresolved_label_exists,
)
from voxint.api.languages import language_label
from voxint.api.presentation import title_from_snapshot
from voxint.db.models import (
    MediaFolder,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    StageRun,
    TranscriptSegment,
)
from voxint.db.search import ts_headline, ts_query, ts_vector
from voxint.speakers.roster import alias_ids


class ReviewFilter(enum.StrEnum):
    """Post-hoc adjudication classification, orthogonal to run status."""

    NEEDED = "needed"  # COMPLETED with >=1 label still needing a human ruling
    RESOLVED = "resolved"  # COMPLETED with nothing left to rule on
    CLAIMED = "claimed"  # a reviewer currently holds a live claim


class InvalidCursorError(ValueError):
    """The pagination cursor is malformed or corrupt."""


@dataclass(frozen=True)
class Cursor:
    """The keyset position: the sort key of the last row already shown."""

    created_at: datetime
    run_id: uuid.UUID

    def encode(self) -> str:
        raw = f"{self.created_at.isoformat()}|{self.run_id}"
        return base64.urlsafe_b64encode(raw.encode()).decode()

    @classmethod
    def decode(cls, token: str) -> "Cursor":
        try:
            raw = base64.urlsafe_b64decode(token.encode()).decode()
            timestamp, run_id = raw.rsplit("|", 1)
            created_at = datetime.fromisoformat(timestamp)
            parsed_id = uuid.UUID(run_id)
        except ValueError as exc:  # bad base64, bad split, bad datetime, bad uuid
            raise InvalidCursorError(f"unparseable cursor {token!r}") from exc
        if created_at.tzinfo is None:
            # Every cursor we mint carries the TIMESTAMPTZ offset. A naive value
            # is a forged token: compared against the tz-aware column it would be
            # cast using the session timezone, silently shifting the seek
            # boundary under any non-UTC session. Reject it rather than paginate
            # off a moved fence.
            raise InvalidCursorError(f"cursor timestamp is not tz-aware: {token!r}")
        return cls(created_at=created_at, run_id=parsed_id)


@dataclass(frozen=True)
class SearchFilters:
    """The search facets, all optional and AND-composed with status/review."""

    q: str | None = None
    speaker_id: uuid.UUID | None = None
    source: str | None = None
    created_from: date | None = None
    created_to: date | None = None
    # Detected-language facet (issue #124): the whisper language code stamped on
    # the run. Exact match; NULL-language rows (queued/failed/legacy) never
    # match a specific code.
    language: str | None = None

    def active(self) -> bool:
        return any(
            (
                self.q,
                self.speaker_id,
                self.source,
                self.created_from,
                self.created_to,
                self.language,
            )
        )


def parse_search_filters(
    *,
    q: str | None,
    speaker: str | None,
    source: str | None,
    created_from: str | None,
    created_to: str | None,
    language: str | None = None,
) -> SearchFilters:
    """Blank/absent values mean 'off', mirroring the status/review parsers."""
    speaker_id: uuid.UUID | None = None
    if speaker not in (None, ""):
        try:
            speaker_id = uuid.UUID(speaker)
        except ValueError as exc:
            raise ValueError(f"invalid speaker id {speaker!r}") from exc

    def parse_date(raw: str | None, name: str) -> date | None:
        if not raw:
            return None
        try:
            parsed = date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {name} date {raw!r}") from exc
        if parsed == date.max:
            # date.max + 1 day (the exclusive upper bound) would overflow with
            # an ArithmeticError the route's ValueError→422 mapping misses.
            raise ValueError(f"invalid {name} date {raw!r}")
        return parsed

    return SearchFilters(
        q=(q.strip() or None) if q is not None else None,
        speaker_id=speaker_id,
        source=source if source not in (None, "") else None,
        created_from=parse_date(created_from, "created_from"),
        created_to=parse_date(created_to, "created_to"),
        # Any non-blank string is a legal facet value (exact match against the
        # stamped code; a value no run carries just yields an empty page) —
        # parameterized SQL, so no validation list to drift from the model.
        # Stripped like q, so "es " from a hand-edited URL still matches.
        language=(language.strip() or None) if language is not None else None,
    )


def _escape_like(fragment: str) -> str:
    """Escape LIKE metacharacters so operator input is a literal substring."""
    return fragment.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class Snippet:
    """One highlighted hit fragment for a run — pre-escaped, template-safe."""

    html: Markup
    start_seconds: float


@dataclass(frozen=True)
class RunListItem:
    """One row of the browser — everything the template renders per run."""

    run_id: uuid.UUID
    status: str
    source_path: str
    created_at: datetime
    unresolved_count: int
    label_count: int
    claim_live: bool
    claimed_by: str | None
    snippet: Snippet | None = None
    # Display title: the run's sidecar title (issue #104, operator intent) when
    # present, else the acquisition-metadata title (issue #36); None otherwise —
    # the template falls back to source_path.
    title: str | None = None
    # Soft-archived (issue #5): True when this run carries an archived_at stamp.
    # Only ever True in the explicit archived view (?archived=1) — the default
    # listing excludes archived runs — but the flag lets the template pill them.
    archived: bool = False
    # Detected language code (issue #124); None for runs not yet transcribed or
    # transcribed before the column existed — the template renders an honest "—".
    language: str | None = None
    # Wall-clock elapsed seconds (created_at to last stage finished_at); None for
    # runs still in progress or with no stage rows. Template uses
    # format_compact_duration.
    elapsed_seconds: float | None = None
    # Settings folder path (from media_folder via media_item); None for uploads
    # or media with no folder assignment.
    folder_path: str | None = None
    # Run-level error message for failed runs; None otherwise.
    error: str | None = None


@dataclass(frozen=True)
class RunsPage:
    """One bounded page plus the cursor that fetches the next (older) one."""

    items: list[RunListItem]
    next_cursor: Cursor | None


def parse_status_filter(raw: str | None) -> RunStatus | None:
    """A blank/absent value means 'all'; anything else must be a real status."""
    if raw in (None, ""):
        return None
    try:
        return RunStatus(raw)
    except ValueError as exc:
        raise ValueError(f"unknown status {raw!r}") from exc


def parse_review_filter(raw: str | None) -> ReviewFilter | None:
    if raw in (None, ""):
        return None
    try:
        return ReviewFilter(raw)
    except ValueError as exc:
        raise ValueError(f"unknown review filter {raw!r}") from exc


def runs_url(
    *,
    status: RunStatus | None = None,
    review: ReviewFilter | None = None,
    filters: SearchFilters | None = None,
    archived: bool = False,
    cursor: Cursor | None = None,
) -> str:
    """Build a ``/runs`` URL preserving the active filters (+ optional cursor)."""
    params: list[tuple[str, str]] = []
    if status is not None:
        params.append(("status", status.value))
    if review is not None:
        params.append(("review", review.value))
    if archived:
        params.append(("archived", "1"))
    if filters is not None:
        if filters.q is not None:
            params.append(("q", filters.q))
        if filters.speaker_id is not None:
            params.append(("speaker", str(filters.speaker_id)))
        if filters.source is not None:
            params.append(("source", filters.source))
        if filters.language is not None:
            params.append(("language", filters.language))
        if filters.created_from is not None:
            params.append(("created_from", filters.created_from.isoformat()))
        if filters.created_to is not None:
            params.append(("created_to", filters.created_to.isoformat()))
    if cursor is not None:
        params.append(("cursor", cursor.encode()))
    return "/runs" + (f"?{urlencode(params)}" if params else "")


# ts_headline sentinels: ASCII markers no transcript should contain. If one
# ever does, the failure mode is a stray highlight in already-escaped text —
# never markup injection, because escaping happens before substitution.
_START_SEL = "[[voxint-hit[["
_STOP_SEL = "]]voxint-hit]]"
_HEADLINE_OPTIONS = (
    f"StartSel={_START_SEL}, StopSel={_STOP_SEL}, "
    'MaxWords=18, MinWords=6, MaxFragments=2, FragmentDelimiter=" … "'
)


def _render_headline(fragment: str) -> Markup:
    """Escape the whole ts_headline output, THEN promote sentinels to <mark>.

    ts_headline emits the document text verbatim around its selectors; a
    transcript can contain hostile markup, so it must never reach the template
    trusted. Escaping first makes the sentinel substitution the only source of
    live HTML.
    """
    escaped = str(escape(fragment))
    return Markup(
        escaped.replace(str(escape(_START_SEL)), "<mark>").replace(
            str(escape(_STOP_SEL)), "</mark>"
        )
    )


def _segment_matches(tsq: ColumnElement[Any]) -> ColumnElement[bool]:
    """The transcript_segments match predicate the run filter and snippet share.

    Both pipeline renderings, OR'd — never coalesced — so the filter and snippet
    paths cannot drift apart on which segments count as hits. Operator-corrected
    text is a THIRD rendering matched by :func:`_corrected_matches`; it is kept
    separate because it lives on ``segment_review_states`` (see that helper).
    """
    return or_(
        ts_vector(TranscriptSegment.raw_text).bool_op("@@")(tsq),
        ts_vector(TranscriptSegment.enhanced_text).bool_op("@@")(tsq),
    )


def _corrected_matches(tsq: ColumnElement[Any]) -> ColumnElement[bool]:
    """Match predicate for operator-corrected text (issue #58, D3) — the third
    independently-searchable rendering, never coalesced with raw/enhanced.

    Deliberately NOT folded into :func:`_segment_matches`: ``corrected_text``
    lives on ``segment_review_states``, and OR-ing a second table's column into
    the transcript_segments predicate would make the planner unable to use the
    partial GIN index (an OR across two tables cannot become a BitmapOr). The
    run filter therefore evaluates this in its OWN correlated ``EXISTS`` over
    ``segment_review_states`` so the index is usable. The ``IS NOT NULL`` guard
    mirrors the index's partial ``WHERE`` so the planner selects it (EXPLAIN-
    tested). Corrected text is sparse, so this branch touches few rows.
    """
    return and_(
        SegmentReviewState.corrected_text.is_not(None),
        ts_vector(SegmentReviewState.corrected_text).bool_op("@@")(tsq),
    )


def _snippets_for(session: Session, run_ids: list[uuid.UUID], q: str) -> dict[uuid.UUID, Snippet]:
    """One highlighted hit per displayed run — the first matching segment.

    Bounded work: DISTINCT ON over only the page's run ids, so ts_headline
    runs on at most page_size short segments. "First by segment_index", not
    "best" — there is no ranking pre-1.0. The headline is computed over the
    text variant that actually matched, in the display precedence
    corrected → enhanced → raw among the renderings that matched (never
    coalesced — a coalesced headline could highlight a rendering that lacks the
    matched term). The LEFT JOIN keeps at most one review row per segment (PK on
    transcript_segment_id), so it never multiplies rows.
    """
    if not run_ids:
        return {}
    tsq = ts_query(q)
    corrected_matches = _corrected_matches(tsq)
    enhanced_matches = ts_vector(TranscriptSegment.enhanced_text).bool_op("@@")(tsq)
    headline = (
        sa_select(
            TranscriptSegment.pipeline_run_id,
            TranscriptSegment.start_seconds,
            case(
                (
                    corrected_matches,
                    ts_headline(SegmentReviewState.corrected_text, tsq, _HEADLINE_OPTIONS),
                ),
                (
                    enhanced_matches,
                    ts_headline(TranscriptSegment.enhanced_text, tsq, _HEADLINE_OPTIONS),
                ),
                else_=ts_headline(TranscriptSegment.raw_text, tsq, _HEADLINE_OPTIONS),
            ).label("fragment"),
        )
        .select_from(TranscriptSegment)
        .outerjoin(
            SegmentReviewState,
            SegmentReviewState.transcript_segment_id == TranscriptSegment.id,
        )
        .where(
            TranscriptSegment.pipeline_run_id.in_(run_ids),
            or_(_segment_matches(tsq), corrected_matches),
        )
    )
    stmt = headline.distinct(TranscriptSegment.pipeline_run_id).order_by(
        TranscriptSegment.pipeline_run_id, TranscriptSegment.segment_index
    )
    return {
        row.pipeline_run_id: Snippet(
            html=_render_headline(row.fragment),
            start_seconds=row.start_seconds,
        )
        for row in session.execute(stmt)
    }


def list_runs(
    session: Session,
    *,
    status: RunStatus | None,
    review: ReviewFilter | None,
    cursor: Cursor | None,
    page_size: int,
    filters: SearchFilters | None = None,
    archived: bool = False,
) -> RunsPage:
    """One bounded, newest-first keyset page of runs matching the filters.

    ``archived`` selects the soft-archive view (issue #5): ``False`` (default)
    hides archived runs entirely; ``True`` shows ONLY archived runs. Applied as a
    plain predicate before the keyset clause so pagination walks the chosen set.
    """
    claim_live = and_(
        PipelineRun.review_claim_expires_at.isnot(None),
        PipelineRun.review_claim_expires_at > func.now(),
    )
    last_finished = (
        sa_select(func.max(StageRun.finished_at))
        .where(StageRun.pipeline_run_id == PipelineRun.id)
        .correlate(PipelineRun)
        .scalar_subquery()
    )
    # updated_at fallback: approximation for legacy/seeded runs with no stage
    # rows. updated_at has onupdate=func.now(), so a later note/archive edit
    # can inflate this — acceptable for the rare legacy case.
    elapsed = cast(
        func.extract(
            "epoch",
            func.coalesce(last_finished, PipelineRun.updated_at) - PipelineRun.created_at,
        ),
        Float,
    )
    stmt = (
        sa_select(
            PipelineRun.id,
            PipelineRun.status,
            PipelineRun.created_at,
            PipelineRun.review_claimed_by,
            PipelineRun.archived_at,
            PipelineRun.error,
            MediaItem.source_path,
            PipelineRun.sidecar,
            PipelineRun.detected_language,
            MediaSourceMetadata.title.label("source_title"),
            MediaFolder.path.label("folder_path"),
            unresolved_label_count(PipelineRun.id).label("unresolved_count"),
            label_count(PipelineRun.id).label("label_count"),
            claim_live.label("claim_live"),
            case(
                (PipelineRun.status.in_(("completed", "failed")), elapsed),
                else_=None,
            ).label("elapsed_seconds"),
        )
        .join(MediaItem, MediaItem.id == PipelineRun.media_item_id)
        # Outer: most media has no metadata snapshot (uploads, pre-#36 runs).
        .outerjoin(MediaSourceMetadata, MediaSourceMetadata.media_item_id == MediaItem.id)
        .outerjoin(MediaFolder, MediaFolder.id == MediaItem.media_folder_id)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        .limit(page_size + 1)
    )

    # Soft-archive (issue #5): default hides archived runs; ?archived=1 shows
    # only them. Orthogonal to status/review, applied before the keyset clause.
    if archived:
        stmt = stmt.where(PipelineRun.archived_at.is_not(None))
    else:
        stmt = stmt.where(PipelineRun.archived_at.is_(None))

    if status is not None:
        stmt = stmt.where(PipelineRun.status == status.value)

    # needed/resolved are the COMPLETED complement of each other; claimed is a
    # live-lease predicate independent of status. Combined with an incompatible
    # status= (e.g. status=failed&review=needed) they yield an empty page, by
    # design — the filters are composable, not status-independent.
    if review is ReviewFilter.NEEDED:
        stmt = stmt.where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            unresolved_label_exists(PipelineRun.id),
        )
    elif review is ReviewFilter.RESOLVED:
        stmt = stmt.where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            ~unresolved_label_exists(PipelineRun.id),
        )
    elif review is ReviewFilter.CLAIMED:
        stmt = stmt.where(claim_live)

    # Search facets: plain AND-composed predicates, applied before the keyset
    # clause so pagination walks exactly the filtered set.
    if filters is not None and filters.q is not None:
        tsq = ts_query(filters.q)
        # Two correlated EXISTS OR'd at the run level, not one predicate OR'd
        # across tables: the corrected rendering lives on segment_review_states,
        # so a single OR would defeat its partial GIN index (see
        # _corrected_matches). Each EXISTS scans one table's own indexes.
        stmt = stmt.where(
            or_(
                sa_select(1)
                .where(
                    TranscriptSegment.pipeline_run_id == PipelineRun.id,
                    _segment_matches(tsq),
                )
                .correlate(PipelineRun)
                .exists(),
                sa_select(1)
                .where(
                    SegmentReviewState.pipeline_run_id == PipelineRun.id,
                    _corrected_matches(tsq),
                )
                .correlate(PipelineRun)
                .exists(),
            )
        )
    if filters is not None and filters.speaker_id is not None:
        stmt = stmt.where(
            speaker_attributed_exists(PipelineRun.id, alias_ids(session, filters.speaker_id))
        )
    if filters is not None and filters.source is not None:
        stmt = stmt.where(
            MediaItem.source_path.ilike(f"%{_escape_like(filters.source)}%", escape="\\")
        )
    if filters is not None and filters.language is not None:
        # Exact match on the stamped code; NULL-language rows (queued/failed/
        # legacy) are excluded by SQL NULL semantics, which is the honest read
        # of "show me the Spanish runs".
        stmt = stmt.where(PipelineRun.detected_language == filters.language)
    if filters is not None and filters.created_from is not None:
        stmt = stmt.where(
            PipelineRun.created_at
            >= datetime.combine(filters.created_from, datetime.min.time(), tzinfo=UTC)
        )
    if filters is not None and filters.created_to is not None:
        stmt = stmt.where(
            PipelineRun.created_at
            < datetime.combine(
                filters.created_to + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            )
        )

    if cursor is not None:
        stmt = stmt.where(
            or_(
                PipelineRun.created_at < cursor.created_at,
                and_(
                    PipelineRun.created_at == cursor.created_at,
                    PipelineRun.id < cursor.run_id,
                ),
            )
        )

    rows = session.execute(stmt).all()
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    snippets = (
        _snippets_for(session, [row.id for row in rows], filters.q)
        if filters is not None and filters.q is not None
        else {}
    )
    items = [
        RunListItem(
            run_id=row.id,
            status=row.status,
            source_path=row.source_path,
            created_at=row.created_at,
            unresolved_count=row.unresolved_count,
            label_count=row.label_count,
            claim_live=row.claim_live,
            claimed_by=row.review_claimed_by if row.claim_live else None,
            snippet=snippets.get(row.id),
            # Operator intent beats scraped context: a sidecar title (issue
            # #104) wins over the acquisition-metadata title. Display-only.
            title=title_from_snapshot(row.sidecar) or row.source_title,
            archived=row.archived_at is not None,
            language=row.detected_language,
            elapsed_seconds=row.elapsed_seconds,
            folder_path=row.folder_path,
            error=row.error,
        )
        for row in rows
    ]
    next_cursor = (
        Cursor(created_at=rows[-1].created_at, run_id=rows[-1].id) if has_more and rows else None
    )
    return RunsPage(items=items, next_cursor=next_cursor)


@dataclass(frozen=True)
class LanguageFacet:
    """One option in the runs browser's detected-language filter."""

    code: str
    label: str


def searchable_languages(
    session: Session, *, archived: bool = False, include: str | None = None
) -> list[LanguageFacet]:
    """The distinct detected languages the current view's runs carry, for the
    filter facet.

    Reads the stamped codes (issue #124), labels them via the pinned map
    ("Spanish (es)"; a raw code for anything the map predates), and orders by
    display label so the dropdown reads alphabetically for a human. NULLs are
    excluded — "not recorded" is not a language. ``archived`` mirrors
    :func:`list_runs`'s soft-archive scoping so the dropdown offers exactly the
    languages the listing can show. ``include`` forces one extra code into the
    options (the active filter): without it, a stale or hand-typed
    ``?language=`` the current view lacks would render the select as "all"
    while the predicate silently stays active. A bounded, low-cardinality scan
    (at most one row per distinct code), fine without an index for
    single-operator data volumes.
    """
    stmt = (
        sa_select(PipelineRun.detected_language)
        .where(PipelineRun.detected_language.is_not(None))
        .distinct()
    )
    if archived:
        stmt = stmt.where(PipelineRun.archived_at.is_not(None))
    else:
        stmt = stmt.where(PipelineRun.archived_at.is_(None))
    codes = set(session.execute(stmt).scalars().all())
    if include is not None:
        codes.add(include)
    facets = [
        LanguageFacet(code=code, label=language_label(code))
        for code in codes
        # The WHERE clause excludes NULLs; the guard narrows the nullable
        # column type for mypy without a cast.
        if code is not None
    ]
    return sorted(facets, key=lambda f: f.label)
