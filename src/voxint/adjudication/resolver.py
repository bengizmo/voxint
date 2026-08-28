"""Read-time speaker attribution: the one resolver the UI and exports share.

Machine proposals and human rulings are never merged in storage — this module
is where they meet, at read time:

- The **effective decision** for a (run, label) is the newest ledger row
  (``created_at DESC, id DESC`` — corrections are appends, never edits).
- A human decision always beats machine evidence. Without one, only a
  *grounded* cosine proposal counts as identity; an ``llm_hint`` name is
  surfaced as a suggestion, never as attribution.
- ``exclude`` suppresses speaker attribution, never transcript text.

A label is **unresolved** (needs adjudication) when it has neither an
effective human decision nor a grounded cosine proposal. A COMPLETED run with
at least one unresolved label is in the adjudication queue.
"""

import enum
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias, TypeVar

from sqlalchemy import (
    ColumnElement,
    ColumnExpressionArgument,
    Exists,
    ScalarSelect,
    and_,
    distinct,
    func,
    or_,
    select,
)
from sqlalchemy.orm import Session, aliased, joinedload

from voxint.api.presentation import title_from_snapshot
from voxint.db.models import (
    AdjudicationDecision,
    AssignmentMethod,
    Decision,
    DiarizationTurn,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    SegmentReviewState,
    Speaker,
    SpeakerAssignment,
)
from voxint.speakers.roster import canonicalize, merge_map

# A run-id the correlated SQL predicates accept: either a literal UUID (single
# run) or an outer column such as ``PipelineRun.id`` (correlated per row).
RunIdRef: TypeAlias = "ColumnExpressionArgument[uuid.UUID] | uuid.UUID"

# The scope key of an override reduction — a whole-segment id, or a word-range
# ``(segment_id, start, end)`` tuple. ``_active_overrides`` is generic over it so
# both grains share one reduction (issue #59 slice 3).
_K = TypeVar("_K")


def _label_unresolved(
    run_id: RunIdRef, label: ColumnExpressionArgument[str]
) -> ColumnElement[bool]:
    """SQL mirror of ``label_states``' resolved/unresolved binary for one label.

    A (run, label) is UNRESOLVED when it has neither a human decision nor a
    grounded cosine proposal. *Any* decision kind (assign/exclude/unknown)
    resolves the label, so the append-only newest-wins precedence is irrelevant
    to this binary — presence of a single decision row is enough. Kept beside
    the Python resolver so the two definitions move together.
    """
    # Explicit multi-level correlation is required: without it SQLAlchemy
    # reintroduces ``pipeline_runs`` into each inner EXISTS FROM, turning the
    # ``pipeline_run_id == run_id`` guard into a cross-run join — a decision or
    # grounding for a label on *any* run would then resolve that label on
    # *every* run. ``correlate`` pins these to the outer run + turn instead.
    has_decision = (
        select(1)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            AdjudicationDecision.diarization_label == label,
            # LABEL scope only: a segment-scope override (issue #54 Phase B) rules
            # one segment, not the label, so it must NOT mark the label resolved.
            AdjudicationDecision.transcript_segment_id.is_(None),
        )
        .correlate(PipelineRun, DiarizationTurn)
        .exists()
    )
    has_grounded_cosine = (
        select(1)
        .where(
            SpeakerAssignment.pipeline_run_id == run_id,
            SpeakerAssignment.diarization_label == label,
            SpeakerAssignment.method == AssignmentMethod.COSINE.value,
            SpeakerAssignment.grounded.is_(True),
        )
        .correlate(PipelineRun, DiarizationTurn)
        .exists()
    )
    return and_(~has_decision, ~has_grounded_cosine)


def unresolved_label_exists(run_id: RunIdRef) -> Exists:
    """EXISTS a diarization-turn label of ``run_id`` that needs a human ruling.

    Labels are anchored in ``diarization_turns`` exactly as ``label_states``
    does — a decision or proposal whose label has no turn is not a label of the
    run and is ignored here too.
    """
    return (
        select(1)
        .where(
            DiarizationTurn.pipeline_run_id == run_id,
            _label_unresolved(run_id, DiarizationTurn.label),
        )
        .exists()
    )


def unresolved_label_count(run_id: RunIdRef) -> ScalarSelect[int]:
    """Count of distinct unresolved turn-labels for ``run_id`` (for display)."""
    return (
        select(func.count(distinct(DiarizationTurn.label)))
        .where(
            DiarizationTurn.pipeline_run_id == run_id,
            _label_unresolved(run_id, DiarizationTurn.label),
        )
        .scalar_subquery()
    )


def speaker_attributed_exists(
    run_id: RunIdRef, speaker_ids: Collection[uuid.UUID]
) -> Exists:
    """EXISTS a turn label of ``run_id`` whose effective attribution is in the set.

    SQL mirror of ``label_states``' attribution branch, kept beside
    ``_label_unresolved`` so the definitions move together. Unlike that binary,
    newest-decision-wins matters here: an ``assign`` superseded by a newer
    ``exclude``/``unknown``/re-``assign`` must not match, so the decision
    branch demands the effective row — no strictly newer ledger row exists,
    "newer" being the exact ``(created_at DESC, id DESC)`` tuple order
    ``effective_decisions`` uses. The cosine branch counts only when NO
    decision of any kind exists (any decision suppresses machine evidence)
    and the proposal is grounded.

    ``speaker_ids`` must be pre-expanded through the merge map
    (``roster.alias_ids``): ledger rows keep merged sources' ids, so matching
    the stored id against the expanded set is equivalent to comparing
    canonicalized identities. Labels are anchored in ``diarization_turns``
    exactly as ``label_states`` does.
    """
    ids = list(speaker_ids)
    newer = aliased(AdjudicationDecision)
    effective_assign = (
        select(1)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            AdjudicationDecision.diarization_label == DiarizationTurn.label,
            AdjudicationDecision.decision.in_(
                [Decision.ASSIGN.value, Decision.AUTO_ENROLL.value]
            ),
            AdjudicationDecision.speaker_id.in_(ids),
            # LABEL scope only (issue #54 Phase B): speaker search stays a
            # label-grain fact. Segment-scope overrides are deliberately not
            # surfaced here (documented v1 limitation); both this row and the
            # `newer` tie-break below must exclude them, or a segment INHERIT
            # could appear to "supersede" a label assign in the SQL mirror while
            # the Python resolver disagrees.
            AdjudicationDecision.transcript_segment_id.is_(None),
            ~(
                select(1)
                .where(
                    newer.pipeline_run_id == run_id,
                    newer.diarization_label == DiarizationTurn.label,
                    newer.transcript_segment_id.is_(None),
                    or_(
                        newer.created_at > AdjudicationDecision.created_at,
                        and_(
                            newer.created_at == AdjudicationDecision.created_at,
                            newer.id > AdjudicationDecision.id,
                        ),
                    ),
                )
                # Pin every outer level: the run/turn (as _label_unresolved
                # warns) AND the candidate decision row, or SQLAlchemy folds
                # adjudication_decisions back into this FROM and the
                # newest-row guard collapses.
                .correlate(PipelineRun, DiarizationTurn, AdjudicationDecision)
                .exists()
            ),
        )
        .correlate(PipelineRun, DiarizationTurn)
        .exists()
    )
    has_decision = (
        select(1)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            AdjudicationDecision.diarization_label == DiarizationTurn.label,
            AdjudicationDecision.transcript_segment_id.is_(None),
        )
        .correlate(PipelineRun, DiarizationTurn)
        .exists()
    )
    grounded_cosine_in_set = (
        select(1)
        .where(
            SpeakerAssignment.pipeline_run_id == run_id,
            SpeakerAssignment.diarization_label == DiarizationTurn.label,
            SpeakerAssignment.method == AssignmentMethod.COSINE.value,
            SpeakerAssignment.grounded.is_(True),
            SpeakerAssignment.speaker_id.in_(ids),
        )
        .correlate(PipelineRun, DiarizationTurn)
        .exists()
    )
    return (
        select(1)
        .where(
            DiarizationTurn.pipeline_run_id == run_id,
            or_(effective_assign, and_(~has_decision, grounded_cosine_in_set)),
        )
        .exists()
    )


def label_count(run_id: RunIdRef) -> ScalarSelect[int]:
    """Count of distinct diarization-turn labels for ``run_id`` (for display)."""
    return (
        select(func.count(distinct(DiarizationTurn.label)))
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .scalar_subquery()
    )


class Resolution(enum.StrEnum):
    """How a label's attribution was settled — or that it wasn't."""

    HUMAN_ASSIGN = "human_assign"
    HUMAN_EXCLUDE = "human_exclude"
    HUMAN_UNKNOWN = "human_unknown"
    GROUNDED_COSINE = "grounded_cosine"
    AUTO_ENROLL = "auto_enroll"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class LabelState:
    """Everything the workbench and exports need to know about one label."""

    label: str
    turn_count: int
    total_seconds: float
    resolution: Resolution
    speaker_id: uuid.UUID | None
    speaker_name: str | None
    # Machine evidence, shown as context regardless of resolution.
    cosine_speaker_id: uuid.UUID | None
    cosine_speaker_name: str | None
    cosine_confidence: float | None
    cosine_grounded: bool
    llm_hint_name: str | None
    effective_decision: AdjudicationDecision | None


def effective_decisions(
    session: Session, run_id: uuid.UUID
) -> dict[str, AdjudicationDecision]:
    """Newest LABEL-scope ledger row per label (created_at DESC, id DESC).

    Segment-scope rows (issue #54 Phase B) are excluded here — GROUND ZERO: this
    feeds ``label_states``, and a newer segment override would otherwise poison a
    whole label's resolution. Segment overrides resolve separately, in
    :func:`segment_states`.
    """
    rows = session.execute(
        select(AdjudicationDecision)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            AdjudicationDecision.transcript_segment_id.is_(None),
        )
        .order_by(
            AdjudicationDecision.diarization_label,
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
    ).scalars()
    effective: dict[str, AdjudicationDecision] = {}
    for row in rows:
        effective.setdefault(row.diarization_label, row)
    return effective


def review_states(
    session: Session, run_id: uuid.UUID
) -> dict[uuid.UUID, SegmentReviewState]:
    """Per-segment operator review state (verified mark + corrected text) for a
    run, keyed by ``transcript_segment_id``. One indexed batch load (no N+1) —
    the overlay the transcript resolver folds over segments (issues #53/#58),
    exactly as :func:`segment_states` folds attribution overrides."""
    rows = session.execute(
        select(SegmentReviewState).where(SegmentReviewState.pipeline_run_id == run_id)
    ).scalars()
    return {row.transcript_segment_id: row for row in rows}


@dataclass(frozen=True)
class SegmentOverride:
    """An active per-segment attribution override (issue #54 Phase B).

    Segment scope carries only ``assign`` (a per-segment speaker) and ``inherit``
    (reset to the label's resolution). Only an *active* override — one whose
    newest segment-scope row is an ``assign`` — becomes a ``SegmentOverride``; a
    newest ``inherit`` (or no row) means the segment simply follows its label, so
    it has no entry here. ``speaker_id`` is canonicalized through merge
    tombstones exactly as ``label_states`` does.
    """

    speaker_id: uuid.UUID
    speaker_name: str | None
    decision: AdjudicationDecision


def segment_states(
    session: Session, run_id: uuid.UUID
) -> dict[uuid.UUID, SegmentOverride]:
    """Active per-segment overrides for a run, keyed by ``transcript_segment_id``.

    One indexed batch load (no N+1). Newest segment-scope row wins per segment;
    an ``inherit`` newest means no active override (the segment follows its
    label), so it is omitted. Speaker ids canonicalize through the same merge map
    the label resolver uses, so the two scopes can never disagree on identity.

    Whole-segment scope only: a row with a word-range (issue #59 slice 3) is a
    finer sub-segment override handled by :func:`word_range_states`, so it is
    excluded here (``start_word_index IS NULL``) — the two grains never mix.
    """
    rows = session.execute(
        select(AdjudicationDecision)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            AdjudicationDecision.transcript_segment_id.is_not(None),
            AdjudicationDecision.start_word_index.is_(None),
        )
        .order_by(
            AdjudicationDecision.transcript_segment_id,
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
    ).scalars()
    newest: dict[uuid.UUID, AdjudicationDecision] = {}
    for row in rows:
        assert row.transcript_segment_id is not None
        newest.setdefault(row.transcript_segment_id, row)
    return _active_overrides(session, newest)


# A word-range scope key: the immutable parent segment id plus a half-open
# ``[start, end)`` word interval (issue #59 slice 3). Keyed on the parent id and
# offsets — never a disposable split-boundary row — so it survives re-split.
WordRangeKey: TypeAlias = "tuple[uuid.UUID, int, int]"


def word_range_states(
    session: Session, run_id: uuid.UUID
) -> dict[WordRangeKey, SegmentOverride]:
    """Active per-word-range overrides for a run, keyed by ``(segment_id, start,
    end)`` (issue #59 slice 3 — sub-segment reassignment).

    The finer-grain sibling of :func:`segment_states`: one indexed batch load (no
    N+1), newest row wins per exact range, a newest ``inherit`` removes the
    override (the range follows its whole-segment/label resolution again). Only
    rows carrying a range (``start_word_index IS NOT NULL``) are considered, so
    whole-segment overrides never leak in. Speaker ids canonicalize through the
    same merge map, so no scope can disagree on identity.
    """
    rows = session.execute(
        select(AdjudicationDecision)
        .where(
            AdjudicationDecision.pipeline_run_id == run_id,
            # A ranged row always carries a segment (the DB CHECK enforces it);
            # asserting it here too — as segment_states does — keeps the key
            # non-null and defends the reduction against any future CHECK relaxation.
            AdjudicationDecision.transcript_segment_id.is_not(None),
            AdjudicationDecision.start_word_index.is_not(None),
        )
        .order_by(
            AdjudicationDecision.transcript_segment_id,
            AdjudicationDecision.start_word_index,
            AdjudicationDecision.end_word_index,
            AdjudicationDecision.created_at.desc(),
            AdjudicationDecision.id.desc(),
        )
    ).scalars()
    newest: dict[WordRangeKey, AdjudicationDecision] = {}
    for row in rows:
        assert (
            row.transcript_segment_id is not None
            and row.start_word_index is not None
            and row.end_word_index is not None
        )
        key = (row.transcript_segment_id, row.start_word_index, row.end_word_index)
        newest.setdefault(key, row)
    return _active_overrides(session, newest)


def _active_overrides(
    session: Session, newest: dict[_K, AdjudicationDecision]
) -> dict[_K, SegmentOverride]:
    """Reduce newest-per-scope rows to active :class:`SegmentOverride`\\ s.

    Shared by :func:`segment_states` and :func:`word_range_states`: keep only the
    scopes whose newest row is an ``assign`` with a speaker (a newest ``inherit``
    drops out — the scope follows its coarser resolution), canonicalize speaker
    ids through merge tombstones, and batch-load display names in one query. The
    key type is opaque (segment id, or a word-range tuple), so both grains reuse
    this identical reduction and can never disagree on identity."""
    tombstones = merge_map(session)
    active: dict[_K, AdjudicationDecision] = {
        key: row
        for key, row in newest.items()
        if row.decision == Decision.ASSIGN.value and row.speaker_id is not None
    }
    speaker_ids = {
        canonicalize(row.speaker_id, tombstones)
        for row in active.values()
        if row.speaker_id is not None
    }
    speaker_ids.discard(None)
    names: dict[uuid.UUID, str] = (
        {
            sid: name
            for sid, name in session.execute(
                select(Speaker.id, Speaker.display_name).where(Speaker.id.in_(speaker_ids))
            ).tuples()
        }
        if speaker_ids
        else {}
    )
    result: dict[_K, SegmentOverride] = {}
    for key, row in active.items():
        assert row.speaker_id is not None  # active rows are assigns; narrows for mypy
        canonical = canonicalize(row.speaker_id, tombstones)
        result[key] = SegmentOverride(
            speaker_id=canonical,
            speaker_name=names.get(canonical),
            decision=row,
        )
    return result


def label_states(session: Session, run_id: uuid.UUID) -> list[LabelState]:
    """Resolve every diarization label of a run, in label order."""
    turn_stats = session.execute(
        select(
            DiarizationTurn.label,
            func.count().label("turns"),
            func.sum(DiarizationTurn.end_seconds - DiarizationTurn.start_seconds),
        )
        .where(DiarizationTurn.pipeline_run_id == run_id)
        .group_by(DiarizationTurn.label)
        .order_by(DiarizationTurn.label)
    ).all()
    proposals = (
        session.execute(
            select(SpeakerAssignment).where(SpeakerAssignment.pipeline_run_id == run_id)
        )
        .scalars()
        .all()
    )
    cosine_by_label = {
        p.diarization_label: p for p in proposals if p.method == AssignmentMethod.COSINE.value
    }
    hint_by_label = {
        p.diarization_label: p
        for p in proposals
        if p.method == AssignmentMethod.LLM_HINT.value
    }
    decisions = effective_decisions(session, run_id)

    # Merged speakers canonicalize at read time: an old ledger assign(B) renders
    # as B's merge target. The ledger row itself (effective_decision) stays the
    # immutable historical reference — canonicalization is presentation.
    tombstones = merge_map(session)

    def canonical(speaker_id: uuid.UUID | None) -> uuid.UUID | None:
        return canonicalize(speaker_id, tombstones) if speaker_id else None

    speaker_ids = {
        canonical(p.speaker_id) for p in cosine_by_label.values() if p.speaker_id
    } | {canonical(d.speaker_id) for d in decisions.values() if d.speaker_id}
    speaker_ids.discard(None)
    names: dict[uuid.UUID, str] = (
        {
            sid: name
            for sid, name in session.execute(
                select(Speaker.id, Speaker.display_name).where(Speaker.id.in_(speaker_ids))
            ).tuples()
        }
        if speaker_ids
        else {}
    )

    states: list[LabelState] = []
    for label, turns, seconds in turn_stats:
        cosine = cosine_by_label.get(label)
        hint = hint_by_label.get(label)
        decision = decisions.get(label)

        speaker_id: uuid.UUID | None = None
        if decision is not None:
            if decision.decision == Decision.ASSIGN.value:
                resolution = Resolution.HUMAN_ASSIGN
                speaker_id = canonical(decision.speaker_id)
            elif decision.decision == Decision.EXCLUDE.value:
                resolution = Resolution.HUMAN_EXCLUDE
            elif decision.decision == Decision.UNKNOWN.value:
                resolution = Resolution.HUMAN_UNKNOWN
            elif decision.decision == Decision.AUTO_ENROLL.value:
                resolution = Resolution.AUTO_ENROLL
                speaker_id = canonical(decision.speaker_id)
            else:
                raise AssertionError(f"unhandled decision type: {decision.decision!r}")
        elif cosine is not None and cosine.grounded:
            resolution = Resolution.GROUNDED_COSINE
            speaker_id = canonical(cosine.speaker_id)
        else:
            resolution = Resolution.UNRESOLVED

        cosine_speaker_id = canonical(cosine.speaker_id) if cosine else None
        states.append(
            LabelState(
                label=label,
                turn_count=int(turns),
                total_seconds=float(seconds or 0.0),
                resolution=resolution,
                speaker_id=speaker_id,
                speaker_name=names.get(speaker_id) if speaker_id else None,
                cosine_speaker_id=cosine_speaker_id,
                cosine_speaker_name=(
                    names.get(cosine_speaker_id) if cosine_speaker_id else None
                ),
                cosine_confidence=cosine.confidence if cosine else None,
                cosine_grounded=bool(cosine.grounded) if cosine else False,
                llm_hint_name=hint.proposed_name if hint else None,
                effective_decision=decision,
            )
        )
    return states


@dataclass(frozen=True)
class QueueEntry:
    run_id: uuid.UUID
    source_path: str
    unresolved_labels: int
    total_labels: int
    claimed_by: str | None
    # Operator-facing display context (issue #56); appended with defaults so
    # older positional/attribute construction stays valid. ``title`` is the
    # run's sidecar title (issue #104) when present, else the
    # acquisition-metadata title (issue #36); ``duration_seconds``
    # is the PROBED media length (``media_items.duration_seconds``), the truth,
    # not the source-claimed metadata figure; ``created_at`` is the run's.
    title: str | None = None
    duration_seconds: float | None = None
    created_at: datetime | None = None


# The queue's ordering options (issue #56). ``oldest`` is the historical FIFO
# fairness order and the default; ``unresolved`` surfaces the runs with the most
# voices still to adjudicate first ("Most voices to resolve"). An unknown value
# degrades to ``oldest`` rather than erroring — the route whitelists, but this
# keeps the function total.
QUEUE_SORTS = ("oldest", "unresolved")


def adjudication_queue(session: Session, *, sort: str = "oldest") -> list[QueueEntry]:
    """COMPLETED runs with at least one unresolved label.

    Ordered oldest-first (FIFO fairness, the default) or — with
    ``sort="unresolved"`` — most-unresolved-first, tie-broken oldest-first.
    Exact per-run resolution at single-operator scale — correctness over a
    clever SQL reduction of the precedence rules. Both the media item and its
    (usually absent) source-metadata snapshot are eager-loaded so enriching a
    row with a friendly title costs no per-run follow-up query.
    """
    now = datetime.now(tz=UTC)
    runs = session.execute(
        select(PipelineRun)
        .options(
            joinedload(PipelineRun.media_item).joinedload(MediaItem.source_metadata)
        )
        .where(
            PipelineRun.status == RunStatus.COMPLETED.value,
            # Soft-archived runs (issue #5) are hidden from the review queue.
            PipelineRun.archived_at.is_(None),
            # Benchmark runs use a reserved source_path prefix and are not
            # operator media -- exclude them from the review queue.
            ~PipelineRun.media_item.has(
                MediaItem.source_path.startswith("benchmark/")
            ),
        )
        # ``id`` is a deterministic secondary key: Postgres makes no ordering
        # promise among rows sharing a ``created_at`` (reachable when several
        # runs are inserted in one transaction under the DB-side ``now()``
        # default), so without it the oldest-first order — and the stable-sort
        # tie-break the ``unresolved`` mode relies on — would vary per request.
        # Matches the ledger's ``(created_at, id)`` discipline.
        .order_by(PipelineRun.created_at, PipelineRun.id)
    ).scalars()
    entries: list[QueueEntry] = []
    for run in runs:
        states = label_states(session, run.id)
        unresolved = sum(1 for s in states if s.resolution is Resolution.UNRESOLVED)
        if unresolved == 0:
            continue
        claim_live = (
            run.review_claim_expires_at is not None and run.review_claim_expires_at > now
        )
        metadata: MediaSourceMetadata | None = run.media_item.source_metadata
        entries.append(
            QueueEntry(
                run_id=run.id,
                source_path=run.media_item.source_path,
                unresolved_labels=unresolved,
                total_labels=len(states),
                claimed_by=run.review_claimed_by if claim_live else None,
                # Operator intent beats scraped context: a sidecar title
                # (issue #104) wins over the acquisition-metadata title.
                title=title_from_snapshot(run.sidecar)
                or (metadata.title if metadata is not None else None),
                duration_seconds=run.media_item.duration_seconds,
                created_at=run.created_at,
            )
        )
    # Stable sort over the oldest-first list: ``unresolved`` reorders by voice
    # count while preserving oldest-first among ties for free.
    if sort == "unresolved":
        entries.sort(key=lambda e: e.unresolved_labels, reverse=True)
    return entries


def review_backlog_count(session: Session) -> int:
    """How many runs are eligible for review — the queue's length, by construction.

    The dashboard's "Continue review (N)" affordance and the review queue it
    links to must never disagree (issue #117). Deriving the count from
    :func:`adjudication_queue` rather than a parallel status tally makes that
    drift impossible: both share the one predicate (``COMPLETED``, not archived,
    at least one unresolved label). The old dashboard counted
    ``AWAITING_ADJUDICATION`` runs — a status a successful pipeline never ends
    in — so its "Review backlog" card was structurally wrong. At single-operator
    scale the handful of ``QueueEntry`` objects this materializes costs nothing;
    correctness of the invariant beats a separate ``COUNT(*)`` that could rot
    away from the queue predicate again.
    """
    return len(adjudication_queue(session))
