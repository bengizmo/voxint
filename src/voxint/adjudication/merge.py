"""Inline speaker merge: rule that several diarization labels are one person.

The workbench's most common correction is fixing an *over-split* speaker — the
diarizer routinely splits one voice into ``SPEAKER_00`` + ``SPEAKER_03``. This
module is the composite mutation behind the workbench's "these labels are the
same speaker in this recording" action:

- It is **run-local**. It records one ``assign`` ledger ruling per label to a
  single survivor speaker within THIS run. It NEVER calls
  :func:`voxint.speakers.roster.merge_speakers` — roster-wide identity surgery
  (which touches other runs, embeddings, and aliases) stays the explicit,
  separately-confirmed ``/speakers`` action. A later deliberate roster merge
  still unifies these run-local rulings retroactively at read time, so deferring
  the global act never paints the operator into a corner.
- It is **atomic**. Every label's ruling commits or rolls back together, under
  the caller's claim lock, so a partial "some labels moved" state is impossible.
- Its idempotency is **composite-safe**. One operator nonce backs several ledger
  rows, so each row gets a deterministic child key derived from the nonce and the
  label set; replaying the whole request returns the original rulings, and
  reusing the nonce for a *different* label set fails loudly instead of silently
  half-applying.
- Its impact is **server-computed**. :func:`preview_merge` reports the exact
  turns/segments a merge touches; the UI shows that, never an advisory client
  count.

Reset/undo is the same as any ruling: append a corrective decision. Nothing here
is destructive.
"""

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import (
    LabelState,
    Resolution,
    effective_decisions,
    label_states,
)
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    Speaker,
    TranscriptSegment,
)
from voxint.speakers.matching import MatchingGates, eligible_label_vectors
from voxint.speakers.roster import is_active


class MergeError(Exception):
    """The merge cannot proceed as requested — operator-visible."""


class MergeConflictError(Exception):
    """A label's ruling changed since the operator previewed the merge.

    The preview the operator confirmed no longer matches the ledger — applying
    would override a decision they never saw. The route maps this to 409 so the
    operator refreshes and re-previews rather than silently clobbering.
    """


@dataclass(frozen=True)
class LabelImpact:
    """What a merge would touch for one label, computed server-side."""

    label: str
    turn_count: int
    total_seconds: float
    segment_count: int
    resolution: Resolution
    current_speaker_id: uuid.UUID | None
    current_speaker_name: str | None
    # The effective ledger row id for this label right now, or None. The
    # optimistic-concurrency token: the confirm form echoes it, apply re-checks.
    expected_decision_id: uuid.UUID | None


@dataclass(frozen=True)
class MergePreview:
    """The exact, server-computed effect of a proposed merge."""

    labels: list[LabelImpact]
    total_turns: int
    total_segments: int
    # True when >=2 of the merged labels currently resolve to DISTINCT active
    # roster speakers — the UI shows the "does not merge them on the roster" note.
    distinct_roster_speakers: bool


@dataclass(frozen=True)
class MergeResult:
    """The committed outcome of an applied merge."""

    survivor_speaker_id: uuid.UUID
    survivor_name: str
    created_speaker: bool
    decision_ids: dict[str, uuid.UUID]
    labels: list[str]
    total_turns: int
    total_segments: int
    # True when this apply was an idempotent replay (every child ruling already
    # existed) rather than a fresh consolidation. The activity emit (issue #162)
    # announces only a fresh merge, never a replay of one. Defaults False for the
    # roster-wide merge path, which does not feed the activity feed.
    is_replay: bool = False


def _labels_digest(labels: list[str]) -> str:
    """Stable 64-bit-hex digest of the label SET (order-independent).

    Folded into each child idempotency key so the key set for a merge of one
    label set is disjoint from the key set for any other label set under the same
    nonce — different merges never share a child row. 64 bits keeps a chance
    collision between two distinct sets negligible.
    """
    joined = "\x00".join(sorted(set(labels)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _child_key(nonce: str, labels: list[str], label: str) -> str:
    # ``merge:`` namespaces these keys away from the bare nonce the single-label
    # decide/enroll routes use, so a merge child row can never be mistaken for a
    # decide row (or vice versa) in the replay probe.
    return f"merge:{nonce}:{_labels_digest(labels)}:{label}"


def _validate_labels(states: dict[str, LabelState], labels: list[str]) -> list[str]:
    """A cleaned, de-duplicated label list, all present in the run and >=2 distinct."""
    seen: list[str] = []
    for label in labels:
        if label not in states:
            raise MergeError(f"no label {label!r} in this run — refresh and retry")
        if label not in seen:
            seen.append(label)
    if len(seen) < 2:
        raise MergeError("select at least two distinct labels to merge")
    return seen


def _segment_counts(session: Session, run_id: uuid.UUID) -> dict[str, int]:
    rows = session.execute(
        select(TranscriptSegment.diarization_label, func.count())
        .where(TranscriptSegment.pipeline_run_id == run_id)
        .group_by(TranscriptSegment.diarization_label)
    ).all()
    return {label: int(count) for label, count in rows if label is not None}


def _impacts(
    session: Session, run_id: uuid.UUID, labels: list[str]
) -> tuple[list[LabelImpact], bool]:
    """Per-label impact rows + whether >=2 labels hold distinct roster speakers."""
    states = {s.label: s for s in label_states(session, run_id)}
    decisions = effective_decisions(session, run_id)
    seg_counts = _segment_counts(session, run_id)
    impacts: list[LabelImpact] = []
    distinct_speakers: set[uuid.UUID] = set()
    for label in labels:
        state = states[label]
        decision = decisions.get(label)
        if state.resolution in (
            Resolution.HUMAN_ASSIGN,
            Resolution.AUTO_ENROLL,
            Resolution.GROUNDED_COSINE,
        ) and state.speaker_id is not None:
            distinct_speakers.add(state.speaker_id)
        impacts.append(
            LabelImpact(
                label=label,
                turn_count=state.turn_count,
                total_seconds=state.total_seconds,
                segment_count=seg_counts.get(label, 0),
                resolution=state.resolution,
                current_speaker_id=state.speaker_id,
                current_speaker_name=state.speaker_name,
                expected_decision_id=decision.id if decision is not None else None,
            )
        )
    return impacts, len(distinct_speakers) >= 2


def preview_merge(
    session: Session, run_id: uuid.UUID, labels: list[str]
) -> MergePreview:
    """Compute the exact effect of merging ``labels`` — reads only, no writes."""
    states = {s.label: s for s in label_states(session, run_id)}
    clean = _validate_labels(states, labels)
    impacts, distinct = _impacts(session, run_id, clean)
    return MergePreview(
        labels=impacts,
        total_turns=sum(i.turn_count for i in impacts),
        total_segments=sum(i.segment_count for i in impacts),
        distinct_roster_speakers=distinct,
    )


def _check_expected(
    session: Session, run_id: uuid.UUID, expected: dict[str, uuid.UUID | None]
) -> None:
    """Reject if any label's effective ruling drifted from what was previewed."""
    current = effective_decisions(session, run_id)
    for label, expected_id in expected.items():
        actual = current.get(label)
        actual_id = actual.id if actual is not None else None
        if actual_id != expected_id:
            raise MergeConflictError(
                f"label {label!r} changed since you previewed — refresh and re-check"
            )


def apply_merge(
    session: Session,
    *,
    run_id: uuid.UUID,
    labels: list[str],
    operator: str,
    nonce: str,
    gates: MatchingGates,
    target_speaker_id: uuid.UUID | None = None,
    target_name: str | None = None,
    expected: dict[str, uuid.UUID | None],
    user_id: uuid.UUID | None = None,
) -> MergeResult:
    """Rule that ``labels`` are one speaker in this run, atomically.

    Exactly one of ``target_speaker_id`` (assign to an existing active roster
    identity) or ``target_name`` (enroll a new speaker from the labels) must be
    given. ``expected`` is the per-label effective-ruling snapshot the operator
    saw at preview — its keys MUST be exactly the merged label set. The caller
    owns the transaction and MUST already hold the run's claim lock; everything
    here commits or rolls back as one.
    """
    if (target_speaker_id is None) == (target_name is None):
        raise MergeError("choose exactly one target: an existing speaker or a new name")

    states = {s.label: s for s in label_states(session, run_id)}
    clean = _validate_labels(states, labels)

    # The expected-state snapshot must describe EXACTLY the labels being merged —
    # not a subset (which would let a drifted label slip through unchecked) and
    # not a superset (stale form). Anything else means the confirm no longer
    # matches its preview.
    if set(expected) != set(clean):
        raise MergeConflictError(
            "the confirm no longer matches its preview — refresh and re-preview"
        )

    # Optimistic concurrency, replay-aware ordering. The claim token proves
    # ownership, not that the ledger is unchanged since the preview — so on a
    # FRESH apply we verify the previewed rulings still hold. But a genuine
    # replay (double-click, network retry) arrives after this operation already
    # rewrote these labels to its OWN rulings, so the previewed expected-state no
    # longer matches; enforcing it then would 409 a legitimate retry. A replay is
    # ONLY when EVERY child ruling of this exact (nonce, label set) already
    # exists; a partial or single-row match (e.g. one pre-existing row) is NOT a
    # replay and must still be drift-checked, so a stale label can never slip
    # through on the back of one colliding row.
    existing_children = session.execute(
        select(func.count())
        .select_from(AdjudicationDecision)
        .where(
            AdjudicationDecision.idempotency_key.in_(
                [_child_key(nonce, clean, label) for label in clean]
            )
        )
    ).scalar_one()
    is_replay = existing_children == len(clean)
    if not is_replay:
        _check_expected(session, run_id, expected)

    created_speaker = False
    decision_ids: dict[str, uuid.UUID] = {}

    if target_speaker_id is not None:
        # FOR SHARE, exactly as the single-label assign route does: a concurrent
        # archive/merge takes FOR UPDATE, so the active check and the appends
        # below serialize with roster curation instead of racing it.
        survivor = session.execute(
            select(Speaker).where(Speaker.id == target_speaker_id).with_for_update(read=True)
        ).scalar_one_or_none()
        if survivor is None:
            raise MergeError(f"no speaker {target_speaker_id}")
        if not is_active(survivor):
            raise MergeError(
                f"speaker {survivor.display_name!r} is no longer an active roster "
                "identity — refresh and pick another"
            )
        assign_labels = clean
    else:
        # Enroll a new speaker from a label that actually has embedded turns to
        # build a centroid from, preferring the one with the most turns. Choosing
        # blindly by turn count could pick a label with NO eligible turns and 400
        # a merge another selected label could have enrolled. If none is eligible
        # the whole merge is refused cleanly (before any speaker is created).
        eligible = eligible_label_vectors(session, run_id, gates)
        enrollable = [label for label in clean if label in eligible]
        if not enrollable:
            raise MergeError(
                "none of the selected labels have speaker audio to create a new "
                "identity from — assign them to an existing speaker instead"
            )
        primary = max(enrollable, key=lambda label: states[label].turn_count)
        try:
            enrolled = enroll_new_speaker(
                session,
                run_id=run_id,
                diarization_label=primary,
                display_name=target_name or "",
                operator=operator,
                idempotency_key=_child_key(nonce, clean, primary),
                gates=gates,
                user_id=user_id,
            )
        except EnrollmentError as exc:
            raise MergeError(str(exc)) from exc
        survivor = session.get(Speaker, enrolled.speaker_id)
        assert survivor is not None
        created_speaker = enrolled.created_speaker
        decision_ids[primary] = enrolled.decision_id
        assign_labels = [label for label in clean if label != primary]

    for label in assign_labels:
        row = record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label=label,
            decision=Decision.ASSIGN,
            operator=operator,
            idempotency_key=_child_key(nonce, clean, label),
            speaker_id=survivor.id,
            user_id=user_id,
        )
        decision_ids[label] = row.id

    impacts, _ = _impacts(session, run_id, clean)
    return MergeResult(
        survivor_speaker_id=survivor.id,
        survivor_name=survivor.display_name,
        created_speaker=created_speaker,
        decision_ids=decision_ids,
        labels=clean,
        total_turns=sum(i.turn_count for i in impacts),
        total_segments=sum(i.segment_count for i in impacts),
        is_replay=is_replay,
    )
