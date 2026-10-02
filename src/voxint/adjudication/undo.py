"""Compensating undo for enroll and merge (issue #158).

Undo appends a REVOKE decision that voids the original ruling. The resolver
excludes both the voided row and the REVOKE row from effective_decisions, so
the pre-void effective state is restored automatically. No snapshot storage
needed.

Segment-scope rulings (issue #573) cannot carry a REVOKE: the schema keeps
REVOKE label-scope only, and the segment resolvers reduce newest-wins without
consulting voids. Their undo instead appends a compensating ruling in the same
scope that re-asserts what the scope followed before: the prior ASSIGN's
speaker, or INHERIT when there was no earlier ruling or the earlier one was
an INHERIT. Newest-wins then yields the pre-ruling state.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, aliased

from voxint.adjudication.ledger import ConflictingReplayError, record_decision
from voxint.adjudication.resolver import effective_decisions, newest_in_scope
from voxint.adjudication.splits import child_ranges
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    Speaker,
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.speakers.roster import (
    alias_ids,
    archive_speaker,
    canonicalize,
    is_active,
    merge_map,
)


class UndoError(Exception):
    """The requested action cannot be undone."""


class UndoDriftError(UndoError):
    """A label was re-ruled after the action."""


class UndoExpiredError(UndoError):
    """The undo grace window has passed."""


def _revoke_for(
    session: Session, decision_id: uuid.UUID
) -> AdjudicationDecision | None:
    return session.execute(
        select(AdjudicationDecision).where(
            AdjudicationDecision.decision == Decision.REVOKE.value,
            AdjudicationDecision.voids_decision_id == decision_id,
        )
    ).scalar_one_or_none()


def _has_live_decisions(session: Session, speaker_id: uuid.UUID) -> bool:
    """Whether any label- or segment-scope speaker ruling remains active.

    Rulings stored against a merged tombstone count for the identity it
    merged into. A label-scope ruling stays live until it is voided, because
    revoking a newer label ruling brings the older one back. A segment-scope
    ruling counts only while it is the newest in its exact scope (issue #718):
    a later ruling or an undo's compensating row supersedes it, and an undo
    that would re-assert it refuses an archived speaker (see
    :func:`undo_segment_decision`).
    """
    ruling = aliased(AdjudicationDecision)
    revoke = aliased(AdjudicationDecision)
    count = session.execute(
        select(func.count())
        .select_from(ruling)
        .where(
            ruling.speaker_id.in_(alias_ids(session, speaker_id)),
            ruling.detached_at.is_(None),
            or_(ruling.transcript_segment_id.is_(None), newest_in_scope(ruling)),
            ~select(revoke.id)
            .where(
                revoke.decision == Decision.REVOKE.value,
                revoke.voids_decision_id == ruling.id,
            )
            .exists(),
        )
    ).scalar_one()
    return bool(count)


def _archive_if_orphaned(
    session: Session,
    speaker_id: uuid.UUID | None,
) -> bool:
    if speaker_id is None:
        return False
    embedding_count = session.execute(
        select(func.count())
        .select_from(SpeakerEmbedding)
        .where(SpeakerEmbedding.speaker_id == speaker_id)
    ).scalar_one()
    if embedding_count or _has_live_decisions(session, speaker_id):
        return False
    speaker = session.get(Speaker, speaker_id)
    if speaker is None or speaker.merged_into_id is not None:
        return False
    archive_speaker(session, speaker_id)
    return True


def _append_revoke(
    session: Session,
    *,
    original: AdjudicationDecision,
    operator: str,
    idempotency_key: str,
    user_id: uuid.UUID | None,
) -> AdjudicationDecision:
    return record_decision(
        session,
        pipeline_run_id=original.pipeline_run_id,
        diarization_label=original.diarization_label,
        decision=Decision.REVOKE,
        operator=operator,
        idempotency_key=idempotency_key,
        voids_decision_id=original.id,
        user_id=user_id,
    )


def undo_enrollment(
    session: Session,
    run_id: uuid.UUID,
    decision_id: uuid.UUID,
    operator: str,
    idempotency_key: str,
    user_id: uuid.UUID | None = None,
) -> dict[str, object]:
    """Undo one label enrollment; the caller owns the transaction."""
    original = session.get(AdjudicationDecision, decision_id)
    if original is None:
        raise UndoError(f"no adjudication decision {decision_id}")
    if (
        original.pipeline_run_id != run_id
        or original.transcript_segment_id is not None
        or original.decision != Decision.ASSIGN.value
    ):
        raise UndoError("only a label-scope enrollment assignment from this run can be undone")
    if original.idempotency_key.startswith("merge:"):
        raise UndoError("use the merge undo endpoint")

    existing = _revoke_for(session, decision_id)
    if existing is not None:
        return {
            "revoke_decision_id": existing.id,
            "voided_decision_id": decision_id,
            "speaker_id": original.speaker_id,
            "speaker_archived": bool(
                original.speaker_id
                and (speaker := session.get(Speaker, original.speaker_id)) is not None
                and speaker.deleted_at is not None
            ),
            "is_replay": True,
        }

    current = effective_decisions(session, run_id).get(original.diarization_label)
    if current is None or current.id != original.id:
        raise UndoDriftError(
            f"label {original.diarization_label!r} changed after enrollment — refresh and retry"
        )

    revoke = _append_revoke(
        session,
        original=original,
        operator=operator,
        idempotency_key=idempotency_key,
        user_id=user_id,
    )
    embedding = session.execute(
        select(SpeakerEmbedding).where(
            SpeakerEmbedding.source_adjudication_decision_id == decision_id
        )
    ).scalar_one_or_none()
    if embedding is not None:
        session.delete(embedding)
    session.flush()
    # Only an enrollment-minted assignment has this provenance row. An
    # arbitrary ASSIGN may be revoked, but must never archive a pre-existing
    # roster identity merely because it currently has no embeddings.
    archived = embedding is not None and _archive_if_orphaned(
        session, original.speaker_id
    )
    session.flush()
    return {
        "revoke_decision_id": revoke.id,
        "voided_decision_id": decision_id,
        "speaker_id": original.speaker_id,
        "speaker_archived": archived,
        "is_replay": False,
    }


def undo_decision(
    session: Session,
    run_id: uuid.UUID,
    decision_id: uuid.UUID,
    operator: str,
    idempotency_key: str,
    grace_seconds: float,
    user_id: uuid.UUID | None = None,
) -> dict[str, object]:
    """Undo a plain label-scope decision; the caller owns the transaction."""
    original = session.get(AdjudicationDecision, decision_id)
    if original is None:
        raise UndoError(f"no adjudication decision {decision_id}")
    if (
        original.pipeline_run_id != run_id
        or original.transcript_segment_id is not None
        or original.decision not in (
            Decision.ASSIGN.value, Decision.EXCLUDE.value, Decision.UNKNOWN.value
        )
    ):
        raise UndoError("only a plain label-scope decision from this run can be undone")

    if original.idempotency_key.startswith("merge:"):
        raise UndoError("use the merge undo endpoint")
    embedding = session.execute(
        select(SpeakerEmbedding).where(
            SpeakerEmbedding.source_adjudication_decision_id == decision_id
        )
    ).scalar_one_or_none()
    if embedding is not None:
        raise UndoError("use the enrollment undo endpoint")

    existing = _revoke_for(session, decision_id)
    if existing is not None:
        return {
            "revoke_decision_id": existing.id,
            "voided_decision_id": decision_id,
            "is_replay": True,
        }

    created_at = original.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if created_at + timedelta(seconds=grace_seconds) <= datetime.now(UTC):
        raise UndoExpiredError("the undo grace window has passed")

    current = effective_decisions(session, run_id).get(original.diarization_label)
    if current is None or current.id != original.id:
        raise UndoDriftError(
            f"label {original.diarization_label!r} was re-ruled after this decision"
        )

    revoke = _append_revoke(
        session,
        original=original,
        operator=operator,
        idempotency_key=idempotency_key,
        user_id=user_id,
    )
    session.flush()
    return {
        "revoke_decision_id": revoke.id,
        "voided_decision_id": decision_id,
        "is_replay": False,
    }


def _check_grace(created_at: datetime, grace_seconds: float, message: str) -> None:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if created_at + timedelta(seconds=grace_seconds) <= datetime.now(UTC):
        raise UndoExpiredError(message)


def _scope_history(
    session: Session, original: AdjudicationDecision
) -> list[AdjudicationDecision]:
    """Every ruling in ``original``'s exact scope, newest first.

    The same reduction order as ``segment_states`` / ``word_range_states``
    (created_at, then id, descending), so ``[0]`` is what the read path applies.
    """
    start = AdjudicationDecision.start_word_index
    end = AdjudicationDecision.end_word_index
    return list(
        session.execute(
            select(AdjudicationDecision)
            .where(
                AdjudicationDecision.pipeline_run_id == original.pipeline_run_id,
                AdjudicationDecision.transcript_segment_id
                == original.transcript_segment_id,
                start.is_(None)
                if original.start_word_index is None
                else start == original.start_word_index,
                end.is_(None)
                if original.end_word_index is None
                else end == original.end_word_index,
            )
            .order_by(
                AdjudicationDecision.created_at.desc(),
                AdjudicationDecision.id.desc(),
            )
        ).scalars()
    )


def undo_segment_decision(
    session: Session,
    run_id: uuid.UUID,
    decision_id: uuid.UUID,
    operator: str,
    idempotency_key: str,
    grace_seconds: float,
    user_id: uuid.UUID | None = None,
) -> dict[str, object]:
    """Undo a segment- or word-range-scope relabel; the caller owns the transaction.

    Appends a compensating ruling in the original's exact scope (see the module
    docstring). Replaying the same ``idempotency_key`` returns the compensating
    row it already wrote, even after the grace window, so a retried request
    whose first attempt committed reports success.
    """
    original = session.get(AdjudicationDecision, decision_id)
    if original is None:
        raise UndoError(f"no adjudication decision {decision_id}")
    if original.pipeline_run_id != run_id:
        raise UndoError("only a segment-scope ruling from this run can be undone")
    if original.detached_at is not None:
        raise UndoDriftError("the segment this ruling applied to no longer exists")
    if original.transcript_segment_id is None or original.decision not in (
        Decision.ASSIGN.value,
        Decision.INHERIT.value,
    ):
        raise UndoError("only a segment-scope ruling from this run can be undone")

    history = _scope_history(session, original)
    position = next(i for i, row in enumerate(history) if row.id == original.id)
    prior = history[position + 1] if position + 1 < len(history) else None
    restore_speaker = (
        prior.speaker_id
        if prior is not None and prior.decision == Decision.ASSIGN.value
        else None
    )
    restore = Decision.ASSIGN if restore_speaker is not None else Decision.INHERIT

    existing = session.execute(
        select(AdjudicationDecision).where(
            AdjudicationDecision.idempotency_key == idempotency_key
        )
    ).scalar_one_or_none()
    if existing is not None:
        # A replay must be this original's compensation: newer than it in the
        # same scope and re-asserting exactly what preceded it. Anything else
        # is the key reused for a different ruling.
        later = {row.id for row in history[:position]}
        if (
            existing.id not in later
            or existing.decision != restore.value
            or existing.speaker_id != restore_speaker
        ):
            raise ConflictingReplayError(idempotency_key)
        return {
            "compensating_decision_id": existing.id,
            "undone_decision_id": decision_id,
            "is_replay": True,
        }

    _check_grace(original.created_at, grace_seconds, "the undo grace window has passed")

    if position != 0:
        raise UndoDriftError("this segment's speaker was changed again after this ruling")
    if restore_speaker is not None:
        # The relabel route only assigns active roster identities; an undo must
        # not re-assert one archived since (an enrollment undo can archive the
        # speaker a superseded ruling named). Check the canonical identity but
        # write the historical id, which the replay comparison above expects.
        restored = session.get(Speaker, canonicalize(restore_speaker, merge_map(session)))
        if restored is None or not is_active(restored):
            raise UndoDriftError(
                "the speaker this segment had before was removed from the roster"
            )
    segment = session.get(TranscriptSegment, original.transcript_segment_id)
    if segment is None:
        # Unreachable while the FK's ON DELETE SET NULL detaches rulings (0066);
        # fail closed rather than write a ruling nothing would apply.
        raise UndoDriftError("the segment this ruling applied to no longer exists")
    if (
        original.start_word_index is not None
        and (original.start_word_index, original.end_word_index)
        not in child_ranges(session, segment)
    ):
        raise UndoDriftError("this part of the segment was split again after this ruling")

    compensating = record_decision(
        session,
        pipeline_run_id=run_id,
        diarization_label=original.diarization_label,
        decision=restore,
        operator=operator,
        idempotency_key=idempotency_key,
        speaker_id=restore_speaker,
        transcript_segment_id=original.transcript_segment_id,
        start_word_index=original.start_word_index,
        end_word_index=original.end_word_index,
        user_id=user_id,
    )
    session.flush()
    return {
        "compensating_decision_id": compensating.id,
        "undone_decision_id": decision_id,
        "is_replay": False,
    }


def undo_merge(
    session: Session,
    run_id: uuid.UUID,
    merge_nonce: str,
    operator: str,
    idempotency_key: str,
    grace_seconds: float,
    user_id: uuid.UUID | None = None,
) -> dict[str, object]:
    """Undo every child ruling of a run-local merge; caller owns the transaction."""
    children = list(
        session.execute(
            select(AdjudicationDecision)
            .where(
                AdjudicationDecision.pipeline_run_id == run_id,
                AdjudicationDecision.idempotency_key.startswith(
                    f"merge:{merge_nonce}:", autoescape=True
                ),
            )
            .order_by(AdjudicationDecision.diarization_label)
        ).scalars()
    )
    if not children:
        raise UndoError(f"no merge {merge_nonce!r} exists for this run")
    if any(
        child.decision != Decision.ASSIGN.value
        or child.transcript_segment_id is not None
        for child in children
    ):
        raise UndoError("the merge contains an invalid child ruling")

    existing = {child.id: _revoke_for(session, child.id) for child in children}
    if all(revoke is not None for revoke in existing.values()):
        replay_archived_ids = [
            speaker_id
            for speaker_id in {
                child.speaker_id for child in children if child.speaker_id is not None
            }
            if (speaker := session.get(Speaker, speaker_id)) is not None
            and speaker.deleted_at is not None
        ]
        return {
            "merge_nonce": merge_nonce,
            "revoke_decision_ids": {
                child.diarization_label: revoke.id
                for child in children
                if (revoke := existing[child.id]) is not None
            },
            "archived_speaker_ids": replay_archived_ids,
            "is_replay": True,
        }

    now = datetime.now(UTC)
    for child in children:
        created_at = child.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if created_at + timedelta(seconds=grace_seconds) <= now:
            raise UndoExpiredError("the merge undo grace window has passed")

    current = effective_decisions(session, run_id)
    for child in children:
        actual = current.get(child.diarization_label)
        if actual is None or actual.id != child.id:
            raise UndoDriftError(
                f"label {child.diarization_label!r} changed after the merge — refresh and retry"
            )

    revokes: dict[str, AdjudicationDecision] = {}
    for child in children:
        revokes[child.diarization_label] = _append_revoke(
            session,
            original=child,
            operator=operator,
            idempotency_key=f"{idempotency_key}:{child.id}",
            user_id=user_id,
        )

    child_ids = [child.id for child in children]
    merge_embeddings = list(
        session.execute(
            select(SpeakerEmbedding).where(
                SpeakerEmbedding.source_adjudication_decision_id.in_(child_ids),
                SpeakerEmbedding.source_pipeline_run_id == run_id,
            )
        ).scalars()
    )
    created_speaker_ids = {embedding.speaker_id for embedding in merge_embeddings}
    for embedding in merge_embeddings:
        session.delete(embedding)
    session.flush()

    archived_ids: list[uuid.UUID] = []
    # An existing target can legitimately have no embeddings. Only the speaker
    # carrying a merge-child enrollment embedding was created by this merge.
    for speaker_id in created_speaker_ids:
        if _archive_if_orphaned(session, speaker_id):
            archived_ids.append(speaker_id)
    session.flush()
    return {
        "merge_nonce": merge_nonce,
        "revoke_decision_ids": {
            label: revoke.id for label, revoke in revokes.items()
        },
        "archived_speaker_ids": archived_ids,
        "is_replay": False,
    }
