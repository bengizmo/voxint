"""Compensating undo for enroll and merge (issue #158).

Undo appends a REVOKE decision that voids the original ruling. The resolver
excludes both the voided row and the REVOKE row from effective_decisions, so
the pre-void effective state is restored automatically. No snapshot storage
needed.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from voxint.adjudication.ledger import record_decision
from voxint.adjudication.resolver import effective_decisions
from voxint.db.models import (
    AdjudicationDecision,
    Decision,
    Speaker,
    SpeakerEmbedding,
)
from voxint.speakers.roster import archive_speaker


class UndoError(Exception):
    """The requested action cannot be undone."""


class UndoDriftError(UndoError):
    """A label was re-ruled after the action."""


class UndoExpiredError(UndoError):
    """The merge undo grace window has passed."""


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
    """Whether any label- or segment-scope speaker ruling remains active."""
    ruling = aliased(AdjudicationDecision)
    revoke = aliased(AdjudicationDecision)
    count = session.execute(
        select(func.count())
        .select_from(ruling)
        .where(
            ruling.speaker_id == speaker_id,
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
