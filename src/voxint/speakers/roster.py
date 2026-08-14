"""Roster curation: rename, merge, archive/restore, remove embeddings (issue #7).

The append-only ``adjudication_decisions`` ledger is never written here — every
operation curates the mutable side of the roster only:

- **Merge** repoints the source's embeddings and machine assignments to the
  target and keeps the source row as a tombstone (``merged_into_id``), so
  historical ledger FKs stay valid and readers canonicalize through
  :func:`merge_map`. Writes collapse chains to depth 1; readers still follow
  chains defensively and fail loudly on a cycle instead of misattributing.
- **Archive** is reversible (``deleted_at``) and deletes the speaker's cosine
  assignments — stale machine grounding must not survive the operator saying
  "this identity is wrong". Embeddings and human decisions are preserved.
- **Removing an embedding** hard-deletes the derived centroid row (the ledger
  decision and the raw ``diarization_turns`` vectors survive, so it is fully
  re-derivable) and deletes the speaker's cosine assignments — assignments do
  not record centroid lineage, so narrower invalidation would be a guess.

The *active* predicate (not merged, not archived) defined here is the single
definition used by matching, the workbench dropdown, and the decide route.
Callers own the transaction, exactly like ``adjudication.enrollment``.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
from sqlalchemy import ColumnElement, CursorResult, and_, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import (
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
)

logger = logging.getLogger(__name__)

MAX_DISPLAY_NAME_LENGTH = 120


class RosterError(Exception):
    """The operation cannot proceed as requested. Operator-visible: the API
    re-renders the roster with this message inline (htmx swaps need a 2xx)."""


class RosterConflictError(RosterError):
    """The roster changed under the operator's form — rendered inline like any
    ``RosterError``; the refreshed roster the response carries IS the retry."""


class RosterNotFoundError(RosterError):
    """The referenced speaker or embedding does not exist (HTTP 404)."""


def normalize_display_name(raw: str) -> str:
    """The one trim + length rule for speaker names (enrollment shares it)."""
    name = raw.strip()
    if not name or len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise ValueError("display name must be 1-120 characters")
    return name


def active_speaker_clause() -> ColumnElement[bool]:
    """SQL predicate for a curatable, matchable roster identity."""
    return and_(Speaker.merged_into_id.is_(None), Speaker.deleted_at.is_(None))


def is_active(speaker: Speaker) -> bool:
    return speaker.merged_into_id is None and speaker.deleted_at is None


def active_speakers(session: Session) -> list[Speaker]:
    """Active roster identities, display-name order — dropdowns and matching."""
    return list(
        session.execute(
            select(Speaker).where(active_speaker_clause()).order_by(Speaker.display_name)
        ).scalars()
    )


def describe_name_owner(owner: Speaker) -> str:
    """Operator guidance when a name is taken: what owns it and what to do.

    Names are globally unique across every lifecycle state — the fix is always
    restore or merge, never re-creating the identity.
    """
    name = owner.display_name
    if owner.deleted_at is not None:
        return f"speaker {name!r} is archived — restore it instead of re-creating it"
    if owner.merged_into_id is not None:
        return f"speaker {name!r} was merged — use the merge target instead"
    return f"speaker {name!r} already exists"


def searchable_speakers(session: Session) -> list[Speaker]:
    """Canonical identities for the runs search facet: active, then archived.

    Archived speakers stay listed (the UI marks them) because their human
    decisions remain effective — hiding them would make those runs
    undiscoverable. Merge tombstones are excluded; their history is found
    through the canonical target via ``alias_ids``.
    """
    return list(
        session.execute(
            select(Speaker)
            .where(Speaker.merged_into_id.is_(None))
            .order_by(Speaker.deleted_at.is_not(None), Speaker.display_name)
        ).scalars()
    )


def merge_map(session: Session) -> dict[uuid.UUID, uuid.UUID]:
    """Every tombstone's target, for read-time canonicalization."""
    return {
        source: target
        for source, target in session.execute(
            select(Speaker.id, Speaker.merged_into_id).where(
                Speaker.merged_into_id.is_not(None)
            )
        ).tuples()
        if target is not None  # guaranteed by the WHERE; narrows the type
    }


def canonicalize(
    speaker_id: uuid.UUID, mapping: dict[uuid.UUID, uuid.UUID]
) -> uuid.UUID:
    """Follow merge tombstones to the canonical identity.

    Writes keep chains at depth 1, but readers follow arbitrary chains
    defensively — historical rows and future migrations make a one-hop
    assumption fragile. A cycle is corrupt data: fail loudly, never
    misattribute silently.
    """
    current = speaker_id
    visited = {current}
    while current in mapping:
        current = mapping[current]
        if current in visited:
            raise RosterError(f"speaker merge chain contains a cycle at {current}")
        visited.add(current)
    return current


def alias_ids(session: Session, speaker_id: uuid.UUID) -> set[uuid.UUID]:
    """Every id whose merge chain lands on ``speaker_id``'s canonical identity.

    The inverse of ``canonicalize``, for SQL predicates that compare stored
    ``speaker_id`` columns against a search target: historical ledger rows
    keep the merged source's id (canonicalization is presentation, never a
    rewrite), so "attributed to X" must match X plus every tombstone that
    canonicalizes into X — chain-safe, like the reader side. The input is
    itself canonicalized first, so a stale reference to a since-merged
    speaker resolves to the same set as its target.
    """
    mapping = merge_map(session)
    target = canonicalize(speaker_id, mapping)
    return {target} | {
        source for source in mapping if canonicalize(source, mapping) == target
    }


@dataclass(frozen=True)
class EmbeddingInfo:
    """One enrollment centroid with its provenance, for the roster page."""

    id: uuid.UUID
    embedding_space: str
    created_at: datetime
    source_pipeline_run_id: uuid.UUID | None
    source_diarization_label: str | None
    # Loaded only for ACTIVE speakers (voiceprint rendering); None on tombstones
    # and archived entries, whose vectors the page never shows.
    vector: tuple[float, ...] | None


@dataclass(frozen=True)
class RosterEntry:
    speaker: Speaker
    embeddings: tuple[EmbeddingInfo, ...]
    assignment_count: int
    last_seen_at: datetime | None
    # Set on tombstones: the display name of the merge target.
    merged_into_name: str | None


@dataclass(frozen=True)
class RosterOverview:
    active: tuple[RosterEntry, ...]
    inactive: tuple[RosterEntry, ...]


def roster_overview(session: Session) -> RosterOverview:
    """Everything the roster page shows, active first, in display-name order."""
    speakers = list(
        session.execute(select(Speaker).order_by(Speaker.display_name)).scalars()
    )
    by_id = {s.id: s for s in speakers}

    # Vectors are heavy (192 floats each) and only the active entries' voiceprints
    # consume them — fetch provenance for everyone, vectors for active speakers only.
    vectors: dict[uuid.UUID, tuple[float, ...]] = {
        embedding_id: tuple(float(v) for v in embedding)
        for embedding_id, embedding in session.execute(
            select(SpeakerEmbedding.id, SpeakerEmbedding.embedding)
            .join(Speaker, Speaker.id == SpeakerEmbedding.speaker_id)
            .where(active_speaker_clause())
        )
    }
    embeddings: dict[uuid.UUID, list[EmbeddingInfo]] = {}
    for row in session.execute(
        select(
            SpeakerEmbedding.id,
            SpeakerEmbedding.speaker_id,
            SpeakerEmbedding.embedding_space,
            SpeakerEmbedding.created_at,
            SpeakerEmbedding.source_pipeline_run_id,
            SpeakerEmbedding.source_diarization_label,
        ).order_by(SpeakerEmbedding.created_at)
    ):
        embeddings.setdefault(row.speaker_id, []).append(
            EmbeddingInfo(
                id=row.id,
                embedding_space=row.embedding_space,
                created_at=row.created_at,
                source_pipeline_run_id=row.source_pipeline_run_id,
                source_diarization_label=row.source_diarization_label,
                vector=vectors.get(row.id),
            )
        )

    assignment_stats = {
        speaker_id: (int(count), last_seen)
        for speaker_id, count, last_seen in session.execute(
            select(
                SpeakerAssignment.speaker_id,
                func.count(),
                func.max(SpeakerAssignment.created_at),
            )
            .where(SpeakerAssignment.speaker_id.is_not(None))
            .group_by(SpeakerAssignment.speaker_id)
        )
    }

    active: list[RosterEntry] = []
    inactive: list[RosterEntry] = []
    for speaker in speakers:
        count, assignment_seen = assignment_stats.get(speaker.id, (0, None))
        own_embeddings = tuple(embeddings.get(speaker.id, ()))
        seen_candidates = [assignment_seen] + [e.created_at for e in own_embeddings]
        seen = [t for t in seen_candidates if t is not None]
        target = by_id.get(speaker.merged_into_id) if speaker.merged_into_id else None
        entry = RosterEntry(
            speaker=speaker,
            embeddings=own_embeddings,
            assignment_count=count,
            last_seen_at=max(seen) if seen else None,
            merged_into_name=target.display_name if target else None,
        )
        (active if is_active(speaker) else inactive).append(entry)
    return RosterOverview(active=tuple(active), inactive=tuple(inactive))


def _require_speaker(
    session: Session, speaker_id: uuid.UUID, *, for_update: bool = False
) -> Speaker:
    """Fetch a speaker or raise. ``for_update`` takes the row lock every
    lifecycle mutation must hold, so archive/restore/embedding-delete serialize
    with merge (which locks the same rows) instead of racing the not-merged-
    and-deleted CHECK into an IntegrityError 500."""
    if for_update:
        session.execute(
            select(Speaker.id).where(Speaker.id == speaker_id).with_for_update()
        )
    speaker = session.get(Speaker, speaker_id)
    if speaker is None:
        raise RosterNotFoundError("speaker not found")
    return speaker


def rename_speaker(session: Session, speaker_id: uuid.UUID, new_name: str) -> Speaker:
    """Rename an active speaker. Pure metadata — decisions store ids, never
    names, so historical attribution re-renders under the new name."""
    speaker = _require_speaker(session, speaker_id, for_update=True)
    if not is_active(speaker):
        raise RosterError("only active speakers can be renamed")
    try:
        name = normalize_display_name(new_name)
    except ValueError as exc:
        raise RosterError(str(exc)) from exc
    if name == speaker.display_name:
        return speaker

    owner = session.execute(
        select(Speaker).where(Speaker.display_name == name, Speaker.id != speaker.id)
    ).scalar_one_or_none()
    if owner is not None:
        raise RosterConflictError(describe_name_owner(owner))
    try:
        # Savepoint: the pre-check races a concurrent write; the unique index
        # decides and the loser gets the same operator-visible conflict.
        with session.begin_nested():
            speaker.display_name = name
    except IntegrityError as exc:
        raise RosterConflictError(f"speaker {name!r} already exists") from exc
    return speaker


@dataclass(frozen=True)
class MergeResult:
    source_id: uuid.UUID
    target_id: uuid.UUID
    embeddings_moved: int
    assignments_moved: int
    aliases_collapsed: int
    already_merged: bool


def merge_speakers(
    session: Session, source_id: uuid.UUID, target_id: uuid.UUID
) -> MergeResult:
    """Merge the source identity into the target ("these were always the same
    person"): embeddings and machine assignments move; the ledger is untouched;
    the source stays as a tombstone that readers canonicalize through."""
    if source_id == target_id:
        raise RosterError("a speaker cannot be merged into itself")

    # Deterministic lock order so two concurrent merges can't deadlock.
    for lock_id in sorted((source_id, target_id)):
        session.execute(
            select(Speaker.id).where(Speaker.id == lock_id).with_for_update()
        )
    source = _require_speaker(session, source_id)
    target = _require_speaker(session, target_id)

    if source.merged_into_id is not None:
        # Replay of a completed merge is success; merged elsewhere is a stale form.
        if canonicalize(source_id, merge_map(session)) == target_id:
            return MergeResult(
                source_id=source_id,
                target_id=target_id,
                embeddings_moved=0,
                assignments_moved=0,
                aliases_collapsed=0,
                already_merged=True,
            )
        raise RosterConflictError(
            f"{source.display_name!r} was already merged into a different speaker"
        )
    if source.deleted_at is not None:
        raise RosterError(
            f"{source.display_name!r} is archived — restore it before merging"
        )
    if not is_active(target):
        raise RosterConflictError(
            f"{target.display_name!r} is no longer an active speaker — refresh and retry"
        )

    embeddings_moved = cast(CursorResult[Any], session.execute(
        update(SpeakerEmbedding)
        .where(SpeakerEmbedding.speaker_id == source_id)
        .values(speaker_id=target_id)
    )).rowcount
    assignments_moved = cast(CursorResult[Any], session.execute(
        update(SpeakerAssignment)
        .where(SpeakerAssignment.speaker_id == source_id)
        .values(speaker_id=target_id)
    )).rowcount
    now = datetime.now(tz=UTC)
    # Collapse existing aliases of the source so chains stay depth 1.
    aliases_collapsed = cast(CursorResult[Any], session.execute(
        update(Speaker)
        .where(Speaker.merged_into_id == source_id)
        .values(merged_into_id=target_id)
    )).rowcount
    source.merged_into_id = target_id
    source.merged_at = now
    session.flush()
    logger.info(
        "merged speaker %s (%r) into %s (%r): %d embeddings, %d assignments, %d aliases",
        source_id,
        source.display_name,
        target_id,
        target.display_name,
        embeddings_moved,
        assignments_moved,
        aliases_collapsed,
    )
    return MergeResult(
        source_id=source_id,
        target_id=target_id,
        embeddings_moved=embeddings_moved,
        assignments_moved=assignments_moved,
        aliases_collapsed=aliases_collapsed,
        already_merged=False,
    )


def archive_speaker(session: Session, speaker_id: uuid.UUID) -> int:
    """Reversibly archive a speaker and delete its cosine assignments (stale
    machine grounding must not survive). Returns assignments deleted."""
    speaker = _require_speaker(session, speaker_id, for_update=True)
    if speaker.merged_into_id is not None:
        raise RosterError("merged speakers are historical tombstones — nothing to archive")
    if speaker.deleted_at is not None:
        return 0  # replayed archive — already done
    speaker.deleted_at = datetime.now(tz=UTC)
    deleted = cast(
        CursorResult[Any],
        session.execute(
            delete(SpeakerAssignment).where(SpeakerAssignment.speaker_id == speaker_id)
        ),
    ).rowcount
    session.flush()
    logger.info(
        "archived speaker %s (%r): deleted %d machine assignments",
        speaker_id,
        speaker.display_name,
        deleted,
    )
    return deleted


def restore_speaker(session: Session, speaker_id: uuid.UUID) -> Speaker:
    """Reverse an archive. Deleted machine assignments are NOT resurrected —
    matching re-proposes on future runs."""
    speaker = _require_speaker(session, speaker_id, for_update=True)
    if speaker.merged_into_id is not None:
        raise RosterError("merged speakers cannot be restored — they live on in the target")
    if speaker.deleted_at is None:
        return speaker  # replayed restore — already active
    speaker.deleted_at = None
    session.flush()
    return speaker


@dataclass(frozen=True)
class EmbeddingRemoval:
    embedding_id: uuid.UUID
    speaker_id: uuid.UUID
    embedding_space: str
    assignments_deleted: int
    remaining_in_space: int


def delete_embedding(
    session: Session, speaker_id: uuid.UUID, embedding_id: uuid.UUID
) -> EmbeddingRemoval:
    """Remove one bad enrollment centroid.

    Safe to hard-delete: the ledger decision that minted it survives (append
    only), the raw turn vectors survive in ``diarization_turns``, and a replayed
    enrollment POST returns at the idempotency check before ever re-minting.
    All of the speaker's cosine assignments are deleted with it — they may have
    been grounded through this centroid and carry no lineage to prove otherwise.
    """
    speaker = _require_speaker(session, speaker_id, for_update=True)
    embedding = session.get(SpeakerEmbedding, embedding_id)
    if embedding is None or embedding.speaker_id != speaker_id:
        raise RosterNotFoundError("embedding not found for this speaker")

    space = embedding.embedding_space
    provenance = (
        embedding.source_pipeline_run_id,
        embedding.source_diarization_label,
        embedding.source_adjudication_decision_id,
    )
    session.delete(embedding)
    assignments_deleted = cast(
        CursorResult[Any],
        session.execute(
            delete(SpeakerAssignment).where(SpeakerAssignment.speaker_id == speaker_id)
        ),
    ).rowcount
    session.flush()
    remaining = int(session.execute(
        select(func.count())
        .select_from(SpeakerEmbedding)
        .where(
            SpeakerEmbedding.speaker_id == speaker_id,
            SpeakerEmbedding.embedding_space == space,
        )
    ).scalar_one())
    logger.info(
        "removed embedding %s of speaker %s (%r) in space %s"
        " (source run=%s label=%r decision=%s): deleted %d assignments, %d left in space",
        embedding_id,
        speaker_id,
        speaker.display_name,
        space,
        *provenance,
        assignments_deleted,
        remaining,
    )
    return EmbeddingRemoval(
        embedding_id=embedding_id,
        speaker_id=speaker_id,
        embedding_space=space,
        assignments_deleted=assignments_deleted,
        remaining_in_space=remaining,
    )


def voiceprint_bars(
    entries: tuple[EmbeddingInfo, ...] | list[EmbeddingInfo], bars: int = 48
) -> list[float] | None:
    """Deterministic bar heights (0..1) derived from a speaker's centroid.

    Presentation, not comparison — but the one-space rule still applies: a
    speaker enrolled in several spaces gets its strip from the space with the
    most embeddings (ties break lexicographically), never from a cross-space
    mean, so "same voice → same strip" holds within the chosen space.
    Entries without a loaded vector (inactive speakers) are ignored.
    """
    with_vectors = [e for e in entries if e.vector is not None]
    if not with_vectors:
        return None
    by_space: dict[str, list[EmbeddingInfo]] = {}
    for entry in with_vectors:
        by_space.setdefault(entry.embedding_space, []).append(entry)
    chosen = min(by_space, key=lambda space: (-len(by_space[space]), space))
    vectors = np.asarray([e.vector for e in by_space[chosen]], dtype=np.float64)
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0.0:
        mean = mean / norm
    heights = np.asarray([float(np.abs(c).mean()) for c in np.array_split(mean, bars)])
    peak = float(heights.max())
    if peak <= 0.0:
        return [0.0] * bars
    return [float(h / peak) for h in heights]
