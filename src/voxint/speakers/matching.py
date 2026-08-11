"""Cosine speaker matching + the single proposal writer.

Everything vector-shaped lives here — turn eligibility, label centroids, roster
centroids, similarity, gates — and every ``speaker_assignments`` row in the
system is written through :func:`replace_run_proposals`. Both queries filter by
``embedding_space``: vectors from different spaces are never compared (the
invariant named in docs/architecture.md).

Method (v1, thresholds in docs/quality-gates.md):

- Label side: eligible turns (embedding present, overlap ratio within gate) are
  L2-normalized and averaged, weighted by usable non-overlap seconds capped per
  turn, then re-normalized.
- Roster side: each speaker's enrollment embeddings (same space) are normalized,
  averaged, re-normalized — one centroid per speaker, never max-over-rows,
  which would let one anomalous enrollment dominate.
- Decision: top-1 cosine must clear the similarity gate, the top-1 vs top-2
  margin gate, and a duration-weighted per-turn vote-agreement gate. Grounding
  applies the same shape of gates with stricter values.
- Unmatched or ineligible labels produce **no row** — absence of evidence is
  not a low-confidence proposal.
"""

import logging
import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from voxint.db.models import (
    AssignmentMethod,
    DiarizationTurn,
    SpeakerAssignment,
    SpeakerEmbedding,
)

if TYPE_CHECKING:
    from voxint.config import Settings

logger = logging.getLogger(__name__)

MAX_PROPOSED_NAME_LENGTH = 120


@dataclass(frozen=True)
class MatchingGates:
    """Eligibility and acceptance thresholds; defaults mirror ``Settings``."""

    max_overlap_ratio: float = 0.20
    turn_weight_cap_seconds: float = 10.0
    min_turns: int = 2
    min_seconds: float = 6.0
    min_cosine: float = 0.60
    min_margin: float = 0.05
    min_vote_agreement: float = 0.60
    grounded_min_turns: int = 3
    grounded_min_seconds: float = 10.0
    grounded_min_cosine: float = 0.70
    grounded_min_margin: float = 0.08
    grounded_min_vote_agreement: float = 0.67


def gates_from_settings(settings: "Settings") -> MatchingGates:
    """The one Settings -> MatchingGates mapping (worker and review UI share it)."""
    return MatchingGates(
        max_overlap_ratio=settings.match_max_overlap_ratio,
        turn_weight_cap_seconds=settings.match_turn_weight_cap_seconds,
        min_turns=settings.match_min_turns,
        min_seconds=settings.match_min_seconds,
        min_cosine=settings.match_min_cosine,
        min_margin=settings.match_min_margin,
        min_vote_agreement=settings.match_min_vote_agreement,
        grounded_min_turns=settings.grounded_min_turns,
        grounded_min_seconds=settings.grounded_min_seconds,
        grounded_min_cosine=settings.grounded_min_cosine,
        grounded_min_margin=settings.grounded_min_margin,
        grounded_min_vote_agreement=settings.grounded_min_vote_agreement,
    )


@dataclass(frozen=True)
class CosineProposal:
    diarization_label: str
    speaker_id: uuid.UUID
    similarity: float  # raw cosine, [-1, 1]
    margin: float  # top-1 minus top-2 similarity (inf with a 1-speaker roster)
    vote_agreement: float  # duration-weighted fraction of turns voting top-1
    grounded: bool


@dataclass(frozen=True)
class NameHintProposal:
    diarization_label: str
    proposed_name: str


class ProposalError(Exception):
    """A proposal violates a matching invariant — a pipeline bug, never data."""


def confidence_from_similarity(similarity: float) -> float:
    """Map raw cosine [-1, 1] to the stored [0, 1] score. A transformed
    similarity, NOT a calibrated probability — documented as such."""
    return min(1.0, max(0.0, (similarity + 1.0) / 2.0))


def eligible_label_vectors(
    session: Session, run_id: uuid.UUID, gates: MatchingGates
) -> dict[str, tuple[str, list[tuple[np.ndarray, float]]]]:
    """Per label: its embedding space and eligible (unit vector, usable seconds)
    pairs. The single definition of turn eligibility — matching and P5 speaker
    enrollment both build centroids from exactly this."""
    turns = (
        session.execute(
            select(DiarizationTurn)
            .where(
                DiarizationTurn.pipeline_run_id == run_id,
                DiarizationTurn.embedding.is_not(None),
            )
            .order_by(DiarizationTurn.turn_index)
        )
        .scalars()
        .all()
    )

    # label -> (space, [(unit vector, usable non-overlap seconds)])
    by_label: dict[str, tuple[str, list[tuple[np.ndarray, float]]]] = {}
    mixed_space_labels: set[str] = set()
    for turn in turns:
        duration = turn.end_seconds - turn.start_seconds
        if duration <= 0 or turn.overlap_seconds / duration > gates.max_overlap_ratio:
            continue
        vector = _unit(np.asarray(turn.embedding, dtype=np.float64))
        if vector is None:
            continue  # zero vector — nothing to compare
        usable = duration - turn.overlap_seconds  # > 0: overlap ratio gate passed
        space = turn.embedding_space
        assert space is not None  # DB CHECK: embedding implies embedding_space
        entry = by_label.setdefault(turn.label, (space, []))
        if entry[0] != space:
            # One label spanning two spaces means mixed embedder output — a
            # pipeline bug; drop the label rather than blend spaces.
            mixed_space_labels.add(turn.label)
            continue
        entry[1].append((vector, usable))
    for label in mixed_space_labels:
        logger.error("run %s label %s spans embedding spaces; skipped", run_id, label)
        del by_label[label]
    return by_label


def label_centroid(
    entries: list[tuple[np.ndarray, float]], cap_seconds: float
) -> np.ndarray | None:
    """Duration-weighted unit centroid of a label's eligible vectors (per-turn
    weight capped so one long monologue can't dominate)."""
    weighted = [(v, min(usable, cap_seconds)) for v, usable in entries]
    return _unit(
        sum((v * w for v, w in weighted), start=np.zeros_like(weighted[0][0]))
    )


def match_speakers(
    session: Session, run_id: uuid.UUID, gates: MatchingGates
) -> tuple[CosineProposal, ...]:
    """Propose roster speakers for this run's diarization labels."""
    by_label = eligible_label_vectors(session, run_id, gates)

    rosters: dict[str, dict[uuid.UUID, np.ndarray]] = {}
    proposals: list[CosineProposal] = []
    for label in sorted(by_label):
        space, entries = by_label[label]
        if len(entries) < gates.min_turns:
            continue
        if sum(usable for _, usable in entries) < gates.min_seconds:
            continue
        if space not in rosters:
            rosters[space] = _roster_centroids(session, space)
        roster = rosters[space]
        if not roster:
            continue

        weighted = [
            (v, min(usable, gates.turn_weight_cap_seconds)) for v, usable in entries
        ]
        centroid = label_centroid(entries, gates.turn_weight_cap_seconds)
        if centroid is None:
            continue

        # Deterministic ordering: similarity desc, then speaker id.
        ranked = sorted(
            ((float(centroid @ c), sid) for sid, c in roster.items()),
            key=lambda pair: (-pair[0], pair[1]),
        )
        top_sim, top_speaker = ranked[0]
        top_sim = max(-1.0, min(1.0, top_sim))  # unit-vector dot, modulo float error
        margin = top_sim - ranked[1][0] if len(ranked) > 1 else math.inf

        agree_weight = sum(w for v, w in weighted if _nearest(v, roster) == top_speaker)
        vote_agreement = agree_weight / sum(w for _, w in weighted)

        accepted = (
            top_sim >= gates.min_cosine
            and margin >= gates.min_margin
            and vote_agreement >= gates.min_vote_agreement
        )
        # Near-misses matter for threshold calibration — keep the numbers.
        logger.debug(
            "run %s label %s: cosine=%.4f margin=%.4f agreement=%.4f -> %s",
            run_id,
            label,
            top_sim,
            margin,
            vote_agreement,
            "proposed" if accepted else "rejected",
        )
        if not accepted:
            continue
        grounded = (
            len(entries) >= gates.grounded_min_turns
            and sum(usable for _, usable in entries) >= gates.grounded_min_seconds
            and top_sim >= gates.grounded_min_cosine
            and margin >= gates.grounded_min_margin
            and vote_agreement >= gates.grounded_min_vote_agreement
        )
        proposals.append(
            CosineProposal(
                diarization_label=label,
                speaker_id=top_speaker,
                similarity=top_sim,
                margin=margin,
                vote_agreement=vote_agreement,
                grounded=grounded,
            )
        )
    return tuple(proposals)


def replace_run_proposals(
    session: Session,
    run_id: uuid.UUID,
    cosine: tuple[CosineProposal, ...],
    name_hints: tuple[NameHintProposal, ...],
) -> None:
    """The single choke point writing ``speaker_assignments`` — validates every
    proposal against the matching invariants, then delete-then-inserts the
    run's rows (idempotent under stage retry)."""
    known_labels = set(
        session.execute(
            select(DiarizationTurn.label).where(DiarizationTurn.pipeline_run_id == run_id)
        )
        .scalars()
        .all()
    )
    seen: set[tuple[str, str]] = set()
    proposals: list[CosineProposal | NameHintProposal] = [*cosine, *name_hints]
    for proposal in proposals:
        method = (
            AssignmentMethod.COSINE
            if isinstance(proposal, CosineProposal)
            else AssignmentMethod.LLM_HINT
        )
        key = (proposal.diarization_label, method.value)
        if key in seen:
            raise ProposalError(f"duplicate proposal for {key}")
        seen.add(key)
        if proposal.diarization_label not in known_labels:
            raise ProposalError(
                f"label {proposal.diarization_label!r} not in run {run_id}'s turn ledger"
            )
    for cp in cosine:
        if not (math.isfinite(cp.similarity) and -1.0 <= cp.similarity <= 1.0):
            raise ProposalError(f"similarity out of range: {cp.similarity}")
        if not (0.0 <= cp.vote_agreement <= 1.0):
            raise ProposalError(f"vote_agreement out of range: {cp.vote_agreement}")
    for hint in name_hints:
        name = hint.proposed_name.strip()
        if not name or len(name) > MAX_PROPOSED_NAME_LENGTH:
            raise ProposalError(f"invalid proposed name: {hint.proposed_name!r:.200}")

    session.execute(
        delete(SpeakerAssignment).where(SpeakerAssignment.pipeline_run_id == run_id)
    )
    for cp in cosine:
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label=cp.diarization_label,
                speaker_id=cp.speaker_id,
                method=AssignmentMethod.COSINE.value,
                confidence=confidence_from_similarity(cp.similarity),
                proposed_name=None,
                grounded=cp.grounded,
            )
        )
    for hint in name_hints:
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run_id,
                diarization_label=hint.diarization_label,
                speaker_id=None,
                method=AssignmentMethod.LLM_HINT.value,
                confidence=None,  # LLM self-reported confidence is not calibrated
                proposed_name=hint.proposed_name.strip(),
                grounded=False,
            )
        )


def _roster_centroids(session: Session, space: str) -> dict[uuid.UUID, np.ndarray]:
    """One unit centroid per enrolled speaker, within one embedding space."""
    rows = session.execute(
        select(SpeakerEmbedding.speaker_id, SpeakerEmbedding.embedding).where(
            SpeakerEmbedding.embedding_space == space
        )
    ).all()
    grouped: dict[uuid.UUID, list[np.ndarray]] = {}
    for speaker_id, embedding in rows:
        vector = _unit(np.asarray(embedding, dtype=np.float64))
        if vector is not None:
            grouped.setdefault(speaker_id, []).append(vector)
    centroids: dict[uuid.UUID, np.ndarray] = {}
    for speaker_id, vectors in grouped.items():
        centroid = _unit(np.mean(vectors, axis=0))
        if centroid is not None:
            centroids[speaker_id] = centroid
    return centroids


def _nearest(vector: np.ndarray, roster: dict[uuid.UUID, np.ndarray]) -> uuid.UUID:
    """This turn's nearest roster speaker; ties break on speaker id."""
    return min(
        ((float(vector @ c), sid) for sid, c in roster.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )[1]


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    return vector / norm
