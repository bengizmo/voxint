"""Qualitative voice-match tiers from the recorded match evidence (issue #159).

Turns ``match_candidates`` diagnostics into the strong / moderate / weak chips
the speakers pages show, graded against the SAME gates the matcher enforces
(``MatchingGates``) so the words and the pipeline can never disagree. The tier
describes VOICE evidence only — the verified badge (a human assignment) is a
separate, orthogonal fact.

Semantics the UI copy must respect:

- Evidence unit = one SURVIVING grounded appearance: a (run, label) pair where
  grounded cosine actually won transcript intervals (the aggregate fold's
  ``grounded_keys``) — a label fully overridden by human rulings contributes
  nothing.
- The per-speaker fold is **best appearance wins** and must be labeled
  "strongest voice evidence", never "overall confidence"; the reveal reports
  every appearance's numbers and the grade counts, so one strong sample cannot
  quietly hide many weak ones.
- A grounded appearance from a run predating migration 0032 has NO diagnostics
  row: that is ``unavailable`` evidence, never "weak" (weak = numbers exist
  and fail today's gates).
- A NULL margin means a one-speaker roster (top-2 undefined, margin infinite):
  it PASSES the margin gate iff ``roster_size == 1``; any other NULL numeric
  makes the appearance unavailable.

``tier_for`` is the single seam a calibrated policy (issue #114) replaces.
"""

import enum
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session

from voxint.db.models import MatchCandidate
from voxint.speakers.matching import MatchingGates


class MatchTier(enum.StrEnum):
    STRONG = "strong"  # clears the grounded gate
    MODERATE = "moderate"  # clears only the accept gate
    WEAK = "weak"  # recorded numbers below today's gates


@dataclass(frozen=True)
class TierEvidence:
    """One surviving grounded appearance's recorded numbers (or their absence)."""

    run_id: uuid.UUID
    label: str
    available: bool
    similarity: float | None = None
    margin: float | None = None  # None = one-speaker roster (infinite margin)
    vote_agreement: float | None = None
    eligible_turns: int = 0
    eligible_seconds: float = 0.0
    roster_size: int | None = None


@dataclass(frozen=True)
class TierSummary:
    """The chip plus the honest breakdown behind the reveal."""

    tier: MatchTier | None  # None = no gradable voice evidence
    strong: int
    moderate: int
    weak: int
    unavailable: int
    evidence: tuple[TierEvidence, ...]

    @property
    def has_voice_evidence(self) -> bool:
        return bool(self.evidence)


def evidence_for(
    session: Session, grounded_keys: Sequence[tuple[uuid.UUID, str]]
) -> list[TierEvidence]:
    """Load the diagnostics for surviving grounded appearances, one batch query.

    A key with no ``match_candidates`` row (pre-0032 run) yields an
    ``available=False`` evidence entry — visible, never silently dropped.
    """
    if not grounded_keys:
        return []
    rows = {
        (row.pipeline_run_id, row.diarization_label): row
        for row in session.execute(
            select(MatchCandidate).where(
                tuple_(
                    MatchCandidate.pipeline_run_id, MatchCandidate.diarization_label
                ).in_(list(grounded_keys))
            )
        ).scalars()
    }
    out: list[TierEvidence] = []
    for run_id, label in grounded_keys:
        row = rows.get((run_id, label))
        if row is None:
            out.append(TierEvidence(run_id=run_id, label=label, available=False))
        else:
            out.append(
                TierEvidence(
                    run_id=run_id,
                    label=label,
                    available=True,
                    similarity=row.similarity,
                    margin=row.margin,
                    vote_agreement=row.vote_agreement,
                    eligible_turns=row.eligible_turns,
                    eligible_seconds=row.eligible_seconds,
                    roster_size=row.roster_size,
                )
            )
    return out


def _margin_passes(evidence: TierEvidence, minimum: float) -> bool:
    if evidence.margin is not None:
        return evidence.margin >= minimum
    # NULL margin = one-speaker roster = infinite margin; anything else NULL
    # is malformed and must not pass.
    return evidence.roster_size == 1


def grade(evidence: TierEvidence, gates: MatchingGates) -> MatchTier | None:
    """One appearance's grade, ``None`` when its numbers are unavailable."""
    if not evidence.available:
        return None
    if evidence.similarity is None or evidence.vote_agreement is None:
        return None
    grounded = (
        evidence.similarity >= gates.grounded_min_cosine
        and _margin_passes(evidence, gates.grounded_min_margin)
        and evidence.vote_agreement >= gates.grounded_min_vote_agreement
        and evidence.eligible_turns >= gates.grounded_min_turns
        and evidence.eligible_seconds >= gates.grounded_min_seconds
    )
    if grounded:
        return MatchTier.STRONG
    accept = (
        evidence.similarity >= gates.min_cosine
        and _margin_passes(evidence, gates.min_margin)
        and evidence.vote_agreement >= gates.min_vote_agreement
        and evidence.eligible_turns >= gates.min_turns
        and evidence.eligible_seconds >= gates.min_seconds
    )
    return MatchTier.MODERATE if accept else MatchTier.WEAK


def tier_for(
    evidence: Sequence[TierEvidence], gates: MatchingGates
) -> TierSummary:
    """Fold appearances into the speaker's chip: best appearance wins.

    THE replaceable seam for issue #114's calibrated policy — the pages bind to
    ``TierSummary`` only, so a new rule swaps in behind this signature.
    """
    counts = {MatchTier.STRONG: 0, MatchTier.MODERATE: 0, MatchTier.WEAK: 0}
    unavailable = 0
    for item in evidence:
        graded = grade(item, gates)
        if graded is None:
            unavailable += 1
        else:
            counts[graded] += 1
    tier: MatchTier | None = None
    for candidate in (MatchTier.STRONG, MatchTier.MODERATE, MatchTier.WEAK):
        if counts[candidate]:
            tier = candidate
            break
    return TierSummary(
        tier=tier,
        strong=counts[MatchTier.STRONG],
        moderate=counts[MatchTier.MODERATE],
        weak=counts[MatchTier.WEAK],
        unavailable=unavailable,
        evidence=tuple(evidence),
    )
