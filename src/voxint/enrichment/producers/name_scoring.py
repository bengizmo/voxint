"""Pure aggregation and scoring of name mentions into scored candidates (#38).

Mentions in, deterministic scored :class:`NameCandidate` list out. Rules:

- **Per-target aggregation.** Run-level candidates aggregate METADATA and
  OTHER mentions; run_label candidates aggregate only SELF mentions from
  segments carrying that diarization label. Evidence never leaks across
  targets — a title mention must not inflate (or create) a cluster-identity
  claim, and one label's self-introduction must not boost another label.
  A SELF mention from a segment *without* a diarization label degrades to
  run-level (there is no cluster to claim).
- **Exact-name grouping.** Keyed by the casefolded normalized name. No
  given-name prefix merging: "John" and "John Smith" score independently
  (merging is unsafe with multiple participants).
- **Explainable score, not pseudo-calibration.** ``base`` is the strongest
  adjusted pattern reliability (suspect segments halve a mention's
  reliability); small flat bonuses reward multi-pattern corroboration,
  source diversity, and a domain-pack seed match; capped below 1.0 so no
  candidate ever reads as certain. No frequency term — repeated weak
  matches of the same pattern are counted, not rewarded.
- **Ambiguity gate.** Single-token names that double as common words
  ("Will", "Mark", "June") are kept only when vouched for by a strong
  self-introduction pattern or an exact seed match.

Weights and bonuses are module constants; changing them changes producer
output, so bump ``SCORING_VERSION``.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from voxint.enrichment.producers.name_patterns import (
    Attribution,
    RawMention,
    SegmentRef,
)

SCORING_VERSION = 1

SUSPECT_RELIABILITY_MULTIPLIER = 0.5
CORROBORATION_BONUS = 0.05
SOURCE_DIVERSITY_BONUS = 0.05
SEED_MATCH_BONUS = 0.05
SCORE_CAP = 0.95
# An ambiguous single-token name needs at least this pattern reliability
# (or an exact seed match) to survive aggregation.
AMBIGUOUS_MIN_RELIABILITY = 0.85


class CandidateLevel(enum.StrEnum):
    """How far a candidate reaches: the whole run, or one diarization label."""

    RUN = "run"
    RUN_LABEL = "run_label"


@dataclass(frozen=True)
class NameCandidate:
    """A scored, evidence-ordered suggestion for one target."""

    name: str
    level: CandidateLevel
    diarization_label: str | None
    score: float
    score_components: dict[str, float]
    mentions: tuple[RawMention, ...]  # deterministic evidence order


def _source_class(mention: RawMention) -> str:
    """The independence class a mention counts toward for diversity."""
    if isinstance(mention.source, SegmentRef):
        return "transcript_self" if mention.attribution is Attribution.SELF else "transcript_other"
    field = mention.source.field
    return "channel" if field in ("channel", "uploader") else field


def _adjusted_reliability(mention: RawMention) -> float:
    if isinstance(mention.source, SegmentRef) and mention.source.suspect:
        return mention.reliability * SUSPECT_RELIABILITY_MULTIPLIER
    return mention.reliability


def _mention_sort_key(mention: RawMention) -> tuple[float, str, str, int, str]:
    """Deterministic evidence order: strongest first, then stable provenance."""
    if isinstance(mention.source, SegmentRef):
        position = mention.source.segment_index
        field = ""
    else:
        position = mention.source.item_index or 0
        field = mention.source.field
    return (
        -_adjusted_reliability(mention),
        _source_class(mention),
        field,
        position,
        mention.pattern_id,
    )


def _passes_ambiguity_gate(mention: RawMention, seed_matched: bool) -> bool:
    if not mention.ambiguous:
        return True
    return seed_matched or mention.reliability >= AMBIGUOUS_MIN_RELIABILITY


def _score(mentions: Sequence[RawMention], seed_matched: bool) -> tuple[float, dict[str, float]]:
    adjusted = [_adjusted_reliability(m) for m in mentions]
    base = max(adjusted)
    distinct_patterns = len({m.pattern_id for m in mentions})
    distinct_sources = len({_source_class(m) for m in mentions})
    bonus_patterns = CORROBORATION_BONUS if distinct_patterns >= 2 else 0.0
    bonus_sources = SOURCE_DIVERSITY_BONUS if distinct_sources >= 2 else 0.0
    bonus_seed = SEED_MATCH_BONUS if seed_matched else 0.0
    score = min(SCORE_CAP, base + bonus_patterns + bonus_sources + bonus_seed)
    suspect_applied = any(isinstance(m.source, SegmentRef) and m.source.suspect for m in mentions)
    components = {
        "base": round(base, 4),
        "bonus_patterns": bonus_patterns,
        "bonus_sources": bonus_sources,
        "bonus_seed": bonus_seed,
        "mention_count": float(len(mentions)),
        "distinct_patterns": float(distinct_patterns),
        "distinct_sources": float(distinct_sources),
        "suspect_penalty_applied": 1.0 if suspect_applied else 0.0,
    }
    return round(score, 4), components


def aggregate(
    mentions: Sequence[RawMention], *, name_seeds: Sequence[str] = ()
) -> list[NameCandidate]:
    """Group mentions per target and name, gate, score, and order them.

    Output order is fully deterministic: run candidates first, then
    run_label candidates by label, then by descending score and name.
    """
    seeds = {seed.strip().casefold() for seed in name_seeds if seed.strip()}

    # (level, label, casefolded name) -> mentions
    groups: dict[tuple[CandidateLevel, str | None, str], list[RawMention]] = {}
    display: dict[tuple[CandidateLevel, str | None, str], str] = {}
    for mention in mentions:
        if mention.attribution is Attribution.SELF and isinstance(mention.source, SegmentRef):
            label = mention.source.diarization_label
            level = CandidateLevel.RUN_LABEL if label is not None else CandidateLevel.RUN
        else:
            label = None
            level = CandidateLevel.RUN
        key = (level, label, mention.name.casefold())
        groups.setdefault(key, []).append(mention)
        display.setdefault(key, mention.name)

    candidates: list[NameCandidate] = []
    for key, grouped in groups.items():
        level, label, folded = key
        seed_matched = folded in seeds
        kept = [m for m in grouped if _passes_ambiguity_gate(m, seed_matched)]
        if not kept:
            continue
        kept.sort(key=_mention_sort_key)
        score, components = _score(kept, seed_matched)
        candidates.append(
            NameCandidate(
                name=display[key],
                level=level,
                diarization_label=label,
                score=score,
                score_components=components,
                mentions=tuple(kept),
            )
        )

    candidates.sort(
        key=lambda c: (
            c.level.value,  # "run" < "run_label"
            c.diarization_label or "",
            -c.score,
            c.name.casefold(),
        )
    )
    return candidates
