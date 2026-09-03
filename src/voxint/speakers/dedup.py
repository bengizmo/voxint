"""Merge-direction planning for probable duplicate speakers."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from voxint.speakers.matching import DuplicatePair

SkipReason = Literal["ambiguous_direction", "overlaps_prior"]

_PLACEHOLDER_RE = re.compile(r"^Voice \d+$")


@dataclass(frozen=True, slots=True)
class PlannedMerge:
    source_id: uuid.UUID
    target_id: uuid.UUID
    pair: DuplicatePair


@dataclass(frozen=True, slots=True)
class SkippedPair:
    pair: DuplicatePair
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class MergePlanResult:
    merges: tuple[PlannedMerge, ...]
    skipped: tuple[SkippedPair, ...]


def plan_dedup_merges(
    pairs: Sequence[DuplicatePair],
    speaker_names: Mapping[uuid.UUID, str],
    merge_threshold: float,
) -> MergePlanResult:
    """Build a safe merge plan from duplicate pairs.

    Pure function: no DB access, no side effects.  Pairs are processed in
    descending similarity order (UUID tie-breaker for determinism).  A pair
    is planned only when exactly one speaker has a placeholder name
    (``Voice N``); the placeholder becomes the merge source.  If either
    speaker already appears in a prior planned merge the pair is skipped
    (disjoint-pair safety).
    """
    planned: list[PlannedMerge] = []
    skipped: list[SkippedPair] = []
    seen: set[uuid.UUID] = set()

    sorted_pairs = sorted(
        pairs,
        key=lambda p: (-p.similarity, p.speaker_a_id, p.speaker_b_id),
    )

    for pair in sorted_pairs:
        if pair.similarity < merge_threshold:
            continue

        name_a = speaker_names.get(pair.speaker_a_id, "")
        name_b = speaker_names.get(pair.speaker_b_id, "")
        a_placeholder = _PLACEHOLDER_RE.fullmatch(name_a) is not None
        b_placeholder = _PLACEHOLDER_RE.fullmatch(name_b) is not None

        if a_placeholder == b_placeholder:
            skipped.append(SkippedPair(pair, "ambiguous_direction"))
            continue

        if pair.speaker_a_id in seen or pair.speaker_b_id in seen:
            skipped.append(SkippedPair(pair, "overlaps_prior"))
            continue

        if a_placeholder:
            source_id, target_id = pair.speaker_a_id, pair.speaker_b_id
        else:
            source_id, target_id = pair.speaker_b_id, pair.speaker_a_id

        planned.append(PlannedMerge(source_id, target_id, pair))
        seen.update((source_id, target_id))

    return MergePlanResult(tuple(planned), tuple(skipped))
