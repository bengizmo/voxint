"""Gold-set stratification + ground-truth provenance core (pure, DB-free).

Building blocks for assembling a name-accuracy gold set:

  * :func:`select_strata` deterministically picks a priority-ordered,
    hash-spread sample from candidate pools.
  * :func:`ground_truth_for` assigns AUTO truth (host name / ABSTAIN) ONLY
    where provenance is clean — a curated channel fact whose host voiceprint is
    groundable for *that item*; everything else goes to the human label queue.
    Without a voice anchor on the item, no slot can be proven to be the host,
    so a channel fact alone never auto-labels a positive.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from voxint.harness.name_accuracy import ABSTAIN


@dataclass(frozen=True)
class GoldRow:
    """The per-item attributes stratification + provenance gating need."""

    id: str
    channel: str | None
    source_type: str | None
    host_groundable: bool  # the curated host has >=1 enrollment embedding on this item
    is_no_host_control: bool  # a channel with no recurring host (negative control)


@dataclass(frozen=True)
class GoldLabel:
    """Provenance-tagged ground truth for one item's host slot."""

    truth: str | None  # a real host name, the ABSTAIN sentinel, or None (unknown)
    host_id: str | None
    label_source: str  # "auto" | "human_queue"
    reason: str


def _h(item_id: str) -> str:
    """Stable pseudo-random sort key (deterministic spread across the corpus)."""
    return hashlib.sha256(item_id.encode()).hexdigest()


def select_strata(
    plan: Sequence[tuple[str, Sequence[str], int]],
    *,
    total_target: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Priority-ordered, hash-deterministic stratified selection.

    ``plan`` is an ordered list of ``(label, candidate_ids, per_stratum_target)``.
    Each stratum draws hash-ordered ids it can still claim (skipping ids an
    earlier-priority stratum already took), capped by both its own target and
    the remaining global ``total_target``. Deterministic across input order and
    processes. Returns ``(id -> label, label -> count)``.
    """
    picked: set[str] = set()
    assign: dict[str, str] = {}
    counts: dict[str, int] = {}
    for label, candidates, n in plan:
        remaining = total_target - len(picked)
        if remaining <= 0:
            counts[label] = 0
            continue
        take = min(n, remaining)
        taken = 0
        for item_id in sorted(candidates, key=_h):
            if taken >= take:
                break
            if item_id in picked:
                continue
            picked.add(item_id)
            assign[item_id] = label
            taken += 1
        counts[label] = taken
    return assign, counts


def ground_truth_for(
    row: GoldRow,
    channel_host: Mapping[str, Mapping[str, str | None]],
) -> GoldLabel:
    """Provenance-clean AUTO truth, or defer to the human queue.

    ``channel_host`` maps a curated channel to ``{"host": name | None,
    "host_id": id | None}``. Resolution:

      * curated channel with a recurring host, host groundable here -> AUTO
        name truth (anchored on the verified ``host_id``).
      * curated channel with a recurring host NOT groundable here -> human
        queue (no voice anchor -> no slot can be proven to be the host).
      * curated channel with no recurring host (e.g. a guest-only show) ->
        AUTO ABSTAIN.
      * no-host control channel -> AUTO ABSTAIN (the over-naming negative
        control).
      * anything else -> human queue.
    """
    entry = channel_host.get(row.channel) if row.channel is not None else None
    if entry is not None:
        host = entry.get("host")
        host_id = entry.get("host_id")
        if host is None:
            return GoldLabel(ABSTAIN, None, "auto", "curated_no_host_channel")
        if not host_id:
            # A host name without a verified identity anchor cannot auto-label
            # a positive — the truth would not be provably about one person.
            return GoldLabel(None, None, "human_queue", "host_unanchored")
        if row.host_groundable:
            return GoldLabel(host, host_id, "auto", "channel_fact_groundable")
        return GoldLabel(None, host_id, "human_queue", "host_not_groundable")
    if row.is_no_host_control:
        return GoldLabel(ABSTAIN, None, "auto", "no_host_control")
    return GoldLabel(None, None, "human_queue", "uncurated_channel")
