"""Speakers overview assembly (issue #159): roster + aggregates + tiers.

The Console 2.0 ``/speakers`` overview's read model, in the ``media_query``
mold: frozen dataclasses assembled server-side, no HTTP concerns. Data comes
from the resolver-backed aggregation (``speakers/aggregate.py``) layered over
the roster (``speakers/roster.py``); voice-match chips from
``speakers/tiers.py`` graded against the live gates. Unknown sort/view values
degrade to the default (the Home ``?window=`` convention) — never a 422.
"""

import uuid
from dataclasses import dataclass
from typing import TypeGuard

from sqlalchemy.orm import Session

from voxint.speakers.aggregate import (
    AggregateResult,
    SpeakerAggregate,
    aggregate_speakers,
    empty_aggregate,
)
from voxint.speakers.matching import MatchingGates
from voxint.speakers.roster import RosterEntry, roster_overview
from voxint.speakers.tiers import TierSummary, evidence_for, tier_for

SORT_LABELS: tuple[tuple[str, str], ...] = (
    ("minutes", "Minutes"),
    ("files", "Files"),
    ("name", "Name"),
    ("seen", "Last heard"),
)
DEFAULT_SORT = "minutes"
VIEWS: tuple[str, ...] = ("cards", "table")
DEFAULT_VIEW = "cards"

_SORT_VALUES = tuple(value for value, _ in SORT_LABELS)


def normalize_sort(raw: str | None) -> str:
    return raw if raw in _SORT_VALUES else DEFAULT_SORT


def normalize_view(raw: str | None) -> str:
    return raw if raw in VIEWS else DEFAULT_VIEW


def sort_is_known(raw: str | None) -> TypeGuard[str]:
    return raw in _SORT_VALUES


@dataclass(frozen=True)
class OverviewRow:
    """One active speaker's overview line: identity + numbers + voice chip."""

    entry: RosterEntry
    aggregate: SpeakerAggregate
    tier: TierSummary


@dataclass(frozen=True)
class SpeakersOverview:
    rows: tuple[OverviewRow, ...]
    inactive: tuple[RosterEntry, ...]
    runs_scanned: int
    verified_count: int
    total_seconds: float


def _sort_rows(rows: list[OverviewRow], sort: str) -> list[OverviewRow]:
    # Every ordering ends on display name for a deterministic tiebreak.
    if sort == "name":
        return sorted(rows, key=lambda r: r.entry.speaker.display_name.casefold())
    if sort == "files":
        return sorted(
            rows,
            key=lambda r: (-r.aggregate.files, r.entry.speaker.display_name.casefold()),
        )
    if sort == "seen":
        # Speakers never heard sort last.
        return sorted(
            rows,
            key=lambda r: (
                r.aggregate.last_seen is None,
                -(r.aggregate.last_seen.timestamp() if r.aggregate.last_seen else 0.0),
                r.entry.speaker.display_name.casefold(),
            ),
        )
    return sorted(
        rows,
        key=lambda r: (-r.aggregate.seconds, r.entry.speaker.display_name.casefold()),
    )


def speakers_overview(
    session: Session, gates: MatchingGates, *, sort: str = DEFAULT_SORT
) -> SpeakersOverview:
    """Assemble the overview: EVERY active roster speaker renders (zero rows
    for the never-attributed — nobody silently vanishes), aggregates from the
    canonical-run fold, tiers from one batched diagnostics load."""
    overview = roster_overview(session)
    result: AggregateResult = aggregate_speakers(session)
    # One diagnostics batch for every speaker's surviving grounded appearances.
    all_keys: list[tuple[uuid.UUID, str]] = []
    for entry in overview.active:
        aggregate = result.by_speaker.get(entry.speaker.id)
        if aggregate is not None:
            all_keys.extend(aggregate.grounded_keys)
    evidence_by_key = {
        (item.run_id, item.label): item for item in evidence_for(session, all_keys)
    }
    rows: list[OverviewRow] = []
    for entry in overview.active:
        aggregate = result.by_speaker.get(entry.speaker.id) or empty_aggregate(
            entry.speaker.id
        )
        evidence = [evidence_by_key[key] for key in aggregate.grounded_keys]
        rows.append(
            OverviewRow(entry=entry, aggregate=aggregate, tier=tier_for(evidence, gates))
        )
    return SpeakersOverview(
        rows=tuple(_sort_rows(rows, sort)),
        inactive=overview.inactive,
        runs_scanned=result.runs_scanned,
        verified_count=sum(1 for r in rows if r.aggregate.verified),
        total_seconds=sum(r.aggregate.seconds for r in rows),
    )
