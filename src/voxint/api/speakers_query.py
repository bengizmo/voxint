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

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from voxint.db.models import (
    ClaimField,
    EnrichmentCandidate,
    ProfileReviewDecision,
    SpeakerAssignment,
)
from voxint.enrichment.queries import CandidateState, effective_state_sql
from voxint.speakers.aggregate import (
    AggregateResult,
    SpeakerAggregate,
    aggregate_speakers,
    empty_aggregate,
)
from voxint.speakers.matching import DuplicatePair, MatchingGates, find_possible_duplicates
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
    heard_name: str | None = None


# Reminders (#159, #181): flat caps + honest truncation (the lib-kit convention).
NAME_SUGGESTION_CAP = 5
UNVERIFIED_CAP = 5
DUPLICATE_CAP = 3
# A speaker below this much attributed speech is not "high-activity" — the
# card nudges toward the biggest unconfirmed voices, not every stray minute.
UNVERIFIED_FLOOR_SECONDS = 5 * 60.0


@dataclass(frozen=True)
class NameSuggestionReminder:
    """Pending (proposed) speaker-name drafts on one recording — reviewed on
    that recording's workbench, the only name-decision surface."""

    run_id: uuid.UUID
    count: int


@dataclass(frozen=True)
class DuplicateReminder:
    """A pair of active speakers whose enrollment centroids are similar enough
    to warrant a merge review (#181)."""

    speaker_a_id: uuid.UUID
    speaker_a_name: str
    speaker_b_id: uuid.UUID
    speaker_b_name: str
    similarity: float


@dataclass(frozen=True)
class SpeakersOverview:
    rows: tuple[OverviewRow, ...]
    inactive: tuple[RosterEntry, ...]
    runs_scanned: int
    verified_count: int
    total_seconds: float
    # Action reminders (#159): capped lists plus true totals so the strip can
    # say "and N more" honestly.
    name_suggestions: tuple[NameSuggestionReminder, ...] = ()
    name_suggestion_total: int = 0
    unverified_active: tuple[OverviewRow, ...] = ()
    unverified_active_total: int = 0
    possible_duplicates: tuple[DuplicateReminder, ...] = ()
    possible_duplicates_total: int = 0


def _pending_name_suggestions(
    session: Session,
) -> tuple[tuple[NameSuggestionReminder, ...], int]:
    """Proposed name drafts grouped by recording, busiest first.

    Name candidates target a (run, label) — never a speaker row (DB shape
    check) — so the reminder groups by run and links its workbench. The count
    is raw proposed drafts (rerun duplicates the workbench would collapse are
    still work waiting there, so the total stays honest about volume).
    """
    rows = session.execute(
        select(EnrichmentCandidate.pipeline_run_id, func.count().label("n"))
        .outerjoin(
            ProfileReviewDecision,
            ProfileReviewDecision.candidate_id == EnrichmentCandidate.id,
        )
        .where(
            EnrichmentCandidate.field == ClaimField.NAME.value,
            EnrichmentCandidate.pipeline_run_id.is_not(None),
            effective_state_sql() == CandidateState.PROPOSED.value,
        )
        .group_by(EnrichmentCandidate.pipeline_run_id)
        .order_by(func.count().desc(), EnrichmentCandidate.pipeline_run_id)
    ).all()
    reminders = tuple(
        NameSuggestionReminder(run_id=row.pipeline_run_id, count=row.n)
        for row in rows[:NAME_SUGGESTION_CAP]
    )
    return reminders, int(sum(row.n for row in rows))


def _possible_duplicate_reminders(
    session: Session,
    gates: MatchingGates,
    name_by_id: dict[uuid.UUID, str],
) -> tuple[tuple[DuplicateReminder, ...], int]:
    """Pairs of active speakers whose centroids clear the grounded threshold."""
    pairs: list[DuplicatePair] = find_possible_duplicates(
        session, gates.grounded_min_cosine
    )
    reminders = tuple(
        DuplicateReminder(
            speaker_a_id=p.speaker_a_id,
            speaker_a_name=name_by_id.get(p.speaker_a_id, "?"),
            speaker_b_id=p.speaker_b_id,
            speaker_b_name=name_by_id.get(p.speaker_b_id, "?"),
            similarity=p.similarity,
        )
        for p in pairs
        if p.speaker_a_id in name_by_id and p.speaker_b_id in name_by_id
    )
    return reminders[:DUPLICATE_CAP], len(reminders)


def _heard_names_for_speakers(
    session: Session,
    rows: list[OverviewRow],
) -> dict[uuid.UUID, str]:
    """For each unverified speaker, find the most recent llm_hint proposed_name
    on any of their attributed (run, label) appearances. Returns speaker_id ->
    proposed_name for speakers that have one."""
    all_keys: list[tuple[uuid.UUID, str]] = []
    speaker_by_key: dict[tuple[uuid.UUID, str], uuid.UUID] = {}
    for row in rows:
        if row.aggregate.verified:
            continue
        for key in row.aggregate.grounded_keys:
            all_keys.append(key)
            speaker_by_key[key] = row.entry.speaker.id
    if not all_keys:
        return {}
    unique_keys = list(set(all_keys))
    hints = session.execute(
        select(
            SpeakerAssignment.pipeline_run_id,
            SpeakerAssignment.diarization_label,
            SpeakerAssignment.proposed_name,
            SpeakerAssignment.created_at,
        )
        .where(
            SpeakerAssignment.method == "llm_hint",
            tuple_(
                SpeakerAssignment.pipeline_run_id,
                SpeakerAssignment.diarization_label,
            ).in_(unique_keys),
        )
        .order_by(SpeakerAssignment.created_at.desc(), SpeakerAssignment.id.desc())
    ).all()
    result: dict[uuid.UUID, str] = {}
    for hint in hints:
        key = (hint.pipeline_run_id, hint.diarization_label)
        speaker_id = speaker_by_key.get(key)
        if speaker_id is not None and speaker_id not in result and hint.proposed_name:
            result[speaker_id] = hint.proposed_name
    return result


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
    heard_names = _heard_names_for_speakers(session, rows)
    if heard_names:
        rows = [
            OverviewRow(
                entry=r.entry,
                aggregate=r.aggregate,
                tier=r.tier,
                heard_name=heard_names.get(r.entry.speaker.id),
            )
            if r.entry.speaker.id in heard_names
            else r
            for r in rows
        ]
    name_suggestions, name_total = _pending_name_suggestions(session)
    # Unverified high-activity (#159): much attributed speech, no confirming
    # human ruling anywhere — the voices most worth an operator look.
    unverified = sorted(
        (
            r
            for r in rows
            if not r.aggregate.verified
            and (
                r.aggregate.seconds >= UNVERIFIED_FLOOR_SECONDS
                or r.aggregate.auto_enrolled
            )
        ),
        key=lambda r: (-r.aggregate.seconds, r.entry.speaker.display_name.casefold()),
    )
    # Possible duplicates (#181): pairs whose enrollment centroids clear the
    # grounded cosine threshold, suggesting the same physical voice.
    name_by_id = {e.speaker.id: e.speaker.display_name for e in overview.active}
    dup_reminders, dup_total = _possible_duplicate_reminders(
        session, gates, name_by_id
    )
    return SpeakersOverview(
        rows=tuple(_sort_rows(rows, sort)),
        inactive=overview.inactive,
        runs_scanned=result.runs_scanned,
        verified_count=sum(1 for r in rows if r.aggregate.verified),
        total_seconds=sum(r.aggregate.seconds for r in rows),
        name_suggestions=name_suggestions,
        name_suggestion_total=name_total,
        unverified_active=tuple(unverified[:UNVERIFIED_CAP]),
        unverified_active_total=len(unverified),
        possible_duplicates=dup_reminders,
        possible_duplicates_total=dup_total,
    )
