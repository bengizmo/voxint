"""AMI speaker-attribution evaluation protocol manifests.

Pure, DB-free helpers for selecting corpus rows, checking session-disjoint
splits, and converting protocol manifests to and from JSON-friendly data.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from voxint.harness.ami_recurrence import (
    MeetingSpeaker,
    RecurrenceReport,
    base_session_id,
)

MeetingRole = Literal["enrollment", "test_genuine", "test_open"]
MeetingSplit = Literal["enrollment", "dev", "confirm"]


class OpenSetSelection(TypedDict):
    """Persisted metadata for deterministic open-set meeting selection."""

    requested: int
    selected: list[str]
    candidates: int


@dataclass(frozen=True)
class AttributionProtocolRow:
    """One gold speaker slot in the attribution evaluation corpus."""

    meeting_id: str
    nxt_agent: str
    channel: int
    host_global_name: str
    base_session_id: str
    role: MeetingRole = "test_genuine"
    split: MeetingSplit = "dev"

    def __post_init__(self) -> None:
        if self.role not in ("enrollment", "test_genuine", "test_open"):
            raise ValueError(f"invalid attribution meeting role {self.role!r}")
        if self.split not in ("enrollment", "dev", "confirm"):
            raise ValueError(f"invalid attribution meeting split {self.split!r}")


@dataclass(frozen=True)
class ExclusionEntry:
    """A meeting deliberately omitted from the evaluation corpus."""

    meeting_id: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ProtocolManifest:
    """Serializable definition of an attribution evaluation corpus."""

    schema_version: int = 2
    corpus: str = "ami_ihm"
    truth_source: str = "corpus_gold"
    selection_seed: str
    split_seed: str
    open_set_selection: OpenSetSelection
    rows: list[AttributionProtocolRow]
    recurrence_report: RecurrenceReport
    exclusions: list[ExclusionEntry]


def build_protocol(
    meetings: dict[str, dict[str, MeetingSpeaker]],
    recurrence: RecurrenceReport,
    *,
    exclude: set[str] | None = None,
    assignments: dict[str, tuple[MeetingRole, MeetingSplit]] | None = None,
) -> list[AttributionProtocolRow]:
    """Build deterministic protocol rows from parsed AMI meeting metadata."""
    del recurrence  # The report is recorded in the manifest, not used to filter rows.
    excluded = exclude or set()
    rows: list[AttributionProtocolRow] = []
    for meeting_id, speakers in sorted(meetings.items()):
        if meeting_id in excluded:
            continue
        for agent, speaker in sorted(speakers.items()):
            if speaker.global_name is None:
                continue
            role, split = (assignments or {}).get(
                meeting_id, ("test_genuine", "dev")
            )
            rows.append(
                AttributionProtocolRow(
                    meeting_id=meeting_id,
                    nxt_agent=agent,
                    channel=speaker.channel,
                    host_global_name=speaker.global_name,
                    base_session_id=base_session_id(meeting_id),
                    role=role,
                    split=split,
                )
            )
    return rows


def _component_hash(split_seed: str, sessions: set[str]) -> str:
    key = ",".join(sorted(sessions))
    return hashlib.sha256(f"{split_seed}:{key}".encode()).hexdigest()


def _alternate_components(
    components: list[set[str]], split_seed: str
) -> dict[str, MeetingSplit]:
    assignments: dict[str, MeetingSplit] = {}
    ordered = sorted(
        components,
        key=lambda component: (_component_hash(split_seed, component), sorted(component)),
    )
    for index, component in enumerate(ordered):
        split: MeetingSplit = "dev" if index % 2 == 0 else "confirm"
        for session in component:
            assignments[session] = split
    return assignments


def assign_roles_and_splits(
    meetings: dict[str, dict[str, MeetingSpeaker]],
    recurrence: RecurrenceReport,
    *,
    enrollment_meetings: set[str],
    test_meetings: set[str],
    open_set_meetings: set[str],
    split_seed: str,
) -> dict[str, tuple[MeetingRole, MeetingSplit]]:
    """Assign selected meetings to roles and metadata-only dev/confirm splits."""
    selected = enrollment_meetings | test_meetings | open_set_meetings
    unknown = selected - meetings.keys()
    if unknown:
        raise ValueError(f"selected meetings absent from metadata: {sorted(unknown)}")
    overlaps = (
        (enrollment_meetings & test_meetings)
        | (enrollment_meetings & open_set_meetings)
        | (test_meetings & open_set_meetings)
    )
    if overlaps:
        raise ValueError(f"meetings assigned to multiple roles: {sorted(overlaps)}")

    test_sessions = {base_session_id(meeting) for meeting in test_meetings}
    adjacency: dict[str, set[str]] = {
        session: set() for session in test_sessions
    }
    for appearances in recurrence.speakers.values():
        sessions = {
            appearance.base_session_id
            for appearance in appearances
            if appearance.base_session_id in test_sessions
        }
        for session in sessions:
            adjacency[session].update(sessions - {session})

    genuine_components: list[set[str]] = []
    unseen = set(test_sessions)
    while unseen:
        root = min(unseen)
        component: set[str] = set()
        pending = [root]
        while pending:
            session = pending.pop()
            if session in component:
                continue
            component.add(session)
            pending.extend(sorted(adjacency[session] - component, reverse=True))
        unseen -= component
        genuine_components.append(component)

    open_sessions = {base_session_id(meeting) for meeting in open_set_meetings}
    session_overlap = test_sessions & open_sessions
    if session_overlap:
        raise ValueError(
            f"genuine/open-set meetings share base sessions: {sorted(session_overlap)}"
        )
    open_components = [{session} for session in open_sessions]
    split_by_session = _alternate_components(genuine_components, split_seed)
    split_by_session.update(_alternate_components(open_components, split_seed))

    result: dict[str, tuple[MeetingRole, MeetingSplit]] = {}
    for meeting in sorted(enrollment_meetings):
        result[meeting] = ("enrollment", "enrollment")
    for meeting in sorted(test_meetings):
        result[meeting] = ("test_genuine", split_by_session[base_session_id(meeting)])
    for meeting in sorted(open_set_meetings):
        result[meeting] = ("test_open", split_by_session[base_session_id(meeting)])
    return result


def validate_split_honesty(rows: list[AttributionProtocolRow]) -> list[str]:
    """Report role/split leakage in a schema-2 protocol row set."""
    violations: list[str] = []
    rows_by_meeting: dict[str, list[AttributionProtocolRow]] = defaultdict(list)
    enrolled_speakers = {
        row.host_global_name for row in rows if row.role == "enrollment"
    }
    test_splits_by_speaker: dict[str, set[MeetingSplit]] = defaultdict(set)

    for row in rows:
        rows_by_meeting[row.meeting_id].append(row)
        if row.role == "enrollment" and row.split != "enrollment":
            violations.append(
                f"enrollment meeting {row.meeting_id!r} is in split {row.split!r}"
            )
        if row.role != "enrollment" and row.split in ("dev", "confirm"):
            test_splits_by_speaker[row.host_global_name].add(row.split)

    for speaker, splits in sorted(test_splits_by_speaker.items()):
        if splits == {"dev", "confirm"}:
            violations.append(
                f"speaker {speaker!r} has test rows in both dev and confirm"
            )
    for meeting_id, meeting_rows in sorted(rows_by_meeting.items()):
        if meeting_rows[0].role != "test_open":
            continue
        leaked = sorted(
            {row.host_global_name for row in meeting_rows} & enrolled_speakers
        )
        if leaked:
            violations.append(
                f"open-set meeting {meeting_id!r} contains enrolled speakers: {leaked}"
            )
    return violations


def validate_session_honesty(
    rows: list[AttributionProtocolRow],
    enrollment_meetings: set[str],
    test_meetings: set[str],
) -> list[str]:
    """Report speakers whose enrollment and test data share a base session."""
    enrollment_sessions: dict[str, set[str]] = {}
    test_sessions: dict[str, set[str]] = {}
    for row in rows:
        if row.meeting_id in enrollment_meetings:
            enrollment_sessions.setdefault(row.host_global_name, set()).add(
                row.base_session_id
            )
        if row.meeting_id in test_meetings:
            test_sessions.setdefault(row.host_global_name, set()).add(row.base_session_id)

    violations: list[str] = []
    for speaker in sorted(enrollment_sessions.keys() & test_sessions.keys()):
        overlap = sorted(enrollment_sessions[speaker] & test_sessions[speaker])
        if overlap:
            violations.append(
                f"speaker {speaker!r} has enrollment/test base-session overlap: "
                f"{', '.join(overlap)}"
            )
    return violations


def serialize_manifest(manifest: ProtocolManifest) -> dict[str, Any]:
    """Convert a protocol manifest to JSON-friendly built-in types."""
    report = manifest.recurrence_report
    return {
        "schema_version": manifest.schema_version,
        "corpus": manifest.corpus,
        "truth_source": manifest.truth_source,
        "selection_seed": manifest.selection_seed,
        "split_seed": manifest.split_seed,
        "open_set_selection": manifest.open_set_selection,
        "rows": [
            {
                "meeting_id": row.meeting_id,
                "nxt_agent": row.nxt_agent,
                "channel": row.channel,
                "host_global_name": row.host_global_name,
                "base_session_id": row.base_session_id,
                "role": row.role,
                "split": row.split,
            }
            for row in manifest.rows
        ],
        "recurrence_report": {
            "n_meetings": report.n_meetings,
            "n_base_sessions": report.n_base_sessions,
            "n_participants": report.n_participants,
            "n_cross_session_speakers": report.n_cross_session_speakers,
            "n_genuine_pairs": report.n_genuine_pairs,
            "n_impostor_pairs": report.n_impostor_pairs,
            "baseline_viable": report.baseline_viable,
            "calibration_viable": report.calibration_viable,
        },
        "exclusions": [
            {"meeting_id": entry.meeting_id, "reason": entry.reason}
            for entry in manifest.exclusions
        ],
    }


def parse_manifest(data: dict[str, Any]) -> ProtocolManifest:
    """Reconstruct a protocol manifest from its JSON-friendly representation."""
    version = data.get("schema_version")
    if version == 1:
        raise ValueError(
            "attribution protocol schema_version 1 has no role/split metadata; "
            "regenerate it as schema_version 2"
        )
    if version != 2:
        raise ValueError(f"unsupported attribution protocol schema_version {version!r}")
    recurrence = data["recurrence_report"]
    selection = data["open_set_selection"]
    return ProtocolManifest(
        schema_version=version,
        corpus=data["corpus"],
        truth_source=data["truth_source"],
        selection_seed=data["selection_seed"],
        split_seed=data["split_seed"],
        open_set_selection=OpenSetSelection(
            requested=selection["requested"],
            selected=list(selection["selected"]),
            candidates=selection["candidates"],
        ),
        rows=[AttributionProtocolRow(**row) for row in data["rows"]],
        recurrence_report=RecurrenceReport(
            n_meetings=recurrence["n_meetings"],
            n_base_sessions=recurrence["n_base_sessions"],
            n_participants=recurrence["n_participants"],
            n_cross_session_speakers=recurrence["n_cross_session_speakers"],
            n_genuine_pairs=recurrence["n_genuine_pairs"],
            n_impostor_pairs=recurrence["n_impostor_pairs"],
            baseline_viable=recurrence["baseline_viable"],
            calibration_viable=recurrence["calibration_viable"],
            speakers={},
        ),
        exclusions=[ExclusionEntry(**entry) for entry in data["exclusions"]],
    )
