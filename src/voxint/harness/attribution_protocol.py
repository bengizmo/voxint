"""AMI speaker-attribution evaluation protocol manifests.

Pure, DB-free helpers for selecting corpus rows, checking session-disjoint
splits, and converting protocol manifests to and from JSON-friendly data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voxint.harness.ami_recurrence import (
    MeetingSpeaker,
    RecurrenceReport,
    base_session_id,
)


@dataclass(frozen=True)
class AttributionProtocolRow:
    """One gold speaker slot in the attribution evaluation corpus."""

    meeting_id: str
    nxt_agent: str
    channel: int
    host_global_name: str
    base_session_id: str


@dataclass(frozen=True)
class ExclusionEntry:
    """A meeting deliberately omitted from the evaluation corpus."""

    meeting_id: str
    reason: str


@dataclass(frozen=True, kw_only=True)
class ProtocolManifest:
    """Serializable definition of an attribution evaluation corpus."""

    schema_version: int = 1
    corpus: str = "ami_ihm"
    truth_source: str = "corpus_gold"
    selection_seed: str
    rows: list[AttributionProtocolRow]
    recurrence_report: RecurrenceReport
    exclusions: list[ExclusionEntry]


def build_protocol(
    meetings: dict[str, dict[str, MeetingSpeaker]],
    recurrence: RecurrenceReport,
    *,
    exclude: set[str] | None = None,
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
            rows.append(
                AttributionProtocolRow(
                    meeting_id=meeting_id,
                    nxt_agent=agent,
                    channel=speaker.channel,
                    host_global_name=speaker.global_name,
                    base_session_id=base_session_id(meeting_id),
                )
            )
    return rows


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
        "rows": [
            {
                "meeting_id": row.meeting_id,
                "nxt_agent": row.nxt_agent,
                "channel": row.channel,
                "host_global_name": row.host_global_name,
                "base_session_id": row.base_session_id,
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
    recurrence = data["recurrence_report"]
    return ProtocolManifest(
        schema_version=data["schema_version"],
        corpus=data["corpus"],
        truth_source=data["truth_source"],
        selection_seed=data["selection_seed"],
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
