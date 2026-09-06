#!/usr/bin/env python3
"""Generate the AMI speaker-attribution evaluation protocol manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from voxint.harness.ami_recurrence import (
    base_session_id,
    check_kill_criterion,
    parse_meetings_full,
)
from voxint.harness.attribution_protocol import (
    ExclusionEntry,
    ProtocolManifest,
    build_protocol,
    serialize_manifest,
    validate_session_honesty,
)

DEFAULT_OUT = Path("media/.benchmark/ami/attribution/protocol.json")
SELECTION_SEED = "most-meetings-per-base-session-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--exclude-meetings",
        nargs="*",
        default=[],
        help="meetings to exclude (e.g. missing gold RTTM)",
    )
    args = parser.parse_args()

    meetings = parse_meetings_full(args.meetings_xml.read_bytes())
    recurrence = check_kill_criterion(meetings)
    assert recurrence.baseline_viable, "AMI recurrence baseline is not viable"

    manual_exclude = set(args.exclude_meetings)
    all_rows = build_protocol(meetings, recurrence, exclude=manual_exclude)
    enrollment_sessions: set[str] = set()
    test_sessions: set[str] = set()
    cross_session_meetings: set[str] = set()

    for global_name in sorted(recurrence.speakers):
        meetings_by_session: dict[str, set[str]] = defaultdict(set)
        for appearance in recurrence.speakers[global_name]:
            meetings_by_session[appearance.base_session_id].add(appearance.meeting_id)
            cross_session_meetings.add(appearance.meeting_id)

        enrollment_session = min(
            meetings_by_session,
            key=lambda session: (-len(meetings_by_session[session]), session),
        )
        enrollment_sessions.add(enrollment_session)
        test_sessions.update(set(meetings_by_session) - {enrollment_session})

    cross_session_meetings -= manual_exclude
    enrollment_meetings = {
        meeting_id
        for meeting_id in meetings
        if meeting_id not in manual_exclude
        and meeting_id in cross_session_meetings
        and base_session_id(meeting_id) in enrollment_sessions
    }
    test_meetings = {
        meeting_id
        for meeting_id in meetings
        if meeting_id not in manual_exclude
        and meeting_id in cross_session_meetings
        and base_session_id(meeting_id) in test_sessions - enrollment_sessions
    }
    selected_meetings = enrollment_meetings | test_meetings
    excluded_meetings = sorted(set(meetings) - selected_meetings - manual_exclude)
    rows = [row for row in all_rows if row.meeting_id in selected_meetings]

    violations = validate_session_honesty(rows, enrollment_meetings, test_meetings)
    assert not violations, "; ".join(violations)

    exclusion_entries = [
        ExclusionEntry(m, "no gold RTTM available")
        for m in sorted(manual_exclude)
        if m not in selected_meetings
    ] + [
        ExclusionEntry(m, "no cross-session speaker")
        for m in excluded_meetings
        if m not in manual_exclude
    ]
    manifest = ProtocolManifest(
        selection_seed=SELECTION_SEED,
        rows=rows,
        recurrence_report=recurrence,
        exclusions=exclusion_entries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(serialize_manifest(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Enrollment meetings: {len(enrollment_meetings)}")
    print(f"Test meetings: {len(test_meetings)}")
    print(f"Cross-session speakers: {recurrence.n_cross_session_speakers}")
    print(f"Genuine pairs: {recurrence.n_genuine_pairs}")
    print(f"Exclusions: {len(excluded_meetings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
