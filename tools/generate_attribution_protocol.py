#!/usr/bin/env python3
"""Generate the AMI speaker-attribution evaluation protocol manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from voxint.harness.ami_recurrence import (
    base_session_id,
    check_kill_criterion,
    parse_meetings_full,
)
from voxint.harness.attribution_protocol import (
    ExclusionEntry,
    OpenSetSelection,
    ProtocolManifest,
    assign_roles_and_splits,
    build_protocol,
    serialize_manifest,
    validate_session_honesty,
    validate_split_honesty,
)

DEFAULT_OUT = Path("media/.benchmark/ami/attribution/protocol.json")
SELECTION_SEED = "most-meetings-per-base-session-v1"
DEFAULT_SPLIT_SEED = "ami-attribution-split-v1"


def _selection_hash(split_seed: str, meeting_id: str) -> str:
    return hashlib.sha256(f"{split_seed}:{meeting_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings-xml", type=Path, required=True)
    parser.add_argument("--gold-rttm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--open-set", type=int, default=0, metavar="N")
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--exclude-meetings",
        nargs="*",
        default=[],
        help="meetings to exclude (e.g. missing gold RTTM)",
    )
    args = parser.parse_args()
    if args.open_set < 0:
        parser.error("--open-set must be non-negative")

    meetings = parse_meetings_full(args.meetings_xml.read_bytes())
    recurrence = check_kill_criterion(meetings)
    assert recurrence.baseline_viable, "AMI recurrence baseline is not viable"

    manual_exclude = set(args.exclude_meetings)
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
    open_set_candidates = sorted(
        (
            meeting_id
            for meeting_id in meetings
            if meeting_id not in manual_exclude
            and meeting_id not in cross_session_meetings
            and (args.gold_rttm_dir / f"{meeting_id}.rttm").is_file()
        ),
        key=lambda meeting_id: (_selection_hash(args.split_seed, meeting_id), meeting_id),
    )
    selected_open_set = open_set_candidates[: args.open_set]
    open_set_meetings = set(selected_open_set)
    selected_meetings = enrollment_meetings | test_meetings | open_set_meetings
    excluded_meetings = sorted(set(meetings) - selected_meetings - manual_exclude)
    assignments = assign_roles_and_splits(
        meetings,
        recurrence,
        enrollment_meetings=enrollment_meetings,
        test_meetings=test_meetings,
        open_set_meetings=open_set_meetings,
        split_seed=args.split_seed,
    )
    rows = build_protocol(
        meetings,
        recurrence,
        exclude=set(meetings) - selected_meetings,
        assignments=assignments,
    )

    violations = validate_session_honesty(rows, enrollment_meetings, test_meetings)
    violations.extend(validate_split_honesty(rows))
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
        split_seed=args.split_seed,
        open_set_selection=OpenSetSelection(
            requested=args.open_set,
            selected=selected_open_set,
            candidates=len(open_set_candidates),
        ),
        rows=rows,
        recurrence_report=recurrence,
        exclusions=exclusion_entries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(serialize_manifest(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    meeting_assignments = Counter(assignments.values())
    for role in ("enrollment", "test_genuine", "test_open"):
        for split in ("enrollment", "dev", "confirm"):
            count = meeting_assignments[(role, split)]
            if count:
                print(f"{role}/{split} meetings: {count}")
    for split in ("dev", "confirm"):
        speakers = {
            row.host_global_name
            for row in rows
            if row.role == "test_genuine" and row.split == split
            and row.host_global_name in recurrence.speakers
        }
        print(f"Cross-session speakers ({split}): {len(speakers)}")
    print(f"Open-set candidates with gold RTTM: {len(open_set_candidates)}")
    print(f"Cross-session speakers: {recurrence.n_cross_session_speakers}")
    print(f"Genuine pairs: {recurrence.n_genuine_pairs}")
    print(f"Exclusions: {len(excluded_meetings)}")
    print("Split validation: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
