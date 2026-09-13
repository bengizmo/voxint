import pytest

from voxint.harness.ami_recurrence import (
    MeetingSpeaker,
    RecurrenceReport,
    check_kill_criterion,
)
from voxint.harness.attribution_protocol import (
    AttributionProtocolRow,
    ExclusionEntry,
    MeetingRole,
    MeetingSplit,
    OpenSetSelection,
    ProtocolManifest,
    assign_roles_and_splits,
    build_protocol,
    parse_manifest,
    serialize_manifest,
    validate_session_honesty,
    validate_split_honesty,
)


def _report() -> RecurrenceReport:
    return RecurrenceReport(
        n_meetings=3,
        n_base_sessions=2,
        n_participants=3,
        n_cross_session_speakers=2,
        n_genuine_pairs=1,
        n_impostor_pairs=2,
        baseline_viable=True,
        calibration_viable=False,
        speakers={},
    )


def _row(
    meeting_id: str,
    speaker: str,
    session: str,
    role: MeetingRole = "test_genuine",
    split: MeetingSplit = "dev",
) -> AttributionProtocolRow:
    return AttributionProtocolRow(meeting_id, "A", 0, speaker, session, role, split)


def test_build_protocol_sorts_rows_and_skips_speakers_without_global_name() -> None:
    meetings = {
        "IS1003a": {"B": MeetingSpeaker("B", 1, "MIO034")},
        "ES2002a": {
            "B": MeetingSpeaker("B", 1, None),
            "A": MeetingSpeaker("A", 0, "FEE005"),
        },
    }

    assert build_protocol(meetings, _report()) == [
        AttributionProtocolRow("ES2002a", "A", 0, "FEE005", "ES2002"),
        AttributionProtocolRow("IS1003a", "B", 1, "MIO034", "IS1003"),
    ]


def test_build_protocol_excludes_meetings() -> None:
    meetings = {
        "ES2002a": {"A": MeetingSpeaker("A", 0, "FEE005")},
        "IS1003a": {"A": MeetingSpeaker("A", 0, "FEE005")},
    }

    assert build_protocol(meetings, _report(), exclude={"ES2002a"}) == [
        AttributionProtocolRow("IS1003a", "A", 0, "FEE005", "IS1003")
    ]


def test_validate_session_honesty_passes_for_disjoint_sessions() -> None:
    rows = [_row("ES2002a", "FEE005", "ES2002"), _row("IS1003a", "FEE005", "IS1003")]

    assert validate_session_honesty(rows, {"ES2002a"}, {"IS1003a"}) == []


def test_validate_session_honesty_catches_overlap_for_same_speaker() -> None:
    rows = [_row("ES2002a", "FEE005", "ES2002"), _row("ES2002b", "FEE005", "ES2002")]

    violations = validate_session_honesty(rows, {"ES2002a"}, {"ES2002b"})

    assert len(violations) == 1
    assert "FEE005" in violations[0]
    assert "ES2002" in violations[0]


def test_validate_session_honesty_returns_multiple_violations() -> None:
    rows = [
        _row("ES2002a", "FEE005", "ES2002"),
        _row("ES2002b", "FEE005", "ES2002"),
        _row("IS1003a", "MIO034", "IS1003"),
        _row("IS1003b", "MIO034", "IS1003"),
    ]

    violations = validate_session_honesty(
        rows, {"ES2002a", "IS1003a"}, {"ES2002b", "IS1003b"}
    )

    assert len(violations) == 2
    assert "FEE005" in violations[0]
    assert "MIO034" in violations[1]


def test_serialize_parse_manifest_round_trip_preserves_persisted_fields() -> None:
    manifest = ProtocolManifest(
        schema_version=2,
        corpus="ami_test",
        truth_source="curated_gold",
        selection_seed="issue-113-a2",
        split_seed="split-v1",
        open_set_selection=OpenSetSelection(
            requested=1, selected=["IB4001"], candidates=4
        ),
        rows=[
            AttributionProtocolRow(
                "ES2002a", "A", 0, "FEE005", "ES2002", "test_genuine", "dev"
            )
        ],
        recurrence_report=_report(),
        exclusions=[ExclusionEntry("IB4001", "headset dropout")],
    )

    serialized = serialize_manifest(manifest)

    assert "speakers" not in serialized["recurrence_report"]
    assert parse_manifest(serialized) == manifest


def test_build_protocol_empty_meetings() -> None:
    assert build_protocol({}, _report()) == []


def test_build_protocol_single_meeting() -> None:
    meetings = {"ES2002a": {"A": MeetingSpeaker("A", 0, "FEE005")}}

    assert build_protocol(meetings, _report()) == [
        AttributionProtocolRow("ES2002a", "A", 0, "FEE005", "ES2002")
    ]


def test_build_protocol_all_meetings_excluded() -> None:
    meetings = {"ES2002a": {"A": MeetingSpeaker("A", 0, "FEE005")}}

    assert build_protocol(meetings, _report(), exclude={"ES2002a"}) == []


def _split_fixture() -> tuple[
    dict[str, dict[str, MeetingSpeaker]],
    set[str],
    set[str],
    set[str],
]:
    meetings: dict[str, dict[str, MeetingSpeaker]] = {}
    enrollment: set[str] = set()
    genuine: set[str] = set()
    opened: set[str] = set()
    for index in range(6):
        speaker = f"cross-{index}"
        enrollment_id = f"EN{index:04d}a"
        test_id = f"TS{index:04d}a"
        enrollment.add(enrollment_id)
        genuine.add(test_id)
        meetings[enrollment_id] = {"A": MeetingSpeaker("A", 0, speaker)}
        meetings[test_id] = {"A": MeetingSpeaker("A", 0, speaker)}
    # Link two test sessions into one component through a shared test speaker.
    meetings["TS0000a"]["B"] = MeetingSpeaker("B", 1, "bridge")
    meetings["TS0001a"]["B"] = MeetingSpeaker("B", 1, "bridge")
    for index in range(6):
        meeting_id = f"OP{index:04d}a"
        opened.add(meeting_id)
        meetings[meeting_id] = {
            "A": MeetingSpeaker("A", 0, f"open-{index}")
        }
    return meetings, enrollment, genuine, opened


def test_assign_roles_and_splits_is_deterministic_and_honest() -> None:
    meetings, enrollment, genuine, opened = _split_fixture()
    recurrence = check_kill_criterion(meetings)
    kwargs = {
        "enrollment_meetings": enrollment,
        "test_meetings": genuine,
        "open_set_meetings": opened,
        "split_seed": "split-v1",
    }

    first = assign_roles_and_splits(meetings, recurrence, **kwargs)
    second = assign_roles_and_splits(meetings, recurrence, **kwargs)
    rows = build_protocol(meetings, recurrence, assignments=first)

    assert first == second
    assert {first[meeting][1] for meeting in genuine} == {"dev", "confirm"}
    assert all(first[meeting] == ("enrollment", "enrollment") for meeting in enrollment)
    open_counts = {
        split: sum(first[meeting][1] == split for meeting in opened)
        for split in ("dev", "confirm")
    }
    assert abs(open_counts["dev"] - open_counts["confirm"]) <= 1
    assert validate_split_honesty(rows) == []

    test_splits_by_speaker: dict[str, set[str]] = {}
    for row in rows:
        if row.role == "test_genuine":
            test_splits_by_speaker.setdefault(row.host_global_name, set()).add(row.split)
    assert all(len(splits) == 1 for splits in test_splits_by_speaker.values())


def test_validate_split_honesty_detects_all_violation_types() -> None:
    rows = [
        _row("EN0001a", "shared", "EN0001", "enrollment", "dev"),
        _row("TS0001a", "shared", "TS0001", "test_genuine", "dev"),
        _row("TS0002a", "shared", "TS0002", "test_genuine", "confirm"),
        _row("OP0001a", "shared", "OP0001", "test_open", "dev"),
    ]

    violations = validate_split_honesty(rows)

    assert len(violations) == 3
    assert any("enrollment meeting" in violation for violation in violations)
    assert any("both dev and confirm" in violation for violation in violations)
    assert any("contains enrolled speakers" in violation for violation in violations)


def test_parse_schema_one_requires_regeneration() -> None:
    with pytest.raises(ValueError, match=r"regenerate.*schema_version 2"):
        parse_manifest({"schema_version": 1})
