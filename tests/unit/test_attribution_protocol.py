from voxint.harness.ami_recurrence import MeetingSpeaker, RecurrenceReport
from voxint.harness.attribution_protocol import (
    AttributionProtocolRow,
    ExclusionEntry,
    ProtocolManifest,
    build_protocol,
    parse_manifest,
    serialize_manifest,
    validate_session_honesty,
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


def _row(meeting_id: str, speaker: str, session: str) -> AttributionProtocolRow:
    return AttributionProtocolRow(meeting_id, "A", 0, speaker, session)


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
        rows=[AttributionProtocolRow("ES2002a", "A", 0, "FEE005", "ES2002")],
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
