"""Pure tests for the gold-anchored attribution roster builder."""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tools.build_attribution_manifest import ManifestError, RunManifest
from tools.build_attribution_roster import (
    EnrolledSource,
    EnrollmentCandidate,
    build_fingerprint_document,
    build_roster,
    roster_fingerprint,
    select_enrollment_candidates,
)

from voxint.harness.ami_recurrence import RecurrenceReport
from voxint.harness.attribution_aligner import (
    AlignmentReport,
    SlotAlignment,
    SlotClassification,
)
from voxint.harness.attribution_protocol import (
    AttributionProtocolRow,
    OpenSetSelection,
    ProtocolManifest,
)


def _alignment(
    label: str,
    name: str | None,
    purity: float,
    duration: float,
    classification: SlotClassification = SlotClassification.GENUINE,
) -> SlotAlignment:
    return SlotAlignment(label, classification, name, purity, 0.8, 0.5, duration)


def _report(*alignments: SlotAlignment) -> AlignmentReport:
    return AlignmentReport(list(alignments), len(alignments), 0, 0, 0, 0, 0)


def test_select_enrollment_candidates_picks_purity_then_duration() -> None:
    run_a = uuid.uuid4()
    run_b = uuid.uuid4()
    reports = {
        "EN0001a": _report(
            _alignment("S0", "Alice", 0.91, 50.0),
            _alignment("S1", "Bob", 0.99, 60.0, SlotClassification.MIXED),
        ),
        "EN0002a": _report(
            _alignment("S2", "Alice", 0.95, 20.0),
            _alignment("S3", "Carol", 0.90, 30.0),
        ),
    }

    selected, unenrollable = select_enrollment_candidates(
        reports,
        {"EN0001a": {"Alice", "Bob"}, "EN0002a": {"Alice", "Carol"}},
        {"EN0001a": run_a, "EN0002a": run_b},
        {"Alice", "Bob", "Carol", "Dana"},
    )

    assert selected["Alice"].diarization_label == "S2"
    assert selected["Carol"].diarization_label == "S3"
    assert unenrollable == ("Bob", "Dana")


def test_select_enrollment_candidates_uses_duration_on_equal_purity() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    selected, _ = select_enrollment_candidates(
        {
            "EN0001a": _report(_alignment("short", "Alice", 0.95, 10.0)),
            "EN0002a": _report(_alignment("long", "Alice", 0.95, 20.0)),
        },
        {"EN0001a": {"Alice"}, "EN0002a": {"Alice"}},
        {"EN0001a": first, "EN0002a": second},
        {"Alice"},
    )
    assert selected["Alice"].diarization_label == "long"


def test_roster_fingerprint_is_order_independent_and_rounded() -> None:
    speaker_a = uuid.uuid4()
    speaker_b = uuid.uuid4()
    source_a = EnrolledSource(
        speaker_a,
        EnrollmentCandidate("Alice", "EN1", uuid.uuid4(), "S0", 0.912345, 12.34567),
    )
    source_b = EnrolledSource(
        speaker_b,
        EnrollmentCandidate("Bob", "EN2", uuid.uuid4(), "S1", 0.987654, 20.11119),
    )
    changed_below_rounding = EnrolledSource(
        speaker_a,
        EnrollmentCandidate("Alice", "EN1", source_a.candidate.run_id, "S0", 0.9123449, 12.3456),
    )

    assert roster_fingerprint([source_a, source_b]) == roster_fingerprint(
        [source_b, source_a]
    )
    assert roster_fingerprint([source_a]) == roster_fingerprint([changed_below_rounding])


def test_fingerprint_document_carries_provenance() -> None:
    source = EnrolledSource(
        uuid.uuid4(),
        EnrollmentCandidate("Alice", "EN1", uuid.uuid4(), "S0", 0.9, 12.0),
    )
    document = build_fingerprint_document(
        [source], git_sha="abc123", exported_at="2026-09-13T12:00:00+00:00"
    )
    assert document["count"] == 1
    assert document["git_sha"] == "abc123"
    assert document["sha256"] == roster_fingerprint([source])
    assert document["entries"][0]["global_name"] == "Alice"


def _protocol() -> ProtocolManifest:
    return ProtocolManifest(
        selection_seed="selection",
        split_seed="split",
        open_set_selection=OpenSetSelection(requested=0, selected=[], candidates=0),
        rows=[
            AttributionProtocolRow(
                "EN1a", "A", 0, "Alice", "EN1", "enrollment", "enrollment"
            )
        ],
        recurrence_report=RecurrenceReport(1, 1, 1, 1, 0, 0, True, False, {}),
        exclusions=[],
    )


def _mock_roster_planning(
    monkeypatch: pytest.MonkeyPatch, candidate: EnrollmentCandidate
) -> None:
    monkeypatch.setattr("tools.build_attribution_roster.resolve_gold_rttm_paths", lambda *_: {})
    monkeypatch.setattr("tools.build_attribution_roster._validate_enrollment_runs", lambda *_: None)
    monkeypatch.setattr("tools.build_attribution_roster._assert_clean_roster", lambda *_: None)
    monkeypatch.setattr("tools.build_attribution_roster._alignment_reports", lambda *_: {})
    monkeypatch.setattr(
        "tools.build_attribution_roster.select_enrollment_candidates",
        lambda *_: ({"Alice": candidate}, ()),
    )
    monkeypatch.setattr("tools.build_attribution_roster.get_settings", Mock())
    monkeypatch.setattr("tools.build_attribution_roster.gates_from_settings", Mock())


def test_build_roster_calls_enrollment_with_replay_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = uuid.uuid4()
    speaker_id = uuid.uuid4()
    candidate = EnrollmentCandidate("Alice", "EN1a", run_id, "S0", 0.95, 20.0)
    _mock_roster_planning(monkeypatch, candidate)
    enroll = Mock(return_value=SimpleNamespace(speaker_id=speaker_id))
    monkeypatch.setattr("tools.build_attribution_roster.enroll_new_speaker", enroll)

    sources, unenrollable, _ = build_roster(
        Mock(),
        protocol=_protocol(),
        run_manifest=RunManifest({"EN1a": run_id}),
        gold_rttm_dir=tmp_path,
        operator="calibration-114",
        allow_existing=False,
        dry_run=False,
    )
    assert sources == [EnrolledSource(speaker_id, candidate)]
    assert unenrollable == ()
    assert enroll.call_args.kwargs["idempotency_key"] == "calib114:Alice"


def test_build_roster_converts_enrollment_failure_to_manifest_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voxint.adjudication.enrollment import EnrollmentError

    run_id = uuid.uuid4()
    candidate = EnrollmentCandidate("Alice", "EN1a", run_id, "S0", 0.95, 20.0)
    _mock_roster_planning(monkeypatch, candidate)
    monkeypatch.setattr(
        "tools.build_attribution_roster.enroll_new_speaker",
        Mock(side_effect=EnrollmentError("no usable centroid")),
    )

    with pytest.raises(ManifestError, match="cannot enroll 'Alice'"):
        build_roster(
            Mock(),
            protocol=_protocol(),
            run_manifest=RunManifest({"EN1a": run_id}),
            gold_rttm_dir=tmp_path,
            operator="calibration-114",
            allow_existing=False,
            dry_run=False,
        )
