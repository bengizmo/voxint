"""Unit tests for attribution role filtering and rematch orchestration."""

import uuid
from unittest.mock import Mock

import pytest
from tools.build_attribution_manifest import ManifestError, RunManifest
from tools.rematch_runs import (
    RunTally,
    format_tally,
    rematch_runs,
    select_runs_by_role,
)

from voxint.harness.ami_recurrence import RecurrenceReport
from voxint.harness.attribution_protocol import (
    AttributionProtocolRow,
    OpenSetSelection,
    ProtocolManifest,
)
from voxint.speakers.matching import MatchingGates


def _protocol(roles: dict[str, str]) -> ProtocolManifest:
    rows = [
        AttributionProtocolRow(
            meeting,
            "A",
            0,
            f"gold-{meeting}",
            meeting[:-1],
            role,  # type: ignore[arg-type]
            "enrollment" if role == "enrollment" else "dev",
        )
        for meeting, role in roles.items()
    ]
    return ProtocolManifest(
        selection_seed="selection",
        split_seed="split",
        open_set_selection=OpenSetSelection(requested=0, selected=[], candidates=0),
        rows=rows,
        recurrence_report=RecurrenceReport(3, 3, 3, 2, 1, 1, True, False, {}),
        exclusions=[],
    )


def test_select_runs_by_role_filters_and_keeps_stable_order() -> None:
    ids = {meeting: uuid.uuid4() for meeting in ("TS1a", "OP1a")}
    selected = select_runs_by_role(
        RunManifest(ids),
        {"TS1a": "test_genuine", "OP1a": "test_open"},
        {"test_open"},
    )
    assert [(item.meeting_id, item.role) for item in selected] == [
        ("OP1a", "test_open")
    ]


def test_select_runs_by_role_rejects_enrollment_selection() -> None:
    with pytest.raises(ManifestError, match="enrollment"):
        select_runs_by_role(
            RunManifest({"EN1a": uuid.uuid4()}),
            {"EN1a": "enrollment"},
            {"test_genuine"},
        )


def test_select_runs_by_role_rejects_unknown_meeting() -> None:
    with pytest.raises(ManifestError, match="absent from protocol"):
        select_runs_by_role(
            RunManifest({"missing": uuid.uuid4()}), {}, {"test_open"}
        )


def test_format_tally_includes_all_decisions_and_roster() -> None:
    tally = RunTally("OP1a", uuid.UUID(int=1), 2, 3, 4, 35)
    assert format_tally(tally).endswith(
        "accepted=2 rejected=3 ineligible=4 roster=35"
    )


def test_rematch_orchestration_leaves_enrollment_count_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = {meeting: uuid.uuid4() for meeting in ("TS1a", "OP1a")}
    protocol = _protocol(
        {"EN1a": "enrollment", "TS1a": "test_genuine", "OP1a": "test_open"}
    )
    session = Mock()
    refreshed: list[uuid.UUID] = []
    monkeypatch.setattr("tools.rematch_runs._validate_completed_runs", lambda *_: None)
    monkeypatch.setattr("tools.rematch_runs._candidate_count", lambda *_: 7)
    monkeypatch.setattr(
        "tools.rematch_runs.refresh_run_matches",
        lambda _session, run_id, _gates: refreshed.append(run_id),
    )
    monkeypatch.setattr(
        "tools.rematch_runs._run_tally",
        lambda _session, selected: RunTally(
            selected.meeting_id, selected.run_id, 1, 2, 3, 4
        ),
    )

    tallies, count = rematch_runs(
        session,
        run_manifest=RunManifest(run_ids),
        protocol=protocol,
        selected_roles={"test_genuine", "test_open"},
        gates=MatchingGates(),
        dry_run=False,
    )

    assert count == 2
    assert set(refreshed) == {run_ids["TS1a"], run_ids["OP1a"]}
    assert len(tallies) == 2
    session.flush.assert_called_once()


def test_rematch_dry_run_performs_no_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid.uuid4()
    session = Mock()
    refresh = Mock()
    monkeypatch.setattr("tools.rematch_runs._validate_completed_runs", lambda *_: None)
    monkeypatch.setattr("tools.rematch_runs._candidate_count", lambda *_: 0)
    monkeypatch.setattr("tools.rematch_runs.refresh_run_matches", refresh)

    tallies, count = rematch_runs(
        session,
        run_manifest=RunManifest({"TS1a": run_id}),
        protocol=_protocol({"TS1a": "test_genuine"}),
        selected_roles={"test_genuine"},
        gates=MatchingGates(),
        dry_run=True,
    )
    assert tallies == ()
    assert count == 1
    refresh.assert_not_called()
    session.flush.assert_not_called()


def test_rematch_fails_if_enrollment_candidate_count_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = {"TS1a": uuid.uuid4()}
    session = Mock()
    monkeypatch.setattr("tools.rematch_runs._validate_completed_runs", lambda *_: None)
    counts = iter((4, 5))
    monkeypatch.setattr("tools.rematch_runs._candidate_count", lambda *_: next(counts))
    monkeypatch.setattr("tools.rematch_runs.refresh_run_matches", Mock())

    with pytest.raises(ManifestError, match="count changed"):
        rematch_runs(
            session,
            run_manifest=RunManifest(run_ids),
            protocol=_protocol({"EN1a": "enrollment", "TS1a": "test_genuine"}),
            selected_roles={"test_genuine"},
            gates=MatchingGates(),
            dry_run=False,
        )
