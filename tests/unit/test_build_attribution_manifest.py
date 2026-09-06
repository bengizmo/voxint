"""Pure tests for the attribution-manifest maintainer bridge."""

import json
import uuid
from pathlib import Path

import pytest
from tools.build_attribution_manifest import (
    KIND,
    ManifestError,
    RunManifest,
    _dumps,
    assemble_manifest,
    parse_run_manifest,
    reshape_match_evidence,
    resolve_gold_rttm_paths,
)


def test_reshape_match_evidence_by_meeting_and_label() -> None:
    run_a = uuid.uuid4()
    run_b = uuid.uuid4()
    speaker = uuid.uuid4()
    candidates = [
        {
            "pipeline_run_id": run_a,
            "diarization_label": "SPEAKER_00",
            "decision": "accepted",
            "top_speaker_id": speaker,
            "similarity": 0.85,
            "margin": 0.12,
            "vote_agreement": 0.9,
            "grounded": True,
            "eligible_turns": 5,
            "eligible_seconds": 45.2,
            "roster_size": 10,
        },
        {
            "pipeline_run_id": run_b,
            "diarization_label": "SPEAKER_01",
            "decision": "ineligible",
            "top_speaker_id": None,
            "similarity": None,
            "margin": None,
            "vote_agreement": None,
            "grounded": None,
            "eligible_turns": 0,
            "eligible_seconds": 0.0,
            "roster_size": None,
        },
    ]

    result = reshape_match_evidence(candidates, {run_a: "ES2002a", run_b: "ES2002b"})

    accepted = result["ES2002a"]["SPEAKER_00"]
    assert accepted == {
        "run_id": str(run_a),
        "pipeline_run_id": str(run_a),
        "similarity": 0.85,
        "margin": 0.12,
        "vote_agreement": 0.9,
        "eligible_turns": 5,
        "eligible_seconds": 45.2,
        "roster_size": 10,
        "top_speaker_id": str(speaker),
        "decision": "accepted",
        "grounded": True,
    }
    assert result["ES2002b"]["SPEAKER_01"]["decision"] == "ineligible"


def test_reshape_keeps_empty_evidence_for_selected_meeting() -> None:
    run_id = uuid.uuid4()
    assert reshape_match_evidence([], {run_id: "ES2002a"}) == {"ES2002a": {}}


def test_reshape_rejects_duplicate_label() -> None:
    run_id = uuid.uuid4()
    row = {
        "pipeline_run_id": run_id,
        "diarization_label": "SPEAKER_00",
    }
    with pytest.raises(ManifestError, match="duplicate match evidence"):
        reshape_match_evidence([row, row], {run_id: "ES2002a"})


def test_assemble_manifest_uses_paths_relative_to_output(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    gold_dir = inputs / "gold"
    gold_dir.mkdir(parents=True)
    protocol = inputs / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    gold = gold_dir / "ES2002a.rttm"
    gold.write_text("", encoding="utf-8")
    out_dir = tmp_path / "exports" / "pass-1"
    run_id = uuid.uuid4()
    enrolled_id = uuid.uuid4()

    manifest = assemble_manifest(
        protocol_path=protocol,
        gold_paths={"ES2002a": gold},
        run_manifest=RunManifest(runs={"ES2002a": run_id}),
        match_evidence={"ES2002a": {}},
        enrolled_speaker_map={"MEO069": str(enrolled_id)},
        out_dir=out_dir,
        git_sha="abc123",
    )

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == KIND
    assert manifest["protocol_path"] == "../../inputs/protocol.json"
    assert manifest["meetings"]["ES2002a"] == {
        "gold_rttm": "../../inputs/gold/ES2002a.rttm",
        "hypothesis_rttm": "hypothesis_rttm/ES2002a.rttm",
        "run_id": str(run_id),
    }
    assert manifest["match_evidence"] == {"ES2002a": {}}
    assert manifest["enrolled_speaker_map"] == {"MEO069": str(enrolled_id)}
    assert manifest["environment"] == {"git_sha": "abc123"}
    assert json.loads(_dumps(manifest)) == manifest


def test_parse_run_manifest() -> None:
    run_id = uuid.uuid4()
    parsed = parse_run_manifest({"schema_version": 1, "runs": {"ES2002a": str(run_id)}})
    assert parsed.runs == {"ES2002a": run_id}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 2, "runs": {}}, "schema_version"),
        ({"schema_version": 1}, "runs"),
        ({"schema_version": 1, "runs": {}}, "must not be empty"),
        (
            {"schema_version": 1, "runs": {"ES2002a": "not-a-uuid"}},
            "not a valid UUID",
        ),
    ],
)
def test_parse_run_manifest_rejects_invalid_input(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        parse_run_manifest(payload)


def test_resolve_gold_rttm_paths(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    expected = gold_dir / "ES2002a.rttm"
    expected.write_text("RTTM\n", encoding="utf-8")

    resolved = resolve_gold_rttm_paths(gold_dir, ["ES2002a"])

    assert resolved == {"ES2002a": expected.resolve()}


def test_resolve_gold_rttm_paths_rejects_missing_meeting(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    with pytest.raises(ManifestError, match=r"ES2002b.*missing gold RTTM"):
        resolve_gold_rttm_paths(gold_dir, ["ES2002b"])
