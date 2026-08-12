"""CLI round trips for ``voxint score`` — file in, report out, errors with line numbers."""

import json
import math
from pathlib import Path

import pytest

from voxint.cli import main

SPACE = "model-a"
DIMS = 3


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _name_items(path: Path) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "item_id": "ep-001",
                "slots": {
                    "S0": {
                        "assigned_name": "Dana Fox",
                        "truth": "Dana Fox",
                        "confidence": 0.9,
                        "duration": 100.0,
                    },
                    "S1": {"assigned_name": None, "truth": "__ABSTAIN__"},
                },
            },
            {
                "item_id": "ep-002",
                "slots": {
                    "S0": {"assigned_name": "Dana Wolfe", "truth": "Dana Fox", "confidence": 0.4},
                    "S1": {"assigned_name": "Ling Wei", "truth": "__NEITHER_DETERMINABLE__"},
                },
            },
        ],
    )


def _thresholds(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tau": 0.6,
                "margin": 0.1,
                "min_duration": 30.0,
                "min_segments": 4,
                "low_band": 0.3,
                "neg_min_total_duration": 300.0,
                "min_enrollment_items": 3,
            }
        ),
        encoding="utf-8",
    )
    return path


def _enrollment(path: Path, *, held_out: bool = True, items: int = 5) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "dims": DIMS,
                "voiceprints": {
                    "host-dana": {
                        "embedding": [1.0, 0.0, 0.0],
                        "enrollment_items": items,
                        "held_out": held_out,
                        "source_item_ids": ["ep-900"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _slot_record(
    item_id: str,
    kind: str,
    embedding: list[float],
    *,
    host_id: str | None = "host-dana",
    total_speech: float = 1000.0,
    space: str = SPACE,
) -> dict[str, object]:
    record: dict[str, object] = {
        "item_id": item_id,
        "kind": kind,
        "embedding_space": space,
        "total_speech": total_speech,
        "slots": {"S0": {"embedding": embedding, "duration": 100.0, "segments": 10}},
    }
    if host_id is not None:
        record["host_id"] = host_id
    return record


# --------------------------------------------------------------------------- #
# name-accuracy
# --------------------------------------------------------------------------- #
def test_name_accuracy_report(tmp_path: Path) -> None:
    items = _name_items(tmp_path / "items.jsonl")
    out = tmp_path / "report.json"
    assert main(["score", "name-accuracy", str(items), "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["schema_version"] == 1
    assert report["n_items"] == 2
    agg = report["aggregate"]
    assert (agg["tp"], agg["fp_wrong"], agg["tn"], agg["excluded"]) == (1, 1, 1, 1)
    assert report["n_slots_scored"] == 3
    assert math.isclose(report["accuracy"], 2 / 3)
    assert "risk_coverage" in report  # confidences were present
    per_item = {rec["item_id"]: rec["verdicts"] for rec in report["per_item"]}
    assert per_item["ep-002"]["S1"] == "EXCLUDED"


def test_name_accuracy_deterministic_output(tmp_path: Path) -> None:
    items = _name_items(tmp_path / "items.jsonl")
    out1, out2 = tmp_path / "r1.json", tmp_path / "r2.json"
    assert main(["score", "name-accuracy", str(items), "--out", str(out1)]) == 0
    assert main(["score", "name-accuracy", str(items), "--out", str(out2)]) == 0
    assert out1.read_bytes() == out2.read_bytes()
    # Atomic write leaves no temp droppings.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_name_accuracy_stdout_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = _name_items(tmp_path / "items.jsonl")
    assert main(["score", "name-accuracy", str(items)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "name_accuracy_report"


def test_name_accuracy_aliases_change_verdicts(tmp_path: Path) -> None:
    items = _write_jsonl(
        tmp_path / "items.jsonl",
        [{"item_id": "e1", "slots": {"S0": {"assigned_name": "Dee", "truth": "Daniela Fox"}}}],
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"schema_version": 1, "aliases": {"Daniela Fox": ["Dee"]}}),
        encoding="utf-8",
    )
    out = tmp_path / "r.json"
    assert main(["score", "name-accuracy", str(items), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["aggregate"]["fp_wrong"] == 1
    assert (
        main(["score", "name-accuracy", str(items), "--aliases", str(aliases), "--out", str(out)])
        == 0
    )
    assert json.loads(out.read_text())["aggregate"]["tp"] == 1


def test_name_accuracy_paired_baseline(tmp_path: Path) -> None:
    baseline = _write_jsonl(
        tmp_path / "base.jsonl",
        [
            {
                "item_id": "e1",
                "slots": {
                    "S0": {"assigned_name": "Dana Fox", "truth": "Dana Fox"},
                    "S1": {"assigned_name": None, "truth": "Ling Wei"},
                },
            }
        ],
    )
    candidate = _write_jsonl(
        tmp_path / "cand.jsonl",
        [
            {
                "item_id": "e1",
                "slots": {
                    "S0": {"assigned_name": None, "truth": "Dana Fox"},  # regression
                    "S1": {"assigned_name": "Ling Wei", "truth": "Ling Wei"},  # fix
                },
            }
        ],
    )
    out = tmp_path / "r.json"
    assert (
        main(
            [
                "score",
                "name-accuracy",
                str(candidate),
                "--baseline",
                str(baseline),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    paired = json.loads(out.read_text())["paired"]
    assert paired["n_slots"] == 2
    assert paired["mcnemar"]["baseline_correct_candidate_wrong"] == 1
    assert paired["mcnemar"]["baseline_wrong_candidate_correct"] == 1
    assert paired["mcnemar"]["net"] == 0
    assert "bootstrap" in paired


def test_name_accuracy_baseline_item_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _name_items(tmp_path / "items.jsonl")
    baseline = _write_jsonl(
        tmp_path / "base.jsonl",
        [{"item_id": "other", "slots": {"S0": {"assigned_name": None, "truth": None}}}],
    )
    assert main(["score", "name-accuracy", str(items), "--baseline", str(baseline)]) == 2
    assert "item_ids differ" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line,fragment",
    [
        ("{not json", "invalid JSON"),
        ('["a list"]', "expected a JSON object"),
        ('{"item_id": "", "slots": {}}', "'item_id'"),
        ('{"item_id": "x", "slots": {}}', "'slots'"),
        ('{"item_id": "x", "slots": {"S0": {"assigned_name": 3}}}', "assigned_name"),
    ],
)
def test_name_accuracy_malformed_line_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], line: str, fragment: str
) -> None:
    good = json.dumps(
        {"item_id": "ok", "slots": {"S0": {"assigned_name": None, "truth": "__ABSTAIN__"}}}
    )
    path = tmp_path / "items.jsonl"
    path.write_text(good + "\n" + line + "\n", encoding="utf-8")
    assert main(["score", "name-accuracy", str(path)]) == 2
    err = capsys.readouterr().err
    assert f"{path}:2" in err and fragment in err


def test_name_accuracy_duplicate_and_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = {"item_id": "x", "slots": {"S0": {"assigned_name": None, "truth": "__ABSTAIN__"}}}
    dup = _write_jsonl(tmp_path / "dup.jsonl", [rec, rec])
    assert main(["score", "name-accuracy", str(dup)]) == 2
    assert "duplicate item_id" in capsys.readouterr().err

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["score", "name-accuracy", str(empty)]) == 2
    assert "no records" in capsys.readouterr().err


def test_name_accuracy_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", "name-accuracy", str(tmp_path / "nope.jsonl")]) == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #
def _run_agreement(tmp_path: Path, records: list[dict[str, object]], **enroll: object) -> int:
    slots = _write_jsonl(tmp_path / "slots.jsonl", records)
    args = [
        "score",
        "agreement",
        "--slots",
        str(slots),
        "--enrollment",
        str(_enrollment(tmp_path / "enroll.json", **enroll)),  # type: ignore[arg-type]
        "--thresholds",
        str(_thresholds(tmp_path / "thresholds.json")),
        "--out",
        str(tmp_path / "verdicts.jsonl"),
    ]
    return main(args)


def _verdicts(tmp_path: Path) -> list[dict[str, object]]:
    lines = (tmp_path / "verdicts.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_agreement_curated_confident_present(tmp_path: Path) -> None:
    rc = _run_agreement(tmp_path, [_slot_record("ep-001", "curated", [1.0, 0.0, 0.0])])
    assert rc == 0
    (rec,) = _verdicts(tmp_path)
    assert rec["verdict"] == "CONFIDENT_HOST_PRESENT"
    assert rec["embedding_space"] == SPACE
    assert rec["host_slot"] == "S0" and rec["item_id"] == "ep-001"


def test_agreement_leakage_abstains(tmp_path: Path) -> None:
    # ep-900 is in the voiceprint's source_item_ids.
    rc = _run_agreement(tmp_path, [_slot_record("ep-900", "curated", [1.0, 0.0, 0.0])])
    assert rc == 0
    (rec,) = _verdicts(tmp_path)
    assert rec["verdict"] == "ABSTAIN" and rec["reason"] == "session_leakage_risk"


def test_agreement_not_held_out_abstains(tmp_path: Path) -> None:
    rc = _run_agreement(
        tmp_path, [_slot_record("ep-001", "curated", [1.0, 0.0, 0.0])], held_out=False
    )
    assert rc == 0
    (rec,) = _verdicts(tmp_path)
    assert rec["reason"] == "session_leakage_risk"


def test_agreement_weak_enrollment_abstains(tmp_path: Path) -> None:
    rc = _run_agreement(
        tmp_path, [_slot_record("ep-001", "curated", [1.0, 0.0, 0.0])], items=1
    )
    assert rc == 0
    (rec,) = _verdicts(tmp_path)
    assert rec["reason"] == "weak_enrollment"


def test_agreement_negative_control_silver_absence(tmp_path: Path) -> None:
    rc = _run_agreement(
        tmp_path,
        [_slot_record("ep-001", "negative_control", [0.0, 1.0, 0.0], host_id=None)],
    )
    assert rc == 0
    (rec,) = _verdicts(tmp_path)
    assert rec["verdict"] == "NO_CURATED_HOST_DETECTED"


@pytest.mark.parametrize(
    "record,fragment",
    [
        (_slot_record("e1", "curated", [1.0, 0.0, 0.0], host_id="ghost"), "unknown host_id"),
        (_slot_record("e1", "curated", [1.0, 0.0]), "dims"),
        (_slot_record("e1", "bogus", [1.0, 0.0, 0.0]), "'kind'"),
        (_slot_record("e1", "curated", [1.0, 0.0, 0.0], host_id=None), "'host_id'"),
    ],
)
def test_agreement_bad_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    record: dict[str, object],
    fragment: str,
) -> None:
    assert _run_agreement(tmp_path, [record]) == 2
    assert fragment in capsys.readouterr().err


def test_agreement_duplicate_item(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rec = _slot_record("e1", "curated", [1.0, 0.0, 0.0])
    assert _run_agreement(tmp_path, [rec, rec]) == 2
    assert "duplicate item_id" in capsys.readouterr().err


def test_agreement_invalid_thresholds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    slots = _write_jsonl(
        tmp_path / "slots.jsonl", [_slot_record("e1", "curated", [1.0, 0.0, 0.0])]
    )
    bad = tmp_path / "bad-thresholds.json"
    payload = json.loads(_thresholds(tmp_path / "t.json").read_text())
    payload["low_band"] = 0.9  # > tau
    bad.write_text(json.dumps(payload), encoding="utf-8")
    rc = main(
        [
            "score",
            "agreement",
            "--slots",
            str(slots),
            "--enrollment",
            str(_enrollment(tmp_path / "enroll.json")),
            "--thresholds",
            str(bad),
        ]
    )
    assert rc == 2
    assert "low_band" in capsys.readouterr().err


def test_agreement_wrong_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slots = _write_jsonl(
        tmp_path / "slots.jsonl", [_slot_record("e1", "curated", [1.0, 0.0, 0.0])]
    )
    enroll = _enrollment(tmp_path / "enroll.json")
    payload = json.loads(enroll.read_text())
    payload["schema_version"] = 99
    enroll.write_text(json.dumps(payload), encoding="utf-8")
    rc = main(
        [
            "score",
            "agreement",
            "--slots",
            str(slots),
            "--enrollment",
            str(enroll),
            "--thresholds",
            str(_thresholds(tmp_path / "t.json")),
        ]
    )
    assert rc == 2
    assert "schema_version" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
def _voter_line(
    item_id: str,
    verdict: str,
    *,
    space: str,
    kind: str = "curated",
    host_slot: str | None = "S0",
    contradiction: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "item_id": item_id,
        "kind": kind,
        "embedding_space": space,
        "verdict": verdict,
        "host_slot": host_slot,
        "contradiction": contradiction,
    }


def test_ensemble_fuses_verdicts(tmp_path: Path) -> None:
    a = _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _voter_line("e1", "CONFIDENT_HOST_PRESENT", space="model-a"),
            _voter_line("e2", "NO_CURATED_HOST_DETECTED", space="model-a", kind="negative_control"),
        ],
    )
    b = _write_jsonl(
        tmp_path / "b.jsonl",
        [
            _voter_line("e1", "CONFIDENT_HOST_PRESENT", space="model-b"),
            _voter_line("e2", "NO_CURATED_HOST_DETECTED", space="model-b", kind="negative_control"),
        ],
    )
    out = tmp_path / "fused.jsonl"
    assert main(["score", "ensemble", str(a), str(b), "--out", str(out)]) == 0
    fused = {json.loads(line)["item_id"]: json.loads(line) for line in out.read_text().splitlines()}
    assert fused["e1"]["verdict"] == "SILVER_HOST_PRESENT"
    assert fused["e2"]["verdict"] == "SILVER_NO_HOST"
    assert fused["e1"]["voter_a"]["embedding_space"] == "model-a"


def test_ensemble_item_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _write_jsonl(tmp_path / "a.jsonl", [_voter_line("e1", "ABSTAIN", space="model-a")])
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e2", "ABSTAIN", space="model-b")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "different item_ids" in capsys.readouterr().err


def test_ensemble_kind_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _write_jsonl(tmp_path / "a.jsonl", [_voter_line("e1", "ABSTAIN", space="model-a")])
    b = _write_jsonl(
        tmp_path / "b.jsonl",
        [_voter_line("e1", "ABSTAIN", space="model-b", kind="negative_control")],
    )
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "disagree on kind" in capsys.readouterr().err


def test_ensemble_mixed_spaces_within_one_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_jsonl(
        tmp_path / "a.jsonl",
        [
            _voter_line("e1", "ABSTAIN", space="model-a"),
            _voter_line("e2", "ABSTAIN", space="model-c"),
        ],
    )
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e1", "ABSTAIN", space="model-b")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "mixed embedding spaces" in capsys.readouterr().err


def test_ensemble_unknown_verdict_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_jsonl(tmp_path / "a.jsonl", [_voter_line("e1", "MAYBE", space="model-a")])
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e1", "ABSTAIN", space="model-b")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "unknown verdict" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Review-finding regressions
# --------------------------------------------------------------------------- #
def test_name_accuracy_negative_duration_is_a_clean_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = _write_jsonl(
        tmp_path / "items.jsonl",
        [
            {
                "item_id": "e1",
                "slots": {
                    "S0": {"assigned_name": "Dana Fox", "truth": "Dana Fox", "duration": -5}
                },
            }
        ],
    )
    assert main(["score", "name-accuracy", str(items)]) == 2
    err = capsys.readouterr().err
    assert f"{items}:1" in err and "'duration'" in err and ">= 0" in err


def test_name_accuracy_non_finite_confidence_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(
        '{"item_id": "e1", "slots": {"S0": {"assigned_name": null, '
        '"truth": "__ABSTAIN__", "confidence": NaN}}}\n',
        encoding="utf-8",
    )
    assert main(["score", "name-accuracy", str(items)]) == 2
    assert "finite" in capsys.readouterr().err


def test_name_accuracy_baseline_truth_drift_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_jsonl(
        tmp_path / "base.jsonl",
        [{"item_id": "e1", "slots": {"S0": {"assigned_name": None, "truth": "Dana Fox"}}}],
    )
    candidate = _write_jsonl(
        tmp_path / "cand.jsonl",
        [{"item_id": "e1", "slots": {"S0": {"assigned_name": None, "truth": "Ling Wei"}}}],
    )
    assert main(["score", "name-accuracy", str(candidate), "--baseline", str(baseline)]) == 2
    assert "disagree on ground truth" in capsys.readouterr().err


def test_agreement_slots_must_declare_matching_space(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Equal-dims vectors from another model must be rejected, not inherited."""
    record = _slot_record("e1", "curated", [1.0, 0.0, 0.0], space="other-model")
    assert _run_agreement(tmp_path, [record]) == 2
    err = capsys.readouterr().err
    assert "does not match" in err and "other-model" in err

    missing = _slot_record("e1", "curated", [1.0, 0.0, 0.0])
    del missing["embedding_space"]
    assert _run_agreement(tmp_path, [missing]) == 2
    assert "'embedding_space'" in capsys.readouterr().err


def test_agreement_nan_threshold_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slots = _write_jsonl(
        tmp_path / "slots.jsonl", [_slot_record("e1", "curated", [1.0, 0.0, 0.0])]
    )
    bad = tmp_path / "t.json"
    text = _thresholds(tmp_path / "base-t.json").read_text()
    bad.write_text(text.replace('"min_duration": 30.0', '"min_duration": NaN'), encoding="utf-8")
    rc = main(
        [
            "score",
            "agreement",
            "--slots",
            str(slots),
            "--enrollment",
            str(_enrollment(tmp_path / "enroll.json")),
            "--thresholds",
            str(bad),
        ]
    )
    assert rc == 2
    assert "min_duration" in capsys.readouterr().err


def test_ensemble_rejects_same_space_voters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_jsonl(tmp_path / "a.jsonl", [_voter_line("e1", "ABSTAIN", space="model-a")])
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e1", "ABSTAIN", space="model-a")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "independent models" in capsys.readouterr().err


def test_ensemble_confident_present_requires_host_slot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two slot-less confident records must not mint a silver label."""
    a = _write_jsonl(
        tmp_path / "a.jsonl",
        [_voter_line("e1", "CONFIDENT_HOST_PRESENT", space="model-a", host_slot=None)],
    )
    b = _write_jsonl(
        tmp_path / "b.jsonl",
        [_voter_line("e1", "CONFIDENT_HOST_PRESENT", space="model-b", host_slot=None)],
    )
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "'host_slot'" in capsys.readouterr().err


def test_ensemble_verdict_kind_consistency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_jsonl(
        tmp_path / "a.jsonl",
        [_voter_line("e1", "NO_CURATED_HOST_DETECTED", space="model-a", kind="curated")],
    )
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e1", "ABSTAIN", space="model-b")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "only valid on a negative control" in capsys.readouterr().err


def test_ensemble_contradiction_must_be_boolean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _voter_line("e1", "ABSTAIN", space="model-a")
    rec["contradiction"] = "false"
    a = _write_jsonl(tmp_path / "a.jsonl", [rec])
    b = _write_jsonl(tmp_path / "b.jsonl", [_voter_line("e1", "ABSTAIN", space="model-b")])
    assert main(["score", "ensemble", str(a), str(b)]) == 2
    assert "'contradiction' must be a boolean" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Group registration
# --------------------------------------------------------------------------- #
def test_score_without_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["score"])
    assert excinfo.value.code == 2
