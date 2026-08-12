"""Gate-metrics assembly: transitions, audits, and the no-information bound."""

import pytest

from voxint.harness import name_accuracy as na
from voxint.harness.gate_metrics import Z_ONE_SIDED_95, assemble_gate_metrics


def _rec(item_id: str, base: str, cand: str) -> dict[str, str]:
    return {"item_id": item_id, "baseline_verdict": base, "candidate_verdict": cand}


def test_transition_counts() -> None:
    scorer = [
        _rec("i1", na.TP, na.FN),  # host dropped
        _rec("i2", na.TP, na.FP_WRONG),  # correct -> wrong swap
        _rec("i3", na.TP, na.TP),  # stable, no count
        _rec("i4", na.FP_OVERNAME, na.FP_OVERNAME),  # persisting over-name
        _rec("i5", na.TN, na.FP_OVERNAME),  # NEW over-name
        _rec("i6", na.FN, na.TP),  # a fix, no regression count
    ]
    gm = assemble_gate_metrics(scorer_per_item=scorer)
    assert gm.host_regression_count == 1
    assert gm.correct_to_wrong_swaps == 1
    assert gm.baseline_fp_overname == 1
    assert gm.candidate_fp_overname == 2
    assert gm.new_fp_overname_additions == 1


def test_strict_controls_counted_only_when_ids_supplied() -> None:
    scorer = [_rec("i1", na.TN, na.FP_OVERNAME), _rec("i2", na.TN, na.FP_OVERNAME)]
    gm = assemble_gate_metrics(scorer_per_item=scorer)
    assert gm.overnaming_strict_count == 0
    gm = assemble_gate_metrics(scorer_per_item=scorer, strict_control_ids=["i2"])
    assert gm.overnaming_strict_count == 1


def test_audit_regressions_and_candidate_host_ok() -> None:
    audit = [
        {"item_id": "a1", "host_ok": True, "candidate_verdict": na.FN},  # regressed
        {"item_id": "a2", "host_ok": True, "candidate_verdict": na.TP},  # held
        {"item_id": "a3", "host_ok": False, "candidate_verdict": na.TN},  # improved
    ]
    gm = assemble_gate_metrics(scorer_per_item=[], audit_per_item=audit)
    assert gm.audit_regression_count == 1
    assert gm.audit_candidate_host_ok == 2


def test_no_information_tally_yields_upper_bound_one() -> None:
    gm = assemble_gate_metrics(scorer_per_item=[])
    assert gm.paired_n == 0
    assert gm.item_regression_rate_upper == 1.0


def test_paired_tally_uses_one_sided_wilson() -> None:
    gm = assemble_gate_metrics(
        scorer_per_item=[], paired_tally={"n_items": 60, "item_regressed": 2}
    )
    expected = na.wilson_ci(2, 60, z=Z_ONE_SIDED_95)[1]
    assert gm.item_regression_rate_upper == pytest.approx(expected)
    assert gm.item_regression_rate_upper < 0.1
    assert gm.paired_n == 60


def test_contamination_is_a_passthrough() -> None:
    gm = assemble_gate_metrics(scorer_per_item=[], contamination_count=3)
    assert gm.contamination_count == 3


def test_invalid_inputs_rejected_not_coerced() -> None:
    with pytest.raises(ValueError, match="bad baseline_verdict"):
        assemble_gate_metrics(scorer_per_item=[_rec("i1", "BOGUS", na.TP)])
    with pytest.raises(ValueError, match="host_ok must be a boolean"):
        assemble_gate_metrics(
            scorer_per_item=[],
            audit_per_item=[{"item_id": "a1", "host_ok": "false", "candidate_verdict": na.TP}],
        )
    with pytest.raises(ValueError, match="bad candidate_verdict"):
        assemble_gate_metrics(
            scorer_per_item=[],
            audit_per_item=[{"item_id": "a1", "host_ok": True, "candidate_verdict": "eh"}],
        )
    with pytest.raises(ValueError, match="contamination_count"):
        assemble_gate_metrics(scorer_per_item=[], contamination_count=-1)
    with pytest.raises(ValueError, match="item_regressed"):
        assemble_gate_metrics(
            scorer_per_item=[], paired_tally={"n_items": 5, "item_regressed": 6}
        )
