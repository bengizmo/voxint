"""Calibration gates contract (issue #114).

Pins the grounded-gate defaults and the calibration rule parameters.
The AMI corpus calibration returned NO_DECISION, so defaults must remain
at the base gates until a future certification succeeds. The evidence
pack in tests/parity/fixtures/attribution/calibration/ records the
outcome for audit.
"""

import json
from pathlib import Path

import pytest

from voxint.harness.calibration import MIN_INDEPENDENT_CLUSTERS, SelectionRule
from voxint.speakers.matching import MatchingGates

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "parity"
    / "fixtures"
    / "attribution"
    / "calibration"
)


def test_grounded_defaults_match_base_gates() -> None:
    """Defaults must not change while calibration status is NO_DECISION."""
    defaults = MatchingGates()
    assert defaults.grounded_min_cosine == 0.70
    assert defaults.grounded_min_margin == 0.08
    assert defaults.grounded_min_vote_agreement == 0.67


def test_base_gates_fixture_matches_code_defaults() -> None:
    """The committed base_gates.json must equal MatchingGates() defaults."""
    base = json.loads((_FIXTURES / "base_gates.json").read_text())
    defaults = MatchingGates()
    assert base["grounded_min_cosine"] == defaults.grounded_min_cosine
    assert base["grounded_min_margin"] == defaults.grounded_min_margin
    assert base["grounded_min_vote_agreement"] == defaults.grounded_min_vote_agreement
    assert base["min_cosine"] == defaults.min_cosine
    assert base["min_margin"] == defaults.min_margin


def test_min_independent_clusters_constant() -> None:
    assert MIN_INDEPENDENT_CLUSTERS == 50


def test_selection_rule_grid() -> None:
    rule = SelectionRule()
    assert rule.cosine_grid == (0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80)
    assert rule.margin_grid == (0.05, 0.08, 0.10, 0.12)
    assert rule.max_far_upper_one_sided == 0.05
    assert rule.confidence == 0.95
    assert rule.min_impostor_clusters == MIN_INDEPENDENT_CLUSTERS


def test_certification_outcome_is_no_decision() -> None:
    """The committed evidence records NO_DECISION; no silent flip."""
    cert = json.loads((_FIXTURES / "certification.json").read_text())
    assert cert["outcome"] == "NO_DECISION"
    assert "insufficient_impostor_clusters" in cert["reasons"]
    assert "far_bound_exceeded" in cert["reasons"]
    assert cert["post"]["auto_wrong"] == 0
    assert cert["post"]["n_impostor_clusters"] == 49
    assert cert["post"]["far_upper_one_sided"] == pytest.approx(0.0523, abs=0.001)


def test_selection_outcome_is_selected() -> None:
    sel = json.loads((_FIXTURES / "selection.json").read_text())
    assert sel["outcome"] == "SELECTED"
    assert sel["chosen_gates"]["grounded_min_cosine"] == 0.80
    assert sel["chosen_gates"]["grounded_min_margin"] == 0.12
