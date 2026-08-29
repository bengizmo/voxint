"""Tests for the calibration tooling (issue #114 Phase 2).

Covers: trial classification (genuine/impostor/unscoreable), gate sweep with
known-optimal-point recovery, Wilson CI integration, PRE/POST comparator,
independence check, and JSONL round-trip serialization.
"""

import json

from voxint.harness.calibration import (
    SweepPoint,
    Trial,
    TrialKind,
    check_independence,
    classify_trial,
    compare,
    gates_from_dict,
    gates_to_dict,
    sweep,
    sweep_point_to_dict,
    trial_from_dict,
    trial_to_dict,
)
from voxint.speakers.matching import MatchingGates


# ---------------------------------------------------------------------------
# Fixtures: synthetic trials with known properties
# ---------------------------------------------------------------------------
def _trial(
    *,
    run_id: str = "aaaa",
    label: str = "SPEAKER_00",
    cosine: float = 0.80,
    margin: float | None = 0.10,
    vote: float = 0.75,
    turns: int = 5,
    seconds: float = 15.0,
    roster: int = 3,
    speaker: str = "spk-A",
    kind: TrialKind = TrialKind.GENUINE,
    anchoring: str = "post_proposal",
    cluster: str | None = None,
) -> Trial:
    return Trial(
        run_id=run_id,
        label=label,
        similarity=cosine,
        margin=margin,
        vote_agreement=vote,
        eligible_turns=turns,
        eligible_seconds=seconds,
        roster_size=roster,
        top_speaker_id=speaker,
        kind=kind,
        truth_anchoring=anchoring,
        cluster_id=cluster or speaker,
    )


def _make_fixture() -> list[Trial]:
    """5 strong genuines, 5 moderate genuines, 5 impostors, 3 unscoreable.

    Strong genuines: cos=0.85, margin=0.12 -> grounded at default gates.
    Moderate genuines: cos=0.72, margin=0.09 -> accepted, not grounded at default.
    Impostors: cos=0.75, margin=0.10 -> accepted, not grounded at default.
    """
    trials: list[Trial] = []
    for i in range(5):
        trials.append(
            _trial(
                run_id=f"run-{i}",
                label=f"SPKR_{i:02d}",
                cosine=0.85,
                margin=0.12,
                vote=0.80,
                turns=5,
                seconds=15.0,
                speaker=f"genuine-strong-{i}",
                kind=TrialKind.GENUINE,
                cluster=f"cluster-strong-{i}",
            )
        )
    for i in range(5):
        trials.append(
            _trial(
                run_id=f"run-{i + 5}",
                label=f"SPKR_{i + 5:02d}",
                cosine=0.72,
                margin=0.09,
                vote=0.70,
                turns=4,
                seconds=12.0,
                speaker=f"genuine-moderate-{i}",
                kind=TrialKind.GENUINE,
                cluster=f"cluster-moderate-{i}",
            )
        )
    for i in range(5):
        trials.append(
            _trial(
                run_id=f"run-{i + 10}",
                label=f"SPKR_{i + 10:02d}",
                cosine=0.75,
                margin=0.10,
                vote=0.75,
                turns=5,
                seconds=15.0,
                speaker=f"impostor-{i}",
                kind=TrialKind.IMPOSTOR,
                cluster=f"cluster-impostor-{i}",
            )
        )
    for i in range(3):
        trials.append(
            _trial(
                run_id=f"run-{i + 15}",
                label=f"SPKR_{i + 15:02d}",
                cosine=0.65,
                margin=0.05,
                kind=TrialKind.UNSCOREABLE,
                cluster=f"__unscoreable__:run-{i + 15}:SPKR_{i + 15:02d}",
            )
        )
    return trials


# ---------------------------------------------------------------------------
# classify_trial
# ---------------------------------------------------------------------------
class TestClassifyTrial:
    def test_genuine_when_speakers_match(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.GENUINE
        assert trial.cluster_id == "spk-A"

    def test_impostor_when_speakers_differ(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.75,
            margin=0.10,
            vote_agreement=0.75,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-B",
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="independent",
        )
        assert trial.kind == TrialKind.IMPOSTOR
        assert trial.cluster_id == "spk-A"

    def test_rejected_genuine(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="rejected",
            similarity=0.55,
            margin=0.02,
            vote_agreement=0.50,
            eligible_turns=2,
            eligible_seconds=5.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.GENUINE

    def test_unscoreable_no_human_decision(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision=None,
            human_speaker_id=None,
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_exclude(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="exclude",
            human_speaker_id=None,
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_unknown(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="unknown",
            human_speaker_id=None,
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_auto_enroll(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="auto_enroll",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_ineligible(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="ineligible",
            similarity=None,
            margin=None,
            vote_agreement=None,
            eligible_turns=0,
            eligible_seconds=0.0,
            roster_size=None,
            top_speaker_id=None,
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_no_mc(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision=None,
            similarity=None,
            margin=None,
            vote_agreement=None,
            eligible_turns=0,
            eligible_seconds=0.0,
            roster_size=None,
            top_speaker_id=None,
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_unscoreable_assign_no_speaker_ids(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id=None,
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.kind == TrialKind.UNSCOREABLE

    def test_cluster_id_genuine(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.85,
            margin=0.12,
            vote_agreement=0.80,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-A",
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="post_proposal",
        )
        assert trial.cluster_id == "spk-A"

    def test_cluster_id_impostor_uses_human_speaker(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision="accepted",
            similarity=0.75,
            margin=0.10,
            vote_agreement=0.75,
            eligible_turns=5,
            eligible_seconds=15.0,
            roster_size=3,
            top_speaker_id="spk-B",
            human_decision="assign",
            human_speaker_id="spk-A",
            truth_anchoring="independent",
        )
        assert trial.cluster_id == "spk-A"

    def test_cluster_id_unscoreable_unique(self) -> None:
        trial = classify_trial(
            run_id="r1",
            label="L0",
            mc_decision=None,
            similarity=None,
            margin=None,
            vote_agreement=None,
            eligible_turns=0,
            eligible_seconds=0.0,
            roster_size=None,
            top_speaker_id=None,
            human_decision=None,
            human_speaker_id=None,
            truth_anchoring="post_proposal",
        )
        assert trial.cluster_id.startswith("__unscoreable__:")


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
class TestSweep:
    def test_default_gates_strong_genuines_auto_attributed(self) -> None:
        """At default grounded gates (cos=0.70, margin=0.08), strong genuines
        (cos=0.85, margin=0.12) are auto-attributed."""
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        assert len(points) == 1
        p = points[0]
        # Strong genuines (cos=0.85, margin=0.12) AND moderate genuines
        # (cos=0.72, margin=0.09) AND impostors (cos=0.75, margin=0.10)
        # all pass grounded at (0.70, 0.08).
        assert p.auto_correct == 10  # both genuine groups
        assert p.auto_wrong_person == 5  # impostors also grounded
        assert p.far > 0

    def test_tight_gates_exclude_impostors(self) -> None:
        """At cos=0.80, margin=0.10 only the strong genuines (cos=0.85) pass."""
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.80],
            margin_grid=[0.10],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.auto_correct == 5
        assert p.auto_wrong_person == 0
        assert p.far == 0.0

    def test_sweep_grid_produces_correct_count(self) -> None:
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.60, 0.70, 0.80],
            margin_grid=[0.05, 0.08, 0.10],
            base_gates=MatchingGates(),
        )
        assert len(points) == 9

    def test_far_zero_when_no_auto_attributions(self) -> None:
        """No trials auto-attributed, but impostors exist: FAR=0 with a CI."""
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.95],
            margin_grid=[0.25],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.auto_correct == 0
        assert p.auto_wrong_person == 0
        assert p.far == 0.0
        # Traditional FAR: 0/5 impostors; Wilson CI upper > 0 (not 1.0).
        assert 0.0 < p.far_ci_upper < 1.0

    def test_far_no_impostors(self) -> None:
        """No impostor trials at all: FAR=0, CI=1.0 (no information)."""
        genuines = [
            _trial(
                run_id=f"r{i}",
                label=f"L{i}",
                cosine=0.85,
                margin=0.12,
                kind=TrialKind.GENUINE,
                cluster=f"c{i}",
            )
            for i in range(5)
        ]
        points = sweep(
            genuines,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.auto_correct == 5
        assert p.auto_wrong_person == 0
        assert p.far == 0.0
        assert p.far_ci_upper == 1.0

    def test_far_ci_upper_above_point_estimate(self) -> None:
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.far_ci_upper >= p.far

    def test_roster_stratum_r1(self) -> None:
        r1_trial = _trial(
            run_id="r1-solo",
            label="L0",
            cosine=0.80,
            margin=None,
            roster=1,
            kind=TrialKind.GENUINE,
            cluster="solo-speaker",
        )
        trials = [r1_trial, *_make_fixture()]
        points = sweep(
            trials,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
            roster_stratum="R=1",
        )
        p = points[0]
        # Only the R=1 trial should be counted; NULL margin passes for R=1.
        assert p.n_scoreable == 1
        assert p.auto_correct == 1

    def test_roster_stratum_r2_plus(self) -> None:
        r1_trial = _trial(roster=1, kind=TrialKind.GENUINE, cluster="solo")
        trials = [r1_trial, *_make_fixture()]
        points = sweep(
            trials,
            cosine_grid=[0.80],
            margin_grid=[0.10],
            base_gates=MatchingGates(),
            roster_stratum="R>=2",
        )
        p = points[0]
        # R=1 trial excluded; only multi-speaker trials from _make_fixture.
        assert p.n_scoreable == 15  # 5+5+5 scoreable (unscoreable excluded)

    def test_unscoreable_excluded_from_sweep(self) -> None:
        trials = _make_fixture()
        points = sweep(
            trials,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        p = points[0]
        # 18 total, 3 unscoreable, 15 scoreable.
        assert p.n_scoreable == 15

    def test_sweep_recovers_optimal_boundary(self) -> None:
        """The sweep should show FAR=0 exactly at the grid point where the
        grounded threshold first excludes all impostors."""
        trials = _make_fixture()
        # Impostors: cos=0.75, margin=0.10
        # Raising grounded_min_cosine to 0.76 excludes them.
        points = sweep(
            trials,
            cosine_grid=[0.75, 0.76],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        at_075 = points[0]
        at_076 = points[1]
        assert at_075.auto_wrong_person > 0
        assert at_076.auto_wrong_person == 0
        assert at_076.far == 0.0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
class TestCompare:
    def test_no_changes_same_gates(self) -> None:
        trials = _make_fixture()
        gates = MatchingGates()
        result = compare(trials, baseline_gates=gates, candidate_gates=gates)
        assert len(result.changes) == 0
        assert result.baseline_auto_correct == result.candidate_auto_correct

    def test_tightening_moves_impostors_out_of_auto(self) -> None:
        trials = _make_fixture()
        baseline = MatchingGates()  # grounded cos=0.70, margin=0.08
        candidate = MatchingGates(
            grounded_min_cosine=0.80,
            grounded_min_margin=0.10,
        )
        result = compare(trials, baseline_gates=baseline, candidate_gates=candidate)
        # The moderate genuines (cos=0.72) and impostors (cos=0.75) lose
        # AUTO_ATTRIBUTE and gain REVIEW.
        assert result.candidate_auto_wrong == 0
        assert result.candidate_auto_correct < result.baseline_auto_correct
        assert len(result.changes) > 0

    def test_changes_report_correct_kinds(self) -> None:
        trials = _make_fixture()
        baseline = MatchingGates()
        candidate = MatchingGates(grounded_min_cosine=0.80, grounded_min_margin=0.10)
        result = compare(trials, baseline_gates=baseline, candidate_gates=candidate)
        impostor_changes = [c for c in result.changes if c.kind == TrialKind.IMPOSTOR]
        genuine_changes = [c for c in result.changes if c.kind == TrialKind.GENUINE]
        assert len(impostor_changes) == 5
        assert all(c.old_band == "auto_attribute" for c in impostor_changes)
        assert all(c.new_band == "review" for c in impostor_changes)
        assert len(genuine_changes) == 5  # moderate genuines

    def test_n_scoreable_excludes_unscoreable(self) -> None:
        trials = _make_fixture()
        gates = MatchingGates()
        result = compare(trials, baseline_gates=gates, candidate_gates=gates)
        assert result.n_scoreable == 15


# ---------------------------------------------------------------------------
# independence check
# ---------------------------------------------------------------------------
class TestIndependence:
    def test_sufficient_clusters(self) -> None:
        trials = [
            _trial(
                run_id=f"r{i}",
                label=f"L{i}",
                kind=TrialKind.GENUINE,
                cluster=f"cluster-{i}",
            )
            for i in range(60)
        ]
        report = check_independence(trials)
        assert report.sufficient is True
        assert report.n_clusters == 60

    def test_insufficient_clusters(self) -> None:
        trials = [
            _trial(
                run_id=f"r{i}",
                label=f"L{i}",
                kind=TrialKind.GENUINE,
                cluster="same-cluster",
            )
            for i in range(10)
        ]
        report = check_independence(trials)
        assert report.sufficient is False
        assert report.n_clusters == 1
        assert report.n_trials == 10

    def test_unscoreable_excluded(self) -> None:
        trials = [
            _trial(kind=TrialKind.UNSCOREABLE, cluster="__unscoreable__:x:y")
            for _ in range(100)
        ]
        report = check_independence(trials)
        assert report.n_trials == 0
        assert report.n_clusters == 0
        assert report.sufficient is False

    def test_cluster_sizes(self) -> None:
        trials = [
            _trial(run_id="r0", label="L0", kind=TrialKind.GENUINE, cluster="A"),
            _trial(run_id="r1", label="L1", kind=TrialKind.GENUINE, cluster="A"),
            _trial(run_id="r2", label="L2", kind=TrialKind.GENUINE, cluster="B"),
        ]
        report = check_independence(trials)
        assert report.cluster_sizes == {"A": 2, "B": 1}


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_trial_round_trip(self) -> None:
        original = _trial(cosine=0.83, margin=0.11, anchoring="independent")
        d = trial_to_dict(original)
        restored = trial_from_dict(d)
        assert restored == original

    def test_trial_round_trip_null_margin(self) -> None:
        original = _trial(margin=None, roster=1)
        d = trial_to_dict(original)
        restored = trial_from_dict(d)
        assert restored == original
        assert restored.margin is None

    def test_trial_json_deterministic(self) -> None:
        t = _trial()
        d = trial_to_dict(t)
        j1 = json.dumps(d, sort_keys=True)
        j2 = json.dumps(d, sort_keys=True)
        assert j1 == j2

    def test_sweep_point_serialization(self) -> None:
        point = SweepPoint(
            cosine=0.70,
            margin=0.08,
            auto_correct=10,
            auto_wrong_person=2,
            review_count=5,
            abstain_count=3,
            n_scoreable=20,
            far=2 / 12,
            far_ci_upper=0.35,
        )
        d = sweep_point_to_dict(point)
        assert d["cosine"] == 0.70
        assert d["auto_correct"] == 10
        assert isinstance(d["far"], float)

    def test_gates_round_trip(self) -> None:
        gates = MatchingGates(
            grounded_min_cosine=0.75,
            grounded_min_margin=0.10,
        )
        d = gates_to_dict(gates)
        restored = gates_from_dict(d)
        assert restored.grounded_min_cosine == 0.75
        assert restored.grounded_min_margin == 0.10
        assert restored.min_cosine == gates.min_cosine

    def test_gates_json_valid(self) -> None:
        gates = MatchingGates()
        d = gates_to_dict(gates)
        text = json.dumps(d, allow_nan=False)
        parsed = json.loads(text)
        assert parsed == d


# ---------------------------------------------------------------------------
# Wilson CI integration (via sweep)
# ---------------------------------------------------------------------------
class TestWilsonCIIntegration:
    def test_wilson_ci_bounds_far(self) -> None:
        """The Wilson CI upper bound should be >= the point FAR.
        At tight gates, FAR=0/5 -> CI upper is informative but < 1.0."""
        trials = _make_fixture()
        # At (0.80, 0.10), impostors (cos=0.75) fail: 0/5 FAR.
        points = sweep(
            trials,
            cosine_grid=[0.80],
            margin_grid=[0.10],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.auto_correct == 5  # strong genuines pass
        assert p.auto_wrong_person == 0
        assert p.far == 0.0
        assert p.far_ci_upper >= p.far
        assert p.far_ci_upper < 1.0  # 0/5 -> informative upper bound

    def test_wilson_ci_all_correct_no_impostors(self) -> None:
        """No impostor trials at all: FAR=0, CI=1.0 (no denominator)."""
        genuines = [
            _trial(
                run_id=f"r{i}",
                label=f"L{i}",
                cosine=0.85,
                margin=0.12,
                kind=TrialKind.GENUINE,
                cluster=f"c{i}",
            )
            for i in range(10)
        ]
        points = sweep(
            genuines,
            cosine_grid=[0.70],
            margin_grid=[0.08],
            base_gates=MatchingGates(),
        )
        p = points[0]
        assert p.auto_correct == 10
        assert p.auto_wrong_person == 0
        assert p.far == 0.0
        assert p.far_ci_upper == 1.0  # no impostor trials -> no information

    def test_wilson_ci_with_impostors(self) -> None:
        """With impostors present, FAR and CI reflect the impostor population."""
        trials = _make_fixture()  # 5 impostors, 10 genuines
        points = sweep(
            trials,
            cosine_grid=[0.80],
            margin_grid=[0.10],
            base_gates=MatchingGates(),
        )
        p = points[0]
        # At tight gates, no impostors auto-attributed (cos=0.75 < 0.80).
        assert p.auto_wrong_person == 0
        assert p.far == 0.0
        # Wilson CI: 0/5 -> upper bound > 0 (informative, not 1.0).
        assert 0.0 < p.far_ci_upper < 1.0
