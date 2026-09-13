"""Pre-registered DEV selection and CONFIRM certification tests."""

import json
from pathlib import Path
from typing import Any

import pytest
from tools import calibrate_policy

from voxint.harness.calibration import (
    SelectionRule,
    Trial,
    TrialDetail,
    TrialKind,
    certify,
    filter_split,
    load_attribution_trials,
    select_candidate,
)
from voxint.harness.name_accuracy import wilson_upper_one_sided
from voxint.speakers.matching import MatchingGates


def _trial(
    *,
    cluster: str,
    kind: TrialKind,
    similarity: float,
    margin: float,
    split: str,
) -> Trial:
    return Trial(
        run_id=f"run-{cluster}",
        label="SPEAKER_00",
        similarity=similarity,
        margin=margin,
        vote_agreement=0.9,
        eligible_turns=5,
        eligible_seconds=15.0,
        roster_size=4,
        top_speaker_id="roster-speaker",
        kind=kind,
        truth_anchoring="corpus_gold",
        cluster_id=cluster,
        detail=(
            TrialDetail.GENUINE
            if kind == TrialKind.GENUINE
            else TrialDetail.IMPOSTOR_OPEN
        ),
        split=split,
    )


def _safe_impostors(count: int, split: str) -> list[Trial]:
    return [
        _trial(
            cluster=f"impostor-{index}",
            kind=TrialKind.IMPOSTOR,
            similarity=0.60,
            margin=0.20,
            split=split,
        )
        for index in range(count)
    ]


def test_select_candidate_uses_pre_registered_lexicographic_order() -> None:
    trials = _safe_impostors(60, "dev")
    trials.extend(
        _trial(
            cluster=f"genuine-{index}",
            kind=TrialKind.GENUINE,
            similarity=0.75,
            margin=0.09,
            split="dev",
        )
        for index in range(3)
    )

    result = select_candidate(trials, MatchingGates(), SelectionRule())

    assert result.outcome == "SELECTED"
    assert result.reason is None
    assert result.chosen_gates is not None
    assert result.chosen_gates.grounded_min_cosine == 0.74
    assert result.chosen_gates.grounded_min_margin == 0.08
    assert len(result.points) == 32
    assert result.n_genuine_clusters == 3
    assert result.n_impostor_clusters == 60


def test_select_candidate_returns_no_decision_when_nothing_is_feasible() -> None:
    trials = [
        _trial(
            cluster=f"impostor-{index}",
            kind=TrialKind.IMPOSTOR,
            similarity=0.95,
            margin=0.30,
            split="dev",
        )
        for index in range(60)
    ]

    result = select_candidate(trials, MatchingGates(), SelectionRule())

    assert result.outcome == "NO_DECISION"
    assert result.reason == "no_feasible_candidate"
    assert result.chosen_gates is None
    assert not any(point.feasible for point in result.points)


def test_certify_rejects_49_impostor_clusters_even_with_zero_wrong() -> None:
    result = certify(
        _safe_impostors(49, "confirm"),
        MatchingGates(),
        MatchingGates(grounded_min_cosine=0.74, grounded_min_margin=0.08),
        SelectionRule(),
    )

    assert result.outcome == "NO_DECISION"
    assert "insufficient_impostor_clusters" in result.reasons
    assert result.post.auto_wrong == 0


def test_certify_accepts_60_clusters_with_zero_wrong() -> None:
    result = certify(
        _safe_impostors(60, "confirm"),
        MatchingGates(),
        MatchingGates(grounded_min_cosine=0.74, grounded_min_margin=0.08),
        SelectionRule(),
    )

    assert result.outcome == "CERTIFIED"
    assert result.reasons == ()
    assert result.post.far_upper_one_sided == pytest.approx(0.0431, abs=0.0001)
    assert result.post.far_upper_one_sided == pytest.approx(
        wilson_upper_one_sided(0, 60)
    )


def test_certify_rejects_one_auto_wrong() -> None:
    trials = _safe_impostors(59, "confirm")
    trials.append(
        _trial(
            cluster="impostor-wrong",
            kind=TrialKind.IMPOSTOR,
            similarity=0.95,
            margin=0.30,
            split="confirm",
        )
    )

    result = certify(
        trials,
        MatchingGates(),
        MatchingGates(grounded_min_cosine=0.74, grounded_min_margin=0.08),
        SelectionRule(),
    )

    assert result.outcome == "NO_DECISION"
    assert "auto_wrong_observed" in result.reasons
    assert result.post.auto_wrong == 1


def _align_trial(split: str) -> dict[str, Any]:
    return {
        "run_id": f"run-{split}",
        "label": "SPEAKER_00",
        "similarity": 0.8,
        "margin": 0.1,
        "vote_agreement": 0.8,
        "eligible_turns": 4,
        "eligible_seconds": 12.0,
        "roster_size": 4,
        "top_speaker_id": "speaker-a",
        "kind": "genuine",
        "detail": "genuine",
        "truth_anchoring": "corpus_gold",
        "cluster_id": "gold-a",
        "meeting_split": split,
        "meeting_role": "test_genuine",
    }


def test_load_and_filter_attribution_trials_by_split() -> None:
    trials = load_attribution_trials(
        {
            "kind": "attribution_trials",
            "trials": [_align_trial("dev"), _align_trial("confirm")],
        }
    )

    assert [trial.split for trial in trials] == ["dev", "confirm"]
    assert [trial.run_id for trial in filter_split(trials, "confirm")] == [
        "run-confirm"
    ]


def test_load_attribution_trials_requires_meeting_split() -> None:
    raw = _align_trial("dev")
    del raw["meeting_split"]

    with pytest.raises(ValueError, match="missing required meeting_split"):
        load_attribution_trials({"kind": "attribution_trials", "trials": [raw]})


def test_select_and_certify_cli_accept_align_json(tmp_path: Path) -> None:
    raw_trials = []
    for split in ("dev", "confirm"):
        for index in range(60):
            raw = _align_trial(split)
            raw.update(
                {
                    "run_id": f"run-{split}-{index}",
                    "similarity": 0.60,
                    "margin": 0.20,
                    "kind": "impostor",
                    "detail": "impostor_open",
                    "cluster_id": f"impostor-{split}-{index}",
                    "meeting_role": "test_open",
                }
            )
            raw_trials.append(raw)
    trials_path = tmp_path / "trials.json"
    selection_path = tmp_path / "selection.json"
    certification_path = tmp_path / "certification.json"
    trials_path.write_text(
        json.dumps({"kind": "attribution_trials", "trials": raw_trials}),
        encoding="utf-8",
    )

    assert calibrate_policy.main(
        ["select", "--trials", str(trials_path), "--out", str(selection_path)]
    ) == 0
    assert calibrate_policy.main(
        [
            "certify",
            "--trials",
            str(trials_path),
            "--candidate",
            str(selection_path),
            "--out",
            str(certification_path),
        ]
    ) == 0

    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["outcome"] == "CERTIFIED"
    assert certification["post"]["n_impostor_clusters"] == 60
