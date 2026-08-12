"""Agreement labeler core: thresholds, slot scoring, gate order, verdicts."""

import pytest

from voxint.harness import agreement as ag
from voxint.harness.vectors import SpaceMismatchError, TaggedVector

SPACE = "model-a"


def _vec(*values: float, space: str = SPACE) -> TaggedVector:
    return TaggedVector(space=space, values=values)


def _thresholds(**overrides: float | int) -> ag.Thresholds:
    base: dict[str, float | int] = {
        "tau": 0.6,
        "margin": 0.1,
        "min_duration": 30.0,
        "min_segments": 4,
        "low_band": 0.3,
        "neg_min_total_duration": 300.0,
        "min_enrollment_items": 3,
    }
    base.update(overrides)
    return ag.Thresholds(
        tau=float(base["tau"]),
        margin=float(base["margin"]),
        min_duration=float(base["min_duration"]),
        min_segments=int(base["min_segments"]),
        low_band=float(base["low_band"]),
        neg_min_total_duration=float(base["neg_min_total_duration"]),
        min_enrollment_items=int(base["min_enrollment_items"]),
    )


# Host voiceprint along the x axis; cosine to a slot is easy to read off.
HOST = _vec(1.0, 0.0)


def _slot(x: float, y: float, duration: float = 100.0, segments: int = 10) -> ag.Slot:
    return ag.Slot(vector=_vec(x, y), duration=duration, segments=segments)


# --------------------------------------------------------------------------- #
# Thresholds validation
# --------------------------------------------------------------------------- #
def test_thresholds_low_band_must_not_exceed_tau() -> None:
    with pytest.raises(ValueError, match="low_band"):
        _thresholds(low_band=0.7, tau=0.6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tau": float("nan")},
        {"tau": 1.5},
        {"margin": -0.1},
        {"min_duration": -1.0},
        {"min_duration": float("nan")},
        {"neg_min_total_duration": float("inf")},
        {"min_segments": -1},
        {"min_enrollment_items": -1},
    ],
)
def test_thresholds_rejects_bad_values(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        _thresholds(**overrides)


# --------------------------------------------------------------------------- #
# score_slots
# --------------------------------------------------------------------------- #
def test_score_slots_sorted_desc_with_name_tiebreak() -> None:
    slots = {
        "S2": _slot(0.5, 0.5),
        "S1": _slot(0.5, 0.5),
        "S0": _slot(1.0, 0.0),
    }
    scored = ag.score_slots(slots, HOST)
    assert [s.slot for s in scored] == ["S0", "S1", "S2"]
    assert scored[0].cosine == pytest.approx(1.0)


def test_score_slots_cross_space_raises() -> None:
    slots = {"S0": ag.Slot(vector=_vec(1.0, 0.0, space="model-b"), duration=1.0, segments=1)}
    with pytest.raises(SpaceMismatchError):
        ag.score_slots(slots, HOST)


# --------------------------------------------------------------------------- #
# label_positive gate order
# --------------------------------------------------------------------------- #
def test_positive_weak_enrollment_short_circuits() -> None:
    res = ag.label_positive({}, HOST, _thresholds(), enrollment_ok=False)
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_WEAK_ENROLLMENT


def test_positive_leakage_reason_passthrough() -> None:
    res = ag.label_positive(
        {},
        HOST,
        _thresholds(),
        enrollment_ok=False,
        enrollment_reason=ag.REASON_SESSION_LEAKAGE_RISK,
    )
    assert res.reason == ag.REASON_SESSION_LEAKAGE_RISK


def test_positive_unknown_enrollment_reason_raises() -> None:
    with pytest.raises(ValueError, match="enrollment_reason"):
        ag.label_positive({}, HOST, _thresholds(), enrollment_ok=False, enrollment_reason="bogus")


def test_positive_no_slots() -> None:
    res = ag.label_positive({}, HOST, _thresholds(), enrollment_ok=True)
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_NO_SLOT_EMBEDDINGS


def test_positive_low_cosine_without_contradiction_when_speech_inadequate() -> None:
    res = ag.label_positive(
        {"S0": _slot(0.1, 1.0)}, HOST, _thresholds(), enrollment_ok=True, total_speech=10.0
    )
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_LOW_COSINE_BAND
    assert not res.contradiction


def test_positive_confident_absence_flags_contradiction() -> None:
    res = ag.label_positive(
        {"S0": _slot(0.1, 1.0)}, HOST, _thresholds(), enrollment_ok=True, total_speech=1000.0
    )
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_LOW_COSINE_BAND
    assert res.contradiction


def test_positive_mid_band_cosine_is_not_a_contradiction() -> None:
    # cos = 0.5: below tau (0.6) but above low_band (0.3).
    res = ag.label_positive(
        {"S0": _slot(1.0, 1.732)}, HOST, _thresholds(), enrollment_ok=True, total_speech=1000.0
    )
    assert res.reason == ag.REASON_LOW_COSINE_BAND and not res.contradiction


def test_positive_short_duration_and_sparse_segments() -> None:
    res = ag.label_positive(
        {"S0": _slot(1.0, 0.0, duration=5.0)}, HOST, _thresholds(), enrollment_ok=True
    )
    assert res.reason == ag.REASON_SHORT_DURATION
    res = ag.label_positive(
        {"S0": _slot(1.0, 0.0, segments=1)}, HOST, _thresholds(), enrollment_ok=True
    )
    assert res.reason == ag.REASON_SHORT_DURATION


def test_positive_near_tie_abstains() -> None:
    slots = {"S0": _slot(1.0, 0.05), "S1": _slot(1.0, 0.1)}
    res = ag.label_positive(slots, HOST, _thresholds(), enrollment_ok=True)
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_NEAR_TIE
    assert res.runner_up_cosine is not None and res.margin is not None


def test_positive_single_slot_auto_passes_margin() -> None:
    res = ag.label_positive({"S0": _slot(1.0, 0.0)}, HOST, _thresholds(), enrollment_ok=True)
    assert res.verdict == ag.CONFIDENT_HOST_PRESENT and res.reason is None
    assert res.host_slot == "S0" and res.top_cosine == pytest.approx(1.0)
    assert res.runner_up_cosine is None and res.margin is None


def test_positive_confident_with_clear_margin() -> None:
    slots = {"S0": _slot(1.0, 0.0), "S1": _slot(0.2, 1.0)}
    res = ag.label_positive(slots, HOST, _thresholds(), enrollment_ok=True)
    assert res.verdict == ag.CONFIDENT_HOST_PRESENT
    assert res.host_slot == "S0"


def test_positive_diagnostics_passthrough() -> None:
    res = ag.label_positive(
        {"S0": _slot(1.0, 0.0)},
        HOST,
        _thresholds(),
        enrollment_ok=True,
        diagnostics={"note": "x"},
    )
    assert res.diagnostics == {"note": "x"}


# --------------------------------------------------------------------------- #
# passes_present_gates
# --------------------------------------------------------------------------- #
def test_passes_present_gates_branches() -> None:
    th = _thresholds()
    assert not ag.passes_present_gates([], th)
    ok = ag.SlotScore(slot="S0", cosine=0.9, duration=100.0, segments=10)
    low = ag.SlotScore(slot="S0", cosine=0.5, duration=100.0, segments=10)
    short = ag.SlotScore(slot="S0", cosine=0.9, duration=1.0, segments=10)
    near = ag.SlotScore(slot="S1", cosine=0.85, duration=100.0, segments=10)
    far = ag.SlotScore(slot="S1", cosine=0.3, duration=100.0, segments=10)
    assert ag.passes_present_gates([ok], th)
    assert not ag.passes_present_gates([low], th)
    assert not ag.passes_present_gates([short], th)
    assert not ag.passes_present_gates([ok, near], th)
    assert ag.passes_present_gates([ok, far], th)


# --------------------------------------------------------------------------- #
# label_negative_control
# --------------------------------------------------------------------------- #
def test_negative_no_slots_or_no_voiceprints() -> None:
    th = _thresholds()
    res = ag.label_negative_control({}, {"h": HOST}, th)
    assert res.reason == ag.REASON_NO_SLOT_EMBEDDINGS
    res = ag.label_negative_control({"S0": _slot(1.0, 0.0)}, {}, th)
    assert res.reason == ag.REASON_NO_SLOT_EMBEDDINGS


def test_negative_curated_host_present_is_contradiction() -> None:
    res = ag.label_negative_control(
        {"S0": _slot(1.0, 0.0)}, {"h": HOST}, _thresholds(), total_speech=1000.0
    )
    assert res.verdict == ag.ABSTAIN and res.contradiction
    assert res.reason == ag.REASON_LOW_COSINE_BAND


def test_negative_silver_absence_requires_adequate_speech() -> None:
    slots = {"S0": _slot(0.1, 1.0)}
    res = ag.label_negative_control(slots, {"h": HOST}, _thresholds(), total_speech=1000.0)
    assert res.verdict == ag.NO_CURATED_HOST_DETECTED and res.reason is None
    res = ag.label_negative_control(slots, {"h": HOST}, _thresholds(), total_speech=10.0)
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_SHORT_DURATION


def test_negative_ambiguous_band_abstains() -> None:
    # cos = 0.5: in [low_band, tau) — ambiguously similar.
    res = ag.label_negative_control(
        {"S0": _slot(1.0, 1.732)}, {"h": HOST}, _thresholds(), total_speech=1000.0
    )
    assert res.verdict == ag.ABSTAIN and res.reason == ag.REASON_LOW_COSINE_BAND
    assert not res.contradiction


def test_negative_contradiction_names_the_triggering_host() -> None:
    """best_candidate_host is highest-cosine; the contradiction trigger can be
    a DIFFERENT host — the diagnostics must name the actual trigger."""
    th = _thresholds()
    # "trigger" clears every present-gate on its slot; "loud" has a higher top
    # cosine but a near-tie between two slots, so it never sets contradiction.
    slots = {
        "S0": _slot(1.0, 0.0),  # trigger's confident slot
        "S1": _slot(0.0, 1.0, duration=100.0, segments=10),
        "S2": _slot(0.05, 1.0, duration=100.0, segments=10),
    }
    vps = {"trigger": HOST, "loud": _vec(0.0, 1.0)}
    res = ag.label_negative_control(slots, vps, th, total_speech=1000.0)
    assert res.contradiction
    assert res.diagnostics["contradiction_hosts"] == ["trigger"]


def test_negative_records_best_candidate_host() -> None:
    # Slot along +x: cosine 0.5 to "near" (ambiguous band), -0.5 to "far".
    vps = {"near": HOST, "far": _vec(-1.0, 0.0)}
    res = ag.label_negative_control(
        {"S0": _slot(1.0, 1.732)}, vps, _thresholds(), total_speech=1000.0
    )
    assert res.diagnostics["best_candidate_host"] == "near"


# --------------------------------------------------------------------------- #
# far_frr_at
# --------------------------------------------------------------------------- #
def test_far_frr_counts() -> None:
    genuine = [0.9, 0.7, 0.4]
    impostor = [0.65, 0.3, 0.2]
    far, frr, n_g, n_i = ag.far_frr_at(genuine, impostor, tau=0.6)
    assert (far, frr, n_g, n_i) == (1, 1, 3, 3)
