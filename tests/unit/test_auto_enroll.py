"""Unit tests for the auto-enrollment feature (#275).

Pure-function tests for resolver dispatch, attribution propagation, aggregate
tracking, transcript rendering, and harness export handling of AUTO_ENROLL.
The DB-coupled ``auto_enroll_run()`` flow is exercised in integration tests.
"""

import uuid
from types import SimpleNamespace

import pytest

from voxint.adjudication.attribution import AttributionScope, winning_attribution
from voxint.adjudication.resolver import Resolution
from voxint.adjudication.transcript import display_name
from voxint.db.models import Decision
from voxint.harness_export.export import _truth_from_state

# -- Decision enum -----------------------------------------------------------


def test_auto_enroll_in_decision_enum() -> None:
    assert Decision.AUTO_ENROLL.value == "auto_enroll"
    assert Decision("auto_enroll") is Decision.AUTO_ENROLL


def test_auto_enroll_in_resolution_enum() -> None:
    assert Resolution.AUTO_ENROLL.value == "auto_enroll"
    assert Resolution("auto_enroll") is Resolution.AUTO_ENROLL


# -- Resolver dispatch -------------------------------------------------------


def test_resolver_exhaustive_dispatch_unknown_decision() -> None:
    """An unrecognized decision value must raise, not silently resolve."""
    # We test this by checking the enum is exhaustive. Adding a new value
    # without a branch would hit the AssertionError in label_states.
    known = {
        Decision.ASSIGN.value,
        Decision.EXCLUDE.value,
        Decision.UNKNOWN.value,
        Decision.INHERIT.value,
        Decision.AUTO_ENROLL.value,
    }
    assert {d.value for d in Decision} == known


# -- Attribution propagation -------------------------------------------------


def _label_state(
    resolution: Resolution, speaker_id: uuid.UUID | None, speaker_name: str | None
) -> SimpleNamespace:
    return SimpleNamespace(resolution=resolution, speaker_id=speaker_id, speaker_name=speaker_name)


def _emission(*, label_state: object = None) -> SimpleNamespace:
    return SimpleNamespace(range_override=None, seg_override=None, label_state=label_state)


def test_attribution_auto_enroll_propagates_speaker() -> None:
    sid = uuid.uuid4()
    state = _label_state(Resolution.AUTO_ENROLL, sid, "Voice 1")
    scope, res, spk_id, spk_name = winning_attribution(_emission(label_state=state))
    assert scope is AttributionScope.LABEL
    assert res is Resolution.AUTO_ENROLL
    assert spk_id == sid
    assert spk_name == "Voice 1"


def test_attribution_auto_enroll_no_name_propagates_id() -> None:
    sid = uuid.uuid4()
    state = _label_state(Resolution.AUTO_ENROLL, sid, None)
    scope, res, spk_id, spk_name = winning_attribution(_emission(label_state=state))
    assert scope is AttributionScope.LABEL
    assert res is Resolution.AUTO_ENROLL
    assert spk_id == sid
    assert spk_name is None


# -- Transcript display name -------------------------------------------------


def _seg(label: str | None) -> SimpleNamespace:
    return SimpleNamespace(diarization_label=label)


def _display_state(resolution: Resolution, speaker_name: str | None) -> SimpleNamespace:
    return SimpleNamespace(resolution=resolution, speaker_name=speaker_name)


def test_display_name_auto_enroll_shows_speaker_name() -> None:
    state = _display_state(Resolution.AUTO_ENROLL, "Voice 3")
    assert display_name(state, _seg("SPEAKER_02")) == "Voice 3"


def test_display_name_auto_enroll_no_name_falls_back_to_label() -> None:
    state = _display_state(Resolution.AUTO_ENROLL, None)
    assert display_name(state, _seg("SPEAKER_02")) == "SPEAKER_02"


# -- Harness export (eval truth) ---------------------------------------------


def test_harness_export_auto_enroll_is_none() -> None:
    """AUTO_ENROLL is not human truth and must not contaminate eval scoring."""
    state = SimpleNamespace(
        effective_decision=SimpleNamespace(decision=Decision.AUTO_ENROLL.value),
        speaker_name="Voice 1",
    )
    assert _truth_from_state(state) is None


def test_harness_export_assign_still_works() -> None:
    state = SimpleNamespace(
        effective_decision=SimpleNamespace(decision=Decision.ASSIGN.value),
        speaker_name="Alice",
    )
    assert _truth_from_state(state) == "Alice"


def test_harness_export_no_decision_is_none() -> None:
    state = SimpleNamespace(effective_decision=None, speaker_name=None)
    assert _truth_from_state(state) is None


# -- Aggregate tracking ------------------------------------------------------


def test_aggregate_auto_enrolled_flag_on_appearance() -> None:
    from voxint.speakers.aggregate import SpeakerAppearance

    a = SpeakerAppearance(
        media_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        media_created_at=None,
        seconds=30.0,
        segments=5,
        human_assigned=False,
        auto_enrolled=True,
    )
    assert a.auto_enrolled is True
    assert a.human_assigned is False


def test_aggregate_auto_enrolled_default_false() -> None:
    from voxint.speakers.aggregate import SpeakerAppearance

    a = SpeakerAppearance(
        media_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        media_created_at=None,
        seconds=30.0,
        segments=5,
        human_assigned=False,
    )
    assert a.auto_enrolled is False


def test_speaker_aggregate_auto_enrolled_flag() -> None:
    from voxint.speakers.aggregate import SpeakerAggregate

    agg = SpeakerAggregate(
        speaker_id=uuid.uuid4(),
        files=1,
        seconds=15.0,
        segments=3,
        first_seen=None,
        last_seen=None,
        verified=False,
        appearances=(),
        auto_enrolled=True,
    )
    assert agg.auto_enrolled is True
    assert agg.verified is False


def test_speaker_aggregate_auto_enrolled_default_false() -> None:
    from voxint.speakers.aggregate import empty_aggregate

    agg = empty_aggregate(uuid.uuid4())
    assert agg.auto_enrolled is False


# -- Auto-enroll module pure helpers -----------------------------------------


def test_cosine_match_empty_roster_returns_none() -> None:
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    centroid = np.array([1.0, 0.0, 0.0])
    entries = [(np.array([1.0, 0.0, 0.0]), 5.0)]
    assert _cosine_match(centroid, {}, entries, MatchingGates()) is None


def test_cosine_match_below_threshold_returns_none() -> None:
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    centroid = np.array([1.0, 0.0, 0.0])
    roster = {uuid.uuid4(): np.array([0.0, 1.0, 0.0])}
    entries = [(np.array([1.0, 0.0, 0.0]), 5.0)]
    assert _cosine_match(centroid, roster, entries, MatchingGates()) is None


def test_cosine_match_above_threshold_returns_speaker() -> None:
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    sid = uuid.uuid4()
    centroid = np.array([1.0, 0.0, 0.0])
    roster = {sid: np.array([1.0, 0.0, 0.0])}
    entries = [(np.array([1.0, 0.0, 0.0]), 5.0)] * 5
    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result == sid


def test_cosine_match_margin_gate_rejects_close_pair() -> None:
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    v1 = np.array([1.0, 0.01, 0.0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([1.0, -0.01, 0.0])
    v2 = v2 / np.linalg.norm(v2)
    centroid = np.array([1.0, 0.0, 0.0])
    centroid = centroid / np.linalg.norm(centroid)
    roster = {uuid.uuid4(): v1, uuid.uuid4(): v2}
    entries = [(centroid, 5.0)] * 5
    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result is None


def test_cosine_match_standard_tier_links_cross_episode() -> None:
    """Cosine 0.669 with unanimous vote links at standard tier (#433)."""
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    # Build vectors with cosine ~0.669 between centroid and speaker A,
    # and a second speaker far away so margin is large.
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    # cos(theta) = 0.669 => theta ~48 deg; blend to hit that.
    centroid = 0.669 * a + np.sqrt(1 - 0.669**2) * b
    centroid = centroid / np.linalg.norm(centroid)
    assert abs(float(centroid @ a) - 0.669) < 0.001

    sid_a = uuid.uuid4()
    sid_far = uuid.uuid4()
    roster = {sid_a: a, sid_far: -a}

    entries = [(centroid, 5.0)] * 5
    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result == sid_a


def test_cosine_match_singleton_guard_requires_grounded_cosine() -> None:
    """With only one roster speaker, the cosine bar rises to grounded (0.70)."""
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    # cos = 0.669 < grounded_min_cosine (0.70) => rejected for singleton
    centroid = 0.669 * a + np.sqrt(1 - 0.669**2) * b
    centroid = centroid / np.linalg.norm(centroid)

    sid = uuid.uuid4()
    roster = {sid: a}
    entries = [(centroid, 5.0)] * 5
    assert _cosine_match(centroid, roster, entries, MatchingGates()) is None


def test_cosine_match_singleton_above_grounded_accepts() -> None:
    """Singleton roster with cosine above grounded threshold links."""
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    # cos = 0.75 > grounded_min_cosine (0.70) => accepted
    centroid = 0.75 * a + np.sqrt(1 - 0.75**2) * b
    centroid = centroid / np.linalg.norm(centroid)

    sid = uuid.uuid4()
    roster = {sid: a}
    entries = [(centroid, 5.0)] * 5
    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result == sid


def test_cosine_match_margin_in_standard_band_accepts() -> None:
    """Margin ~0.068 passes at standard tier (>= 0.05) but would fail grounded (>= 0.08)."""
    import math

    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    # Place centroid at angle 0.15 rad from speaker_a, speaker_b at -0.25 rad.
    # margin = cos(0.15) - cos(0.40) ≈ 0.989 - 0.921 ≈ 0.068
    speaker_a = np.array([1.0, 0.0])
    speaker_b = np.array([math.cos(-0.25), math.sin(-0.25)])
    centroid = np.array([math.cos(0.15), math.sin(0.15)])

    sid_a = uuid.uuid4()
    sid_b = uuid.uuid4()
    roster = {sid_a: speaker_a, sid_b: speaker_b}

    margin = float(centroid @ speaker_a) - float(centroid @ speaker_b)
    assert 0.05 <= margin < 0.08

    entries = [(centroid, 5.0)] * 5
    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result == sid_a


def test_cosine_match_vote_in_standard_band_accepts() -> None:
    """Vote agreement 0.625 passes at standard tier (>= 0.60) but would fail grounded (>= 0.67)."""
    import numpy as np

    from voxint.speakers.auto_enroll import _cosine_match
    from voxint.speakers.matching import MatchingGates

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])

    sid_a = uuid.uuid4()
    sid_b = uuid.uuid4()
    roster = {sid_a: a, sid_b: -a}

    # Centroid well above cosine and margin thresholds
    centroid = 0.80 * a + np.sqrt(1 - 0.80**2) * b
    centroid = centroid / np.linalg.norm(centroid)

    # 5 of 8 turns vote for speaker A => vote_agreement = 5/8 = 0.625
    vote_a = (centroid, 5.0)
    vote_b_vec = 0.80 * (-a) + np.sqrt(1 - 0.80**2) * c
    vote_b_vec = vote_b_vec / np.linalg.norm(vote_b_vec)
    vote_b = (vote_b_vec, 5.0)
    entries = [vote_a] * 5 + [vote_b] * 3

    result = _cosine_match(centroid, roster, entries, MatchingGates())
    assert result == sid_a


# -- Settings ----------------------------------------------------------------


def test_auto_enroll_setting_default_true() -> None:
    from voxint.config import Settings

    s = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://x:x@localhost/x",
        redis_url="redis://localhost:6379/0",
    )
    assert s.auto_enroll is True


def test_auto_enroll_setting_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from voxint.config import Settings

    monkeypatch.setenv("AUTO_ENROLL", "false")
    s = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://x:x@localhost/x",
        redis_url="redis://localhost:6379/0",
    )
    assert s.auto_enroll is False


# -- Idempotency key format -------------------------------------------------


def test_idempotency_key_format() -> None:
    run_id = uuid.uuid4()
    label = "SPEAKER_00"
    key = f"auto_enroll:{run_id}:{label}"
    assert key.startswith("auto_enroll:")
    assert str(run_id) in key
    assert label in key
