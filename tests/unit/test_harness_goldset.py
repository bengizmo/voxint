"""Goldset stratification determinism + provenance-gated auto-labeling."""

from voxint.harness.goldset_strata import GoldLabel, GoldRow, ground_truth_for, select_strata
from voxint.harness.name_accuracy import ABSTAIN


def test_select_strata_deterministic_across_input_order() -> None:
    candidates = [f"item-{i}" for i in range(20)]
    plan = [("s1", candidates, 5), ("s2", candidates, 5)]
    plan_shuffled = [("s1", list(reversed(candidates)), 5), ("s2", candidates, 5)]
    a = select_strata(plan, total_target=10)
    b = select_strata(plan_shuffled, total_target=10)
    assert a == b


def test_select_strata_priority_and_no_double_pick() -> None:
    candidates = [f"item-{i}" for i in range(10)]
    assign, counts = select_strata(
        [("first", candidates, 6), ("second", candidates, 6)], total_target=20
    )
    assert counts == {"first": 6, "second": 4}
    assert len(assign) == 10  # every id assigned exactly once


def test_select_strata_global_cap() -> None:
    candidates = [f"item-{i}" for i in range(10)]
    assign, counts = select_strata(
        [("a", candidates[:5], 5), ("b", candidates[5:], 5)], total_target=7
    )
    assert counts == {"a": 5, "b": 2}
    assert len(assign) == 7
    # A stratum after the cap is exhausted records zero.
    _, counts = select_strata([("a", candidates, 10), ("late", candidates, 5)], total_target=10)
    assert counts["late"] == 0


CHANNEL_HOST: dict[str, dict[str, str | None]] = {
    "acme-cast": {"host": "Dana Fox", "host_id": "spk-dana"},
    "guest-only-show": {"host": None, "host_id": None},
}


def _row(
    channel: str | None, groundable: bool = False, control: bool = False
) -> GoldRow:
    return GoldRow(
        id="item-1",
        channel=channel,
        source_type="podcast",
        host_groundable=groundable,
        is_no_host_control=control,
    )


def test_groundable_curated_host_auto_labels() -> None:
    label = ground_truth_for(_row("acme-cast", groundable=True), CHANNEL_HOST)
    assert label == GoldLabel("Dana Fox", "spk-dana", "auto", "channel_fact_groundable")


def test_ungroundable_curated_host_goes_to_human_queue() -> None:
    label = ground_truth_for(_row("acme-cast", groundable=False), CHANNEL_HOST)
    assert label.truth is None and label.label_source == "human_queue"
    assert label.reason == "host_not_groundable"
    assert label.host_id == "spk-dana"


def test_host_without_identity_anchor_goes_to_human_queue() -> None:
    """A host name with no verified host_id must never auto-label a positive."""
    table: dict[str, dict[str, str | None]] = {
        "acme-cast": {"host": "Dana Fox", "host_id": None}
    }
    label = ground_truth_for(_row("acme-cast", groundable=True), table)
    assert label.truth is None and label.label_source == "human_queue"
    assert label.reason == "host_unanchored"


def test_curated_no_host_channel_auto_abstains() -> None:
    label = ground_truth_for(_row("guest-only-show"), CHANNEL_HOST)
    assert label.truth == ABSTAIN and label.reason == "curated_no_host_channel"


def test_no_host_control_auto_abstains() -> None:
    label = ground_truth_for(_row("newsy-callsign", control=True), CHANNEL_HOST)
    assert label.truth == ABSTAIN and label.reason == "no_host_control"


def test_uncurated_channel_goes_to_human_queue() -> None:
    label = ground_truth_for(_row("mystery-channel"), CHANNEL_HOST)
    assert label.label_source == "human_queue" and label.reason == "uncurated_channel"
    label = ground_truth_for(_row(None), CHANNEL_HOST)
    assert label.reason == "uncurated_channel"
