"""Pure idempotency-key helpers for inline merge (issue #54).

The child-key derivation is the composite mutation's replay safety: one operator
nonce backs several ledger rows, so each row's key must be deterministic, unique
per label, and bound to the label SET — reusing the nonce for a different set
must collide loudly rather than half-apply under matching keys.
"""

import pytest

from voxint.adjudication.merge import MergeError, _child_key, _labels_digest, apply_merge


def test_digest_is_order_independent() -> None:
    assert _labels_digest(["S0", "S1"]) == _labels_digest(["S1", "S0"])


def test_digest_ignores_duplicates() -> None:
    assert _labels_digest(["S0", "S1", "S0"]) == _labels_digest(["S0", "S1"])


def test_digest_distinguishes_different_label_sets() -> None:
    assert _labels_digest(["S0", "S1"]) != _labels_digest(["S0", "S2"])


def test_child_key_is_deterministic_and_per_label() -> None:
    a = _child_key("nonce123", ["S0", "S1"], "S0")
    b = _child_key("nonce123", ["S1", "S0"], "S0")  # order-independent
    c = _child_key("nonce123", ["S0", "S1"], "S1")  # different label
    assert a == b
    assert a != c
    # Namespaced away from the decide/enroll routes' bare nonce.
    assert a.startswith("merge:nonce123:") and a.endswith(":S0")


def test_child_key_binds_to_the_label_set() -> None:
    # Same nonce + same label, but a different merge set => different key, so a
    # nonce reused across two different merges cannot collide into a matching row.
    same_set = _child_key("n", ["S0", "S1"], "S0")
    other_set = _child_key("n", ["S0", "S2"], "S0")
    assert same_set != other_set


def test_apply_requires_exactly_one_target() -> None:
    # The XOR guard is the first statement, before any DB access — reached with no
    # session. Both-or-neither target is a caller bug, refused loudly.
    for kwargs in ({}, {"target_speaker_id": __import__("uuid").uuid4(), "target_name": "x"}):
        with pytest.raises(MergeError):
            apply_merge(
                None,  # type: ignore[arg-type]  # guard fires before session is touched
                run_id=__import__("uuid").uuid4(),
                labels=["S0", "S1"],
                operator="op",
                nonce="nonce123",
                gates=None,  # type: ignore[arg-type]
                expected={},  # required now; the XOR guard fires before it is read
                **kwargs,
            )
