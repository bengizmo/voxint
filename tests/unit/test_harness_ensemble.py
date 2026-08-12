"""Two-voter verdict fusion + the cross-embedding-space guardrail."""

import ast
import inspect

import pytest

from voxint.harness import ensemble as en
from voxint.harness.agreement import (
    ABSTAIN,
    CONFIDENT_HOST_PRESENT,
    NO_CURATED_HOST_DETECTED,
    LabelResult,
)


def _res(
    verdict: str, host_slot: str | None = None, contradiction: bool = False
) -> LabelResult:
    return LabelResult(verdict=verdict, host_slot=host_slot, contradiction=contradiction)


# --------------------------------------------------------------------------- #
# combine_curated
# --------------------------------------------------------------------------- #
def test_contradiction_from_either_voter_flags_review() -> None:
    ok = _res(CONFIDENT_HOST_PRESENT, "S0")
    bad = _res(ABSTAIN, contradiction=True)
    for a, b in ((ok, bad), (bad, ok)):
        d = en.combine_curated(a, b)
        assert d.verdict == en.FLAG_REVIEW and d.agreement == "contradiction"


def test_both_confident_same_slot_is_silver() -> None:
    d = en.combine_curated(_res(CONFIDENT_HOST_PRESENT, "S0"), _res(CONFIDENT_HOST_PRESENT, "S0"))
    assert d.verdict == en.SILVER_HOST_PRESENT and d.agreement == "agree_present"


def test_both_confident_different_slots_flags_review() -> None:
    d = en.combine_curated(_res(CONFIDENT_HOST_PRESENT, "S0"), _res(CONFIDENT_HOST_PRESENT, "S1"))
    assert d.verdict == en.FLAG_REVIEW and d.agreement == "slot_disagree"


def test_single_voter_confident_names_the_voter() -> None:
    d = en.combine_curated(
        _res(CONFIDENT_HOST_PRESENT, "S0"),
        _res(ABSTAIN),
        voter_a_name="space-a",
        voter_b_name="space-b",
    )
    assert d.verdict == en.FLAG_REVIEW and d.reason == "single_voter_confident:space-a"
    d = en.combine_curated(
        _res(ABSTAIN),
        _res(CONFIDENT_HOST_PRESENT, "S0"),
        voter_a_name="space-a",
        voter_b_name="space-b",
    )
    assert d.reason == "single_voter_confident:space-b"


def test_agree_abstain() -> None:
    d = en.combine_curated(_res(ABSTAIN), _res(ABSTAIN))
    assert d.verdict == en.ENSEMBLE_ABSTAIN and d.agreement == "agree_abstain"


# --------------------------------------------------------------------------- #
# combine_neg_control
# --------------------------------------------------------------------------- #
def test_neg_contradiction_is_false_accept_suspect() -> None:
    d = en.combine_neg_control(_res(NO_CURATED_HOST_DETECTED), _res(ABSTAIN, contradiction=True))
    assert d.verdict == en.FLAG_REVIEW and d.agreement == "false_accept_suspect"


def test_neg_both_absent_is_silver() -> None:
    d = en.combine_neg_control(_res(NO_CURATED_HOST_DETECTED), _res(NO_CURATED_HOST_DETECTED))
    assert d.verdict == en.SILVER_NO_HOST


def test_neg_one_abstain_abstains() -> None:
    d = en.combine_neg_control(_res(NO_CURATED_HOST_DETECTED), _res(ABSTAIN))
    assert d.verdict == en.ENSEMBLE_ABSTAIN and d.agreement == "one_or_both_abstain"


# --------------------------------------------------------------------------- #
# Cross-embedding-space guardrail: the ensemble layer cannot see vectors.
# --------------------------------------------------------------------------- #
def test_ensemble_module_never_imports_numpy_or_vectors() -> None:
    tree = ast.parse(inspect.getsource(en))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("numpy" in mod or "vectors" in mod for mod in imported), imported


@pytest.mark.parametrize("fn", [en.combine_curated, en.combine_neg_control])
def test_ensemble_signatures_accept_only_label_results(fn: object) -> None:
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional and all(p.annotation is LabelResult for p in positional)
