"""Regression-gate metrics assembly: paired verdict records -> one metrics dict.

Compares a *candidate* attribution run against a *baseline* on the same items
and assembles the counts a release gate would read: correct-host drops,
correct-to-wrong swaps, over-naming introduction, audited-subset regressions,
and a one-sided Wilson upper bound on the item-level regression rate.

Two counts are NOT mechanically derivable from verdicts and so are
*adjudicated inputs*, never invented here:

  * ``contamination_count`` — cross-show over-naming needs a host ->
    home-channel map (host ground truth is not auto-derivable). Passed in.
  * ``overnaming_strict_count`` — over-naming on strict-abstain control items;
    computed from the verdict records *restricted to* the caller-supplied set
    of strict-control item ids (the manifest knows which controls are strict).

Pure, DB-free, dict-in/dict-out — same pattern as
:mod:`voxint.harness.name_accuracy`.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from voxint.harness.name_accuracy import (
    FN,
    FP_OVERNAME,
    FP_WRONG,
    TN,
    TP,
    VERDICTS,
    wilson_ci,
)

# One-sided 95% normal quantile. The certification question is one-sided ("is
# the regression rate provably <= cap?"), so the upper bound uses z=1.645 — NOT
# the two-sided 1.96 that wilson_ci defaults to.
Z_ONE_SIDED_95 = 1.645

# Verdicts under which a slot's name surface is acceptable: the correct person,
# or a correct abstain.
HOST_OK_VERDICTS = frozenset({TP, TN})


@dataclass(frozen=True)
class GateMetrics:
    """The assembled gate inputs. All counts are item-level."""

    host_regression_count: int  # baseline TP -> candidate FN (host dropped)
    correct_to_wrong_swaps: int  # baseline TP -> candidate FP_WRONG
    contamination_count: int  # adjudicated input, passed through
    overnaming_strict_count: int  # candidate FP_OVERNAME on strict controls
    baseline_fp_overname: int
    candidate_fp_overname: int
    new_fp_overname_additions: int  # candidate over-names that baseline didn't
    audit_regression_count: int  # audited subset: host-ok -> not host-ok
    audit_candidate_host_ok: int  # audited subset: candidate host-ok count
    item_regression_rate_upper: float  # one-sided 95% Wilson upper bound
    paired_n: int  # audited paired-comparison item count


def _verdicts(rec: Mapping[str, Any]) -> tuple[object, object]:
    return rec.get("baseline_verdict"), rec.get("candidate_verdict")


def assemble_gate_metrics(
    *,
    scorer_per_item: Sequence[Mapping[str, Any]],
    audit_per_item: Sequence[Mapping[str, Any]] = (),
    paired_tally: Mapping[str, Any] | None = None,
    contamination_count: int = 0,
    strict_control_ids: Iterable[object] | None = None,
) -> GateMetrics:
    """Assemble :class:`GateMetrics` from paired per-item verdict records.

    ``scorer_per_item`` records carry ``item_id``, ``baseline_verdict``, and
    ``candidate_verdict`` (values from the name-accuracy verdict vocabulary).
    ``audit_per_item`` is an optional human-audited subset with a binding
    ``host_ok`` boolean for the baseline. ``paired_tally`` carries ``n_items``
    and ``item_regressed`` from a paired (e.g. blinded) comparison; ``n_items``
    of 0 yields an upper bound of 1.0 (no information — a downstream gate
    should refuse to certify rather than pass vacuously).
    """
    strict = set(strict_control_ids) if strict_control_ids is not None else None
    if contamination_count < 0:
        raise ValueError(f"contamination_count must be >= 0, got {contamination_count}")

    host_regression_count = 0
    correct_to_wrong_swaps = 0
    baseline_fp_overname = 0
    candidate_fp_overname = 0
    new_fp_overname_additions = 0
    overnaming_strict_count = 0

    for rec in scorer_per_item:
        base, cand = _verdicts(rec)
        # A typo'd verdict silently counted as "no transition" would understate
        # regressions — reject it instead.
        for label, value in (("baseline_verdict", base), ("candidate_verdict", cand)):
            if value not in VERDICTS:
                raise ValueError(f"record {rec.get('item_id')!r}: bad {label} {value!r}")
        # Correct-host -> dropped / swapped: the central regression modes.
        if base == TP and cand == FN:
            host_regression_count += 1
        elif base == TP and cand == FP_WRONG:
            correct_to_wrong_swaps += 1

        # Over-naming surface (FP_OVERNAME only ever occurs on an abstain truth).
        if base == FP_OVERNAME:
            baseline_fp_overname += 1
        if cand == FP_OVERNAME:
            candidate_fp_overname += 1
            # A candidate over-name that was NOT a baseline over-name is a
            # newly-introduced over-name — the binding "did the change start
            # over-naming?" guard.
            if base != FP_OVERNAME:
                new_fp_overname_additions += 1
            # Over-naming on a strict-abstain control (only counted when the
            # caller said which item ids are strict controls).
            if strict is not None and rec.get("item_id") in strict:
                overnaming_strict_count += 1

    audit_regression_count = 0
    audit_candidate_host_ok = 0
    for rec in audit_per_item:
        base_ok = rec.get("host_ok")
        if not isinstance(base_ok, bool):
            raise ValueError(
                f"audit record {rec.get('item_id')!r}: host_ok must be a boolean, "
                f"got {base_ok!r}"
            )
        cand = rec.get("candidate_verdict")
        if cand not in VERDICTS:
            raise ValueError(
                f"audit record {rec.get('item_id')!r}: bad candidate_verdict {cand!r}"
            )
        cand_ok = cand in HOST_OK_VERDICTS
        if base_ok and not cand_ok:
            audit_regression_count += 1
        if cand_ok:
            audit_candidate_host_ok += 1

    tally = paired_tally or {}
    n_items = int(tally.get("n_items", 0))
    item_regressed = int(tally.get("item_regressed", 0))
    if n_items < 0 or not 0 <= item_regressed <= max(n_items, 0):
        raise ValueError(
            f"paired_tally must satisfy 0 <= item_regressed <= n_items, "
            f"got {item_regressed}/{n_items}"
        )
    item_regression_rate_upper = wilson_ci(item_regressed, n_items, z=Z_ONE_SIDED_95)[1]

    return GateMetrics(
        host_regression_count=host_regression_count,
        correct_to_wrong_swaps=correct_to_wrong_swaps,
        contamination_count=int(contamination_count),
        overnaming_strict_count=overnaming_strict_count,
        baseline_fp_overname=baseline_fp_overname,
        candidate_fp_overname=candidate_fp_overname,
        new_fp_overname_additions=new_fp_overname_additions,
        audit_regression_count=audit_regression_count,
        audit_candidate_host_ok=audit_candidate_host_ok,
        item_regression_rate_upper=item_regression_rate_upper,
        paired_n=n_items,
    )
