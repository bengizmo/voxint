"""Versioned Platt calibration policies for synthdetect scores.

The service returns raw logits; this module applies calibrated risk via
``score = sigmoid(A * logit + B)`` where A and B are fit on a known corpus.
Each policy records its provenance so a model update without recalibration
is detectable (policy records which inference_space it was fit on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPolicy:
    platt_a: float
    platt_b: float
    fit_inference_space: str
    brier: float
    fit_date: str


CALIBRATION_POLICIES: dict[str, CalibrationPolicy] = {
    "platt-m2-s0e11-dev-v1": CalibrationPolicy(
        platt_a=4.28,
        platt_b=-11.42,
        fit_inference_space="w2v2-aasist-df-m2-s0e11",
        brier=0.0066,
        fit_date="2026-08-27",
    ),
}

DEFAULT_POLICY_ID = "platt-m2-s0e11-dev-v1"
DEFAULT_INFERENCE_SPACE = "w2v2-aasist-df-m2-s0e11"


def apply_calibration(raw_logit: float, policy_id: str) -> float:
    """Apply Platt scaling: sigmoid(A * logit + B) -> calibrated risk in [0, 1]."""
    policy = CALIBRATION_POLICIES.get(policy_id)
    if policy is None:
        raise ValueError(f"unknown calibration policy: {policy_id!r}")
    return 1.0 / (1.0 + math.exp(-(policy.platt_a * raw_logit + policy.platt_b)))
