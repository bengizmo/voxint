"""Acoustic agreement labeler core (pure, DB-free) — one embedding voter.

Given an item's per-diarized-slot centroid embeddings and a *held-out*
reference voiceprint for a curated host, emit a conservative verdict +
evidence. Outputs are **SILVER** (acoustic evidence), never gold truth:

  * held-out enrollment only — never a voiceprint built from the item being
    labeled (the caller attests via ``enrollment_ok`` / leakage provenance);
  * abstain on near-ties, on cosine below the positive threshold, and on
    short/sparse slots;
  * a *confident absence* (``NO_CURATED_HOST_DETECTED``) is asserted only on
    negative-control channels with adequate speech coverage;
  * on a curated channel, a confidently-absent host is an ``ABSTAIN`` flagged
    as a ``contradiction`` (a candidate channel-fact error for human review).

All vectors are :class:`~voxint.harness.vectors.TaggedVector`; scoring a slot
against a voiceprint from a different embedding space raises. Cross-*voter*
agreement never happens here — verdicts combine in
:mod:`voxint.harness.ensemble`, which cannot see vectors at all.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from voxint.harness.vectors import TaggedVector, cosine

# Verdict vocabulary.
CONFIDENT_HOST_PRESENT = "CONFIDENT_HOST_PRESENT"  # silver positive
NO_CURATED_HOST_DETECTED = "NO_CURATED_HOST_DETECTED"  # silver negative
ABSTAIN = "ABSTAIN"

# Abstain reasons (the only values ``LabelResult.reason`` takes for an ABSTAIN).
REASON_SHORT_DURATION = "short_duration"
REASON_LOW_COSINE_BAND = "low_cosine_band"
REASON_NEAR_TIE = "near_tie"
REASON_NO_SLOT_EMBEDDINGS = "no_slot_embeddings"
REASON_WEAK_ENROLLMENT = "weak_enrollment"
REASON_SESSION_LEAKAGE_RISK = "session_leakage_risk"

ABSTAIN_REASONS = frozenset(
    {
        REASON_SHORT_DURATION,
        REASON_LOW_COSINE_BAND,
        REASON_NEAR_TIE,
        REASON_NO_SLOT_EMBEDDINGS,
        REASON_WEAK_ENROLLMENT,
        REASON_SESSION_LEAKAGE_RISK,
    }
)


@dataclass(frozen=True)
class Thresholds:
    """Frozen decision thresholds (chosen by impostor-trial calibration).

    Attributes:
        tau: minimum top-slot cosine for ``CONFIDENT_HOST_PRESENT``.
        margin: minimum top1 - top2 cosine gap (near-tie guard; >=2 slots only).
        min_duration: minimum net host-slot duration (seconds).
        min_segments: minimum host-slot segment count.
        low_band: top-slot cosine strictly below this, on a curated channel with
            adequate speech, is a *confident absence* (contradiction flag). On a
            negative-control channel, best-host cosine below this (with adequate
            speech) yields ``NO_CURATED_HOST_DETECTED``. ``low_band <= tau``.
        neg_min_total_duration: minimum item total speech (seconds) for any
            confident-absence call.
        min_enrollment_items: minimum held-out enrollment items for a usable
            voiceprint (enforced by the caller; recorded here for provenance).
    """

    tau: float
    margin: float
    min_duration: float
    min_segments: int
    low_band: float
    neg_min_total_duration: float
    min_enrollment_items: int

    def __post_init__(self) -> None:
        for name in ("tau", "margin", "low_band", "min_duration", "neg_min_total_duration"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v!r}")
        if not -1.0 <= self.tau <= 1.0 or not -1.0 <= self.low_band <= 1.0:
            raise ValueError("tau and low_band must be within [-1, 1]")
        if self.low_band > self.tau:
            raise ValueError(f"low_band ({self.low_band}) must be <= tau ({self.tau})")
        if self.margin < 0.0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")
        if self.min_duration < 0.0 or self.neg_min_total_duration < 0.0:
            raise ValueError("duration floors must be >= 0")
        if self.min_segments < 0 or self.min_enrollment_items < 0:
            raise ValueError("count floors must be >= 0")


@dataclass(frozen=True)
class Slot:
    """One diarized slot's centroid embedding + coverage evidence."""

    vector: TaggedVector
    duration: float
    segments: int


@dataclass(frozen=True)
class SlotScore:
    """One diarized slot scored against a voiceprint."""

    slot: str
    cosine: float
    duration: float
    segments: int


@dataclass(frozen=True)
class LabelResult:
    """A per-item acoustic-agreement verdict + the evidence behind it."""

    verdict: str
    reason: str | None = None
    host_slot: str | None = None
    top_cosine: float | None = None
    runner_up_cosine: float | None = None
    margin: float | None = None
    host_slot_duration: float | None = None
    host_slot_segments: int | None = None
    contradiction: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


def score_slots(slots: Mapping[str, Slot], voiceprint: TaggedVector) -> list[SlotScore]:
    """Score every slot against ``voiceprint``, sorted cosine-descending.

    Raises on a cross-space or dims-mismatched slot (corrupt input is an error,
    not a skip — a silently dropped slot would bias the verdict). Ties break on
    slot name for determinism.
    """
    scored = [
        SlotScore(
            slot=str(name),
            cosine=cosine(voiceprint, slot.vector),
            duration=slot.duration,
            segments=slot.segments,
        )
        for name, slot in slots.items()
    ]
    scored.sort(key=lambda s: (-s.cosine, s.slot))
    return scored


def _adequate_speech(total_speech: float | None, thresholds: Thresholds) -> bool:
    """Whether item total speech clears the confident-absence floor."""
    return total_speech is not None and total_speech >= thresholds.neg_min_total_duration


def passes_present_gates(scored: Sequence[SlotScore], thresholds: Thresholds) -> bool:
    """Whether the top slot clears every ``CONFIDENT_HOST_PRESENT`` gate.

    (cosine >= tau, duration >= min, segments >= min, and — when >=2 slots —
    the top1-top2 margin >= threshold.) Single-slot items auto-pass the margin
    gate. Exposed so a calibration grid sweep can score a slot list once and
    test many threshold points without recomputing cosines.
    """
    if not scored:
        return False
    top = scored[0]
    if top.cosine < thresholds.tau:
        return False
    if top.duration < thresholds.min_duration or top.segments < thresholds.min_segments:
        return False
    return not (len(scored) >= 2 and (top.cosine - scored[1].cosine) < thresholds.margin)


def label_positive(
    slots: Mapping[str, Slot],
    host_voiceprint: TaggedVector,
    thresholds: Thresholds,
    *,
    enrollment_ok: bool,
    total_speech: float | None = None,
    enrollment_reason: str = REASON_WEAK_ENROLLMENT,
    diagnostics: dict[str, Any] | None = None,
) -> LabelResult:
    """Decide whether the curated host is acoustically present in this item.

    Gate order: weak_enrollment -> no_slot_embeddings -> low_cosine_band (top
    cosine < tau; ``contradiction=True`` when top cosine < low_band AND item
    speech is adequate — a curated host confidently absent on their own channel
    is a candidate channel-fact error) -> short_duration (duration OR segment
    count below floor) -> near_tie (>=2 slots and margin < threshold) ->
    ``CONFIDENT_HOST_PRESENT``. Single-slot items auto-pass the margin.

    ``enrollment_ok`` is decided by the caller (enough held-out, non-leaking
    enrollment items); ``enrollment_reason`` lets it distinguish too-few
    (``weak_enrollment``) from leakage (``session_leakage_risk``).
    """
    diag = dict(diagnostics or {})

    if not enrollment_ok:
        if enrollment_reason not in ABSTAIN_REASONS:
            raise ValueError(f"unknown enrollment_reason {enrollment_reason!r}")
        return LabelResult(verdict=ABSTAIN, reason=enrollment_reason, diagnostics=diag)

    scored = score_slots(slots, host_voiceprint)
    if not scored:
        return LabelResult(
            verdict=ABSTAIN, reason=REASON_NO_SLOT_EMBEDDINGS, diagnostics=diag
        )

    top = scored[0]
    runner_up = scored[1].cosine if len(scored) >= 2 else None
    margin = (top.cosine - runner_up) if runner_up is not None else None

    def _result(verdict: str, reason: str | None, contradiction: bool = False) -> LabelResult:
        return LabelResult(
            verdict=verdict,
            reason=reason,
            host_slot=top.slot,
            top_cosine=top.cosine,
            runner_up_cosine=runner_up,
            margin=margin,
            host_slot_duration=top.duration,
            host_slot_segments=top.segments,
            contradiction=contradiction,
            diagnostics=diag,
        )

    # Cosine below the positive threshold -> not confident present.
    if top.cosine < thresholds.tau:
        contradiction = top.cosine < thresholds.low_band and _adequate_speech(
            total_speech, thresholds
        )
        return _result(ABSTAIN, REASON_LOW_COSINE_BAND, contradiction=contradiction)

    # Cosine high enough, but the slot is too short / sparse to trust.
    if top.duration < thresholds.min_duration or top.segments < thresholds.min_segments:
        return _result(ABSTAIN, REASON_SHORT_DURATION)

    # Two slots both look like the host -> cannot pick one (over-segmentation
    # or a genuine voice collision); abstain rather than guess.
    if runner_up is not None and margin is not None and margin < thresholds.margin:
        return _result(ABSTAIN, REASON_NEAR_TIE)

    return _result(CONFIDENT_HOST_PRESENT, None)


def label_negative_control(
    slots: Mapping[str, Slot],
    candidate_voiceprints: Mapping[str, TaggedVector],
    thresholds: Thresholds,
    *,
    total_speech: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> LabelResult:
    """Decide "no curated host present" on a negative-control (no-host) channel.

    For each curated host voiceprint, score the item's slots:
      * if ANY curated host clears the present-gates -> a curated host appears
        on a no-host channel: ``contradiction=True``, ``ABSTAIN`` (flag for
        review — the absence call would be wrong, but we don't silver-label a
        positive off a negative-control draw);
      * else if the best cosine across all hosts is < low_band AND item speech
        is adequate -> ``NO_CURATED_HOST_DETECTED`` (silver negative);
      * else ``ABSTAIN`` — ``low_cosine_band`` when a host is ambiguously
        similar (best cosine in ``[low_band, tau)``), ``short_duration`` when
        speech is inadequate to assert absence.
    """
    diag = dict(diagnostics or {})

    if not slots:
        return LabelResult(
            verdict=ABSTAIN, reason=REASON_NO_SLOT_EMBEDDINGS, diagnostics=diag
        )

    best_cosine: float | None = None
    best_host: str | None = None
    best_slot: str | None = None
    contradiction_hosts: list[str] = []

    for host, vp in candidate_voiceprints.items():
        scored = score_slots(slots, vp)
        if not scored:
            continue
        if passes_present_gates(scored, thresholds):
            contradiction_hosts.append(host)
        if best_cosine is None or scored[0].cosine > best_cosine:
            best_cosine = scored[0].cosine
            best_host = host
            best_slot = scored[0].slot

    contradiction = bool(contradiction_hosts)

    if best_cosine is None:
        return LabelResult(
            verdict=ABSTAIN, reason=REASON_NO_SLOT_EMBEDDINGS, diagnostics=diag
        )

    # ``best_candidate_host`` is highest-cosine, which is NOT necessarily the
    # host that cleared the present-gates — a reviewer chasing a contradiction
    # needs the actual trigger(s), so record them separately.
    diag.setdefault("best_candidate_host", best_host)
    if contradiction_hosts:
        diag.setdefault("contradiction_hosts", sorted(contradiction_hosts))

    def _result(verdict: str, reason: str | None) -> LabelResult:
        return LabelResult(
            verdict=verdict,
            reason=reason,
            host_slot=best_slot,
            top_cosine=best_cosine,
            contradiction=contradiction,
            diagnostics=diag,
        )

    if contradiction:
        return _result(ABSTAIN, REASON_LOW_COSINE_BAND)

    if best_cosine < thresholds.low_band:
        if _adequate_speech(total_speech, thresholds):
            return _result(NO_CURATED_HOST_DETECTED, None)
        # Confidently low cosine but not enough speech to assert absence.
        return _result(ABSTAIN, REASON_SHORT_DURATION)

    # A curated host is ambiguously similar — neither confidently present nor absent.
    return _result(ABSTAIN, REASON_LOW_COSINE_BAND)


def far_frr_at(
    genuine: Sequence[float],
    impostor: Sequence[float],
    tau: float,
) -> tuple[int, int, int, int]:
    """Count false accepts / false rejects at a cosine threshold ``tau``.

    A trial is *accepted* iff its cosine ``>= tau``. ``genuine`` trials are
    same-person (a slot vs its TRUE held-out voiceprint); ``impostor`` trials
    are different-person. Returns ``(far_count, frr_count, n_genuine,
    n_impostor)`` where ``far_count`` = impostors accepted and ``frr_count`` =
    genuines rejected. Rates + one-sided CIs come from
    :func:`voxint.harness.name_accuracy.wilson_ci`.
    """
    far_count = sum(1 for c in impostor if c >= tau)
    frr_count = sum(1 for c in genuine if c < tau)
    return far_count, frr_count, len(genuine), len(impostor)
