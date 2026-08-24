#!/usr/bin/env python3
"""Pinned, verifiable facts about the synthdetect eval sources (issue #144).

Single source of truth for the synthetic-speech-detection evaluation harness:
the detector model registry (repo/commit/weights) and the benchmark dataset
pins, kept as reviewable DATA (not buried in a script) so a contract test can
machine-check them and a reviewer can read them.

Two integrity rails ride on this module:

* **License gating.** Every model carries a ``license_class``:
  ``shippable`` (permissive MIT/BSD/Apache -- may be vendored and
  redistributed), ``noncommercial`` (CC-BY-NC-SA and kin -- evaluated in the
  harness and offered only as user-fetched opt-in, never redistributed by
  Voxint), or ``unlicensed`` (no grant at all -- Nes2Net today, whose author has
  not published a license). An ``unlicensed`` model is NOT runnable: the harness
  refuses to load it until the author grants a license, so the registry can name
  it for completeness without ever redistributing or executing it.
* **Sha provenance.** Weight file shas and upstream commits start CANDIDATE
  (``sha256=None`` / ``commit=None``): placeholders the ``verify-sources`` pass
  confirms against real downloaded bytes and freezes. ``weights_pinned()`` is
  False until then, so no code can claim a model is serve-ready on the strength
  of an unverified sha. As of S2b (2026-08-24) the default detector
  ``w2v2-aasist`` is FROZEN (real-byte shas + commit pinned, ``weights_pinned()``
  True); the other registered models remain CANDIDATE.

The two versioned identities the plan separates (inference space vs calibration
policy) are NOT both here: this module pins the inference-space INPUTS (weights +
repo + windowing); the calibration policy is produced by ``synthdetect_eval.py
calibrate`` in a later session and committed as its own artifact.

Provenance of these pins (from ``voxint_synthetic_speech_detection_research.md``,
2026-08-24). Numbers are the published, single-source EERs a maintainer's S3+
reproduction run confirms against real bytes; the tolerances are PROVISIONAL
until ratified from measured rerun variance (see ``docs/gpu-contracts.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Bump when the registry SHAPE or the selection seed changes -- recorded in a
# manifest/report so a reshuffle or a pin refresh is a visible, deliberate event.
SOURCES_VERSION: Final = "synthdetect-sources-v1"

# The one seed for every deterministic choice this harness makes downstream
# (corpus split assignment, bootstrap resampling). A single named constant so a
# reshuffle is a one-line, reviewable diff, never an accident.
SELECTION_SEED: Final = "voxint-synthdetect-144"

# The license class vocabulary, ordered from most to least permissive. A model
# whose class is not one of these is a registry bug (asserted at import).
LICENSE_CLASSES: Final = ("shippable", "noncommercial", "unlicensed")


class SourcesError(Exception):
    """A registry-integrity or license-gating problem (fail closed)."""


# --------------------------------------------------------------------------- #
# Weight files, reproduction targets, windowing (per-model data)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeightFile:
    """One vendored weight artifact.

    ``sha256`` is None while CANDIDATE (S1): the S2 ``verify-sources`` pass
    downloads the file, records the real digest, and freezes it here. ``url`` is
    provenance only -- nothing in S1 fetches it. ``license_spdx`` is per-FILE
    because an MIT repo does not automatically license every artifact it links
    (Google-Drive checkpoints and SSL bases are verified individually).
    """

    filename: str
    role: str
    url: str
    sha256: str | None
    size_bytes: int | None
    license_spdx: str

    def pinned(self) -> bool:
        """True once the real sha is frozen (S2); False while CANDIDATE."""
        return self.sha256 is not None


@dataclass(frozen=True)
class ReproductionTarget:
    """A published benchmark number this model must reproduce, and its gate.

    ``tolerance_status`` is ``provisional`` until S3 ratifies it from measured
    rerun variance (a miss is a STOP, never a silent tolerance widening). The
    ``protocol`` note pins the score polarity / crop rule / key file so the gate
    is unambiguous before any GPU run (the pre-registration doctrine).
    """

    benchmark: str
    metric: str  # "eer" | "tpr_clean" | "fpr_unmarked"
    published_pct: float
    tolerance_pp: float
    tolerance_status: str  # "provisional" | "ratified"
    protocol: str


@dataclass(frozen=True)
class WindowingPolicy:
    """Upstream vs production windowing -- part of the inference space.

    The upstream protocol (a fixed ``upstream_window_samples`` crop, or
    arbitrary-length for models that accept it) is what gate-1 reproduction uses.
    The production path (merge same-speaker turns separated by <
    ``merge_gap_s``, chunk into ``production_window_s`` windows at
    ``production_hop_s``, ``pooling`` the per-window logits) is a DIFFERENT
    function and is validated separately in M1. ``pooling`` is pinned here
    because raw-mean vs logit-mean vs max change the decision surface.
    """

    sample_rate_hz: int
    upstream_window_samples: int | None  # None => model accepts arbitrary length
    production_window_s: float
    production_hop_s: float
    merge_gap_s: float
    pooling: str  # "logit-mean"


@dataclass(frozen=True)
class ModelEntry:
    """One detector model: its code/weights identity, license, gates, windowing.

    ``inference_space`` is the durable identity governed by the parity gate;
    ``commit`` is None while CANDIDATE (pinned in S2). ``harness_only`` marks a
    model evaluated in M1 but deliberately NOT a shipped-service candidate in v1
    (AudioSeal watermark detection -- deferred to a v1.1 ship/skip decision
    backed by the measured harness numbers).
    """

    model_id: str
    inference_space: str
    family: str
    repo: str
    commit: str | None
    license_class: str
    code_license_spdx: str
    weights_license_spdx: str
    attribution: str
    default: bool
    harness_only: bool
    windowing: WindowingPolicy
    weights: tuple[WeightFile, ...]
    reproduction_targets: tuple[ReproductionTarget, ...]
    notes: str

    def enabled(self) -> bool:
        """False for an ``unlicensed`` model -- it never runs (no license grant)."""
        return self.license_class != "unlicensed"

    def weights_pinned(self) -> bool:
        """True only when every weight file has a frozen (non-CANDIDATE) sha."""
        return bool(self.weights) and all(w.pinned() for w in self.weights)


@dataclass(frozen=True)
class BenchmarkDataset:
    """One benchmark corpus pin for gate-1 reproduction.

    ``license_status`` is ``unverified`` in S1: the exact ASVspoof / In-the-Wild
    terms are confirmed at acquisition (plan open question 2). Audio is never
    committed; the corpus root is always a CLI arg, never hardcoded.
    """

    dataset_id: str
    name: str
    license_spdx: str | None
    license_status: str  # "verified" | "unverified"
    provenance_url: str
    keys: str
    split: str
    notes: str


# --------------------------------------------------------------------------- #
# The model registry (CANDIDATE shas/commits in S1)
# --------------------------------------------------------------------------- #
# AASIST-family models crop to a fixed 64,600-sample (~4.0375 s) window at
# 16 kHz; the production path merges same-speaker turns (<1 s gap), chunks into
# 4 s windows, and logit-mean pools. Models that accept arbitrary length
# (AntiDeepfake) set ``upstream_window_samples=None`` but share the production
# policy so the deployed function is identical across models.
_AASIST_WINDOWING: Final = WindowingPolicy(
    sample_rate_hz=16000,
    upstream_window_samples=64600,
    production_window_s=4.0,
    production_hop_s=4.0,
    merge_gap_s=1.0,
    pooling="logit-mean",
)
_ARBITRARY_WINDOWING: Final = WindowingPolicy(
    sample_rate_hz=16000,
    upstream_window_samples=None,
    production_window_s=4.0,
    production_hop_s=4.0,
    merge_gap_s=1.0,
    pooling="logit-mean",
)

MODELS: Final[dict[str, ModelEntry]] = {
    # The shipped default candidate: permissive MIT code AND weights, the
    # community reference baseline (wav2vec2-XLS-R + AASIST).
    "w2v2-aasist": ModelEntry(
        model_id="w2v2-aasist",
        inference_space="synthdetect-w2v2aasist-v1",
        family="wav2vec2-xls-r + aasist",
        repo="TakHemlata/SSL_Anti-spoofing",
        commit="4acaa61dcef5f7610f43aa4d0b29c4559b970cd2",  # frozen S2b (2026-08-24)
        license_class="shippable",
        code_license_spdx="MIT",
        weights_license_spdx="MIT",
        attribution="Tak et al., SSL_Anti-spoofing (MIT)",
        default=True,
        harness_only=False,
        windowing=_AASIST_WINDOWING,
        weights=(
            WeightFile(
                filename="LA_model.pth",
                role="aasist_checkpoint",
                # provenance only: the checkpoint lives in the upstream Drive folder
                url="https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB",
                sha256="bd6f36097259fe54e7004eb983651e5304d807be81156dbd04faccb70d91e10c",
                size_bytes=1271633441,
                license_spdx="MIT",
            ),
            WeightFile(
                filename="xlsr2_300m.pt",
                role="xlsr_ssl_base",
                url="https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt",
                sha256="b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9",
                size_bytes=3808868242,
                license_spdx="MIT",
            ),
        ),
        reproduction_targets=(
            ReproductionTarget(
                benchmark="asvspoof2021_df",
                metric="eer",
                published_pct=2.85,
                tolerance_pp=0.3,
                tolerance_status="provisional",
                protocol=(
                    "ASVspoof 2021 DF eval, official keys; 64,600-sample crop; "
                    "the runner journals scores as higher = more synthetic (invert "
                    "the checkpoint's bona-fide logits before writing); official "
                    "ASVspoof scorer"
                ),
            ),
            ReproductionTarget(
                benchmark="itw",
                metric="eer",
                published_pct=10.5,
                tolerance_pp=1.0,
                tolerance_status="provisional",
                protocol="In-the-Wild, author condition; 64,600-sample crop",
            ),
        ),
        notes="318M params; PyTorch + pinned fairseq; Google-Drive checkpoint.",
    ),
    # Best raw generalization but NON-COMMERCIAL weights: harness-eval and
    # user-fetched opt-in only, never redistributed. Raw scores only -- the
    # default calibration policy is NOT inherited (meaningless for other weights).
    "antideepfake-xlsr-2b": ModelEntry(
        model_id="antideepfake-xlsr-2b",
        inference_space="synthdetect-antideepfake-xlsr2b-v1",
        family="xls-r-2b + fc head",
        repo="nii-yamagishilab/AntiDeepfake",
        commit=None,  # CANDIDATE
        license_class="noncommercial",
        code_license_spdx="BSD-3-Clause",
        weights_license_spdx="CC-BY-NC-SA-4.0",
        attribution="NII Yamagishi Lab, AntiDeepfake (code BSD-3; weights CC-BY-NC-SA-4.0)",
        default=False,
        harness_only=False,
        windowing=_ARBITRARY_WINDOWING,
        weights=(
            WeightFile(
                filename="xlsr_2b_antideepfake.pt",
                role="antideepfake_checkpoint",
                url="https://huggingface.co/nii-yamagishilab/AntiDeepfake",
                sha256=None,  # CANDIDATE
                size_bytes=None,
                license_spdx="CC-BY-NC-SA-4.0",
            ),
        ),
        reproduction_targets=(
            ReproductionTarget(
                benchmark="itw",
                metric="eer",
                published_pct=1.23,
                tolerance_pp=0.5,
                tolerance_status="provisional",
                protocol="In-the-Wild, XLS-R-2B zero-shot; arbitrary-length 16 kHz input",
            ),
        ),
        notes="Best zero-shot ITW EER; ~2B params (tight on a 3060); non-commercial weights.",
    ),
    # Meta AudioSeal watermark detector: permissive, but HARNESS-ONLY in M1.
    # It only detects AudioSeal-marked audio (near-zero field coverage for
    # adversarial content), so it is deferred from the shipped service to a v1.1
    # ship/skip decision. Its gates are a clean-audio TPR and an unmarked-audio
    # FPR, never an EER.
    "audioseal": ModelEntry(
        model_id="audioseal",
        inference_space="synthdetect-audioseal-v1",
        family="audioseal watermark detector",
        repo="facebookresearch/audioseal",
        commit=None,  # CANDIDATE
        license_class="shippable",
        code_license_spdx="MIT",
        weights_license_spdx="MIT",
        attribution="Meta AI, AudioSeal (MIT)",
        default=False,
        harness_only=True,
        windowing=_ARBITRARY_WINDOWING,
        weights=(
            WeightFile(
                filename="audioseal_detector_16khz.pth",
                role="audioseal_detector",
                url="https://huggingface.co/facebook/audioseal",
                sha256=None,  # CANDIDATE
                size_bytes=None,
                license_spdx="MIT",
            ),
        ),
        reproduction_targets=(
            ReproductionTarget(
                benchmark="audioseal_marked",
                metric="tpr_clean",
                published_pct=99.0,
                tolerance_pp=1.0,
                tolerance_status="provisional",
                protocol="AudioSeal-marked clips, clean; TPR >= 99% AND an unmarked-audio FPR gate",
            ),
        ),
        notes="Harness-only in M1; watermark detection deferred to a v1.1 decision.",
    ),
    # Nes2Net: best accuracy-per-parameter, but NO license file (all rights
    # reserved). Named for completeness; UNLICENSED so it never runs. Enabled
    # only if the author grants a license (emailing them is an S1 task).
    "nes2net": ModelEntry(
        model_id="nes2net",
        inference_space="synthdetect-nes2net-v1",
        family="wav2vec2-xls-r + nes2net-x",
        repo="Liu-Tianchi/Nes2Net_ASVspoof_ITW",
        commit=None,  # CANDIDATE
        license_class="unlicensed",
        code_license_spdx="NOASSERTION",
        weights_license_spdx="NOASSERTION",
        attribution="Liu et al., Nes2Net (no license file; contact author)",
        default=False,
        harness_only=False,
        windowing=_AASIST_WINDOWING,
        weights=(
            WeightFile(
                filename="nes2net_x.pth",
                role="nes2net_checkpoint",
                url="https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW",
                sha256=None,  # CANDIDATE
                size_bytes=None,
                license_spdx="NOASSERTION",
            ),
        ),
        reproduction_targets=(
            ReproductionTarget(
                benchmark="asvspoof2021_df",
                metric="eer",
                published_pct=1.49,
                tolerance_pp=0.5,
                tolerance_status="provisional",
                protocol="ASVspoof 2021 DF; 64,600-sample crop (blocked: unlicensed)",
            ),
        ),
        notes="~511k-param back-end; SOTA acc/param; DISABLED until a license is granted.",
    ),
}

# --------------------------------------------------------------------------- #
# The benchmark dataset registry (gate-1 reproduction corpora)
# --------------------------------------------------------------------------- #
BENCHMARKS: Final[dict[str, BenchmarkDataset]] = {
    "asvspoof2021_df": BenchmarkDataset(
        dataset_id="asvspoof2021_df",
        name="ASVspoof 2021 Deepfake (DF) eval",
        license_spdx=None,
        license_status="unverified",  # confirmed at acquisition (open question 2)
        provenance_url="https://www.asvspoof.org/index2021.html",
        keys="official ASVspoof 2021 DF keys + metadata",
        split="eval",
        notes="~611k trials; anchored once overnight on a 3090-class node after subset shakeout.",
    ),
    "itw": BenchmarkDataset(
        dataset_id="itw",
        name="In-the-Wild (ITW)",
        license_spdx=None,
        license_status="unverified",
        provenance_url="https://deepfake-total.com/in_the_wild",
        keys="bundled real/spoof labels",
        split="eval",
        notes="~38 h, 58 public figures; the de facto real-world generalization benchmark.",
    ),
    # A synthesized-marker corpus for the AudioSeal harness-only gates; not a
    # public download but a generated stratum (marked + unmarked copies).
    "audioseal_marked": BenchmarkDataset(
        dataset_id="audioseal_marked",
        name="AudioSeal marked/unmarked pairs",
        license_spdx=None,
        license_status="unverified",
        provenance_url="generated locally from AudioSeal + bona fide sources",
        keys="marked vs unmarked provenance",
        split="eval",
        notes="Harness-only; evidence for the v1.1 AudioSeal ship/skip decision.",
    ),
}


# --------------------------------------------------------------------------- #
# Runnability guard + accessors
# --------------------------------------------------------------------------- #
def runnable(model: ModelEntry) -> bool:
    """True iff the harness may load and execute this model.

    Today this is exactly ``model.enabled()`` (license grant present), separated
    as its own predicate so a later gate (e.g. requiring pinned weights before a
    real inference run) can tighten it in one place.
    """
    return model.enabled()


def assert_runnable(model: ModelEntry) -> None:
    """Raise :class:`SourcesError` if a model must not be run (fail closed)."""
    if not runnable(model):
        raise SourcesError(
            f"model {model.model_id!r} is {model.license_class} and refuses to run "
            f"(no license grant); resolve the license before evaluating it"
        )


def default_model() -> ModelEntry:
    """The single shipped default candidate (exactly one ``default=True`` entry)."""
    defaults = [m for m in MODELS.values() if m.default]
    if len(defaults) != 1:
        raise SourcesError(f"registry must have exactly one default model, found {len(defaults)}")
    return defaults[0]


def get_model(model_id: str) -> ModelEntry:
    """Look up a model by id, raising :class:`SourcesError` if unknown."""
    try:
        return MODELS[model_id]
    except KeyError:
        raise SourcesError(f"unknown model {model_id!r}; known: {sorted(MODELS)}") from None


def _validate_registry() -> None:
    """Assert registry integrity at import (a bad pin is a bug, not runtime data).

    Checks: every model's key matches its ``model_id``; ``license_class`` is a
    known value; an ``unlicensed`` model is never runnable; exactly one default;
    every reproduction target names a known benchmark; benchmark keys match ids.
    """
    for key, model in MODELS.items():
        if key != model.model_id:
            raise SourcesError(f"model registry key {key!r} != model_id {model.model_id!r}")
        if model.license_class not in LICENSE_CLASSES:
            raise SourcesError(
                f"model {key!r} has unknown license_class {model.license_class!r}"
            )
        if model.license_class == "unlicensed" and runnable(model):
            raise SourcesError(f"unlicensed model {key!r} must not be runnable")
        for target in model.reproduction_targets:
            if target.benchmark not in BENCHMARKS:
                raise SourcesError(
                    f"model {key!r} target names unknown benchmark {target.benchmark!r}"
                )
    if sum(1 for m in MODELS.values() if m.default) != 1:
        raise SourcesError("registry must declare exactly one default model")
    for key, bench in BENCHMARKS.items():
        if key != bench.dataset_id:
            raise SourcesError(f"benchmark key {key!r} != dataset_id {bench.dataset_id!r}")


_validate_registry()
