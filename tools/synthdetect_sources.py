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
  ``w2v2-aasist`` is FROZEN and QUALIFIED (real-byte shas + commit pinned,
  ``weights_pinned()`` True, GPU determinism + smoke evidence dated). As of S3
  (2026-08-25) its DF-tuned sibling ``w2v2-aasist-df`` is likewise FROZEN and
  QUALIFIED: its DF-checkpoint sha is frozen from real bytes and its own dated GPU
  determinism + smoke verdict passed on RTX 3060 hardware
  (``docs/reports/synthdetect-gpu-smoke-df-2026-08-25.md``), including the strict
  ``module.`` DataParallel-unwrap load. The remaining registered models are still
  CANDIDATE.

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

import re
from dataclasses import dataclass
from typing import Final

# Bump when the registry SHAPE or the selection seed changes -- recorded in a
# manifest/report so a reshuffle or a pin refresh is a visible, deliberate event.
# v2 (2026-08-25, S3): added ReproductionTarget.gate_role + the DF sibling model.
# v3 (2026-08-25, S3): added WeightFile.state_dict_key_prefix (DF checkpoint's
#     nn.DataParallel "module." unwrap), declared as data + closed vocabulary.
# v4 (2026-08-26, S5): added the versioned degradation recipe registry (a closed
#     vocabulary of deterministic ffmpeg transforms; the argv builders live in
#     synthdetect_corpus.py). Additive-noise is deferred: its SNR mix needs a
#     measured parent RMS and so is not a pure argv (see the S5 pre-registration).
# v5 (2026-08-26, S5 PR-3): frozen cohort policy (one degraded child per
#     calibration parent, hash-assigned from six single-recipe chains).
SOURCES_VERSION: Final = "synthdetect-sources-v5"

# A frozen weight sha is a lowercase hex sha256; a pinned model commit is a
# lowercase hex git sha1. Enforced at import so a truncated/placeholder digest
# cannot masquerade as a real pin (a non-None sha alone is not evidence).
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")

# The reproduction-target metric and tolerance vocabularies. A value outside
# these is a registry bug (asserted at import, alongside gate_role).
METRICS: Final = ("eer", "tpr_clean", "fpr_unmarked")
TOLERANCE_STATUSES: Final = ("provisional", "ratified")

# The only state-dict key prefixes a checkpoint may declare (WeightFile). None is
# load-verbatim (the shipped default); ``"module."`` is the nn.DataParallel-save
# unwrap. A new transform must be a deliberate, reviewed registry change, not an
# accident, so the set is closed and asserted at import.
STATE_DICT_KEY_PREFIXES: Final = (None, "module.")

# The one seed for every deterministic choice this harness makes downstream
# (corpus split assignment, bootstrap resampling). A single named constant so a
# reshuffle is a one-line, reviewable diff, never an accident.
SELECTION_SEED: Final = "voxint-synthdetect-144"

# The license class vocabulary, ordered from most to least permissive. A model
# whose class is not one of these is a registry bug (asserted at import).
LICENSE_CLASSES: Final = ("shippable", "noncommercial", "unlicensed")

# The reproduction-target gate roles. A ``stop_gate`` target's ratified miss is
# a STOP; a ``diagnostic`` target is measured and reported but never gates. A
# target whose role is not one of these is a registry bug (asserted at import).
GATE_ROLES: Final = ("stop_gate", "diagnostic")


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
    # A state-dict key prefix this checkpoint's bytes carry that the runner must
    # strip before a strict load, declared as data (never inferred at load time).
    # ``"module."`` marks a checkpoint saved from an ``nn.DataParallel``-wrapped
    # model: every key is ``module.``-prefixed, and on a single GPU the unwrapped
    # forward is numerically identical. None (the default) means load the bytes
    # verbatim -- the shipped default's proven path. The runner fails closed unless
    # every key uniformly carries the declared prefix, and keeps ``strict=True``
    # after stripping, so a mis-declared prefix can never mask a wrong checkpoint.
    state_dict_key_prefix: str | None = None

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

    ``gate_role`` separates a hard reproduction gate from a tracked observation:

    * ``stop_gate`` -- the published number is a checkpoint-and-protocol-exact
      anchor; a miss (once the tolerance is ratified) is a STOP.
    * ``diagnostic`` -- the number is MEASURED and REPORTED but is NOT a
      stop-gate, because a checkpoint-exact, citable anchor for THIS model on
      THIS benchmark is not pinned. A diagnostic is generalization evidence, not
      a pass/fail bar. (S3 decision, 2026-08-25: the In-the-Wild number for the
      ASVspoof2019-LA-trained ``w2v2-aasist`` checkpoint is diagnostic; the one
      hard DF anchor lives on ``w2v2-aasist-df`` with its DF-trained checkpoint.)
    """

    benchmark: str
    metric: str  # "eer" | "tpr_clean" | "fpr_unmarked"
    published_pct: float
    tolerance_pp: float
    tolerance_status: str  # "provisional" | "ratified"
    protocol: str
    gate_role: str  # "stop_gate" | "diagnostic" -- REQUIRED, no default: a
    # forgotten role must be a construction error, never a silent hard STOP.


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
        """True only when the model commit AND every weight file sha are frozen.

        The pin doctrine freezes the code identity (``commit``) and the weight
        shas together, so a model with real shas but a CANDIDATE (``None``)
        commit is not serve-ready. Requiring both here keeps ``weights_pinned``
        honest with the freeze ceremony rather than sha-only.
        """
        return (
            self.commit is not None
            and bool(self.weights)
            and all(w.pinned() for w in self.weights)
        )


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
            # The published 2.85% ASVspoof 2021 DF EER is NOT a property of this
            # ASVspoof2019-LA-trained checkpoint: it is achieved by the DF-tuned
            # Best_LA_model_for_DF.pth, pinned as `w2v2-aasist-df` below. This
            # entry stays the production DEFAULT (shippable MIT weights), and its
            # only reproduction number is a DIAGNOSTIC In-the-Wild generalization
            # observation, measured and reported but never a stop-gate (no
            # checkpoint-exact citable ITW anchor for THIS checkpoint). (S3
            # decision, 2026-08-25; see docs/gpu-contracts.md.)
            ReproductionTarget(
                benchmark="itw",
                metric="eer",
                published_pct=10.5,
                tolerance_pp=1.0,
                tolerance_status="provisional",
                protocol="In-the-Wild, author condition; 64,600-sample crop",
                gate_role="diagnostic",
            ),
        ),
        notes=(
            "318M params; PyTorch + pinned fairseq; Google-Drive checkpoint. "
            "The production DEFAULT (LA_model.pth, ASVspoof2019 LA); the hard DF "
            "reproduction anchor lives on w2v2-aasist-df."
        ),
    ),
    # The DF-tuned sibling of the default: SAME code/runtime/XLS-R base, a
    # DIFFERENT aasist checkpoint (Best_LA_model_for_DF.pth) and therefore a
    # DIFFERENT inference space. This is the checkpoint that produces the upstream
    # 2.85% ASVspoof 2021 DF EER, so it -- not the production default -- carries
    # the hard DF stop-gate (S3 decision, 2026-08-25). QUALIFIED (S3, 2026-08-25):
    # the weight sha is frozen from real bytes (receipt below) AND its own dated
    # GPU determinism + smoke verdict passed on RTX 3060 hardware, including the
    # strict "module." DataParallel-unwrap load
    # (docs/reports/synthdetect-gpu-smoke-df-2026-08-25.md).
    "w2v2-aasist-df": ModelEntry(
        model_id="w2v2-aasist-df",
        inference_space="synthdetect-w2v2aasistdf-v1",
        family="wav2vec2-xls-r + aasist",
        repo="TakHemlata/SSL_Anti-spoofing",
        commit="4acaa61dcef5f7610f43aa4d0b29c4559b970cd2",  # same vendored model.py
        license_class="shippable",
        code_license_spdx="MIT",
        weights_license_spdx="MIT",
        attribution="Tak et al., SSL_Anti-spoofing (MIT)",
        default=False,
        harness_only=False,
        windowing=_AASIST_WINDOWING,
        weights=(
            WeightFile(
                filename="Best_LA_model_for_DF.pth",
                role="aasist_checkpoint",
                # provenance only: the DF-tuned checkpoint in the upstream Drive
                # folder (upstream README's designated DF checkpoint). Receipt:
                # docs/reports/synthdetect-weight-receipt-df-2026-08-25.md
                url="https://drive.google.com/uc?id=1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG",
                sha256="1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804",
                size_bytes=1271642081,
                license_spdx="MIT",
                # This checkpoint was saved from an nn.DataParallel-wrapped model:
                # all 674 keys are ``module.``-prefixed (the default's are not).
                # Declared here so the runner strips it before a strict load; on a
                # single GPU this is numerically identical to upstream's eval.
                # Verified cold on maintainer GPU hardware (S3 smoke, 2026-08-25).
                state_dict_key_prefix="module.",
            ),
            # The XLS-R front end is byte-identical to the default's base.
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
                    "ASVspoof scorer. The Gate-1 PASS is the FULL eval cohort "
                    "(~611k trials); a seeded subset is a preflight + Gate-2 "
                    "cohort only and never inherits the full-set EER tolerance."
                ),
                gate_role="stop_gate",
            ),
        ),
        notes=(
            "DF-tuned checkpoint (Best_LA_model_for_DF.pth); the upstream 2.85% "
            "ASVspoof 2021 DF anchor. QUALIFIED (S3, 2026-08-25): dated GPU "
            "determinism + smoke verdict on RTX 3060, strict module.-unwrap load."
        ),
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
                gate_role="stop_gate",
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
                gate_role="stop_gate",
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
                # A checkpoint-exact DF anchor, so its role IS stop_gate -- but the
                # model is unlicensed and never runnable, so the import-time
                # "one runnable DF stop-gate" rail excludes it. If a license lands,
                # that rail fires (two runnable DF stop-gates) and forces a
                # deliberate resolution rather than a silent second hard gate.
                gate_role="stop_gate",
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
# Degradation recipes (S5): a closed, versioned vocabulary of deterministic
# ffmpeg transforms. This module pins WHAT each recipe is (data); the argv
# builders and chain serialization live in synthdetect_corpus.py (logic). The
# recipe_id is the corpus identity of the transform, so it is versioned (`-v1`)
# and its bytes are reproducible only on the pinned realization toolchain
# (container digest + codec library versions), never as universal ffmpeg
# reproducibility. Additive-noise is intentionally absent: an SNR mix needs the
# measured parent RMS, which is audio-dependent and so cannot be a pure argv; it
# lands with the executor. See docs/gpu-contracts.md, the S5 pre-registration.
# --------------------------------------------------------------------------- #
DEGRADATION_FAMILIES: Final = ("codec", "telephony", "speed")

# A recipe id is a lowercase token joined only by '-'; '|' is the chain
# separator and whitespace/dots are forbidden, so a recipe id is always a safe
# manifest token and a chain string round-trips unambiguously.
_RECIPE_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class DegradationRecipe:
    """One deterministic ffmpeg transform, as reviewable data.

    ``lossy`` recipes are a real round trip (canonical PCM -> pinned encoder ->
    ``intermediate_format`` bitstream -> pinned decoder -> canonical PCM), so
    ``intermediate_format`` names the encoded container. A non-lossy recipe (a
    timeline filter such as ``speed``) is a single pass straight back to canonical
    PCM and carries no ``intermediate_format``. ``encode_args`` are the ffmpeg
    output options for the transform pass, emitted verbatim by the builder between
    the pinned raw-input framing and the intermediate/output path.
    """

    recipe_id: str
    family: str
    implementation: str
    lossy: bool
    encode_args: tuple[str, ...]
    intermediate_format: str


DEGRADATION_RECIPES: Final[dict[str, DegradationRecipe]] = {
    "mp3-cbr48-v1": DegradationRecipe(
        recipe_id="mp3-cbr48-v1",
        family="codec",
        implementation="libmp3lame",
        lossy=True,
        encode_args=("-c:a", "libmp3lame", "-b:a", "48k", "-ar", "16000", "-ac", "1"),
        intermediate_format="mp3",
    ),
    "opus-voip-cbr16-f20-v1": DegradationRecipe(
        recipe_id="opus-voip-cbr16-f20-v1",
        family="codec",
        implementation="libopus",
        lossy=True,
        encode_args=(
            "-c:a", "libopus", "-b:a", "16k", "-vbr", "off",
            "-application", "voip", "-frame_duration", "20", "-ar", "16000", "-ac", "1",
        ),
        intermediate_format="opus",
    ),
    "aac-lc-cbr48-v1": DegradationRecipe(
        recipe_id="aac-lc-cbr48-v1",
        family="codec",
        implementation="aac",
        lossy=True,
        encode_args=(
            "-c:a", "aac", "-b:a", "48k", "-profile:a", "aac_low", "-ar", "16000", "-ac", "1",
        ),
        intermediate_format="adts",
    ),
    "g711-mulaw-8k-v1": DegradationRecipe(
        recipe_id="g711-mulaw-8k-v1",
        family="telephony",
        implementation="pcm_mulaw",
        lossy=True,
        encode_args=("-c:a", "pcm_mulaw", "-ar", "8000", "-ac", "1"),
        intermediate_format="wav",
    ),
    "amr-nb-122-v1": DegradationRecipe(
        recipe_id="amr-nb-122-v1",
        family="telephony",
        implementation="libopencore_amrnb",
        lossy=True,
        encode_args=("-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-ar", "8000", "-ac", "1"),
        intermediate_format="amr",
    ),
    "speed-atempo-0p90-v1": DegradationRecipe(
        recipe_id="speed-atempo-0p90-v1",
        family="speed",
        implementation="atempo",
        lossy=False,
        # Single pass to canonical PCM: the -ar/-ac/-c:a framing comes from the
        # builder's canonical output, so the recipe only names the timeline filter.
        encode_args=("-filter:a", "atempo=0.90"),
        intermediate_format="",
    ),
}


def get_recipe(recipe_id: str) -> DegradationRecipe:
    """Look up a degradation recipe by id, raising :class:`SourcesError` if unknown."""
    try:
        return DEGRADATION_RECIPES[recipe_id]
    except KeyError:
        raise SourcesError(
            f"unknown degradation recipe {recipe_id!r}; known: {sorted(DEGRADATION_RECIPES)}"
        ) from None


def _validate_recipes(recipes: dict[str, DegradationRecipe] = DEGRADATION_RECIPES) -> None:
    """Assert recipe-registry integrity at import (a bad recipe is a bug).

    Every key matches its ``recipe_id``; the id is a safe token (no ``|``,
    whitespace, or dot); ``family`` is known; ``implementation`` (the reviewable
    encoder/decoder claim) and ``encode_args`` are non-empty; a lossy recipe names
    a non-empty ``intermediate_format`` and a non-lossy one names none.
    """
    for key, recipe in recipes.items():
        if key != recipe.recipe_id:
            raise SourcesError(f"recipe key {key!r} != recipe_id {recipe.recipe_id!r}")
        if not _RECIPE_ID_RE.match(recipe.recipe_id):
            raise SourcesError(
                f"recipe id {recipe.recipe_id!r} must be a lowercase '-'-joined token"
            )
        if recipe.family not in DEGRADATION_FAMILIES:
            raise SourcesError(
                f"recipe {key!r} has unknown family {recipe.family!r} "
                f"(allowed: {DEGRADATION_FAMILIES})"
            )
        if not recipe.implementation.strip():
            raise SourcesError(f"recipe {key!r} has an empty implementation")
        if not recipe.encode_args:
            raise SourcesError(f"recipe {key!r} has empty encode_args")
        if recipe.lossy and not recipe.intermediate_format:
            raise SourcesError(f"lossy recipe {key!r} must name an intermediate_format")
        if not recipe.lossy and recipe.intermediate_format:
            raise SourcesError(
                f"non-lossy recipe {key!r} must not carry an intermediate_format "
                f"(got {recipe.intermediate_format!r})"
            )


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


def _validate_registry(
    models: dict[str, ModelEntry] = MODELS,
    benchmarks: dict[str, BenchmarkDataset] = BENCHMARKS,
) -> None:
    """Assert registry integrity at import (a bad pin is a bug, not runtime data).

    Structural rails: every model's key matches its ``model_id``;
    ``license_class`` is a known value; an ``unlicensed`` model is never
    runnable; exactly one default; ``inference_space`` ids are unique across
    models; benchmark keys match ids.

    Pin rails: a non-None weight ``sha256`` is a 64-char lowercase hex digest
    with a positive ``size_bytes``, and a non-None model ``commit`` is a 40-char
    lowercase hex sha -- a truncated or placeholder value must not pass for a
    real freeze.

    Reproduction-target rails: every target names a known ``benchmark``,
    ``metric``, ``tolerance_status``, and ``gate_role``, with a positive
    ``tolerance_pp``.

    Gate rails (the S3 doctrine, now machine-checked, not test-only): the shipped
    default carries NO ``stop_gate`` target (it must not gate on a checkpoint it
    is not), and at most one RUNNABLE model may claim the ASVspoof-DF EER
    stop-gate (a second, e.g. an unblocked Nes2Net, must be resolved deliberately
    rather than silently activating a competing hard gate).

    Parameterised on ``models``/``benchmarks`` so the failure paths are testable;
    the import-time call below validates the real registry.
    """
    seen_spaces: dict[str, str] = {}
    runnable_df_stop_gates: list[str] = []
    for key, model in models.items():
        if key != model.model_id:
            raise SourcesError(f"model registry key {key!r} != model_id {model.model_id!r}")
        if model.license_class not in LICENSE_CLASSES:
            raise SourcesError(
                f"model {key!r} has unknown license_class {model.license_class!r}"
            )
        if model.license_class == "unlicensed" and runnable(model):
            raise SourcesError(f"unlicensed model {key!r} must not be runnable")
        if model.inference_space in seen_spaces:
            raise SourcesError(
                f"model {key!r} reuses inference_space {model.inference_space!r} "
                f"already claimed by {seen_spaces[model.inference_space]!r}"
            )
        seen_spaces[model.inference_space] = key
        if model.commit is not None and not _COMMIT_RE.match(model.commit):
            raise SourcesError(
                f"model {key!r} commit {model.commit!r} is not a 40-char hex sha"
            )
        for weight in model.weights:
            if weight.sha256 is not None and not _SHA256_RE.match(weight.sha256):
                raise SourcesError(
                    f"model {key!r} weight {weight.filename!r} sha256 "
                    f"{weight.sha256!r} is not a 64-char hex digest"
                )
            if weight.sha256 is not None and not (weight.size_bytes and weight.size_bytes > 0):
                raise SourcesError(
                    f"model {key!r} weight {weight.filename!r} is sha-pinned but has "
                    f"no positive size_bytes ({weight.size_bytes!r})"
                )
            if weight.state_dict_key_prefix not in STATE_DICT_KEY_PREFIXES:
                raise SourcesError(
                    f"model {key!r} weight {weight.filename!r} declares an unknown "
                    f"state_dict_key_prefix {weight.state_dict_key_prefix!r} "
                    f"(allowed: {STATE_DICT_KEY_PREFIXES})"
                )
        for target in model.reproduction_targets:
            if target.benchmark not in benchmarks:
                raise SourcesError(
                    f"model {key!r} target names unknown benchmark {target.benchmark!r}"
                )
            if target.metric not in METRICS:
                raise SourcesError(
                    f"model {key!r} target has unknown metric {target.metric!r}"
                )
            if target.tolerance_status not in TOLERANCE_STATUSES:
                raise SourcesError(
                    f"model {key!r} target has unknown tolerance_status "
                    f"{target.tolerance_status!r}"
                )
            if target.tolerance_pp <= 0:
                raise SourcesError(
                    f"model {key!r} target has non-positive tolerance_pp {target.tolerance_pp!r}"
                )
            if target.gate_role not in GATE_ROLES:
                raise SourcesError(
                    f"model {key!r} target has unknown gate_role {target.gate_role!r}"
                )
            if (
                runnable(model)
                and target.benchmark == "asvspoof2021_df"
                and target.metric == "eer"
                and target.gate_role == "stop_gate"
            ):
                runnable_df_stop_gates.append(key)
            if model.default and target.gate_role == "stop_gate":
                raise SourcesError(
                    f"default model {key!r} must not carry a stop_gate target "
                    f"(benchmark {target.benchmark!r}); it cannot gate on a checkpoint it is not"
                )
    if sum(1 for m in models.values() if m.default) != 1:
        raise SourcesError("registry must declare exactly one default model")
    if len(runnable_df_stop_gates) > 1:
        raise SourcesError(
            f"more than one runnable model claims the ASVspoof-DF EER stop-gate: "
            f"{runnable_df_stop_gates!r}; exactly one DF anchor may gate at a time"
        )
    for key, bench in benchmarks.items():
        if key != bench.dataset_id:
            raise SourcesError(f"benchmark key {key!r} != dataset_id {bench.dataset_id!r}")


# --------------------------------------------------------------------------- #
# Frozen cohort policy (S5 PR-3)
# --------------------------------------------------------------------------- #
S5_COHORT_VERSION: Final[int] = 1
S5_COHORT_SELECTION_POLICY: Final[str] = "hash-assign-v1"

FROZEN_COHORT_CHAINS: Final[tuple[tuple[str, ...], ...]] = (
    ("mp3-cbr48-v1",),
    ("opus-voip-cbr16-f20-v1",),
    ("aac-lc-cbr48-v1",),
    ("g711-mulaw-8k-v1",),
    ("amr-nb-122-v1",),
    ("speed-atempo-0p90-v1",),
)


def _validate_cohort_chains(
    chains: tuple[tuple[str, ...], ...] = FROZEN_COHORT_CHAINS,
    recipes: dict[str, DegradationRecipe] = DEGRADATION_RECIPES,
) -> None:
    """Assert cohort-chain integrity at import."""
    if not chains:
        raise SourcesError("FROZEN_COHORT_CHAINS must be non-empty")
    seen_serialized: set[str] = set()
    for chain in chains:
        if not chain:
            raise SourcesError("FROZEN_COHORT_CHAINS contains an empty chain")
        for recipe_id in chain:
            if recipe_id not in recipes:
                raise SourcesError(
                    f"FROZEN_COHORT_CHAINS names unknown recipe {recipe_id!r}"
                )
        serialized = "|".join(chain)
        if serialized in seen_serialized:
            raise SourcesError(
                f"FROZEN_COHORT_CHAINS contains duplicate chain {serialized!r}"
            )
        seen_serialized.add(serialized)


_validate_registry()
_validate_recipes()
_validate_cohort_chains()
