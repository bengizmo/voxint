#!/usr/bin/env python3
"""Diarization (DER/JER) + ASR (WER) quality harness for Voxint (issue #97).

A MAINTAINER instrument, never shipped to users. It scores hypothesis
diarization and transcripts against public ground truth (AMI + VoxConverse) so
the numerics doctrine can be honored with measured evidence when the GPU knobs
change (#96). It is a **tripwire, not a benchmark**: the 14-file subset cannot
prove non-regression, only catch gross breakage, and every threshold must be
measured from a zero-change noise floor rather than reasoned.

This module holds the three pieces that are testable without a worker:

* the diarization scorer, built on the vetted ``pyannote.metrics`` (never a
  hand-rolled DER: optimal mapping + collar + overlap + UEM crop is a
  silent-bug generator), fed through pyannote's own accumulators so the pooled
  number is a true micro-average, not a mean of per-file rates;
* the ASR scorer, reused verbatim from the frozen bakeoff WER stack
  (``tests/parity/whisper_bakeoff_score.py`` + the sha-pinned Whisper
  normalizer), applied to raw reference and raw hypothesis together;
* the ``report`` renderer, a pure metrics-JSON -> Markdown step that writes a
  dated ``docs/reports/eval-quality-baseline-*.md`` in house style. It scores
  each corpus separately (one metrics JSON per corpus, never a mixed
  accumulator) so no single grand AMI+VoxConverse number is ever published, and
  when a corpus is scored K times it renders the zero-change noise band
  (max spread per metric) that thresholds must be measured from.

The ``score`` subcommand is manifest-driven and corpus-layout-agnostic: it takes
a JSON manifest mapping each recording to its hypothesis/reference file paths, so
no ground-truth path or hostname is ever hardcoded (clean-room). Producing that
manifest by running the pipeline on the subset (submit -> poll -> read DB ->
emit relabelled hypothesis RTTM/text) is the ``run`` step, which needs a live
worker and lands with its live validation.

Correctness invariants baked in here (from the plan's 3-model consult):

* **Relabel.** ``to_rttm`` writes the run UUID; DER needs the corpus recording
  id. ``parse_rttm`` re-keys every hypothesis to the recording id so it aligns
  with the reference RTTM/UEM.
* **Collar is total width.** ``pyannote.metrics`` ``collar`` is the centered
  total width: NIST +/-250 ms is ``collar=0.5``. Validated by a fixture.
* **Primary metric is strict** (``collar=0.0``, ``skip_overlap=False``); the
  forgiving ``collar=0.5`` / skip-overlap number is a labelled diagnostic only.
* **UEMs are applied**; reference and hypothesis are cropped consistently, and a
  UEM is an explicit choice per recording (a path, or a deliberate null), never
  an omission that silently changes the protocol.
* **JER is a delta-only signal.** pyannote 4.1's speaker mapping maximizes raw
  co-occurrence, not the official DIHARD Jaccard-optimal assignment, so its
  absolute JER is never DIHARD-comparable (see ``JER_MAPPING``); DER is the
  gate-bearing primary. A true DIHARD JER is a deferred follow-up.
* Every report stamps a scorer-side environment manifest (git sha, scorer
  versions, the frozen normalizer fingerprint, the manifest hash, the scored
  cohort) so an aggregate can never move without the report explaining why.
* **Cohort binding (fail closed).** A ``run``-produced manifest carries a
  ``cohort`` block: the per-input records (audio + reference_rttm + uem +
  wer_reference, byte length and sha256, explicit nulls for absent roles) plus
  the pipeline-environment identity. ``score`` recomputes the cohort hash from
  the bytes it ACTUALLY scored (it re-derives the reference/uem/wer-reference
  shas and rejects any that disagree with the attestation; only the audio and
  the pipeline environment stay pure ``run`` attestations), then stamps
  ``environment.cohort_sha256``. ``report`` refuses to render unless every run
  of a corpus shares that identity and an identical recording set, so calling K
  runs a zero-change noise floor is a checked claim, not a hope. This is
  accidental-drift integrity for a single-operator harness, not defence against
  a hostile caller (without the audio or a signature the audio/environment
  attestations are unverifiable, which is acceptable here).

Run:

    uv run --isolated --extra parity --extra eval-quality \\
        tools/eval_quality.py score --manifest <manifest.json> --out <metrics.json>

    uv run --isolated --extra parity --extra eval-quality \\
        tools/eval_quality.py report --date 2026-08-20 --out <report.md> \\
        --run ami=<ami_run1.json> --run ami=<ami_run2.json> \\
        --run voxconverse=<vc_run1.json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The frozen `run`-step contracts (pure, unit-covered): pipeline-environment
# identity + the domain-separated cohort descriptor/hash. `score` reuses them so
# a metrics JSON carries the SAME cohort identity `run` will stamp, recomputed
# here from the bytes actually scored rather than trusted from the manifest.
# ``eval_run`` imports only the bakeoff ``_us`` helper (no pyannote/jiwer), so it
# is safe to import at module load in every lane, including ``run``/``report``.
sys.path.insert(0, str(REPO / "tools"))
import eval_run  # noqa: E402

# The heavy scoring stack (pyannote.core/metrics + the frozen jiwer WER stack) is
# loaded LAZILY by ``_load_scoring`` and used ONLY by ``score``. ``report`` is a
# pure metrics-JSON -> Markdown renderer and ``run`` is a submit/poll/export
# driver: neither needs pyannote or jiwer, so importing this module (and running
# those two subcommands) must succeed with the scoring extras absent. The names
# below are module globals populated on first ``_load_scoring()`` call; the
# scoring functions reference them at call time (after ``cmd_score`` loads them).
Annotation: Any = None
Segment: Any = None
Timeline: Any = None
DiarizationErrorRate: Any = None
JaccardErrorRate: Any = None
NORMALIZER_VERSION: str = ""
runtime_fingerprint: Any = None
score_pooled: Any = None


def _load_scoring() -> None:
    """Import the pyannote + jiwer scoring stack into module globals (once).

    Called at the top of ``cmd_score`` only. A missing extra raises the normal
    ``ImportError`` here rather than at module import, so ``import eval_quality``,
    ``run``, and ``report`` all work in the dev lane without the parity/
    eval-quality extras installed.
    """
    global Annotation, Segment, Timeline, DiarizationErrorRate, JaccardErrorRate
    global NORMALIZER_VERSION, runtime_fingerprint, score_pooled
    if Annotation is not None:
        return
    from pyannote.core import Annotation as _Annotation
    from pyannote.core import Segment as _Segment
    from pyannote.core import Timeline as _Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate as _DER
    from pyannote.metrics.diarization import JaccardErrorRate as _JER

    # Frozen WER stack (parity extra): jiwer + the sha-pinned Whisper normalizer.
    from tests.parity.bakeoff.normalize import NORMALIZER_VERSION as _NV
    from tests.parity.bakeoff.normalize import runtime_fingerprint as _rf
    from tests.parity.whisper_bakeoff_score import score_pooled as _sp

    Annotation, Segment, Timeline = _Annotation, _Segment, _Timeline
    DiarizationErrorRate, JaccardErrorRate = _DER, _JER
    NORMALIZER_VERSION, runtime_fingerprint, score_pooled = _NV, _rf, _sp

# The strict, gate-bearing diarization protocol vs the forgiving diagnostic one.
STRICT = {"collar": 0.0, "skip_overlap": False}
DIAGNOSTIC = {"collar": 0.5, "skip_overlap": True}

# The scoring protocol is a property of THIS harness, never the caller's to set:
# two runs are a legitimate zero-change pair only if scored under the same
# strict+diagnostic protocol, so the constant is baked into the cohort descriptor
# (and thus the hash), not read from the manifest.
HARNESS_PROTOCOL = {"strict": STRICT, "diagnostic": DIAGNOSTIC}

# The four cohort input roles. `audio` drives the hypothesis; reference_rttm/uem/
# wer_reference are the ground truth the score is taken against. Hypothesis files
# are OUTPUTS, never cohort inputs (two zero-change runs differ there by design,
# which is exactly what the noise floor measures). A corpus that lacks a role
# (VoxConverse has no UEM or WER) carries an EXPLICIT null record for it.
COHORT_ROLES = ("audio", "reference_rttm", "uem", "wer_reference")

# pyannote 4.1's JaccardErrorRate maps hypothesis to reference speakers by
# MAXIMIZING raw co-occurrence duration, whereas the official DIHARD/dscore JER
# MINIMIZES the pairwise Jaccard-error assignment. The two disagree materially
# under severe miss/false-alarm imbalance (a valid fixture gives pyannote 0.962
# vs Jaccard-optimal 0.773). So this harness's JER is a SELF-CONSISTENT delta
# signal only — never a DIHARD-comparable absolute. Recorded in the report as
# ``jer_mapping`` so no reader mistakes it for the official metric; a true
# DIHARD JER is a deferred follow-up if an absolute claim is ever needed.
JER_MAPPING = "pyannote-4.1-cooccurrence (delta-only; not DIHARD Jaccard-optimal)"

SCHEMA_VERSION = 1


def _git_sha() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - env dependent
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


class EvalError(Exception):
    """A user-facing input problem (bad manifest, malformed RTTM/UEM)."""


# --------------------------------------------------------------------------- #
# Parsing (manual so we control the recording id and never touch temp files)
# --------------------------------------------------------------------------- #
def parse_rttm(text: str, recording_id: str) -> Annotation:
    """Parse RTTM ``SPEAKER`` lines into one Annotation keyed to ``recording_id``.

    The file-id column is IGNORED: ``to_rttm`` emits the run UUID there, but the
    scorer needs every hypothesis and reference re-keyed to the corpus recording
    id so DER aligns them. Non-positive and malformed intervals are rejected
    rather than silently dropped.
    """
    _load_scoring()
    annotation = Annotation(uri=recording_id)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        if len(parts) < 9:
            raise EvalError(f"RTTM line {lineno}: expected >=9 fields, got {len(parts)}")
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError as exc:
            raise EvalError(f"RTTM line {lineno}: bad start/duration: {exc}") from exc
        label = parts[7]
        if duration <= 0.0:
            raise EvalError(f"RTTM line {lineno}: non-positive duration {duration}")
        segment = Segment(start, start + duration)
        # A unique track per line, NOT the ``annotation[segment] = label``
        # shorthand (which reuses the default track "_"): two speakers with
        # exactly-coincident boundaries would otherwise overwrite one another,
        # silently deleting overlap and moving DER/JER.
        annotation[segment, annotation.new_track(segment)] = label
    return annotation


def parse_uem(text: str, recording_id: str) -> Timeline:
    """Parse a NIST UEM into a Timeline, keeping only ``recording_id`` rows."""
    _load_scoring()
    segments: list[Segment] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise EvalError(f"UEM line {lineno}: expected 4 fields, got {parts!r}")
        if parts[0] != recording_id:
            continue
        start, end = float(parts[2]), float(parts[3])
        if end <= start:
            raise EvalError(f"UEM line {lineno}: non-positive region {start}..{end}")
        segments.append(Segment(start, end))
    if not segments:
        raise EvalError(f"UEM has no region for recording {recording_id!r}")
    return Timeline(segments, uri=recording_id)


# --------------------------------------------------------------------------- #
# Diarization scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiarItem:
    """One recording's parsed diarization inputs."""

    recording_id: str
    reference: Annotation
    hypothesis: Annotation
    uem: Timeline | None


@dataclass
class DiarResult:
    """Per-recording detailed components + the pooled (accumulated) rates."""

    protocol: str
    collar: float
    skip_overlap: bool
    per_recording: dict[str, dict[str, float]] = field(default_factory=dict)
    pooled_der: float = 0.0
    pooled_jer: float = 0.0


def reference_metadata(reference: Annotation, uem: Timeline | None) -> dict[str, float]:
    """Corpus-descriptive stats for the report table, taken from the reference.

    Computed on the reference cropped to the scored region (the UEM), so the
    speaker count and overlap fraction describe exactly what DER scored, not the
    whole file. ``reference_overlap_pct`` is the share of reference *speech*
    time (union of all speakers) where two or more speakers are simultaneously
    active. These are descriptive only; they never enter a rate.
    """
    _load_scoring()
    scoped = reference.crop(uem, mode="intersection") if uem is not None else reference
    speech = scoped.get_timeline().support().duration()
    overlap = scoped.get_overlap().duration()
    # Wall-clock evaluated span: the UEM duration when one is applied, else the
    # region the reference actually covers. This is NOT the DER ``total_s``
    # denominator, which speaker-sums overlap and so exceeds wall-clock on
    # multi-speaker meetings; the table shows real minutes, the rate keeps total_s.
    evaluated = uem.duration() if uem is not None else speech
    return {
        "speaker_count": float(len(scoped.labels())),
        "reference_overlap_pct": (overlap / speech * 100.0) if speech > 0.0 else 0.0,
        "evaluated_s": float(evaluated),
    }


def score_diarization_set(
    items: list[DiarItem], *, collar: float, skip_overlap: bool, protocol: str
) -> DiarResult:
    """Score a set of recordings, pooling via pyannote's own accumulators.

    One ``DiarizationErrorRate`` / ``JaccardErrorRate`` instance is fed every
    recording, so ``abs(metric)`` is the micro-average over the pooled duration
    components (not a mean of per-file rates, which would over-weight short
    files). Per-recording detailed components are captured for the report table.
    """
    _load_scoring()
    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    result = DiarResult(protocol=protocol, collar=collar, skip_overlap=skip_overlap)
    for item in items:
        uem = item.uem
        components = der(item.reference, item.hypothesis, uem=uem, detailed=True)
        file_jer = jer(item.reference, item.hypothesis, uem=uem)
        result.per_recording[item.recording_id] = {
            "der": float(components["diarization error rate"]),
            "jer": float(file_jer),
            "confusion_s": float(components["confusion"]),
            "missed_s": float(components["missed detection"]),
            "false_alarm_s": float(components["false alarm"]),
            "total_s": float(components["total"]),
            "uem_applied": item.uem is not None,
            **reference_metadata(item.reference, uem),
        }
    result.pooled_der = abs(der)
    result.pooled_jer = abs(jer)
    return result


# --------------------------------------------------------------------------- #
# ASR scoring (reuse the frozen bakeoff WER stack verbatim)
# --------------------------------------------------------------------------- #
def score_wer(items: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Pooled micro-average WER over ``(recording_id, ref_text, hyp_text)``.

    ``score_pooled`` normalizes raw reference and raw hypothesis together with
    the frozen Whisper normalizer, then pools integer edit counts.
    """
    _load_scoring()
    pooled = score_pooled(items)
    return {
        "pooled_wer": pooled.wer,
        "pooled_wer_pp": pooled.wer_pp,
        "substitutions": pooled.substitutions,
        "deletions": pooled.deletions,
        "insertions": pooled.insertions,
        "ref_words": pooled.ref_words,
        "per_recording": {
            s.name: {"wer": s.wer, "ref_words": s.ref_words} for s in pooled.files
        },
    }


# --------------------------------------------------------------------------- #
# Manifest-driven `score` command
# --------------------------------------------------------------------------- #
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"{path}: {exc.strerror or exc}") from exc


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvalError(f"{path}: {exc.strerror or exc}") from exc


def _decode(data: bytes, path: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvalError(f"{path}: not valid UTF-8: {exc}") from exc


def _observe(data: bytes) -> tuple[int, str]:
    """Byte length + sha256 of the exact buffer scored (accidental-drift guard)."""
    return len(data), hashlib.sha256(data).hexdigest()


def _resolve(base_dir: Path, path_str: str) -> Path:
    """Resolve a manifest entry path against the manifest directory.

    A relative path is taken relative to ``base_dir`` (the manifest's own
    directory) so a self-contained bundle scores wherever it is moved; an
    absolute path is honored unchanged.
    """
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


# A per-recording map of the OBSERVED bytes for cohort-input roles score can see:
# reference_rttm (always), uem (None when the entry scores the whole file), and
# wer_reference (filled in during WER load). None means an explicit null role.
Observed = dict[str, dict[str, tuple[int, str] | None]]


def _load_diar_items(
    entries: list[dict[str, Any]], manifest_dir: Path
) -> tuple[list[DiarItem], Observed]:
    items: list[DiarItem] = []
    observed: Observed = {}
    seen: set[str] = set()
    for entry in entries:
        rec_id = entry["recording_id"]
        if rec_id in seen:
            raise EvalError(f"duplicate diarization recording_id {rec_id!r}")
        seen.add(rec_id)
        # A UEM must be an explicit choice: a path, or null to score the whole
        # recording. Omitting the key would let pyannote silently approximate the
        # scoring region from the reference+hypothesis extents (only a runtime
        # warning), changing the protocol without the report showing it.
        if "uem" not in entry:
            raise EvalError(
                f"{rec_id}: diarization entry must set 'uem' explicitly "
                "(a UEM path, or null to score the whole recording)"
            )
        # Read reference and UEM as bytes ONCE, then hash and decode the SAME
        # buffer, so the cohort hash provably covers what was scored (not a
        # re-read that could bind different bytes). The hypothesis is an output,
        # never a cohort input, so it needs no observation.
        ref_bytes = _read_bytes(_resolve(manifest_dir, entry["reference_rttm"]))
        reference = parse_rttm(_decode(ref_bytes, entry["reference_rttm"]), rec_id)
        hypothesis = parse_rttm(_read(_resolve(manifest_dir, entry["hypothesis_rttm"])), rec_id)
        uem_path = entry["uem"]
        if uem_path is not None:
            uem_bytes = _read_bytes(_resolve(manifest_dir, uem_path))
            uem: Timeline | None = parse_uem(_decode(uem_bytes, uem_path), rec_id)
            uem_obs: tuple[int, str] | None = _observe(uem_bytes)
        else:
            uem = None
            uem_obs = None
        items.append(DiarItem(rec_id, reference, hypothesis, uem))
        observed[rec_id] = {"reference_rttm": _observe(ref_bytes), "uem": uem_obs}
    return items, observed


def _environment_manifest(
    manifest_bytes: bytes, diar_ids: list[str], wer_ids: list[str]
) -> dict[str, Any]:
    """Bind a report to exactly what produced it (the plan's env manifest).

    The scorer-side half: git sha (dirty-marked), scorer package versions, the
    frozen normalizer fingerprint, the manifest bytes hash, and the sorted
    scored cohort. The pipeline-side half (model-weight shas, GPU/driver/CUDA,
    decode params) is stamped by the ``run`` step that produces the hypotheses.
    """
    return {
        "git_sha": _git_sha(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "diarization_cohort": sorted(diar_ids),
        "wer_cohort": sorted(wer_ids),
        "scorer_versions": {
            "pyannote.metrics": _pkg_version("pyannote.metrics"),
            "pyannote.core": _pkg_version("pyannote.core"),
            "jiwer": _pkg_version("jiwer"),
        },
        "normalizer_version": NORMALIZER_VERSION,
        "normalizer_runtime": runtime_fingerprint(),
    }


# --------------------------------------------------------------------------- #
# Cohort binding: recompute the #97 cohort identity from the SCORED bytes
# --------------------------------------------------------------------------- #
# A `run`-produced manifest carries a `cohort` block attesting the inputs and the
# pipeline environment that produced the hypotheses. `score` does NOT trust it
# blindly: for every input role it can actually read (reference_rttm, uem,
# wer_reference) it derives the sha from the bytes it scored and rejects any
# disagreement with the attestation. Only the audio bytes (not read at score
# time) and the pipeline-environment identity remain pure `run` attestations.
# This is accidental-drift integrity for a single-operator harness, not hostile-
# caller provenance: without the audio or a signature a caller could fabricate
# the audio/environment records, which is acceptable here and documented, not
# defended against.


def _cohort_records(
    inputs: Any,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[int, str] | None]]:
    """Validate the cohort ``inputs`` array; return (split_by_id, records).

    ``records`` maps ``(id, role)`` to ``(byte_len, sha256)`` or ``None`` for an
    explicit null role. Enforces exactly the four :data:`COHORT_ROLES` per id,
    unique ``(id, role)`` pairs, one consistent split per id, coherent
    null/hash/length pairs, and non-null ``audio``/``reference_rttm`` records.
    """
    if not isinstance(inputs, list) or not inputs:
        raise EvalError("cohort.inputs must be a non-empty array")
    records: dict[tuple[str, str], tuple[int, str] | None] = {}
    split_by_id: dict[str, str] = {}
    roles_by_id: dict[str, set[str]] = {}
    allowed = {"id", "split", "role", "byte_len", "sha256"}
    for rec in inputs:
        if not isinstance(rec, dict):
            raise EvalError("cohort.inputs entries must be objects")
        extra = set(rec) - allowed
        if extra:
            raise EvalError(f"cohort input has unexpected keys: {sorted(extra)}")
        rid, role = rec.get("id"), rec.get("role")
        if not isinstance(rid, str) or not rid:
            raise EvalError(f"cohort input has a bad id: {rid!r}")
        if role not in COHORT_ROLES:
            raise EvalError(f"cohort input {rid!r} has unknown role {role!r}")
        split = rec.get("split")
        if not isinstance(split, str) or not split:
            raise EvalError(f"cohort input {rid}/{role} has a bad split: {split!r}")
        if rid in split_by_id and split_by_id[rid] != split:
            raise EvalError(f"cohort id {rid!r} has inconsistent splits")
        split_by_id[rid] = split
        if (rid, role) in records:
            raise EvalError(f"cohort has a duplicate {rid}/{role} record")
        byte_len, sha = rec.get("byte_len"), rec.get("sha256")
        where = f"cohort input {rid}/{role}"
        if (byte_len is None) != (sha is None):
            raise EvalError(f"{where}: byte_len and sha256 must both be set or both null")
        value: tuple[int, str] | None
        if byte_len is None:
            if role in ("audio", "reference_rttm"):
                raise EvalError(f"{where} must not be null")
            value = None
        else:
            if not isinstance(byte_len, int) or isinstance(byte_len, bool) or byte_len < 0:
                raise EvalError(f"{where}: byte_len must be a non-negative int")
            hexset = "0123456789abcdef"
            is_hex = isinstance(sha, str) and len(sha) == 64 and all(c in hexset for c in sha)
            if not is_hex:
                raise EvalError(f"{where}: sha256 must be 64 lowercase hex chars")
            value = (byte_len, sha)
        records[(rid, role)] = value
        roles_by_id.setdefault(rid, set()).add(role)
    for rid, roles in roles_by_id.items():
        if roles != set(COHORT_ROLES):
            missing = sorted(set(COHORT_ROLES) - roles)
            raise EvalError(f"cohort id {rid!r} must carry all four roles; missing {missing}")
    return split_by_id, records


def _role_input(rid: str, role: str, obs: tuple[int, str] | None) -> eval_run.CohortInput:
    """A ``CohortInput`` from an observed ``(byte_len, sha256)`` pair or a null role."""
    return eval_run.CohortInput(rid, role, obs[0] if obs else None, obs[1] if obs else None)


def _verify_observed(
    rid: str,
    role: str,
    observed: tuple[int, str] | None,
    records: dict[tuple[str, str], tuple[int, str] | None],
) -> None:
    """Reject a scored-byte observation that disagrees with its cohort record."""
    attested = records[(rid, role)]
    if observed != attested:
        raise EvalError(
            f"cohort input {rid}/{role} disagrees with the bytes scored "
            f"(cohort {attested}, scored {observed})"
        )


def _bind_cohort(
    cohort: dict[str, Any],
    diar_observed: Observed,
    wer_observed: dict[str, tuple[int, str]],
    diar_ids: list[str],
    wer_ids: list[str],
) -> dict[str, Any]:
    """Recompute the cohort identity from the scored bytes and return the stamp.

    Returns ``{"cohort_sha256", "corpus", "pipeline_environment"}`` to merge into
    the metrics ``environment``. Raises :class:`EvalError` on any disagreement
    between the manifest attestation and what was actually scored.
    """
    if not isinstance(cohort, dict):
        raise EvalError("manifest 'cohort' must be an object")
    corpus = cohort.get("corpus")
    if not isinstance(corpus, str) or not corpus:
        raise EvalError("cohort.corpus must be a non-empty string")
    pipeline_env = cohort.get("pipeline_environment")
    if not isinstance(pipeline_env, dict):
        raise EvalError("cohort.pipeline_environment must be an object")
    try:
        eval_run.validate_pipeline_environment(pipeline_env)
        env_hash = eval_run.pipeline_environment_hash(pipeline_env)
        split_by_id, records = _cohort_records(cohort.get("inputs"))
    except eval_run.RunError as exc:
        raise EvalError(str(exc)) from exc

    cohort_ids = set(split_by_id)
    if set(diar_ids) != cohort_ids:
        raise EvalError(
            f"scored diarization set {sorted(diar_ids)} does not equal the cohort "
            f"recording set {sorted(cohort_ids)}"
        )
    # WER is scored for exactly the recordings whose wer_reference role is non-null
    # (all AMI, no VoxConverse); an implicit subset is forbidden.
    expected_wer = {rid for rid in cohort_ids if records[(rid, "wer_reference")] is not None}
    if set(wer_ids) != expected_wer:
        raise EvalError(
            f"scored WER set {sorted(wer_ids)} does not equal the recordings with a "
            f"non-null wer_reference {sorted(expected_wer)}"
        )

    # Verify every locally observable role against the bytes scored, then build
    # the descriptor from those observed values (audio stays a pure attestation).
    inputs: list[eval_run.CohortInput] = []
    for rid in sorted(cohort_ids):
        ref_obs = diar_observed[rid]["reference_rttm"]
        _verify_observed(rid, "reference_rttm", ref_obs, records)
        uem_obs = diar_observed[rid]["uem"]
        _verify_observed(rid, "uem", uem_obs, records)
        wer_obs = wer_observed.get(rid)
        _verify_observed(rid, "wer_reference", wer_obs, records)
        audio = records[(rid, "audio")]
        assert audio is not None and ref_obs is not None  # enforced above
        inputs += [
            _role_input(rid, "audio", audio),
            _role_input(rid, "reference_rttm", ref_obs),
            _role_input(rid, "uem", uem_obs),
            _role_input(rid, "wer_reference", wer_obs),
        ]

    try:
        descriptor = eval_run.cohort_descriptor(
            corpus, split_by_id, inputs, env_hash, HARNESS_PROTOCOL
        )
        cohort_hash = eval_run.cohort_sha256(descriptor)
    except eval_run.RunError as exc:
        raise EvalError(str(exc)) from exc

    claimed = cohort.get("cohort_sha256")
    if claimed is not None and claimed != cohort_hash:
        raise EvalError(
            f"cohort.cohort_sha256 {claimed!r} does not match the recompute {cohort_hash!r}"
        )
    return {"cohort_sha256": cohort_hash, "corpus": corpus, "pipeline_environment": pipeline_env}


def cmd_score(args: argparse.Namespace) -> int:
    _load_scoring()
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # Manifest entry paths resolve against the MANIFEST'S directory, not the
    # process CWD, so a self-contained bundle (the `run` step copies the
    # references + hypotheses beside the manifest and writes relative paths)
    # scores identically wherever it is moved — host, container, or a clean
    # checkout. An absolute path in a manifest is honored as-is.
    manifest_dir = manifest_path.resolve().parent
    diar_entries = manifest.get("diarization", [])
    wer_entries = manifest.get("wer", [])
    if not diar_entries and not wer_entries:
        raise EvalError(f"{args.manifest}: manifest has no 'diarization' or 'wer' entries")

    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": "eval_quality_report"}

    diar_ids = [e["recording_id"] for e in diar_entries]
    wer_ids = [e["recording_id"] for e in wer_entries]
    diar_observed: Observed = {}
    wer_observed: dict[str, tuple[int, str]] = {}

    if diar_entries:
        items, diar_observed = _load_diar_items(diar_entries, manifest_dir)
        strict = score_diarization_set(items, protocol="strict", **STRICT)
        diagnostic = score_diarization_set(items, protocol="diagnostic", **DIAGNOSTIC)
        report["diarization"] = {
            "jer_mapping": JER_MAPPING,
            "strict": _diar_json(strict),
            "diagnostic": _diar_json(diagnostic),
        }

    if wer_entries:
        if len(set(wer_ids)) != len(wer_ids):
            raise EvalError("duplicate recording_id in 'wer' entries")
        triples: list[tuple[str, str, str]] = []
        for e in wer_entries:
            rid = e["recording_id"]
            # Read the WER reference once, then hash and decode the same buffer.
            ref_bytes = _read_bytes(_resolve(manifest_dir, e["reference_text"]))
            ref_text = _decode(ref_bytes, e["reference_text"])
            triples.append((rid, ref_text, _read(_resolve(manifest_dir, e["hypothesis_text"]))))
            wer_observed[rid] = _observe(ref_bytes)
        report["wer"] = score_wer(triples)

    environment = _environment_manifest(manifest_bytes, diar_ids, wer_ids)
    # A `run`-produced manifest carries a cohort block; recompute its identity
    # from the scored bytes and stamp it so `report` can prove the run is a
    # zero-change peer of another. A cohort-less manifest keeps today's shape
    # (no cohort_sha256), and `report` will refuse to render it (fail closed).
    if "cohort" in manifest:
        environment.update(
            _bind_cohort(manifest["cohort"], diar_observed, wer_observed, diar_ids, wer_ids)
        )
    report["environment"] = environment

    out = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)
    return 0


def _diar_json(result: DiarResult) -> dict[str, Any]:
    # pooled_der is a duration micro-average; global_jer is reference-speaker
    # weighted (sum of per-speaker Jaccard errors / total reference speakers) —
    # they aggregate differently, so they are named differently on purpose. See
    # JER_MAPPING for why the JER absolute is delta-only.
    return {
        "collar": result.collar,
        "skip_overlap": result.skip_overlap,
        "pooled_der": result.pooled_der,
        "global_jer": result.pooled_jer,
        "per_recording": result.per_recording,
    }


# --------------------------------------------------------------------------- #
# `report` command: metrics JSON -> dated Markdown (house style, pure function)
# --------------------------------------------------------------------------- #
def _pct(fraction: float) -> str:
    """A rate in [0, 1] as a percentage string (DER/JER/WER are fractions)."""
    return f"{fraction * 100:.2f}%"


def _pp(delta_fraction: float) -> str:
    """A difference of two rate fractions as percentage points."""
    return f"{delta_fraction * 100:.2f} pp"


def _dur(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _spread(values: list[float]) -> float:
    """The max-minus-min spread; the conservative zero-change noise band."""
    return (max(values) - min(values)) if values else 0.0


def _strict(run: dict[str, Any]) -> dict[str, Any]:
    return run["diarization"]["strict"]


def _diagnostic(run: dict[str, Any]) -> dict[str, Any]:
    return run["diarization"]["diagnostic"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _corpus_recordings(runs: list[dict[str, Any]]) -> list[str]:
    """The corpus recording set. After :func:`_validate_corpus_runs` every run
    shares an identical diarization set, so the first run's ids are authoritative
    (no permissive intersection that could hide a dropped or invented row)."""
    return sorted(_strict(runs[0])["per_recording"])


def _scorer_identity(env: dict[str, Any]) -> tuple[str, str, str]:
    """The metric-determining scorer identity (versions + normalizer).

    ``git_sha`` is deliberately excluded: a pure-refactor harness commit changes
    it without moving any number, so a git-sha mix is a header WARNING, not a
    fatal error. What actually determines the metric is the scorer package
    versions and the frozen normalizer, so THOSE must match across a noise floor.
    """
    return (
        json.dumps(env.get("scorer_versions", {}), sort_keys=True),
        env.get("normalizer_version", ""),
        env.get("normalizer_runtime", ""),
    )


def _validate_corpus_runs(corpus: str, runs: list[dict[str, Any]]) -> None:
    """Fail closed unless the runs are genuinely comparable (the plan's guard).

    Every run of a corpus must be cohort-bound, carry the same cohort identity,
    be labelled for this corpus, and expose identical DER and WER recording sets;
    a multi-run noise floor must additionally share one scorer identity. Only
    then is calling the max spread a zero-change band an honest claim.
    """
    hashes: set[str] = set()
    diar_sets: set[frozenset[str]] = set()
    wer_flags: set[bool] = set()
    wer_sets: set[frozenset[str]] = set()
    for run in runs:
        env = run.get("environment", {})
        cohort_hash = env.get("cohort_sha256")
        if not cohort_hash:
            raise EvalError(
                f"{corpus}: a scored run is not cohort-bound (no environment.cohort_sha256). "
                "report requires cohort identity; re-score from a run-produced manifest."
            )
        hashes.add(cohort_hash)
        run_corpus = env.get("corpus")
        if run_corpus != corpus:
            raise EvalError(
                f"{corpus}: a run declares environment.corpus {run_corpus!r} but was supplied "
                f"as --run {corpus}=... (corpus label mismatch)"
            )
        strict_ids = set(_strict(run)["per_recording"])
        diag_ids = set(_diagnostic(run)["per_recording"])
        env_ids = set(env.get("diarization_cohort", []))
        if not strict_ids == diag_ids == env_ids:
            raise EvalError(f"{corpus}: a run's strict/diagnostic/cohort diarization sets disagree")
        diar_sets.add(frozenset(strict_ids))
        has_wer = "wer" in run
        wer_flags.add(has_wer)
        wset = set(run["wer"]["per_recording"]) if has_wer else set()
        if has_wer and wset != set(env.get("wer_cohort", [])):
            raise EvalError(f"{corpus}: a run's WER set disagrees with environment.wer_cohort")
        wer_sets.add(frozenset(wset))
    if len(hashes) != 1:
        raise EvalError(f"{corpus}: runs do not share one cohort_sha256 (not a zero-change set)")
    if len(diar_sets) != 1:
        raise EvalError(f"{corpus}: runs have different diarization recording sets")
    if len(wer_flags) != 1:
        raise EvalError(f"{corpus}: some runs carry WER and some do not")
    if len(wer_sets) != 1:
        raise EvalError(f"{corpus}: runs have different WER recording sets")
    if len(runs) > 1 and len({_scorer_identity(r.get("environment", {})) for r in runs}) != 1:
        raise EvalError(
            f"{corpus}: a multi-run noise floor mixes scorer versions or normalizer identity; "
            "the spread would not be a clean zero-change band"
        )


def _report_header(date: str, runs: list[dict[str, Any]]) -> list[str]:
    envs = [r.get("environment", {}) for r in runs]
    shas = sorted({e.get("git_sha", "unknown") for e in envs})
    manifests = sorted(
        {e.get("manifest_sha256", "")[:12] for e in envs if e.get("manifest_sha256")}
    )
    pmvers = sorted({e.get("scorer_versions", {}).get("pyannote.metrics", "?") for e in envs})
    normvers = sorted({e.get("normalizer_version", "?") for e in envs})
    sha_txt = shas[0] if len(shas) == 1 else ", ".join(shas)
    lines = [
        f"> Eval-quality baseline. Generated {date}. "
        f"Pipeline git sha `{sha_txt}`, Voxint {_pkg_version('voxint')}.",
        f"> Scorer pyannote.metrics {', '.join(pmvers)}, "
        f"Whisper normalizer {', '.join(normvers)}. "
        f"Corpus manifest sha `{', '.join(manifests) or 'n/a'}`.",
    ]
    if len(shas) > 1:
        lines.append(
            "> Warning: the scored runs do not share one pipeline git sha, so the "
            "noise band below mixes code states and is not a clean zero-change floor."
        )
    return lines


def _corpus_section(corpus: str, runs: list[dict[str, Any]]) -> list[str]:
    has_wer = all("wer" in r for r in runs)
    n = len(runs)
    strict_der = [_strict(r)["pooled_der"] for r in runs]
    diag_der = [_diagnostic(r)["pooled_der"] for r in runs]
    strict_jer = [_strict(r)["global_jer"] for r in runs]
    diag_jer = [_diagnostic(r)["global_jer"] for r in runs]
    wer = [r["wer"]["pooled_wer"] for r in runs] if has_wer else []

    lines = [f"## {corpus}", ""]
    lines.append(
        f"Runs scored: {n}."
        if n > 1
        else "Runs scored: 1 (single run, no noise band)."
    )
    lines += [
        "",
        "| metric | strict | diagnostic |",
        "| --- | --- | --- |",
        f"| Pooled DER | {_pct(_mean(strict_der))} | {_pct(_mean(diag_der))} |",
        f"| Global JER | {_pct(_mean(strict_jer))} | {_pct(_mean(diag_jer))} |",
    ]
    if has_wer:
        lines.append(f"| Pooled WER | {_pct(_mean(wer))} | n/a |")
    lines.append("")

    if n > 1:
        band = [
            f"pooled DER {_pp(_spread(strict_der))}",
            f"global JER {_pp(_spread(strict_jer))}",
        ]
        if has_wer:
            band.append(f"pooled WER {_pp(_spread(wer))}")
        lines += [
            f"### Noise floor ({n} zero-change runs)",
            "",
            "Pooled max spread (strict protocol): " + ", ".join(band) + ".",
            "",
        ]
        worst = _worst_file_spread(corpus, runs, has_wer)
        if worst:
            lines += [worst, ""]

    lines += _per_recording_table(runs, has_wer)
    return lines


def _worst_file_spread(corpus: str, runs: list[dict[str, Any]], has_wer: bool) -> str:
    recs = _corpus_recordings(runs)
    der_worst = ("", 0.0)
    for rec in recs:
        spread = _spread([_strict(r)["per_recording"][rec]["der"] for r in runs])
        if spread > der_worst[1]:
            der_worst = (rec, spread)
    parts = [f"DER {der_worst[0]} {_pp(der_worst[1])}"] if der_worst[0] else []
    if has_wer:
        wer_worst = ("", 0.0)
        for rec in recs:
            vals = [r["wer"]["per_recording"].get(rec, {}).get("wer") for r in runs]
            vals = [v for v in vals if v is not None]
            if len(vals) == len(runs):
                spread = _spread(vals)
                if spread > wer_worst[1]:
                    wer_worst = (rec, spread)
        if wer_worst[0]:
            parts.append(f"WER {wer_worst[0]} {_pp(wer_worst[1])}")
    return ("Worst single-file spread: " + "; ".join(parts) + ".") if parts else ""


def _per_recording_table(runs: list[dict[str, Any]], has_wer: bool) -> list[str]:
    recs = _corpus_recordings(runs)
    header = "| recording | evaluated | speakers | ref overlap | strict DER | strict JER |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    if has_wer:
        header += " WER |"
        rule += " --- |"
    lines = ["Per recording:", "", header, rule]
    for rec in recs:
        per = [_strict(r)["per_recording"][rec] for r in runs]
        dur = _mean([p["evaluated_s"] for p in per])
        speakers = round(_mean([p["speaker_count"] for p in per]))
        overlap = _mean([p["reference_overlap_pct"] for p in per])
        der = _mean([p["der"] for p in per])
        jer = _mean([p["jer"] for p in per])
        row = (
            f"| {rec} | {_dur(dur)} | {speakers} | {overlap:.1f}% "
            f"| {_pct(der)} | {_pct(jer)} |"
        )
        if has_wer:
            wvals = [r["wer"]["per_recording"].get(rec, {}).get("wer") for r in runs]
            wvals = [v for v in wvals if v is not None]
            row += f" {_pct(_mean(wvals))} |" if wvals else " n/a |"
        lines.append(row)
    lines.append("")
    return lines


def render_report(date: str, runs_by_corpus: dict[str, list[dict[str, Any]]]) -> str:
    """Render one or more per-corpus metrics reports into a house-style doc.

    Each corpus is a separate section with its own pooled numbers; there is
    deliberately no combined figure. When a corpus carries K > 1 runs the
    zero-change noise band (max spread per metric) is rendered so downstream
    thresholds can be measured, not guessed.
    """
    all_runs = [run for runs in runs_by_corpus.values() for run in runs]
    if not all_runs:
        raise EvalError("report: no runs supplied")
    lines = _report_header(date, all_runs)
    lines += [
        "",
        f"# Eval-quality baseline ({date})",
        "",
        "This is a maintainer tripwire, not a benchmark. The AMI and VoxConverse "
        "subsets are a known-condition regression corpus, not a held-out "
        "generalization set: both corpora informed the pyannote 3.1 tuning that "
        "Voxint pins, so these numbers catch gross pipeline breakage and gate the "
        "single-GPU memory change (issue #96). They are not a claim about accuracy "
        "on unseen recordings. Each corpus is scored separately; there is "
        "deliberately no combined AMI plus VoxConverse figure.",
        "",
        "The strict protocol (collar 0.0, overlap scored) is the gate. The "
        "diagnostic protocol (collar 0.5, overlap skipped) is a forgiving "
        "cross-check only. JER uses the pyannote 4.1 co-occurrence mapping and is a "
        "delta-only signal, never a DIHARD-comparable absolute; DER is the primary "
        "metric.",
        "",
    ]
    for corpus in sorted(runs_by_corpus):
        runs = runs_by_corpus[corpus]
        _validate_corpus_runs(corpus, runs)
        lines += _corpus_section(corpus, runs)
    return "\n".join(lines).rstrip() + "\n"


def _parse_run_specs(specs: list[str]) -> dict[str, list[dict[str, Any]]]:
    runs_by_corpus: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise EvalError(f"--run expects CORPUS=path.json, got {spec!r}")
        corpus, _, path = spec.partition("=")
        corpus = corpus.strip()
        if not corpus:
            raise EvalError(f"--run has an empty corpus label: {spec!r}")
        data = json.loads(_read(Path(path)))
        if data.get("kind") != "eval_quality_report":
            raise EvalError(f"{path}: not an eval_quality_report metrics JSON")
        if "diarization" not in data:
            raise EvalError(f"{path}: metrics JSON has no 'diarization' block to report")
        runs_by_corpus.setdefault(corpus, []).append(data)
    return runs_by_corpus


def cmd_report(args: argparse.Namespace) -> int:
    runs_by_corpus = _parse_run_specs(args.run)
    text = render_report(args.date, runs_by_corpus)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------- #
# `run` command: live driver (submit -> poll -> read DB -> export a bundle)
# --------------------------------------------------------------------------- #
# The driver imports NO pyannote/jiwer (Step 0): it submits the subset, polls the
# DB, exports the relabelled hypothesis RTTM + AMI WER text, and writes a
# SELF-CONTAINED score bundle (references + hypotheses copied in, manifest paths
# relative to the manifest). ``score`` consumes that bundle later, in the parity
# lane. All live seams (voxint.db / voxint.export / voxint.ingest / the CLI
# publish path) are imported lazily inside ``cmd_run`` so ``import eval_quality``
# and this subcommand load without a scoring stack installed.

# Bundle layout under --out-dir: copied ground-truth inputs and exported
# hypotheses live in fixed subdirectories so the manifest can reference them by
# relative path and the whole directory is portable (host <-> container).
BUNDLE_INPUTS = "inputs"
BUNDLE_HYPS = "hypotheses"
JOURNAL_NAME = "journal.json"

# The container/worker images whose digests fingerprint the pipeline identity.
FINGERPRINT_CONTAINERS = ("api", "worker", "whisper", "pyannote", "titanet")


def _load_driver() -> Any:
    """Import the live voxint seams the driver needs (lazy; no scoring stack)."""
    from types import SimpleNamespace

    from sqlalchemy import select

    from voxint.cli import _publish_or_defer
    from voxint.config import get_settings
    from voxint.db.models import DiarizationTurn, MediaItem, PipelineRun, TranscriptSegment
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.export import to_rttm
    from voxint.ingest.service import submit_media_item_if_new

    return SimpleNamespace(
        select=select,
        publish=_publish_or_defer,
        get_settings=get_settings,
        DiarizationTurn=DiarizationTurn,
        MediaItem=MediaItem,
        PipelineRun=PipelineRun,
        TranscriptSegment=TranscriptSegment,
        build_engine=build_engine,
        build_session_factory=build_session_factory,
        session_scope=session_scope,
        to_rttm=to_rttm,
        submit_media_item_if_new=submit_media_item_if_new,
    )


# --------------------------------------------------------------------------- #
# Pure export shaping (testable with duck-typed rows; no worker, no pyannote)
# --------------------------------------------------------------------------- #
def assert_monotonic_unique(indices: list[int], what: str) -> None:
    """Reject a non-monotonic or duplicated index sequence (ordering guard)."""
    seen: set[int] = set()
    prev: int | None = None
    for idx in indices:
        if idx in seen:
            raise EvalError(f"{what}: duplicate index {idx}")
        if prev is not None and idx < prev:
            raise EvalError(f"{what}: index {idx} is out of order after {prev}")
        seen.add(idx)
        prev = idx


def segments_to_word_lists(segments: list[Any]) -> list[list[dict[str, Any]]]:
    """Ordered per-segment word lists for the AMI WER hypothesis (strict nulls).

    ``segments`` are ``TranscriptSegment`` rows ALREADY ordered by
    ``segment_index`` (the caller supplies the DB order). A segment whose
    ``words`` is NULL but whose ``raw_text`` is non-empty is a hard error: the
    run lost its word timing, and silently treating it as ``[]`` would fabricate
    a perfect-empty hypothesis and move WER. A genuinely empty segment (no
    ``raw_text``) contributes an empty list.
    """
    assert_monotonic_unique([s.segment_index for s in segments], "transcript segment_index")
    out: list[list[dict[str, Any]]] = []
    for s in segments:
        if s.words is None:
            if (s.raw_text or "").strip():
                raise EvalError(
                    f"transcript segment {s.segment_index} has non-empty text but NULL word "
                    "timing; cannot build a WER hypothesis (word timestamps missing)"
                )
            out.append([])
        else:
            out.append(list(s.words))
    return out


# --------------------------------------------------------------------------- #
# Cohort observation + self-contained bundle (pure file IO; round-trippable)
# --------------------------------------------------------------------------- #
def observe_role_files(resolved: list[Any]) -> dict[str, dict[str, tuple[int, str] | None]]:
    """Hash the four cohort role files per recording (explicit null for absent)."""
    observed: dict[str, dict[str, tuple[int, str] | None]] = {}
    for r in resolved:
        observed[r.recording_id] = {
            "audio": _observe(_read_bytes(r.audio)),
            "reference_rttm": _observe(_read_bytes(r.reference_rttm)),
            "uem": _observe(_read_bytes(r.uem)) if r.uem is not None else None,
            "wer_reference": (
                _observe(_read_bytes(r.wer_reference)) if r.wer_reference is not None else None
            ),
        }
    return observed


def build_cohort_inputs(
    resolved: list[Any], observed: dict[str, dict[str, tuple[int, str] | None]]
) -> tuple[dict[str, str], list[eval_run.CohortInput]]:
    """The single (split_by_id, CohortInput[]) builder reused for journal + manifest."""
    split_by_id = {r.recording_id: r.split for r in resolved}
    inputs: list[eval_run.CohortInput] = []
    for r in resolved:
        roles = observed[r.recording_id]
        for role in COHORT_ROLES:
            inputs.append(_role_input(r.recording_id, role, roles[role]))
    return split_by_id, inputs


def _copy_bytes(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(_read_bytes(src))


def write_bundle(
    out_dir: Path,
    resolved: list[Any],
    hyp_rttm_by_id: dict[str, str],
    wer_hyp_by_id: dict[str, str],
    cohort_block: dict[str, Any],
) -> Path:
    """Write a self-contained score bundle; return the manifest path.

    Copies each recording's reference RTTM/UEM (and WER reference for AMI) into
    ``<out_dir>/inputs`` and writes the exported hypothesis RTTM (and AMI WER
    text) into ``<out_dir>/hypotheses``, then emits ``manifest.json`` with paths
    RELATIVE to the manifest directory. ``score`` (which now resolves relative
    paths against the manifest's directory) can therefore score the bundle
    wherever it is moved.
    """
    out_dir = Path(out_dir)
    diar_entries: list[dict[str, Any]] = []
    wer_entries: list[dict[str, Any]] = []
    for r in resolved:
        rid = r.recording_id
        ref_rel = f"{BUNDLE_INPUTS}/{rid}.reference.rttm"
        hyp_rel = f"{BUNDLE_HYPS}/{rid}.hypothesis.rttm"
        _copy_bytes(r.reference_rttm, out_dir / ref_rel)
        (out_dir / hyp_rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / hyp_rel).write_text(hyp_rttm_by_id[rid], encoding="utf-8")
        if r.uem is not None:
            uem_rel: str | None = f"{BUNDLE_INPUTS}/{rid}.uem"
            _copy_bytes(r.uem, out_dir / uem_rel)
        else:
            uem_rel = None
        diar_entries.append(
            {
                "recording_id": rid,
                "reference_rttm": ref_rel,
                "hypothesis_rttm": hyp_rel,
                "uem": uem_rel,
            }
        )
        if r.wer_reference is not None:
            wref_rel = f"{BUNDLE_INPUTS}/{rid}.wer_reference.txt"
            whyp_rel = f"{BUNDLE_HYPS}/{rid}.wer_hypothesis.txt"
            _copy_bytes(r.wer_reference, out_dir / wref_rel)
            (out_dir / whyp_rel).write_text(wer_hyp_by_id[rid], encoding="utf-8")
            wer_entries.append(
                {"recording_id": rid, "reference_text": wref_rel, "hypothesis_text": whyp_rel}
            )
    manifest: dict[str, Any] = {"diarization": diar_entries, "cohort": cohort_block}
    if wer_entries:
        manifest["wer"] = wer_entries
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return manifest_path


# --------------------------------------------------------------------------- #
# Host-side environment fingerprint (probe shells out; refusal logic is pure)
# --------------------------------------------------------------------------- #
def _probe(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip()


def probe_fingerprint(container_prefix: str, cuda_visible: str | None) -> dict[str, Any]:
    """Probe the running deploy host-side (docker image digests + nvidia-smi).

    Best-effort: any leg that cannot be observed (docker/nvidia-smi absent, a
    container not running) is recorded as ``probe_failed`` and drops the mode to
    ``degraded``. A ``degraded`` fingerprint never yields a score-ready manifest
    (the refusal is in :func:`require_verified_fingerprints`), so a partial probe
    fails closed rather than silently attesting an unknown environment.
    """
    images: dict[str, str | None] = {}
    status: dict[str, str] = {}
    for svc in FINGERPRINT_CONTAINERS:
        name = f"{container_prefix}-{svc}-1"
        digest = _probe(["docker", "inspect", "--format", "{{.Image}}", name])
        images[svc] = digest
        status[svc] = "observed" if digest else "probe_failed"
    gpu_raw = _probe(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]
    )
    if gpu_raw:
        first = gpu_raw.splitlines()[0].split(",")
        gpu: dict[str, str] | None = {
            "name": first[0].strip(),
            "uuid": first[1].strip() if len(first) > 1 else "",
            "driver": first[2].strip() if len(first) > 2 else "",
        }
        status["gpu"] = "observed"
    else:
        gpu = None
        status["gpu"] = "probe_failed"
    mode = "full" if all(v == "observed" for v in status.values()) else "degraded"
    return {
        "mode": mode,
        "images": images,
        "gpu": gpu,
        "cuda_visible_devices": cuda_visible,
        "probe_status": status,
    }


def require_verified_fingerprints(
    before: dict[str, Any], after: dict[str, Any], static_env: dict[str, Any]
) -> None:
    """Refuse a manifest unless the environment is fully, consistently observed.

    Fails closed on: a degraded probe (something unobservable), a mid-batch
    change (``before`` != ``after``), or a live probe that disagrees with the
    static ``--pipeline-env`` identity (the running whisper image digest must
    equal the static ``code.image_digest``, and the live GPU name the static
    ``gpu.name``). Passing means the run's hypotheses were produced under exactly
    the attested identity.
    """
    if before.get("mode") != "full":
        raise EvalError(f"environment fingerprint is degraded before the batch: {before}")
    if after.get("mode") != "full":
        raise EvalError(f"environment fingerprint is degraded after the batch: {after}")
    if before != after:
        raise EvalError("environment fingerprint changed during the batch (before != after)")
    static_digest = static_env.get("code", {}).get("image_digest")
    live_digest = before.get("images", {}).get("whisper")
    if static_digest and live_digest and static_digest != live_digest:
        raise EvalError(
            f"static pipeline-env image_digest {static_digest!r} disagrees with the running "
            f"whisper image {live_digest!r}"
        )
    static_gpu = static_env.get("gpu", {}).get("name")
    live_gpu = (before.get("gpu") or {}).get("name")
    if static_gpu and live_gpu and static_gpu != live_gpu:
        raise EvalError(
            f"static pipeline-env gpu.name {static_gpu!r} disagrees with the live GPU {live_gpu!r}"
        )


# --------------------------------------------------------------------------- #
# The live orchestration (crash-safe; validated on the maintainer host, worker idle)
# --------------------------------------------------------------------------- #
def _load_subset_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(_read(path))
    if isinstance(data, dict):
        # The corpus-tooling subset wraps the entries under "files"; accept the
        # legacy "items" spelling too. A bare array is also fine.
        for key in ("items", "files"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise EvalError(
            f"{path}: subset must be a JSON array or an object with an 'items' or 'files' array"
        )
    return data


def _staged_source_path(media_subdir: str, batch_id: str, recording_id: str) -> str:
    """The MEDIA_ROOT-relative, unique-per-(batch, recording) staged path."""
    return f"{media_subdir.strip('/')}/{batch_id}/{recording_id}.wav"


def _reconcile_run(driver: Any, session: Any, staged_rel: str) -> Any:
    """Adopt the single run for a staged source_path (the idempotency join)."""
    rows = (
        session.execute(
            driver.select(driver.PipelineRun)
            .join(driver.MediaItem, driver.PipelineRun.media_item_id == driver.MediaItem.id)
            .where(driver.MediaItem.source_path == staged_rel)
        )
        .scalars()
        .all()
    )
    if len(rows) != 1:
        raise EvalError(
            f"reconcile for {staged_rel!r} found {len(rows)} runs (expected exactly 1); "
            "refusing to guess"
        )
    return rows[0]


def _export_completed(
    driver: Any, session: Any, run_id: Any, resolved: Any
) -> tuple[str, str | None]:
    """Read a completed run's DB rows and return (hypothesis_rttm, wer_text|None)."""
    turns = (
        session.execute(
            driver.select(driver.DiarizationTurn)
            .where(driver.DiarizationTurn.pipeline_run_id == run_id)
            .order_by(driver.DiarizationTurn.turn_index)
        )
        .scalars()
        .all()
    )
    assert_monotonic_unique([t.turn_index for t in turns], "diarization turn_index")
    hyp_rttm = driver.to_rttm(turns, resolved.recording_id)

    wer_text: str | None = None
    if resolved.wer_reference is not None:
        segments = (
            session.execute(
                driver.select(driver.TranscriptSegment)
                .where(driver.TranscriptSegment.pipeline_run_id == run_id)
                .order_by(driver.TranscriptSegment.segment_index)
            )
            .scalars()
            .all()
        )
        word_lists = segments_to_word_lists(segments)
        uem_text = _read(resolved.uem)
        uem_regions = _uem_regions_us(uem_text, resolved.recording_id)
        wer_text = eval_run.ami_hypothesis_text(word_lists, uem_regions)
    return hyp_rttm, wer_text


def _uem_regions_us(text: str, recording_id: str) -> list[tuple[int, int]]:
    """UEM regions in integer microseconds (same _us rule as the reference)."""
    import prepare_bakeoff_corpus as bake

    regions: list[tuple[int, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0] != recording_id:
            continue
        regions.append((bake._us(float(parts[2])), bake._us(float(parts[3]))))
    if not regions:
        raise EvalError(f"UEM has no region for {recording_id!r}")
    return regions


def cmd_run(args: argparse.Namespace) -> int:
    import uuid as _uuid

    driver = _load_driver()
    corpus = args.corpus
    corpus_root = Path(args.corpus_root or _require_env("EVAL_CORPUS_ROOT", args.corpus_root))
    database_url = args.database_url or _require_env("DATABASE_URL", args.database_url)
    batch_id = args.batch_id or _uuid.uuid4().hex
    media_root = Path(args.media_root) if args.media_root else driver.get_settings().media_root
    tol = float(args.duration_tol)

    items = eval_run.load_subset(_load_subset_entries(Path(args.subset)), corpus)
    only = [s for s in args.only.split(",") if s] if args.only else None
    items = eval_run.select_only(items, only)
    resolved = [eval_run.resolve_item(corpus_root, it) for it in items]
    by_id = {r.recording_id: r for r in resolved}

    # Preflight every recording BEFORE any submission (catches a truncated file).
    for it, r in zip(items, resolved, strict=True):
        measured = eval_run.measure_wav_seconds(r.audio)
        ref_end = eval_run.rttm_max_end_seconds(_read(r.reference_rttm))
        uem_end = (
            eval_run.uem_max_end_seconds(_read(r.uem), r.recording_id)
            if r.uem is not None
            else None
        )
        problems = eval_run.check_duration(measured, it.extent_s, ref_end, uem_end, tol)
        if problems:
            raise EvalError(f"{r.recording_id}: preflight failed: {'; '.join(problems)}")

    pipeline_env = eval_run.validate_pipeline_environment(
        json.loads(_read(Path(args.pipeline_env)))
    )

    observed = observe_role_files(resolved)
    split_by_id, cohort_inputs = build_cohort_inputs(resolved, observed)
    descriptor = eval_run.cohort_descriptor(
        corpus,
        split_by_id,
        cohort_inputs,
        eval_run.pipeline_environment_hash(pipeline_env),
        HARNESS_PROTOCOL,
    )
    cohort_hash = eval_run.cohort_sha256(descriptor)

    fp_before = probe_fingerprint(args.container_prefix, args.cuda_visible_devices)

    out_dir = Path(args.out_dir)
    with eval_run.out_dir_lock(out_dir):
        journal = _load_or_init_journal(out_dir, corpus, cohort_hash, pipeline_env, batch_id)
        # A non-empty journal must be honored ONLY under an explicit --resume, so a
        # re-run into a stale out-dir cannot silently SKIP_DONE into a fake
        # zero-change pass (each noise-floor pass gets a fresh out-dir + batch-id).
        if journal.get("items") and not args.resume:
            raise EvalError(
                f"{out_dir} already holds a journal with results; pass --resume to continue it "
                "or use a fresh --out-dir (noise-floor passes must not share an out-dir)"
            )
        decisions = eval_run.plan_resume(
            journal, [r.recording_id for r in resolved],
            resume=args.resume, retry_failed=args.retry_failed,
        )
        for decision in decisions:
            _drive_one(
                driver, journal, out_dir, by_id[decision.recording_id], decision,
                database_url=database_url, media_root=media_root,
                media_subdir=args.media_subdir, batch_id=batch_id,
                interval=float(args.interval), timeout=float(args.timeout),
            )

        fp_after = probe_fingerprint(args.container_prefix, args.cuda_visible_devices)
        require_verified_fingerprints(fp_before, fp_after, pipeline_env)

        # All-or-nothing: score needs the full cohort, so refuse a partial bundle.
        missing = [r.recording_id for r in resolved
                   if (journal["items"].get(r.recording_id) or {}).get("status") != "completed"]
        if missing:
            raise EvalError(f"cannot write a scoreable bundle; not completed: {sorted(missing)}")

        hyp_rttm_by_id = {rid: journal["items"][rid]["hypothesis_rttm"] for rid in by_id}
        wer_hyp_by_id = {
            rid: journal["items"][rid]["wer_text"]
            for rid in by_id
            if by_id[rid].wer_reference is not None
        }
        cohort_block = {
            "schema_version": eval_run.COHORT_SCHEMA_VERSION,
            "corpus": corpus,
            "pipeline_environment": pipeline_env,
            "inputs": descriptor["inputs"],
            "cohort_sha256": cohort_hash,
            "batch_id": batch_id,
        }
        manifest_path = write_bundle(out_dir, resolved, hyp_rttm_by_id, wer_hyp_by_id, cohort_block)
    print(manifest_path)
    return 0


def _require_env(name: str, override: str | None) -> str:
    import os

    if override:
        return override
    value = os.environ.get(name)
    if not value:
        raise EvalError(f"{name} is required (pass the flag or set the environment variable)")
    return value


def _journal_path(out_dir: Path) -> Path:
    return Path(out_dir) / JOURNAL_NAME


def _load_or_init_journal(
    out_dir: Path, corpus: str, cohort_hash: str, pipeline_env: dict[str, Any], batch_id: str
) -> dict[str, Any]:
    path = _journal_path(out_dir)
    if path.exists():
        journal = json.loads(_read(path))
        try:
            eval_run.validate_journal(journal, corpus=corpus, cohort_hash=cohort_hash)
        except eval_run.RunError as exc:
            raise EvalError(str(exc)) from exc
        return journal
    journal = eval_run.new_journal(corpus, cohort_hash, pipeline_env)
    journal["batch_id"] = batch_id
    eval_run.write_json_atomic(path, journal)
    return journal


def _record(out_dir: Path, journal: dict[str, Any], rid: str, patch: dict[str, Any]) -> None:
    """Merge a patch into one journal item and durably persist (write-ahead)."""
    item = journal["items"].setdefault(rid, {})
    item.update(patch)
    eval_run.write_json_atomic(_journal_path(out_dir), journal)


def _drive_one(
    driver: Any, journal: dict[str, Any], out_dir: Path, resolved: Any,
    decision: Any, *, database_url: str, media_root: Path, media_subdir: str,
    batch_id: str, interval: float, timeout: float,
) -> None:
    """Submit/reconcile/poll/export ONE recording, crash-safe by the journal."""
    import time

    rid = resolved.recording_id
    if decision.action == eval_run.ACTION_SKIP_DONE:
        return
    if decision.action == eval_run.ACTION_STOP:
        raise EvalError(f"{rid}: {decision.reason}")

    staged_rel = _staged_source_path(media_subdir, batch_id, rid)
    engine = driver.build_engine(database_url)
    try:
        factory = driver.build_session_factory(engine)
        run_uuid = (journal["items"].get(rid) or {}).get("run_uuid")

        if decision.action in (eval_run.ACTION_SUBMIT, eval_run.ACTION_RETRY):
            audio_sha = observe_role_files([resolved])[rid]["audio"]
            staged = media_root / staged_rel
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(_read_bytes(resolved.audio))
            if _observe(_read_bytes(staged)) != audio_sha:
                raise EvalError(f"{rid}: staged audio sha != corpus audio sha")
            _record(out_dir, journal, rid,
                    {"status": "submitting", "source_path": staged_rel, "batch_id": batch_id})
            with driver.session_scope(factory) as session:
                run = driver.submit_media_item_if_new(session, staged_rel)
            if run is None:
                with factory() as session:
                    run = _reconcile_run(driver, session, staged_rel)
            run_uuid = str(run.id)
            _record(out_dir, journal, rid, {"status": "queued", "run_uuid": run_uuid})
            driver.publish(run.id)

        if run_uuid is None:
            raise EvalError(f"{rid}: no run to poll (internal invariant)")

        deadline = time.monotonic() + timeout
        while True:
            with factory() as session:
                run = session.get(driver.PipelineRun, _as_uuid(run_uuid))
                if run is None:
                    raise EvalError(f"{rid}: run {run_uuid} vanished from the DB")
                status = eval_run.map_db_status(str(run.status))
                if status == "completed":
                    hyp_rttm, wer_text = _export_completed(driver, session, run.id, resolved)
                    break
                if status in eval_run.FAILURE_STATES:
                    _record(out_dir, journal, rid, {"status": status})
                    raise EvalError(f"{rid}: run {run_uuid} ended {status}")
            if time.monotonic() >= deadline:
                _record(out_dir, journal, rid, {"status": status})
                raise EvalError(f"{rid}: timed out after {timeout}s in status {status}")
            time.sleep(interval)

        artifacts = {"hypothesis_rttm_sha256": _observe(hyp_rttm.encode())[1]}
        patch: dict[str, Any] = {"status": "completed", "hypothesis_rttm": hyp_rttm}
        if wer_text is not None:
            artifacts["wer_text_sha256"] = _observe(wer_text.encode())[1]
            patch["wer_text"] = wer_text
        patch["artifacts"] = artifacts
        _record(out_dir, journal, rid, patch)
    finally:
        engine.dispose()


def _as_uuid(value: str) -> Any:
    import uuid as _uuid

    return _uuid.UUID(value)


def _add_run_parser(sub: Any) -> None:
    run_p = sub.add_parser(
        "run", help="drive the live pipeline over a corpus subset and emit a score bundle"
    )
    run_p.add_argument("--corpus", required=True, choices=("ami", "voxconverse"))
    run_p.add_argument("--corpus-root", help="ground-truth root (or EVAL_CORPUS_ROOT)")
    run_p.add_argument("--subset", required=True, help="scoring-subset JSON (array or {items:[]})")
    run_p.add_argument("--only", help="comma-separated recording ids to restrict to")
    run_p.add_argument("--out-dir", required=True, help="bundle + journal output directory")
    run_p.add_argument("--pipeline-env", required=True, help="static pipeline-environment JSON")
    run_p.add_argument("--database-url", help="postgres URL (or DATABASE_URL)")
    run_p.add_argument("--media-root", help="host path to stage audio into (default: settings)")
    run_p.add_argument("--media-subdir", default="eval", help="MEDIA_ROOT subdir for staged audio")
    run_p.add_argument("--container-prefix", default="voxint", help="docker compose project prefix")
    run_p.add_argument("--cuda-visible-devices", help="record CUDA_VISIBLE_DEVICES in fingerprint")
    run_p.add_argument("--interval", default="10", help="poll interval seconds (default 10)")
    run_p.add_argument("--timeout", default="3600", help="per-run poll timeout seconds (def 3600)")
    run_p.add_argument("--duration-tol", default="2.0", help="WAV duration tolerance seconds")
    run_p.add_argument("--batch-id", help="batch id (default: a fresh uuid per pass)")
    run_p.add_argument("--resume", action="store_true", help="honor an existing --out-dir journal")
    run_p.add_argument("--retry-failed", action="store_true", help="re-submit failed recordings")
    run_p.set_defaults(fn=cmd_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    score_p = sub.add_parser("score", help="score a hypotheses+reference manifest")
    score_p.add_argument("--manifest", required=True, help="paths manifest JSON")
    score_p.add_argument("--out", help="metrics JSON path (default: stdout)")
    score_p.set_defaults(fn=cmd_score)

    report_p = sub.add_parser("report", help="render metrics JSON to a dated Markdown report")
    report_p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="CORPUS=path.json",
        help="a scored metrics JSON tagged by corpus; repeat for K runs / more corpora",
    )
    report_p.add_argument(
        "--date", required=True, help="report date, YYYY-MM-DD (stamped in the header)"
    )
    report_p.add_argument("--out", help="Markdown path (default: stdout)")
    report_p.set_defaults(fn=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
