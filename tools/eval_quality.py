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

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Frozen WER stack (parity extra): jiwer + the sha-pinned Whisper normalizer.
from tests.parity.bakeoff.normalize import (  # noqa: E402
    NORMALIZER_VERSION,
    runtime_fingerprint,
)
from tests.parity.whisper_bakeoff_score import score_pooled  # noqa: E402

# The frozen `run`-step contracts (pure, unit-covered): pipeline-environment
# identity + the domain-separated cohort descriptor/hash. `score` reuses them so
# a metrics JSON carries the SAME cohort identity `run` will stamp, recomputed
# here from the bytes actually scored rather than trusted from the manifest.
sys.path.insert(0, str(REPO / "tools"))
import eval_run  # noqa: E402

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


# A per-recording map of the OBSERVED bytes for cohort-input roles score can see:
# reference_rttm (always), uem (None when the entry scores the whole file), and
# wer_reference (filled in during WER load). None means an explicit null role.
Observed = dict[str, dict[str, tuple[int, str] | None]]


def _load_diar_items(entries: list[dict[str, Any]]) -> tuple[list[DiarItem], Observed]:
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
        ref_bytes = _read_bytes(Path(entry["reference_rttm"]))
        reference = parse_rttm(_decode(ref_bytes, entry["reference_rttm"]), rec_id)
        hypothesis = parse_rttm(_read(Path(entry["hypothesis_rttm"])), rec_id)
        uem_path = entry["uem"]
        if uem_path is not None:
            uem_bytes = _read_bytes(Path(uem_path))
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
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
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
        items, diar_observed = _load_diar_items(diar_entries)
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
            ref_bytes = _read_bytes(Path(e["reference_text"]))
            ref_text = _decode(ref_bytes, e["reference_text"])
            triples.append((rid, ref_text, _read(Path(e["hypothesis_text"]))))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
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
