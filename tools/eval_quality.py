#!/usr/bin/env python3
"""Diarization (DER/JER) + ASR (WER) quality harness for Voxint (issue #97).

A MAINTAINER instrument, never shipped to users. It scores hypothesis
diarization and transcripts against public ground truth (AMI + VoxConverse) so
the numerics doctrine can be honored with measured evidence when the GPU knobs
change (#96). It is a **tripwire, not a benchmark**: the 14-file subset cannot
prove non-regression, only catch gross breakage, and every threshold must be
measured from a zero-change noise floor rather than reasoned.

This module holds the two pieces that are testable without a worker:

* the diarization scorer, built on the vetted ``pyannote.metrics`` (never a
  hand-rolled DER: optimal mapping + collar + overlap + UEM crop is a
  silent-bug generator), fed through pyannote's own accumulators so the pooled
  number is a true micro-average, not a mean of per-file rates;
* the ASR scorer, reused verbatim from the frozen bakeoff WER stack
  (``tests/parity/whisper_bakeoff_score.py`` + the sha-pinned Whisper
  normalizer), applied to raw reference and raw hypothesis together.

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

Run:

    uv run --isolated --extra parity --extra eval-quality \\
        tools/eval_quality.py score --manifest <manifest.json> --out <metrics.json>
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

# The strict, gate-bearing diarization protocol vs the forgiving diagnostic one.
STRICT = {"collar": 0.0, "skip_overlap": False}
DIAGNOSTIC = {"collar": 0.5, "skip_overlap": True}

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


def _load_diar_items(entries: list[dict[str, Any]]) -> list[DiarItem]:
    items: list[DiarItem] = []
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
        reference = parse_rttm(_read(Path(entry["reference_rttm"])), rec_id)
        hypothesis = parse_rttm(_read(Path(entry["hypothesis_rttm"])), rec_id)
        uem_path = entry["uem"]
        uem = parse_uem(_read(Path(uem_path)), rec_id) if uem_path is not None else None
        items.append(DiarItem(rec_id, reference, hypothesis, uem))
    return items


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

    if diar_entries:
        items = _load_diar_items(diar_entries)
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
        triples = [
            (e["recording_id"], _read(Path(e["reference_text"])), _read(Path(e["hypothesis_text"])))
            for e in wer_entries
        ]
        report["wer"] = score_wer(triples)

    report["environment"] = _environment_manifest(manifest_bytes, diar_ids, wer_ids)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score_p = sub.add_parser("score", help="score a hypotheses+reference manifest")
    score_p.add_argument("--manifest", required=True, help="paths manifest JSON")
    score_p.add_argument("--out", help="metrics JSON path (default: stdout)")
    score_p.set_defaults(fn=cmd_score)
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
