#!/usr/bin/env python3
"""Speaker-attribution evaluation driver (#113 A4).

A MAINTAINER instrument, never shipped to users. It takes the three artifacts
the A5 GPU baseline step will produce (a protocol manifest, gold + hypothesis
RTTMs, and per-label match evidence) and scores them against corpus gold
speaker identity: align gold timing to predicted diarization slots, build
FAR/FRR trials, aggregate, and render a dated Markdown report.

Subcommands:

  protocol  inspect/validate an attribution protocol manifest
  align     gold RTTM + hypothesis RTTM + match evidence -> alignment + trials
  score     trials JSON -> attribution metrics JSON
  report    metrics JSON -> dated Markdown report

Usage::

    uv run python tools/eval_attribution.py protocol \\
        --manifest protocol.json

    uv run python tools/eval_attribution.py align \\
        --manifest align-input.json --out trials.json

    uv run python tools/eval_attribution.py score \\
        --trials trials.json --out metrics.json

    uv run python tools/eval_attribution.py report \\
        --run metrics.json --date 2026-09-05 --out report.md

The ``align`` step reads a self-contained input manifest pointing to RTTMs
and evidence files (relative paths resolve against the manifest's directory).
The ``score`` step is pure: trials JSON in, metrics JSON out. The ``report``
step renders one or more metrics JSONs into a dated Markdown report; when
given multiple runs it computes a noise-floor spread.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from voxint.harness.attribution_aligner import (  # noqa: E402
    AttributionTrial,
    Interval,
    SlotAlignment,
    SlotClassification,
    aggregate_trials,
    align_slots,
    build_trials,
)
from voxint.harness.attribution_protocol import (  # noqa: E402
    parse_manifest as parse_protocol_manifest,
)
from voxint.harness.calibration import Trial, TrialKind  # noqa: E402
from voxint.speakers.matching import MatchingGates  # noqa: E402

SCHEMA_VERSION = 1
ALIGN_INPUT_SCHEMA = 1


class EvalError(Exception):
    """A user-facing input problem (bad manifest, malformed RTTM)."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else base / p


def _git_sha() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (subprocess.SubprocessError, OSError):  # pragma: no cover
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


# --------------------------------------------------------------------------- #
# RTTM parsing (no pyannote dependency)
# --------------------------------------------------------------------------- #
def parse_rttm_intervals(text: str) -> dict[str, list[Interval]]:
    """Parse RTTM ``SPEAKER`` lines into ``{speaker_label: [Interval, ...]}``.

    The file-id column (field 1) is ignored. Non-positive and malformed
    intervals are rejected.
    """
    intervals: dict[str, list[Interval]] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split()
        if parts[0] != "SPEAKER":
            continue
        if len(parts) < 9:
            raise EvalError(
                f"RTTM line {lineno}: expected >=9 fields, got {len(parts)}"
            )
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError as exc:
            raise EvalError(
                f"RTTM line {lineno}: bad start/duration: {exc}"
            ) from exc
        if not math.isfinite(start) or start < 0.0:
            raise EvalError(f"RTTM line {lineno}: invalid start {start}")
        if not math.isfinite(duration) or duration <= 0.0:
            raise EvalError(f"RTTM line {lineno}: non-positive duration {duration}")
        label = parts[7]
        intervals.setdefault(label, []).append(Interval(start, start + duration))
    return intervals


# --------------------------------------------------------------------------- #
# ``protocol`` subcommand
# --------------------------------------------------------------------------- #
def cmd_protocol(args: argparse.Namespace) -> int:
    try:
        data = json.loads(_read(Path(args.manifest)))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"{args.manifest}: {exc}") from exc
    manifest = parse_protocol_manifest(data)
    report = manifest.recurrence_report
    lines = [
        f"Protocol: {manifest.corpus}, truth={manifest.truth_source}",
        f"  Schema version: {manifest.schema_version}",
        f"  Selection seed: {manifest.selection_seed}",
        f"  Rows: {len(manifest.rows)}",
        f"  Exclusions: {len(manifest.exclusions)}",
        f"  Meetings: {report.n_meetings}",
        f"  Base sessions: {report.n_base_sessions}",
        f"  Participants: {report.n_participants}",
        f"  Cross-session speakers: {report.n_cross_session_speakers}",
        f"  Genuine pairs: {report.n_genuine_pairs}",
        f"  Impostor pairs: {report.n_impostor_pairs}",
        f"  Baseline viable: {report.baseline_viable}",
        f"  Calibration viable: {report.calibration_viable}",
    ]
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------- #
# ``align`` subcommand
# --------------------------------------------------------------------------- #
def _validate_align_input(data: dict[str, Any]) -> None:
    version = data.get("schema_version")
    if version != ALIGN_INPUT_SCHEMA:
        raise EvalError(
            f"expected align input schema_version {ALIGN_INPUT_SCHEMA}, got {version}"
        )
    kind = data.get("kind")
    if kind != "attribution_align_input":
        raise EvalError(f"expected kind 'attribution_align_input', got {kind!r}")
    for key in ("meetings", "match_evidence", "enrolled_speaker_map", "protocol_path"):
        if key not in data:
            raise EvalError(f"align input missing required key {key!r}")
    if not isinstance(data["meetings"], dict) or not data["meetings"]:
        raise EvalError("align input must have a non-empty 'meetings' map")


def _serialize_alignment(alignment_report: Any) -> dict[str, Any]:
    return {
        "n_total_slots": alignment_report.n_total_slots,
        "n_genuine": alignment_report.n_genuine,
        "n_impostor": alignment_report.n_impostor,
        "n_mixed": alignment_report.n_mixed,
        "n_unscoreable": alignment_report.n_unscoreable,
        "n_no_gold_overlap": alignment_report.n_no_gold_overlap,
        "per_slot": [
            {
                "slot_label": a.slot_label,
                "classification": a.classification.value,
                "dominant_gold_speaker": a.dominant_gold_speaker,
                "purity": a.purity,
                "coverage": a.coverage,
                "margin": a.margin,
                "slot_duration": a.slot_duration,
            }
            for a in alignment_report.alignments
        ],
    }


def _serialize_trial(at: AttributionTrial, meeting_id: str) -> dict[str, Any]:
    t = at.trial
    return {
        "meeting_id": meeting_id,
        "run_id": t.run_id,
        "label": t.label,
        "similarity": t.similarity,
        "margin": t.margin,
        "vote_agreement": t.vote_agreement,
        "eligible_turns": t.eligible_turns,
        "eligible_seconds": t.eligible_seconds,
        "roster_size": t.roster_size,
        "top_speaker_id": t.top_speaker_id,
        "kind": t.kind.value,
        "truth_anchoring": t.truth_anchoring,
        "cluster_id": t.cluster_id,
        "slot_label": at.slot_label,
        "slot_classification": at.alignment.classification.value,
        "dominant_gold_speaker": at.alignment.dominant_gold_speaker,
        "slot_purity": at.alignment.purity,
        "slot_coverage": at.alignment.coverage,
        "slot_margin": at.alignment.margin,
        "slot_duration": at.alignment.slot_duration,
    }


def cmd_align(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        data = json.loads(_read(manifest_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"{args.manifest}: {exc}") from exc
    _validate_align_input(data)
    base = manifest_path.resolve().parent

    try:
        protocol_data = json.loads(_read(_resolve(base, data["protocol_path"])))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"protocol: {exc}") from exc
    protocol = parse_protocol_manifest(protocol_data)

    meetings = data["meetings"]
    evidence_by_meeting: dict[str, dict[str, dict[str, Any]]] = data["match_evidence"]
    enrolled: dict[str, str] = data["enrolled_speaker_map"]
    warnings: list[str] = []

    all_trials: list[dict[str, Any]] = []
    alignments_out: dict[str, Any] = {}

    for meeting_id in sorted(meetings):
        meeting_info = meetings[meeting_id]
        try:
            gold_path = _resolve(base, meeting_info["gold_rttm"])
            hyp_path = _resolve(base, meeting_info["hypothesis_rttm"])
            gold_intervals = parse_rttm_intervals(_read(gold_path))
            slot_intervals = parse_rttm_intervals(_read(hyp_path))
        except (OSError, KeyError) as exc:
            raise EvalError(f"meeting {meeting_id}: {exc}") from exc

        meeting_evidence = evidence_by_meeting.get(meeting_id, {})
        if not meeting_evidence:
            warnings.append(f"meeting {meeting_id}: no match evidence")

        turn_counts = {
            label: ev.get("eligible_turns", 0)
            for label, ev in meeting_evidence.items()
        }

        orphan_labels = set(meeting_evidence) - set(slot_intervals)
        if orphan_labels:
            warnings.append(
                f"meeting {meeting_id}: evidence labels not in hypothesis: "
                f"{sorted(orphan_labels)}"
            )

        alignment = align_slots(
            gold_intervals,
            slot_intervals,
            slot_turn_counts=turn_counts if turn_counts else None,
        )

        trials = build_trials(
            alignment,
            meeting_evidence,
            enrolled,
            truth_source=protocol.truth_source,
        )

        alignments_out[meeting_id] = _serialize_alignment(alignment)
        for at in trials:
            all_trials.append(_serialize_trial(at, meeting_id))

    if warnings:
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)

    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "attribution_trials",
        "protocol_summary": {
            "corpus": protocol.corpus,
            "truth_source": protocol.truth_source,
            "n_rows": len(protocol.rows),
            "n_meetings": len(meetings),
        },
        "alignment": alignments_out,
        "trials": all_trials,
        "warnings": warnings,
        "environment": {
            "git_sha": _git_sha(),
            "voxint_version": _pkg_version("voxint"),
        },
    }

    out = _dumps(output) + "\n"
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)
    return 0


# --------------------------------------------------------------------------- #
# ``score`` subcommand
# --------------------------------------------------------------------------- #
_REQUIRED_TRIAL_FIELDS = {"label", "kind", "slot_label", "slot_classification"}


def deserialize_trial(d: dict[str, Any]) -> AttributionTrial:
    """Reconstruct an ``AttributionTrial`` from its JSON representation."""
    missing = _REQUIRED_TRIAL_FIELDS - set(d)
    if missing:
        raise EvalError(f"trial missing required fields: {sorted(missing)}")
    try:
        trial = Trial(
            run_id=str(d.get("run_id") or ""),
            label=d["label"],
            similarity=d.get("similarity"),
            margin=d.get("margin"),
            vote_agreement=d.get("vote_agreement"),
            eligible_turns=d.get("eligible_turns", 0),
            eligible_seconds=d.get("eligible_seconds", 0.0),
            roster_size=d.get("roster_size"),
            top_speaker_id=d.get("top_speaker_id"),
            kind=TrialKind(d["kind"]),
            truth_anchoring=d.get("truth_anchoring", ""),
            cluster_id=d.get("cluster_id", ""),
        )
        alignment = SlotAlignment(
            slot_label=d["slot_label"],
            classification=SlotClassification(d["slot_classification"]),
            dominant_gold_speaker=d.get("dominant_gold_speaker"),
            purity=d.get("slot_purity", 0.0),
            coverage=d.get("slot_coverage", 0.0),
            margin=d.get("slot_margin", 0.0),
            slot_duration=d.get("slot_duration", 0.0),
        )
    except (ValueError, TypeError) as exc:
        raise EvalError(f"trial deserialization: {exc}") from exc
    return AttributionTrial(
        trial=trial,
        slot_label=d["slot_label"],
        alignment=alignment,
    )


def cmd_score(args: argparse.Namespace) -> int:
    try:
        data = json.loads(_read(Path(args.trials)))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"{args.trials}: {exc}") from exc
    if data.get("kind") != "attribution_trials":
        raise EvalError(
            f"expected kind 'attribution_trials', got {data.get('kind')!r}"
        )
    if data.get("schema_version") != SCHEMA_VERSION:
        raise EvalError(
            f"expected schema_version {SCHEMA_VERSION}, "
            f"got {data.get('schema_version')}"
        )

    trials = [deserialize_trial(d) for d in data["trials"]]

    gates = None
    if args.gates:
        try:
            gates_data = json.loads(_read(Path(args.gates)))
            gates = MatchingGates(**gates_data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise EvalError(f"--gates: {exc}") from exc

    summary = aggregate_trials(trials, gates)

    from dataclasses import asdict

    effective_gates = gates or MatchingGates()

    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "attribution_metrics",
        "n_genuine_trials": summary.n_genuine_trials,
        "n_impostor_trials": summary.n_impostor_trials,
        "n_unscoreable": summary.n_unscoreable,
        "n_auto_correct": summary.n_auto_correct,
        "n_auto_wrong": summary.n_auto_wrong,
        "n_review": summary.n_review,
        "n_abstain": summary.n_abstain,
        "far": summary.far,
        "far_ci_upper": summary.far_ci_upper,
        "frr": summary.frr,
        "frr_ci_upper": summary.frr_ci_upper,
        "coverage": summary.coverage,
        "n_speaker_clusters": summary.n_speaker_clusters,
        "alignment_attrition": summary.alignment_attrition,
        "gates": asdict(effective_gates),
        "environment": {
            "git_sha": _git_sha(),
            "voxint_version": _pkg_version("voxint"),
            "source_trials_kind": data.get("kind"),
            "n_trials_loaded": len(trials),
        },
    }

    out = _dumps(output) + "\n"
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)
    return 0


# --------------------------------------------------------------------------- #
# ``report`` subcommand
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _spread(values: list[float]) -> float:
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def render_report(date: str, runs: list[dict[str, Any]]) -> str:
    """Render one or more attribution metrics JSONs into a house-style report.

    Multiple runs compute a noise-floor spread. The report states limitations
    honestly: close-talk only, baseline not certification, Wilson CIs
    descriptive only.
    """
    if not runs:
        raise EvalError("report: no runs supplied")

    for run in runs:
        if run.get("kind") != "attribution_metrics":
            raise EvalError(
                f"expected kind 'attribution_metrics', got {run.get('kind')!r}"
            )
        if run.get("schema_version") != SCHEMA_VERSION:
            raise EvalError(
                f"expected schema_version {SCHEMA_VERSION}, "
                f"got {run.get('schema_version')}"
            )

    envs = [r.get("environment", {}) for r in runs]
    shas = sorted({e.get("git_sha", "unknown") for e in envs})
    sha_txt = shas[0] if len(shas) == 1 else ", ".join(shas)
    n = len(runs)

    far_values = [r["far"] for r in runs]
    frr_values = [r["frr"] for r in runs]
    coverage_values = [r["coverage"] for r in runs]

    lines = [
        f"# Speaker-attribution baseline ({date})",
        "",
        f"> Generated {date}. Pipeline git sha `{sha_txt}`, "
        f"Voxint {_pkg_version('voxint')}.",
        "",
        "This is a maintainer baseline for the speaker-attribution pipeline. "
        "It measures the false accept rate (FAR), false reject rate (FRR), and "
        "auto-attribution coverage against corpus gold labels. These numbers "
        "are not a claim about accuracy on unseen recordings; they catch "
        "gross attribution breakage and establish the operating point the "
        "calibration phase will tighten.",
        "",
        "Truth source is corpus gold (AMI global participant IDs), not "
        "post-proposal human adjudication. Session independence is enforced "
        "at protocol build time (A2): enrollment and test data must not "
        "share a base session.",
        "",
        "## Results",
        "",
        "| metric | value | 95% CI upper |",
        "| --- | --- | --- |",
        f"| FAR | {_pct(_mean(far_values))} "
        f"| {_pct(_mean([r['far_ci_upper'] for r in runs]))} |",
        f"| FRR | {_pct(_mean(frr_values))} "
        f"| {_pct(_mean([r['frr_ci_upper'] for r in runs]))} |",
        f"| Auto-attribution coverage | {_pct(_mean(coverage_values))} | |",
        "",
        "## Trial counts",
        "",
        "| category | count |",
        "| --- | --- |",
        f"| Genuine trials | {round(_mean([r['n_genuine_trials'] for r in runs]))} |",
        f"| Impostor trials | {round(_mean([r['n_impostor_trials'] for r in runs]))} |",
        f"| Unscoreable | {round(_mean([r['n_unscoreable'] for r in runs]))} |",
        f"| Auto correct | {round(_mean([r['n_auto_correct'] for r in runs]))} |",
        f"| Auto wrong | {round(_mean([r['n_auto_wrong'] for r in runs]))} |",
        f"| Review | {round(_mean([r['n_review'] for r in runs]))} |",
        f"| Abstain | {round(_mean([r['n_abstain'] for r in runs]))} |",
        f"| Speaker clusters | {round(_mean([r['n_speaker_clusters'] for r in runs]))} |",
        "",
    ]

    all_keys: set[str] = set()
    for r in runs:
        all_keys.update(r.get("alignment_attrition", {}))
    if all_keys:
        lines += [
            "## Alignment attrition",
            "",
            "| classification | count |",
            "| --- | --- |",
        ]
        for cls in sorted(all_keys):
            avg = round(
                _mean([r.get("alignment_attrition", {}).get(cls, 0) for r in runs])
            )
            lines.append(f"| {cls} | {avg} |")
        lines.append("")

    if n > 1:
        lines += [
            f"## Noise floor ({n} zero-change runs)",
            "",
            f"FAR spread: {_pct(_spread(far_values))}. "
            f"FRR spread: {_pct(_spread(frr_values))}. "
            f"Coverage spread: {_pct(_spread(coverage_values))}.",
            "",
        ]

    lines += [
        "## Limitations",
        "",
        "Close-talk IHM microphones only (AMI). "
        "Baseline-only status, not calibration certification. "
        "Effective sample counts may be small. "
        "Wilson CIs assume independent labels; with clustered speakers "
        "the true interval may be wider.",
        "",
    ]

    return "\n".join(lines).rstrip() + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    runs: list[dict[str, Any]] = []
    for path_str in args.run:
        try:
            data = json.loads(_read(Path(path_str)))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvalError(f"{path_str}: {exc}") from exc
        runs.append(data)
    text = render_report(args.date, runs)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speaker-attribution evaluation driver (#113 A4)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    proto_p = sub.add_parser(
        "protocol", help="inspect/validate a protocol manifest"
    )
    proto_p.add_argument("--manifest", required=True, help="protocol manifest JSON")
    proto_p.set_defaults(fn=cmd_protocol)

    align_p = sub.add_parser(
        "align",
        help="gold + hypothesis RTTM + evidence -> alignment + trials",
    )
    align_p.add_argument(
        "--manifest",
        required=True,
        help="align input manifest JSON (points to RTTMs and evidence)",
    )
    align_p.add_argument("--out", help="trials JSON path (default: stdout)")
    align_p.set_defaults(fn=cmd_align)

    score_p = sub.add_parser(
        "score", help="trials -> attribution metrics JSON"
    )
    score_p.add_argument("--trials", required=True, help="trials JSON path")
    score_p.add_argument("--gates", help="optional matching gates override JSON")
    score_p.add_argument("--out", help="metrics JSON path (default: stdout)")
    score_p.set_defaults(fn=cmd_score)

    report_p = sub.add_parser(
        "report", help="metrics JSON -> dated Markdown report"
    )
    report_p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PATH",
        help="metrics JSON (repeat for noise floor)",
    )
    report_p.add_argument(
        "--date", required=True, help="report date YYYY-MM-DD"
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
