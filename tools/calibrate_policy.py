"""Calibration CLI for the three-band confidence policy (#114 Phase 2).

Maintainer tool (not part of the shipped ``voxint`` CLI). Follows the pattern
of ``tools/export_match_evidence.py``: the ``export`` subcommand reads the
database; ``sweep``, ``compare``, and ``baseline`` operate on files only.

Usage::

    uv run python -m tools.calibrate_policy export \\
        --run-ids <uuid>,<uuid> --truth-anchoring post_proposal \\
        --out trials.jsonl

    uv run python -m tools.calibrate_policy sweep \\
        --trials trials.jsonl --out sweep_report.json

    uv run python -m tools.calibrate_policy compare \\
        --trials trials.jsonl \\
        --baseline baseline_gates.json --candidate candidate_gates.json

    uv run python -m tools.calibrate_policy baseline --out baseline_gates.json
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from voxint.harness.calibration import (
    CompareResult,
    Trial,
    check_independence,
    classify_trial,
    compare,
    gates_from_dict,
    gates_to_dict,
    sweep,
    sweep_point_to_dict,
    trial_from_dict,
    trial_to_dict,
)
from voxint.speakers.matching import MatchingGates

_TRUTH_ANCHORING_VALUES = ("independent", "post_proposal")

_DEFAULT_COSINE_GRID = [
    0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90,
]
_DEFAULT_MARGIN_GRID = [
    0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20,
]


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _load_trials(path: Path) -> list[Trial]:
    trials: list[Trial] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(trial_from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"error: {path}:{lineno}: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
    return trials


def _load_gates(path: Path) -> MatchingGates:
    try:
        text = path.read_text(encoding="utf-8")
        return gates_from_dict(json.loads(text))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: cannot load gates from {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _parse_grid(raw: str | None, default: list[float]) -> list[float]:
    if raw is None:
        return default
    return sorted(float(x.strip()) for x in raw.split(","))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _cmd_export(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from voxint.adjudication.resolver import effective_decisions
    from voxint.db.models import MatchCandidate
    from voxint.db.session import build_engine, build_session_factory, session_scope
    from voxint.speakers.roster import canonicalize, merge_map

    truth_anchoring = args.truth_anchoring
    if truth_anchoring not in _TRUTH_ANCHORING_VALUES:
        print(
            f"error: --truth-anchoring must be one of "
            f"{', '.join(_TRUTH_ANCHORING_VALUES)}; got {truth_anchoring!r}",
            file=sys.stderr,
        )
        return 2

    try:
        run_ids = [uuid.UUID(r.strip()) for r in args.run_ids.split(",")]
    except ValueError as exc:
        print(f"error: invalid run id: {exc}", file=sys.stderr)
        return 2

    engine = build_engine()
    try:
        factory = build_session_factory(engine)
        tombstones: dict[uuid.UUID, uuid.UUID] = {}

        with session_scope(factory) as session:
            tombstones = merge_map(session)
            trials: list[Trial] = []

            for run_id in run_ids:
                mc_rows = (
                    session.execute(
                        select(MatchCandidate).where(
                            MatchCandidate.pipeline_run_id == run_id
                        )
                    )
                    .scalars()
                    .all()
                )
                decisions = effective_decisions(session, run_id)

                for mc in mc_rows:
                    decision = decisions.get(mc.diarization_label)
                    human_decision = decision.decision if decision else None

                    human_raw = decision.speaker_id if decision else None
                    human_canonical = canonicalize(human_raw, tombstones) if human_raw else None
                    human_speaker_id = str(human_canonical) if human_canonical else None

                    machine_raw = mc.top_speaker_id
                    machine_canonical = (
                        canonicalize(machine_raw, tombstones) if machine_raw else None
                    )
                    machine_speaker_id = (
                        str(machine_canonical) if machine_canonical else None
                    )

                    trial = classify_trial(
                        run_id=str(run_id),
                        label=mc.diarization_label,
                        mc_decision=mc.decision,
                        similarity=mc.similarity,
                        margin=mc.margin,
                        vote_agreement=mc.vote_agreement,
                        eligible_turns=mc.eligible_turns,
                        eligible_seconds=mc.eligible_seconds,
                        roster_size=mc.roster_size,
                        top_speaker_id=machine_speaker_id,
                        human_decision=human_decision,
                        human_speaker_id=human_speaker_id,
                        truth_anchoring=truth_anchoring,
                    )
                    trials.append(trial)
    finally:
        engine.dispose()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(_dumps(trial_to_dict(trial)) + "\n")

    kinds: dict[str, int] = {"genuine": 0, "impostor": 0, "unscoreable": 0}
    for t in trials:
        kinds[t.kind.value] += 1
    print(f"Exported {len(trials)} trials to {out_path}")
    for k, v in sorted(kinds.items()):
        print(f"  {k}: {v}")

    independence = check_independence(trials)
    print(f"  clusters: {independence.n_clusters}")
    if not independence.sufficient:
        print(
            f"  WARNING: only {independence.n_clusters} independent clusters "
            f"(minimum {50} for a reliable decision)"
        )
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _cmd_sweep(args: argparse.Namespace) -> int:
    trials = _load_trials(Path(args.trials))
    base_gates = _load_gates(Path(args.gates)) if args.gates else MatchingGates()

    cosine_grid = _parse_grid(args.cosine_grid, _DEFAULT_COSINE_GRID)
    margin_grid = _parse_grid(args.margin_grid, _DEFAULT_MARGIN_GRID)

    strata = [args.stratum] if args.stratum else [None, "R=1", "R>=2"]
    report: dict[str, Any] = {"strata": {}}

    for stratum in strata:
        label = stratum or "all"
        points = sweep(
            trials,
            cosine_grid=cosine_grid,
            margin_grid=margin_grid,
            base_gates=base_gates,
            roster_stratum=stratum,
        )
        report["strata"][label] = [sweep_point_to_dict(p) for p in points]

    report["base_gates"] = gates_to_dict(base_gates)
    report["cosine_grid"] = cosine_grid
    report["margin_grid"] = margin_grid

    independence = check_independence(trials)
    report["independence"] = {
        "n_clusters": independence.n_clusters,
        "n_trials": independence.n_trials,
        "sufficient": independence.sufficient,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    total_points = sum(len(v) for v in report["strata"].values())
    print(f"Sweep: {total_points} grid points across {len(strata)} strata written to {out_path}")
    if not independence.sufficient:
        print(
            f"WARNING: only {independence.n_clusters} independent clusters "
            f"(minimum {50} for a reliable decision)"
        )
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _print_compare(result: CompareResult) -> None:
    print(f"Scoreable trials: {result.n_scoreable}")
    print()
    print("            auto_correct  auto_wrong  review  abstain")
    print(
        f"  baseline  {result.baseline_auto_correct:>12}  "
        f"{result.baseline_auto_wrong:>10}  "
        f"{result.baseline_review:>6}  "
        f"{result.baseline_abstain:>7}"
    )
    print(
        f"  candidate {result.candidate_auto_correct:>12}  "
        f"{result.candidate_auto_wrong:>10}  "
        f"{result.candidate_review:>6}  "
        f"{result.candidate_abstain:>7}"
    )
    delta_ac = result.candidate_auto_correct - result.baseline_auto_correct
    delta_aw = result.candidate_auto_wrong - result.baseline_auto_wrong
    delta_rv = result.candidate_review - result.baseline_review
    delta_ab = result.candidate_abstain - result.baseline_abstain
    print(
        f"  delta     {delta_ac:>+12}  {delta_aw:>+10}  "
        f"{delta_rv:>+6}  {delta_ab:>+7}"
    )
    if result.changes:
        print(f"\n{len(result.changes)} labels changed band:")
        for ch in result.changes:
            sim = f"{ch.similarity:.3f}" if ch.similarity is not None else "n/a"
            print(
                f"  {ch.run_id[:8]}.. {ch.label}: "
                f"{ch.old_band} -> {ch.new_band} "
                f"(cos={sim}, {ch.kind})"
            )
    else:
        print("\nNo labels changed band.")


def _cmd_compare(args: argparse.Namespace) -> int:
    trials = _load_trials(Path(args.trials))
    baseline = _load_gates(Path(args.baseline))
    candidate = _load_gates(Path(args.candidate))

    result = compare(trials, baseline_gates=baseline, candidate_gates=candidate)
    _print_compare(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": gates_to_dict(baseline),
            "candidate": gates_to_dict(candidate),
            "n_scoreable": result.n_scoreable,
            "baseline_tallies": {
                "auto_correct": result.baseline_auto_correct,
                "auto_wrong": result.baseline_auto_wrong,
                "review": result.baseline_review,
                "abstain": result.baseline_abstain,
            },
            "candidate_tallies": {
                "auto_correct": result.candidate_auto_correct,
                "auto_wrong": result.candidate_auto_wrong,
                "review": result.candidate_review,
                "abstain": result.candidate_abstain,
            },
            "changes": [
                {
                    "run_id": ch.run_id,
                    "label": ch.label,
                    "old_band": ch.old_band,
                    "new_band": ch.new_band,
                    "top_speaker_id": ch.top_speaker_id,
                    "similarity": ch.similarity,
                    "kind": ch.kind.value,
                }
                for ch in result.changes
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        print(f"\nFull report written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def _cmd_baseline(args: argparse.Namespace) -> int:
    from voxint.config import get_settings
    from voxint.speakers.matching import gates_from_settings

    settings = get_settings()
    gates = gates_from_settings(settings)
    snapshot = gates_to_dict(gates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(f"Baseline gates written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate_policy",
        description="Calibration tooling for the three-band confidence policy (#114).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = sub.add_parser(
        "export", help="Export calibration trials from the database to JSONL."
    )
    p_export.add_argument(
        "--run-ids",
        required=True,
        help="Comma-separated pipeline run UUIDs to export.",
    )
    p_export.add_argument(
        "--truth-anchoring",
        required=True,
        choices=_TRUTH_ANCHORING_VALUES,
        help="How the human truth relates to the machine proposal.",
    )
    p_export.add_argument("--out", required=True, help="Output JSONL file path.")

    # sweep
    p_sweep = sub.add_parser(
        "sweep", help="Run a gate sweep on exported trials."
    )
    p_sweep.add_argument("--trials", required=True, help="Input trials JSONL file.")
    p_sweep.add_argument("--out", required=True, help="Output JSON report file.")
    p_sweep.add_argument(
        "--gates",
        default=None,
        help="Base gates JSON file (default: built-in MatchingGates defaults).",
    )
    p_sweep.add_argument(
        "--cosine-grid",
        default=None,
        help="Comma-separated grounded cosine values to sweep.",
    )
    p_sweep.add_argument(
        "--margin-grid",
        default=None,
        help="Comma-separated grounded margin values to sweep.",
    )
    p_sweep.add_argument(
        "--stratum",
        default=None,
        choices=["R=1", "R>=2"],
        help="Restrict sweep to a roster-size stratum (default: all + both).",
    )

    # compare
    p_compare = sub.add_parser(
        "compare", help="PRE/POST band diff between two gate configurations."
    )
    p_compare.add_argument("--trials", required=True, help="Input trials JSONL file.")
    p_compare.add_argument("--baseline", required=True, help="Baseline gates JSON.")
    p_compare.add_argument("--candidate", required=True, help="Candidate gates JSON.")
    p_compare.add_argument("--out", default=None, help="Optional output JSON report.")

    # baseline
    p_baseline = sub.add_parser(
        "baseline", help="Snapshot current gates from settings to a JSON file."
    )
    p_baseline.add_argument("--out", required=True, help="Output JSON file.")

    args = parser.parse_args(argv)
    dispatch = {
        "export": _cmd_export,
        "sweep": _cmd_sweep,
        "compare": _cmd_compare,
        "baseline": _cmd_baseline,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
