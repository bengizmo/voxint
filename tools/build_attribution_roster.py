#!/usr/bin/env python3
# ruff: noqa: E402
"""Build the gold-anchored AMI attribution roster (#114 A3).

Maintainer-only: select one high-quality enrollment slot per AMI global
participant, enroll all selected identities atomically, and write the durable
gold-name mapping plus a reproducibility fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session
from tools.build_attribution_manifest import (
    ManifestError,
    RunManifest,
    _dumps,
    _load_json,
    _load_protocol,
    _write_atomic,
    parse_run_manifest,
    resolve_gold_rttm_paths,
)
from tools.eval_attribution import EvalError, parse_rttm_intervals

from voxint.adjudication.enrollment import EnrollmentError, enroll_new_speaker
from voxint.adjudication.ledger import ConflictingReplayError
from voxint.config import get_settings
from voxint.db.models import (
    AdjudicationDecision,
    DiarizationTurn,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerEmbedding,
)
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.harness.attribution_aligner import (
    AlignmentReport,
    Interval,
    SlotClassification,
    align_slots,
)
from voxint.harness.attribution_protocol import ProtocolManifest
from voxint.speakers.matching import gates_from_settings
from voxint.speakers.roster import active_speaker_clause

ENROLLED_MAP_NAME = "enrolled_speaker_map.json"
FINGERPRINT_NAME = "roster_fingerprint.json"
DEFAULT_OPERATOR = "calibration-114"


@dataclass(frozen=True)
class EnrollmentCandidate:
    """Best scoreable diarization slot available for one gold identity."""

    global_name: str
    meeting_id: str
    run_id: uuid.UUID
    diarization_label: str
    purity: float
    slot_duration: float


@dataclass(frozen=True)
class EnrolledSource:
    """Roster UUID paired with the gold-aligned slot that created it."""

    speaker_id: uuid.UUID
    candidate: EnrollmentCandidate


def select_enrollment_candidates(
    reports: Mapping[str, AlignmentReport],
    enrollment_names_by_meeting: Mapping[str, set[str]],
    run_ids: Mapping[str, uuid.UUID],
    expected_names: set[str],
) -> tuple[dict[str, EnrollmentCandidate], tuple[str, ...]]:
    """Pick one deterministic best scoreable slot per enrollment identity."""
    candidates: list[EnrollmentCandidate] = []
    for meeting_id, report in reports.items():
        allowed = enrollment_names_by_meeting.get(meeting_id, set())
        run_id = run_ids.get(meeting_id)
        if run_id is None:
            continue
        for item in report.alignments:
            name = item.dominant_gold_speaker
            if item.classification != SlotClassification.GENUINE or name not in allowed:
                continue
            assert name is not None
            candidates.append(
                EnrollmentCandidate(
                    global_name=name,
                    meeting_id=meeting_id,
                    run_id=run_id,
                    diarization_label=item.slot_label,
                    purity=item.purity,
                    slot_duration=item.slot_duration,
                )
            )
    candidates.sort(
        key=lambda item: (
            -item.purity,
            -item.slot_duration,
            item.meeting_id,
            item.diarization_label,
            str(item.run_id),
        )
    )
    selected: dict[str, EnrollmentCandidate] = {}
    for candidate in candidates:
        selected.setdefault(candidate.global_name, candidate)
    return selected, tuple(sorted(expected_names - selected.keys()))


def roster_fingerprint(sources: Sequence[EnrolledSource]) -> str:
    """Hash the stable enrollment identity/source facts."""
    entries = sorted(
        (
            source.candidate.global_name,
            str(source.speaker_id),
            str(source.candidate.run_id),
            source.candidate.diarization_label,
            round(source.candidate.purity, 4),
            round(source.candidate.slot_duration, 3),
        )
        for source in sources
    )
    payload = {"count": len(entries), "entries": entries}
    return hashlib.sha256(_dumps(payload).encode()).hexdigest()


def build_fingerprint_document(
    sources: Sequence[EnrolledSource], *, git_sha: str | None, exported_at: str
) -> dict[str, Any]:
    """Build the JSON artifact that explains and fingerprints the roster."""
    entries = [
        {
            "global_name": source.candidate.global_name,
            "speaker_uuid": str(source.speaker_id),
            "source_run_id": str(source.candidate.run_id),
            "source_label": source.candidate.diarization_label,
            "purity": round(source.candidate.purity, 4),
            "slot_duration": round(source.candidate.slot_duration, 3),
        }
        for source in sorted(sources, key=lambda item: item.candidate.global_name)
    ]
    return {
        "schema_version": 1,
        "count": len(entries),
        "sha256": roster_fingerprint(sources),
        "entries": entries,
        "git_sha": git_sha,
        "exported_at": exported_at,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None


def _protocol_enrollment_names(
    protocol: ProtocolManifest,
) -> tuple[dict[str, set[str]], set[str]]:
    by_meeting: dict[str, set[str]] = {}
    all_names: set[str] = set()
    for row in protocol.rows:
        if row.role != "enrollment":
            continue
        by_meeting.setdefault(row.meeting_id, set()).add(row.host_global_name)
        all_names.add(row.host_global_name)
    if not all_names:
        raise ManifestError("protocol contains no enrollment-role speakers")
    return by_meeting, all_names


def _validate_enrollment_runs(
    session: Session, run_manifest: RunManifest, meeting_ids: set[str]
) -> None:
    selected = {
        meeting: run_id
        for meeting, run_id in run_manifest.runs.items()
        if meeting in meeting_ids
    }
    runs = session.scalars(select(PipelineRun).where(PipelineRun.id.in_(selected.values())))
    by_id = {run.id: run for run in runs}
    for meeting, run_id in selected.items():
        run = by_id.get(run_id)
        if run is None:
            raise ManifestError(f"meeting {meeting}: pipeline run {run_id} does not exist")
        if run.status != RunStatus.COMPLETED.value:
            raise ManifestError(
                f"meeting {meeting}: pipeline run {run_id} is {run.status!r}, not completed"
            )


def _assert_clean_roster(session: Session, operator: str, allow_existing: bool) -> None:
    if allow_existing:
        return
    active_ids = set(session.scalars(select(Speaker.id).where(active_speaker_clause())))
    owned_ids = set(
        session.scalars(
            select(SpeakerEmbedding.speaker_id)
            .join(
                AdjudicationDecision,
                AdjudicationDecision.id
                == SpeakerEmbedding.source_adjudication_decision_id,
            )
            .where(AdjudicationDecision.operator == operator)
        )
    )
    foreign = active_ids - owned_ids
    if foreign:
        raise ManifestError(
            "active roster contains speakers not created by operator "
            f"{operator!r}: {len(foreign)} (use --allow-existing to override)"
        )


def _alignment_reports(
    session: Session,
    run_manifest: RunManifest,
    enrollment_names_by_meeting: Mapping[str, set[str]],
    gold_paths: Mapping[str, Path],
) -> dict[str, AlignmentReport]:
    reports: dict[str, AlignmentReport] = {}
    for meeting_id in sorted(enrollment_names_by_meeting.keys() & run_manifest.runs.keys()):
        run_id = run_manifest.runs[meeting_id]
        turns = list(
            session.scalars(
                select(DiarizationTurn)
                .where(DiarizationTurn.pipeline_run_id == run_id)
                .order_by(DiarizationTurn.turn_index)
            )
        )
        slots: dict[str, list[Interval]] = {}
        counts: Counter[str] = Counter()
        for turn in turns:
            slots.setdefault(turn.label, []).append(
                Interval(turn.start_seconds, turn.end_seconds)
            )
            counts[turn.label] += 1
        try:
            gold = parse_rttm_intervals(gold_paths[meeting_id].read_text(encoding="utf-8"))
        except (OSError, EvalError) as exc:
            raise ManifestError(f"meeting {meeting_id}: cannot read gold RTTM: {exc}") from exc
        reports[meeting_id] = align_slots(
            gold,
            slots,
            slot_turn_counts=dict(counts),
        )
    return reports


def build_roster(
    session: Session,
    *,
    protocol: ProtocolManifest,
    run_manifest: RunManifest,
    gold_rttm_dir: Path,
    operator: str,
    allow_existing: bool,
    dry_run: bool,
) -> tuple[list[EnrolledSource], tuple[str, ...], dict[str, EnrollmentCandidate]]:
    """Plan and optionally enroll the full gold roster in the caller transaction."""
    names_by_meeting, expected_names = _protocol_enrollment_names(protocol)
    unknown = set(run_manifest.runs) - {row.meeting_id for row in protocol.rows}
    if unknown:
        raise ManifestError(f"run manifest meetings absent from protocol: {sorted(unknown)}")
    enrollment_meetings = names_by_meeting.keys() & run_manifest.runs.keys()
    gold_paths = resolve_gold_rttm_paths(gold_rttm_dir, enrollment_meetings)
    _validate_enrollment_runs(session, run_manifest, set(names_by_meeting))
    _assert_clean_roster(session, operator, allow_existing)
    reports = _alignment_reports(
        session, run_manifest, names_by_meeting, gold_paths
    )
    selected, unenrollable = select_enrollment_candidates(
        reports, names_by_meeting, run_manifest.runs, expected_names
    )
    if dry_run:
        return [], unenrollable, selected

    gates = gates_from_settings(get_settings())
    sources: list[EnrolledSource] = []
    for name, candidate in sorted(selected.items()):
        try:
            result = enroll_new_speaker(
                session,
                run_id=candidate.run_id,
                diarization_label=candidate.diarization_label,
                display_name=name,
                operator=operator,
                idempotency_key=f"calib114:{name}",
                gates=gates,
            )
        except (EnrollmentError, ConflictingReplayError) as exc:
            raise ManifestError(f"cannot enroll {name!r}: {exc}") from exc
        sources.append(EnrolledSource(result.speaker_id, candidate))
    return sources, unenrollable, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--gold-rttm-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)

    try:
        protocol = _load_protocol(args.protocol)
        run_manifest = parse_run_manifest(_load_json(args.runs, "run manifest"))
        engine = build_engine()
        try:
            factory = build_session_factory(engine)
            with session_scope(factory) as session:
                sources, unenrollable, selected = build_roster(
                    session,
                    protocol=protocol,
                    run_manifest=run_manifest,
                    gold_rttm_dir=args.gold_rttm_dir,
                    operator=args.operator,
                    allow_existing=args.allow_existing,
                    dry_run=args.dry_run,
                )
        finally:
            engine.dispose()
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Enrollment candidates: {len(selected)}")
    print(f"Unenrollable: {len(unenrollable)}")
    for name in unenrollable:
        print(f"  {name}")
    if args.dry_run:
        print("Dry run: no roster or artifact writes performed")
        return 0

    enrolled_map = {
        source.candidate.global_name: str(source.speaker_id) for source in sources
    }
    exported_at = datetime.now(UTC).isoformat()
    fingerprint = build_fingerprint_document(
        sources, git_sha=_git_sha(), exported_at=exported_at
    )
    _write_atomic(args.out_dir / ENROLLED_MAP_NAME, _dumps(enrolled_map) + "\n")
    _write_atomic(args.out_dir / FINGERPRINT_NAME, _dumps(fingerprint) + "\n")
    print(f"Enrolled identities: {len(sources)}")
    print(args.out_dir / ENROLLED_MAP_NAME)
    print(args.out_dir / FINGERPRINT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
