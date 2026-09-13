#!/usr/bin/env python3
# ruff: noqa: E402
"""Re-derive attribution match evidence against the offline gold roster (#114 A4)."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tools.build_attribution_manifest import (
    ManifestError,
    RunManifest,
    _load_json,
    _load_protocol,
    parse_run_manifest,
)

from voxint.config import get_settings
from voxint.db.models import DiarizationTurn, MatchCandidate, PipelineRun, RunStatus
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.harness.attribution_protocol import MeetingRole, ProtocolManifest
from voxint.speakers.matching import MatchingGates, gates_from_settings, roster_centroids
from voxint.speakers.reembed import refresh_run_matches

TEST_ROLES: frozenset[MeetingRole] = frozenset({"test_genuine", "test_open"})


@dataclass(frozen=True)
class SelectedRun:
    """One protocol test meeting selected for CPU rematching."""

    meeting_id: str
    run_id: uuid.UUID
    role: MeetingRole


@dataclass(frozen=True)
class RunTally:
    """Post-rematch candidate decisions and effective roster size."""

    meeting_id: str
    run_id: uuid.UUID
    accepted: int
    rejected: int
    ineligible: int
    roster_size: int


def meeting_roles(protocol: ProtocolManifest) -> dict[str, MeetingRole]:
    """Collapse protocol rows to one validated role per meeting."""
    roles: dict[str, MeetingRole] = {}
    for row in protocol.rows:
        previous = roles.setdefault(row.meeting_id, row.role)
        if previous != row.role:
            raise ManifestError(
                f"protocol meeting {row.meeting_id}: inconsistent roles"
            )
    return roles


def select_runs_by_role(
    run_manifest: RunManifest,
    roles_by_meeting: Mapping[str, MeetingRole],
    selected_roles: set[MeetingRole],
) -> tuple[SelectedRun, ...]:
    """Validate meeting membership and select non-enrollment runs by role."""
    invalid_roles = selected_roles - TEST_ROLES
    if invalid_roles:
        raise ManifestError(
            f"roles cannot include enrollment or unknown values: {sorted(invalid_roles)}"
        )
    selected: list[SelectedRun] = []
    for meeting_id, run_id in sorted(run_manifest.runs.items()):
        try:
            role = roles_by_meeting[meeting_id]
        except KeyError as exc:
            raise ManifestError(
                f"run manifest meeting {meeting_id!r} is absent from protocol"
            ) from exc
        if role == "enrollment":
            raise ManifestError(
                f"meeting {meeting_id}: enrollment-role runs cannot be listed for rematching"
            )
        if role in selected_roles:
            selected.append(SelectedRun(meeting_id, run_id, role))
    if not selected:
        raise ManifestError("no runs match the selected roles")
    return tuple(selected)


def format_tally(tally: RunTally) -> str:
    """Render one stable operator-readable per-run tally line."""
    return (
        f"{tally.meeting_id} ({tally.run_id}): "
        f"accepted={tally.accepted} rejected={tally.rejected} "
        f"ineligible={tally.ineligible} roster={tally.roster_size}"
    )


def _validate_completed_runs(session: Session, run_manifest: RunManifest) -> None:
    runs = session.scalars(
        select(PipelineRun).where(PipelineRun.id.in_(run_manifest.runs.values()))
    )
    by_id = {run.id: run for run in runs}
    for meeting_id, run_id in run_manifest.runs.items():
        run = by_id.get(run_id)
        if run is None:
            raise ManifestError(f"meeting {meeting_id}: pipeline run {run_id} does not exist")
        if run.status != RunStatus.COMPLETED.value:
            raise ManifestError(
                f"meeting {meeting_id}: pipeline run {run_id} is {run.status!r}, not completed"
            )


def _candidate_count(session: Session, run_ids: Sequence[uuid.UUID]) -> int:
    if not run_ids:
        return 0
    return int(
        session.scalar(
            select(func.count(MatchCandidate.id)).where(
                MatchCandidate.pipeline_run_id.in_(run_ids)
            )
        )
        or 0
    )


def _run_tally(session: Session, selected: SelectedRun) -> RunTally:
    rows = session.execute(
        select(MatchCandidate.decision, func.count(MatchCandidate.id))
        .where(MatchCandidate.pipeline_run_id == selected.run_id)
        .group_by(MatchCandidate.decision)
    ).tuples()
    counts: Counter[str] = Counter({decision: count for decision, count in rows})
    spaces = set(
        session.scalars(
            select(DiarizationTurn.embedding_space)
            .where(
                DiarizationTurn.pipeline_run_id == selected.run_id,
                DiarizationTurn.embedding_space.is_not(None),
            )
            .distinct()
        )
    )
    roster_ids: set[uuid.UUID] = set()
    for space in spaces:
        if space is not None:
            roster_ids.update(roster_centroids(session, space))
    return RunTally(
        meeting_id=selected.meeting_id,
        run_id=selected.run_id,
        accepted=counts["accepted"],
        rejected=counts["rejected"],
        ineligible=counts["ineligible"],
        roster_size=len(roster_ids),
    )


def rematch_runs(
    session: Session,
    *,
    run_manifest: RunManifest,
    protocol: ProtocolManifest,
    selected_roles: set[MeetingRole],
    gates: MatchingGates,
    dry_run: bool,
) -> tuple[tuple[RunTally, ...], int]:
    """Validate all listed runs and atomically refresh the selected test runs."""
    roles = meeting_roles(protocol)
    selected = select_runs_by_role(run_manifest, roles, selected_roles)
    _validate_completed_runs(session, run_manifest)
    enrollment_ids = [
        run_id
        for meeting_id, run_id in run_manifest.runs.items()
        if roles[meeting_id] == "enrollment"
    ]
    enrollment_before = _candidate_count(session, enrollment_ids)
    if not dry_run:
        for item in selected:
            refresh_run_matches(session, item.run_id, gates)
        session.flush()
    enrollment_after = _candidate_count(session, enrollment_ids)
    if enrollment_after != enrollment_before:
        raise ManifestError(
            "enrollment-run match candidate count changed during rematch: "
            f"{enrollment_before} -> {enrollment_after}"
        )
    tallies = () if dry_run else tuple(_run_tally(session, item) for item in selected)
    return tallies, len(selected)


def _parse_roles(raw: str) -> set[MeetingRole]:
    values = {value.strip() for value in raw.split(",") if value.strip()}
    invalid = values - set(TEST_ROLES)
    if not values or invalid:
        raise ManifestError(
            "--roles must contain test_genuine and/or test_open; "
            f"invalid values: {sorted(invalid)}"
        )
    return {value for value in values if value in TEST_ROLES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--roles", default="test_genuine,test_open")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        protocol = _load_protocol(args.protocol)
        run_manifest = parse_run_manifest(_load_json(args.runs, "run manifest"))
        selected_roles = _parse_roles(args.roles)
        gates = gates_from_settings(get_settings())
        engine = build_engine()
        try:
            factory = build_session_factory(engine)
            with session_scope(factory) as session:
                tallies, selected_count = rematch_runs(
                    session,
                    run_manifest=run_manifest,
                    protocol=protocol,
                    selected_roles=selected_roles,
                    gates=gates,
                    dry_run=args.dry_run,
                )
        finally:
            engine.dispose()
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run: {selected_count} runs validated; no writes performed")
    else:
        for tally in tallies:
            print(format_tally(tally))
        print(f"Rematched runs: {selected_count}")
    print("Enrollment runs were left untouched (candidate count unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
