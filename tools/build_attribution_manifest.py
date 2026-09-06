#!/usr/bin/env python3
"""Build the DB-backed speaker-attribution alignment input (#113).

This maintainer-only bridge exports stored diarization turns and match evidence
from Postgres.  The attribution harness remains file-based and DB-free.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.db.models import DiarizationTurn, MatchCandidate, PipelineRun, RunStatus
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.export import to_rttm
from voxint.harness.attribution_protocol import (
    ProtocolManifest,
)
from voxint.harness.attribution_protocol import (
    parse_manifest as parse_protocol_manifest,
)

SCHEMA_VERSION = 1
KIND = "attribution_align_input"
ALIGN_INPUT_NAME = "align-input.json"
HYPOTHESIS_DIR_NAME = "hypothesis_rttm"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class ManifestError(Exception):
    """A user-facing input or export error."""


@dataclass(frozen=True)
class RunManifest:
    """Validated mapping from corpus meeting id to pipeline run id."""

    runs: dict[str, uuid.UUID]


def _dumps(payload: Any) -> str:
    """Serialize exactly as the attribution harness does."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _write_atomic(path: Path, text: str) -> None:
    """Write text via a same-directory temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{where}: expected a JSON object")
    return value


def _parse_uuid(value: Any, where: str) -> uuid.UUID:
    if not isinstance(value, str):
        raise ManifestError(f"{where}: expected a UUID string, got {value!r}")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ManifestError(f"{where}: not a valid UUID: {value!r}") from exc


def parse_run_manifest(payload: Any) -> RunManifest:
    """Validate the schema-1 meeting-to-run selection manifest."""
    doc = _require_mapping(payload, "run manifest")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"run manifest: schema_version must be {SCHEMA_VERSION}, "
            f"got {doc.get('schema_version')!r}"
        )
    raw_runs = _require_mapping(doc.get("runs"), "run manifest.runs")
    if not raw_runs:
        raise ManifestError("run manifest.runs: must not be empty")

    runs: dict[str, uuid.UUID] = {}
    seen_ids: set[uuid.UUID] = set()
    for meeting_id, raw_run_id in raw_runs.items():
        if not isinstance(meeting_id, str) or not meeting_id.strip():
            raise ManifestError(
                f"run manifest.runs: meeting IDs must be non-empty strings, got {meeting_id!r}"
            )
        run_id = _parse_uuid(raw_run_id, f"run manifest.runs[{meeting_id!r}]")
        if run_id in seen_ids:
            raise ManifestError(f"run manifest.runs: duplicate run id {run_id}")
        runs[meeting_id] = run_id
        seen_ids.add(run_id)
    return RunManifest(runs=runs)


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read {description} {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{description} {path} is not valid JSON: {exc.msg}") from exc


def _load_protocol(path: Path) -> ProtocolManifest:
    payload = _load_json(path, "protocol")
    if not isinstance(payload, dict):
        raise ManifestError(f"protocol {path}: expected a JSON object")
    try:
        protocol = parse_protocol_manifest(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"protocol {path}: invalid attribution protocol: {exc}") from exc
    if protocol.schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"protocol {path}: schema_version must be {SCHEMA_VERSION}, "
            f"got {protocol.schema_version!r}"
        )
    return protocol


def parse_enrolled_speaker_map(payload: Any) -> dict[str, str]:
    """Validate and normalize the gold-name to roster-speaker UUID map."""
    doc = _require_mapping(payload, "enrolled speaker map")
    enrolled: dict[str, str] = {}
    for global_name, raw_speaker_id in doc.items():
        if not isinstance(global_name, str) or not global_name.strip():
            raise ManifestError("enrolled speaker map: keys must be non-empty strings")
        speaker_id = _parse_uuid(raw_speaker_id, f"enrolled speaker map[{global_name!r}]")
        enrolled[global_name] = str(speaker_id)
    return enrolled


def resolve_gold_rttm_paths(gold_rttm_dir: Path, meeting_ids: Iterable[str]) -> dict[str, Path]:
    """Resolve and require one regular ``{meeting_id}.rttm`` file per meeting."""
    if not gold_rttm_dir.is_dir():
        raise ManifestError(f"gold RTTM directory does not exist: {gold_rttm_dir}")
    resolved: dict[str, Path] = {}
    for meeting_id in sorted(meeting_ids):
        path = (gold_rttm_dir / f"{meeting_id}.rttm").resolve()
        if not path.is_file():
            raise ManifestError(f"meeting {meeting_id}: missing gold RTTM {path}")
        resolved[meeting_id] = path
    return resolved


_EVIDENCE_FIELDS = (
    "similarity",
    "margin",
    "vote_agreement",
    "eligible_turns",
    "eligible_seconds",
    "roster_size",
    "top_speaker_id",
    "decision",
    "grounded",
)


def reshape_match_evidence(
    candidates: Iterable[Mapping[str, Any]],
    meeting_by_run: Mapping[Any, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reshape flat MatchCandidate-like mappings by meeting and local label."""
    normalized_meetings = {str(run_id): meeting for run_id, meeting in meeting_by_run.items()}
    result: dict[str, dict[str, dict[str, Any]]] = {
        meeting_id: {} for meeting_id in sorted(normalized_meetings.values())
    }
    for index, candidate in enumerate(candidates):
        raw_run_id = candidate.get("pipeline_run_id")
        run_id = str(raw_run_id)
        try:
            meeting_id = normalized_meetings[run_id]
        except KeyError as exc:
            raise ManifestError(
                f"match candidate {index}: unselected pipeline_run_id {raw_run_id!r}"
            ) from exc
        label = candidate.get("diarization_label")
        if not isinstance(label, str) or not label:
            raise ManifestError(f"match candidate {index}: invalid diarization_label {label!r}")
        if label in result[meeting_id]:
            raise ManifestError(f"meeting {meeting_id}: duplicate match evidence for {label}")

        evidence = {field: candidate.get(field) for field in _EVIDENCE_FIELDS}
        evidence["run_id"] = run_id
        evidence["pipeline_run_id"] = run_id
        top_speaker_id = evidence["top_speaker_id"]
        if top_speaker_id is not None:
            evidence["top_speaker_id"] = str(top_speaker_id)
        result[meeting_id][label] = evidence
    return result


def _relative_path(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=manifest_dir.resolve())).as_posix()


def assemble_manifest(
    *,
    protocol_path: Path,
    gold_paths: Mapping[str, Path],
    run_manifest: RunManifest,
    match_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    enrolled_speaker_map: Mapping[str, str],
    out_dir: Path,
    git_sha: str | None,
) -> dict[str, Any]:
    """Assemble the JSON-friendly align input with paths relative to its file."""
    manifest_dir = out_dir.resolve()
    meetings: dict[str, dict[str, str]] = {}
    evidence: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for meeting_id in sorted(run_manifest.runs):
        if meeting_id not in gold_paths:
            raise ManifestError(f"meeting {meeting_id}: no resolved gold RTTM")
        meetings[meeting_id] = {
            "gold_rttm": _relative_path(gold_paths[meeting_id], manifest_dir),
            "hypothesis_rttm": f"{HYPOTHESIS_DIR_NAME}/{meeting_id}.rttm",
            "run_id": str(run_manifest.runs[meeting_id]),
        }
        evidence[meeting_id] = match_evidence.get(meeting_id, {})
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "protocol_path": _relative_path(protocol_path, manifest_dir),
        "meetings": meetings,
        "match_evidence": evidence,
        "enrolled_speaker_map": dict(enrolled_speaker_map),
        "environment": {"git_sha": git_sha},
    }


def _candidate_mapping(candidate: MatchCandidate) -> dict[str, Any]:
    return {
        "pipeline_run_id": candidate.pipeline_run_id,
        "diarization_label": candidate.diarization_label,
        **{field: getattr(candidate, field) for field in _EVIDENCE_FIELDS},
    }


def export_database_artifacts(
    session: Session, run_manifest: RunManifest
) -> tuple[dict[str, str], dict[str, dict[str, dict[str, Any]]]]:
    """Validate selected runs, then render their RTTMs and stored evidence."""
    selected_ids = list(run_manifest.runs.values())
    runs = session.execute(select(PipelineRun).where(PipelineRun.id.in_(selected_ids))).scalars()
    run_by_id = {run.id: run for run in runs}
    for meeting_id, run_id in run_manifest.runs.items():
        run = run_by_id.get(run_id)
        if run is None:
            raise ManifestError(f"meeting {meeting_id}: pipeline run {run_id} does not exist")
        if run.status != RunStatus.COMPLETED.value:
            raise ManifestError(
                f"meeting {meeting_id}: pipeline run {run_id} is {run.status!r}, not completed"
            )

    rttm_by_meeting: dict[str, str] = {}
    candidate_rows: list[dict[str, Any]] = []
    for meeting_id, run_id in sorted(run_manifest.runs.items()):
        turns = (
            session.execute(
                select(DiarizationTurn)
                .where(DiarizationTurn.pipeline_run_id == run_id)
                .order_by(DiarizationTurn.turn_index)
            )
            .scalars()
            .all()
        )
        rttm_by_meeting[meeting_id] = to_rttm(turns, str(run_id))

        candidates = (
            session.execute(
                select(MatchCandidate)
                .where(MatchCandidate.pipeline_run_id == run_id)
                .order_by(MatchCandidate.diarization_label)
            )
            .scalars()
            .all()
        )
        candidate_rows.extend(_candidate_mapping(candidate) for candidate in candidates)

    meeting_by_run = {run_id: meeting for meeting, run_id in run_manifest.runs.items()}
    return rttm_by_meeting, reshape_match_evidence(candidate_rows, meeting_by_run)


def _git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None


def _validate_inputs(
    protocol_path: Path,
    run_manifest_path: Path,
    gold_rttm_dir: Path,
    enrolled_map_path: Path,
) -> tuple[RunManifest, dict[str, Path], dict[str, str]]:
    protocol = _load_protocol(protocol_path)
    run_manifest = parse_run_manifest(_load_json(run_manifest_path, "run manifest"))
    protocol_meetings = {row.meeting_id for row in protocol.rows}
    unknown = sorted(set(run_manifest.runs) - protocol_meetings)
    if unknown:
        raise ManifestError(f"run manifest meetings absent from protocol: {unknown}")
    gold_paths = resolve_gold_rttm_paths(gold_rttm_dir, run_manifest.runs)
    enrolled = parse_enrolled_speaker_map(_load_json(enrolled_map_path, "enrolled speaker map"))
    return run_manifest, gold_paths, enrolled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_attribution_manifest",
        description="Export DB evidence into an attribution align-input bundle (#113).",
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--gold-rttm-dir", required=True, type=Path)
    parser.add_argument("--enrolled-speaker-map", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        run_manifest, gold_paths, enrolled = _validate_inputs(
            args.protocol,
            args.run_manifest,
            args.gold_rttm_dir,
            args.enrolled_speaker_map,
        )
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    engine = build_engine()
    try:
        factory = build_session_factory(engine)
        try:
            with session_scope(factory) as session:
                rttm_by_meeting, evidence = export_database_artifacts(session, run_manifest)
        except ManifestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        engine.dispose()

    manifest = assemble_manifest(
        protocol_path=args.protocol,
        gold_paths=gold_paths,
        run_manifest=run_manifest,
        match_evidence=evidence,
        enrolled_speaker_map=enrolled,
        out_dir=args.out_dir,
        git_sha=_git_sha(_REPO_ROOT),
    )
    manifest_text = _dumps(manifest) + "\n"
    for meeting_id, rttm_text in sorted(rttm_by_meeting.items()):
        _write_atomic(args.out_dir / HYPOTHESIS_DIR_NAME / f"{meeting_id}.rttm", rttm_text)
    output_path = args.out_dir / ALIGN_INPUT_NAME
    _write_atomic(output_path, manifest_text)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
