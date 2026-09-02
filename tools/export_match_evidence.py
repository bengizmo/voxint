"""Maintainer driver (#113, step 5): DB -> ``voxint score`` input files.

The observational matcher evidence (#113) lives in the database; the scoring
harness (``voxint score``) reads plain files and never touches a database. This
driver bridges the two: it reads a small run-selection manifest, calls the
``voxint.harness_export`` exporters against the live database, and writes the
JSON/JSONL files the harness consumes, plus the evidence snapshot they must be
read against.

It is a maintainer tool, not part of the shipped ``voxint`` CLI: the score
command stays DB-free by contract, so the one piece that reads the database is
kept out here (precedent: ``tools/qualify_local_llm.py``,
``tools/generate_bakeoff_baseline.py``). It exports STORED evidence only; it
never re-runs TitaNet or the matcher, so an exported baseline reflects
production numerics exactly.

Manifest (schema_version 1)::

    {
      "schema_version": 1,
      "embedding_space": "titanet-large-v2",
      "name_accuracy": {
        "truth_anchoring": "independent",
        "run_ids": ["<uuid>", "..."]
      },
      "agreement": {
        "runs": [
          {"run_id": "<uuid>", "kind": "curated", "host_id": "<uuid>"},
          {"run_id": "<uuid>", "kind": "negative_control"}
        ],
        "roster_speaker_ids": ["<uuid>", "..."]
      }
    }

Both lanes are optional; at least one must be present. ``embedding_space`` is
required and shared by every artifact, so enrollment voiceprints and slot
vectors are always compared in one space. Files land in ``--out-dir``:

- ``snapshot.json``               (always)
- ``name_accuracy_items.jsonl``   (name_accuracy lane)
- ``enrollment.json``             (agreement lane)
- ``agreement_slots.jsonl``       (agreement lane)

Usage::

    uv run python -m tools.export_match_evidence \\
        --manifest baseline-selection.json --out-dir docs/reports/baseline-export
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from voxint.config import Settings, get_settings
from voxint.db.session import build_engine, build_session_factory, session_scope
from voxint.harness_export import (
    ExportError,
    TruthAnchoring,
    agreement_enrollment,
    agreement_slots,
    evidence_snapshot,
    name_accuracy_items,
)
from voxint.speakers.matching import gates_from_settings

MANIFEST_SCHEMA_VERSION = 1

KIND_CURATED = "curated"
KIND_NEGATIVE_CONTROL = "negative_control"
_KINDS = (KIND_CURATED, KIND_NEGATIVE_CONTROL)

FILE_SNAPSHOT = "snapshot.json"
FILE_NAME_ACCURACY_ITEMS = "name_accuracy_items.jsonl"
FILE_ENROLLMENT = "enrollment.json"
FILE_AGREEMENT_SLOTS = "agreement_slots.jsonl"

_REPO_ROOT = Path(__file__).resolve().parent.parent


class ManifestError(Exception):
    """A malformed run-selection manifest. The message is user-facing."""


# --------------------------------------------------------------------------- #
# serialization (byte-identical to voxint.harness.score_cli)
# --------------------------------------------------------------------------- #
def _dumps(payload: Any) -> str:
    """Serialize one JSON value the way the score harness serializes its own.

    Mirrors ``voxint.harness.score_cli._dumps`` (sort_keys, no NaN/Infinity) so a
    repeated export from an unchanged database is byte-for-byte reproducible and
    the files can never carry a non-finite number the scorer would reject.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _dump_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_dumps(record) + "\n" for record in records)


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + rename)."""
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


# --------------------------------------------------------------------------- #
# manifest model + parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgreementRun:
    run_id: uuid.UUID
    kind: str
    host_id: uuid.UUID | None


@dataclass(frozen=True)
class NameAccuracyLane:
    truth_anchoring: TruthAnchoring
    run_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class AgreementLane:
    runs: tuple[AgreementRun, ...]
    roster_speaker_ids: tuple[uuid.UUID, ...] | None


@dataclass(frozen=True)
class Manifest:
    embedding_space: str
    name_accuracy: NameAccuracyLane | None
    agreement: AgreementLane | None


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{where}: expected a JSON object")
    return value


def _uuid(value: Any, where: str) -> uuid.UUID:
    if not isinstance(value, str):
        raise ManifestError(f"{where}: expected a UUID string, got {value!r}")
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ManifestError(f"{where}: not a valid UUID: {value!r}") from exc


def _uuid_list(value: Any, where: str, *, allow_empty: bool = False) -> list[uuid.UUID]:
    if not isinstance(value, list):
        raise ManifestError(f"{where}: expected a list")
    ids = [_uuid(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if not ids and not allow_empty:
        raise ManifestError(f"{where}: must not be empty")
    seen: set[uuid.UUID] = set()
    for run_id in ids:
        if run_id in seen:
            raise ManifestError(f"{where}: duplicate id {run_id}")
        seen.add(run_id)
    return ids


def _parse_name_accuracy(block: Any) -> NameAccuracyLane:
    payload = _require_mapping(block, "name_accuracy")
    raw_anchoring = payload.get("truth_anchoring")
    allowed = ", ".join(a.value for a in TruthAnchoring)
    if not isinstance(raw_anchoring, str):
        raise ManifestError(
            f"name_accuracy.truth_anchoring: must be one of {allowed}; "
            f"got {raw_anchoring!r}"
        )
    try:
        anchoring = TruthAnchoring(raw_anchoring)
    except ValueError as exc:
        raise ManifestError(
            f"name_accuracy.truth_anchoring: must be one of {allowed}; "
            f"got {raw_anchoring!r}"
        ) from exc
    run_ids = _uuid_list(payload.get("run_ids"), "name_accuracy.run_ids")
    return NameAccuracyLane(truth_anchoring=anchoring, run_ids=tuple(run_ids))


def _parse_agreement_run(block: Any, where: str) -> AgreementRun:
    payload = _require_mapping(block, where)
    run_id = _uuid(payload.get("run_id"), f"{where}.run_id")
    kind = payload.get("kind")
    if kind not in _KINDS:
        raise ManifestError(
            f"{where}.kind: must be one of {', '.join(_KINDS)}; got {kind!r}"
        )
    host_present = payload.get("host_id") is not None
    if kind == KIND_CURATED:
        if not host_present:
            raise ManifestError(f"{where}: a curated run must name its host_id")
        host_id: uuid.UUID | None = _uuid(payload["host_id"], f"{where}.host_id")
    else:
        if host_present:
            raise ManifestError(
                f"{where}: a negative_control run must not carry a host_id"
            )
        host_id = None
    return AgreementRun(run_id=run_id, kind=kind, host_id=host_id)


def _parse_agreement(block: Any) -> AgreementLane:
    payload = _require_mapping(block, "agreement")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ManifestError("agreement.runs: must be a non-empty list")
    runs = tuple(
        _parse_agreement_run(item, f"agreement.runs[{i}]")
        for i, item in enumerate(raw_runs)
    )
    seen: set[uuid.UUID] = set()
    for run in runs:
        if run.run_id in seen:
            raise ManifestError(f"agreement.runs: duplicate run id {run.run_id}")
        seen.add(run.run_id)
    roster_raw = payload.get("roster_speaker_ids")
    roster = (
        tuple(_uuid_list(roster_raw, "agreement.roster_speaker_ids"))
        if roster_raw is not None
        else None
    )
    return AgreementLane(runs=runs, roster_speaker_ids=roster)


def parse_manifest(payload: Any) -> Manifest:
    """Validate a run-selection manifest, failing closed on any problem.

    A bad selection must be rejected before the database is touched, never
    silently narrowed: a partial export would bias a baseline without anyone
    noticing.
    """
    doc = _require_mapping(payload, "manifest")
    if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest: schema_version must be {MANIFEST_SCHEMA_VERSION}, "
            f"got {doc.get('schema_version')!r}"
        )
    space = doc.get("embedding_space")
    if not isinstance(space, str) or not space:
        raise ManifestError("manifest: embedding_space is required (a non-empty string)")

    name_accuracy = (
        _parse_name_accuracy(doc["name_accuracy"])
        if doc.get("name_accuracy") is not None
        else None
    )
    agreement = (
        _parse_agreement(doc["agreement"])
        if doc.get("agreement") is not None
        else None
    )
    if name_accuracy is None and agreement is None:
        raise ManifestError(
            "manifest: at least one of name_accuracy or agreement must be present"
        )
    return Manifest(
        embedding_space=space, name_accuracy=name_accuracy, agreement=agreement
    )


# --------------------------------------------------------------------------- #
# artifact assembly
# --------------------------------------------------------------------------- #
def _union_run_ids(manifest: Manifest) -> list[uuid.UUID]:
    """Ordered-unique run ids referenced by any lane (for the snapshot)."""
    ordered: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    lanes: list[Iterable[uuid.UUID]] = []
    if manifest.name_accuracy is not None:
        lanes.append(manifest.name_accuracy.run_ids)
    if manifest.agreement is not None:
        lanes.append(run.run_id for run in manifest.agreement.runs)
    for run_id in (rid for lane in lanes for rid in lane):
        if run_id not in seen:
            seen.add(run_id)
            ordered.append(run_id)
    return ordered


def build_artifacts(
    session: Session,
    settings: Settings,
    manifest: Manifest,
    *,
    exported_at: str,
    git_sha: str | None,
) -> dict[str, str]:
    """Render a manifest into ``{filename: file text}`` (pure given the session).

    Keeping IO and the clock out of this function makes it byte-for-byte
    testable and lets the same mapping round-trip through the real
    ``voxint score`` parsers. The evidence snapshot is always emitted; each lane
    emits its files only when its block is present. The snapshot's embedding
    space is fixed to the manifest's, so every artifact shares one space.
    """
    space = manifest.embedding_space
    artifacts: dict[str, str] = {}

    snapshot = evidence_snapshot(
        session,
        settings,
        _union_run_ids(manifest),
        exported_at=exported_at,
        git_sha=git_sha,
        embedding_space=space,
    )
    artifacts[FILE_SNAPSHOT] = _dumps(snapshot) + "\n"

    if manifest.name_accuracy is not None:
        items = name_accuracy_items(
            session,
            manifest.name_accuracy.run_ids,
            truth_anchoring=manifest.name_accuracy.truth_anchoring,
        )
        artifacts[FILE_NAME_ACCURACY_ITEMS] = _dump_jsonl(items)

    if manifest.agreement is not None:
        agreement = manifest.agreement
        enrollment = agreement_enrollment(
            session,
            space,
            roster_speaker_ids=agreement.roster_speaker_ids,
        )
        slots = agreement_slots(
            session,
            [run.run_id for run in agreement.runs],
            kind_by_run={run.run_id: run.kind for run in agreement.runs},
            host_by_run={
                run.run_id: run.host_id
                for run in agreement.runs
                if run.host_id is not None
            },
            gates=gates_from_settings(settings),
            embedding_space=space,
        )
        artifacts[FILE_ENROLLMENT] = _dumps(enrollment) + "\n"
        artifacts[FILE_AGREEMENT_SLOTS] = _dump_jsonl(slots)

    return artifacts


def write_artifacts(artifacts: Mapping[str, str], out_dir: Path) -> list[Path]:
    """Atomically write each artifact into ``out_dir``; return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(artifacts):
        path = out_dir / name
        _write_atomic(path, artifacts[name])
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# git provenance
# --------------------------------------------------------------------------- #
def _git_sha(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_tree_dirty(repo: Path) -> bool:
    """True if tracked files differ from HEAD (untracked files are ignored)."""
    try:
        return (
            subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=repo).returncode != 0
        )
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_manifest_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc.strerror or exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc.msg}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_match_evidence",
        description="Export live match evidence into voxint score input files (#113).",
    )
    parser.add_argument("--manifest", required=True, help="run-selection manifest JSON")
    parser.add_argument("--out-dir", required=True, help="directory to write artifacts into")
    parser.add_argument(
        "--git-sha",
        default=None,
        help="record this code git sha in the snapshot (default: current HEAD)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="export even though the working tree has uncommitted tracked changes",
    )
    args = parser.parse_args(argv)

    try:
        manifest = parse_manifest(_load_manifest_file(Path(args.manifest)))
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    git_sha = args.git_sha if args.git_sha is not None else _git_sha(_REPO_ROOT)
    if git_sha is not None and not args.allow_dirty and _git_tree_dirty(_REPO_ROOT):
        print(
            "error: the working tree has uncommitted tracked changes, so the "
            f"recorded git sha ({git_sha[:12]}) would not match the code that "
            "produced this export. Commit or stash first, or pass --allow-dirty.",
            file=sys.stderr,
        )
        return 2

    exported_at = datetime.now(UTC).isoformat()

    engine = build_engine()
    try:
        factory = build_session_factory(engine)
        settings = get_settings()
        try:
            with session_scope(factory) as session:
                artifacts = build_artifacts(
                    session,
                    settings,
                    manifest,
                    exported_at=exported_at,
                    git_sha=git_sha,
                )
        except ExportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        written = write_artifacts(artifacts, Path(args.out_dir))
    finally:
        engine.dispose()

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
