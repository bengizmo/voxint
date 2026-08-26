"""Pure-logic tests for the ``tools/export_match_evidence.py`` maintainer driver.

The driver's DB-backed responsibilities (``build_artifacts`` over seeded runs and
the full round trip through ``voxint score``) live in the integration suite of
the same basename. This module covers only the parts that touch no database:
fail-closed manifest parsing, deterministic serialization byte-identical to the
score harness, atomic writes, the git helpers, and the CLI paths that fail closed
BEFORE any database is constructed (guarded here by the ``no_db`` tripwire).
"""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from tools import export_match_evidence as drv
from tools.export_match_evidence import ManifestError, main, parse_manifest, write_artifacts

from voxint.harness.score_cli import _dumps as harness_dumps
from voxint.harness_export import TruthAnchoring

SPACE = "titanet-large-v1"
ANCHOR = TruthAnchoring.INDEPENDENT.value


@pytest.fixture()
def no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the CLI error paths fail closed before the database is constructed.

    These ``main()`` tests are only unit tests because they return before
    ``build_engine()`` is ever called; wire the engine builder to blow up so a
    regression that reaches it fails loudly instead of silently touching a DB.
    """

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("build_engine must not be reached on the error path")

    monkeypatch.setattr(drv, "build_engine", _boom)


# --------------------------------------------------------------------------- #
# manifest parsing
# --------------------------------------------------------------------------- #
def _na_block(run_ids: list[Any]) -> dict[str, Any]:
    return {"truth_anchoring": ANCHOR, "run_ids": run_ids}


def test_parse_name_accuracy_only() -> None:
    rid = str(uuid.uuid4())
    manifest = parse_manifest(
        {
            "schema_version": 1,
            "embedding_space": SPACE,
            "name_accuracy": _na_block([rid]),
        }
    )
    assert manifest.embedding_space == SPACE
    assert manifest.agreement is None
    assert manifest.name_accuracy is not None
    assert manifest.name_accuracy.truth_anchoring is TruthAnchoring.INDEPENDENT
    assert manifest.name_accuracy.run_ids == (uuid.UUID(rid),)


def test_parse_agreement_curated_and_negative() -> None:
    host = str(uuid.uuid4())
    curated = str(uuid.uuid4())
    negative = str(uuid.uuid4())
    manifest = parse_manifest(
        {
            "schema_version": 1,
            "embedding_space": SPACE,
            "agreement": {
                "runs": [
                    {"run_id": curated, "kind": "curated", "host_id": host},
                    {"run_id": negative, "kind": "negative_control"},
                ],
                "roster_speaker_ids": [host],
            },
        }
    )
    assert manifest.name_accuracy is None
    assert manifest.agreement is not None
    assert manifest.agreement.roster_speaker_ids == (uuid.UUID(host),)
    kinds = {run.run_id: (run.kind, run.host_id) for run in manifest.agreement.runs}
    assert kinds[uuid.UUID(curated)] == ("curated", uuid.UUID(host))
    assert kinds[uuid.UUID(negative)] == ("negative_control", None)


def test_parse_rejects_bad_schema_version() -> None:
    with pytest.raises(ManifestError, match="schema_version"):
        parse_manifest(
            {
                "schema_version": 2,
                "embedding_space": SPACE,
                "name_accuracy": _na_block([str(uuid.uuid4())]),
            }
        )


def test_parse_requires_embedding_space() -> None:
    with pytest.raises(ManifestError, match="embedding_space"):
        parse_manifest({"schema_version": 1, "name_accuracy": _na_block([str(uuid.uuid4())])})
    with pytest.raises(ManifestError, match="embedding_space"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": "",
                "name_accuracy": _na_block([str(uuid.uuid4())]),
            }
        )


def test_parse_requires_at_least_one_lane() -> None:
    with pytest.raises(ManifestError, match="at least one"):
        parse_manifest({"schema_version": 1, "embedding_space": SPACE})


def test_parse_rejects_bad_truth_anchoring() -> None:
    with pytest.raises(ManifestError, match="truth_anchoring"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": {"truth_anchoring": "sideways", "run_ids": [str(uuid.uuid4())]},
            }
        )


def test_parse_rejects_empty_and_duplicate_run_ids() -> None:
    with pytest.raises(ManifestError, match="must not be empty"):
        parse_manifest(
            {"schema_version": 1, "embedding_space": SPACE, "name_accuracy": _na_block([])}
        )
    dup = str(uuid.uuid4())
    with pytest.raises(ManifestError, match="duplicate id"):
        parse_manifest(
            {"schema_version": 1, "embedding_space": SPACE, "name_accuracy": _na_block([dup, dup])}
        )


def test_parse_rejects_bad_uuid() -> None:
    with pytest.raises(ManifestError, match="not a valid UUID"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": _na_block(["not-a-uuid"]),
            }
        )


def test_parse_rejects_bad_kind() -> None:
    with pytest.raises(ManifestError, match="kind"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "agreement": {"runs": [{"run_id": str(uuid.uuid4()), "kind": "golden"}]},
            }
        )


def test_parse_curated_requires_host() -> None:
    with pytest.raises(ManifestError, match="host_id"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "agreement": {"runs": [{"run_id": str(uuid.uuid4()), "kind": "curated"}]},
            }
        )


def test_parse_negative_control_rejects_host() -> None:
    with pytest.raises(ManifestError, match="must not carry a host_id"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "agreement": {
                    "runs": [
                        {
                            "run_id": str(uuid.uuid4()),
                            "kind": "negative_control",
                            "host_id": str(uuid.uuid4()),
                        }
                    ]
                },
            }
        )


def test_parse_rejects_empty_agreement_runs() -> None:
    with pytest.raises(ManifestError, match="non-empty list"):
        parse_manifest(
            {"schema_version": 1, "embedding_space": SPACE, "agreement": {"runs": []}}
        )


def test_parse_rejects_duplicate_agreement_run() -> None:
    dup = str(uuid.uuid4())
    with pytest.raises(ManifestError, match="duplicate run id"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "agreement": {
                    "runs": [
                        {"run_id": dup, "kind": "negative_control"},
                        {"run_id": dup, "kind": "negative_control"},
                    ]
                },
            }
        )


def test_parse_rejects_non_object() -> None:
    with pytest.raises(ManifestError, match="expected a JSON object"):
        parse_manifest([1, 2, 3])


def test_parse_rejects_non_string_uuid() -> None:
    with pytest.raises(ManifestError, match="expected a UUID string"):
        parse_manifest(
            {"schema_version": 1, "embedding_space": SPACE, "name_accuracy": _na_block([123])}
        )


def test_parse_rejects_non_list_run_ids() -> None:
    with pytest.raises(ManifestError, match="expected a list"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": {"truth_anchoring": ANCHOR, "run_ids": "nope"},
            }
        )


def test_parse_rejects_non_string_truth_anchoring() -> None:
    with pytest.raises(ManifestError, match="truth_anchoring"):
        parse_manifest(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": {"truth_anchoring": 7, "run_ids": [str(uuid.uuid4())]},
            }
        )


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def test_dumps_matches_harness_serialization() -> None:
    payload = {"z": 1, "a": {"nested": [3, 2, 1]}, "unicode": "Zoë"}
    assert drv._dumps(payload) == harness_dumps(payload)


def test_dumps_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        drv._dumps({"x": float("inf")})


def test_dump_jsonl_is_newline_terminated() -> None:
    text = drv._dump_jsonl([{"b": 1}, {"a": 2}])
    lines = text.splitlines()
    assert text.endswith("\n")
    assert [json.loads(line) for line in lines] == [{"b": 1}, {"a": 2}]
    # sorted keys within each record
    assert lines[0] == '{"b": 1}'


def test_write_atomic_creates_dirs_and_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.json"
    drv._write_atomic(target, '{"ok": true}')
    assert target.read_text() == '{"ok": true}'
    # no temp files left behind
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


# --------------------------------------------------------------------------- #
# write_artifacts
# --------------------------------------------------------------------------- #
def test_write_artifacts_writes_all(tmp_path: Path) -> None:
    written = write_artifacts({"a.json": "1\n", "b.jsonl": "2\n"}, tmp_path / "out")
    assert [p.name for p in written] == ["a.json", "b.jsonl"]
    assert (tmp_path / "out" / "a.json").read_text() == "1\n"


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def test_git_sha_none_outside_repo(tmp_path: Path) -> None:
    assert drv._git_sha(tmp_path) is None


def test_git_sha_present_in_repo() -> None:
    sha = drv._git_sha(drv._REPO_ROOT)
    assert sha is not None and len(sha) == 40


# --------------------------------------------------------------------------- #
# CLI error paths (fail closed before any database access)
# --------------------------------------------------------------------------- #
def test_main_bad_manifest_returns_2(
    no_db: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "m.json"
    bad.write_text("{ not json")
    assert main(["--manifest", str(bad), "--out-dir", str(tmp_path / "o")]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_main_missing_manifest_returns_2(
    no_db: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--manifest", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")]) == 2
    assert "cannot read manifest" in capsys.readouterr().err


def test_main_invalid_manifest_returns_2(
    no_db: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Valid JSON that fails schema validation is a distinct pre-DB exit from a
    # JSON syntax error: main() surfaces the ManifestError as exit 2 before the
    # engine is built.
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "embedding_space": SPACE,
                "name_accuracy": _na_block([str(uuid.uuid4())]),
            }
        )
    )
    assert main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "o")]) == 2
    assert "schema_version" in capsys.readouterr().err


def test_main_refuses_dirty_tree(
    no_db: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": {"truth_anchoring": "independent", "run_ids": [str(uuid.uuid4())]},
            }
        )
    )
    monkeypatch.setattr(drv, "_git_sha", lambda repo: "a" * 40)
    monkeypatch.setattr(drv, "_git_tree_dirty", lambda repo: True)
    # A dirty tree must fail closed before the database is touched.
    assert main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "o")]) == 2
    assert "uncommitted tracked changes" in capsys.readouterr().err
