"""Tests for the ``tools/export_match_evidence.py`` maintainer driver (#113).

The exporters themselves are covered in ``test_harness_export.py``; this suite
covers the driver's own responsibilities: fail-closed manifest parsing, the
lane -> file mapping, deterministic serialization byte-identical to the score
harness, atomic writes, and a full round trip from a manifest through the driver
into the real ``voxint score`` commands.
"""

import contextlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tools import export_match_evidence as drv
from tools.export_match_evidence import (
    AgreementLane,
    AgreementRun,
    Manifest,
    ManifestError,
    NameAccuracyLane,
    build_artifacts,
    main,
    parse_manifest,
    write_artifacts,
)

# Reuse the exporter suite's DB builders so the driver is exercised over the same
# fixtures the exporters are validated against.
from tests.integration.test_harness_export import (
    E0,
    SPACE,
    _grounded_run,
    add_speaker,
    add_turn,
    make_run,
    run_matcher,
)
from voxint.cli import main as voxint_main
from voxint.config import Settings
from voxint.harness.score_cli import _dumps as harness_dumps
from voxint.harness_export import ExportError, TruthAnchoring

ANCHOR = TruthAnchoring.INDEPENDENT.value


@pytest.fixture()
def session(session_factory: sessionmaker[Session]):  # type: ignore[no-untyped-def]
    with session_factory() as s:
        yield s


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
# build_artifacts
# --------------------------------------------------------------------------- #
def _agreement_manifest(
    curated: uuid.UUID, host: uuid.UUID, negative: uuid.UUID
) -> Manifest:
    return Manifest(
        embedding_space=SPACE,
        name_accuracy=None,
        agreement=AgreementLane(
            runs=(
                AgreementRun(curated, "curated", host),
                AgreementRun(negative, "negative_control", None),
            ),
            roster_speaker_ids=None,
        ),
    )


def test_build_name_accuracy_only(session: Session) -> None:
    run_id, _alice = _grounded_run(session)
    manifest = Manifest(
        embedding_space=SPACE,
        name_accuracy=NameAccuracyLane(TruthAnchoring.INDEPENDENT, (run_id,)),
        agreement=None,
    )
    artifacts = build_artifacts(
        session, Settings(), manifest, exported_at="2026-08-20T00:00:00+00:00", git_sha="abc123"
    )
    assert set(artifacts) == {drv.FILE_SNAPSHOT, drv.FILE_NAME_ACCURACY_ITEMS}
    snap = json.loads(artifacts[drv.FILE_SNAPSHOT])
    assert snap["code"]["git_sha"] == "abc123"
    assert snap["run_ids"] == [str(run_id)]
    items = [json.loads(line) for line in artifacts[drv.FILE_NAME_ACCURACY_ITEMS].splitlines()]
    assert len(items) == 1
    assert items[0]["item_id"] == str(run_id)


def test_build_agreement_emits_enrollment_and_slots(session: Session) -> None:
    curated = make_run(session)
    negative = make_run(session)
    alice = add_speaker(session, "Alice", [E0, E0], source_runs=[curated, curated])
    for i in range(4):
        add_turn(session, curated, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    for i in range(4):
        add_turn(session, negative, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()

    manifest = _agreement_manifest(curated, alice, negative)
    artifacts = build_artifacts(
        session, Settings(), manifest, exported_at="2026-08-20T00:00:00+00:00", git_sha=None
    )
    assert set(artifacts) == {
        drv.FILE_SNAPSHOT,
        drv.FILE_ENROLLMENT,
        drv.FILE_AGREEMENT_SLOTS,
    }
    enroll = json.loads(artifacts[drv.FILE_ENROLLMENT])
    assert str(alice) in enroll["voiceprints"]
    slots = [json.loads(line) for line in artifacts[drv.FILE_AGREEMENT_SLOTS].splitlines()]
    kinds = {rec["item_id"]: rec["kind"] for rec in slots}
    assert kinds[str(curated)] == "curated"
    assert kinds[str(negative)] == "negative_control"
    # snapshot git_sha is null when none is injected
    assert json.loads(artifacts[drv.FILE_SNAPSHOT])["code"]["git_sha"] is None


def test_build_both_lanes_union_dedupes_snapshot(session: Session) -> None:
    # A run that appears in BOTH lanes must be deduped for the snapshot, which
    # otherwise rejects a duplicate selection.
    run_id, alice = _grounded_run(session)
    manifest = Manifest(
        embedding_space=SPACE,
        name_accuracy=NameAccuracyLane(TruthAnchoring.INDEPENDENT, (run_id,)),
        agreement=AgreementLane(
            runs=(AgreementRun(run_id, "curated", alice),),
            roster_speaker_ids=None,
        ),
    )
    artifacts = build_artifacts(
        session, Settings(), manifest, exported_at="2026-08-20T00:00:00+00:00", git_sha=None
    )
    assert set(artifacts) == {
        drv.FILE_SNAPSHOT,
        drv.FILE_NAME_ACCURACY_ITEMS,
        drv.FILE_ENROLLMENT,
        drv.FILE_AGREEMENT_SLOTS,
    }
    assert json.loads(artifacts[drv.FILE_SNAPSHOT])["run_ids"] == [str(run_id)]


def test_build_is_deterministic(session: Session) -> None:
    run_id, _alice = _grounded_run(session)
    manifest = Manifest(
        embedding_space=SPACE,
        name_accuracy=NameAccuracyLane(TruthAnchoring.INDEPENDENT, (run_id,)),
        agreement=None,
    )
    first = build_artifacts(
        session, Settings(), manifest, exported_at="2026-08-20T00:00:00+00:00", git_sha="x"
    )
    second = build_artifacts(
        session, Settings(), manifest, exported_at="2026-08-20T00:00:00+00:00", git_sha="x"
    )
    assert first == second


def test_build_propagates_export_error(session: Session) -> None:
    run_id = make_run(session)  # no turns -> no embedding space
    session.flush()
    manifest = Manifest(
        embedding_space=SPACE,
        name_accuracy=None,
        agreement=AgreementLane(
            runs=(AgreementRun(run_id, "negative_control", None),),
            roster_speaker_ids=None,
        ),
    )
    with pytest.raises(ExportError):
        build_artifacts(
            session, Settings(), manifest, exported_at="t", git_sha=None
        )


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
# CLI + round trip through voxint score
# --------------------------------------------------------------------------- #
def test_main_bad_manifest_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "m.json"
    bad.write_text("{ not json")
    assert main(["--manifest", str(bad), "--out-dir", str(tmp_path / "o")]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_main_missing_manifest_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--manifest", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")]) == 2
    assert "cannot read manifest" in capsys.readouterr().err


def test_main_refuses_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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


def test_main_end_to_end_and_score_round_trip(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A curated run grounding onto Alice, enrolled from a separate held-out run.
    enroll_run = make_run(session)
    scored = make_run(session)
    alice = add_speaker(session, "Alice", [E0, E0, E0], source_runs=[enroll_run] * 3)
    for i in range(4):
        add_turn(session, scored, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    run_matcher(session, scored)
    session.commit()

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "embedding_space": SPACE,
                "name_accuracy": {"truth_anchoring": "independent", "run_ids": [str(scored)]},
                "agreement": {
                    "runs": [{"run_id": str(scored), "kind": "curated", "host_id": str(alice)}],
                    "roster_speaker_ids": [str(alice)],
                },
            }
        )
    )
    out_dir = tmp_path / "export"

    # Drive main() against the fixture session. A stub engine keeps main()'s
    # engine.dispose() off the shared, session-scoped test engine.
    monkeypatch.setattr(drv, "build_engine", _StubEngine)
    monkeypatch.setattr(drv, "session_scope", _passthrough_session_scope(session))
    monkeypatch.setattr(drv, "get_settings", Settings)
    monkeypatch.setattr(drv, "_git_tree_dirty", lambda repo: False)

    exit_code = main(
        ["--manifest", str(manifest_path), "--out-dir", str(out_dir), "--git-sha", "deadbeef"]
    )
    assert exit_code == 0

    # All four artifacts exist.
    for name in (
        drv.FILE_SNAPSHOT,
        drv.FILE_NAME_ACCURACY_ITEMS,
        drv.FILE_ENROLLMENT,
        drv.FILE_AGREEMENT_SLOTS,
    ):
        assert (out_dir / name).exists()
    assert json.loads((out_dir / drv.FILE_SNAPSHOT).read_text())["code"]["git_sha"] == "deadbeef"

    # name-accuracy round-trips through the real scorer.
    na_report = tmp_path / "na.json"
    assert (
        voxint_main(
            [
                "score",
                "name-accuracy",
                str(out_dir / drv.FILE_NAME_ACCURACY_ITEMS),
                "--out",
                str(na_report),
            ]
        )
        == 0
    )
    assert json.loads(na_report.read_text())  # a non-empty report

    # agreement round-trips through the real scorer.
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tau": 0.6,
                "margin": 0.05,
                "min_duration": 10.0,
                "min_segments": 3,
                "low_band": 0.3,
                "neg_min_total_duration": 5.0,
                "min_enrollment_items": 2,
            }
        )
    )
    verdicts = tmp_path / "verdicts.jsonl"
    assert (
        voxint_main(
            [
                "score",
                "agreement",
                "--slots",
                str(out_dir / drv.FILE_AGREEMENT_SLOTS),
                "--enrollment",
                str(out_dir / drv.FILE_ENROLLMENT),
                "--thresholds",
                str(thresholds),
                "--out",
                str(verdicts),
            ]
        )
        == 0
    )
    assert verdicts.read_text().strip()  # at least one verdict line


class _StubEngine:
    """Stand-in for the driver's engine so main()'s dispose() is a no-op."""

    def dispose(self) -> None:  # pragma: no cover - trivial
        pass


def _passthrough_session_scope(fixture_session: Session):  # type: ignore[no-untyped-def]
    """A session_scope replacement that hands back the fixture session unclosed.

    The driver opens its own session_scope; in-test we reuse the fixture session
    (already holding the seeded rows) so main() reads what the test wrote without
    a second connection racing the shared test database.
    """

    @contextlib.contextmanager
    def _scope(_factory):  # type: ignore[no-untyped-def]
        yield fixture_session

    return _scope
