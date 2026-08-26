"""DB-backed tests for the ``tools/export_match_evidence.py`` maintainer driver (#113).

The driver's pure responsibilities (fail-closed manifest parsing, deterministic
serialization, atomic writes, git helpers, and the CLI paths that fail closed
before any database access) are covered in the unit suite of the same basename.
This module covers what needs a real ``Session``: ``build_artifacts`` over
seeded runs — the lane -> file mapping and snapshot dedupe — and a full round
trip from a manifest through the driver into the real ``voxint score`` commands.
The exporters themselves are covered in ``test_harness_export.py``.
"""

import contextlib
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tools import export_match_evidence as drv
from tools.export_match_evidence import (
    AgreementLane,
    AgreementRun,
    Manifest,
    NameAccuracyLane,
    build_artifacts,
    main,
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
from voxint.harness_export import ExportError, TruthAnchoring


@pytest.fixture()
def session(session_factory: sessionmaker[Session]):  # type: ignore[no-untyped-def]
    with session_factory() as s:
        yield s


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
# CLI + round trip through voxint score
# --------------------------------------------------------------------------- #
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
