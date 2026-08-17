"""Migration 0021 (waveform_peaks artifact kind), up/down + constraints (issue #57).

Real alembic up/down against the shared test database (head restored in
teardown): the widened kind CHECK accepts 'waveform_peaks' and still rejects
unknown kinds, the partial unique index enforces one peaks row per run while
leaving other kinds unconstrained, downgrade removes the rows it must, and the
model's ArtifactKind enum stays in lockstep with the live CHECK.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from voxint.db.models import ArtifactKind

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def alembic_cfg(engine: Engine) -> Iterator[Config]:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    try:
        yield cfg
    finally:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        command.upgrade(cfg, "head")


def _seed_run(engine: Engine) -> uuid.UUID:
    media_id, run_id = uuid.uuid4(), uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO media_items (id, source_path) VALUES (:id, :p)"),
            {"id": media_id, "p": f"incoming/{media_id}.wav"},
        )
        conn.execute(
            text(
                "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
                " VALUES (:id, :m, 'completed', 0)"
            ),
            {"id": run_id, "m": media_id},
        )
        conn.commit()
    return run_id


def _insert_artifact(engine: Engine, run_id: uuid.UUID, kind: str) -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO audio_artifacts (id, pipeline_run_id, kind, path)"
                " VALUES (:id, :r, :k, :p)"
            ),
            {"id": uuid.uuid4(), "r": run_id, "k": kind, "p": f"artifacts/{run_id}/x"},
        )
        conn.commit()


def test_upgrade_accepts_waveform_peaks_and_rejects_unknown(engine: Engine) -> None:
    run_id = _seed_run(engine)
    _insert_artifact(engine, run_id, "waveform_peaks")
    with pytest.raises(IntegrityError):
        _insert_artifact(engine, run_id, "definitely_not_a_kind")


def test_partial_unique_index_scopes_to_waveform_peaks(engine: Engine) -> None:
    run_id = _seed_run(engine)
    # Multiple chunk rows for one run stay legal (the index is partial).
    _insert_artifact(engine, run_id, "chunk")
    _insert_artifact(engine, run_id, "chunk")
    _insert_artifact(engine, run_id, "waveform_peaks")
    with pytest.raises(IntegrityError):
        _insert_artifact(engine, run_id, "waveform_peaks")
    # A second run gets its own peaks row.
    other = _seed_run(engine)
    _insert_artifact(engine, other, "waveform_peaks")


def test_downgrade_removes_rows_index_and_check(
    engine: Engine, alembic_cfg: Config
) -> None:
    run_id = _seed_run(engine)
    _insert_artifact(engine, run_id, "waveform_peaks")
    _insert_artifact(engine, run_id, "preprocessed_audio")

    command.downgrade(alembic_cfg, "0020")
    with engine.connect() as conn:
        kinds = conn.execute(
            text("SELECT kind FROM audio_artifacts WHERE pipeline_run_id = :r"),
            {"r": run_id},
        ).scalars().all()
        assert kinds == ["preprocessed_audio"]  # peaks row deleted, others kept
        indexes = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'audio_artifacts'"
            )
        ).scalars().all()
        assert "uq_audio_artifacts_waveform_peaks" not in indexes
    with pytest.raises(IntegrityError):
        _insert_artifact(engine, run_id, "waveform_peaks")  # narrowed CHECK

    command.upgrade(alembic_cfg, "head")
    _insert_artifact(engine, run_id, "waveform_peaks")


def test_model_enum_matches_live_check(engine: Engine) -> None:
    # Lockstep guard: every ArtifactKind value must be accepted by the live
    # CHECK, and the CHECK definition must mention each one — so adding an enum
    # value without a migration (or vice versa) fails here.
    with engine.connect() as conn:
        definition = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint"
                " WHERE conname = 'audio_artifacts_kind_check'"
            )
        ).scalar_one()
    for kind in ArtifactKind:
        assert f"'{kind.value}'" in definition, (kind, definition)
    run_id = _seed_run(engine)
    for kind in ArtifactKind:
        if kind is not ArtifactKind.WAVEFORM_PEAKS:
            _insert_artifact(engine, run_id, kind.value)
