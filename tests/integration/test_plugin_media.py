"""The plugin audio accessor against a real Postgres (issue #137).

``run_audio_descriptor`` is the hardened seam a plugin uses to reach a run's
processed audio. It returns a confined descriptor only for a COMPLETED run whose
preprocessed-audio artifact resolves to a regular file inside ``media_root``;
everything else fails closed, and a reclaimed intermediate is reported, not
served. These cases mirror the reclamation / path-safety coverage the GC sweep
carries, from the consuming side.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from voxint.plugins.media import (
    AudioMissing,
    AudioUnconfined,
    RunNotCompleted,
    run_audio_descriptor,
)

WAV_BYTES = b"RIFFmock-normalized-wav-payload"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    return root


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def _seed_run(session: Session, *, status: str = "completed") -> uuid.UUID:
    rid, mid = uuid.uuid4(), uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
    )
    session.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision)"
            " VALUES (:rid, :mid, :st, 0)"
        ),
        {"rid": rid, "mid": mid, "st": status},
    )
    session.commit()
    return rid


def _seed_artifact(
    session: Session,
    media_root: Path,
    run_id: uuid.UUID,
    *,
    rel: str | None = None,
    write_file: bool = True,
    reclaimed_bytes: int | None = None,
) -> str:
    rel = rel or f"artifacts/{run_id}/normalized.wav"
    reclaimed = reclaimed_bytes is not None
    session.execute(
        text(
            "INSERT INTO audio_artifacts (id, pipeline_run_id, kind, path,"
            " reclaimed_at, reclaimed_bytes)"
            " VALUES (:id, :rid, 'preprocessed_audio', :p, :ra, :rb)"
        ),
        {
            "id": uuid.uuid4(),
            "rid": run_id,
            "p": rel,
            "ra": datetime.now(tz=UTC) if reclaimed else None,
            "rb": reclaimed_bytes,
        },
    )
    session.commit()
    if write_file:
        target = media_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(WAV_BYTES)
    return rel


def test_descriptor_for_completed_run(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    rel = _seed_artifact(session, media_root, rid)
    desc = run_audio_descriptor(session, rid, media_root=media_root)
    assert desc.run_id == rid
    assert desc.media_relative_path == rel
    assert desc.size_bytes == len(WAV_BYTES)
    assert desc.reclaimed is False


def test_reclaimed_is_reported_not_raised(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, write_file=False, reclaimed_bytes=12345)
    desc = run_audio_descriptor(session, rid, media_root=media_root)
    assert desc.reclaimed is True
    assert desc.size_bytes == 12345


def test_non_completed_run_fails_closed(session: Session, media_root: Path) -> None:
    rid = _seed_run(session, status="running")
    _seed_artifact(session, media_root, rid)
    with pytest.raises(RunNotCompleted):
        run_audio_descriptor(session, rid, media_root=media_root)


def test_unknown_run_fails_closed(session: Session, media_root: Path) -> None:
    with pytest.raises(RunNotCompleted):
        run_audio_descriptor(session, uuid.uuid4(), media_root=media_root)


def test_missing_artifact_fails_closed(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    with pytest.raises(AudioMissing):
        run_audio_descriptor(session, rid, media_root=media_root)


def test_multiple_artifacts_is_missing(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, rel=f"artifacts/{rid}/a.wav")
    _seed_artifact(session, media_root, rid, rel=f"artifacts/{rid}/b.wav")
    with pytest.raises(AudioMissing):
        run_audio_descriptor(session, rid, media_root=media_root)


def test_escaping_path_is_unconfined(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, rel="../outside.wav", write_file=False)
    with pytest.raises(AudioUnconfined):
        run_audio_descriptor(session, rid, media_root=media_root)
