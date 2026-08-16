"""Media retention / GC sweep (issue #15), against a real Postgres.

Covers the reclamation core end-to-end: eligibility (terminal status + age),
the source-alias and tutorial-run guards, idempotence, missing-file tolerance,
path safety (symlink / escape fail closed), per-row failure isolation, batching
with oldest-first ordering, and the FOR UPDATE SKIP LOCKED concurrency claim.
"""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from voxint.media.reclaim import (
    configured_tutorial_run_id,
    reclaim_expired_intermediates,
    run_intermediate_reclaimed_at,
)

CUTOFF = datetime.now(tz=UTC) - timedelta(days=1)
WAV_BYTES = b"RIFFmock-normalized-wav-payload"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path / "media"


def _seed_run(
    session: Session,
    *,
    status: str = "completed",
    age_days: int = 10,
    run_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Insert a media item + run with an explicit (past) updated_at."""
    rid = run_id or uuid.uuid4()
    mid = uuid.uuid4()
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": mid, "sp": f"incoming/{mid}/source"},
    )
    session.execute(
        text(
            "INSERT INTO pipeline_runs (id, media_item_id, status, revision,"
            " created_at, updated_at)"
            " VALUES (:rid, :mid, :st, 0, now() - make_interval(days => :d),"
            " now() - make_interval(days => :d))"
        ),
        {"rid": rid, "mid": mid, "st": status, "d": age_days},
    )
    session.commit()
    return rid


def _seed_artifact(
    session: Session,
    media_root: Path,
    run_id: uuid.UUID,
    *,
    kind: str = "preprocessed_audio",
    rel: str | None = None,
    write_file: bool = True,
    reclaimed: bool = False,
) -> Path:
    rel = rel or f"artifacts/{run_id}/normalized.wav"
    session.execute(
        text(
            "INSERT INTO audio_artifacts (id, pipeline_run_id, kind, path,"
            " reclaimed_at, reclaimed_bytes)"
            " VALUES (:id, :rid, :k, :p, :ra, :rb)"
        ),
        {
            "id": uuid.uuid4(),
            "rid": run_id,
            "k": kind,
            "p": rel,
            "ra": datetime.now(tz=UTC) if reclaimed else None,
            "rb": 0 if reclaimed else None,
        },
    )
    session.commit()
    target = media_root / rel
    if write_file:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(WAV_BYTES)
    return target


def _sweep(
    session: Session,
    media_root: Path,
    *,
    batch_limit: int = 500,
    tutorial_run_id: uuid.UUID | None = None,
):  # type: ignore[no-untyped-def]
    return reclaim_expired_intermediates(
        session,
        media_root=media_root,
        cutoff=CUTOFF,
        batch_limit=batch_limit,
        tutorial_run_id=tutorial_run_id,
    )


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def test_reclaims_expired_terminal_runs(session: Session, media_root: Path) -> None:
    for status in ("completed", "cancelled"):
        rid = _seed_run(session, status=status)
        wav = _seed_artifact(session, media_root, rid)
        summary = _sweep(session, media_root)
        assert summary.reclaimed == 1
        assert summary.bytes == len(WAV_BYTES)
        assert not wav.exists()
        assert run_intermediate_reclaimed_at(session, rid) is not None
        row = session.execute(
            text("SELECT reclaimed_bytes FROM audio_artifacts WHERE pipeline_run_id = :r"),
            {"r": rid},
        ).one()
        assert row[0] == len(WAV_BYTES)


def test_non_terminal_runs_are_never_reclaimed(session: Session, media_root: Path) -> None:
    wavs = []
    for status in ("queued", "running", "awaiting_adjudication", "failed"):
        rid = _seed_run(session, status=status)
        wavs.append(_seed_artifact(session, media_root, rid))
    summary = _sweep(session, media_root)
    assert summary.selected == 0
    assert all(w.exists() for w in wavs)


def test_recent_terminal_run_excluded(session: Session, media_root: Path) -> None:
    rid = _seed_run(session, status="completed", age_days=0)  # updated just now
    wav = _seed_artifact(session, media_root, rid)
    summary = _sweep(session, media_root)
    assert summary.selected == 0
    assert wav.exists()
    assert run_intermediate_reclaimed_at(session, rid) is None


def test_only_preprocessed_audio_kind_is_eligible(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    # A non-intermediate artifact kind on an eligible run is left alone.
    other = _seed_artifact(
        session, media_root, rid, kind="transcript_export", rel=f"artifacts/{rid}/export.json"
    )
    summary = _sweep(session, media_root)
    assert summary.selected == 0
    assert other.exists()


def test_already_reclaimed_is_skipped(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, write_file=False, reclaimed=True)
    summary = _sweep(session, media_root)
    assert summary.selected == 0


def test_idempotent_second_sweep(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid)
    first = _sweep(session, media_root)
    assert first.reclaimed == 1
    second = _sweep(session, media_root)
    assert second.selected == 0


def test_missing_file_tolerated(session: Session, media_root: Path) -> None:
    # Orphan case (prepare's delete-without-unlink, or an interrupted reclaim):
    # the row is stamped, bytes 0, no error.
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, write_file=False)
    summary = _sweep(session, media_root)
    assert summary.missing == 1
    assert summary.reclaimed == 0
    assert summary.bytes == 0
    assert run_intermediate_reclaimed_at(session, rid) is not None


def test_source_alias_is_protected(session: Session, media_root: Path) -> None:
    # An artifact path that also appears as some run's source_path must never be
    # unlinked — that would violate the source-retention guarantee.
    rid = _seed_run(session)
    shared = f"artifacts/{rid}/normalized.wav"
    session.execute(
        text("INSERT INTO media_items (id, source_path) VALUES (:mid, :sp)"),
        {"mid": uuid.uuid4(), "sp": shared},
    )
    session.commit()
    wav = _seed_artifact(session, media_root, rid, rel=shared)
    summary = _sweep(session, media_root)
    assert summary.selected == 0
    assert wav.exists()


def test_tutorial_run_excluded(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    wav = _seed_artifact(session, media_root, rid)
    summary = _sweep(session, media_root, tutorial_run_id=rid)
    assert summary.selected == 0
    assert wav.exists()


def test_symlink_is_not_followed_or_unlinked(
    session: Session, media_root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "precious.dat"
    outside.write_bytes(b"do not touch")
    rid = _seed_run(session)
    rel = f"artifacts/{rid}/normalized.wav"
    session.execute(
        text(
            "INSERT INTO audio_artifacts (id, pipeline_run_id, kind, path)"
            " VALUES (:id, :rid, 'preprocessed_audio', :p)"
        ),
        {"id": uuid.uuid4(), "rid": rid, "p": rel},
    )
    session.commit()
    link = media_root / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    summary = _sweep(session, media_root)
    assert summary.failed == 1
    assert summary.reclaimed == 0
    assert link.is_symlink()  # left in place, not unlinked
    assert outside.exists() and outside.read_bytes() == b"do not touch"
    assert run_intermediate_reclaimed_at(session, rid) is None  # left for review


def test_path_escaping_media_root_fails_closed(session: Session, media_root: Path) -> None:
    rid = _seed_run(session)
    _seed_artifact(session, media_root, rid, rel="../escape/normalized.wav", write_file=False)
    summary = _sweep(session, media_root)
    assert summary.failed == 1
    assert run_intermediate_reclaimed_at(session, rid) is None


def test_batch_limit_oldest_first_then_drains(session: Session, media_root: Path) -> None:
    ages = {10: None, 20: None, 30: None}  # oldest = 30 days
    wavs: dict[int, Path] = {}
    for age in ages:
        rid = _seed_run(session, age_days=age)
        wavs[age] = _seed_artifact(session, media_root, rid)
    first = _sweep(session, media_root, batch_limit=2)
    assert first.reclaimed == 2
    # The two OLDEST (30, 20 days) went first.
    assert not wavs[30].exists() and not wavs[20].exists()
    assert wavs[10].exists()
    second = _sweep(session, media_root, batch_limit=2)
    assert second.reclaimed == 1
    assert not wavs[10].exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission bits")
def test_per_row_failure_isolation(session: Session, media_root: Path) -> None:
    # One row whose file cannot be unlinked (read-only parent dir) fails in
    # isolation; the batch still reclaims the healthy row.
    good = _seed_run(session, age_days=30)
    good_wav = _seed_artifact(session, media_root, good)
    bad = _seed_run(session, age_days=10)
    bad_wav = _seed_artifact(session, media_root, bad)
    bad_wav.parent.chmod(0o500)  # r-x: unlink denied
    try:
        summary = _sweep(session, media_root)
        assert summary.reclaimed == 1
        assert summary.failed == 1
        assert not good_wav.exists()
        assert bad_wav.exists()
    finally:
        bad_wav.parent.chmod(0o700)


def test_for_update_skip_locked_claims_rows(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A row claimed by one in-flight sweep is invisible to a concurrent one:
    # the second selection SKIP LOCKEDs it and comes back empty (no double work).
    from voxint.media.reclaim import _select_eligible

    with session_factory() as seed:
        rid = _seed_run(seed)
        _seed_artifact(seed, media_root, rid)

    with session_factory() as a, session_factory() as b:
        claimed = _select_eligible(a, cutoff=CUTOFF, batch_limit=10, tutorial_run_id=None)
        assert len(claimed) == 1  # a holds the row lock (uncommitted)
        contended = _select_eligible(b, cutoff=CUTOFF, batch_limit=10, tutorial_run_id=None)
        assert contended == []  # b skips the locked row
        a.rollback()


def test_configured_tutorial_run_id_reads_singleton(session: Session) -> None:
    assert configured_tutorial_run_id(session) is None
    rid = _seed_run(session)
    session.execute(
        text(
            "INSERT INTO app_settings (id, onboarding_complete, tutorial_run_id)"
            " VALUES (1, true, :r)"
        ),
        {"r": rid},
    )
    session.commit()
    assert configured_tutorial_run_id(session) == rid
