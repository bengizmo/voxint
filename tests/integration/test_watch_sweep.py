"""Watch-folder ingest sweep (issue #60), against a real Postgres.

Drives :func:`voxint.ingest.watch.sweep_watch_folders` directly (the task is a thin
wrapper): the effective-gate recheck, the scan+settle+submit pipeline, the
commit-before-publish + broker-defer path, the "already known" skip, and the
persisted status summary. Publishing is a fake callable so no broker is needed.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.setup_wizard import ScanResult
from voxint.config import Settings
from voxint.db.models import AppSettings, MediaItem, PipelineRun, RunStatus
from voxint.ingest.watch import WatchSweepSummary, _store_summary, sweep_watch_folders

FOLDER = "clips"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    (root / FOLDER).mkdir(parents=True)
    return root


def _settings(media_root: Path, **over: object) -> Settings:
    over.setdefault("watch_folder_settle_seconds", 0)  # immediate unless overridden
    return Settings(_env_file=None, media_root=media_root, **over)


def _seed_settings_row(
    factory: sessionmaker[Session],
    *,
    folders: list[str],
    enabled: bool | None,
) -> None:
    with factory() as s:
        s.add(
            AppSettings(
                id=1,
                onboarding_complete=True,
                media_folders=folders,
                watch_folder_enabled=enabled,
            )
        )
        s.commit()


def _drop(media_root: Path, name: str, *, folder: str = FOLDER) -> Path:
    f = media_root / folder / name
    f.write_bytes(b"RIFF" + name.encode())
    return f


def _runs(factory: sessionmaker[Session]) -> list[PipelineRun]:
    with factory() as s:
        return list(s.query(PipelineRun).all())


def _media_paths(factory: sessionmaker[Session]) -> set[str]:
    with factory() as s:
        return {m.source_path for m in s.query(MediaItem).all()}


class _Publisher:
    """Records published run ids; ``ok=False`` simulates a broker outage."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.published: list[uuid.UUID] = []

    def __call__(self, run_id: uuid.UUID) -> bool:
        self.published.append(run_id)
        return self.ok


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def test_new_file_is_ingested_and_published(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    pub = _Publisher()

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=pub)

    assert summary.picked_up == 1
    assert summary.already_known == 0
    assert _media_paths(session_factory) == {f"{FOLDER}/a.wav"}
    runs = _runs(session_factory)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.QUEUED.value
    assert pub.published == [runs[0].id]  # commit-before-publish handed off the run


def test_disabled_is_a_noop(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # env default off AND no runtime override → nothing walked or submitted.
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=None)
    _drop(media_root, "a.wav")
    pub = _Publisher()

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=pub)

    assert summary == WatchSweepSummary()  # all-zero, completed_at None
    assert _runs(session_factory) == []
    assert pub.published == []


def test_runtime_override_enables_without_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # env default off, but the app_settings override flips it on (no restart path).
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())
    assert summary.picked_up == 1


def test_already_known_file_is_skipped(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    with session_factory() as s:  # pre-claim the path
        s.add(MediaItem(source_path=f"{FOLDER}/a.wav"))
        s.commit()
    pub = _Publisher()

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=pub)

    assert summary.picked_up == 0
    assert summary.already_known == 1
    assert _runs(session_factory) == []  # no run minted for a known file
    assert pub.published == []


def test_second_sweep_is_idempotent(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    first = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())
    assert first.picked_up == 1
    second = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())
    assert second.picked_up == 0
    assert second.already_known == 1
    assert len(_runs(session_factory)) == 1  # no duplicate run


def test_too_fresh_file_is_settling_then_ingested(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    f = _drop(media_root, "a.wav")
    # A long settle window → the just-dropped file is still copying this pass.
    summary = sweep_watch_folders(
        session_factory,
        _settings(media_root, watch_folder_settle_seconds=3600),
        publish=_Publisher(),
    )
    assert summary.picked_up == 0
    assert summary.settling == 1
    assert _runs(session_factory) == []
    # Back-date it so it's now quiescent; a later sweep picks it up.
    old = os.stat(f).st_mtime - 10_000
    os.utime(f, (old, old))
    # settle=0 ignores the fresh ctime for this assertion (accept immediately).
    later = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())
    assert later.picked_up == 1


def test_broker_outage_defers_but_keeps_queued_run(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    pub = _Publisher(ok=False)  # broker down

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=pub)

    assert summary.picked_up == 1
    assert summary.deferred == 1
    runs = _runs(session_factory)
    assert len(runs) == 1 and runs[0].status == RunStatus.QUEUED.value  # durable, for recovery
    assert pub.published == [runs[0].id]  # it WAS attempted


def test_summary_is_persisted_for_the_status_line(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    _drop(media_root, "b.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    with session_factory() as s:
        stored = s.get(AppSettings, 1)
        assert stored is not None
        assert stored.watch_folder_last_sweep is not None
        assert stored.watch_folder_last_sweep["picked_up"] == 2
        assert stored.watch_folder_last_sweep["completed_at"] == summary.completed_at


def test_vanished_candidate_is_counted_not_fatal(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate that disappears / becomes unreadable between the scan and the
    # settle stat is counted (stat_errors) and skipped, never crashing the sweep.
    import voxint.ingest.watch as watch_mod

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    monkeypatch.setattr(watch_mod, "classify_settle", lambda *a, **k: watch_mod.SettleState.SKIP)

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    assert summary.picked_up == 0
    assert summary.stat_errors == 1
    assert _runs(session_factory) == []


def test_domain_pack_collision_is_skipped_not_fatal(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A freeze-time domain-pack collision (issue #84) / unresolvable pack is a
    # PERSISTENT operator config error. It must be logged + counted (stat_errors) and
    # the file skipped — never propagate out and crash the recurring beat sweep (which
    # would silently stop ingesting every folder). Issue #84 review finding.
    import voxint.ingest.watch as watch_mod
    from voxint.domain_packs.base import DomainPackError

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "bad.wav")
    _drop(media_root, "good.wav")

    real_submit = watch_mod.submit_media_item_if_new

    def _submit(session: object, rel: str, **kw: object) -> object:
        if rel.endswith("bad.wav"):
            raise DomainPackError("domain pack corrections are not idempotent: rule 'b'")
        return real_submit(session, rel, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(watch_mod, "submit_media_item_if_new", _submit)

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    # The good file still ingests; the colliding one is counted, not fatal.
    assert summary.picked_up == 1
    assert summary.stat_errors == 1
    assert _media_paths(session_factory) == {f"{FOLDER}/good.wav"}


def test_race_loss_at_submit_counts_as_already_known(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If another sweep/submission claims a settled path between the scan and the
    # submit, submit_media_item_if_new returns None — counted "already known", not
    # picked up, and no run is minted.
    import voxint.ingest.watch as watch_mod

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    monkeypatch.setattr(watch_mod, "submit_media_item_if_new", lambda *a, **k: None)

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    assert summary.picked_up == 0
    assert summary.already_known == 1
    assert _runs(session_factory) == []


def test_real_unique_conflict_at_submit_does_not_poison_the_batch(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The REAL race path (not a mock of submit): a candidate the scan reported as
    # net-new already has a MediaItem by submit time, so the actual UNIQUE(source_path)
    # violation fires. submit_media_item_if_new contains it in a SAVEPOINT, so the loser
    # rolls back to None WITHOUT poisoning the batch — the genuinely-new sibling in the
    # same sweep still commits its run. Forcing the scan to report the pre-claimed path
    # as net-new is how we drive the conflict deterministically.
    import voxint.ingest.watch as watch_mod

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    _drop(media_root, "b.wav")
    with session_factory() as s:  # a.wav already claimed; scan below still reports it net-new
        s.add(MediaItem(source_path=f"{FOLDER}/a.wav"))
        s.commit()
    monkeypatch.setattr(
        watch_mod,
        "scan_media_folders",
        lambda *a, **k: ScanResult(
            candidates=[f"{FOLDER}/a.wav", f"{FOLDER}/b.wav"],
            inspected=2,
            hit_entry_cap=False,
            hit_file_cap=False,
            root_missing=False,
            already_known=0,
        ),
    )

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    assert summary.picked_up == 1  # b.wav survived the batch despite a's real conflict
    assert summary.already_known == 1  # a.wav's UNIQUE conflict → counted, not raised
    assert _media_paths(session_factory) == {f"{FOLDER}/a.wav", f"{FOLDER}/b.wav"}
    runs = _runs(session_factory)
    assert len(runs) == 1  # exactly one run minted — for b.wav, not the pre-claimed a.wav


def test_missing_media_root_is_surfaced_not_a_silent_empty_sweep(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # An unmounted drive / share (media root gone) must read as an honest problem,
    # not a healthy "picked up 0 new files" — root_missing rides into the summary.
    gone = tmp_path / "unmounted"  # never created
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)

    summary = sweep_watch_folders(session_factory, _settings(gone), publish=_Publisher())

    assert summary.root_missing is True
    assert summary.picked_up == 0
    assert _runs(session_factory) == []


def test_store_summary_newest_wins_ignores_an_out_of_order_staler_sweep(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Two overlapping sweeps: the guarded, row-locked write must keep the FRESHER
    # summary even when a staler one commits afterwards (an out-of-order finisher).
    settings = _settings(media_root)
    fresh = WatchSweepSummary(picked_up=5, completed_at="2026-08-18T12:00:00+00:00")
    stale = WatchSweepSummary(picked_up=1, completed_at="2026-08-18T10:00:00+00:00")
    newer = WatchSweepSummary(picked_up=9, completed_at="2026-08-18T14:00:00+00:00")

    _store_summary(session_factory, settings, fresh)
    _store_summary(session_factory, settings, stale)  # older → must NOT clobber
    with session_factory() as s:
        assert s.get(AppSettings, 1).watch_folder_last_sweep["picked_up"] == 5

    _store_summary(session_factory, settings, newer)  # newer → wins
    with session_factory() as s:
        assert s.get(AppSettings, 1).watch_folder_last_sweep["picked_up"] == 9


def test_reserved_and_symlink_paths_are_not_ingested(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # Registering the media root itself must not ingest the incoming/ tree Voxint
    # owns, and a symlinked media file is never followed (scan_media_folders guards).
    _seed_settings_row(session_factory, folders=["."], enabled=True)
    (media_root / "incoming").mkdir()
    _drop(media_root, "owned.wav", folder="incoming")
    real = tmp_path / "outside.wav"  # OUTSIDE the media root, so only the symlink is seen
    real.write_bytes(b"RIFFreal")
    (media_root / FOLDER / "link.wav").symlink_to(real)
    _drop(media_root, "good.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root), publish=_Publisher())

    assert _media_paths(session_factory) == {f"{FOLDER}/good.wav"}
    assert summary.picked_up == 1
