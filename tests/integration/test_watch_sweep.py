"""Watch-folder ingest sweep (issue #60), against a real Postgres.

Drives :func:`voxint.ingest.watch.sweep_watch_folders` directly (the task is a thin
wrapper): the effective-gate recheck, the scan+settle+submit pipeline, the
commit-before-publish via :class:`~voxint.ingest.service.SubmissionResult`, the
"already known" skip, and the persisted status summary. ``SubmissionResult.publish``
is monkeypatched so no broker is needed.
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.api.setup_wizard import ScanResult
from voxint.config import Settings
from voxint.db.models import AppSettings, MediaFolder, MediaItem, PipelineRun, RunStatus
from voxint.ingest.service import SubmissionResult
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
        # Since #153 the sweep walks the media_folders relation (watch=true), not the
        # legacy app_settings column; register a row per folder.
        for folder in folders:
            s.add(MediaFolder(path=folder, watch=True))
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


class _PublishRecorder:
    """Records published run ids; ``ok=False`` simulates a broker outage."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.published: list[uuid.UUID] = []

    def __call__(self, result_self: SubmissionResult) -> bool:
        self.published.append(result_self.run_id)
        return self.ok


@pytest.fixture(autouse=True)
def _mock_publish(monkeypatch: pytest.MonkeyPatch) -> _PublishRecorder:
    """Replace SubmissionResult.publish with a no-broker recorder."""
    recorder = _PublishRecorder()
    monkeypatch.setattr(SubmissionResult, "publish", recorder)
    return recorder


@pytest.fixture()
def publish_recorder(_mock_publish: _PublishRecorder) -> _PublishRecorder:
    """Explicit handle when a test needs to inspect what was published."""
    return _mock_publish


@pytest.fixture()
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as s:
        yield s


def test_new_file_is_ingested_and_published(
    session_factory: sessionmaker[Session], media_root: Path, publish_recorder: _PublishRecorder
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 1
    assert summary.already_known == 0
    assert _media_paths(session_factory) == {f"{FOLDER}/a.wav"}
    runs = _runs(session_factory)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.QUEUED.value
    assert publish_recorder.published == [runs[0].id]


def test_disabled_is_a_noop(
    session_factory: sessionmaker[Session], media_root: Path, publish_recorder: _PublishRecorder
) -> None:
    # env default off AND no runtime override → nothing walked or submitted.
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=None)
    _drop(media_root, "a.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary == WatchSweepSummary()  # all-zero, completed_at None
    assert _runs(session_factory) == []
    assert publish_recorder.published == []


def test_runtime_override_enables_without_env(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # env default off, but the app_settings override flips it on (no restart path).
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root))
    assert summary.picked_up == 1


def test_already_known_file_is_skipped(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    with session_factory() as s:  # pre-claim the path
        s.add(MediaItem(source_path=f"{FOLDER}/a.wav"))
        s.commit()

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 0
    assert summary.already_known == 1
    assert _runs(session_factory) == []  # no run minted for a known file


def test_second_sweep_is_idempotent(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    first = sweep_watch_folders(session_factory, _settings(media_root))
    assert first.picked_up == 1
    second = sweep_watch_folders(session_factory, _settings(media_root))
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
    )
    assert summary.picked_up == 0
    assert summary.settling == 1
    assert _runs(session_factory) == []
    # Back-date it so it's now quiescent; a later sweep picks it up.
    old = os.stat(f).st_mtime - 10_000
    os.utime(f, (old, old))
    # settle=0 ignores the fresh ctime for this assertion (accept immediately).
    later = sweep_watch_folders(session_factory, _settings(media_root))
    assert later.picked_up == 1


def test_broker_outage_defers_but_keeps_queued_run(
    session_factory: sessionmaker[Session],
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish_recorder: _PublishRecorder,
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    publish_recorder.ok = False  # broker down

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 1
    assert summary.deferred == 1
    runs = _runs(session_factory)
    assert len(runs) == 1 and runs[0].status == RunStatus.QUEUED.value  # durable, for recovery
    assert publish_recorder.published == [runs[0].id]  # it WAS attempted


def test_summary_is_persisted_for_the_status_line(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "a.wav")
    _drop(media_root, "b.wav")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

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

    summary = sweep_watch_folders(session_factory, _settings(media_root))

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

    summary = sweep_watch_folders(session_factory, _settings(media_root))

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

    summary = sweep_watch_folders(session_factory, _settings(media_root))

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

    summary = sweep_watch_folders(session_factory, _settings(media_root))

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

    summary = sweep_watch_folders(session_factory, _settings(gone))

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

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert _media_paths(session_factory) == {f"{FOLDER}/good.wav"}
    assert summary.picked_up == 1


# --- YAML sidecars (issue #104) --------------------------------------------------


def _drop_sidecar(media_root: Path, name: str, text: str, *, folder: str = FOLDER) -> Path:
    f = media_root / folder / name
    f.write_text(text, encoding="utf-8")
    return f


def _run_for(factory: sessionmaker[Session], source_path: str) -> PipelineRun:
    with factory() as s:
        media = s.query(MediaItem).filter_by(source_path=source_path).one()
        return s.query(PipelineRun).filter_by(media_item_id=media.id).one()


def test_sidecar_pair_ingests_with_all_fields_applied(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # A pack the sidecar can name explicitly, alongside the default.
    packs = tmp_path / "packs"
    (packs / "hvac").mkdir(parents=True)
    (packs / "hvac" / "manifest.yaml").write_text(
        "name: hvac\nname_seeds: [Pack Seed]\n", encoding="utf-8"
    )
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    _drop_sidecar(
        media_root,
        "talk.wav.yaml",
        "title: Spring keynote\n"
        "speakers: [Jane Doe, Pack Seed]\n"
        "domain_pack: hvac\n"
        "notes: recorded on stage\n"
        "content_item_id: 42\n",
    )

    summary = sweep_watch_folders(
        session_factory,
        _settings(media_root, domain_packs_dir=packs),
    )

    assert summary.picked_up == 1
    assert summary.sidecar_errors == 0
    run = _run_for(session_factory, f"{FOLDER}/talk.wav")
    # The whole mapping (reference keys included) is frozen on the run.
    assert run.sidecar == {
        "title": "Spring keynote",
        "speakers": ["Jane Doe", "Pack Seed"],
        "domain_pack": "hvac",
        "notes": "recorded on stage",
        "content_item_id": 42,
    }
    assert run.operator_notes == "recorded on stage"
    # The sidecar's pack was resolved as the explicit name, and its speakers
    # unioned into the frozen snapshot's name_seeds (pack seeds keep priority,
    # exact-string dedupe drops the repeat of "Pack Seed").
    assert run.domain_pack is not None
    assert run.domain_pack["name"] == "hvac"
    assert run.domain_pack["name_seeds"] == ["Pack Seed", "Jane Doe"]


def test_malformed_sidecar_holds_then_fixed_file_ingests(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    bad = _drop_sidecar(media_root, "talk.wav.yaml", "title: [unclosed\n")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    # The media is HELD, not dropped and not half-ingested.
    assert summary.picked_up == 0
    assert summary.sidecar_errors == 1
    assert _media_paths(session_factory) == set()

    # The operator fixes the sidecar; the next sweep ingests the pair.
    bad.write_text("title: Fixed now\n", encoding="utf-8")
    later = sweep_watch_folders(session_factory, _settings(media_root))
    assert later.picked_up == 1
    assert later.sidecar_errors == 0
    assert _run_for(session_factory, f"{FOLDER}/talk.wav").sidecar == {"title": "Fixed now"}


def test_unknown_pack_in_sidecar_is_a_sidecar_error(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    _drop_sidecar(media_root, "talk.wav.yaml", "domain_pack: no-such-pack\n")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    # Attributed to the SIDECAR (fix that file), not the folder config.
    assert summary.sidecar_errors == 1
    assert summary.stat_errors == 0
    assert _media_paths(session_factory) == set()


def test_reference_only_sidecar_ingests_with_nothing_applied(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "ep.mp3")
    _drop_sidecar(
        media_root,
        "ep.mp3.yaml",
        "content_item_id: 7\nsource_type: rss_feed\npublished: 2026-01-15\n",
    )

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 1
    run = _run_for(session_factory, f"{FOLDER}/ep.mp3")
    # Preserved for provenance (dates ISO-normalized), nothing applied.
    assert run.sidecar == {
        "content_item_id": 7,
        "source_type": "rss_feed",
        "published": "2026-01-15",
    }
    assert run.operator_notes is None
    assert run.domain_pack is not None and run.domain_pack["name"] == "generic"


def test_stem_ambiguity_holds_both_while_full_name_pair_ingests(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    # Ambiguous: one stem sidecar, two same-stem media files.
    _drop(media_root, "clip.wav")
    _drop(media_root, "clip.mp4")
    _drop_sidecar(media_root, "clip.yaml", "title: whose?\n")
    # Unambiguous full-name pair beside them.
    _drop(media_root, "other.wav")
    _drop_sidecar(media_root, "other.wav.yaml", "title: mine\n")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    # Both same-stem media files are held (the sidecar could describe either);
    # the full-name pair ingests untouched.
    assert summary.sidecar_errors == 2
    assert summary.picked_up == 1
    assert _media_paths(session_factory) == {f"{FOLDER}/other.wav"}


def test_sidecar_still_settling_holds_media_as_settling(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Media settled, sidecar still being written → the PAIR waits (settling),
    # never a half-applied ingest. ctime cannot be back-dated, so the settle
    # classifier is faked per-file (the SKIP precedent above).
    import voxint.ingest.watch as watch_mod

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    _drop_sidecar(media_root, "talk.wav.yaml", "title: T\n")

    def _classify(path: Path, *, now: float, settle_seconds: float) -> watch_mod.SettleState:
        if path.name.endswith(".yaml"):
            return watch_mod.SettleState.TOO_FRESH
        return watch_mod.SettleState.SETTLED

    monkeypatch.setattr(watch_mod, "classify_settle", _classify)

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 0
    assert summary.settling == 1
    assert summary.sidecar_errors == 0
    assert _media_paths(session_factory) == set()


def test_sidecar_arriving_after_ingest_is_too_late(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    first = sweep_watch_folders(session_factory, _settings(media_root))
    assert first.picked_up == 1

    # The sidecar shows up late: the media is already known, nothing re-triggers,
    # and the run keeps its NULL sidecar (frozen at submit).
    _drop_sidecar(media_root, "talk.wav.yaml", "title: too late\n")
    later = sweep_watch_folders(session_factory, _settings(media_root))
    assert later.picked_up == 0
    assert later.already_known == 1
    run = _run_for(session_factory, f"{FOLDER}/talk.wav")
    assert run.sidecar is None
    assert run.operator_notes is None


def test_held_sidecar_cannot_starve_later_files_past_the_cap(
    session_factory: sessionmaker[Session], media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The net-new cap applies to SUBMISSIONS, not scan candidates: a permanently
    # held pair at the FRONT of the walk order must not occupy the only cap slot
    # forever. The scan is faked to pin the walk order (scandir order is
    # arbitrary); the files are real.
    import voxint.ingest.watch as watch_mod

    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "bad.wav")
    _drop_sidecar(media_root, "bad.wav.yaml", "title: [unclosed\n")
    _drop(media_root, "good.wav")
    _drop(media_root, "extra.wav")

    def _scan(*a: object, **kw: object) -> ScanResult:
        assert kw.get("apply_file_cap") is False  # the sweep must opt out
        return ScanResult(
            candidates=[f"{FOLDER}/bad.wav", f"{FOLDER}/good.wav", f"{FOLDER}/extra.wav"],
            inspected=3,
            hit_entry_cap=False,
            hit_file_cap=False,
            root_missing=False,
            already_known=0,
        )

    monkeypatch.setattr(watch_mod, "scan_media_folders", _scan)

    summary = sweep_watch_folders(
        session_factory,
        _settings(media_root, setup_scan_max_files=1),
    )

    # bad is held WITHOUT consuming the single cap slot; good takes it; extra
    # waits for the next sweep behind the honest hit_file_cap flag.
    assert summary.sidecar_errors == 1
    assert summary.picked_up == 1
    assert summary.hit_file_cap is True
    assert _media_paths(session_factory) == {f"{FOLDER}/good.wav"}


def test_blank_sidecar_file_ingests_and_stamps_empty_mapping(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A comment-only sidecar is a valid stub: the pair ingests, nothing is
    # applied, and the run records {} — distinct from NULL (no sidecar at all).
    _seed_settings_row(session_factory, folders=[FOLDER], enabled=True)
    _drop(media_root, "talk.wav")
    _drop_sidecar(media_root, "talk.wav.yaml", "# nothing for Voxint yet\n")

    summary = sweep_watch_folders(session_factory, _settings(media_root))

    assert summary.picked_up == 1
    assert summary.sidecar_errors == 0
    run = _run_for(session_factory, f"{FOLDER}/talk.wav")
    assert run.sidecar == {}
    assert run.operator_notes is None
