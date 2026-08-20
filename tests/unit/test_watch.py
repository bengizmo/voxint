"""Unit tests for the pure watch-folder helpers (issue #60): the settle
classification and the sweep-summary shape. No DB or broker involved."""

import os
from pathlib import Path

from voxint.ingest.watch import SettleState, WatchSweepSummary, classify_settle


def _newest(path: Path) -> float:
    st = os.stat(path, follow_symlinks=False)
    return max(st.st_mtime, st.st_ctime)


def test_settled_once_quiescent_long_enough(tmp_path: Path) -> None:
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    now = _newest(f) + 100.0  # 100 s after its newest timestamp
    assert classify_settle(f, now=now, settle_seconds=10) is SettleState.SETTLED


def test_too_fresh_within_the_settle_window(tmp_path: Path) -> None:
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    now = _newest(f) + 5.0
    assert classify_settle(f, now=now, settle_seconds=60) is SettleState.TOO_FRESH


def test_settle_zero_accepts_immediately(tmp_path: Path) -> None:
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    assert classify_settle(f, now=_newest(f), settle_seconds=0) is SettleState.SETTLED


def test_far_future_timestamp_is_settled_not_stranded(tmp_path: Path) -> None:
    # A file whose newest timestamp is implausibly ahead of the clock (a NAS/SMB
    # mount with a fast server clock, or a wrong-date recorder stamp) must NOT be
    # stranded as TOO_FRESH forever — waiting can never make a bogus future settle.
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    now = _newest(f) - 10_000  # "now" is 10 000 s before the file's newest stamp
    assert classify_settle(f, now=now, settle_seconds=60) is SettleState.SETTLED


def test_small_future_skew_still_waits(tmp_path: Path) -> None:
    # A skew smaller than the settle window is plausibly an in-progress copy, so the
    # file still waits normally (it settles once wall-clock passes newest+settle).
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    now = _newest(f) - 30  # newest is 30 s ahead; settle window is 60 s
    assert classify_settle(f, now=now, settle_seconds=60) is SettleState.TOO_FRESH


def test_uses_max_of_mtime_and_ctime_not_mtime_alone(tmp_path: Path) -> None:
    # A file whose mtime was back-dated (a copy tool preserving the source mtime)
    # but whose ctime is recent must NOT be treated as settled — the ctime guard.
    f = tmp_path / "clip.wav"
    f.write_bytes(b"data")
    st = os.stat(f)
    old = st.st_mtime - 10_000
    os.utime(f, (old, old))  # back-date atime+mtime; ctime stays ~now
    fresh_ctime = os.stat(f).st_ctime
    # A realistic "now" is at/after the fresh ctime (ctime can't be in the future).
    # Just inside the settle window of that ctime → TOO_FRESH: proves ctime, not the
    # back-dated mtime (which alone would read as long-settled), drives the decision.
    assert classify_settle(f, now=fresh_ctime + 30, settle_seconds=60) is SettleState.TOO_FRESH
    # and far past both → settled
    assert classify_settle(f, now=fresh_ctime + 100, settle_seconds=60) is SettleState.SETTLED


def test_missing_file_is_skip(tmp_path: Path) -> None:
    assert classify_settle(tmp_path / "gone.wav", now=1e12, settle_seconds=0) is SettleState.SKIP


def test_directory_is_skip(tmp_path: Path) -> None:
    d = tmp_path / "adir"
    d.mkdir()
    assert classify_settle(d, now=1e12, settle_seconds=0) is SettleState.SKIP


def test_symlink_is_skip(tmp_path: Path) -> None:
    target = tmp_path / "real.wav"
    target.write_bytes(b"data")
    link = tmp_path / "link.wav"
    link.symlink_to(target)
    # lstat sees the symlink itself (not a regular file) → SKIP, never followed.
    assert classify_settle(link, now=1e12, settle_seconds=0) is SettleState.SKIP


def test_summary_defaults_all_zero_never_run() -> None:
    d = WatchSweepSummary().as_dict()
    assert d == {
        "picked_up": 0,
        "already_known": 0,
        "settling": 0,
        "deferred": 0,
        "stat_errors": 0,
        "sidecar_errors": 0,
        "hit_entry_cap": False,
        "hit_file_cap": False,
        "root_missing": False,
        "completed_at": None,
    }


def test_summary_round_trips_values() -> None:
    d = WatchSweepSummary(
        picked_up=3, already_known=12, settling=2, completed_at="2026-08-18T10:42:00+00:00"
    ).as_dict()
    assert d["picked_up"] == 3
    assert d["already_known"] == 12
    assert d["settling"] == 2
    assert d["completed_at"] == "2026-08-18T10:42:00+00:00"
