"""Unit tests for the yt-dlp subprocess boundary (no network, no yt-dlp binary).

``run_download_command`` and the argv ``build_ytdlp_downloader`` produces are
exercised with harmless stand-in commands (``true``/``sh``/a recording script),
so the process-group timeout kill and the flag lockdown are tested deterministically.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from voxint.media.ytdlp import (
    AcquisitionError,
    _isolate_child,
    _kill_process_group,
    build_ytdlp_downloader,
    run_download_command,
)
from voxint.pipeline.engine import StageFailedError
from voxint.pipeline.stages.context import StageDataError
from voxint.worker.tasks import retryable_cause


def test_success_returns_none() -> None:
    # `true` exits 0 and writes nothing; the caller inspects the dir, not us.
    run_download_command(["true"], timeout_seconds=5)


def test_nonzero_exit_raises_with_stderr_tail() -> None:
    with pytest.raises(AcquisitionError, match="exit 3") as excinfo:
        run_download_command(["sh", "-c", "echo boom 1>&2; exit 3"], timeout_seconds=5)
    assert "boom" in str(excinfo.value)


def test_missing_binary_raises() -> None:
    with pytest.raises(AcquisitionError, match="failed to execute"):
        run_download_command(["/nonexistent/yt-dlp-xyz"], timeout_seconds=5)


def _read_pid(pidfile: Path) -> int:
    deadline = time.time() + 5
    while time.time() < deadline and not pidfile.exists():
        time.sleep(0.02)
    return int(pidfile.read_text().strip())


def _wait_gone(pid: int, *, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    """A wall-clock timeout terminates the child's entire process group, so a
    descendant the download spawned (a muxer, a fragment fetcher) cannot outlive
    the lease. The shell backgrounds a long sleep, records its pid, and waits;
    after the timeout the recorded pid must be gone."""
    pidfile = tmp_path / "child.pid"
    argv = ["sh", "-c", f'sleep 30 & echo $! > "{pidfile}"; wait']

    with pytest.raises(AcquisitionError, match="wall-clock"):
        run_download_command(argv, timeout_seconds=0.5)

    assert _wait_gone(_read_pid(pidfile)), "descendant survived the process-group kill"


def test_timeout_kills_group_even_when_leader_exits_first(tmp_path: Path) -> None:
    """Regression for the getpgid race: the shell backgrounds a long sleep and
    exits immediately, so the group LEADER is gone (and reaped) before the timeout
    fires. Signalling proc.pid directly must still reap the surviving descendant —
    os.getpgid on the reaped leader would fail and silently leave it running."""
    pidfile = tmp_path / "child.pid"
    argv = ["sh", "-c", f'sleep 30 & echo $! > "{pidfile}"; exit 0']

    with pytest.raises(AcquisitionError, match="wall-clock"):
        run_download_command(argv, timeout_seconds=0.5)

    assert _wait_gone(_read_pid(pidfile)), (
        "descendant survived after the group leader exited first"
    )


def test_kill_process_group_escalates_to_sigkill(tmp_path: Path) -> None:
    """A process group that ignores SIGTERM is reaped by the SIGKILL escalation
    after the grace period (exercised with a short grace to stay fast)."""
    trap = tmp_path / "trap.sh"
    pidfile = tmp_path / "child.pid"
    trap.write_text(
        "#!/bin/sh\n"
        'trap "" TERM\n'  # the leader ignores SIGTERM
        "sh -c 'trap \"\" TERM; while :; do sleep 1; done' &\n"  # so does the child
        f'echo $! > "{pidfile}"\n'
        "wait\n"
    )
    trap.chmod(0o755)
    proc = subprocess.Popen(
        [str(trap)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=_isolate_child,
    )
    child_pid = _read_pid(pidfile)

    _kill_process_group(proc, grace_seconds=0.3)  # SIGTERM ignored → SIGKILL fires

    assert _wait_gone(child_pid), "SIGTERM-ignoring group survived the SIGKILL escalation"
    proc.wait(timeout=5)


def test_downloader_argv_locks_down_yt_dlp(tmp_path: Path) -> None:
    """The production argv keeps a single audio item, caps size, bounds sockets,
    disables ambient config, and passes the URL only after the ``--`` terminator."""
    recorder = tmp_path / "record.sh"
    argfile = tmp_path / "args.txt"
    recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argfile}"\n')
    recorder.chmod(0o755)
    dest = tmp_path / "out"
    dest.mkdir()

    downloader = build_ytdlp_downloader(
        timeout_seconds=10, socket_timeout_seconds=7.0, ytdlp_bin=str(recorder)
    )
    downloader("https://example.com/video", dest, 4242)

    args = argfile.read_text().splitlines()
    assert "--no-config" in args
    assert "--no-playlist" in args
    assert args[args.index("--max-downloads") + 1] == "1"
    assert args[args.index("--format") + 1] == "bestaudio/best"
    assert args[args.index("--max-filesize") + 1] == "4242"
    assert args[args.index("--socket-timeout") + 1] == "7.0"
    assert str(dest / "source.%(ext)s") in args
    # The URL is the final token, guarded by the argument terminator.
    assert args[-2:] == ["--", "https://example.com/video"]


def test_build_stage_context_wires_the_real_downloader() -> None:
    """Production wiring smoke test: build_stage_context binds a callable yt-dlp
    downloader and propagates the authoritative size cap from settings."""
    from voxint.config import Settings
    from voxint.pipeline.stages.context import build_stage_context

    ctx = build_stage_context(Settings(ytdlp_max_bytes=777))
    assert ctx.ytdlp_max_bytes == 777
    assert callable(ctx.downloader)


def test_acquisition_error_is_not_auto_retried() -> None:
    """The worker only auto-retries ServiceError; an AcquisitionError (or a
    StageDataError) is deterministic, so the run stays FAILED for manual Requeue."""
    from voxint.db.models import Stage

    acquire_failure = StageFailedError(Stage.ACQUIRE, AcquisitionError("bot check"))
    assert retryable_cause(acquire_failure) is False
    stage_data_failure = StageFailedError(Stage.ACQUIRE, StageDataError("missing file"))
    assert retryable_cause(stage_data_failure) is False
