"""Unit tests for the yt-dlp subprocess boundary (no network, no yt-dlp binary).

``run_download_command`` and the argv ``build_ytdlp_downloader`` produces are
exercised with harmless stand-in commands (``true``/``sh``/a recording script),
so the process-group timeout kill and the flag lockdown are tested deterministically.
"""

import os
import shlex
import signal
import subprocess
import time
import traceback
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
    while time.time() < deadline:
        if pidfile.exists():
            content = pidfile.read_text().strip()
            if content:
                return int(content)
        time.sleep(0.02)
    raise TimeoutError(f"{pidfile} was not written within 5 s")


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


def test_kill_process_group_swallows_eperm(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS/BSD zombie-reparented-to-launchd case (issue #26): once the group
    leader is reaped and the survivor is a zombie owned by launchd, killpg returns
    EPERM (PermissionError), not ESRCH. Teardown must swallow it on BOTH the
    SIGTERM and SIGKILL call sites so it never escapes and masks the intended
    redacted AcquisitionError. Linux returns ESRCH, so this is only reachable via
    monkeypatch here."""

    class _FakeProc:
        pid = 424242

        def wait(self, timeout: float | None = None) -> int:
            return 0  # leader already reaped; return promptly, no TimeoutExpired

    signals: list[int] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        signals.append(sig)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", _fake_killpg)

    # Must not raise — both PermissionErrors are suppressed inside the teardown.
    _kill_process_group(_FakeProc(), grace_seconds=0.0)  # type: ignore[arg-type]

    assert signals == [signal.SIGTERM, signal.SIGKILL], (
        "both killpg call sites must run despite EPERM on the first"
    )


def test_kill_process_group_does_not_swallow_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against the suppression widening to `suppress(Exception)`: only the
    two benign errnos (ESRCH/EPERM) are swallowed. A genuinely unexpected teardown
    error must still propagate, not be hidden."""

    class _FakeProc:
        pid = 424242

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def _fake_killpg(pgid: int, sig: int) -> None:
        raise RuntimeError("unexpected teardown failure")

    monkeypatch.setattr(os, "killpg", _fake_killpg)

    with pytest.raises(RuntimeError, match="unexpected teardown failure"):
        _kill_process_group(_FakeProc(), grace_seconds=0.0)  # type: ignore[arg-type]


def test_timeout_error_does_not_leak_argv_in_traceback() -> None:
    """The timeout AcquisitionError must NOT chain the raw TimeoutExpired, whose
    __str__ embeds the whole argv (the signed source URL). Assert the sentinel is
    absent from the ENTIRE rendered traceback and that the cause is suppressed —
    str(exc) alone would false-pass since the top-level message is already clean.
    """
    sentinel = "TIMEOUT-ARGV-SENTINEL-abc123"
    url = f"https://cdn.example.com/media.mp3?token={sentinel}&sig=deadbeef"
    argv = ["sh", "-c", "sleep 30", "--", url]

    try:
        run_download_command(argv, timeout_seconds=0.5)
    except AcquisitionError as exc:
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        assert exc.__cause__ is None  # TimeoutExpired cause suppressed
        assert sentinel not in rendered
        assert sentinel not in str(exc)
        assert "wall-clock" in str(exc)  # still names the failure
    else:  # pragma: no cover - the sleep must outlast the 0.5s timeout
        raise AssertionError("expected an AcquisitionError from the timeout")


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


def _record_argv(
    tmp_path: Path, *, proxy: str = "", cookies_file: "Path | None" = None
) -> list[str]:
    """Build the production downloader against a recorder stub and return the argv
    it was invoked with (no network, no real yt-dlp)."""
    recorder = tmp_path / "record.sh"
    argfile = tmp_path / "args.txt"
    recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argfile}"\n')
    recorder.chmod(0o755)
    dest = tmp_path / "out"
    dest.mkdir()
    downloader = build_ytdlp_downloader(
        timeout_seconds=10,
        socket_timeout_seconds=7.0,
        ytdlp_bin=str(recorder),
        proxy=proxy,
        cookies_file=cookies_file,
    )
    downloader("https://example.com/video", dest, 4242)
    return argfile.read_text().splitlines()


def test_downloader_argv_adds_6g_lockdown_and_pins_direct_egress(tmp_path: Path) -> None:
    """Slice 6g lockdown: plugin dirs cleared, post-processor exec neutralised, and
    --proxy pinned to an explicit empty string (direct) even when unset — so an
    ambient HTTP(S)_PROXY can't reroute egress. --cookies is absent when unset."""
    args = _record_argv(tmp_path)
    assert "--no-plugin-dirs" in args  # no local/remote plugin loading
    assert "--no-exec" in args  # no post-processor command execution
    assert args[args.index("--proxy") + 1] == ""  # explicit direct, not omitted
    assert "--cookies" not in args


def test_downloader_argv_wires_proxy_and_cookies_only_when_set(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    args = _record_argv(tmp_path, proxy="socks5://203.0.113.5:1080", cookies_file=cookies)
    assert args[args.index("--proxy") + 1] == "socks5://203.0.113.5:1080"
    assert args[args.index("--cookies") + 1] == str(cookies)
    # Egress options ride among the flags, before the ``--`` URL terminator.
    term = args.index("--")
    assert args.index("--proxy") < term and args.index("--cookies") < term
    assert args[-2:] == ["--", "https://example.com/video"]


def test_downloader_argv_captures_info_json_alongside_the_download(
    tmp_path: Path,
) -> None:
    """Metadata capture (issue #36) rides the SAME invocation — no second
    network call — writing a clean info-JSON to the pinned filename, with
    playlist metafiles suppressed so the stage's file-count guard holds."""
    from voxint.media.ytdlp import INFO_JSON_FILENAME

    args = _record_argv(tmp_path)
    assert "--write-info-json" in args
    assert "--clean-info-json" in args
    assert "--no-write-playlist-metafiles" in args
    dest = tmp_path / "out"
    # The typed infojson template pins the exact output name...
    assert f"infojson:{dest / INFO_JSON_FILENAME}" in args
    # ...riding among the flags, before the ``--`` URL terminator, with the
    # media template unchanged.
    term = args.index("--")
    assert args.index(f"infojson:{dest / INFO_JSON_FILENAME}") < term
    assert str(dest / "source.%(ext)s") in args
    assert args[-2:] == ["--", "https://example.com/video"]


def test_run_download_command_scrubs_extra_secret_from_stderr() -> None:
    """A cookies path echoed as prose in stderr (no --cookies flag) is scrubbed via
    the extra_secrets channel the downloader threads through — the structural
    redactor alone could not catch a bare path."""
    path = "/secrets/COOKIE-PATH-SENTINEL/cookies.txt"
    blob = f"ERROR: unable to load cookies [Errno 2]: '{path}'"
    with pytest.raises(AcquisitionError) as exc:
        run_download_command(
            ["sh", "-c", f"printf '%s' {shlex.quote(blob)} 1>&2; exit 1"],
            timeout_seconds=10,
            extra_secrets=(path,),
        )
    assert "COOKIE-PATH-SENTINEL" not in str(exc.value)
    assert path not in str(exc.value)
    assert "exit 1" in str(exc.value)  # the failure class is still legible


def test_build_stage_context_wires_the_real_downloader() -> None:
    """Production wiring smoke test: build_stage_context binds a callable yt-dlp
    downloader and propagates the authoritative size cap from settings."""
    from voxint.config import Settings
    from voxint.pipeline.stages.context import build_stage_context

    ctx = build_stage_context(Settings(ytdlp_max_bytes=777))
    assert ctx.ytdlp_max_bytes == 777
    assert callable(ctx.downloader)


def test_build_stage_context_diarization_ceiling_defaults_unset() -> None:
    """No-hint runs stay byte-identical to pre-#128 (issue #128 review): with the
    ceiling unset the context carries None, so diarize_embed passes no bound and
    the client posts only the path (see test_diarize_sends_no_bounds_by_default).
    An install-wide ceiling flows through as an explicit int."""
    from voxint.config import Settings
    from voxint.pipeline.stages.context import build_stage_context

    assert build_stage_context(Settings(_env_file=None)).diarization_max_speakers is None
    assert build_stage_context(Settings(diarization_max_speakers=4)).diarization_max_speakers == 4


def test_acquisition_error_is_not_auto_retried() -> None:
    """The worker only auto-retries ServiceError; an AcquisitionError (or a
    StageDataError) is deterministic, so the run stays FAILED for manual Requeue."""
    from voxint.db.models import Stage

    acquire_failure = StageFailedError(Stage.ACQUIRE, AcquisitionError("bot check"))
    assert retryable_cause(acquire_failure) is False
    stage_data_failure = StageFailedError(Stage.ACQUIRE, StageDataError("missing file"))
    assert retryable_cause(stage_data_failure) is False
