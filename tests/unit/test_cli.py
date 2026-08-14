import io
import uuid

import pytest

from voxint import __version__
from voxint.cli import main


def _block_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make build_engine explode so a test proves it exits BEFORE any DB access."""
    import voxint.db.session as db_session

    def _no_db(*_a: object, **_k: object) -> object:
        raise AssertionError("build_engine must not run on this path")

    monkeypatch.setattr(db_session, "build_engine", _no_db)


def test_no_args_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "voxint" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_fetch_refuses_when_ytdlp_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ytdlp_enabled is enforced at the submission surface: the CLI refuses with a
    # nonzero exit BEFORE it opens a DB session — proven by making build_engine
    # explode if it is ever reached. No DB needed, so this stays a unit test.
    monkeypatch.setenv("YTDLP_ENABLED", "false")
    import voxint.db.session as db_session

    def _no_db(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("build_engine must not run when ingestion is disabled")

    monkeypatch.setattr(db_session, "build_engine", _no_db)
    assert main(["fetch", "https://www.example.com/video"]) == 2
    assert "disabled" in capsys.readouterr().out


def test_fetch_with_no_url_argument_or_stdin_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A URL can be piped in instead of passed positionally (which would leak a
    # signed URL to ps/proc and shell history). With ingestion enabled but no
    # positional URL and empty stdin, the CLI errors (exit 2) BEFORE opening a DB
    # session — proven by making build_engine explode if reached.
    monkeypatch.setenv("YTDLP_ENABLED", "true")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    import voxint.db.session as db_session

    def _no_db(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("build_engine must not run when no URL was provided")

    monkeypatch.setattr(db_session, "build_engine", _no_db)
    assert main(["fetch"]) == 2
    assert "no URL provided" in capsys.readouterr().out


def test_export_rejects_bad_uuid_via_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["export", "not-a-uuid", "--format", "json"])
    assert exc.value.code == 2


def test_export_refuses_existing_output_before_db(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: object
) -> None:
    # An existing -o file (no --force) is refused BEFORE opening a DB session, so
    # the operator's file is never at risk and no DB is required.
    _block_db(monkeypatch)
    existing = tmp_path / "out.srt"  # type: ignore[operator]
    existing.write_text("keep me")
    assert main(["export", str(uuid.uuid4()), "--format", "srt", "-o", str(existing)]) == 2
    assert "exists" in capsys.readouterr().out
    assert existing.read_text() == "keep me"


def test_export_rejects_unknown_format() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["export", str(uuid.uuid4()), "--format", "docx"])
    assert exc.value.code == 2


def test_list_rejects_unknown_status_before_db(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _block_db(monkeypatch)
    assert main(["list", "--status", "bogus"]) == 2
    assert "unknown status" in capsys.readouterr().out


def test_list_rejects_out_of_range_limit_before_db(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The bound is checked after settings load but before the DB session.
    _block_db(monkeypatch)
    assert main(["list", "--limit", "999"]) == 2
    assert "between 1 and 500" in capsys.readouterr().out


def test_doctor_returns_2_on_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A SettingsError (already credential-sanitized) is a CLI/config error → exit 2,
    # surfaced before any dependency probing.
    import voxint.config as config

    def _boom() -> object:
        raise config.SettingsError("bad config")

    # _doctor imports get_settings from voxint.config at call time, so patching it
    # on the module is what the handler will see.
    monkeypatch.setattr(config, "get_settings", _boom)
    assert main(["doctor"]) == 2
    assert "bad config" in capsys.readouterr().out


def test_doctor_prints_results_and_maps_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Stub the DB engine and the diagnostics run so this exercises _doctor's
    # printing + exit-code wiring (not the checks, which test_diagnostics covers).
    import voxint.db.session as db_session
    import voxint.diagnostics as diagnostics

    class _Engine:
        def dispose(self) -> None:
            pass

    monkeypatch.setattr(db_session, "build_engine", lambda *a, **k: _Engine())

    canned = [
        diagnostics.CheckResult("postgres", True, True, "connected"),
        diagnostics.CheckResult("redis", False, True, "unreachable (ConnectionError)"),
        diagnostics.CheckResult("hugging face token", False, False, "rejected (401)"),
    ]
    monkeypatch.setattr(diagnostics, "run_diagnostics", lambda *a, **k: canned)

    assert main(["doctor"]) == 1  # a hard dep (redis) is down
    out = capsys.readouterr().out
    assert "[ok  ] postgres: connected" in out
    assert "[FAIL] redis:" in out
    assert "[warn] hugging face token:" in out  # advisory failure ⇒ warn, not FAIL
    assert "a hard dependency is down" in out


# ---- watch / submit --wait: the poll loop (no DB, injected clock/sleep) ------


def _make_factory(scripts: list[object]) -> tuple[object, dict[str, int]]:
    """A fake session factory: poll N returns run state scripts[N] (last repeats).

    Each element is ``(status, current_stage)`` for a run, or ``None`` for a
    missing run. Mirrors ``_poll_until_stop``'s one-``get``-per-fresh-session use.
    """
    from types import SimpleNamespace

    calls = {"n": 0}

    def factory() -> object:
        idx = min(calls["n"], len(scripts) - 1)
        item = scripts[idx]
        calls["n"] += 1

        class _Sess:
            def __enter__(self_: object) -> object:
                return self_

            def __exit__(self_: object, *_a: object) -> bool:
                return False

            def get(self_: object, _model: object, _rid: object) -> object:
                if item is None:
                    return None
                status, stage = item  # type: ignore[misc]
                return SimpleNamespace(status=status, current_stage=stage)

        return _Sess()

    return factory, calls


def _poll(scripts: list[object], **kw: object) -> tuple[int, list[str], int]:
    """Drive ``_poll_until_stop`` with a fake clock that advances on sleep."""
    from voxint.cli import _poll_until_stop

    factory, calls = _make_factory(scripts)
    clock = {"t": 0.0}
    buf: list[str] = []
    defaults: dict[str, object] = dict(
        interval=2.0,
        timeout=100.0,
        monotonic=lambda: clock["t"],
        sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        write=buf.append,
        isatty=False,
    )
    defaults.update(kw)
    code = _poll_until_stop(factory, uuid.uuid4(), **defaults)  # type: ignore[arg-type]
    return code, buf, calls["n"]


def test_poll_completed_immediately_exits_zero() -> None:
    code, buf, polls = _poll([("completed", "finalize")])
    assert code == 0
    assert polls == 1
    assert any("completed" in line for line in buf)


def test_poll_running_then_completed() -> None:
    code, buf, polls = _poll([("running", "transcribe"), ("completed", "-")])
    assert code == 0
    assert polls == 2
    # non-tty prints only transitions: running line, then completed line
    joined = "".join(buf)
    assert "running" in joined
    assert "completed" in joined


def test_poll_failed_exits_one() -> None:
    code, _buf, _polls = _poll([("failed", "transcribe")])
    assert code == 1


def test_poll_cancelled_exits_one() -> None:
    code, _buf, _polls = _poll([("cancelled", "prepare")])
    assert code == 1


def test_poll_awaiting_adjudication_exits_three() -> None:
    # awaiting_adjudication can resume to running (state machine), so it is a
    # paused outcome, NOT success — its own exit code, never 0.
    code, _buf, _polls = _poll([("awaiting_adjudication", "enhance_match")])
    assert code == 3


def test_poll_missing_run_exits_two() -> None:
    code, buf, _polls = _poll([None])
    assert code == 2
    assert any("no run" in line for line in buf)


def test_poll_timeout_exits_124() -> None:
    code, buf, _polls = _poll([("running", "transcribe")], timeout=5.0, interval=2.0)
    assert code == 124
    assert any("timeout" in line for line in buf)


def test_watch_rejects_bad_interval_before_db(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_db(monkeypatch)
    assert main(["watch", str(uuid.uuid4()), "--interval", "0"]) == 2


def test_watch_rejects_negative_timeout_before_db(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_db(monkeypatch)
    assert main(["watch", str(uuid.uuid4()), "--timeout", "-1"]) == 2


def test_watch_rejects_bad_uuid_via_argparse() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["watch", "not-a-uuid"])
    assert exc.value.code == 2
