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
