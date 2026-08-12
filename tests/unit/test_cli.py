import io

import pytest

from voxint import __version__
from voxint.cli import main


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
