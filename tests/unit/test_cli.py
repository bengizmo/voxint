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
