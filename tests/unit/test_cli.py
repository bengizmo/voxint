import contextlib
import io
import uuid
from collections.abc import Iterator

import pytest

from voxint import __version__
from voxint.cli import build_parser, main
from voxint.db.models import AppSettings


def _block_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make build_engine explode so a test proves it exits BEFORE any DB access."""
    import voxint.db.session as db_session

    def _no_db(*_a: object, **_k: object) -> object:
        raise AssertionError("build_engine must not run on this path")

    monkeypatch.setattr(db_session, "build_engine", _no_db)


def _stub_db_row(monkeypatch: pytest.MonkeyPatch, row: AppSettings | None = None) -> None:
    """Stub the CLI's DB-session layer so effective-flag resolution (issue #74)
    runs without a real database, using ``row`` (default ``None`` ⇒ env governs)
    as the resolved ``app_settings`` singleton.

    The #74 CLI gates read the effective (row-over-env) value from the DB rather
    than a bare env flag (decision 6: participate in effective settings, fail
    honestly on an unavailable DB — never silently env-fallback). These unit
    tests inject the row directly so they stay DB-free while still exercising the
    resolved gate.
    """
    import voxint.app_settings as app_settings
    import voxint.cli as cli
    import voxint.db.session as db_session

    class _FakeEngine:
        def dispose(self) -> None:  # the CLI disposes engines in a finally
            pass

    monkeypatch.setattr(cli, "_engine_or_report", lambda **_k: (_FakeEngine(), 0))
    monkeypatch.setattr(db_session, "build_session_factory", lambda _e: object())

    @contextlib.contextmanager
    def _scope(_factory: object) -> Iterator[object]:
        yield object()

    monkeypatch.setattr(db_session, "session_scope", _scope)
    monkeypatch.setattr(app_settings, "get_app_settings", lambda _s: row)


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
    # ytdlp_enabled is now resolved as the effective (row-over-env) value from the
    # DB (issue #74, decision 6): a UI toggle governs the CLI, so the gate must
    # read the row rather than a bare env flag. Env-disabled with no stored
    # override refuses...
    monkeypatch.setenv("YTDLP_ENABLED", "false")
    _stub_db_row(monkeypatch, row=None)  # no override → env governs
    assert main(["fetch", "https://www.example.com/video"]) == 2
    assert "disabled" in capsys.readouterr().out

    # ...and a stored row disable wins over an env enable.
    monkeypatch.setenv("YTDLP_ENABLED", "true")
    _stub_db_row(monkeypatch, row=AppSettings(id=1, ytdlp_enabled=False))
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


def test_export_accepts_markdown_format() -> None:
    # md joins the shared formatter set (issue #65); argparse must accept it and
    # carry the timestamps flag (which md honors, like txt).
    args = build_parser().parse_args(
        ["export", str(uuid.uuid4()), "--format", "md", "--no-timestamps"]
    )
    assert args.format == "md"
    assert args.timestamps is False


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


def test_doctor_fails_verdict_when_a_plugin_cannot_load(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Dependencies all green, but a builtin that would abort real api/worker
    # startup must not let doctor report "all hard dependencies OK".
    import voxint.db.session as db_session
    import voxint.diagnostics as diagnostics
    import voxint.plugins as plugins

    class _Engine:
        def dispose(self) -> None:
            pass

    monkeypatch.setattr(db_session, "build_engine", lambda *a, **k: _Engine())
    monkeypatch.setattr(
        diagnostics,
        "run_diagnostics",
        lambda *a, **k: [diagnostics.CheckResult("postgres", True, True, "connected")],
    )

    def _boom(_settings: object) -> object:
        raise plugins.PluginError("duplicate plugin id 'x'")

    # _doctor does `from voxint.plugins import load_plugins` at call time, so the
    # module attribute is what it binds.
    monkeypatch.setattr(plugins, "load_plugins", _boom)

    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] plugins: duplicate plugin id 'x'" in out
    assert "a plugin failed to load" in out


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


def test_watch_rejects_nan_interval_before_db(monkeypatch: pytest.MonkeyPatch) -> None:
    # argparse(float) accepts nan/inf; nan interval would reach time.sleep(nan)
    # and raise mid-loop — reject it up front (before any DB access).
    _block_db(monkeypatch)
    assert main(["watch", str(uuid.uuid4()), "--interval", "nan"]) == 2


def test_watch_rejects_inf_timeout_before_db(monkeypatch: pytest.MonkeyPatch) -> None:
    # inf/nan timeout would silently disable the 124 deadline (remaining <= 0
    # never true) — the headline watch contract. Reject it.
    _block_db(monkeypatch)
    assert main(["watch", str(uuid.uuid4()), "--timeout", "inf"]) == 2
    assert main(["watch", str(uuid.uuid4()), "--timeout", "nan"]) == 2


def test_poll_tty_redraws_line_and_terminates_with_newline() -> None:
    # On a TTY the progress line is redrawn in place (\r + clear) and the final
    # state ends the line with a newline.
    code, buf, _polls = _poll([("completed", "finalize")], isatty=True)
    assert code == 0
    joined = "".join(buf)
    assert "\r" in joined
    assert joined.endswith("\n")


def test_poll_unknown_status_notes_once_then_times_out() -> None:
    # A status this CLI doesn't know keeps polling (forward-compat) but says so
    # exactly once, rather than waiting silently to the timeout.
    code, buf, _polls = _poll([("weird_new_status", "transcribe")], timeout=1.0, interval=0.5)
    assert code == 124
    notes = [line for line in buf if "unrecognized run status" in line]
    assert len(notes) == 1


def test_watch_keyboardinterrupt_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ctrl-C during the poll maps to 130 (the shell's SIGINT convention); the
    # engine is still disposed on the way out.
    import voxint.cli as cli
    import voxint.db.session as db_session

    disposed: list[bool] = []

    class _FakeEngine:
        def dispose(self) -> None:
            disposed.append(True)

    monkeypatch.setattr(cli, "_engine_or_report", lambda **_k: (_FakeEngine(), 0))
    monkeypatch.setattr(db_session, "build_session_factory", lambda _e: object())

    def _boom(*_a: object, **_k: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_poll_until_stop", _boom)
    assert main(["watch", str(uuid.uuid4())]) == 130
    assert disposed == [True]  # finally still ran


def test_research_refuses_when_disabled_before_any_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # voxint_web_research is enforced at the surface as the effective (row-over-env)
    # value resolved from the DB (issue #74): exit 2 BEFORE any DNS or socket work
    # — proven by making getaddrinfo explode if ever reached. Row=None ⇒ env (false)
    # governs.
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "false")
    _stub_db_row(monkeypatch, row=None)
    import socket

    def _no_dns(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("DNS must not be touched while research is disabled")

    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)
    assert main(["research", "search", "anything"]) == 2
    assert "disabled" in capsys.readouterr().out
    assert main(["research", "read", "https://example.com/"]) == 2
    assert "disabled" in capsys.readouterr().out


def test_research_search_prints_normalized_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import voxint.research as research
    from voxint.research.search import SearchOutcome, SearchResult

    def _fake_search(query: str, **_kwargs: object) -> SearchOutcome:
        assert query == "hydronics podcast"
        return SearchOutcome(
            ok=True,
            error=None,
            error_detail="",
            results=(
                SearchResult(
                    title="A Title", url="https://a.example.com/", snippet="snip"
                ),
            ),
            dropped_results=2,
        )

    monkeypatch.setattr(research, "web_search", _fake_search)
    _stub_db_row(monkeypatch)  # env governs (voxint_web_research=true)
    assert main(["research", "search", "hydronics podcast"]) == 0
    out = capsys.readouterr().out
    assert "https://a.example.com/" in out
    assert "A Title" in out
    assert "2 result(s) dropped" in out


def test_research_read_prints_extracted_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import voxint.research as research
    from voxint.research.fetch import FetchOutcome

    def _fake_read(url: str, **_kwargs: object) -> FetchOutcome:
        assert url == "https://a.example.com/page"
        return FetchOutcome(
            ok=True,
            error=None,
            error_detail="",
            text="the page text",
            title="Page",
            final_url="https://a.example.com/page",
            host="a.example.com",
            bytes_fetched=13,
            hops=0,
        )

    monkeypatch.setattr(research, "read_url", _fake_read)
    _stub_db_row(monkeypatch)  # env governs (voxint_web_research=true)
    assert main(["research", "read", "https://a.example.com/page"]) == 0
    out = capsys.readouterr().out
    assert "# Page" in out
    assert "the page text" in out


def test_research_read_failure_exits_2_without_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import voxint.research as research
    from voxint.research.fetch import FetchOutcome

    def _fake_read(url: str, **_kwargs: object) -> FetchOutcome:
        return FetchOutcome(
            ok=False,
            error="policy_refused",
            error_detail="host 'x.example.com' resolves to a non-public address",
            host="x.example.com",
        )

    monkeypatch.setattr(research, "read_url", _fake_read)
    _stub_db_row(monkeypatch)  # env governs (voxint_web_research=true)
    assert main(["research", "read", "https://x.example.com/?token=SECRETQ"]) == 2
    out = capsys.readouterr().out
    assert "policy_refused" in out
    assert "SECRETQ" not in out


def test_research_read_prints_query_stripped_final_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A redirect can mint a signed token into final_url; the terminal (and
    # scrollback) must never retain it (review finding).
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searxng.example:8888")
    import voxint.research as research
    from voxint.research.fetch import FetchOutcome

    def _fake_read(url: str, **_kwargs: object) -> FetchOutcome:
        return FetchOutcome(
            ok=True,
            error=None,
            error_detail="",
            text="body",
            final_url="https://a.example.com/doc?token=SECRETQTOKEN",
            host="a.example.com",
            bytes_fetched=4,
        )

    monkeypatch.setattr(research, "read_url", _fake_read)
    _stub_db_row(monkeypatch)  # env governs (voxint_web_research=true)
    assert main(["research", "read", "https://a.example.com/doc"]) == 0
    out = capsys.readouterr().out
    assert "SECRETQTOKEN" not in out
    assert "query omitted" in out
    assert "https://a.example.com/doc" in out


def test_research_read_accepts_url_on_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searxng.example:8888")
    monkeypatch.setattr("sys.stdin", io.StringIO("https://a.example.com/x\n"))
    import voxint.research as research
    from voxint.research.fetch import FetchOutcome

    def _fake_read(url: str, **_kwargs: object) -> FetchOutcome:
        assert url == "https://a.example.com/x"
        return FetchOutcome(
            ok=True, error=None, error_detail="", text="piped",
            final_url="https://a.example.com/x", host="a.example.com",
        )

    monkeypatch.setattr(research, "read_url", _fake_read)
    _stub_db_row(monkeypatch)  # env governs (voxint_web_research=true)
    assert main(["research", "read"]) == 0
    assert "piped" in capsys.readouterr().out


def test_research_fails_honestly_when_db_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Decision 6: the CLI resolves the effective gate from the DB and FAILS
    # HONESTLY when the DB is unavailable — it must never silently fall back to
    # env (which could bypass a UI disable). Env has research ON, but the engine
    # cannot be built, so the command exits 2 and touches no network.
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import socket

    import voxint.cli as cli
    import voxint.research as research

    def _no_engine(**_k: object) -> object:
        return (None, 2)

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("no network or search when the DB is unavailable")

    monkeypatch.setattr(cli, "_engine_or_report", _no_engine)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(research, "web_search", _boom)
    assert main(["research", "search", "anything"]) == 2


def test_research_fails_honestly_on_unreachable_db(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A well-formed but UNREACHABLE db raises inside the session (lazy connect),
    # NOT in _engine_or_report — the CLI must still exit 2 with a DSN-free message
    # (issue #74, decision 6), never a raw traceback, and touch no network.
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import socket

    from sqlalchemy.exc import OperationalError

    import voxint.app_settings as app_settings
    import voxint.research as research

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("no network/search when the DB is unreachable")

    def _refused(_s: object) -> object:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    _stub_db_row(monkeypatch)  # fake engine + session scope
    monkeypatch.setattr(app_settings, "get_app_settings", _refused)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(research, "web_search", _boom)
    assert main(["research", "search", "anything"]) == 2
    out = capsys.readouterr().out
    assert "database unavailable" in out
    assert "connection refused" not in out  # DSN/detail-free message


def test_research_search_row_disable_wins_over_env_enable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A stored row disable must veto an env enable (the arc's whole point): env
    # has research ON, the row turns it OFF, so the command refuses before DNS.
    monkeypatch.setenv("VOXINT_WEB_RESEARCH", "true")
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://searx.lan:8888")
    import socket

    import voxint.research as research

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("must not reach network/search when row disables research")

    _stub_db_row(monkeypatch, row=AppSettings(id=1, voxint_web_research=False))
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(research, "web_search", _boom)
    assert main(["research", "search", "anything"]) == 2
    assert "disabled" in capsys.readouterr().out
