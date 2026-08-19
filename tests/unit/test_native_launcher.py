"""Offline tests for scripts/native/voxint-native.sh via its library-mode seam.

The launcher sources with ``VOXINT_NATIVE_LIB=1`` (no main), which lets these
tests exercise the pure-shell logic — per-service argv and env assembly, the
DATABASE_URL/REDIS_URL builders, launchd plist generation, and ``pwd -P``
MEDIA_ROOT resolution — without launchd, a Postgres/Redis, or Docker. The
functions under test never write into the repo, so the script is sourced in
place; ``VOXINT_NATIVE_HOME`` points every test at a throwaway tmp tree.

This mirrors tests/unit/test_metal_launcher.py: the two native launchers share
the same plist/env-parity discipline, so they share a test shape.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REAL_REPO = Path(__file__).resolve().parents[2]
NATIVE_SCRIPT = REAL_REPO / "scripts" / "native" / "voxint-native.sh"

CORE_SERVICES = ("api", "worker", "beat")


def run_lib(
    home: Path,
    script: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VOXINT_NATIVE_LIB"] = "1"
    env["VOXINT_NATIVE_HOME"] = str(home)
    # Scrub the caller-overridable knobs so a developer's shell env cannot
    # perturb the asserted defaults.
    for key in (
        "VOXINT_NATIVE_DB_USER",
        "VOXINT_NATIVE_DB_NAME",
        "VOXINT_NATIVE_DB_PASSWORD",
        "VOXINT_NATIVE_PG_PORT",
        "VOXINT_NATIVE_REDIS_PORT",
        "VOXINT_NATIVE_API_HOST",
        "VOXINT_NATIVE_API_PORT",
        "VOXINT_NATIVE_BREW_PREFIX",
        "VOXINT_NATIVE_PG_BINDIR",
        "VOXINT_NATIVE_PASSWORD",
        "VOXINT_NATIVE_CSRF_SECRET",
        "VOXINT_NATIVE_MEDIA_ROOT",
        # Slice-3 knobs: a developer's ambient value must not perturb the
        # asserted defaults (e.g. VOXINT_NATIVE_WITH_MODELS=0 would flip the
        # delegation default; the URL/dir overrides steer staging + env).
        "VOXINT_NATIVE_WITH_MODELS",
        "VOXINT_NATIVE_ASR_URL",
        "VOXINT_NATIVE_DIARIZER_URL",
        "VOXINT_NATIVE_EMBEDDER_URL",
        "VOXINT_NATIVE_FRONTEND_DIR",
        "VOXINT_NATIVE_APP_ASSETS_DIR",
        "VOXINT_NATIVE_LOG_MAX_MB",
        "VOXINT_NATIVE_LOG_ARCHIVES",
        # Upgrade-db old-cluster bindir override: an ambient value would leak
        # into validate_native_inputs and perturb the #12 validation tests.
        "VOXINT_NATIVE_OLD_PG_BINDIR",
    ):
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    full = f'source "{NATIVE_SCRIPT}"\n{script}'
    return subprocess.run(
        ["bash", "-c", full],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
    )


def env_lines(proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )


def argv_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    assert proc.returncode == 0, proc.stderr
    return [ln for ln in proc.stdout.splitlines() if ln != ""]


# --------------------------------------------------------------------------- #
# Connection-string builders
# --------------------------------------------------------------------------- #
def test_database_url_default(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "native_database_url")
    assert proc.returncode == 0
    assert proc.stdout == "postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint"


def test_database_url_honours_overrides(tmp_path: Path) -> None:
    proc = run_lib(
        tmp_path,
        "native_database_url",
        extra_env={
            "VOXINT_NATIVE_DB_USER": "vx",
            "VOXINT_NATIVE_DB_PASSWORD": "s3cret",
            "VOXINT_NATIVE_DB_NAME": "vxdb",
            "VOXINT_NATIVE_PG_PORT": "5455",
        },
    )
    assert proc.stdout == "postgresql+psycopg://vx:s3cret@127.0.0.1:5455/vxdb"


def test_redis_url_default_and_override(tmp_path: Path) -> None:
    assert run_lib(tmp_path, "native_redis_url").stdout == "redis://127.0.0.1:6379/0"
    proc = run_lib(
        tmp_path, "native_redis_url", extra_env={"VOXINT_NATIVE_REDIS_PORT": "6399"}
    )
    assert proc.stdout == "redis://127.0.0.1:6399/0"


# --------------------------------------------------------------------------- #
# #7 — DSN reserved-character handling: the password may legitimately contain
# RFC-3986 reserved characters (only a newline is rejected upstream), so both
# DSN composers must percent-encode the userinfo. A string built from a raw
# reserved char would either mis-parse or corrupt the URL SQLAlchemy consumes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("voxint", "voxint"),  # unreserved: encoding is a no-op
        ("AZaz09-._~", "AZaz09-._~"),  # the full unreserved set survives verbatim
        ("p@ss:w/rd", "p%40ss%3Aw%2Frd"),  # the classic DSN-breaking trio
        ("a?b#c&d=e", "a%3Fb%23c%26d%3De"),  # query/fragment/sub-delims
        ("%already", "%25already"),  # a literal % is itself encoded (no double pass)
        ("sp ace+plus", "sp%20ace%2Bplus"),  # space and + both reserved
        ("café", "caf%C3%A9"),  # multibyte UTF-8 encodes per-byte under LC_ALL=C
    ],
)
def test_percent_encode_matches_rfc3986(
    tmp_path: Path, value: str, expected: str
) -> None:
    # The Bash encoder must agree byte-for-byte with urllib.parse.quote(safe="")
    # (which the Python composer uses) so the two DSN builders stay in lockstep.
    from urllib.parse import quote

    assert quote(value, safe="") == expected  # guards the expectation itself
    proc = run_lib(tmp_path, f'percent_encode {shlex.quote(value)}')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


def test_database_url_password_round_trips_through_make_url(tmp_path: Path) -> None:
    # A password full of reserved characters must survive the Bash composer:
    # SQLAlchemy's own parser must recover the exact original password, user,
    # host, port, and database. String-equality on the URL is insufficient —
    # parse it back and compare the decoded fields (codex).
    from sqlalchemy.engine import make_url

    password = "p@ss:w/rd?#&%+ x"
    proc = run_lib(
        tmp_path,
        "native_database_url",
        extra_env={
            "VOXINT_NATIVE_DB_USER": "voxint",
            "VOXINT_NATIVE_DB_PASSWORD": password,
            "VOXINT_NATIVE_DB_NAME": "vxdb",
            "VOXINT_NATIVE_PG_PORT": "5432",
        },
    )
    assert proc.returncode == 0, proc.stderr
    url = make_url(proc.stdout)
    assert url.password == password
    assert url.username == "voxint"
    assert url.host == "127.0.0.1"
    assert url.port == 5432
    assert url.database == "vxdb"


def test_python_and_bash_composers_agree_on_reserved_password() -> None:
    # The second composer (tools/native_e2e_lifecycle.py) must encode identically,
    # so the acceptance lane's engine and the launcher's alembic hit the same DSN.
    from sqlalchemy.engine import make_url
    from tools.native_e2e_lifecycle import LOOPBACK, NativeConfig

    password = "p@ss:w/rd?#&%+ x"
    cfg = NativeConfig(
        pg_port="5432",
        redis_port="6379",
        api_port="8000",
        db_password=password,
        voxint_password="pw",
        csrf_secret="c" * 32,
        db_user="voxint",
        db_name="vxdb",
    )
    url = make_url(cfg.database_url)
    assert url.password == password
    assert url.username == "voxint"
    assert url.host == LOOPBACK
    assert url.database == "vxdb"


# --------------------------------------------------------------------------- #
# #12 — VOXINT_NATIVE_OLD_PG_BINDIR (the upgrade-db old-cluster override) must
# pass through the same validate_native_inputs control-char gate as the other
# operator-settable path knobs; a newline could otherwise forge a second plist
# env record downstream. Consumed far from the gate, so it must be validated
# up front rather than only where it is read.
# --------------------------------------------------------------------------- #
def test_validate_accepts_clean_old_pg_bindir(tmp_path: Path) -> None:
    # A normal path and the empty (unset) default both pass.
    for val in ("/opt/homebrew/opt/postgresql@17/bin", ""):
        proc = run_lib(
            tmp_path,
            "validate_native_inputs && echo OK",
            extra_env={"VOXINT_NATIVE_OLD_PG_BINDIR": val},
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout


@pytest.mark.parametrize("payload", ["a\nb", "x\rY"])
def test_validate_rejects_control_char_in_old_pg_bindir(
    tmp_path: Path, payload: str
) -> None:
    # Both LF and CR forge a record, so both are rejected with the same message.
    proc = run_lib(
        tmp_path,
        "validate_native_inputs",
        extra_env={"VOXINT_NATIVE_OLD_PG_BINDIR": payload},
    )
    assert proc.returncode != 0
    assert "VOXINT_NATIVE_OLD_PG_BINDIR must not contain a newline" in proc.stderr


# --------------------------------------------------------------------------- #
# #6 — worker/beat liveness. They have no /healthz, so `status` derived their
# health from `launchctl print` exit=0 alone, printing a bare [supervised] even
# for a crash-looping job. launchd_job_state parses the (non-API) print output
# conservatively: a running job is running even with a stale non-zero prior
# exit; a not-running job with a non-zero exit is restarting; anything
# unparsable is "state unknown" -- never a false-healthy bare state.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "block,expected",
    [
        # running, clean exit
        ("\tstate = running\n\tpid = 42\n\tlast exit code = 0\n", "running"),
        # running but a STALE non-zero exit from a prior KeepAlive spawn: healthy
        ("\tstate = running\n\tpid = 42\n\tlast exit code = 1\n", "running"),
        # crash loop: throttled between respawns with a non-zero exit
        ("\tstate = waiting\n\tlast exit code = 1\n", "restarting (last exit 1)"),
        # killed by signal (older macOS spells it 'last exit status')
        ("\tstate = not running\n\tlast exit status = 15\n", "restarting (last exit 15)"),
        # loaded but the state/exit fields are absent -> never bare-healthy
        ("\tactive count = 0\n\tpath = /x\n", "state unknown"),
        # non-numeric exit, not running -> unparsable -> state unknown
        ("\tstate = not running\n\tlast exit code = (never exited)\n", "state unknown"),
    ],
)
def test_launchd_job_state_classifies(
    tmp_path: Path, block: str, expected: str
) -> None:
    proc = run_lib(tmp_path, f"launchd_job_state {shlex.quote(block)}")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected


# --------------------------------------------------------------------------- #
# Program argv — one source of truth for plists AND `run --foreground`
# --------------------------------------------------------------------------- #
def test_api_program_args(tmp_path: Path) -> None:
    args = argv_lines(run_lib(tmp_path, "native_program_args api"))
    assert args == [f"{tmp_path}/venv/bin/voxint", "serve"]


def test_worker_program_args(tmp_path: Path) -> None:
    args = argv_lines(run_lib(tmp_path, "native_program_args worker"))
    assert args == [
        f"{tmp_path}/venv/bin/celery",
        "-A",
        "voxint.worker.app",
        "worker",
        "--loglevel=INFO",
    ]


def test_beat_program_args_use_owned_schedule_path(tmp_path: Path) -> None:
    args = argv_lines(run_lib(tmp_path, "native_program_args beat"))
    assert args == [
        f"{tmp_path}/venv/bin/celery",
        "-A",
        "voxint.worker.app",
        "beat",
        "--loglevel=INFO",
        "-s",
        f"{tmp_path}/celerybeat-schedule",
    ]


def test_unknown_service_is_an_error(tmp_path: Path) -> None:
    assert run_lib(tmp_path, "native_program_args migrate").returncode != 0
    assert run_lib(tmp_path, "native_service_env migrate /m").returncode != 0


# --------------------------------------------------------------------------- #
# Env assembly
# --------------------------------------------------------------------------- #
def test_shared_env_present_for_all_services(tmp_path: Path) -> None:
    for svc in CORE_SERVICES:
        env = env_lines(run_lib(tmp_path, f"native_service_env {svc} /media/root"))
        assert env["DATABASE_URL"] == (
            "postgresql+psycopg://voxint:voxint@127.0.0.1:5432/voxint"
        )
        assert env["REDIS_URL"] == "redis://127.0.0.1:6379/0"
        assert env["MEDIA_ROOT"] == "/media/root"
        assert env["COMPUTE_TIER"] == "metal"
        assert env["PYTHONUNBUFFERED"] == "1"
        # Model services are reached over loopback; ports are owned by the metal
        # launcher and bound here by a contract test below.
        assert env["ASR_URL"] == "http://127.0.0.1:8022"
        assert env["DIARIZER_URL"] == "http://127.0.0.1:8024"
        assert env["EMBEDDER_URL"] == "http://127.0.0.1:8021"


def test_path_puts_venv_then_homebrew_first(tmp_path: Path) -> None:
    # launchd inherits no login PATH; the worker resolves the bare "ffmpeg"/
    # "ffprobe" names, so Homebrew's bin must be on PATH ahead of the system.
    env = env_lines(run_lib(tmp_path, "native_service_env worker /media/root"))
    assert env["PATH"] == (
        f"{tmp_path}/venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )


def test_api_env_carries_host_and_port(tmp_path: Path) -> None:
    env = env_lines(run_lib(tmp_path, "native_service_env api /media/root"))
    assert env["API_HOST"] == "127.0.0.1"
    assert env["API_PORT"] == "8080"


def test_worker_and_beat_omit_api_host_port(tmp_path: Path) -> None:
    for svc in ("worker", "beat"):
        env = env_lines(run_lib(tmp_path, f"native_service_env {svc} /media/root"))
        assert "API_HOST" not in env
        assert "API_PORT" not in env


def test_secrets_only_emitted_when_set(tmp_path: Path) -> None:
    # An empty value would override the app's own default with a weaker one, so
    # the launcher must omit the key entirely when unset.
    env = env_lines(run_lib(tmp_path, "native_service_env api /media/root"))
    assert "VOXINT_PASSWORD" not in env
    assert "CSRF_SECRET" not in env
    env = env_lines(
        run_lib(
            tmp_path,
            "native_service_env api /media/root",
            extra_env={
                "VOXINT_NATIVE_PASSWORD": "hunter2",
                "VOXINT_NATIVE_CSRF_SECRET": "x" * 32,
            },
        )
    )
    assert env["VOXINT_PASSWORD"] == "hunter2"
    assert env["CSRF_SECRET"] == "x" * 32


# --------------------------------------------------------------------------- #
# launchd plist generation
# --------------------------------------------------------------------------- #
def render(tmp_path: Path, svc: str, media: str = "/media/root") -> dict:
    out = tmp_path / "out.plist"
    proc = run_lib(tmp_path, f'render_plist {svc} "{media}" "{out}"')
    assert proc.returncode == 0, proc.stderr
    with out.open("rb") as fh:
        return plistlib.load(fh)


def test_plist_label_program_and_workingdir(tmp_path: Path) -> None:
    plist = render(tmp_path, "worker")
    assert plist["Label"] == "com.voxint.native.worker"
    assert plist["ProgramArguments"] == argv_lines(
        run_lib(tmp_path, "native_program_args worker")
    )
    # alembic.ini + the package live at the repo root; celery/alembic config
    # discovery expects to run from there.
    assert plist["WorkingDirectory"] == str(REAL_REPO)


def test_plist_env_matches_service_env_exactly(tmp_path: Path) -> None:
    # launchd inherits no shell environment; the dict must carry everything
    # native_service_env assembles, with no drift between the two code paths.
    for svc in CORE_SERVICES:
        plist = render(tmp_path, svc)
        expected = env_lines(run_lib(tmp_path, f"native_service_env {svc} /media/root"))
        assert plist["EnvironmentVariables"] == expected


def test_plist_supervision_matches_restart_doctrine(tmp_path: Path) -> None:
    # KeepAlive/SuccessfulExit=false == `restart: unless-stopped`: crashes
    # restart, clean exits and bootout stay down.
    plist = render(tmp_path, "api")
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["StandardOutPath"] == f"{tmp_path}/logs/api.log"
    assert plist["StandardErrorPath"] == f"{tmp_path}/logs/api.log"


def test_plist_xml_escapes_hostile_media_root(tmp_path: Path) -> None:
    plist = render(tmp_path, "worker", media="/media/a&b<c>d")
    assert plist["EnvironmentVariables"]["MEDIA_ROOT"] == "/media/a&b<c>d"


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_plist_passes_plutil_lint(tmp_path: Path) -> None:
    out = tmp_path / "out.plist"
    proc = run_lib(tmp_path, f'render_plist api /media/root "{out}"')
    assert proc.returncode == 0, proc.stderr
    lint = subprocess.run(
        ["plutil", "-lint", "-s", str(out)], capture_output=True, text=True
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr


# --------------------------------------------------------------------------- #
# MEDIA_ROOT resolution (pwd -P physical path)
# --------------------------------------------------------------------------- #
def test_resolve_media_root_physical(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    proc = run_lib(tmp_path, f'resolve_media_root "{media}"')
    assert proc.returncode == 0
    # `pwd -P` prints a trailing newline; command substitution strips it in the
    # real callers, so compare on the stripped value.
    assert proc.stdout.rstrip("\n") == str(media.resolve())


def test_resolve_media_root_missing_dir_fails(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, f'resolve_media_root "{tmp_path}/nope"')
    assert proc.returncode != 0


def test_env_value_from_file_strips_quotes(tmp_path: Path) -> None:
    envf = tmp_path / ".env"
    envf.write_text("MEDIA_ROOT='./media'\nOTHER=x\n")
    proc = run_lib(tmp_path, f'env_value_from_file MEDIA_ROOT "{envf}"')
    assert proc.stdout == "./media"


# NOTE: the cross-file drift guards (native argv ↔ compose.yaml, native model
# URLs ↔ metal launcher ports, and MEDIA_ROOT resolution parity) live in
# tests/contracts/test_native_launcher_contract.py, where the pin-parity
# invariants belong. This module stays focused on the launcher's pure logic.


# --------------------------------------------------------------------------- #
# Slice 2 — managed datastores, secrets, and persisted state
# --------------------------------------------------------------------------- #
def test_postgres_program_args(tmp_path: Path) -> None:
    args = argv_lines(run_lib(tmp_path, "native_program_args postgres"))
    assert args == [
        "/opt/homebrew/opt/postgresql@17/bin/postgres",
        "-D",
        f"{tmp_path}/pgdata",
        "-p",
        "5432",
        "-k",
        f"{tmp_path}/run",
        "-c",
        "listen_addresses=127.0.0.1",
    ]


def test_redis_program_args(tmp_path: Path) -> None:
    args = argv_lines(run_lib(tmp_path, "native_program_args redis"))
    assert args == [
        "/opt/homebrew/bin/redis-server",
        "--port",
        "6379",
        "--bind",
        "127.0.0.1",
        "--dir",
        str(tmp_path),
    ]


def test_redis_carries_no_baked_env(tmp_path: Path) -> None:
    # Redis config rides entirely on argv flags; its env is empty.
    proc = run_lib(tmp_path, "native_service_env redis /media/root")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_postgres_carries_only_locale_env(tmp_path: Path) -> None:
    # launchd inherits no locale; without a valid LC_ALL the macOS postmaster
    # dies ("became multithreaded during startup"). Config still rides on argv.
    env = env_lines(run_lib(tmp_path, "native_service_env postgres /media/root"))
    assert env == {"LC_ALL": "C", "LANG": "C"}


def test_datastore_plist_env_and_argv(tmp_path: Path) -> None:
    pg = render(tmp_path, "postgres")
    assert pg["Label"] == "com.voxint.native.postgres"
    assert pg["EnvironmentVariables"] == {"LC_ALL": "C", "LANG": "C"}
    assert pg["ProgramArguments"] == argv_lines(
        run_lib(tmp_path, "native_program_args postgres")
    )
    assert pg["KeepAlive"] == {"SuccessfulExit": False}
    redis = render(tmp_path, "redis")
    assert redis["EnvironmentVariables"] == {}


def test_generate_secret_is_64_hex(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "generate_secret")
    assert proc.returncode == 0
    assert len(proc.stdout) == 64
    assert all(c in "0123456789abcdef" for c in proc.stdout)


def test_next_free_port_returns_a_free_port(tmp_path: Path) -> None:
    # An almost-certainly-free high port comes back unchanged.
    proc = run_lib(tmp_path, "next_free_port 54999")
    assert proc.returncode == 0
    assert proc.stdout == "54999"


def test_load_state_restores_persisted_values(tmp_path: Path) -> None:
    (tmp_path / "state.env").write_text(
        "PG_PORT=5433\nREDIS_PORT=6380\nAPI_PORT=8081\n"
        "DB_PASSWORD=abc123\nVOXINT_PASSWORD=pw\nCSRF_SECRET=cs\n"
    )
    proc = run_lib(
        tmp_path,
        'load_state\nprintf "%s|%s|%s|%s\\n" '
        '"$NATIVE_PG_PORT" "$NATIVE_REDIS_PORT" "$NATIVE_DB_PASSWORD" '
        '"$VOXINT_NATIVE_PASSWORD"',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "5433|6380|abc123|pw"


def test_explicit_env_override_wins_over_state(tmp_path: Path) -> None:
    # A persisted PG_PORT must not clobber an operator's explicit override.
    (tmp_path / "state.env").write_text("PG_PORT=5433\n")
    proc = run_lib(
        tmp_path,
        'load_state\nprintf "%s\\n" "$NATIVE_PG_PORT"',
        extra_env={"VOXINT_NATIVE_PG_PORT": "5999"},
    )
    assert proc.stdout.strip() == "5999"


def test_write_state_env_is_mode_0600(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "write_state_env")
    assert proc.returncode == 0, proc.stderr
    state = tmp_path / "state.env"
    assert state.exists()
    assert (state.stat().st_mode & 0o777) == 0o600
    body = state.read_text()
    assert "PG_PORT=" in body and "DB_PASSWORD=" in body


# --------------------------------------------------------------------------- #
# #71 slice 2a — Postgres major-version skew detection
# --------------------------------------------------------------------------- #
# `cluster_pg_major` / `bindir_pg_major` read the cluster's on-disk major and
# the binaries' major OFFLINE; the `cmd_up` guard (require_cluster_binary_major_
# match) and the doctor line refuse a mismatch that would otherwise fail the
# postmaster with a cryptic "database files are incompatible with server".


def _write_pg_version(home: Path, contents: str) -> None:
    """Lay down $home/pgdata/PG_VERSION (NATIVE_PGDATA), as initdb would."""
    pgdata = home / "pgdata"
    pgdata.mkdir(parents=True, exist_ok=True)
    (pgdata / "PG_VERSION").write_text(contents)


def _make_pg_bindir(
    bindir: Path, version_line: str | None, *, exit_rc: int = 0
) -> Path:
    """A stub bindir whose ``postgres --version`` prints version_line.

    ``version_line=None`` writes a binary that has no version output (just
    ``exit exit_rc``); ``exit_rc`` lets a test model a binary that errors.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    pg = bindir / "postgres"
    if version_line is None:
        pg.write_text(f"#!/bin/bash\nexit {exit_rc}\n")
    else:
        pg.write_text(
            "#!/bin/bash\n"
            'if [ "$1" = "--version" ]; then\n'
            f"  printf '%s\\n' {shlex.quote(version_line)}\n"
            f"  exit {exit_rc}\n"
            "fi\n"
            "exit 0\n"
        )
    pg.chmod(0o755)
    return bindir


@pytest.mark.parametrize(
    "contents,expected",
    [("17\n", "17"), ("  18 \n", "18"), ("17", "17"), ("21\n", "21")],
)
def test_cluster_pg_major_reads_valid(
    tmp_path: Path, contents: str, expected: str
) -> None:
    _write_pg_version(tmp_path, contents)
    proc = run_lib(tmp_path, "cluster_pg_major")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


@pytest.mark.parametrize(
    "contents",
    ["", "   \n", "abc", "9.6\n", "17x", "1 7\n", "1\t7", "1\n7\n"],
)
def test_cluster_pg_major_rejects_invalid(tmp_path: Path, contents: str) -> None:
    # Empty / whitespace / non-numeric / pre-10 dotted / trailing-garbage / and
    # INTERNAL whitespace or multi-line (a corrupted "1 7" / two-line file) all
    # fail closed with NO stdout, so a damaged cluster is never read as a version.
    _write_pg_version(tmp_path, contents)
    proc = run_lib(tmp_path, "cluster_pg_major")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_cluster_pg_major_missing_file(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "cluster_pg_major")  # no pgdata/PG_VERSION at all
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


@pytest.mark.parametrize(
    "version_line,expected",
    [
        ("postgres (PostgreSQL) 17.5", "17"),
        ("postgres (PostgreSQL) 18.0", "18"),
        ("postgres (PostgreSQL) 18beta1", "18"),
        ("postgres (PostgreSQL) 21rc1", "21"),
        # The real Homebrew build appends a vendor suffix -- the parser must read
        # the token after "(PostgreSQL) ", not the last one ("(Homebrew)").
        ("postgres (PostgreSQL) 17.11 (Homebrew)", "17"),
        ("postgres (PostgreSQL) 18.2 (Homebrew)", "18"),
    ],
)
def test_bindir_pg_major_parses(
    tmp_path: Path, version_line: str, expected: str
) -> None:
    bindir = _make_pg_bindir(tmp_path / "b", version_line)
    proc = run_lib(tmp_path, f"bindir_pg_major {shlex.quote(str(bindir))}")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


@pytest.mark.parametrize(
    "version_line",
    ["17.5", "not-postgres 18", "postgres (weird build)", "PostgreSQL 17.5"],
)
def test_bindir_pg_major_rejects_without_marker(
    tmp_path: Path, version_line: str
) -> None:
    # Output lacking the canonical "(PostgreSQL) <ver>" shape fails closed rather
    # than guessing a major from an arbitrary token.
    bindir = _make_pg_bindir(tmp_path / "b", version_line)
    proc = run_lib(tmp_path, f"bindir_pg_major {shlex.quote(str(bindir))}")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_bindir_pg_major_missing_binary(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, f"bindir_pg_major {shlex.quote(str(tmp_path / 'nope'))}")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_bindir_pg_major_nonzero_exit(tmp_path: Path) -> None:
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.5", exit_rc=3)
    proc = run_lib(tmp_path, f"bindir_pg_major {shlex.quote(str(bindir))}")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_bindir_pg_major_unparseable_output(tmp_path: Path) -> None:
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (weird build)")
    proc = run_lib(tmp_path, f"bindir_pg_major {shlex.quote(str(bindir))}")
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""


def test_up_guard_passes_on_match(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.5")
    proc = run_lib(
        tmp_path,
        "require_cluster_binary_major_match",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)},
    )
    assert proc.returncode == 0, proc.stderr


def test_up_guard_refuses_on_mismatch(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 18.1")
    proc = run_lib(
        tmp_path,
        "require_cluster_binary_major_match",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)},
    )
    assert proc.returncode != 0
    assert "major mismatch" in proc.stderr
    assert "v17" in proc.stderr and "v18" in proc.stderr
    # Actionable guidance: install the matching major AND repoint the bindir at it
    # (installing the formula alone does not change which binaries the launcher uses).
    assert "brew install postgresql@17" in proc.stderr
    assert "VOXINT_NATIVE_PG_BINDIR" in proc.stderr


def test_up_guard_fails_closed_on_damaged_cluster(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "garbage\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.5")
    proc = run_lib(
        tmp_path,
        "require_cluster_binary_major_match",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)},
    )
    assert proc.returncode != 0
    assert "damaged" in proc.stderr


def test_up_guard_fails_closed_on_unreadable_binary(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    proc = run_lib(
        tmp_path,
        "require_cluster_binary_major_match",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(tmp_path / "nope")},
    )
    assert proc.returncode != 0
    assert "cannot determine" in proc.stderr


def _doctor_env(bindir: Path) -> dict[str, str]:
    # Skip metal delegation (cmd_doctor shells out to the real metal launcher
    # otherwise) and force the "Postgres not running" SKIP with an unused port so
    # the doctor run stays hermetic and fast.
    return {
        "VOXINT_NATIVE_PG_BINDIR": str(bindir),
        "VOXINT_NATIVE_WITH_MODELS": "0",
        "VOXINT_NATIVE_PG_PORT": "59999",
    }


def test_doctor_reports_major_match(tmp_path: Path) -> None:
    # Real Homebrew vendor-suffixed output must read as a PASS, not a probe error.
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.11 (Homebrew)")
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=_doctor_env(bindir))
    assert "[PASS] cluster major (v17) matches the installed binaries" in proc.stderr


def test_doctor_reports_major_skew(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 18.1 (Homebrew)")
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=_doctor_env(bindir))
    # The [FAIL] marker proves doctor_report ran with FAIL (which sets DOCTOR_RC).
    assert "[FAIL] cluster is v17 but the installed binaries are v18" in proc.stderr
    assert "up will refuse" in proc.stderr


# --------------------------------------------------------------------------- #
# Slice 3 — model delegation, frontend staging, and log rotation
# --------------------------------------------------------------------------- #
def test_metal_script_points_at_the_sibling_launcher(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "metal_script")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == str(REAL_REPO / "scripts" / "metal" / "voxint-metal.sh")


def test_models_delegated_default_on_and_env_off(tmp_path: Path) -> None:
    # Default: the native launcher drives the metal launcher.
    on = run_lib(tmp_path, 'models_delegated && echo yes || echo no')
    assert on.stdout.strip() == "yes"
    # VOXINT_NATIVE_WITH_MODELS=0 (the --no-models persistence) skips delegation.
    off = run_lib(
        tmp_path,
        'models_delegated && echo yes || echo no',
        extra_env={"VOXINT_NATIVE_WITH_MODELS": "0"},
    )
    assert off.stdout.strip() == "no"


def test_no_models_flag_detected_and_stripped(tmp_path: Path) -> None:
    # The predicate main uses to flip the flag, and the filter that removes the
    # token from the positional args (leaving order + non-flag args intact).
    hit = run_lib(
        tmp_path, 'no_models_flag_present up --no-models && echo hit || echo miss'
    )
    assert hit.stdout.strip() == "hit"
    miss = run_lib(
        tmp_path, 'no_models_flag_present logs api && echo hit || echo miss'
    )
    assert miss.stdout.strip() == "miss"
    kept = run_lib(tmp_path, "args_without_no_models logs --no-models api")
    assert kept.stdout.splitlines() == ["logs", "api"]


def test_app_asset_paths_honour_overrides(tmp_path: Path) -> None:
    proc = run_lib(
        tmp_path,
        'printf "%s\\n%s\\n" "$(app_assets_dir)" "$(app_manifest_path)"',
        extra_env={"VOXINT_NATIVE_APP_ASSETS_DIR": str(tmp_path / "app")},
    )
    lines = proc.stdout.splitlines()
    assert lines[0] == str(tmp_path / "app")
    assert lines[1] == str(tmp_path / "app" / ".vite" / "manifest.json")


def test_stage_frontend_dist_overlays_and_keeps_gitkeep(tmp_path: Path) -> None:
    dist = tmp_path / "frontend" / "dist"
    (dist / ".vite").mkdir(parents=True)
    (dist / ".vite" / "manifest.json").write_text('{"main":{"file":"new.js"}}')
    (dist / "new.js").write_text("new")
    app = tmp_path / "app"
    app.mkdir()
    (app / ".gitkeep").write_text("keep\n")
    # A previous build's manifest + hashed asset already staged. Overlay MUST
    # replace the manifest (authoritative) and add the new asset, while leaving
    # the old hashed asset in place so a still-running api does not 404 mid-
    # upgrade -- and it must never touch the tracked .gitkeep.
    (app / ".vite").mkdir()
    (app / ".vite" / "manifest.json").write_text('{"main":{"file":"old.js"}}')
    (app / "old.js").write_text("old")

    proc = run_lib(
        tmp_path,
        "stage_frontend_dist",
        extra_env={
            "VOXINT_NATIVE_FRONTEND_DIR": str(tmp_path / "frontend"),
            "VOXINT_NATIVE_APP_ASSETS_DIR": str(app),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert (app / "new.js").is_file()  # new asset added
    assert (app / ".gitkeep").is_file()  # tracked placeholder survives
    # Manifest replaced by the new build (the only source the fresh api reads).
    assert "new.js" in (app / ".vite" / "manifest.json").read_text()
    # Old hashed asset lingers unreferenced (overlay, not wipe) -- harmless.
    assert (app / "old.js").is_file()


def test_native_model_urls_default_to_metal_loopback_ports(tmp_path: Path) -> None:
    env = env_lines(run_lib(tmp_path, "native_service_env api /media/root"))
    assert env["ASR_URL"] == "http://127.0.0.1:8022"
    assert env["DIARIZER_URL"] == "http://127.0.0.1:8024"
    assert env["EMBEDDER_URL"] == "http://127.0.0.1:8021"


def test_native_model_urls_honour_remote_overrides(tmp_path: Path) -> None:
    # --no-models with the models on other hardware only works if the override
    # actually reaches the baked env (launchd inherits no ambient ASR_URL).
    env = env_lines(
        run_lib(
            tmp_path,
            "native_service_env api /media/root",
            extra_env={
                "VOXINT_NATIVE_ASR_URL": "http://gpubox:9002",
                "VOXINT_NATIVE_DIARIZER_URL": "http://gpubox:9004",
                "VOXINT_NATIVE_EMBEDDER_URL": "http://gpubox:9001",
            },
        )
    )
    assert env["ASR_URL"] == "http://gpubox:9002"
    assert env["DIARIZER_URL"] == "http://gpubox:9004"
    assert env["EMBEDDER_URL"] == "http://gpubox:9001"


def test_rotate_logs_reports_truncation_failure(tmp_path: Path) -> None:
    # A read-only log the archive-copy can read but the truncate cannot empty
    # must surface as a failure. cmd_rotate_logs calls rotate_log_file under an
    # errexit-suppressing `|| rc=1`, so without the guarded truncate the failure
    # would be masked and cmd_rotate_logs would (wrongly) return 0.
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "api.log"
    log.write_bytes(b"x" * (1024 * 1024 + 10))
    log.chmod(0o444)  # read-only: cp succeeds, `: > log` fails
    try:
        proc = run_lib(
            tmp_path,
            "cmd_rotate_logs",
            extra_env={
                "VOXINT_NATIVE_LOG_MAX_MB": "1",
                "VOXINT_NATIVE_LOG_ARCHIVES": "2",
            },
        )
        assert proc.returncode != 0
    finally:
        log.chmod(0o644)


def test_stage_frontend_dist_fails_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "frontend" / "dist").mkdir(parents=True)  # dist but no manifest
    proc = run_lib(
        tmp_path,
        "stage_frontend_dist",
        extra_env={
            "VOXINT_NATIVE_FRONTEND_DIR": str(tmp_path / "frontend"),
            "VOXINT_NATIVE_APP_ASSETS_DIR": str(tmp_path / "app"),
        },
    )
    assert proc.returncode != 0
    assert "manifest" in proc.stderr


def test_build_frontend_skips_gracefully_without_frontend_dir(tmp_path: Path) -> None:
    # No frontend/ dir -> setup degrades (returns 0, says it skipped) rather than
    # failing the whole install.
    proc = run_lib(
        tmp_path,
        "build_frontend",
        extra_env={"VOXINT_NATIVE_FRONTEND_DIR": str(tmp_path / "nope")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "skipping island build" in proc.stderr


def test_rotate_log_file_rotates_when_oversized(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "api.log"
    log.write_bytes(b"x" * (1024 * 1024 + 10))  # just over 1 MB
    proc = run_lib(tmp_path, f'rotate_log_file "{log}" 1 2')
    assert proc.returncode == 0, proc.stderr
    assert log.read_bytes() == b""  # truncated in place (copytruncate)
    archives = [p for p in logs.iterdir() if p.name != "api.log"]
    assert len(archives) == 1 and archives[0].name.startswith("api_")


def test_rotate_log_file_keeps_below_threshold(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "api.log"
    log.write_text("small\n")
    proc = run_lib(tmp_path, f'rotate_log_file "{log}" 1 2')
    assert proc.returncode == 0, proc.stderr
    assert log.read_text() == "small\n"
    assert list(logs.iterdir()) == [log]  # no archive created


def test_prune_log_archives_keeps_newest_n(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "api.log").write_text("live\n")
    stamps = [
        "2026-08-10-00-00-00",
        "2026-08-11-00-00-00",
        "2026-08-12-00-00-00",
        "2026-08-13-00-00-00",
    ]
    for s in stamps:
        (logs / f"api_{s}.log").write_text("old\n")
    proc = run_lib(tmp_path, f'prune_log_archives "{logs / "api.log"}" 2')
    assert proc.returncode == 0, proc.stderr
    remaining = sorted(p.name for p in logs.iterdir() if p.name != "api.log")
    assert remaining == ["api_2026-08-12-00-00-00.log", "api_2026-08-13-00-00-00.log"]


def test_prune_log_archives_no_matching_archives_succeeds(tmp_path: Path) -> None:
    # pipefail regression pin (#11): with zero rotated archives, the internal
    # `ls | grep -E | ...` pipeline's `grep` exits 1 (no match). Under
    # `set -o pipefail` that becomes the pipeline's status; without the `|| true`
    # guard the function would return 1 and falsely fail log rotation. "Nothing to
    # prune" is success -- assert return 0 and that the live log is left intact.
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "api.log").write_text("live\n")
    proc = run_lib(tmp_path, f'prune_log_archives "{logs / "api.log"}" 2')
    assert proc.returncode == 0, proc.stderr
    assert list(logs.iterdir()) == [logs / "api.log"]


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_logrotate_plist_lints_and_invokes_native_rotate(tmp_path: Path) -> None:
    out = tmp_path / "lr.plist"
    proc = run_lib(tmp_path, f'render_logrotate_plist "{out}"')
    assert proc.returncode == 0, proc.stderr
    # plutil is the same gate the launcher applies before bootstrap.
    lint = subprocess.run(
        ["plutil", "-lint", "-s", str(out)], capture_output=True, text=True
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr
    doc = plistlib.loads(out.read_bytes())
    assert doc["Label"] == "com.voxint.native.logrotate"
    assert doc["ProgramArguments"] == [
        "/bin/bash",
        str(REAL_REPO / "scripts" / "native" / "voxint-native.sh"),
        "rotate-logs",
    ]
    assert doc["StartCalendarInterval"] == {"Hour": 3, "Minute": 17}
    assert doc["RunAtLoad"] is False
    assert doc["EnvironmentVariables"]["VOXINT_NATIVE_HOME"] == str(tmp_path)
    assert "VOXINT_NATIVE_LOG_MAX_MB" in doc["EnvironmentVariables"]


def test_rotate_logs_command_covers_all_service_logs(tmp_path: Path) -> None:
    # rotate-logs must sweep every supervised log AND its own; an oversized log
    # for each is rotated in one pass.
    logs = tmp_path / "logs"
    logs.mkdir()
    for svc in ("postgres", "redis", "api", "worker", "beat", "logrotate"):
        (logs / f"{svc}.log").write_bytes(b"x" * (1024 * 1024 + 10))
    proc = run_lib(
        tmp_path,
        "cmd_rotate_logs",
        extra_env={"VOXINT_NATIVE_LOG_MAX_MB": "1", "VOXINT_NATIVE_LOG_ARCHIVES": "3"},
    )
    assert proc.returncode == 0, proc.stderr
    for svc in ("postgres", "redis", "api", "worker", "beat", "logrotate"):
        assert (logs / f"{svc}.log").read_bytes() == b""
        archives = [p for p in logs.iterdir() if p.name.startswith(f"{svc}_")]
        assert len(archives) == 1


# --------------------------------------------------------------------------- #
# Slice 2: restore --fresh (destructive rebuild from a dump)
# --------------------------------------------------------------------------- #
# A representative `pg_restore --list` listing: the pgvector EXTENSION + its
# COMMENT must be filtered out; schema/alembic/app entries must pass through.
SAMPLE_TOC = "\n".join(
    (
        ";     Archive created at ...",
        "2; 3079 16386 EXTENSION - vector ",
        "4502; 0 0 COMMENT - EXTENSION vector ",
        "218; 1259 16714 TABLE public alembic_version voxint",
        "4471; 0 16714 TABLE DATA public alembic_version voxint",
        "4164; 2606 16718 CONSTRAINT public alembic_version alembic_version_pkc voxint",
        "300; 1259 20000 TABLE public pipeline_runs voxint",
        "301; 0 20000 TABLE DATA public pipeline_runs voxint",
    )
)


def test_filter_vector_toc_entries_drops_only_the_extension(tmp_path: Path) -> None:
    proc = run_lib(
        tmp_path,
        'printf %s "$SAMPLE_TOC" | filter_vector_toc_entries',
        extra_env={"SAMPLE_TOC": SAMPLE_TOC},
    )
    assert proc.returncode == 0, proc.stderr
    kept = proc.stdout.splitlines()
    # The two pgvector TOC entries are gone...
    assert not any("EXTENSION - vector" in ln for ln in kept)
    assert not any("COMMENT - EXTENSION vector" in ln for ln in kept)
    # ...but every application entry (schema, alembic_version, app tables) stays.
    assert any("alembic_version" in ln for ln in kept)
    assert any("pipeline_runs" in ln for ln in kept)
    assert sum(1 for ln in kept if "voxint" in ln) == 5


def test_filter_vector_toc_entries_is_a_noop_without_vector(tmp_path: Path) -> None:
    listing = "218; 1259 16714 TABLE public alembic_version voxint\n"
    proc = run_lib(
        tmp_path,
        'printf %s "$LISTING" | filter_vector_toc_entries',
        extra_env={"LISTING": listing},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == listing.strip()


def test_restore_requires_a_dump_file(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "cmd_restore")
    assert proc.returncode != 0
    assert "usage:" in proc.stderr and "restore [--fresh] <dump-file>" in proc.stderr


def test_restore_rejects_unknown_option(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "cmd_restore --bogus /tmp/x.dump")
    assert proc.returncode != 0
    assert "unknown restore option: --bogus" in proc.stderr


def test_restore_fresh_parses_flag_and_validates_dump_exists(tmp_path: Path) -> None:
    # The flag is consumed and the file positional captured, so a missing dump
    # fails at the existence check (not at flag parsing).
    proc = run_lib(tmp_path, "cmd_restore --fresh /no/such/path.dump")
    assert proc.returncode != 0
    assert "no such dump: /no/such/path.dump" in proc.stderr


#: A representative pg_restore --list TOC for a real voxint dump, one entry per
#: line: the pgvector EXTENSION + its COMMENT (filtered out) and the
#: alembic_version table entry the pre-drop identity gate keys on.
_STUB_TOC_VOXINT = (
    "2; 3079 16386 EXTENSION - vector ",
    "218; 1259 16714 TABLE public alembic_version voxint",
    "300; 1259 20000 TABLE public pipeline_runs voxint",
)
#: A structurally valid but NON-voxint dump: no alembic_version / pipeline_runs.
_STUB_TOC_FOREIGN = (
    "2; 3079 16386 EXTENSION - vector ",
    "300; 1259 20000 TABLE public some_other_table postgres",
)


def _make_stub_bindir(
    tmp_path: Path,
    *,
    list_rc: int = 0,
    restore_rc: int = 0,
    toc: tuple[str, ...] = _STUB_TOC_VOXINT,
    running_labels: tuple[str, ...] = (),
    db_present_before: bool = False,
    dump_rc: int = 0,
    presence_probe_rc: int = 0,
    job_blocks: dict[str, str] | None = None,
) -> Path:
    """A fake $NATIVE_PG_BINDIR + a hermetic launchctl on PATH (see _stub_env).

    - pg_isready: always ready (so fresh_restore skips launchd bootstrap).
    - psql: answers the scalar probes fresh_restore issues (SHOW data_directory,
      OIDs, emptiness, alembic presence) and logs every statement to psql.log;
      records the DROP via a marker file. ``db_present_before`` makes the
      pre-mutation ``SELECT 1 FROM pg_database`` presence probe report the target
      as EXISTING (before any DROP marker) so the pre-restore safety backup
      actually fires; after a DROP it reports absent so the fresh-restore
      empty-gate still passes. Default False keeps the historical "absent"
      behaviour (safety backup skipped) for tests that don't exercise it.
    - pg_restore: `--list` prints ``toc`` (or exits ``list_rc``); a restore
      invocation logs its argv to pg_restore.log and exits ``restore_rc``.
    - pg_dump: logs its argv to pg_dump.log and exits ``dump_rc`` (non-zero
      simulates a failed safety backup).
    - events.log: pg_dump, the mutating psql statements (``drop_database`` /
      ``create_extension``), and a REAL pg_restore (not `--list`) each append
      their name here in call order, so a test can prove the safety backup ran
      BEFORE the FIRST mutation via a single shared event log (separate per-bin
      logs can't). ``presence_probe_rc`` makes only the ``SELECT 1 FROM
      pg_database`` presence probe exit non-zero, to drive the fail-closed branch.
    - launchctl: a HERMETIC stub (never the host's real launchctl) so the tests
      can neither be perturbed by nor perturb the developer's real launchd jobs.
      `print` succeeds only for a label listed in ``running_labels``; `bootout`
      removes it. This lets a test pin the services-must-be-down gate.
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir()
    logdir = tmp_path / "stublog"
    logdir.mkdir()
    running = logdir / "running_labels"
    running.write_text("".join(f"{label}\n" for label in running_labels))
    # Optional per-label `launchctl print` body (for #6 status liveness). A
    # running label with no explicit block gets a default healthy running body.
    for label, block in (job_blocks or {}).items():
        (logdir / f"block_{label}").write_text(block)

    (bindir / "pg_isready").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bindir / "psql").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/psql.log"\n'
        f'MARK="{logdir}/dropped"\n'
        f'EVENTS="{logdir}/events.log"\n'
        "sql=\"\"; mode=stmt\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        "    -tAc) mode=scalar; shift; sql=$1 ;;\n"
        "    -c)   mode=stmt;   shift; sql=$1 ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        'printf "%s\\n" "$sql" >> "$LOG"\n'
        # Record the MUTATING statements in the shared event log too, so a test
        # can prove the safety backup ran before the FIRST mutation (the DROP on
        # --fresh, the CREATE EXTENSION on plain) -- not merely before pg_restore.
        'case "$sql" in\n'
        '  *"DROP DATABASE"*)   printf "drop_database\\n" >> "$EVENTS" ;;\n'
        '  *"CREATE EXTENSION"*) printf "create_extension\\n" >> "$EVENTS" ;;\n'
        "esac\n"
        'case "$sql" in\n'
        # SHOW data_directory must echo exactly $NATIVE_PGDATA so the
        # managed-cluster identity check passes ($VOXINT_NATIVE_HOME/pgdata).
        '  *data_directory*) echo "$VOXINT_NATIVE_HOME/pgdata"; exit 0 ;;\n'
        '  *"DROP DATABASE"*) : > "$MARK"; exit 0 ;;\n'
        '  *"count(*)"*) echo 0; exit 0 ;;\n'
        '  *information_schema*) echo 1; exit 0 ;;\n'
        '  *"oid FROM pg_database"*) [ -f "$MARK" ] && echo 20000 || echo 16386; exit 0 ;;\n'
        # Presence probe: report EXISTING before any DROP (so the safety backup
        # fires) only when db_present_before is set; absent after the DROP marker
        # (so the fresh empty-gate still passes). Default → always empty. A
        # non-zero presence_probe_rc makes ONLY this probe fail, to exercise the
        # helper's fail-closed branch (a probe error must not read as "absent").
        '  *"SELECT 1 FROM pg_database"*)\n'
        f"    [ {presence_probe_rc} -eq 0 ] || exit {presence_probe_rc}\n"
        f'    if [ {1 if db_present_before else 0} -eq 1 ] && [ ! -f "$MARK" ]; then echo 1; fi\n'
        "    exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    toc_args = " ".join(f'"{line}"' for line in toc)
    (bindir / "pg_restore").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/pg_restore.log"\n'
        f'LFILE="{logdir}/restore_L.toc"\n'
        'if [ "$1" = "--list" ]; then\n'
        f"  [ {list_rc} -eq 0 ] || exit {list_rc}\n"
        f'  printf "%s\\n" {toc_args}\n'
        "  exit 0\n"
        "fi\n"
        # A real restore invocation: log the argv AND copy the actual -L file so a
        # test can assert the filtered TOC handed to pg_restore is vector-free.
        'printf "%s\\n" "$*" >> "$LOG"\n'
        f'printf "pg_restore\\n" >> "{logdir}/events.log"\n'
        'prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "-L" ]; then cp "$a" "$LFILE" 2>/dev/null || true; fi\n'
        '  prev=$a\n'
        "done\n"
        f"exit {restore_rc}\n"
    )
    (bindir / "pg_dump").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{logdir}/pg_dump.log"\n'
        f'printf "pg_dump\\n" >> "{logdir}/events.log"\n'
        'printf "PGDMP-stub\\n"\n'  # bytes on stdout for cmd_backup's `> "$out"`
        f"exit {dump_rc}\n"
    )
    (bindir / "launchctl").write_text(
        "#!/usr/bin/env bash\n"
        f'RUNNING="{running}"\n'
        'cmd=$1; shift\n'
        'if [ "$cmd" = "print" ]; then\n'
        '  label=${1##*/}\n'
        '  if grep -qxF "$label" "$RUNNING" 2>/dev/null; then\n'
        f'    blk="{logdir}/block_$label"\n'
        '    if [ -f "$blk" ]; then cat "$blk"; else\n'
        "      printf '\\tstate = running\\n\\tlast exit code = 0\\n'\n"
        "    fi\n"
        "    exit 0\n"
        "  fi\n"
        "  exit 1\n"
        "fi\n"
        'if [ "$cmd" = "bootout" ]; then\n'
        '  label=${1##*/}\n'
        '  grep -vxF "$label" "$RUNNING" > "$RUNNING.new" 2>/dev/null || true\n'
        '  mv "$RUNNING.new" "$RUNNING" 2>/dev/null || true\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    for name in ("pg_isready", "psql", "pg_restore", "pg_dump", "launchctl"):
        (bindir / name).chmod(0o755)
    return bindir


def _stub_env(bindir: Path) -> dict[str, str]:
    """extra_env pointing the launcher at the stub bindir and the hermetic
    launchctl (prepended to PATH so bare `launchctl` never hits the host)."""
    return {
        "VOXINT_NATIVE_PG_BINDIR": str(bindir),
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
    }


def _prime_install(tmp_path: Path) -> None:
    """Minimal on-disk state so managed_cluster passes and alembic is stubbed."""
    (tmp_path / "pgdata").mkdir()
    (tmp_path / "pgdata" / "PG_VERSION").write_text("17\n")  # managed_cluster true
    venvbin = tmp_path / "venv" / "bin"
    venvbin.mkdir(parents=True)
    (venvbin / "alembic").write_text("#!/usr/bin/env bash\nexit 0\n")
    (venvbin / "alembic").chmod(0o755)


def test_fresh_restore_preflights_before_dropping(tmp_path: Path) -> None:
    # An invalid archive must be rejected BEFORE any destructive DDL: the psql
    # stub's DROP marker must never appear.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, list_rc=1)
    dump = tmp_path / "bad.dump"
    dump.write_text("not a real archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not a valid pg_restore archive" in proc.stderr
    assert not (tmp_path / "stublog" / "dropped").exists()
    # No restore invocation was logged either.
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_fresh_restore_uses_safe_restore_flags(tmp_path: Path) -> None:
    # Full happy path over stubs: assert the restore command line is the honest,
    # ownership-correct one (filtered TOC, no --clean) and the emptiness gate ran.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, list_rc=0)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    assert "EMPTY_DB PASS" in proc.stderr
    assert "restore --fresh complete" in proc.stderr
    # The destructive drop did happen (marker present) after preflight.
    assert (tmp_path / "stublog" / "dropped").exists()
    restore_argv = (tmp_path / "stublog" / "pg_restore.log").read_text()
    assert "-L " in restore_argv
    assert "--no-owner" in restore_argv
    assert "--single-transaction" in restore_argv
    assert "--exit-on-error" in restore_argv
    assert "--clean" not in restore_argv  # the footgun path must NOT be used
    # The extension was preinstalled by the superuser (not left to the dump).
    psql_calls = (tmp_path / "stublog" / "psql.log").read_text()
    assert "CREATE EXTENSION vector" in psql_calls
    assert "TEMPLATE template0" in psql_calls


def test_fresh_restore_takes_safety_backup(tmp_path: Path) -> None:
    # --fresh drops the database, so an existing target must be dumped to a 0600
    # pre-image FIRST. This locks the shared helper refactor: fresh keeps taking
    # its safety backup, under its own filename label, before the destructive drop.
    _prime_install(tmp_path)
    (tmp_path / "backups").mkdir()
    bindir = _make_stub_bindir(tmp_path, db_present_before=True)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    assert "SAFETY_BACKUP" in proc.stderr
    backups = list((tmp_path / "backups").glob("pre-fresh-restore-*.dump"))
    assert len(backups) == 1, backups
    assert oct(backups[0].stat().st_mode & 0o777) == "0o600"
    assert not list((tmp_path / "backups").glob("*.partial"))
    # The safety dump (pg_dump) precedes the FIRST mutation -- for --fresh that is
    # the destructive DROP DATABASE -- and the restore, in the shared event log.
    assert (tmp_path / "stublog" / "dropped").exists()
    events = (tmp_path / "stublog" / "events.log").read_text().split()
    assert events.index("pg_dump") < events.index("drop_database") < events.index(
        "pg_restore"
    ), events


def test_fresh_restore_refuses_when_core_service_running(tmp_path: Path) -> None:
    # The services-must-be-down gate is destructive-safety-critical: if any of
    # api/worker/beat is still supervised, fresh_restore must refuse BEFORE any
    # drop. The hermetic launchctl stub reports the api job as running.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, running_labels=("com.voxint.native.api",))
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "refusing destructive restore" in proc.stderr
    assert "api" in proc.stderr
    # Nothing was dropped and no restore ran.
    assert not (tmp_path / "stublog" / "dropped").exists()
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_fresh_restore_refuses_non_voxint_dump_before_dropping(tmp_path: Path) -> None:
    # A structurally valid pg_restore archive of some OTHER database must not
    # destroy the live voxint DB: the pre-drop identity gate (alembic_version in
    # the TOC) refuses it BEFORE the drop.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, toc=_STUB_TOC_FOREIGN)
    dump = tmp_path / "foreign.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not a voxint dump" in proc.stderr
    assert not (tmp_path / "stublog" / "dropped").exists()
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_fresh_restore_refuses_foreign_postmaster(tmp_path: Path) -> None:
    # postgres_reachable accepts any listener on the port; before any drop,
    # fresh_restore must confirm SHOW data_directory matches the managed cluster.
    # Point the psql stub at a mismatched data_directory to simulate a foreign
    # postmaster holding the port.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path)
    # Override the psql stub so SHOW data_directory returns a foreign path.
    (bindir / "psql").write_text(
        "#!/usr/bin/env bash\n"
        "sql=\"\"\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -tAc|-c) shift; sql=$1 ;; esac\n'
        "  shift\n"
        "done\n"
        'case "$sql" in *data_directory*) echo /some/foreign/pgdata; exit 0 ;; esac\n'
        "exit 0\n"
    )
    (bindir / "psql").chmod(0o755)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not the managed cluster" in proc.stderr
    assert not (tmp_path / "stublog" / "dropped").exists()


def test_filter_vector_toc_entries_keeps_similarly_named_extensions(tmp_path: Path) -> None:
    # The filter must strip ONLY the exact `vector` extension, never a different
    # extension whose name merely begins with "vector" (e.g. vectorize).
    listing = "\n".join(
        (
            "2; 3079 16386 EXTENSION - vector ",
            "3; 3079 16400 EXTENSION - vectorize ",
            "4502; 0 0 COMMENT - EXTENSION vector ",
            "4503; 0 0 COMMENT - EXTENSION vectorscale ",
            "300; 1259 20000 TABLE public speaker_embeddings voxint",
        )
    )
    proc = run_lib(
        tmp_path,
        'printf %s "$LISTING" | filter_vector_toc_entries',
        extra_env={"LISTING": listing},
    )
    assert proc.returncode == 0, proc.stderr
    kept = proc.stdout.splitlines()
    # The exact vector extension + its COMMENT are gone...
    assert not any(ln.rstrip().endswith("EXTENSION - vector") for ln in kept)
    assert not any("COMMENT - EXTENSION vector " in ln for ln in kept)
    # ...but the similarly-named extensions and the app table survive.
    assert any("vectorize" in ln for ln in kept)
    assert any("vectorscale" in ln for ln in kept)
    assert any("speaker_embeddings" in ln for ln in kept)


def test_restore_rejects_extra_positional_dump(tmp_path: Path) -> None:
    # A destructive command must not silently pick the last of several paths.
    proc = run_lib(tmp_path, "cmd_restore --fresh /tmp/a.dump /tmp/b.dump")
    assert proc.returncode != 0
    assert "exactly one dump file" in proc.stderr


# --------------------------------------------------------------------------- #
# #71 slice 1: plain `restore` (in-place, vector-safe) + vector-free backups
# --------------------------------------------------------------------------- #
def test_plain_restore_uses_safe_clean_flags(tmp_path: Path) -> None:
    # Happy path over stubs: the plain (non-fresh) restore must run the
    # ownership-correct, atomic in-place command -- a FILTERED TOC (`-L`),
    # `--clean --if-exists` for replacement, and `--single-transaction
    # --exit-on-error` for rollback -- and it must NOT drop/recreate the database
    # (that is `--fresh`'s job). The pgvector extension is (idempotently)
    # preinstalled by the superuser, never recreated by the voxint role.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, list_rc=0)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    assert "restore complete" in proc.stderr
    assert "restored in place" in proc.stderr
    restore_argv = (tmp_path / "stublog" / "pg_restore.log").read_text()
    assert "-L " in restore_argv
    assert "--clean" in restore_argv
    assert "--if-exists" in restore_argv
    assert "--no-owner" in restore_argv
    assert "--single-transaction" in restore_argv
    assert "--exit-on-error" in restore_argv
    # In-place: the DB was NOT dropped and NOT recreated from template0.
    assert not (tmp_path / "stublog" / "dropped").exists()
    psql_calls = (tmp_path / "stublog" / "psql.log").read_text()
    assert "TEMPLATE template0" not in psql_calls
    # The extension is preinstalled idempotently (IF NOT EXISTS -- the target
    # usually already has it), so a bare CREATE never errors on a present ext.
    assert "CREATE EXTENSION IF NOT EXISTS vector" in psql_calls


def test_plain_restore_takes_safety_backup(tmp_path: Path) -> None:
    # A plain (in-place) restore of a valid-but-wrong/older dump silently REPLACES
    # live data -- --single-transaction only rolls back a FAILED restore, never an
    # unwanted successful one. So the plain path must take a recoverable pre-image
    # FIRST, before any mutation, exactly like --fresh does. Prove the ordering
    # with a single shared event log: the safety backup (pg_dump) must run strictly
    # BEFORE the restore (pg_restore) -- separate per-bin logs cannot prove order.
    _prime_install(tmp_path)
    (tmp_path / "backups").mkdir()
    bindir = _make_stub_bindir(tmp_path, db_present_before=True)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    # The safety backup ran, was announced, and landed as a 0600 pre-restore dump.
    assert "SAFETY_BACKUP" in proc.stderr
    dump_argv = (tmp_path / "stublog" / "pg_dump.log").read_text()
    assert "--exclude-extension=vector" in dump_argv
    assert "-Fc" in dump_argv
    backups = list((tmp_path / "backups").glob("pre-restore-*.dump"))
    assert len(backups) == 1, backups
    assert oct(backups[0].stat().st_mode & 0o777) == "0o600"
    assert not list((tmp_path / "backups").glob("*.partial"))  # finalized, no leak
    # Ordering: the safety backup (pg_dump) runs strictly before the FIRST
    # mutation -- for the plain path that is CREATE EXTENSION -- and before the
    # restore itself. A regression placing the backup after either would fail.
    events = (tmp_path / "stublog" / "events.log").read_text().split()
    assert events.index("pg_dump") < events.index("create_extension") < events.index(
        "pg_restore"
    ), events
    # And the restore itself still ran to completion.
    assert "restore complete" in proc.stderr


def test_plain_restore_aborts_if_safety_backup_fails(tmp_path: Path) -> None:
    # A failed safety backup must ABORT before anything is touched: refusing is
    # safer than restoring in place with no recoverable pre-image. Nothing
    # downstream runs -- no pg_restore, no CREATE EXTENSION -- and no artefacts
    # leak: neither a finalized dump nor a stray .partial.
    _prime_install(tmp_path)
    (tmp_path / "backups").mkdir()
    bindir = _make_stub_bindir(tmp_path, db_present_before=True, dump_rc=1)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "safety backup failed" in proc.stderr
    # Nothing downstream ran: no restore invocation, no extension DDL.
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()
    events_file = tmp_path / "stublog" / "events.log"
    if events_file.exists():
        assert "pg_restore" not in events_file.read_text().split()
    psql_calls = (tmp_path / "stublog" / "psql.log").read_text()
    assert "CREATE EXTENSION" not in psql_calls
    # No finalized safety dump and no leaked partial.
    assert not list((tmp_path / "backups").glob("pre-restore-*.dump"))
    assert not list((tmp_path / "backups").glob("*.partial"))


def test_plain_restore_aborts_if_presence_probe_fails(tmp_path: Path) -> None:
    # Fail-closed: if the "does the database exist?" probe ERRORS, it must NOT be
    # read as "absent" and skipped past on the way to a mutation -- that would
    # restore in place with no pre-image. A probe error aborts the whole operation
    # before the backup even starts: no dump, no DDL, no restore, no artifacts.
    _prime_install(tmp_path)
    (tmp_path / "backups").mkdir()
    bindir = _make_stub_bindir(tmp_path, presence_probe_rc=1)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "could not check whether" in proc.stderr  # the fail-closed probe branch
    # The backup never even started, and nothing downstream ran.
    assert not (tmp_path / "stublog" / "pg_dump.log").exists()
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()
    psql_calls = (tmp_path / "stublog" / "psql.log").read_text()
    assert "CREATE EXTENSION" not in psql_calls
    assert not (tmp_path / "stublog" / "dropped").exists()
    assert not list((tmp_path / "backups").glob("*.dump"))
    assert not list((tmp_path / "backups").glob("*.partial"))


def test_plain_restore_safety_backup_is_collision_safe(tmp_path: Path) -> None:
    # Two restores in the same wall-clock second share a date stamp; the backup
    # name must NOT let `mv` clobber an earlier same-second backup. With `date`
    # frozen and both the unsuffixed final AND its partial pre-existing, the helper
    # must write a SUFFIXED backup and leave both existing files untouched.
    _prime_install(tmp_path)
    (tmp_path / "backups").mkdir()
    bindir = _make_stub_bindir(tmp_path, db_present_before=True)
    # Freeze the timestamp so the collision is deterministic.
    (bindir / "date").write_text('#!/usr/bin/env bash\necho "2026-01-01-00-00-00"\n')
    (bindir / "date").chmod(0o755)
    stamp = "2026-01-01-00-00-00"
    existing = tmp_path / "backups" / f"pre-restore-{stamp}.dump"
    existing.write_text("OLD")
    existing_partial = tmp_path / "backups" / f"pre-restore-{stamp}.dump.partial"
    existing_partial.write_text("OLDPARTIAL")
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    # A suffixed backup was written; the pre-existing files are byte-for-byte intact.
    suffixed = tmp_path / "backups" / f"pre-restore-{stamp}-2.dump"
    assert suffixed.exists()
    assert oct(suffixed.stat().st_mode & 0o777) == "0o600"
    assert f"SAFETY_BACKUP {suffixed}" in proc.stderr
    assert existing.read_text() == "OLD"
    assert existing_partial.read_text() == "OLDPARTIAL"


def test_plain_restore_refuses_when_core_service_running(tmp_path: Path) -> None:
    # `--clean` against a live app is a data-corruption footgun: the plain path
    # must carry the SAME services-down gate as `--fresh` and refuse BEFORE any
    # restore runs.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, running_labels=("com.voxint.native.worker",))
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "refusing destructive restore" in proc.stderr
    assert "worker" in proc.stderr
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_plain_restore_refuses_non_voxint_dump(tmp_path: Path) -> None:
    # A valid pg_restore archive of some OTHER database must not `--clean`-mutate
    # the live voxint DB: the archive-identity gate (alembic_version in the TOC)
    # refuses it BEFORE any restore.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, toc=_STUB_TOC_FOREIGN)
    dump = tmp_path / "foreign.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not a voxint dump" in proc.stderr
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_plain_restore_refuses_foreign_alembic_project(tmp_path: Path) -> None:
    # A dump from a DIFFERENT alembic project (has alembic_version but none of
    # voxint's tables) must not `--clean`-mutate the voxint DB: the identity gate
    # also requires a voxint-specific table (pipeline_runs).
    _prime_install(tmp_path)
    foreign_alembic = (
        "2; 3079 16386 EXTENSION - vector ",
        "218; 1259 16714 TABLE public alembic_version someapp",
        "300; 1259 20000 TABLE public widgets someapp",
    )
    bindir = _make_stub_bindir(tmp_path, toc=foreign_alembic)
    dump = tmp_path / "otherapp.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "no pipeline_runs table" in proc.stderr
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_plain_restore_refuses_foreign_postmaster(tmp_path: Path) -> None:
    # The listener on the port must be proven to be the managed cluster before any
    # `--clean` mutation: a mismatched SHOW data_directory is refused.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path)
    (bindir / "psql").write_text(
        "#!/usr/bin/env bash\n"
        "sql=\"\"\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -tAc|-c) shift; sql=$1 ;; esac\n'
        "  shift\n"
        "done\n"
        'case "$sql" in *data_directory*) echo /some/foreign/pgdata; exit 0 ;; esac\n'
        "exit 0\n"
    )
    (bindir / "psql").chmod(0o755)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not the managed cluster" in proc.stderr
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_plain_restore_rejects_invalid_archive(tmp_path: Path) -> None:
    # A non-archive must be rejected BEFORE any restore runs.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, list_rc=1)
    dump = tmp_path / "bad.dump"
    dump.write_text("not a real archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "not a valid pg_restore archive" in proc.stderr
    assert not (tmp_path / "stublog" / "pg_restore.log").exists()


def test_backup_excludes_vector_extension(tmp_path: Path) -> None:
    # New dumps must omit the pgvector EXTENSION (+ its COMMENT) so a later
    # restore never has to strip them: pg_dump is invoked with
    # --exclude-extension=vector AND the custom format (-Fc) restore expects.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path)
    (tmp_path / "backups").mkdir()
    proc = run_lib(tmp_path, "cmd_backup", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    pg_dump_argv = (tmp_path / "stublog" / "pg_dump.log").read_text()
    assert "--exclude-extension=vector" in pg_dump_argv
    assert "-Fc" in pg_dump_argv  # custom format -- plain SQL would break restore


def test_plain_restore_L_file_is_vector_free(tmp_path: Path) -> None:
    # The safety-critical property end-to-end: the actual -L list handed to
    # pg_restore (not just the presence of `-L`) must carry NO vector EXTENSION/
    # COMMENT entry, or --clean could recreate the privileged extension as voxint.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, toc=SAMPLE_TOC.splitlines())
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    lfile = (tmp_path / "stublog" / "restore_L.toc").read_text()
    assert "EXTENSION - vector" not in lfile
    assert "COMMENT - EXTENSION vector" not in lfile
    # ...and it is a real, non-empty list with the app entries preserved.
    assert "alembic_version" in lfile
    assert "pipeline_runs" in lfile


def test_fresh_restore_L_file_is_vector_free(tmp_path: Path) -> None:
    # Same end-to-end filter proof for the destructive path.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, toc=SAMPLE_TOC.splitlines())
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    proc = run_lib(tmp_path, f"cmd_restore --fresh {dump}", extra_env=_stub_env(bindir))
    assert proc.returncode == 0, proc.stderr
    lfile = (tmp_path / "stublog" / "restore_L.toc").read_text()
    assert "EXTENSION - vector" not in lfile
    assert "alembic_version" in lfile


def test_plain_restore_reports_rollback_on_pg_restore_failure(tmp_path: Path) -> None:
    # A failed pg_restore must fail-closed with the honest "left unchanged"
    # rollback message and leave no temp files behind.
    _prime_install(tmp_path)
    bindir = _make_stub_bindir(tmp_path, restore_rc=1)
    dump = tmp_path / "good.dump"
    dump.write_text("archive")
    # Isolate TMPDIR so the leaked-temp assertion sees only this test's temps.
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    env = {**_stub_env(bindir), "TMPDIR": str(tmpdir)}
    proc = run_lib(tmp_path, f"cmd_restore {dump}", extra_env=env)
    assert proc.returncode != 0
    assert "left unchanged" in proc.stderr
    assert "dump file is untouched" in proc.stderr
    # The plain path never drops the DB, so no drop marker either.
    assert not (tmp_path / "stublog" / "dropped").exists()
    # No restore temp files leaked (the EXIT trap + explicit rm cover the failure).
    leaked = list(tmpdir.glob("voxint-restore-*"))
    assert leaked == [], f"leaked restore temp files: {leaked}"


# --------------------------------------------------------------------------- #
# #71 slice 2b — `upgrade-db` (dump/restore major-version upgrade)
# --------------------------------------------------------------------------- #
# The destructive major-version upgrade: dump the OLD server with the NEW pg_dump
# over a private socket, prove the archive restorable BEFORE touching pgdata,
# atomically rename the old cluster aside as a rollback, initdb the new major, and
# rebuild via fresh_restore in a subshell. These offline tests drive the whole
# orchestration over stub binaries with an event log, plus the pure helpers
# (version gate, resolve_old_pg_bindir, rollback shape) in isolation.


def _upg_run(
    home: Path, args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return run_lib(home, f"cmd_upgrade_db {args}", extra_env=extra_env)


# --- pure version gate ----------------------------------------------------- #
def test_upgrade_same_major_is_a_noop(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.5")
    proc = _upg_run(tmp_path, "", extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)})
    assert proc.returncode == 0, proc.stderr
    assert "already on Postgres 17 -- nothing to do" in proc.stderr


def test_upgrade_rejects_downgrade(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "18\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 17.5")
    proc = _upg_run(tmp_path, "", extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)})
    assert proc.returncode != 0
    assert "downgrades are not supported" in proc.stderr


def test_upgrade_rejects_skipped_major(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 19.0")
    proc = _upg_run(tmp_path, "", extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)})
    assert proc.returncode != 0
    assert "only one-major-forward upgrades are supported" in proc.stderr


def test_upgrade_fails_closed_on_damaged_cluster(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "garbage\n")
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 18.1")
    proc = _upg_run(tmp_path, "", extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)})
    assert proc.returncode != 0
    assert "damaged" in proc.stderr


def test_upgrade_no_managed_cluster(tmp_path: Path) -> None:
    bindir = _make_pg_bindir(tmp_path / "b", "postgres (PostgreSQL) 18.1")
    proc = _upg_run(tmp_path, "", extra_env={"VOXINT_NATIVE_PG_BINDIR": str(bindir)})
    assert proc.returncode != 0
    assert "no managed cluster to upgrade" in proc.stderr


# --- argument parsing ------------------------------------------------------ #
def test_upgrade_rejects_unknown_option(tmp_path: Path) -> None:
    proc = _upg_run(tmp_path, "--bogus")
    assert proc.returncode != 0
    assert "unknown upgrade-db option: --bogus" in proc.stderr


def test_upgrade_rejects_positional(tmp_path: Path) -> None:
    proc = _upg_run(tmp_path, "somefile")
    assert proc.returncode != 0
    assert "takes no positional arguments" in proc.stderr


def test_upgrade_rehearse_and_rollback_mutually_exclusive(tmp_path: Path) -> None:
    proc = _upg_run(tmp_path, "--rehearse --rollback")
    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stderr


# --- resolve_old_pg_bindir ------------------------------------------------- #
def _no_brew_env(tmp_path: Path) -> dict[str, str]:
    """Prepend a `brew` stub that fails, so the brew-keg branch of
    resolve_old_pg_bindir cannot resolve — without clobbering PATH (which would
    hide `bash` itself). The real PATH stays after it."""
    binp = tmp_path / "nobrew"
    binp.mkdir(exist_ok=True)
    (binp / "brew").write_text("#!/usr/bin/env bash\nexit 1\n")
    (binp / "brew").chmod(0o755)
    return {"PATH": f"{binp}:{os.environ.get('PATH', '')}"}


def test_resolve_old_bindir_override_wins(tmp_path: Path) -> None:
    old = _make_pg_bindir(tmp_path / "old", "postgres (PostgreSQL) 17.5")
    proc = run_lib(
        tmp_path,
        "resolve_old_pg_bindir 17 0",
        extra_env={"VOXINT_NATIVE_OLD_PG_BINDIR": str(old)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(old)


def test_resolve_old_bindir_override_wrong_major_fails(tmp_path: Path) -> None:
    # An override pointing at the WRONG major must fail closed, not be trusted.
    old = _make_pg_bindir(tmp_path / "old", "postgres (PostgreSQL) 16.9")
    proc = run_lib(
        tmp_path,
        "resolve_old_pg_bindir 17 0",
        extra_env={"VOXINT_NATIVE_OLD_PG_BINDIR": str(old)},
    )
    assert proc.returncode != 0
    assert "is Postgres v16, not the cluster's v17" in proc.stderr


def test_resolve_old_bindir_rehearse_uses_current(tmp_path: Path) -> None:
    # In --rehearse the current bindir may serve as the "old" one iff same major.
    cur = _make_pg_bindir(tmp_path / "cur", "postgres (PostgreSQL) 17.5")
    proc = run_lib(
        tmp_path,
        "resolve_old_pg_bindir 17 1",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(cur), **_no_brew_env(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(cur)


def test_resolve_old_bindir_current_not_used_without_rehearse(tmp_path: Path) -> None:
    # Outside rehearsal the current (new-major) bindir must NOT be accepted as old;
    # with no override and no brew keg, it fails closed with the install hint.
    cur = _make_pg_bindir(tmp_path / "cur", "postgres (PostgreSQL) 17.5")
    proc = run_lib(
        tmp_path,
        "resolve_old_pg_bindir 17 0",
        extra_env={"VOXINT_NATIVE_PG_BINDIR": str(cur), **_no_brew_env(tmp_path)},
    )
    assert proc.returncode != 0
    assert "brew install postgresql@17" in proc.stderr


# --- require_stack_fully_down ---------------------------------------------- #
def test_stack_down_refuses_when_core_service_running(tmp_path: Path) -> None:
    bindir = _make_stub_bindir(tmp_path, running_labels=("com.voxint.native.worker",))
    proc = run_lib(tmp_path, "require_stack_fully_down", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "worker" in proc.stderr


def test_stack_down_refuses_when_datastore_supervised(tmp_path: Path) -> None:
    bindir = _make_stub_bindir(tmp_path, running_labels=("com.voxint.native.postgres",))
    proc = run_lib(tmp_path, "require_stack_fully_down", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "managed datastores still supervised" in proc.stderr
    assert "postgres" in proc.stderr


# --- rollback shape (idempotent, never deletes) ---------------------------- #
def _hermetic_launchctl_env(tmp_path: Path) -> dict[str, str]:
    """A PATH whose launchctl always reports 'not loaded' (so _stop_managed_postgres
    is a hermetic no-op) without a full stub bindir."""
    binp = tmp_path / "lc"
    binp.mkdir()
    (binp / "launchctl").write_text("#!/usr/bin/env bash\nexit 1\n")
    (binp / "launchctl").chmod(0o755)
    return {"PATH": f"{binp}:{os.environ.get('PATH', '')}"}


def test_rollback_preserves_partial_and_restores_old(tmp_path: Path) -> None:
    # Auto-rollback shape: a partial NEW pgdata + the retained OLD cluster. The
    # partial must be preserved as pgdata.failed-<stamp> (NEVER deleted) and the
    # retained old cluster renamed back to pgdata.
    (tmp_path / "pgdata").mkdir()
    (tmp_path / "pgdata" / "marker").write_text("PARTIAL")
    retained = tmp_path / "pgdata.pg17-2026-08-18-10-30-00"
    retained.mkdir()
    (retained / "PG_VERSION").write_text("17\n")
    proc = run_lib(
        tmp_path,
        "_upgrade_rollback 17 2026-08-18-10-30-00",
        extra_env=_hermetic_launchctl_env(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "17\n"  # old restored
    failed = tmp_path / "pgdata.failed-2026-08-18-10-30-00"
    assert failed.is_dir() and (failed / "marker").read_text() == "PARTIAL"  # kept
    assert not retained.exists()  # moved back to pgdata


def test_rollback_is_idempotent(tmp_path: Path) -> None:
    # A second rollback with the retained dir already gone must be a safe no-op —
    # it must NOT set aside the good restored pgdata again.
    (tmp_path / "pgdata").mkdir()
    (tmp_path / "pgdata" / "PG_VERSION").write_text("17\n")  # already restored
    proc = run_lib(
        tmp_path,
        "_upgrade_rollback 17 2026-08-18-10-30-00",
        extra_env=_hermetic_launchctl_env(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert "nothing to restore" in proc.stderr
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "17\n"
    assert not list(tmp_path.glob("pgdata.failed-*"))  # the good cluster untouched


def _down_stack_env(tmp_path: Path) -> dict[str, str]:
    """A hermetic env where require_stack_fully_down passes: launchctl reports
    nothing loaded and the PG port is a free high port nothing listens on."""
    return {**_hermetic_launchctl_env(tmp_path), "VOXINT_NATIVE_PG_PORT": "59998"}


def test_rollback_refuses_to_move_if_postgres_wont_stop(tmp_path: Path) -> None:
    # If the managed Postgres cannot be stopped (launchctl keeps reporting it
    # loaded), rollback must NOT rename any data directory -- moving a live cluster
    # would corrupt it. The retained + partial dirs must be left exactly as they are.
    (tmp_path / "pgdata").mkdir()
    (tmp_path / "pgdata" / "marker").write_text("PARTIAL")
    retained = tmp_path / "pgdata.pg17-2026-08-18-10-30-00"
    retained.mkdir()
    (retained / "PG_VERSION").write_text("17\n")
    # A launchctl whose `print` always succeeds => the label never unloads =>
    # _stop_managed_postgres times out and returns non-zero.
    binp = tmp_path / "stuck"
    binp.mkdir()
    (binp / "launchctl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (binp / "launchctl").chmod(0o755)
    env = {"PATH": f"{binp}:{os.environ.get('PATH', '')}"}
    proc = run_lib(tmp_path, "_upgrade_rollback 17 2026-08-18-10-30-00", extra_env=env)
    assert proc.returncode == 0, proc.stderr  # trap-safe: never fail/exit
    assert "could not stop the managed Postgres" in proc.stderr
    # Nothing was moved: both dirs are intact and unchanged.
    assert (tmp_path / "pgdata" / "marker").read_text() == "PARTIAL"
    assert (retained / "PG_VERSION").read_text() == "17\n"
    assert not list(tmp_path.glob("pgdata.failed-*"))


def test_manual_rollback_refuses_multiple_retained(tmp_path: Path) -> None:
    (tmp_path / "pgdata.pg17-2026-08-18-11-00-00").mkdir()
    (tmp_path / "pgdata.pg17-2026-08-18-12-00-00").mkdir()
    proc = _upg_run(tmp_path, "--rollback", extra_env=_down_stack_env(tmp_path))
    assert proc.returncode != 0
    assert "found 2 retained clusters" in proc.stderr


def test_manual_rollback_no_retained_is_noop(tmp_path: Path) -> None:
    proc = _upg_run(tmp_path, "--rollback", extra_env=_down_stack_env(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "nothing to roll back" in proc.stderr


def test_manual_rollback_refuses_when_stack_up(tmp_path: Path) -> None:
    # A manual rollback swaps the data directory, so it must refuse while a core
    # service is still supervised (guards against app writes during the swap).
    (tmp_path / "pgdata.pg17-2026-08-18-11-00-00").mkdir()
    bindir = _make_stub_bindir(tmp_path, running_labels=("com.voxint.native.api",))
    proc = _upg_run(tmp_path, "--rollback", extra_env=_stub_env(bindir))
    assert proc.returncode != 0
    assert "core services still running" in proc.stderr


# --- cmd_up interrupted-upgrade guard -------------------------------------- #
def test_up_refuses_interrupted_upgrade_shape(tmp_path: Path) -> None:
    # No live pgdata, but a retained old cluster from a half-done upgrade: `up`
    # must refuse and point at --rollback rather than silently using operator DBs.
    (tmp_path / "pgdata.pg17-2026-08-18-11-00-00").mkdir()
    venvbin = tmp_path / "venv" / "bin"
    venvbin.mkdir(parents=True)
    (venvbin / "voxint").write_text("#!/usr/bin/env bash\nexit 0\n")
    (venvbin / "voxint").chmod(0o755)
    proc = run_lib(tmp_path, "cmd_up", extra_env={"VOXINT_NATIVE_WITH_MODELS": "0"})
    assert proc.returncode != 0
    assert "an upgrade did not finish" in proc.stderr
    assert "upgrade-db --rollback" in proc.stderr


# --- full orchestration over stubs ----------------------------------------- #
def _make_upgrade_bindir(
    bindir: Path,
    major: str,
    logdir: Path,
    *,
    dump_rc: int = 0,
    restore_rc: int = 0,
    toc: tuple[str, ...] = _STUB_TOC_VOXINT,
    inv_db: str = "",
    inv_ext: str = "",
) -> Path:
    """A stub Postgres bindir for the upgrade orchestration.

    ``postgres --version`` reports ``major``; pg_ctl start/stop toggle a
    postmaster.pid under -D and log to pg_ctl.log; initdb writes PG_VERSION
    (``major``) into the new -D and logs; pg_dump/pg_restore/createdb/psql are
    stubbed and log to ``logdir``. pg_isready + the hermetic launchctl share a
    ``pg_up`` marker so the datastore reads DOWN before bootstrap and UP after,
    letting require_stack_fully_down pass and wait_for_postgres succeed.
    ``inv_db``/``inv_ext`` drive the source-inventory gate.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    up = logdir / "pg_up"

    (bindir / "postgres").write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--version" ]; then printf "postgres (PostgreSQL) %s\\n" '
        f"{shlex.quote(major)}; exit 0; fi\nexit 0\n"
    )
    (bindir / "pg_ctl").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/pg_ctl.log"\n'
        "datadir=\"\"; action=\"\"\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        "    -D) shift; datadir=$1 ;;\n"
        "    start|stop|restart|status) action=$1 ;;\n"
        "  esac\n"
        "  shift\n"
        "done\n"
        'printf "%s %s\\n" "$action" "$datadir" >> "$LOG"\n'
        'case "$action" in\n'
        '  start) : > "$datadir/postmaster.pid" ;;\n'
        '  stop)  rm -f "$datadir/postmaster.pid" ;;\n'
        "esac\n"
        "exit 0\n"
    )
    (bindir / "initdb").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/initdb.log"\n'
        "datadir=\"\"\n"
        'while [ $# -gt 0 ]; do case "$1" in -D) shift; datadir=$1 ;; esac; shift; done\n'
        'printf "initdb %s\\n" "$datadir" >> "$LOG"\n'
        'mkdir -p "$datadir"\n'
        f'printf "%s\\n" {shlex.quote(major)} > "$datadir/PG_VERSION"\n'
        "exit 0\n"
    )
    (bindir / "pg_isready").write_text(
        "#!/usr/bin/env bash\n"
        f'[ -f "{up}" ] && exit 0\nexit 1\n'
    )
    # The verify query is `current_setting('server_version_num')::int / 10000`, so
    # the RESULT the stub returns is the bare major, not the raw setting.
    svnum = major
    (bindir / "psql").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/psql.log"\n'
        f'MARK="{logdir}/dropped"\n'
        'sql=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -tAc|-c) shift; sql=$1 ;; esac\n'
        "  shift\n"
        "done\n"
        'printf "%s\\n" "$sql" >> "$LOG"\n'
        'case "$sql" in\n'
        f'  *server_version_num*) echo {svnum}; exit 0 ;;\n'
        '  *"<->"*) echo 1.414; exit 0 ;;\n'
        f'  *datistemplate*) printf "%s" {shlex.quote(inv_db)}; echo; exit 0 ;;\n'
        f'  *pg_extension*) printf "%s" {shlex.quote(inv_ext)}; echo; exit 0 ;;\n'
        '  *data_directory*) echo "$VOXINT_NATIVE_HOME/pgdata"; exit 0 ;;\n'
        '  *"DROP DATABASE"*) : > "$MARK"; exit 0 ;;\n'
        '  *"count(*)"*) echo 0; exit 0 ;;\n'
        '  *information_schema*) echo 1; exit 0 ;;\n'
        '  *"oid FROM pg_database"*) [ -f "$MARK" ] && echo 20000 || echo 16386; exit 0 ;;\n'
        '  *"SELECT 1 FROM pg_database"*) exit 0 ;;\n'
        '  *pg_roles*) exit 0 ;;\n'
        "esac\n"
        "exit 0\n"
    )
    toc_args = " ".join(f'"{line}"' for line in toc)
    (bindir / "pg_restore").write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{logdir}/pg_restore.log"\n'
        'if [ "$1" = "--list" ]; then\n'
        f'  printf "%s\\n" {toc_args}\n'
        "  exit 0\n"
        "fi\n"
        'printf "%s\\n" "$*" >> "$LOG"\n'
        f"exit {restore_rc}\n"
    )
    (bindir / "pg_dump").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{logdir}/pg_dump.log"\n'
        f"[ {dump_rc} -eq 0 ] || exit {dump_rc}\n"
        'printf "PGDMP-stub\\n"\n'
        "exit 0\n"
    )
    (bindir / "createdb").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{logdir}/createdb.log"\nexit 0\n'
    )
    (bindir / "launchctl").write_text(
        "#!/usr/bin/env bash\n"
        f'UP="{up}"\n'
        'cmd=$1\n'
        'case "$cmd" in\n'
        '  print) exit 1 ;;\n'  # nothing supervised in these hermetic tests
        f'  bootstrap) : > "$UP"; exit 0 ;;\n'  # new cluster comes UP
        f'  bootout) rm -f "$UP"; exit 0 ;;\n'  # maintenance PG goes down
        "esac\n"
        "exit 0\n"
    )
    for name in (
        "postgres", "pg_ctl", "initdb", "pg_isready", "psql",
        "pg_restore", "pg_dump", "createdb", "launchctl",
    ):
        (bindir / name).chmod(0o755)
    return bindir


def _prime_upgrade(tmp_path: Path, cluster_major: str = "17") -> tuple[Path, Path, dict]:
    """A primed install for the upgrade orchestration: a managed cluster at
    ``cluster_major``, a NEW-major bindir (18) as NATIVE_PG_BINDIR, and an OLD-major
    bindir (``cluster_major``) as the explicit VOXINT_NATIVE_OLD_PG_BINDIR override
    (so the test never depends on a real brew keg). Returns (newbin, logdir, env)."""
    pgdata = tmp_path / "pgdata"
    pgdata.mkdir()
    (pgdata / "PG_VERSION").write_text(f"{cluster_major}\n")
    venvbin = tmp_path / "venv" / "bin"
    venvbin.mkdir(parents=True)
    for tool in ("alembic",):
        (venvbin / tool).write_text("#!/usr/bin/env bash\nexit 0\n")
        (venvbin / tool).chmod(0o755)
    (venvbin / "voxint").write_text("#!/usr/bin/env bash\nexit 0\n")
    (venvbin / "voxint").chmod(0o755)
    logdir = tmp_path / "stublog"
    logdir.mkdir()
    newbin = _make_upgrade_bindir(tmp_path / "newbin", "18", logdir)
    oldbin = _make_upgrade_bindir(tmp_path / "oldbin", cluster_major, logdir)
    env = {
        "VOXINT_NATIVE_PG_BINDIR": str(newbin),
        "VOXINT_NATIVE_OLD_PG_BINDIR": str(oldbin),
        "VOXINT_NATIVE_WITH_MODELS": "0",
        "PATH": f"{newbin}:{os.environ.get('PATH', '')}",
    }
    return newbin, logdir, env


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_upgrade_happy_path(tmp_path: Path) -> None:
    # The full happy path brings the new cluster up, which generates + plutil-lints
    # the launchd plists -- a macOS-only step, so this end-to-end rung is macOS-gated
    # (the version-gate / arg-parse / rollback tests above stay portable and run on
    # Linux CI). See the require_macos VOXINT_NATIVE_LIB seam.
    _newbin, logdir, env = _prime_upgrade(tmp_path)
    proc = _upg_run(tmp_path, "", extra_env=env)
    assert proc.returncode == 0, proc.stderr
    assert "Upgrade complete: Postgres 17 -> 18" in proc.stderr
    # The old cluster is retained as a rollback; a NEW cluster is live at pgdata.
    retained = list(tmp_path.glob("pgdata.pg17-*"))
    assert len(retained) == 1, f"expected one retained cluster, got {retained}"
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "18\n"  # initdb NEW
    # The dump was taken with the NEW pg_dump and the cross-version-safe flags.
    dump_log = (logdir / "pg_dump.log").read_text()
    assert "--exclude-extension=vector" in dump_log
    assert "--quote-all-identifiers" in dump_log
    # A pre-upgrade dump landed in backups/ (finalized, not left as .partial).
    dumps = list((tmp_path / "backups").glob("voxint-pre-upgrade-17-to-18-*.dump"))
    assert len(dumps) == 1
    assert not list((tmp_path / "backups").glob("*.partial"))


def test_upgrade_no_mv_if_dump_fails(tmp_path: Path) -> None:
    # If pg_dump fails, NOTHING destructive happens: pgdata stays put, no retained
    # dir is created, and the old server is stopped (pre-swap cleanup ran).
    newbin, logdir, env = _prime_upgrade(tmp_path)
    # Rebuild the NEW bindir with a failing pg_dump.
    _make_upgrade_bindir(newbin, "18", logdir, dump_rc=1)
    proc = _upg_run(tmp_path, "", extra_env=env)
    assert proc.returncode != 0
    assert "pg_dump failed" in proc.stderr
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "17\n"  # untouched
    assert not list(tmp_path.glob("pgdata.pg17-*"))  # never renamed
    assert not list((tmp_path / "backups").glob("*.partial"))  # partial cleaned up


def test_upgrade_rolls_back_when_restore_fails(tmp_path: Path) -> None:
    # A failure AFTER the cutover (fresh_restore's pg_restore errors) must trigger
    # the rollback trap: the retained old cluster is restored to pgdata and the
    # partial new one preserved as pgdata.failed-*. This also proves fresh_restore's
    # subshell trap-clear did not erase the upgrade's own rollback trap.
    newbin, logdir, env = _prime_upgrade(tmp_path)
    _make_upgrade_bindir(newbin, "18", logdir, restore_rc=1)
    proc = _upg_run(tmp_path, "", extra_env=env)
    assert proc.returncode != 0
    # Rolled back: pgdata is the OLD cluster again (v17), retained dir consumed.
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "17\n"
    assert not list(tmp_path.glob("pgdata.pg17-*"))
    # The partial new cluster was preserved, never deleted.
    assert list(tmp_path.glob("pgdata.failed-*"))


def test_upgrade_source_inventory_refuses_extra_db(tmp_path: Path) -> None:
    # An extra non-template database a single-db dump cannot carry must be refused
    # BEFORE any dump or rename.
    _newbin, logdir, env = _prime_upgrade(tmp_path)
    oldbin = tmp_path / "oldbin"
    _make_upgrade_bindir(oldbin, "17", logdir, inv_db="analytics")
    proc = _upg_run(tmp_path, "", extra_env=env)
    assert proc.returncode != 0
    assert "extra databases" in proc.stderr
    assert "analytics" in proc.stderr
    assert (tmp_path / "pgdata" / "PG_VERSION").read_text() == "17\n"  # untouched
    assert not list(tmp_path.glob("pgdata.pg17-*"))


def test_upgrade_source_inventory_refuses_extra_extension(tmp_path: Path) -> None:
    _newbin, logdir, env = _prime_upgrade(tmp_path)
    oldbin = tmp_path / "oldbin"
    _make_upgrade_bindir(oldbin, "17", logdir, inv_ext="postgis")
    proc = _upg_run(tmp_path, "", extra_env=env)
    assert proc.returncode != 0
    assert "unexpected extensions" in proc.stderr
    assert "postgis" in proc.stderr
    assert not list(tmp_path.glob("pgdata.pg17-*"))


# --------------------------------------------------------------------------- #
# #6 (wiring) — cmd_status must surface a crash-looping worker/beat rather than
# printing a bare [supervised]. Uses the hermetic launchctl stub's optional
# per-label print body; unused datastore ports keep the run hermetic and fast.
# --------------------------------------------------------------------------- #
def _status_env(bindir: Path) -> dict[str, str]:
    env = _stub_env(bindir)
    env.update(
        {
            "VOXINT_NATIVE_WITH_MODELS": "0",  # skip metal delegation
            "VOXINT_NATIVE_PG_PORT": "59991",  # unused -> "not reachable"
            "VOXINT_NATIVE_REDIS_PORT": "59992",
            "VOXINT_NATIVE_API_PORT": "59993",  # unused -> healthz "unreachable"
        }
    )
    return env


def test_status_surfaces_crashloop_worker_and_healthy_beat(tmp_path: Path) -> None:
    crash = "\tstate = waiting\n\tlast exit code = 1\n"
    bindir = _make_stub_bindir(
        tmp_path,
        running_labels=(
            "com.voxint.native.api",
            "com.voxint.native.worker",
            "com.voxint.native.beat",
        ),
        job_blocks={"com.voxint.native.worker": crash},  # beat -> default healthy
    )
    proc = run_lib(tmp_path, "cmd_status", extra_env=_status_env(bindir))
    assert proc.returncode == 0, proc.stderr
    worker_line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("worker")
    )
    beat_line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("beat"))
    # The crash-looping worker is no longer a bare healthy [supervised].
    assert "restarting (last exit 1)" in worker_line
    assert "[supervised]" in worker_line
    # A genuinely running beat reads running.
    assert beat_line.rstrip().endswith("running")


def test_status_not_loaded_worker_has_no_liveness_suffix(tmp_path: Path) -> None:
    # A NOT-loaded worker keeps the plain two-column form (no launchd body to read).
    bindir = _make_stub_bindir(tmp_path, running_labels=())
    proc = run_lib(tmp_path, "cmd_status", extra_env=_status_env(bindir))
    assert proc.returncode == 0, proc.stderr
    worker_line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("worker")
    )
    assert "[NOT loaded]" in worker_line
    assert "restarting" not in worker_line
    assert "running" not in worker_line


# --------------------------------------------------------------------------- #
# #10 — doctor foreign-postmaster detection. When a managed cluster exists and
# Postgres is reachable, doctor must prove the reachable postmaster is OURS
# (SHOW data_directory == $NATIVE_PGDATA), aggregating the verdict through
# doctor_report (never fail/exit, so later checks still run).
# --------------------------------------------------------------------------- #
def _make_doctor_identity_bindir(
    bindir: Path, *, datadir_echo: str, datadir_rc: int = 0
) -> Path:
    """A doctor bindir: postgres v17, pg_isready reachable, and a psql that
    answers SHOW data_directory with ``datadir_echo`` (or exits ``datadir_rc``)."""
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "postgres").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then\n'
        "  printf '%s\\n' 'postgres (PostgreSQL) 17.5'\n"
        "fi\n"
        "exit 0\n"
    )
    (bindir / "pg_isready").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bindir / "psql").write_text(
        "#!/usr/bin/env bash\n"
        'sql=""\n'
        'while [ $# -gt 0 ]; do case "$1" in -tAc) shift; sql=$1 ;; esac; shift; done\n'
        'case "$sql" in\n'
        f"  *data_directory*) [ {datadir_rc} -eq 0 ] || exit {datadir_rc}; "
        f"printf '%s\\n' {shlex.quote(datadir_echo)}; exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    for name in ("postgres", "pg_isready", "psql"):
        (bindir / name).chmod(0o755)
    return bindir


def _identity_doctor_env(bindir: Path) -> dict[str, str]:
    # Reachable Postgres (real pg_isready in the stub bindir), no metal, unused
    # Redis port so redis reads a clean SKIP.
    return {
        "VOXINT_NATIVE_PG_BINDIR": str(bindir),
        "VOXINT_NATIVE_WITH_MODELS": "0",
        "VOXINT_NATIVE_REDIS_PORT": "59998",
    }


def test_doctor_confirms_managed_cluster_identity(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_doctor_identity_bindir(
        tmp_path / "b", datadir_echo=str(tmp_path / "pgdata")
    )
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=_identity_doctor_env(bindir))
    assert "[PASS] the postmaster on :5432 is the managed cluster" in proc.stderr
    assert "NOT the managed cluster" not in proc.stderr


def test_doctor_flags_foreign_postmaster(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_doctor_identity_bindir(
        tmp_path / "b", datadir_echo="/opt/homebrew/var/postgresql@17"
    )
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=_identity_doctor_env(bindir))
    assert "[FAIL] the postmaster on :5432 is NOT the managed cluster" in proc.stderr
    assert "/opt/homebrew/var/postgresql@17" in proc.stderr
    assert proc.returncode != 0  # doctor_report FAIL set DOCTOR_RC


def test_doctor_flags_unqueryable_data_directory(tmp_path: Path) -> None:
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_doctor_identity_bindir(
        tmp_path / "b", datadir_echo="", datadir_rc=2
    )
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=_identity_doctor_env(bindir))
    assert "[FAIL] could not query data_directory" in proc.stderr
    assert proc.returncode != 0


def test_doctor_identity_tolerates_noncanonical_home(tmp_path: Path) -> None:
    # A non-canonical VOXINT_NATIVE_HOME (here a trailing slash) makes
    # NATIVE_PGDATA=".../home//pgdata", while Postgres normalizes what SHOW
    # data_directory returns (".../home/pgdata"). A literal compare would
    # mislabel the correct cluster as foreign; canonical_dir must prevent that.
    _write_pg_version(tmp_path, "17\n")
    bindir = _make_doctor_identity_bindir(
        tmp_path / "b", datadir_echo=str(tmp_path / "pgdata")  # canonical form
    )
    env = _identity_doctor_env(bindir)
    env["VOXINT_NATIVE_HOME"] = f"{tmp_path}/"  # trailing slash -> doubled sep
    proc = run_lib(tmp_path, "cmd_doctor", extra_env=env)
    assert "[PASS] the postmaster on :5432 is the managed cluster" in proc.stderr
    assert "NOT the managed cluster" not in proc.stderr
