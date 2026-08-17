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
