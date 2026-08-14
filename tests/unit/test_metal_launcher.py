"""Offline tests for scripts/metal/voxint-metal.sh via its library-mode seam.

The launcher sources with ``VOXINT_METAL_LIB=1`` (no main), which lets these
tests exercise the pure-shell logic — per-service env assembly, launchd plist
generation, provenance sha verification, vendored-config generation, and
``pwd -P`` MEDIA_ROOT resolution — without network, launchd, or a Docker
daemon. The functions under test never write into the repo, so the script is
sourced in place; ``VOXINT_METAL_HOME`` points every test at a throwaway tmp
tree, and ``VOXINT_METAL_PYTHON`` pins the JSON/YAML verification python to
this test interpreter (the real flow uses the pyannote venv's).
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REAL_REPO = Path(__file__).resolve().parents[2]
METAL_SCRIPT = REAL_REPO / "scripts" / "metal" / "voxint-metal.sh"
PYANNOTE_PROVENANCE = REAL_REPO / "services" / "pyannote" / "models" / "provenance.json"
VENDORED_CONFIG = REAL_REPO / "services" / "pyannote" / "models" / "config.vendored.yaml"


def run_lib(
    home: Path,
    script: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VOXINT_METAL_LIB"] = "1"
    env["VOXINT_METAL_HOME"] = str(home)
    env["VOXINT_METAL_PYTHON"] = sys.executable
    env.pop("TITANET_ORT_PROVIDERS", None)
    env.pop("VOXINT_METAL_DIARIZER_DEVICE", None)
    if extra_env:
        env.update(extra_env)
    full = f'source "{METAL_SCRIPT}"\n{script}'
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


# --------------------------------------------------------------------------- #
# Static per-service rows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("svc", "port"), [("whisper", "8022"), ("pyannote", "8024"), ("titanet", "8021")]
)
def test_service_ports(tmp_path: Path, svc: str, port: str) -> None:
    proc = run_lib(tmp_path, f"service_port {svc}")
    assert proc.returncode == 0
    assert proc.stdout == port


def test_compose_overlay_ports_bind_to_service_ports(tmp_path: Path) -> None:
    # compose.metal.yaml points api/worker at host.docker.internal:<port> and
    # the native services bind whatever service_port() assigns — two files,
    # one invariant. The overlay contract test and test_service_ports each pin
    # their own side to literals, so a port moved in only one place would keep
    # both green while the worker calls a dead port; this binds them directly
    # (metal review follow-up).
    import yaml

    doc = yaml.safe_load((REAL_REPO / "compose.metal.yaml").read_text())
    url_to_service = {
        "ASR_URL": "whisper",
        "DIARIZER_URL": "pyannote",
        "EMBEDDER_URL": "titanet",
    }
    for caller in ("api", "worker"):
        env = doc["services"][caller]["environment"]
        for key, svc in url_to_service.items():
            compose_port = env[key].rsplit(":", 1)[1]
            proc = run_lib(tmp_path, f"service_port {svc}")
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout == compose_port, (
                f"{caller} {key} targets port {compose_port} but "
                f"service_port {svc} binds {proc.stdout}"
            )


def test_titanet_reuses_the_parity_measured_requirements(tmp_path: Path) -> None:
    # The committed ONNX parity verdict binds to exactly the CPU image's
    # dependency chain — the metal venv must not get its own flavor file.
    proc = run_lib(tmp_path, "service_requirements titanet")
    assert proc.stdout == "services/titanet/requirements.cpu.txt"
    for svc in ("whisper", "pyannote"):
        proc = run_lib(tmp_path, f"service_requirements {svc}")
        assert proc.stdout == f"services/{svc}/requirements.metal.txt"


def test_unknown_service_is_an_error(tmp_path: Path) -> None:
    assert run_lib(tmp_path, "service_port nemo").returncode != 0
    assert run_lib(tmp_path, "service_requirements nemo").returncode != 0


# --------------------------------------------------------------------------- #
# Env assembly — one source of truth for plists AND `run --foreground`
# --------------------------------------------------------------------------- #
def test_whisper_env_is_cpu_int8_with_pinned_cache(tmp_path: Path) -> None:
    env = env_lines(run_lib(tmp_path, "service_env whisper /media/root"))
    assert env["MEDIA_ROOT"] == "/media/root"
    assert env["DEVICE"] == "cpu"  # v1: CT2 on CPU, Metal ASR is a follow-up
    assert env["WHISPER_MODEL"] == "large-v2"
    assert env["COMPUTE_TYPE"] == "int8"
    assert env["WHISPER_DOWNLOAD_ROOT"] == f"{tmp_path}/models/whisper"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_pyannote_env_forces_mps_and_vendored_pipeline(tmp_path: Path) -> None:
    env = env_lines(run_lib(tmp_path, "service_env pyannote /media/root"))
    assert env["DIARIZER_DEVICE"] == "mps"  # forced: no silent CPU degradation
    assert (
        env["VOXINT_VENDORED_PIPELINE"]
        == f"{tmp_path}/models/pyannote/vendored/config.yaml"
    )


def test_pyannote_device_override_for_ab_measurement(tmp_path: Path) -> None:
    env = env_lines(
        run_lib(
            tmp_path,
            "service_env pyannote /media/root",
            extra_env={"VOXINT_METAL_DIARIZER_DEVICE": "cpu"},
        )
    )
    assert env["DIARIZER_DEVICE"] == "cpu"


def test_titanet_env_onnx_engine_no_providers_by_default(tmp_path: Path) -> None:
    env = env_lines(run_lib(tmp_path, "service_env titanet /media/root"))
    assert env["EMBED_ENGINE"] == "onnx"
    assert env["TITANET_ONNX_PATH"] == f"{tmp_path}/models/titanet/titanet-large.onnx"
    # Unset means the CPU EP default the parity verdict measured — the plist
    # must not carry an empty override.
    assert "TITANET_ORT_PROVIDERS" not in env


def test_titanet_env_threads_explicit_providers(tmp_path: Path) -> None:
    env = env_lines(
        run_lib(
            tmp_path,
            "service_env titanet /media/root",
            extra_env={"TITANET_ORT_PROVIDERS": "CoreMLExecutionProvider"},
        )
    )
    assert env["TITANET_ORT_PROVIDERS"] == "CoreMLExecutionProvider"


# --------------------------------------------------------------------------- #
# launchd plist generation
# --------------------------------------------------------------------------- #
def render(tmp_path: Path, svc: str, media: str = "/media/root") -> dict:
    out = tmp_path / "out.plist"
    proc = run_lib(tmp_path, f'render_plist {svc} "{media}" "{out}"')
    assert proc.returncode == 0, proc.stderr
    with out.open("rb") as fh:
        return plistlib.load(fh)


def test_plist_program_binds_loopback_and_right_port(tmp_path: Path) -> None:
    plist = render(tmp_path, "pyannote")
    assert plist["Label"] == "com.voxint.metal.pyannote"
    args = plist["ProgramArguments"]
    assert args[0] == f"{tmp_path}/venvs/pyannote/bin/python"
    assert args[1:4] == ["-m", "uvicorn", "app.main:app"]
    assert args[-4:] == ["--host", "127.0.0.1", "--port", "8024"]
    assert plist["WorkingDirectory"] == str(REAL_REPO / "services" / "pyannote")


def test_plist_env_matches_service_env_exactly(tmp_path: Path) -> None:
    # launchd inherits no shell environment; the dict must carry everything
    # service_env assembles, with no drift between the two code paths.
    plist = render(tmp_path, "titanet")
    expected = env_lines(run_lib(tmp_path, "service_env titanet /media/root"))
    assert plist["EnvironmentVariables"] == expected


def test_plist_supervision_matches_restart_doctrine(tmp_path: Path) -> None:
    # KeepAlive/SuccessfulExit=false == `restart: unless-stopped`: crashes
    # restart, clean exits and bootout stay down.
    plist = render(tmp_path, "whisper")
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["StandardOutPath"] == f"{tmp_path}/logs/whisper.log"
    assert plist["StandardErrorPath"] == f"{tmp_path}/logs/whisper.log"


def test_plist_xml_escapes_hostile_paths(tmp_path: Path) -> None:
    plist = render(tmp_path, "whisper", media="/media/a&b<c>d")
    assert plist["EnvironmentVariables"]["MEDIA_ROOT"] == "/media/a&b<c>d"


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_plist_passes_plutil_lint(tmp_path: Path) -> None:
    out = tmp_path / "out.plist"
    proc = run_lib(tmp_path, f'render_plist pyannote /media/root "{out}"')
    assert proc.returncode == 0, proc.stderr
    lint = subprocess.run(
        ["plutil", "-lint", "-s", str(out)], capture_output=True, text=True
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr


# --------------------------------------------------------------------------- #
# Provenance sha verification
# --------------------------------------------------------------------------- #
def _fake_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    files = {}
    for name in ("segmentation-3.0.bin", "wespeaker-voxceleb-resnet34-LM.bin"):
        payload = f"fake weights for {name}".encode()
        (ckpt_dir / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest()}
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"files": files}))
    return ckpt_dir, prov


def test_pyannote_sha_verify_passes_on_match(tmp_path: Path) -> None:
    ckpt_dir, prov = _fake_checkpoints(tmp_path)
    proc = run_lib(tmp_path, f'verify_pyannote_checkpoints "{ckpt_dir}" "{prov}"')
    assert proc.returncode == 0, proc.stderr


def test_pyannote_sha_verify_fails_on_tamper(tmp_path: Path) -> None:
    ckpt_dir, prov = _fake_checkpoints(tmp_path)
    (ckpt_dir / "segmentation-3.0.bin").write_bytes(b"tampered")
    proc = run_lib(tmp_path, f'verify_pyannote_checkpoints "{ckpt_dir}" "{prov}"')
    assert proc.returncode != 0
    assert "sha256 mismatch" in proc.stderr


def test_pyannote_sha_verify_fails_on_missing_file(tmp_path: Path) -> None:
    ckpt_dir, prov = _fake_checkpoints(tmp_path)
    (ckpt_dir / "wespeaker-voxceleb-resnet34-LM.bin").unlink()
    proc = run_lib(tmp_path, f'verify_pyannote_checkpoints "{ckpt_dir}" "{prov}"')
    assert proc.returncode != 0
    assert "missing checkpoint" in proc.stderr


def test_titanet_sha_verify_round_trip(tmp_path: Path) -> None:
    onnx = tmp_path / "titanet-large.onnx"
    onnx.write_bytes(b"fake onnx graph")
    prov = tmp_path / "provenance.json"
    prov.write_text(
        json.dumps({"onnx_sha256": hashlib.sha256(b"fake onnx graph").hexdigest()})
    )
    ok = run_lib(tmp_path, f'verify_titanet_onnx "{onnx}" "{prov}"')
    assert ok.returncode == 0, ok.stderr
    onnx.write_bytes(b"tampered graph")
    bad = run_lib(tmp_path, f'verify_titanet_onnx "{onnx}" "{prov}"')
    assert bad.returncode != 0
    assert "sha256 mismatch" in bad.stderr


# --------------------------------------------------------------------------- #
# Vendored config generation — against the REAL committed config + provenance
# --------------------------------------------------------------------------- #
def _vendored_dest(tmp_path: Path) -> Path:
    dest = tmp_path / "models" / "pyannote" / "vendored"
    (dest / "pyannote").mkdir(parents=True)
    for name in ("segmentation-3.0.bin", "wespeaker-voxceleb-resnet34-LM.bin"):
        (dest / "pyannote" / name).write_bytes(b"placeholder")
    return dest


def test_generated_config_repoints_paths_and_passes_param_check(
    tmp_path: Path,
) -> None:
    dest = _vendored_dest(tmp_path)
    proc = run_lib(
        tmp_path,
        f'generate_vendored_config "{VENDORED_CONFIG}" "{dest}" "{PYANNOTE_PROVENANCE}"',
    )
    assert proc.returncode == 0, proc.stderr
    text = (dest / "config.yaml").read_text()
    assert "/app/vendored/" not in text, "container paths survived the rewrite"
    assert f"{dest}/pyannote/wespeaker-voxceleb-resnet34-LM.bin" in text
    assert f"{dest}/pyannote/segmentation-3.0.bin" in text
    # The knife-edge clustering threshold must survive verbatim.
    assert "0.7045654963945799" in text


def test_generated_config_rejects_param_drift(tmp_path: Path) -> None:
    # A tampered source config (here: the clustering threshold) must fail the
    # provenance param check, not silently ship different numerics.
    dest = _vendored_dest(tmp_path)
    tampered = tmp_path / "config.tampered.yaml"
    tampered.write_text(
        VENDORED_CONFIG.read_text().replace("0.7045654963945799", "0.65")
    )
    proc = run_lib(
        tmp_path,
        f'generate_vendored_config "{tampered}" "{dest}" "{PYANNOTE_PROVENANCE}"',
    )
    assert proc.returncode != 0
    assert "clustering_threshold" in proc.stderr


def test_generated_config_requires_checkpoints_present(tmp_path: Path) -> None:
    dest = _vendored_dest(tmp_path)
    (dest / "pyannote" / "segmentation-3.0.bin").unlink()
    proc = run_lib(
        tmp_path,
        f'generate_vendored_config "{VENDORED_CONFIG}" "{dest}" "{PYANNOTE_PROVENANCE}"',
    )
    assert proc.returncode != 0


def test_pipe_in_metal_home_is_refused(tmp_path: Path) -> None:
    dest = tmp_path / "with|pipe"
    (dest / "pyannote").mkdir(parents=True)
    proc = run_lib(
        tmp_path,
        f'generate_vendored_config "{VENDORED_CONFIG}" "{dest}" "{PYANNOTE_PROVENANCE}"',
    )
    assert proc.returncode != 0
    assert "must not contain" in proc.stderr


def test_ampersand_in_metal_home_survives_the_rewrite(tmp_path: Path) -> None:
    # sed replacement text expands '&' to the matched string — a legal path
    # like "Ben & Co" must come through verbatim, not corrupted.
    dest = tmp_path / "Ben & Co" / "vendored"
    (dest / "pyannote").mkdir(parents=True)
    for name in ("segmentation-3.0.bin", "wespeaker-voxceleb-resnet34-LM.bin"):
        (dest / "pyannote" / name).write_bytes(b"placeholder")
    proc = run_lib(
        tmp_path,
        f'generate_vendored_config "{VENDORED_CONFIG}" "{dest}" "{PYANNOTE_PROVENANCE}"',
    )
    assert proc.returncode == 0, proc.stderr
    text = (dest / "config.yaml").read_text()
    assert f"{dest}/pyannote/segmentation-3.0.bin" in text
    assert "/app/vendored/" not in text


# --------------------------------------------------------------------------- #
# MEDIA_ROOT resolution — pwd -P physical paths
# --------------------------------------------------------------------------- #
def test_media_root_resolves_symlinks_physically(tmp_path: Path) -> None:
    real = tmp_path / "real-media"
    real.mkdir()
    link = tmp_path / "media-link"
    link.symlink_to(real)
    proc = run_lib(tmp_path, f'resolve_media_root "{link}"')
    assert proc.returncode == 0
    # Physical resolution: the symlink itself AND tmp-dir prefix symlinks
    # (macOS /tmp -> /private/tmp) are gone.
    assert proc.stdout.rstrip("\n") == str(real.resolve())


def test_media_root_relative_resolves_against_repo_root(tmp_path: Path) -> None:
    (tmp_path / "repo-media").mkdir()
    # REPO_ROOT is a plain global in the sourced script — repoint it at tmp
    # to model a checkout whose .env says MEDIA_ROOT=./repo-media.
    proc = run_lib(
        tmp_path, f'REPO_ROOT="{tmp_path}"\nresolve_media_root "./repo-media"'
    )
    assert proc.returncode == 0
    assert proc.stdout.rstrip("\n") == str((tmp_path / "repo-media").resolve())


def test_media_root_missing_dir_fails(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, f'resolve_media_root "{tmp_path}/nope"')
    assert proc.returncode != 0


def test_env_file_parsing_last_assignment_wins(tmp_path: Path) -> None:
    envf = tmp_path / ".env"
    envf.write_text(
        "MEDIA_ROOT=/first\nVOXINT_IMAGE_TAG=0.7.0\nMEDIA_ROOT=/second\n"
    )
    proc = run_lib(tmp_path, f'media_root_from_env_file "{envf}"')
    assert proc.stdout.rstrip("\n") == "/second"
    proc = run_lib(tmp_path, f'image_tag_from_env_file "{envf}"')
    assert proc.stdout.rstrip("\n") == "0.7.0"


def test_whisper_revision_pin_matches_parity_lane() -> None:
    # The launcher downloads at WHISPER_HF_REVISION; the whisper parity lane
    # only accepts that exact snapshot. Textual pin-parity so they cannot
    # drift apart silently (parsed, not imported: the parity module skips
    # itself off-Mac at import time).
    import re

    script_pin = re.search(
        r"^WHISPER_HF_REVISION=([0-9a-f]{40})$", METAL_SCRIPT.read_text(), re.M
    )
    lane = (
        METAL_SCRIPT.parents[2] / "tests" / "parity" / "test_whisper_metal.py"
    ).read_text()
    lane_pin = re.search(r'^WHISPER_HF_REVISION = "([0-9a-f]{40})"$', lane, re.M)
    assert script_pin and lane_pin
    assert script_pin.group(1) == lane_pin.group(1)


def test_diarizer_override_rejects_auto(tmp_path: Path) -> None:
    # 'auto' would cascade to CPU when MPS is missing — silently, under
    # launchd supervision. Only the two honest values are allowed.
    for value, ok in (("mps", True), ("cpu", True), ("auto", False), ("gpu", False)):
        proc = run_lib(
            tmp_path,
            f'VOXINT_METAL_DIARIZER_DEVICE="{value}" validate_diarizer_override',
        )
        assert (proc.returncode == 0) is ok, (value, proc.stderr)
        if not ok:
            assert "silent CPU fallback" in proc.stderr


def test_whisper_manifest_round_trip_and_refusals(tmp_path: Path) -> None:
    mh = tmp_path / "metal-home"
    wdir = mh / "models" / "whisper"
    (wdir / "snapshots").mkdir(parents=True)
    (wdir / "snapshots" / "model.bin").write_bytes(b"weights")
    # Hub bookkeeping must be EXCLUDED from the manifest (it churns).
    (wdir / ".locks").mkdir()
    (wdir / ".locks" / "x.lock").write_bytes(b"")
    ok = run_lib(tmp_path, f'write_whisper_manifest "{mh}" && whisper_weights_ok "{mh}"')
    assert ok.returncode == 0, ok.stderr
    manifest = wdir / ".voxint-manifest.sha256"
    text = manifest.read_text()
    assert text.startswith("# revision: ")
    assert ".locks" not in text
    # Corruption fails the check.
    (wdir / "snapshots" / "model.bin").write_bytes(b"corrupted")
    assert run_lib(tmp_path, f'whisper_weights_ok "{mh}"').returncode != 0
    (wdir / "snapshots" / "model.bin").write_bytes(b"weights")
    # An EMPTY manifest must refuse, never bless whatever is on disk.
    manifest.write_text("")
    assert run_lib(tmp_path, f'whisper_weights_ok "{mh}"').returncode != 0
    # A manifest recorded under a different pin must refuse.
    run_lib(tmp_path, f'write_whisper_manifest "{mh}"')
    stale = manifest.read_text().replace("# revision: ", "# revision: 0000dead")
    manifest.write_text(stale)
    assert run_lib(tmp_path, f'whisper_weights_ok "{mh}"').returncode != 0


def test_env_file_parsing_strips_installer_quoting(tmp_path: Path) -> None:
    # The installer writes MEDIA_ROOT single-quoted (dotenv_squote); Compose
    # interpolation strips one matched pair of quotes, so the launcher must
    # read the same unquoted value or the two worlds disagree on the path.
    envf = tmp_path / ".env"
    envf.write_text(
        "MEDIA_ROOT='/with spaces/media'\nVOXINT_IMAGE_TAG=\"0.8.0\"\r\n"
    )
    proc = run_lib(tmp_path, f'media_root_from_env_file "{envf}"')
    assert proc.stdout.rstrip("\n") == "/with spaces/media"
    proc = run_lib(tmp_path, f'image_tag_from_env_file "{envf}"')
    assert proc.stdout.rstrip("\n") == "0.8.0"


# --------------------------------------------------------------------------- #
# Log rotation — copytruncate semantics (launchd holds the StandardOutPath fd)
# --------------------------------------------------------------------------- #
def test_rotate_log_below_threshold_is_untouched(tmp_path: Path) -> None:
    log = tmp_path / "whisper.log"
    log.write_bytes(b"small\n")
    proc = run_lib(tmp_path, f'rotate_log_file "{log}" 1 5')
    assert proc.returncode == 0, proc.stderr
    assert log.read_bytes() == b"small\n"
    assert list(tmp_path.glob("whisper_*.log")) == []


def test_rotate_log_missing_file_is_a_noop(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, f'rotate_log_file "{tmp_path}/absent.log" 1 5')
    assert proc.returncode == 0, proc.stderr


def test_rotate_log_archives_and_truncates_in_place(tmp_path: Path) -> None:
    # copytruncate: the archive gets the bytes, the LIVE file keeps its inode
    # (launchd's fd) and ends up empty — an mv-style rotation would leave the
    # running service writing into the archive until restart.
    log = tmp_path / "whisper.log"
    payload = b"x" * (1024 * 1024 + 1)
    log.write_bytes(payload)
    inode_before = log.stat().st_ino
    proc = run_lib(tmp_path, f'rotate_log_file "{log}" 1 5')
    assert proc.returncode == 0, proc.stderr
    assert log.stat().st_size == 0
    assert log.stat().st_ino == inode_before, "rotation replaced the live inode"
    archives = list(tmp_path.glob("whisper_*.log"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == payload


def test_prune_keeps_only_newest_archives(tmp_path: Path) -> None:
    log = tmp_path / "pyannote.log"
    log.write_bytes(b"live")
    stamps = [f"2026-08-{day:02d}-03-17-00" for day in (10, 11, 12, 13)]
    for stamp in stamps:
        (tmp_path / f"pyannote_{stamp}.log").write_bytes(b"old")
    # An unrelated file matching neither name nor stamp shape must survive.
    (tmp_path / "pyannote_notes.log").write_bytes(b"keep")
    proc = run_lib(tmp_path, f'prune_log_archives "{log}" 2')
    assert proc.returncode == 0, proc.stderr
    remaining = sorted(p.name for p in tmp_path.glob("pyannote_*.log"))
    assert remaining == [
        f"pyannote_{stamps[-2]}.log",
        f"pyannote_{stamps[-1]}.log",
        "pyannote_notes.log",
    ]
    assert log.read_bytes() == b"live"


def test_cmd_rotate_logs_covers_all_services_and_its_own_log(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    for name in ("whisper", "pyannote", "titanet", "logrotate"):
        (logs_dir / f"{name}.log").write_bytes(b"y" * (1024 * 1024 + 1))
    proc = run_lib(
        tmp_path, "cmd_rotate_logs", extra_env={"VOXINT_METAL_LOG_MAX_MB": "1"}
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("whisper", "pyannote", "titanet", "logrotate"):
        assert (logs_dir / f"{name}.log").stat().st_size == 0
        assert len(list(logs_dir.glob(f"{name}_*.log"))) == 1


def test_logrotate_plist_is_a_daily_oneshot(tmp_path: Path) -> None:
    out = tmp_path / "logrotate.plist"
    proc = run_lib(tmp_path, f'render_logrotate_plist "{out}"')
    assert proc.returncode == 0, proc.stderr
    with out.open("rb") as fh:
        plist = plistlib.load(fh)
    assert plist["Label"] == "com.voxint.metal.logrotate"
    args = plist["ProgramArguments"]
    assert args[0] == "/bin/bash"
    assert args[1].endswith("scripts/metal/voxint-metal.sh")
    assert args[2] == "rotate-logs"
    # launchd inherits no shell env — the job must carry its own home + knobs.
    env = plist["EnvironmentVariables"]
    assert env["VOXINT_METAL_HOME"] == str(tmp_path)
    assert env["VOXINT_METAL_LOG_MAX_MB"] == "50"
    assert env["VOXINT_METAL_LOG_ARCHIVES"] == "5"
    # One-shot daily: calendar-fired, never RunAtLoad, never KeepAlive (a
    # KeepAlive here would respawn the rotation in a loop).
    assert plist["RunAtLoad"] is False
    assert plist["StartCalendarInterval"] == {"Hour": 3, "Minute": 17}
    assert "KeepAlive" not in plist
    assert plist["StandardOutPath"] == f"{tmp_path}/logs/logrotate.log"


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_no_args_prints_usage_and_fails(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VOXINT_METAL_HOME"] = str(tmp_path)
    proc = subprocess.run(
        ["bash", str(METAL_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "usage:" in proc.stderr


def test_run_requires_foreground_flag(tmp_path: Path) -> None:
    proc = run_lib(tmp_path, "cmd_run whisper")
    assert proc.returncode != 0
    assert "--foreground" in proc.stderr
