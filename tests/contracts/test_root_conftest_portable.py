"""The repo-root ``tests/conftest.py`` must load without the voxint app (#647).

The Metal parity lane (``.github/workflows/metal-lane.yml``) runs pytest from
each model service's own venv, which holds only the shipped service pins plus
pytest. Pytest loads the root conftest before any module under ``tests/``, so a
module-level voxint import there (direct or transitive) fails collection of
every parity test in that lane. It did from 2026-08-27 until #647, unnoticed,
because the lane only runs on a macOS schedule. These checks rebuild that
situation in a subprocess so the regression fails ``lint-test`` instead.

``sys.modules["voxint"] = None`` is the blocker: ``importlib.util.find_spec``
then returns ``None`` and any ``import voxint...`` raises
``ModuleNotFoundError``, which is what a venv without the package does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.contracts.conftest import REPO_ROOT

# The modules metal-lane.yml runs from the service venvs.
METAL_LANE_MODULES = (
    "tests/parity/test_whisper_metal.py",
    "tests/parity/test_pyannote_metal.py",
    "tests/parity/test_titanet_onnx.py",
)

BLOCK_VOXINT = "import sys\nsys.modules['voxint'] = None\n"

COLLECT = """
import json
import pytest

class _Record:
    def pytest_collection_finish(self, session):
        print("NODEIDS=" + json.dumps(sorted(item.nodeid for item in session.items)))

raise SystemExit(
    pytest.main([*MODULES, "--collect-only", "-p", "no:cacheprovider"], plugins=[_Record()])
)
"""

BREAK_DEPS = """
import importlib.abc
import sys

class _Broken(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "voxint.api.routers.deps":
            raise ImportError("simulated broken voxint install")
        return None

sys.meta_path.insert(0, _Broken())
"""


def _python(code: str) -> subprocess.CompletedProcess[str]:
    # A service venv has plain pytest and no plugins; drop the parent run's
    # pytest and coverage state so the child is a clean, plugin-free session.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "COV_CORE_"))}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _collect(*, block_voxint: bool) -> list[str]:
    preamble = BLOCK_VOXINT if block_voxint else ""
    proc = _python(f"{preamble}MODULES = {list(METAL_LANE_MODULES)!r}\n{COLLECT}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    marker = [line for line in proc.stdout.splitlines() if line.startswith("NODEIDS=")]
    assert len(marker) == 1, proc.stdout + proc.stderr
    node_ids: list[str] = json.loads(marker[0].removeprefix("NODEIDS="))
    return node_ids


def test_metal_lane_modules_collect_without_voxint() -> None:
    blocked = _collect(block_voxint=True)
    assert blocked, "no parity tests collected with voxint blocked"
    # The guard must not change WHICH tests the lane runs.
    assert blocked == _collect(block_voxint=False)


def test_autouse_fixture_is_a_noop_without_voxint(tmp_path: Path) -> None:
    # Requesting the fixture by name makes a renamed or unregistered fixture an
    # error, so a pass shows it ran and did nothing, not that it never ran.
    probe = "def test_probe(_isolate_template_loader):\n    pass\n"
    (tmp_path / "test_probe.py").write_text(probe)
    args = [str(tmp_path), "-p", "tests.conftest", "-p", "no:cacheprovider"]
    args += ["--rootdir", str(tmp_path)]
    proc = _python(
        f"{BLOCK_VOXINT}import pytest\nimport tests.conftest as root\n"
        f"assert root.deps is None\nraise SystemExit(pytest.main({args!r}))\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout, proc.stdout


def test_broken_voxint_install_still_fails_loudly() -> None:
    # The guard asks only whether voxint exists. An installed-but-broken app
    # must surface its ImportError, never be treated as absent.
    proc = _python(f"{BREAK_DEPS}import tests.conftest\n")
    assert proc.returncode != 0, proc.stdout
    assert "simulated broken voxint install" in proc.stderr, proc.stderr
