"""Install-kind marker pins (#317).

``install_kind()`` reads ``VOXINT_INSTALL_KIND``; the two packaging seams that
stamp it must keep doing so or every install regresses to "Install type
unknown" silently:

  * the app-image Dockerfile bakes ``docker`` (ONCE, in the Dockerfile — every
    compose overlay runs images built from it, so ENV inheritance covers them);
  * the native launcher's rendered service env sets ``native`` for each core
    service (asserted through the ``native_service_env`` seam, the same env
    block the launchd plists serialize — not a script grep).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.contracts.test_native_launcher_contract import native_env

REPO = Path(__file__).resolve().parents[2]


def test_app_dockerfile_bakes_docker_marker() -> None:
    dockerfile = (REPO / "Dockerfile").read_text()
    assert re.search(r"^ENV VOXINT_INSTALL_KIND=docker$", dockerfile, re.M), (
        "the app-image Dockerfile must bake ENV VOXINT_INSTALL_KIND=docker "
        "(the single docker-home source for the settings status page)"
    )


@pytest.mark.parametrize("svc", ["api", "worker", "beat"])
def test_native_launcher_env_sets_native_marker(svc: str, tmp_path: Path) -> None:
    env = native_env(f"native_service_env {svc} /media/root", tmp_path)
    assert env["VOXINT_INSTALL_KIND"] == "native"
