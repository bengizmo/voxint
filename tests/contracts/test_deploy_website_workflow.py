"""deploy-website workflow contract (#648).

The deploy target lives in repository settings, not in the public workflow file,
and an unconfigured repository must not turn every docs push red. Pinned here:

* the host and path come from repository variables, and the workflow names no
  literal IPv4 host;
* the deploy step runs only when the configuration check says it is ready;
* the configuration check itself (run for real under ``bash -e``, as Actions
  runs it) skips on no configuration, deploys on full configuration, and fails
  on a partial one with an error annotation, so a half-configured deploy is loud
  rather than silent.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.contracts.conftest import REPO_ROOT

_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-website.yml"
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _steps() -> list[dict[str, Any]]:
    doc = yaml.safe_load(_WORKFLOW.read_text())
    steps: list[dict[str, Any]] = doc["jobs"]["build-deploy"]["steps"]
    return steps


def _step(name: str) -> dict[str, Any]:
    for step in _steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"deploy-website.yml has no {name!r} step")


def _run_config_check(host: str, path: str, key_set: str) -> tuple[int, str]:
    script = _step("Check deploy configuration")["run"]
    # Actions runs a bash step as ``bash -e -o pipefail {0}``.
    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / "github_output"
        output.touch()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GITHUB_OUTPUT": str(output),
            "DEPLOY_HOST": host,
            "DEPLOY_PATH": path,
            "DEPLOY_KEY_SET": key_set,
        }
        proc = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, output.read_text() + proc.stdout


def test_deploy_target_comes_from_repository_settings() -> None:
    env = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["build-deploy"]["env"]
    assert env["DEPLOY_HOST"] == "${{ vars.VOXINT_DEPLOY_HOST }}"
    assert env["DEPLOY_PATH"] == "${{ vars.VOXINT_DEPLOY_PATH }}"
    deploy = _step("Deploy to VPS")["run"]
    assert '"deploy@${DEPLOY_HOST}:${DEPLOY_PATH}"' in deploy
    assert 'ssh-keyscan -H "$DEPLOY_HOST"' in deploy


def test_workflow_names_no_literal_host() -> None:
    assert not _IPV4.search(_WORKFLOW.read_text()), (
        "deploy-website.yml must take its host from vars.VOXINT_DEPLOY_HOST, not a literal"
    )


def test_deploy_step_is_gated_on_the_configuration_check() -> None:
    deploy = _step("Deploy to VPS")
    assert deploy.get("if") == "steps.deploy-config.outputs.ready == 'true'"
    assert _step("Check deploy configuration").get("id") == "deploy-config"


def test_unconfigured_repository_skips_the_deploy() -> None:
    code, output = _run_config_check("", "", "false")
    assert code == 0
    assert "ready=false" in output.splitlines()
    assert "::notice title=Deploy skipped::" in output


def test_full_configuration_deploys() -> None:
    code, output = _run_config_check("site.example", "/var/www/site/", "true")
    assert code == 0
    assert output.strip() == "ready=true"
    assert "::" not in output


@pytest.mark.parametrize(
    ("host", "path", "key_set"),
    [
        ("site.example", "", "false"),
        ("", "/var/www/site/", "false"),
        ("", "", "true"),
        ("site.example", "/var/www/site/", "false"),
        ("site.example", "", "true"),
        ("", "/var/www/site/", "true"),
    ],
)
def test_partial_configuration_fails(host: str, path: str, key_set: str) -> None:
    code, output = _run_config_check(host, path, key_set)
    assert code != 0
    assert "ready=true" not in output
    assert "::error title=Deploy misconfigured::" in output
