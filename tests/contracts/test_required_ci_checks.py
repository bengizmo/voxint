"""Required CI checks contract (#698).

GitHub branch protection on ``main`` requires the check runs named in
``REQUIRED_CHECKS``. That set lives in repository settings, which a test cannot
read, so this pins what the setting depends on inside the repo:

* every required check is a ``ci.yml`` job that runs on every pull request
  update and reports its real result. GitHub counts a skipped job as passing a
  required check, so a required job takes no job- or step-level ``if:`` and no
  ``needs:`` (a failed dependency skips it). A ``pull_request`` path, branch,
  or ``types`` filter would leave some PR updates with no report at all. A
  ``name:`` override or a matrix renames the check run away from the name
  protection expects. ``continue-on-error`` turns a failure into a pass.
* each required job still runs its gate commands, compared line by line, so
  ``npm test || true`` or a deleted step fails here, and ``npm test`` still
  means ``vitest run``.
* the contributor docs list exactly this set.

When the branch-protection set or a gate command changes, update this file in
the same commit.
"""

import json
import re
from typing import Any

import yaml

from tests.contracts.conftest import REPO_ROOT

REQUIRED_CHECKS = ("lint-test", "secrets-scan", "coverage", "frontend")

# Exact command lines each required job must run. A command may sit inside a
# multi-line ``run:`` script; it must appear there as a whole line.
_GATE_COMMANDS = {
    "lint-test": (
        "uv run ruff check .",
        "uv run mypy",
        "uv run pytest tests/unit tests/contracts -n auto",
        "uv run pytest tests/parity",
        "uv run pytest tests/integration -n 8",
    ),
    "coverage": ("uv run pytest tests --cov=voxint -n 8",),
    "frontend": (
        "npm ci",
        "npm run lint",
        "npm run typecheck",
        "npm test",
        "npm run build",
        "npm audit --omit=dev --audit-level=high",
    ),
    "secrets-scan": (
        "gitleaks git --config .gitleaks.toml --redact --verbose .",
        "gitleaks dir --config .gitleaks.toml --redact --verbose .",
    ),
}

_CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
_PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"
_PR_FILTERS = ("paths", "paths-ignore", "branches", "branches-ignore", "types")
# Job keys that can skip, rename, or soften a required check run.
_FORBIDDEN_JOB_KEYS = ("if", "needs", "name", "strategy", "continue-on-error")


def _ci() -> dict[Any, Any]:
    doc: dict[Any, Any] = yaml.safe_load(_CI_YML.read_text())
    return doc


def test_required_checks_run_on_every_pull_request() -> None:
    doc = _ci()
    # PyYAML reads the bare ``on:`` key as boolean True.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict) and "pull_request" in triggers, (
        "ci.yml must trigger on pull_request"
    )
    pr = triggers["pull_request"] or {}
    filters = sorted(set(pr) & set(_PR_FILTERS))
    assert not filters, (
        f"ci.yml pull_request trigger must not filter ({filters}): a PR update it "
        "skips never reports its required checks"
    )


def test_required_checks_are_unconditional_ci_jobs() -> None:
    jobs = _ci().get("jobs") or {}
    assert set(_GATE_COMMANDS) == set(REQUIRED_CHECKS)
    for check in REQUIRED_CHECKS:
        assert check in jobs, f"required check {check!r} is not a ci.yml job"
        job = jobs[check] or {}
        present = sorted(set(job) & set(_FORBIDDEN_JOB_KEYS))
        assert not present, (
            f"required job {check!r} must not set {present}: it could skip, "
            "rename, or soften the check run"
        )
        for step in job.get("steps") or []:
            for key in ("if", "continue-on-error"):
                assert key not in step, (
                    f"required job {check!r} step {step.get('name')!r} must not set {key}"
                )


def test_required_jobs_run_their_gate_commands() -> None:
    # A required check is only as strong as what it runs: the vitest suite ran
    # nowhere in CI before #698.
    jobs = _ci().get("jobs") or {}
    for check, commands in _GATE_COMMANDS.items():
        lines = {
            line.strip()
            for step in jobs[check].get("steps") or []
            for line in str(step.get("run", "")).splitlines()
        }
        for command in commands:
            assert command in lines, f"required job {check!r} must run `{command}`"


def test_npm_test_runs_vitest() -> None:
    # `npm test` in the frontend job is only a gate while the script runs vitest.
    scripts = json.loads(_PACKAGE_JSON.read_text())["scripts"]
    assert scripts["test"] == "vitest run", (
        f"frontend/package.json scripts.test must stay `vitest run`, got {scripts['test']!r}"
    )


def test_contributing_lists_exactly_the_required_checks() -> None:
    text = _CONTRIBUTING.read_text()
    intro = "The required checks are:"
    assert intro in text, f"CONTRIBUTING.md lost its {intro!r} table"
    table = text.split(intro, 1)[1].lstrip("\n").split("\n\n", 1)[0]
    listed = re.findall(r"^\| `([\w-]+)` \|", table, flags=re.MULTILINE)
    assert sorted(listed) == sorted(REQUIRED_CHECKS), (
        f"CONTRIBUTING.md lists {listed}, branch protection requires {list(REQUIRED_CHECKS)}"
    )
