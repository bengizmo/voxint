"""Required CI checks contract (#698).

GitHub branch protection on ``main`` requires the check runs named in
``REQUIRED_CHECKS``. That set lives in repository settings, which a test cannot
read, so this pins the two things in the repo that the setting depends on:

* every required check is a ``ci.yml`` job that always runs on a pull request
  and can only report its real result: no ``pull_request`` path or branch
  filter (a filtered-out PR never reports and waits forever), no job-level
  ``if:`` (a skipped job satisfies a required check), no ``name:`` override
  (the check run would carry a different name than the one protection
  expects), and no ``continue-on-error`` (a failure would read as a pass);
* the contributor docs list exactly this set, so they cannot drift from it
  again.

When the branch-protection set changes, update ``REQUIRED_CHECKS`` in the same
commit.
"""

import re

import yaml

from tests.contracts.conftest import REPO_ROOT

REQUIRED_CHECKS = ("lint-test", "secrets-scan", "coverage", "frontend")

_CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
_PR_FILTERS = ("paths", "paths-ignore", "branches", "branches-ignore")


def _ci() -> dict:
    return yaml.safe_load(_CI_YML.read_text())


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
        f"ci.yml pull_request trigger must not filter ({filters}): a PR it skips "
        "never reports its required checks"
    )


def test_required_checks_are_unconditional_ci_jobs() -> None:
    jobs = _ci().get("jobs") or {}
    for check in REQUIRED_CHECKS:
        assert check in jobs, f"required check {check!r} is not a ci.yml job"
        job = jobs[check] or {}
        assert "if" not in job, (
            f"required job {check!r} must not set if: (a skipped job passes the check)"
        )
        assert job.get("name", check) == check, (
            f"required job {check!r} must not rename its check run"
        )
        assert job.get("continue-on-error") in (None, False), (
            f"required job {check!r} must not set continue-on-error"
        )
        for step in job.get("steps") or []:
            assert step.get("continue-on-error") in (None, False), (
                f"required job {check!r} step {step.get('name')!r} must not set continue-on-error"
            )


def test_frontend_job_runs_every_frontend_gate() -> None:
    # A required check is only as strong as what it runs: the vitest suite ran
    # nowhere in CI before #698.
    job = (_ci().get("jobs") or {})["frontend"]
    runs = [str(step.get("run", "")).strip() for step in job.get("steps") or []]
    for command in ("npm ci", "npm run lint", "npm run typecheck", "npm test", "npm run build"):
        assert command in runs, f"the frontend job must run `{command}`, got {runs}"


def test_contributing_lists_exactly_the_required_checks() -> None:
    text = _CONTRIBUTING.read_text()
    intro = "The required checks are:"
    assert intro in text, f"CONTRIBUTING.md lost its {intro!r} table"
    table = text.split(intro, 1)[1].lstrip("\n").split("\n\n", 1)[0]
    listed = re.findall(r"^\| `([\w-]+)` \|", table, flags=re.MULTILINE)
    assert sorted(listed) == sorted(REQUIRED_CHECKS), (
        f"CONTRIBUTING.md lists {listed}, branch protection requires {list(REQUIRED_CHECKS)}"
    )
