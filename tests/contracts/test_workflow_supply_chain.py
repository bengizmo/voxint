"""CI supply-chain hardening contract (security audit F1/F2/F4).

Pins three invariants over ``.github/workflows/`` so a later edit cannot quietly
undo the 2026-08-18 audit remediation:

* **F1** every third-party action is referenced by a 40-hex commit SHA (never a
  mutable ``@vN`` tag), each carrying a same-line ``# vX.Y.Z`` provenance comment.
  A moved tag can swap an action's code under us; a pinned commit cannot.
* **F2** ``ci.yml`` and ``metal-lane.yml`` declare a top-level least-privilege
  token (``permissions: {contents: read}``), and every ``release.yml`` job keeps a
  least-privilege ``permissions`` mapping (``write`` only where the release must
  push images). This scopes the GitHub lane only; the Forgejo Actions mirror
  ignores ``permissions`` by design.
* **F4** the gitleaks bootstrap in ``ci.yml`` verifies a sha256 BEFORE extracting,
  and a checksum failure prevents ``tar`` from ever running.

The checks parse structure rather than matching lines, and F4 is *behavioral*:
it runs the actual step script against stubbed ``curl`` / ``sha256sum`` / ``tar``
so a bypass like ``sha256sum -c - || true`` or extracting a different archive is
caught, not just a reordered pair of textual lines.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from tests.contracts.conftest import REPO_ROOT

_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# A pinned remote action: ``owner/repo`` (optionally ``/subpath``) @ 40-hex sha.
_SHA_PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
# A pinned container action: ``docker://host/image@sha256:<64-hex>`` (allowed).
_DOCKER_PINNED = re.compile(r"^docker://\S+@sha256:[0-9a-f]{64}$")
# The provenance comment must be the trailing ``# vX.Y.Z`` pin label Dependabot
# maintains, not merely a semver mentioned anywhere on the line (so a stray
# ``# see v0.0.0`` can never stand in for a real tag annotation).
_VERSION_COMMENT = re.compile(r"#\s*v\d+\.\d+\.\d+\s*$")


def _workflow_files() -> list[Path]:
    files = sorted(_WORKFLOWS_DIR.glob("*.yml")) + sorted(_WORKFLOWS_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {_WORKFLOWS_DIR}"
    return files


def _iter_uses(node: yaml.Node):
    """Yield ``(uses_value, line_1based)`` for every ``uses:`` mapping value.

    Walks the composed YAML node tree (not text lines) so every valid spelling
    of a step or job-level ``uses`` is caught, and the value node's source mark
    recovers the line for the comment convention.
    """
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if (
                isinstance(key, yaml.ScalarNode)
                and key.value == "uses"
                and isinstance(value, yaml.ScalarNode)
            ):
                yield value.value, value.start_mark.line + 1
            yield from _iter_uses(value)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _iter_uses(item)


# --------------------------------------------------------------------------- F1


def test_every_action_is_sha_pinned_with_version_comment() -> None:
    """F1: no floating tags; each remote pin carries a ``# vX.Y.Z`` comment."""
    offenders: list[str] = []
    seen_uses = 0
    for path in _workflow_files():
        text = path.read_text()
        lines = text.splitlines()
        root = yaml.compose(text)
        if root is None:
            continue
        for ref, lineno in _iter_uses(root):
            seen_uses += 1
            where = f"{path.name}:{lineno}"
            if ref.startswith("./"):
                continue  # local action / reusable workflow: bound to repo code
            if _DOCKER_PINNED.match(ref):
                continue
            if not _SHA_PINNED.match(ref):
                offenders.append(f"{where}: unpinned or unknown `uses:` form -> {ref}")
                continue
            src = lines[lineno - 1] if 0 <= lineno - 1 < len(lines) else ""
            if not _VERSION_COMMENT.search(src):
                offenders.append(f"{where}: SHA pin lacks a `# vX.Y.Z` comment -> {ref}")
    assert seen_uses, "expected to find `uses:` mappings in the workflows"
    assert not offenders, "unpinned / uncommented actions:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- F2

_ALLOWED_PERMISSION_VALUES = {"read", "write", "none"}
# Scopes a release job may hold at ``write`` (it publishes container images).
# Widening this is a conscious edit, not an accident.
_ALLOWED_WRITE_SCOPES = {"packages"}


def test_ci_and_metal_lane_declare_least_privilege_token() -> None:
    """F2: the two GitHub-only lanes cap the default token to read-only."""
    for name in ("ci.yml", "metal-lane.yml"):
        doc = yaml.safe_load((_WORKFLOWS_DIR / name).read_text())
        perms = doc.get("permissions")
        assert perms == {"contents": "read"}, (
            f"{name}: top-level `permissions` must be exactly "
            f"{{'contents': 'read'}}, got {perms!r}"
        )
        # A job-level `permissions:` block replaces the default token entirely,
        # so the realistic F2 regression is not editing the top-level line but
        # widening one job (e.g. `permissions: {packages: write}` on the
        # PR-triggered secrets-scan). Every job here is read-only today; require
        # that none re-declares permissions at all.
        for job_name, job in (doc.get("jobs") or {}).items():
            assert "permissions" not in (job or {}), (
                f"{name}:{job_name} overrides the read-only default token — a "
                f"job-level `permissions:` widening escapes the top-level cap"
            )


def test_release_jobs_keep_least_privilege_permissions() -> None:
    """F2: every release.yml job states a least-privilege `permissions` mapping.

    Rejects the string shorthands (`write-all`/`read-all`), `null`, and any
    `write` scope outside the release's legitimate image-push need. An explicit
    empty mapping (`permissions: {}`, a pure-compute opt-out) is allowed.
    """
    doc = yaml.safe_load((_WORKFLOWS_DIR / "release.yml").read_text())
    jobs = doc.get("jobs", {})
    assert jobs, "release.yml has no jobs"
    problems: list[str] = []
    for name, job in jobs.items():
        job = job or {}
        if "permissions" not in job:
            problems.append(f"{name}: no explicit `permissions`")
            continue
        perms = job["permissions"]
        if perms == {}:
            continue
        if not isinstance(perms, dict):
            problems.append(f"{name}: `permissions` must be a mapping, got {perms!r}")
            continue
        for scope, value in perms.items():
            if value not in _ALLOWED_PERMISSION_VALUES:
                problems.append(f"{name}.{scope}: illegal permission value {value!r}")
            if value == "write" and scope not in _ALLOWED_WRITE_SCOPES:
                problems.append(
                    f"{name}.{scope}: `write` not allowed (only {sorted(_ALLOWED_WRITE_SCOPES)})"
                )
    assert not problems, "release.yml permission problems:\n" + "\n".join(problems)


# --------------------------------------------------------------------------- F4

_STUBS = {
    # Records its `-o <path>` and creates the file, so the download "succeeds".
    "curl": """#!/usr/bin/env bash
out=""
while [ $# -gt 0 ]; do
  case "$1" in -o) shift; out="$1";; esac
  shift
done
[ -n "$out" ] && : > "$out"
printf 'curl\\t%s\\n' "$out" >> "$STUB_LOG"
exit 0
""",
    # Consumes the piped "<digest>  <path>" line, records the path, exits per RC.
    "sha256sum": """#!/usr/bin/env bash
line="$(cat)"
printf 'sha256sum\\t%s\\n' "$line" >> "$STUB_LOG"
exit "${STUB_SHA_RC:-0}"
""",
    # Records the archive it was asked to extract (the arg after -xzf / -x...f).
    "tar": """#!/usr/bin/env bash
file=""
prev=""
for a in "$@"; do
  case "$prev" in -xzf|-xf|-xJf|-xjf) file="$a";; esac
  prev="$a"
done
printf 'tar\\t%s\\n' "$file" >> "$STUB_LOG"
exit 0
""",
}


def _gitleaks_run_script() -> str:
    doc = yaml.safe_load((_WORKFLOWS_DIR / "ci.yml").read_text())
    for job in doc.get("jobs", {}).values():
        for step in (job or {}).get("steps", []) or []:
            if step.get("name") == "Install gitleaks":
                return step["run"]
    raise AssertionError("ci.yml has no step named 'Install gitleaks'")


def _run_gitleaks_step(sha_rc: int, tmp: Path) -> tuple[int, list[tuple[str, str]]]:
    """Run the real step script against stubs; return (exit_code, log rows)."""
    stub_dir = tmp / "bin"
    stub_dir.mkdir()
    for name, body in _STUBS.items():
        p = stub_dir / name
        p.write_text(body)
        p.chmod(0o755)
    log = tmp / "log.tsv"
    log.write_text("")
    runner_temp = tmp / "runner"
    runner_temp.mkdir()

    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "STUB_SHA_RC": str(sha_rc),
        "RUNNER_TEMP": str(runner_temp),
        "GITLEAKS_VERSION": "8.24.3",
        "GITLEAKS_SHA256": "0" * 64,
    }
    proc = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _gitleaks_run_script()],
        env=env,
        capture_output=True,
        text=True,
    )
    rows = [
        (parts[0], parts[1] if len(parts) > 1 else "")
        for parts in (ln.split("\t") for ln in log.read_text().splitlines())
    ]
    return proc.returncode, rows


def test_gitleaks_checksum_failure_prevents_extraction() -> None:
    """F4: a bad checksum aborts the step and `tar` is never invoked."""
    with tempfile.TemporaryDirectory() as d:
        code, rows = _run_gitleaks_step(sha_rc=1, tmp=Path(d))
    assert code != 0, "step must fail when the gitleaks checksum does not verify"
    assert not any(cmd == "tar" for cmd, _ in rows), (
        "tar ran despite a failed checksum:\n" + "\n".join(map(str, rows))
    )


def test_gitleaks_verifies_and_extracts_the_same_downloaded_archive() -> None:
    """F4: on success, curl -o, sha256sum, and tar all name the same file."""
    with tempfile.TemporaryDirectory() as d:
        code, rows = _run_gitleaks_step(sha_rc=0, tmp=Path(d))
    assert code == 0, "step must succeed when the checksum verifies"
    by_cmd = dict(rows)
    assert {"curl", "sha256sum", "tar"} <= by_cmd.keys(), f"missing commands: {rows}"

    downloaded = by_cmd["curl"]
    verified = by_cmd["sha256sum"].split()[-1]  # trailing path of "<digest>  <path>"
    extracted = by_cmd["tar"]
    assert downloaded and downloaded == verified == extracted, (
        "download / verify / extract must all reference the same archive; got "
        f"curl={downloaded!r} sha256sum={verified!r} tar={extracted!r}"
    )


def test_gitleaks_gate_is_structurally_fail_closed() -> None:
    """F4 (structural): the digest env exists and is well-formed, the install
    step verifies it, and nothing downgrades the gate to advisory.

    The behavioral tests above stub the environment, so on their own they stay
    green even if the real ``GITLEAKS_SHA256`` env entry is deleted, the URL is
    repointed at a different artifact, or a ``continue-on-error: true`` turns the
    fail-closed gate into an advisory. Pin those blind spots here.
    """
    doc = yaml.safe_load((_WORKFLOWS_DIR / "ci.yml").read_text())
    env = doc.get("env") or {}
    assert "GITLEAKS_VERSION" in env, "ci.yml env is missing GITLEAKS_VERSION"
    digest = str(env.get("GITLEAKS_SHA256", ""))
    assert re.fullmatch(r"[0-9a-f]{64}", digest), (
        f"ci.yml env GITLEAKS_SHA256 must be a 64-hex sha256, got {digest!r}"
    )

    install_job = install_step = None
    for job in (doc.get("jobs") or {}).values():
        for step in (job or {}).get("steps") or []:
            if step.get("name") == "Install gitleaks":
                install_job, install_step = job, step
    assert install_step is not None, "ci.yml has no 'Install gitleaks' step"

    # The gate must stay fail-closed: a job- or step-level continue-on-error
    # would let a checksum failure pass as advisory while the behavioral tests
    # (which only run the script) stay green.
    assert (install_job or {}).get("continue-on-error") in (None, False), (
        "the secrets-scan job must not set continue-on-error (fail-closed gate)"
    )
    assert install_step.get("continue-on-error") in (None, False), (
        "the 'Install gitleaks' step must not set continue-on-error"
    )

    # It must verify the repo-held digest and must not fall back to the release's
    # co-published checksums.txt (same trust boundary as the tarball; the audit
    # deliberately does not trust it).
    script = install_step["run"]
    assert "GITLEAKS_SHA256" in script, (
        "the install step must verify $GITLEAKS_SHA256, not a fetched checksum"
    )
    assert "checksums.txt" not in script, (
        "the install step must not trust the release's co-published checksums.txt"
    )
