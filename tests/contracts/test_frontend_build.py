"""Contract + route tests for the frontend island foundation (issue #48).

Two independent layers guard the auth invariant: a *structural* test that no
``StaticFiles`` import exists (catches a future contributor swapping the route
for a convenience mount, which a 401 check alone cannot distinguish from "route
simply absent") and *behavioral* TestClient tests that the route authenticates,
contains traversal, and sets honest cache headers.
"""

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.contracts.conftest import REPO_ROOT
from voxint.api import app as app_module
from voxint.api.app import create_app
from voxint.config import Settings

CREDS = ("reviewer", "s3cret")

_DOCKERFILE = REPO_ROOT / "Dockerfile"
_APP_PY = REPO_ROOT / "src" / "voxint" / "api" / "app.py"
_CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
_NVMRC = REPO_ROOT / "frontend" / ".nvmrc"
_PYPROJECT = REPO_ROOT / "pyproject.toml"
_VITE_CONFIG = REPO_ROOT / "frontend" / "vite.config.ts"
_MAIN_TS = REPO_ROOT / "frontend" / "src" / "main.ts"
_ENTRIES_DIR = REPO_ROOT / "frontend" / "src" / "entries"
_RUN_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "run.html"
_LABELS_HTML = REPO_ROOT / "src" / "voxint" / "api" / "templates" / "fragments" / "labels.html"


# --------------------------------------------------------------------------- #
# 1. Dockerfile has the frontend stage, and copies its dist into the package.
# --------------------------------------------------------------------------- #
def test_dockerfile_has_frontend_stage() -> None:
    text = _DOCKERFILE.read_text()
    assert re.search(r"^FROM node:22-slim AS frontend$", text, re.MULTILINE), (
        "Dockerfile lost its `FROM node:22-slim AS frontend` build stage"
    )
    assert re.search(
        r"^COPY --from=frontend\s+\S+\s+\./src/voxint/api/static/app$", text, re.MULTILINE
    ), "Dockerfile no longer copies the built frontend into src/voxint/api/static/app"


def test_dockerfile_runtime_stage_has_no_node() -> None:
    # The shipping stage (base) must not install Node; Node lives only in the
    # discarded `frontend` stage. Assert no `apt-get install ... nodejs`.
    text = _DOCKERFILE.read_text()
    assert "nodejs" not in text, "runtime image must not install Node"


# --------------------------------------------------------------------------- #
# 2. No StaticFiles anywhere in app.py — the single most important auth guard.
# --------------------------------------------------------------------------- #
def test_no_staticfiles_mount_in_app() -> None:
    # The invariant is "no StaticFiles import and no StaticFiles instantiation"
    # (the *word* legitimately appears in comments explaining WHY the routes are
    # not mounts). Both an import and a mount call bypass the operator auth
    # dependency, and "everything but /healthz authenticates" is absolute.
    text = _APP_PY.read_text()
    msg = (
        "app.py must never import or instantiate StaticFiles: mounts bypass the "
        "operator auth dependency. Serve assets through /static/app instead."
    )
    assert "import StaticFiles" not in text, msg
    assert not re.search(r"\bStaticFiles\s*\(", text), msg


def test_vite_base_matches_asset_route_prefix() -> None:
    # Assets are served ONLY under the app_asset route prefix, so Vite's `base`
    # must equal it: otherwise the modulepreload helper and CSS/chunk deps emit
    # root-absolute /assets/... URLs that 404, silently breaking hydration for
    # any future island that shares a code-split chunk (issue #48 review). Pin
    # the two together so they cannot drift.
    app_text = _APP_PY.read_text()
    route = re.search(r'@app\.get\("(/static/app/)\{asset_path:path\}"\)', app_text)
    assert route, "app.py lost the /static/app/{asset_path:path} route"
    prefix = route.group(1)
    vite_text = _VITE_CONFIG.read_text()
    assert re.search(rf'base:\s*"{re.escape(prefix)}"', vite_text), (
        f"vite.config.ts `base` must equal the asset route prefix {prefix!r} so "
        "preload/CSS-dependency URLs resolve under the served root"
    )


# --------------------------------------------------------------------------- #
# 3. Traversal-containment invariant, tested directly against the asset root.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attack",
    ["../../etc/passwd", "../secret.js", "/etc/passwd"],
)
def test_asset_root_containment_rejects_escape(attack: str) -> None:
    root = app_module._APP_ASSETS_DIR
    candidate = (root / attack).resolve()
    assert not candidate.is_relative_to(root), (
        f"{attack!r} must resolve OUTSIDE the asset root; the route relies on this"
    )


def test_asset_root_containment_accepts_nested() -> None:
    root = app_module._APP_ASSETS_DIR
    candidate = (root / "assets/main-abcdef12.js").resolve()
    assert candidate.is_relative_to(root)


# --------------------------------------------------------------------------- #
# 4. Behavioral route tests (TestClient, real app, no DB needed).
# --------------------------------------------------------------------------- #
@pytest.fixture
def asset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "app").resolve()
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "main-D_zxMlff.js").write_text("console.log('main')")
    (root / "assets" / "tailwind-BVSDHUy6.css").write_text("body{color:red}")
    (root / "unhashed.js").write_text("console.log('unhashed')")
    (root / "weird.bin").write_bytes(b"\x00\x01")
    monkeypatch.setattr(app_module, "_APP_ASSETS_DIR", root)
    return root


@pytest.fixture
def client() -> TestClient:
    settings = Settings(voxint_user=CREDS[0], voxint_password=CREDS[1])
    return TestClient(create_app(settings=settings), raise_server_exceptions=False)


def test_asset_route_challenges_unauthenticated(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/assets/main-D_zxMlff.js")
    assert resp.status_code == 401
    assert "WWW-Authenticate" in resp.headers


def test_hashed_js_served_with_immutable_cache(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/assets/main-D_zxMlff.js", auth=CREDS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/javascript")
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert "console.log('main')" in resp.text


def test_hashed_css_served_with_css_type(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/assets/tailwind-BVSDHUy6.css", auth=CREDS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")


def test_unhashed_asset_not_immutably_cached(client: TestClient, asset_root: Path) -> None:
    # An unhashed name could change in place; pinning it in browsers would be
    # dishonest, so it must NOT get the immutable header.
    resp = client.get("/static/app/unhashed.js", auth=CREDS)
    assert resp.status_code == 200
    assert "immutable" not in resp.headers.get("cache-control", "")


def test_unknown_suffix_falls_back_to_octet_stream(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/weird.bin", auth=CREDS)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")


def test_missing_asset_is_404(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/assets/nope.js", auth=CREDS)
    assert resp.status_code == 404


def test_directory_is_404(client: TestClient, asset_root: Path) -> None:
    resp = client.get("/static/app/assets", auth=CREDS)
    assert resp.status_code == 404


def test_symlink_escape_is_404(
    client: TestClient, asset_root: Path, tmp_path: Path
) -> None:
    # A symlink inside the root pointing outside must 404 (containment runs on
    # the resolved path). This exercises the is_relative_to guard behaviorally,
    # unaffected by any URL dot-segment normalization the client might apply.
    outside = tmp_path / "outside.js"
    outside.write_text("secret")
    (asset_root / "escape.js").symlink_to(outside)
    resp = client.get("/static/app/escape.js", auth=CREDS)
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 4b. Runtime npm dependencies stay exactly {react, react-dom} (issue #57).
#     The waveform strip was deliberately hand-rolled instead of vendoring
#     wavesurfer.js; this pin turns that no-new-runtime-dep decision into an
#     enforced invariant rather than a review norm. Widening it is a deliberate
#     act: update this test in the same commit, with the reasoning recorded.
# --------------------------------------------------------------------------- #
def test_runtime_npm_dependencies_are_exactly_react() -> None:
    package_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text())
    assert set(package_json["dependencies"]) == {"react", "react-dom"}, (
        "frontend runtime dependencies changed — if deliberate, update this "
        "contract in the same commit and record why the new dep earns its place"
    )


# --------------------------------------------------------------------------- #
# 5. Node-version pin parity across Dockerfile, ci.yml, and .nvmrc.
# --------------------------------------------------------------------------- #
def test_node_version_pins_agree() -> None:
    dockerfile = _DOCKERFILE.read_text()
    ci = _CI_YML.read_text()
    nvmrc = _NVMRC.read_text().strip()

    m_docker = re.search(r"FROM node:(\d+)-slim AS frontend", dockerfile)
    m_ci = re.search(r'node-version:\s*"(\d+)"', ci)
    assert m_docker is not None, "Dockerfile frontend stage lost its node: pin"
    assert m_ci is not None, "ci.yml frontend job lost its node-version pin"

    versions = {m_docker.group(1), m_ci.group(1), nvmrc}
    assert versions == {"22"}, f"Node version pins drifted: {sorted(versions)}"


# --------------------------------------------------------------------------- #
# 6. The wheel packages the static/app asset tree, so the prebuilt island
#    bundles the Dockerfile writes there ship in the installed layout.
# --------------------------------------------------------------------------- #
def test_wheel_packages_include_the_asset_tree() -> None:
    # `packages = ["src/voxint"]` ships everything under the package. static/app
    # is inside it and must not be excluded (a force-include would double-add and
    # break the non-editable wheel — see pyproject comment).
    text = _PYPROJECT.read_text()
    assert re.search(r'packages\s*=\s*\[\s*"src/voxint"\s*\]', text), (
        "wheel target must package src/voxint (which contains static/app)"
    )
    assert "[tool.hatch.build.targets.wheel.force-include]" not in text, (
        "no force-include table: static/app is inside the packaged tree, so "
        "force-including it double-adds every file and aborts the non-editable "
        "wheel build (the word may still appear in an explanatory comment)"
    )


def test_built_wheel_contains_static_app_tree(tmp_path: Path) -> None:
    # Thorough end-to-end proof: build the wheel and assert the static/app tree
    # ships. In a source checkout only .gitkeep lives there; the Dockerfile
    # overlays the hashed bundles the same way, so their inclusion rides on this
    # exact mechanism.
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    out = tmp_path / "wheel"
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"uv build failed:\n{result.stderr}"
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any(n.startswith("voxint/api/static/app/") for n in names), (
        "built wheel does not ship the voxint/api/static/app tree"
    )


# --------------------------------------------------------------------------- #
# Manifest helper + hash detection (module-level, no route).
# --------------------------------------------------------------------------- #
def test_looks_hashed_distinguishes_fingerprinted_names() -> None:
    assert app_module._looks_hashed("main-D_zxMlff.js")
    assert app_module._looks_hashed("transcript-player-aYPQ60Ce.js")
    assert not app_module._looks_hashed("main.js")
    assert not app_module._looks_hashed("tailwind.css")


def test_load_asset_manifest_missing_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_APP_MANIFEST_PATH", tmp_path / "absent.json")
    assert app_module._load_asset_manifest() == {}


def test_load_asset_manifest_invalid_json_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json")
    monkeypatch.setattr(app_module, "_APP_MANIFEST_PATH", bad)
    assert app_module._load_asset_manifest() == {}


def test_load_asset_manifest_maps_entry_stem_to_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "src/main.ts": {"file": "assets/main-D_zxMlff.js", "name": "main"},
        "src/styles/tailwind.css": {"file": "assets/tailwind-BVSDHUy6.css"},
        "src/entries/transcript-player.tsx": {
            "file": "assets/transcript-player-aYPQ60Ce.js",
            "name": "transcript-player",
        },
        "noise": {"notfile": 1},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(app_module, "_APP_MANIFEST_PATH", path)
    resolved = app_module._load_asset_manifest()
    assert resolved == {
        "main": "/static/app/assets/main-D_zxMlff.js",
        "tailwind": "/static/app/assets/tailwind-BVSDHUy6.css",
        "transcript-player": "/static/app/assets/transcript-player-aYPQ60Ce.js",
    }


def test_asset_url_reads_the_loaded_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "_APP_ASSET_URLS", {"main": "/static/app/assets/main-abc12345.js"}
    )
    assert app_module.asset_url("main") == "/static/app/assets/main-abc12345.js"
    assert app_module.asset_url("nope") is None


# --------------------------------------------------------------------------- #
# 7. Island wiring: every island is registered in main.ts, has an entry file,
#    and is a Vite rollup input. Adding an island touches exactly these three
#    places (issue #48 contract) — this pins that for every island so a new one
#    can't half-land (issues #49/#55 add `workbench-player`; #84 adds
#    `corrections-editor`).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "island",
    ["transcript-player", "workbench-player", "review-stepper", "corrections-editor"],
)
def test_island_registered_in_main_ts(island: str) -> None:
    text = _MAIN_TS.read_text()
    assert re.search(
        rf'"{re.escape(island)}":\s*\(\)\s*=>\s*import\("\./entries/{re.escape(island)}"\)',
        text,
    ), f"main.ts registry is missing the {island!r} island"


@pytest.mark.parametrize(
    "island",
    ["transcript-player", "workbench-player", "review-stepper", "corrections-editor"],
)
def test_island_entry_file_exists(island: str) -> None:
    entry = _ENTRIES_DIR / f"{island}.tsx"
    assert entry.is_file(), f"missing island entry file {entry}"
    # The entry must export a `mount` the shared loader can call.
    assert "export function mount(" in entry.read_text(), (
        f"{entry} must export a mount(el) the shared loader invokes"
    )


@pytest.mark.parametrize(
    "island",
    ["transcript-player", "workbench-player", "review-stepper", "corrections-editor"],
)
def test_island_is_a_vite_input(island: str) -> None:
    text = _VITE_CONFIG.read_text()
    assert re.search(
        rf'"{re.escape(island)}":\s*"src/entries/{re.escape(island)}\.tsx"',
        text,
    ), f"vite.config.ts rollupOptions.input is missing the {island!r} entry"


# --------------------------------------------------------------------------- #
# 8. The workbench-player mount node lives OUTSIDE #labels, so it survives the
#    htmx innerHTML swap the decision cards do (issue #49 review). If it were
#    inside, every ruling would blow away the audio element mid-review.
# --------------------------------------------------------------------------- #
def test_workbench_mount_is_outside_labels() -> None:
    html = _RUN_HTML.read_text()
    island = html.find('data-island="workbench-player"')
    labels = html.find('<div id="labels">')
    assert island != -1, "run.html lost the workbench-player island mount"
    assert labels != -1, "run.html lost the #labels container"
    assert island < labels, (
        "the workbench-player island must be rendered BEFORE (outside) #labels so "
        "an htmx innerHTML swap of the decision cards never removes the <audio>"
    )
    # And the mount ships a bare <audio> fallback so JS-off still plays.
    mount_region = html[island:labels]
    assert "<audio" in mount_region, "the island mount must wrap the <audio> fallback"


# --------------------------------------------------------------------------- #
# 9. Every server-rendered seek button is disabled + type="button": disabled so
#    JS-off shows no false affordance (the island enables it only when seeking is
#    safe, issue #55); type="button" so it never submits an adjudication form.
# --------------------------------------------------------------------------- #
def test_seek_buttons_are_disabled_and_typed_button() -> None:
    html = _LABELS_HTML.read_text()
    buttons = re.findall(r"<button\b[^>]*\bdata-voxint-seek\b[^>]*>", html, re.DOTALL)
    assert len(buttons) >= 2, (
        "expected at least the per-segment and per-speaker seek buttons in labels.html"
    )
    for button in buttons:
        assert 'type="button"' in button, (
            f"seek button must be type=button so it never submits a form: {button!r}"
        )
        assert re.search(r"\bdisabled\b", button), (
            f"seek button must render disabled by default (honest JS-off): {button!r}"
        )
