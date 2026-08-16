"""Contract + route tests for the frontend island foundation (issue #48).

Two independent layers guard the auth invariant: a *structural* test that no
``StaticFiles`` import exists (catches a future contributor swapping the route
for a convenience mount, which a 401 check alone cannot distinguish from "route
simply absent") and *behavioral* TestClient tests that the route authenticates,
contains traversal, and sets honest cache headers.
"""

import json
import re
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
# 6. The wheel packages the built asset tree (force-include is present).
# --------------------------------------------------------------------------- #
def test_pyproject_force_includes_asset_tree() -> None:
    text = _PYPROJECT.read_text()
    assert "[tool.hatch.build.targets.wheel.force-include]" in text
    assert re.search(
        r'"src/voxint/api/static/app"\s*=\s*"voxint/api/static/app"', text
    ), "wheel force-include for the frontend asset tree is missing"


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
