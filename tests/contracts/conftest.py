"""Loader for the services' torch-free modules.

The GPU services are self-contained images, not part of the voxint package, so
contract tests import their schema/path/postprocess modules from file paths
under unique module names. Anything loaded here must stay importable without
torch/GPU deps — that's the services' side of the contract-test bargain.
"""

import importlib
import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICES_DIR = REPO_ROOT / "services"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_service_main(service: str) -> ModuleType:
    """Import a service's FastAPI app (``app.main``) for route-level tests.

    The services all use the package name ``app``, so previously-imported
    ``app*`` modules are swapped out around the import and the result is
    re-registered under a service-unique name.
    """
    unique = f"voxint_service_{service}_main"
    if unique in sys.modules:
        return sys.modules[unique]

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "app" or k.startswith("app.")}
    sys.path.insert(0, str(SERVICES_DIR / service))
    try:
        mod = importlib.import_module("app.main")
    finally:
        sys.path.remove(str(SERVICES_DIR / service))
        for k in list(sys.modules):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]
        sys.modules.update(saved)
    sys.modules[unique] = mod
    return mod


def load_service_module(service: str, module: str) -> ModuleType:
    """Import one torch-free module from a service's ``app`` package.

    Same package-swap dance as ``load_service_main`` (rather than a bare
    file-path load) so intra-package imports like
    ``from app.preprocess import ...`` resolve.
    """
    name = f"voxint_contract_{service}_{module}"
    if name in sys.modules:
        return sys.modules[name]

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "app" or k.startswith("app.")}
    sys.path.insert(0, str(SERVICES_DIR / service))
    try:
        mod = importlib.import_module(f"app.{module}")
    finally:
        sys.path.remove(str(SERVICES_DIR / service))
        for k in list(sys.modules):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]
        sys.modules.update(saved)
    sys.modules[name] = mod
    return mod


@contextmanager
def service_package(service: str) -> Iterator[None]:
    """Temporarily make one service's ``app`` package importable.

    For calling service functions that lazily ``import app.<something>`` at
    call time (e.g. the titanet engine factory) — the load_service_* helpers
    tear the package down after import, so those lazy imports need the
    package context restored around the call.
    """
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "app" or k.startswith("app.")}
    sys.path.insert(0, str(SERVICES_DIR / service))
    try:
        yield
    finally:
        sys.path.remove(str(SERVICES_DIR / service))
        for k in list(sys.modules):
            if k == "app" or k.startswith("app."):
                del sys.modules[k]
        sys.modules.update(saved)


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES_DIR / name).read_text())


# --- Shared CSS-source parsing for the base.html contracts (#94) -------------
# test_theme_toggle_css and test_speaker_palette_parity guard the SAME
# "dark blocks must stay identical" invariant by parsing the SAME file; a
# parser fix applied to one private copy but not the other would let the two
# contracts disagree about what the CSS says, so the brace-walker lives here
# once (multi-model review finding, 2-of-3).


def strip_css_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def selector_bodies(css: str, selector_pattern: str) -> list[tuple[int, str]]:
    """``(start_index, body)`` of every block whose selector matches.

    NOT a single broad regex: the dark media query nests a ``:root`` block, so
    bodies are extracted by walking brace depth from each selector's opening
    brace. Unbalanced braces are a hard failure, never a silent empty result.
    """
    bodies: list[tuple[int, str]] = []
    for match in re.finditer(selector_pattern + r"\s*\{", css):
        open_brace = css.index("{", match.end() - 1)
        depth = 0
        for i in range(open_brace, len(css)):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append((match.start(), css[open_brace + 1 : i]))
                    break
        else:
            raise AssertionError("unbalanced braces in base.html CSS")
    return bodies


@pytest.fixture
def whisper_schemas() -> ModuleType:
    return load_service_module("whisper", "schemas")


@pytest.fixture
def pyannote_schemas() -> ModuleType:
    return load_service_module("pyannote", "schemas")


@pytest.fixture
def titanet_schemas() -> ModuleType:
    return load_service_module("titanet", "schemas")
