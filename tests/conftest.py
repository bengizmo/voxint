"""Repo-wide test fixtures.

Autouse isolation for the process-global Jinja template loader. The review
console shares one ``Jinja2Templates`` singleton
(``voxint.api.routers.deps.templates``), and ``create_app`` rewrites its loader
when a plugin ships templates (the #138 seam), caching the pristine core loader
in the ``deps._CORE_TEMPLATE_LOADER`` module global. Neither is restored on app
teardown (in production the singleton is built once), so an integration test that
builds an app with an active plugin leaks a ``ChoiceLoader`` into the shared
singleton. A later unit test that asserts the pristine loader is restored then
fails, but only when both land on one xdist worker (the full-suite ``coverage``
job runs unit + integration together under ``-n``; ``lint-test`` runs unit +
contracts only, so it never sees the leak). Snapshot and restore both around
every test so template-loader state can never cross a test boundary.

This file must stay importable WITHOUT the voxint app installed (#647). The
Metal parity lane runs pytest from each model service's own venv, which holds
only the shipped service pins plus pytest; a module-level voxint import here
broke collection of every parity test in that lane. The app is imported only
when it is installed, and the fixture does nothing otherwise (with no app there
is no template loader to leak). ``find_spec`` only asks whether the package
exists, so a broken voxint install still fails loudly at import.
``tests/contracts/test_root_conftest_portable.py`` pins this.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from types import ModuleType

import pytest

deps: ModuleType | None = (
    importlib.import_module("voxint.api.routers.deps")
    if importlib.util.find_spec("voxint") is not None
    else None
)


@pytest.fixture(autouse=True)
def _isolate_template_loader() -> Iterator[None]:
    if deps is None:
        yield
        return
    saved_loader = deps.templates.env.loader
    saved_core = deps._CORE_TEMPLATE_LOADER
    try:
        yield
    finally:
        deps.templates.env.loader = saved_loader
        deps._CORE_TEMPLATE_LOADER = saved_core
