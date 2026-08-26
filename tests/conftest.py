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
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from voxint.api.routers import deps


@pytest.fixture(autouse=True)
def _isolate_template_loader() -> Iterator[None]:
    saved_loader = deps.templates.env.loader
    saved_core = deps._CORE_TEMPLATE_LOADER
    try:
        yield
    finally:
        deps.templates.env.loader = saved_loader
        deps._CORE_TEMPLATE_LOADER = saved_core
