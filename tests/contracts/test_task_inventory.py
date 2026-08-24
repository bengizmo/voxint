"""Contract: the Celery task-name inventory is frozen (issue #137).

Task names are identity contract. When the plugin epic (#136) moves a feature's
task into a plugin package, the task name is grandfathered — ``voxint.translate_run``
stays ``voxint.translate_run`` — so a redelivered message and any pinned route keep
working. This golden pins every ``voxint.*`` task the worker registers so a
conversion that renames a task (or a new plugin that squats an existing name)
fails loudly. New greenfield plugin tasks use the ``voxint.plugin.<id>.*``
convention and are added to the golden in the same change.
"""

from __future__ import annotations

import json

import voxint.worker.tasks  # noqa: F401  (import registers the tasks on the app)
from tests.contracts.conftest import REPO_ROOT
from voxint.worker.app import app

_GOLDEN = REPO_ROOT / "tests" / "contracts" / "fixtures" / "task_inventory.json"


def test_task_inventory_matches_golden() -> None:
    golden = json.loads(_GOLDEN.read_text())
    actual = sorted(name for name in app.tasks if name.startswith("voxint."))
    assert actual == golden, (
        "Celery task inventory changed. If this is intentional (a new plugin task "
        f"or a deliberate rename), regenerate {_GOLDEN.relative_to(REPO_ROOT)}; a "
        "grandfathered task name must never change during a plugin conversion."
    )


def test_core_task_names_guard_matches_inventory() -> None:
    """The worker's plugin-vs-core collision guard (#138) must know every core task.

    ``worker/app.py`` hardcodes ``_CORE_TASK_NAMES`` to reject a plugin declaring a
    task that shadows a core one, deriving it WITHOUT importing the task modules.
    Pin it to the golden so adding a core task without updating the guard (which
    would silently let a plugin squat the new name) fails here.
    """
    from voxint.worker.app import _CORE_TASK_NAMES

    golden = json.loads(_GOLDEN.read_text())
    assert sorted(_CORE_TASK_NAMES) == golden
