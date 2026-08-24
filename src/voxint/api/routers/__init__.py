"""Per-area console routers (Console 2.0, issue #151).

``app.py`` historically defined every console route in one module. P0b splits
that surface into one module per console area, each exporting FastAPI
``APIRouter`` instances that ``app.py``'s factory registers. Shared
dependencies, auth, and template plumbing live in :mod:`voxint.api.routers.deps`.

Layout:

- ``deps.py``      — shared dependencies, auth gate, templates, asset manifest
- ``home.py``      — scaffold (routes arrive in P1)
- ``media.py``     — scaffold (routes arrive in P2)
- ``projects.py``  — scaffold (routes arrive in P2)
- ``jobs.py``      — scaffold (routes arrive in P2)
- ``editor.py``    — scaffold (routes arrive in P3)
- ``speakers.py``, ``settings.py``, ``legacy_review.py``, ``legacy_runs.py``
  — today's routes, moved behavior-preserving from ``app.py``

Import rules (acyclic): ``deps`` imports no router module and never ``app``;
area modules may import ``deps``; ``app.py`` imports both.
"""
