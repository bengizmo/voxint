"""Per-area routers for the Console 2.0 decomposition (issue #151).

The ~7.5k-line ``voxint.api.app`` is being split into one router module per
feature area, behavior-preserving, one family per commit. Shared dependency,
auth, onboarding-gate, and CSRF-verification seams live in :mod:`deps`; each
area module builds an :class:`fastapi.APIRouter` that ``app.create_app`` mounts.
Router modules import from :mod:`deps` and leaf modules, never from
``voxint.api.app`` (that would be a cycle).
"""
