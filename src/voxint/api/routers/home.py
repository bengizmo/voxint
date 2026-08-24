"""Scaffold for the home area (/ and the home queue).

Empty by design: P1 promotes the review queue to the console home
(Console 2.0 epic #149). Until then this router
carries no routes and is not registered; the module exists so the P0b layout
is complete and later phases only add routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from voxint.api.routers.deps import require_onboarded

router = APIRouter(dependencies=[Depends(require_onboarded)])
