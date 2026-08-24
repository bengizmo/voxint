"""Scaffold for the editor area (/editor routes).

Empty by design: P3 introduces the editor that replaces review
(Console 2.0 epic #149). Until then this router
carries no routes and is not registered; the module exists so the P0b layout
is complete and later phases only add routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from voxint.api.routers.deps import require_onboarded

router = APIRouter(dependencies=[Depends(require_onboarded)])
