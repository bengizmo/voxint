"""Public API: system status and identity probe."""

from typing import Any

from fastapi import APIRouter

from voxint.api.api_app import ApiKeyDep

router = APIRouter(tags=["system"])


@router.get("/status")
def api_status(identity: ApiKeyDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "user": identity.username,
        "role": identity.role,
    }


@router.get("/whoami")
def whoami(identity: ApiKeyDep) -> dict[str, Any]:
    return {
        "username": identity.username,
        "role": identity.role,
        "user_id": str(identity.user_id) if identity.user_id else None,
    }
