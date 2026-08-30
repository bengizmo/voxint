"""Public API: API key management (admin only)."""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from voxint.api.api_app import ApiAdminDep, ApiSessionDep
from voxint.api_keys import create_api_key, list_api_keys, revoke_api_key
from voxint.db.models import ApiKey

router = APIRouter(prefix="/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    user_id: uuid.UUID | None = None


class KeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    user_id: uuid.UUID | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    revoked_at: str | None


class CreatedKeyResponse(KeyResponse):
    plaintext_token: str


@router.get("")
def list_keys(
    identity: ApiAdminDep,
    session: ApiSessionDep,
) -> list[KeyResponse]:
    rows = list_api_keys(session, include_revoked=False)
    return [
        KeyResponse(
            id=r.id,
            name=r.name,
            key_prefix=r.key_prefix,
            user_id=r.user_id,
            created_at=r.created_at.isoformat(),
            last_used_at=r.last_used_at.isoformat() if r.last_used_at else None,
            expires_at=r.expires_at.isoformat() if r.expires_at else None,
            revoked_at=None,
        )
        for r in rows
    ]


@router.post("", status_code=201)
def create_key(
    body: CreateKeyRequest,
    identity: ApiAdminDep,
    session: ApiSessionDep,
) -> CreatedKeyResponse:
    try:
        created = create_api_key(session, name=body.name, user_id=body.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = session.get(ApiKey, created.id)
    return CreatedKeyResponse(
        id=created.id,
        name=created.name,
        key_prefix=created.key_prefix,
        user_id=row.user_id if row else body.user_id,
        created_at=row.created_at.isoformat() if row else "",
        last_used_at=None,
        expires_at=None,
        revoked_at=None,
        plaintext_token=created.plaintext_token,
    )


@router.delete("/{key_id}")
def delete_key(
    key_id: uuid.UUID,
    identity: ApiAdminDep,
    session: ApiSessionDep,
) -> KeyResponse:
    row = revoke_api_key(session, key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="key not found")
    return KeyResponse(
        id=row.id,
        name=row.name,
        key_prefix=row.key_prefix,
        user_id=row.user_id,
        created_at=row.created_at.isoformat(),
        last_used_at=row.last_used_at.isoformat() if row.last_used_at else None,
        expires_at=row.expires_at.isoformat() if row.expires_at else None,
        revoked_at=row.revoked_at.isoformat() if row.revoked_at else None,
    )
