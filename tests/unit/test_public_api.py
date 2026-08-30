"""Integration-style tests for the /api/v1/ public API surface.

Exercises the full request path through the mounted sub-app: bearer auth,
route dispatch, JSON error handling, and response shape. Uses the real
app with monkeypatched dependencies to avoid needing a live database.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from voxint.api.app import create_app
from voxint.api_keys import TOKEN_PREFIX, create_api_key
from voxint.config import Settings
from voxint.db.models import (
    ApiKey,
    Base,
    MediaItem,
    PipelineRun,
    User,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _strip_pg_checks(*tables: Any) -> list[tuple[Any, list[Any]]]:
    from sqlalchemy import CheckConstraint as CC

    removed_all: list[tuple[Any, list[Any]]] = []
    for table in tables:
        removed = [
            c
            for c in table.constraints
            if isinstance(c, CC)
            and c.name
            and (
                "format_check" in c.name
                or "nonempty_check" in c.name
                or "no_colon_check" in c.name
                or "status_check" in c.name
                or "current_stage_check" in c.name
                or "revision_nonneg" in c.name
                or "claim_shape" in c.name
                or "sidecar_object" in c.name
                or "diarization" in c.name
                or "detected_language" in c.name
                or "pairing" in c.name
                or "probability" in c.name
            )
        ]
        for c in removed:
            table.constraints.remove(c)
        removed_all.append((table, removed))
    return removed_all


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    tables_needed = [
        User.__table__,
        ApiKey.__table__,
        PipelineRun.__table__,
        MediaItem.__table__,
    ]
    stripped = _strip_pg_checks(*tables_needed)
    try:
        Base.metadata.create_all(engine, tables=tables_needed)
    finally:
        for table, checks in stripped:
            for c in checks:
                table.append_constraint(c)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        yield session
    engine.dispose()


@pytest.fixture()
def api_key(db_session: Session) -> str:
    created = create_api_key(db_session, name="test-key")
    db_session.commit()
    return created.plaintext_token


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        database_url="sqlite://",
        media_root="/tmp/voxint-test-media",
        voxint_api_key="env-test-key-for-api",
    )


@pytest.fixture()
def client(
    db_session: Session, settings: Settings
) -> Iterator[TestClient]:
    factory = sessionmaker(db_session.bind, expire_on_commit=False)
    app = create_app(settings=settings, session_factory=factory)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestBearerAuth:
    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/whoami")
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "unauthorized"

    def test_bad_token_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/whoami", headers=_auth_header("bogus"))
        assert resp.status_code == 401

    def test_env_key_authenticates(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/whoami",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "api:env"
        assert body["role"] == "admin"

    def test_db_key_authenticates(
        self, client: TestClient, api_key: str
    ) -> None:
        resp = client.get(
            "/api/v1/whoami", headers=_auth_header(api_key)
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "admin"
        assert "api:" in body["username"]

    def test_session_cookie_rejected(self, client: TestClient) -> None:
        client.cookies.set("voxint_session", "fake-session-token")
        resp = client.get("/api/v1/whoami")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Status / whoami
# ---------------------------------------------------------------------------

class TestStatusEndpoints:
    def test_status_ok(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/status",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_whoami_fields(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/whoami",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "username" in body
        assert "role" in body
        assert "user_id" in body


# ---------------------------------------------------------------------------
# Keys CRUD
# ---------------------------------------------------------------------------

class TestKeysApi:
    def test_create_key(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/keys",
            json={"name": "my-key"},
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "plaintext_token" in body
        assert body["plaintext_token"].startswith(TOKEN_PREFIX)
        assert body["name"] == "my-key"

    def test_list_keys(self, client: TestClient) -> None:
        client.post(
            "/api/v1/keys",
            json={"name": "list-test"},
            headers=_auth_header("env-test-key-for-api"),
        )
        resp = client.get(
            "/api/v1/keys",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        names = [k["name"] for k in body]
        assert "list-test" in names

    def test_revoke_key(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/v1/keys",
            json={"name": "revoke-me"},
            headers=_auth_header("env-test-key-for-api"),
        )
        key_id = create_resp.json()["id"]
        resp = client.delete(
            f"/api/v1/keys/{key_id}",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        assert resp.json()["revoked_at"] is not None

    def test_revoked_key_cannot_authenticate(
        self, client: TestClient
    ) -> None:
        create_resp = client.post(
            "/api/v1/keys",
            json={"name": "short-lived"},
            headers=_auth_header("env-test-key-for-api"),
        )
        body = create_resp.json()
        token = body["plaintext_token"]
        key_id = body["id"]

        check = client.get(
            "/api/v1/whoami", headers=_auth_header(token)
        )
        assert check.status_code == 200

        client.delete(
            f"/api/v1/keys/{key_id}",
            headers=_auth_header("env-test-key-for-api"),
        )

        check2 = client.get(
            "/api/v1/whoami", headers=_auth_header(token)
        )
        assert check2.status_code == 401


# ---------------------------------------------------------------------------
# Runs listing
# ---------------------------------------------------------------------------

class TestRunsApi:
    def _seed_run(
        self, db_session: Session, *, status: str = "completed"
    ) -> uuid.UUID:
        media = MediaItem(
            id=uuid.uuid4(),
            source_path=f"incoming/{uuid.uuid4()}/test.wav",
        )
        db_session.add(media)
        db_session.flush()
        run = PipelineRun(
            id=uuid.uuid4(),
            media_item_id=media.id,
            status=status,
            revision=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        db_session.add(run)
        db_session.commit()
        return run.id

    def test_list_runs_empty(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/runs",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["has_more"] is False
        assert body["next_cursor"] is None

    def test_list_runs_with_data(
        self, client: TestClient, db_session: Session
    ) -> None:
        run_id = self._seed_run(db_session)
        resp = client.get(
            "/api/v1/runs",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == str(run_id)
        assert body["items"][0]["status"] == "completed"

    def test_get_run_detail(
        self, client: TestClient, db_session: Session
    ) -> None:
        run_id = self._seed_run(db_session)
        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(run_id)
        assert "revision" in body
        assert "current_stage" in body

    def test_get_run_not_found(self, client: TestClient) -> None:
        resp = client.get(
            f"/api/v1/runs/{uuid.uuid4()}",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 404

    def test_filter_by_status(
        self, client: TestClient, db_session: Session
    ) -> None:
        self._seed_run(db_session, status="completed")
        self._seed_run(db_session, status="failed")
        resp = client.get(
            "/api/v1/runs?status=completed",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert all(r["status"] == "completed" for r in body["items"])


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------

class TestOpenApi:
    def test_openapi_json(self, client: TestClient) -> None:
        resp = client.get("/api/v1/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/keys" in schema["paths"]
        assert "/runs" in schema["paths"]

    def test_docs_page(self, client: TestClient) -> None:
        resp = client.get("/api/v1/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()

    def test_console_docs_still_disabled(self, client: TestClient) -> None:
        resp = client.get("/docs")
        assert resp.status_code in (404, 401)

    def test_console_openapi_still_disabled(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code in (404, 401)


# ---------------------------------------------------------------------------
# Error format
# ---------------------------------------------------------------------------

class TestErrorFormat:
    def test_json_error_envelope(self, client: TestClient) -> None:
        resp = client.get("/api/v1/whoami")
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]

    def test_404_returns_json(self, client: TestClient) -> None:
        resp = client.get(
            "/api/v1/nonexistent",
            headers=_auth_header("env-test-key-for-api"),
        )
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "not_found"
