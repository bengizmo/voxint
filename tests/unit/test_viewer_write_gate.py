"""Viewer role write gate: viewers can browse but not mutate (#363)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from voxint.api.auth import AuthContext
from voxint.api.routers.deps import _require_write_access, viewer_write_guard


def _ctx(role: str) -> AuthContext:
    return AuthContext(user_id=None, username="test", role=role)


def _request(method: str) -> SimpleNamespace:
    return SimpleNamespace(method=method)


class TestRequireWriteAccess:
    def test_admin_passes(self) -> None:
        result = _require_write_access(_ctx("admin"))
        assert result.role == "admin"

    def test_reviewer_passes(self) -> None:
        result = _require_write_access(_ctx("reviewer"))
        assert result.role == "reviewer"

    def test_viewer_blocked(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _require_write_access(_ctx("viewer"))
        assert exc_info.value.status_code == 403


class TestViewerWriteGuard:
    @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
    def test_safe_methods_pass_for_viewer(self, method: str) -> None:
        viewer_write_guard(_request(method), _ctx("viewer"))  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_mutation_methods_blocked_for_viewer(self, method: str) -> None:
        with pytest.raises(HTTPException) as exc_info:
            viewer_write_guard(_request(method), _ctx("viewer"))  # type: ignore[arg-type]
        assert exc_info.value.status_code == 403

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_mutation_methods_pass_for_admin(self, method: str) -> None:
        viewer_write_guard(_request(method), _ctx("admin"))  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    def test_mutation_methods_pass_for_reviewer(self, method: str) -> None:
        viewer_write_guard(_request(method), _ctx("reviewer"))  # type: ignore[arg-type]


class TestShellCanWrite:
    def test_admin_can_write(self) -> None:
        from voxint.api.routers.deps import _shell_template_context
        from voxint.config import Settings

        settings = Settings(database_url="postgresql+psycopg://x/x")
        state = SimpleNamespace(
            settings=settings,
            csrf_secret="test",
            current_user=AuthContext(user_id=None, username="a", role="admin"),
        )
        app = SimpleNamespace(state=state)
        request = SimpleNamespace(app=app, state=state)
        ctx = _shell_template_context(request)  # type: ignore[arg-type]
        assert ctx["shell"]["can_write"] is True

    def test_reviewer_can_write(self) -> None:
        from voxint.api.routers.deps import _shell_template_context
        from voxint.config import Settings

        settings = Settings(database_url="postgresql+psycopg://x/x")
        state = SimpleNamespace(
            settings=settings,
            csrf_secret="test",
            current_user=AuthContext(
                user_id=None, username="r", role="reviewer"
            ),
        )
        app = SimpleNamespace(state=state)
        request = SimpleNamespace(app=app, state=state)
        ctx = _shell_template_context(request)  # type: ignore[arg-type]
        assert ctx["shell"]["can_write"] is True

    def test_viewer_cannot_write(self) -> None:
        from voxint.api.routers.deps import _shell_template_context
        from voxint.config import Settings

        settings = Settings(database_url="postgresql+psycopg://x/x")
        state = SimpleNamespace(
            settings=settings,
            csrf_secret="test",
            current_user=AuthContext(
                user_id=None, username="v", role="viewer"
            ),
        )
        app = SimpleNamespace(state=state)
        request = SimpleNamespace(app=app, state=state)
        ctx = _shell_template_context(request)  # type: ignore[arg-type]
        assert ctx["shell"]["can_write"] is False
