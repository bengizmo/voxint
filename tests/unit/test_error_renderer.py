"""Content-negotiated HTML and JSON error responses."""

import pytest
from fastapi import HTTPException, Request
from starlette.testclient import TestClient

from voxint.api.app import create_app
from voxint.config import Settings

SECRET = "sensitive traceback detail"


@pytest.fixture()
def app_client(tmp_path) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(settings=Settings(media_root=tmp_path))

    async def http_error(request: Request) -> None:
        status_code = int(request.path_params["status_code"])
        raise HTTPException(status_code=status_code, detail=SECRET)

    async def unhandled_error() -> None:
        raise RuntimeError(SECRET)

    app.add_api_route("/_test/error/{status_code}", http_error)
    app.add_api_route("/_test/unhandled", unhandled_error)
    return TestClient(app, raise_server_exceptions=False)


def _assert_security_headers(response) -> None:  # type: ignore[no-untyped-def]
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_http_exception_404_html(app_client: TestClient) -> None:
    response = app_client.get("/_test/error/404", headers={"accept": "text/html"})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "Page not found" in response.text
    assert "The page you were looking for does not exist or has been moved." in response.text


def test_http_exception_404_json(app_client: TestClient) -> None:
    response = app_client.get(
        "/_test/error/404", headers={"accept": "application/json"}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": SECRET}


def test_http_exception_500_html_hides_detail(app_client: TestClient) -> None:
    response = app_client.get("/_test/error/500", headers={"accept": "text/html"})

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("text/html")
    assert "Something went wrong" in response.text
    assert "Internal Server Error" in response.text
    assert SECRET not in response.text


def test_unhandled_exception_html_hides_traceback(app_client: TestClient) -> None:
    response = app_client.get("/_test/unhandled", headers={"accept": "text/html"})

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("text/html")
    assert "Internal Server Error" in response.text
    assert SECRET not in response.text
    assert "Traceback" not in response.text


def test_unhandled_exception_json_hides_traceback(app_client: TestClient) -> None:
    response = app_client.get(
        "/_test/unhandled", headers={"accept": "application/json"}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert SECRET not in response.text
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    ("path", "accept"),
    [
        ("/_test/error/404", "text/html"),
        ("/_test/error/404", "application/json"),
        ("/_test/error/500", "text/html"),
        ("/_test/unhandled", "text/html"),
        ("/_test/unhandled", "application/json"),
    ],
)
def test_security_headers_on_all_error_responses(
    app_client: TestClient, path: str, accept: str
) -> None:
    _assert_security_headers(app_client.get(path, headers={"accept": accept}))


def test_http_exception_500_json_hides_detail(app_client: TestClient) -> None:
    response = app_client.get(
        "/_test/error/500", headers={"accept": "application/json"}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert SECRET not in response.text


def test_http_exception_403_html_shows_detail(app_client: TestClient) -> None:
    response = app_client.get("/_test/error/403", headers={"accept": "text/html"})

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("text/html")
    assert "Forbidden" in response.text
    assert SECRET in response.text


def test_html_error_detail_is_escaped(app_client: TestClient) -> None:
    app = app_client.app

    async def xss_error(request: Request) -> None:
        raise HTTPException(status_code=400, detail='<script>alert("xss")</script>')

    app.add_api_route("/_test/xss", xss_error)  # type: ignore[union-attr]
    response = app_client.get("/_test/xss", headers={"accept": "text/html"})

    assert response.status_code == 400
    assert "<script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_5xx_headers_not_leaked(app_client: TestClient) -> None:
    app = app_client.app

    async def leaky_500(request: Request) -> None:
        raise HTTPException(
            status_code=500,
            detail="secret",
            headers={"X-Debug": "leak"},
        )

    app.add_api_route("/_test/leaky500", leaky_500)  # type: ignore[union-attr]
    response = app_client.get(
        "/_test/leaky500", headers={"accept": "application/json"}
    )

    assert response.status_code == 500
    assert "X-Debug" not in response.headers


def test_4xx_headers_preserved(app_client: TestClient) -> None:
    app = app_client.app

    async def auth_error(request: Request) -> None:
        raise HTTPException(
            status_code=401,
            detail="unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    app.add_api_route("/_test/auth", auth_error)  # type: ignore[union-attr]
    response = app_client.get(
        "/_test/auth", headers={"accept": "application/json"}
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_nonexistent_route_returns_html_to_browser(app_client: TestClient) -> None:
    response = app_client.get("/definitely-not-a-route", headers={"accept": "text/html"})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "Page not found" in response.text
    _assert_security_headers(response)
