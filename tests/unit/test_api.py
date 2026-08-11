from fastapi.testclient import TestClient

from voxint import __version__
from voxint.api.app import app


def test_healthz() -> None:
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}
