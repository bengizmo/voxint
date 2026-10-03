"""Effective row configuration determines every endpoint the readiness gate probes."""

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.unit.test_gpu_phase import phase_settings
from voxint.db.models import AppSettings
from voxint.gpu_phase.llm_probe import probe_llm


@pytest.mark.parametrize(
    "bundled,byo", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("failure", [None, "bundle", "byo", "exception"])
def test_probe_effective_endpoints(
    session_factory: sessionmaker[Session],
    bundled: bool,
    byo: bool,
    failure: str | None,
) -> None:
    settings = phase_settings(
        llm_enabled=False,
        llm_base_url="https://env.example/v1",
        llm_model="env-model",
        llm_api_key="env-key",
        llm_bundled_enabled=False,
        llm_bundled_base_url="https://bundle.example/v1",
        llm_bundled_model="bundle-model",
    )
    urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        assert request.extensions["timeout"]["read"] == 3.0
        if request.url.host == "byo.example":
            assert request.headers["Authorization"] == "Bearer row-key"
        else:
            assert "Authorization" not in request.headers
        if failure == "exception":
            raise RuntimeError("private endpoint secret")
        return httpx.Response(503 if request.url.host == f"{failure}.example" else 200)

    with session_factory() as session:
        session.add(
            AppSettings(
                id=1,
                llm_enabled=True,
                llm_bundled_enabled=bundled,
                llm_base_url="https://byo.example/v1" if byo else "https://bundle.example/v1",
                llm_model="row-model",
                llm_api_key="row-key",
            )
        )
        session.flush()
        with httpx.Client(transport=httpx.MockTransport(respond), timeout=3.0) as client:
            result = probe_llm(settings, session, client)
        expected_urls = (["https://bundle.example/v1/models"] if bundled else []) + (
            ["https://byo.example/v1/models"] if byo else []
        )
        assert urls == expected_urls
        if not bundled and not byo:
            assert result is None
        else:
            assert result is not (
                failure == "exception"
                or (bundled and failure == "bundle")
                or (byo and failure == "byo")
            )


def test_disabled_row_skips_probe(session_factory: sessionmaker[Session]) -> None:
    settings = phase_settings(llm_enabled=True, llm_base_url="https://env.example/v1")
    with session_factory() as session:
        session.add(AppSettings(id=1, llm_enabled=False))
        session.flush()
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: pytest.fail("unexpected probe"))
        ) as client:
            assert probe_llm(settings, session, client) is None


def test_probe_database_exception_is_closed(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(session: Session) -> None:
        raise RuntimeError("private key")

    monkeypatch.setattr("voxint.gpu_phase.llm_probe.get_app_settings", broken)
    with session_factory() as session:
        assert probe_llm(phase_settings(), session) is False


def test_owned_client_timeout_and_env_configuration(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = phase_settings(
        llm_enabled=True,
        llm_base_url="https://env.example/v1",
        llm_model="env-model",
    )
    urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        assert set(request.extensions["timeout"].values()) == {3.0}
        return httpx.Response(200)

    client_type = httpx.Client
    clients: list[httpx.Client] = []

    def make_client(*, timeout: float) -> httpx.Client:
        client = client_type(transport=httpx.MockTransport(respond), timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("voxint.gpu_phase.llm_probe.httpx.Client", make_client)
    with session_factory() as session:
        assert probe_llm(settings, session) is True
    assert urls == ["https://env.example/v1/models"]
    assert len(clients) == 1 and clients[0].is_closed
