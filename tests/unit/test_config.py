from pathlib import Path

from voxint.config import Settings


def test_defaults_are_localhost_and_llm_disabled() -> None:
    s = Settings(_env_file=None)
    assert s.api_host == "127.0.0.1"
    assert s.llm_enabled is False
    assert s.domain_pack_path is None
    assert s.media_root == Path("/data/media")


def test_env_overrides(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("API_PORT", "9090")
    s = Settings(_env_file=None)
    assert s.llm_enabled is True
    assert s.api_port == 9090
