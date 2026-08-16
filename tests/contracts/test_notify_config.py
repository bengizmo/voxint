"""Contract: the run-notification settings stay documented and gated (issue #12).

Mirrors ``test_run_assets_config.py`` for the webhook surface: every ``notify_*``
field documented in ``.env.example`` under its env-var name (a field added
without documentation fails here), disabled by default, and the enable-time
validator — enabling REQUIRES a public URL and a strong secret (fail-closed at
startup, sanitized message with no secret)."""

import re

import pytest
from pydantic import ValidationError

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings

_NOTIFY_FIELDS = [name for name in Settings.model_fields if name.startswith("notify_")]

_URL = "https://hooks.example.com/voxint"
_SECRET = "a-sufficiently-long-secret"


def test_every_notify_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _NOTIFY_FIELDS, "notify settings fields disappeared"
    missing = [
        name
        for name in _NOTIFY_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_notify_default_off() -> None:
    assert Settings(_env_file=None).notify_enabled is False


def test_notify_enabled_requires_url() -> None:
    with pytest.raises(ValidationError, match="notify_webhook_url"):
        Settings(_env_file=None, notify_enabled=True, notify_webhook_secret=_SECRET)


def test_notify_enabled_requires_strong_secret() -> None:
    with pytest.raises(ValidationError, match="notify_webhook_secret"):
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_url=_URL,
            notify_webhook_secret="short",
        )


def test_notify_enabled_rejects_non_public_url() -> None:
    with pytest.raises(ValidationError, match="notify_webhook_url"):
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_url="http://localhost/hook",
            notify_webhook_secret=_SECRET,
        )


def test_notify_enable_time_validator_never_echoes_secret() -> None:
    try:
        Settings(
            _env_file=None,
            notify_enabled=True,
            notify_webhook_url=_URL,
            notify_webhook_secret="short",
        )
    except ValidationError as exc:
        assert "short" not in str(exc)
    else:  # pragma: no cover - the row above must raise
        raise AssertionError("expected a ValidationError")


def test_notify_enabled_constructs_with_url_and_secret() -> None:
    settings = Settings(
        _env_file=None,
        notify_enabled=True,
        notify_webhook_url=_URL,
        notify_webhook_secret=_SECRET,
    )
    assert settings.notify_enabled is True
    assert settings.notify_webhook_url == _URL
