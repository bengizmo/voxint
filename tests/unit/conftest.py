"""Unit-test isolation from the operator's real dotenv.

``Settings`` declares ``env_file=".env"`` (CWD-relative), so running the unit
suite from a repo root that carries a live installer-generated ``.env`` would
leak operator values (passwords, ports) into assertions that expect defaults.
Null the dotenv source for every unit test; explicit ``_env_file`` arguments
and real process env vars still win, matching CI behavior.
"""

import pytest

from voxint.config import Settings


@pytest.fixture(autouse=True)
def _no_operator_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
