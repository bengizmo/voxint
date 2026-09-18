"""Settings badges compare saved choices with installation defaults (#550)."""

import pytest

from voxint.api.routers.deps import templates
from voxint.app_settings import feature_flag_state
from voxint.config import Settings
from voxint.db.models import AppSettings


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(
    ("env_default", "saved", "changed"),
    [
        (False, None, False),
        (True, None, False),
        (False, False, False),
        (True, True, False),
        (False, True, True),
        (True, False, True),
    ],
)
def test_switch_changed_badge(
    monkeypatch: pytest.MonkeyPatch,
    env_default: bool,
    saved: bool | None,
    changed: bool,
    disabled: bool,
) -> None:
    monkeypatch.setenv("YTDLP_ENABLED", str(env_default).lower())
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    row = AppSettings(id=1, ytdlp_enabled=saved)
    state = feature_flag_state(row, "ytdlp_enabled")
    effective = settings.ytdlp_enabled if saved is None else saved
    # Dependency gating can display an off switch without changing its saved choice.
    body = templates.env.get_template("settings/_switch.html").module.tri_switch(  # type: ignore[attr-defined]
        name="ytdlp_enabled",
        label="URL ingestion",
        help_text="Download media from URLs.",
        state=state,
        effective=effective and not disabled,
        env_default=settings.ytdlp_enabled,
        disabled=disabled,
    )

    assert ('class="switch-badge"' in body) is changed
    # A pinned value can still be reset to inheritance when it matches the default.
    assert ('name="reset_flag"' in body) is (saved is not None)
