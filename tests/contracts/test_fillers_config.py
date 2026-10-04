"""Versioned English preset contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from voxint.config import Settings, SettingsError, get_settings
from voxint.export.filler_lists import NEVER, PRESET_VERSION, TIER_1, TIER_2, normalize_entries


def test_preset_literals() -> None:
    assert PRESET_VERSION == "en-1"
    assert TIER_1 == ("um", "uh", "umm", "uhh", "uhm", "erm")
    assert TIER_2 == (
        "hm",
        "mm",
        "mmm",
        "you know",
        "kind of",
        "sort of",
        "I mean",
        "I guess",
        "I suppose",
        "or something",
        "you know what I mean",
        "you see",
    )
    assert NEVER == ("like", "so", "right", "well")


def test_disjoint_valid_presets() -> None:
    first, second, never = (
        {entry.casefold() for entry in tier} for tier in (TIER_1, TIER_2, NEVER)
    )
    assert first.isdisjoint(second)
    assert never.isdisjoint(first | second)
    for entry in (*TIER_1, *TIER_2):
        assert normalize_entries([entry]) == (entry,)


def test_environment_documented() -> None:
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    assert "# VOXINT_FILLERS_ADD=" in example
    assert "# VOXINT_FILLERS_KEEP=" in example


@pytest.mark.parametrize("name", ["voxint_fillers_add", "voxint_fillers_keep"])
@pytest.mark.parametrize(
    "invalid", ["private-value-123", "six words are too many per entry", "x" * 41]
)
def test_invalid_environment_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    invalid: str,
) -> None:
    monkeypatch.chdir(tmp_path)  # Never read the developer's .env.
    with pytest.raises(ValidationError) as direct:
        Settings(_env_file=None, **{name: invalid})
    message = direct.value.errors(include_input=False)[0]["msg"]
    assert name.upper() in message
    assert invalid not in message
    assert "Use " in message
    monkeypatch.setenv(name.upper(), invalid)
    with pytest.raises(SettingsError) as sanitized:
        get_settings()
    assert name in str(sanitized.value)
    assert invalid not in str(sanitized.value)


def test_valid_environment_preserves_raw_and_normalizes_at_resolution() -> None:
    from voxint.app_settings import resolve_effective_filler_list

    raw = " You   know, you know, ,I mean "
    settings = Settings(_env_file=None, voxint_fillers_add=raw, voxint_fillers_keep=" UM, um")
    assert settings.voxint_fillers_add == raw
    effective = resolve_effective_filler_list(None, settings)
    assert effective.added == ("You know", "I mean")
    assert effective.kept == ("UM",)


def test_cli_invalid_filler_environment_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from voxint.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOXINT_FILLERS_ADD", "private-value-123")
    assert (
        main(["export", "00000000-0000-0000-0000-000000000001", "--format", "md", "--drop-fillers"])
        == 2
    )
    output = capsys.readouterr().out
    assert "voxint_fillers_add" in output
    assert "private-value-123" not in output
