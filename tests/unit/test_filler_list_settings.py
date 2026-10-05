"""Independent saved-list and environment precedence, without a database."""

from typing import Any

import pytest

from voxint.app_settings import resolve_effective_filler_list
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.export.filler_lists import FillerListError

SAVED_INVALID = "The saved filler word list is not valid. Save it again in settings."


@pytest.mark.parametrize("add", [None, {}, {"fr": ["bonjour"]}, {"en": []}, {"en": ["You know"]}])
@pytest.mark.parametrize("keep", [None, {}, {"en": []}, {"en": ["UM"]}])
@pytest.mark.parametrize("env", [False, True])
def test_independent_precedence(
    add: dict[str, Any] | None,
    keep: dict[str, Any] | None,
    env: bool,
) -> None:
    settings = Settings(
        _env_file=None,
        voxint_fillers_add="I mean" if env else " ",
        voxint_fillers_keep="uh" if env else "",
    )
    result = resolve_effective_filler_list(
        AppSettings(fillers_add=add, fillers_keep=keep), settings
    )
    overridden = []
    for name, stored, fallback, actual, source in (
        ("add", add, "I mean", result.added, result.add_source),
        ("keep", keep, "uh", result.kept, result.keep_source),
    ):
        wins = stored is not None and "en" in stored
        assert actual == (tuple(stored["en"]) if wins else (fallback,) if env else ())
        assert source == ("settings" if wins else "environment" if env else "none")
        if wins and env:
            overridden.append(name)
    assert result.env_overridden == tuple(overridden)
    assert ("um" in result.words) == ("UM" not in result.kept)


def test_missing_row_and_normalization() -> None:
    settings = Settings(
        _env_file=None,
        voxint_fillers_add=" You   know, you know, weiß ",
        voxint_fillers_keep=" UM,um ",
    )
    result = resolve_effective_filler_list(None, settings)
    assert result.added == ("You know", "weiß")
    assert result.kept == ("UM",)
    assert result.phrases == ("You know",)
    assert "weiß" in result.words
    assert "um" not in result.words
    assert result.add_source == result.keep_source == "environment"
    assert result.env_overridden == ()
    default = resolve_effective_filler_list(
        None, Settings(_env_file=None, voxint_fillers_add="", voxint_fillers_keep="")
    )
    assert default.is_default
    assert default.add_source == default.keep_source == "none"


@pytest.mark.parametrize(
    "stored",
    [
        {"en": "um"},
        {"en": None},
        {"en": ["you know", 1]},
        {"en": {"bad": "entry"}},
        ["you know"],
        "you know",
    ],
)
@pytest.mark.parametrize("name", ["add", "keep"])
def test_malformed_saved_list_is_refused(stored: Any, name: str) -> None:
    settings = Settings(_env_file=None, voxint_fillers_add="", voxint_fillers_keep="")
    row = AppSettings(**{f"fillers_{name}": stored})
    with pytest.raises(FillerListError) as refused:
        resolve_effective_filler_list(row, settings)
    assert str(refused.value) == SAVED_INVALID


def test_invalid_saved_entry_names_the_rule_not_the_value() -> None:
    settings = Settings(_env_file=None, voxint_fillers_add="", voxint_fillers_keep="")
    row = AppSettings(fillers_add={"en": ["private-value-123"]})
    with pytest.raises(FillerListError) as refused:
        resolve_effective_filler_list(row, settings)
    message = str(refused.value)
    assert message.startswith(f"{SAVED_INVALID} ")
    assert "Use letters" in message
    assert "private-value-123" not in message


@pytest.mark.parametrize("blank", [",,", " , ,", ","])
def test_separator_only_environment_is_unset(blank: str) -> None:
    settings = Settings(_env_file=None, voxint_fillers_add=blank, voxint_fillers_keep=blank)
    result = resolve_effective_filler_list(None, settings)
    assert result.add_source == result.keep_source == "none"
    assert result.is_default
    saved = resolve_effective_filler_list(
        AppSettings(fillers_add={"en": ["you know"]}, fillers_keep={"en": []}), settings
    )
    assert saved.add_source == saved.keep_source == "settings"
    assert saved.env_overridden == ()
