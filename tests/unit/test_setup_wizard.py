"""Pure logic for the first-run setup wizard: step model + field validation.

No database and no app — these exercise the normalization/validation primitives
and the LLM-enable guard directly. The scan walk (which needs a Session for the
net-new query) is covered in the integration suite.
"""

from pathlib import Path

import pytest

from voxint.api.setup_wizard import (
    MAX_LLM_KEY_CHARS,
    MAX_MEDIA_FOLDERS,
    MAX_VOCABULARY_TERM_CHARS,
    MAX_VOCABULARY_TERMS,
    SetupValidationError,
    WizardStep,
    next_step,
    normalize_llm_api_key,
    normalize_llm_base_url,
    normalize_llm_model,
    normalize_media_folders,
    normalize_vocabulary,
    parse_step,
    validate_llm_enable,
)
from voxint.config import Settings


def _settings(**overrides: object) -> Settings:
    # _env_file=None so a stray dev .env can't bleed into a unit test.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# ------------------------------------------------------------------ step model


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("welcome", WizardStep.WELCOME),
        ("media", WizardStep.MEDIA),
        ("finish", WizardStep.FINISH),
        (None, WizardStep.WELCOME),
        ("", WizardStep.WELCOME),
        ("bogus", WizardStep.WELCOME),  # unknown → welcome, never a 422
    ],
)
def test_parse_step(raw: str | None, expected: WizardStep) -> None:
    assert parse_step(raw) is expected


def test_next_step_advances_and_saturates_at_finish() -> None:
    assert next_step(WizardStep.WELCOME) is WizardStep.MEDIA
    assert next_step(WizardStep.SERVICES) is WizardStep.FINISH
    assert next_step(WizardStep.FINISH) is WizardStep.FINISH  # fixed point


# ------------------------------------------------------------- media folders


def test_normalize_media_folders_relative_dirs_deduped(tmp_path: Path) -> None:
    (tmp_path / "podcasts").mkdir()
    (tmp_path / "interviews" / "2026").mkdir(parents=True)
    out = normalize_media_folders(
        [" podcasts ", "interviews/2026", "podcasts", "", "."], tmp_path
    )
    # "." is the media root itself (allowed); order preserved, dupes dropped.
    assert out == ["podcasts", "interviews/2026", "."]


def test_normalize_media_folders_rejects_absolute(tmp_path: Path) -> None:
    with pytest.raises(SetupValidationError):
        normalize_media_folders(["/etc"], tmp_path)


def test_normalize_media_folders_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(SetupValidationError):
        normalize_media_folders(["../escape"], tmp_path)


def test_normalize_media_folders_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(SetupValidationError):
        normalize_media_folders(["nope"], tmp_path)


@pytest.mark.parametrize("reserved", ["incoming", "artifacts", "incoming/sub"])
def test_normalize_media_folders_rejects_reserved_tree(
    tmp_path: Path, reserved: str
) -> None:
    # incoming/ and artifacts/ are Voxint-owned; registering them (or a subfolder)
    # would re-ingest the pipeline's own uploads/outputs.
    (tmp_path / reserved).mkdir(parents=True)
    with pytest.raises(SetupValidationError, match="reserved"):
        normalize_media_folders([reserved], tmp_path)


def test_normalize_media_folders_rejects_file(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    with pytest.raises(SetupValidationError):
        normalize_media_folders(["a.wav"], tmp_path)


def test_normalize_media_folders_caps_count(tmp_path: Path) -> None:
    for i in range(MAX_MEDIA_FOLDERS + 1):
        (tmp_path / f"d{i}").mkdir()
    with pytest.raises(SetupValidationError):
        normalize_media_folders(
            [f"d{i}" for i in range(MAX_MEDIA_FOLDERS + 1)], tmp_path
        )


# ---------------------------------------------------------------- vocabulary


def test_normalize_vocabulary_splits_strips_dedups() -> None:
    raw = "Ductless mini-split\n NUCA \n\nNUCA\nDr. Rosen\n"
    assert normalize_vocabulary(raw) == ["Ductless mini-split", "NUCA", "Dr. Rosen"]


def test_normalize_vocabulary_rejects_overlong_term() -> None:
    with pytest.raises(SetupValidationError):
        normalize_vocabulary("x" * (MAX_VOCABULARY_TERM_CHARS + 1))


def test_normalize_vocabulary_caps_count() -> None:
    with pytest.raises(SetupValidationError):
        normalize_vocabulary("\n".join(f"t{i}" for i in range(MAX_VOCABULARY_TERMS + 1)))


def test_normalize_vocabulary_empty_is_empty_list() -> None:
    assert normalize_vocabulary("   \n\n ") == []


# -------------------------------------------------------------- llm fields


@pytest.mark.parametrize("raw", ["", "   "])
def test_normalize_llm_base_url_blank_is_none(raw: str) -> None:
    assert normalize_llm_base_url(raw) is None


def test_normalize_llm_base_url_accepts_localhost() -> None:
    # A local inference endpoint is legitimate (unlike ingest URLs — no SSRF gate).
    assert normalize_llm_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


@pytest.mark.parametrize(
    "bad",
    [
        "ftp://host/v1",
        "not-a-url",
        "http://user:pw@host/v1",  # embedded credentials
        "http:// space /v1",  # whitespace
        "http:///v1",  # scheme but no host
        "http://[::1/v1",  # unclosed IPv6 bracket → urlsplit ValueError
        "http://host:notaport/v1",  # bad :port — must fail here, not in httpx later
        "http://host:99999999/v1",  # out-of-range port
        "https://host/" + "a" * 3000,  # over the length ceiling
    ],
)
def test_normalize_llm_base_url_rejects_bad(bad: str) -> None:
    with pytest.raises(SetupValidationError):
        normalize_llm_base_url(bad)


def test_normalize_llm_model_blank_is_none() -> None:
    assert normalize_llm_model("  ") is None
    assert normalize_llm_model(" gpt-4o-mini ") == "gpt-4o-mini"


def test_normalize_llm_model_rejects_overlong() -> None:
    with pytest.raises(SetupValidationError):
        normalize_llm_model("m" * 5000)


# ------------------------------------------------------------ llm api key field


def test_normalize_llm_api_key_blank_is_none() -> None:
    # Blank = no-change sentinel (the password field is never prefilled, so an empty
    # submission must leave the stored key untouched — distinct from removal).
    assert normalize_llm_api_key("") is None
    assert normalize_llm_api_key("   ") is None


def test_normalize_llm_api_key_strips_surrounding_whitespace() -> None:
    assert normalize_llm_api_key("  sk-test-123  ") == "sk-test-123"


def test_normalize_llm_api_key_passthrough() -> None:
    assert normalize_llm_api_key("sk-abc_DEF-123") == "sk-abc_DEF-123"


@pytest.mark.parametrize("bad", ["sk test", "sk\tabc", "sk\nabc", "sk\x00abc", "sk\x7fabc"])
def test_normalize_llm_api_key_rejects_inner_whitespace_or_control(bad: str) -> None:
    with pytest.raises(SetupValidationError, match="whitespace or control"):
        normalize_llm_api_key(bad)


@pytest.mark.parametrize("bad", ["sk-héllo", "sk-café", "sk-日本", "sk-\U0001f600"])
def test_normalize_llm_api_key_rejects_non_ascii(bad: str) -> None:
    # A non-ASCII key would crash httpx's latin-1 Authorization-header encoding at
    # run/doctor time; reject it at save so enablement fails closed with a message.
    with pytest.raises(SetupValidationError, match="printable ASCII"):
        normalize_llm_api_key(bad)


def test_normalize_llm_api_key_rejects_overlong() -> None:
    with pytest.raises(SetupValidationError, match=str(MAX_LLM_KEY_CHARS)):
        normalize_llm_api_key("s" * (MAX_LLM_KEY_CHARS + 1))


def test_normalize_llm_api_key_at_cap_passes() -> None:
    key = "s" * MAX_LLM_KEY_CHARS
    assert normalize_llm_api_key(key) == key


def test_normalize_llm_api_key_message_never_echoes_value() -> None:
    # A validation message must never carry the submitted secret.
    secret = "sk-super-secret-value with a space"
    try:
        normalize_llm_api_key(secret)
    except SetupValidationError as exc:
        assert "sk-super-secret" not in str(exc)
    else:  # pragma: no cover - the value has a space, so it must raise
        raise AssertionError("expected rejection")


# ------------------------------------------------------------- llm enable guard


def test_validate_llm_enable_ok_with_key_and_fitting_budget() -> None:
    # The effective key is passed in (row-wins-over-env resolved by the caller); the
    # env key on settings is irrelevant to the presence check now.
    validate_llm_enable("sk-effective", _settings(llm_api_key=""))  # does not raise


def test_validate_llm_enable_rejects_missing_key() -> None:
    # Empty effective key (no row, no env) → refuse to enable.
    with pytest.raises(SetupValidationError, match="No LLM API key"):
        validate_llm_enable("", _settings(llm_api_key=""))


def test_validate_llm_enable_rejects_budget_over_lease() -> None:
    # llm_enabled stays False so the env-time validator doesn't fire on construction;
    # the wizard guard is what must catch the over-lease budget at enable time. A
    # present effective key gets past the presence check so the budget guard fires.
    settings = _settings(
        llm_api_key="sk-test",
        llm_enabled=False,
        llm_run_budget_seconds=999999.0,
        stage_lease_seconds=21600,
    )
    with pytest.raises(SetupValidationError, match="lease"):
        validate_llm_enable("sk-effective", settings)
