"""Pure precedence resolvers for the in-UI LLM API key (issue #10).

``resolve_effective_llm_api_key`` / ``resolve_effective_llm_endpoint`` /
``effective_llm_key_source`` are the single source of the DB-row-wins-over-env
rule threaded through every LLM client construction. These tests pin the
precedence + canonical-stripping contract the rest of the feature relies on.
"""

from voxint.app_settings import (
    effective_llm_key_source,
    resolve_effective_llm_api_key,
    resolve_effective_llm_endpoint,
)
from voxint.config import Settings
from voxint.db.models import AppSettings


def _settings(
    *,
    llm_api_key: str = "",
    llm_base_url: str = "https://env.example/v1",
    llm_model: str = "env-model",
) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


# ----------------------------------------------- resolve_effective_llm_api_key


def test_key_none_row_falls_back_to_env() -> None:
    assert resolve_effective_llm_api_key(None, _settings(llm_api_key="sk-env")) == "sk-env"


def test_key_blank_row_falls_back_to_env() -> None:
    row = AppSettings(id=1, llm_api_key="   ")
    assert resolve_effective_llm_api_key(row, _settings(llm_api_key="sk-env")) == "sk-env"


def test_key_row_wins_over_env() -> None:
    row = AppSettings(id=1, llm_api_key="sk-row")
    assert resolve_effective_llm_api_key(row, _settings(llm_api_key="sk-env")) == "sk-row"


def test_key_row_value_is_stripped() -> None:
    row = AppSettings(id=1, llm_api_key="  sk-row  ")
    assert resolve_effective_llm_api_key(row, _settings(llm_api_key="sk-env")) == "sk-row"


def test_key_env_value_is_stripped() -> None:
    assert resolve_effective_llm_api_key(None, _settings(llm_api_key="  sk-env  ")) == "sk-env"


def test_key_none_anywhere_is_empty_string() -> None:
    assert resolve_effective_llm_api_key(None, _settings(llm_api_key="")) == ""
    row = AppSettings(id=1, llm_api_key=None)
    assert resolve_effective_llm_api_key(row, _settings(llm_api_key="")) == ""


# ---------------------------------------------- resolve_effective_llm_endpoint


def test_endpoint_none_row_uses_env() -> None:
    base, model = resolve_effective_llm_endpoint(None, _settings())
    assert base == "https://env.example/v1"
    assert model == "env-model"


def test_endpoint_row_wins_when_set() -> None:
    row = AppSettings(id=1, llm_base_url="https://row.example/v1", llm_model="row-model")
    base, model = resolve_effective_llm_endpoint(row, _settings())
    assert base == "https://row.example/v1"
    assert model == "row-model"


def test_endpoint_null_row_fields_fall_back_per_field() -> None:
    row = AppSettings(id=1, llm_base_url="https://row.example/v1", llm_model=None)
    base, model = resolve_effective_llm_endpoint(row, _settings())
    assert base == "https://row.example/v1"
    assert model == "env-model"  # NULL model falls back independently


# ------------------------------------------------- effective_llm_key_source


def test_source_stored_when_row_non_blank() -> None:
    row = AppSettings(id=1, llm_api_key="sk-row")
    assert effective_llm_key_source(row, _settings(llm_api_key="sk-env")) == "stored"


def test_source_environment_when_only_env() -> None:
    row = AppSettings(id=1, llm_api_key="   ")  # blank row does not count as stored
    assert effective_llm_key_source(row, _settings(llm_api_key="sk-env")) == "environment"
    assert effective_llm_key_source(None, _settings(llm_api_key="sk-env")) == "environment"


def test_source_none_when_neither() -> None:
    assert effective_llm_key_source(None, _settings(llm_api_key="")) == "none"
    row = AppSettings(id=1, llm_api_key="")
    assert effective_llm_key_source(row, _settings(llm_api_key="")) == "none"


# --------------------------------------------------------------- secret absence


def test_appsettings_repr_never_leaks_the_key() -> None:
    # AppSettings defines no custom __repr__, so the default shows only class + PK;
    # this pins that a stored key can't surface through repr()/str().
    row = AppSettings(id=1, llm_api_key="sk-super-secret-repr")
    assert "sk-super-secret-repr" not in repr(row)
    assert "sk-super-secret-repr" not in str(row)
