"""Tri-state feature-flag resolvers + the effective-flag validator (issue #74).

These pin the keystone contract of the #47 settings arc: every in-UI-editable
flag resolves DB-row-wins-over-env, an all-NULL row is byte-for-byte identical to
"no row" (the load-bearing NULL-parity invariant), the web-research provider
config threads through one frozen value object, and the five cross-flag
invariants live in exactly one validator shared with ``config.py``.
"""

from voxint.app_settings import (
    EffectiveFlags,
    feature_flag_state,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_enrichment_names_llm_enabled,
    resolve_effective_enrichment_run_assets_autogenerate,
    resolve_effective_enrichment_run_assets_enabled,
    resolve_effective_enrichment_web_research_enabled,
    resolve_effective_source_authority_domains,
    resolve_effective_voxint_web_research,
    resolve_effective_web_research,
    resolve_effective_web_search_api_key,
    resolve_effective_web_search_base_url,
    resolve_effective_ytdlp_enabled,
    validate_effective_flags,
)
from voxint.config import Settings
from voxint.db.models import AppSettings

_BASE_URL = "http://searx.lan:8888"

# (resolver, column/config name, env value used for the "inherit" assertions)
_BOOL_RESOLVERS = (
    (resolve_effective_enrichment_names_enabled, "enrichment_names_enabled"),
    (resolve_effective_enrichment_names_llm_enabled, "enrichment_names_llm_enabled"),
    (resolve_effective_enrichment_run_assets_enabled, "enrichment_run_assets_enabled"),
    (resolve_effective_enrichment_run_assets_autogenerate, "enrichment_run_assets_autogenerate"),
    (resolve_effective_voxint_web_research, "voxint_web_research"),
    (resolve_effective_enrichment_web_research_enabled, "enrichment_web_research_enabled"),
    (resolve_effective_ytdlp_enabled, "ytdlp_enabled"),
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _env_all_on() -> Settings:
    """A fully-enabled, invariant-valid env (every #74 flag True)."""
    return _settings(
        llm_enabled=True,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=True,
        enrichment_run_assets_enabled=True,
        enrichment_run_assets_autogenerate=True,
        voxint_web_research=True,
        enrichment_web_research_enabled=True,
        ytdlp_enabled=True,
        web_search_base_url=_BASE_URL,
    )


def _env_all_off() -> Settings:
    """A fully-disabled, invariant-valid env (every #74 flag False) — set
    explicitly because some flags (e.g. enrichment_names_enabled) default True."""
    return _settings(
        llm_enabled=False,
        enrichment_names_enabled=False,
        enrichment_names_llm_enabled=False,
        enrichment_run_assets_enabled=False,
        enrichment_run_assets_autogenerate=False,
        voxint_web_research=False,
        enrichment_web_research_enabled=False,
        ytdlp_enabled=False,
    )


# ----------------------------------------------------------- boolean resolvers
# Env-time cross-flag validation forbids building a Settings with a single
# dependent flag on, so precedence is exercised against a fully-valid all-on env
# and the default all-off env; the ROW (a plain ORM object) carries the single
# overridden flag freely.


def test_bool_none_row_inherits_env() -> None:
    on, off = _env_all_on(), _env_all_off()
    for resolver, name in _BOOL_RESOLVERS:
        assert resolver(None, on) is True, name
        assert resolver(None, off) is False, name


def test_bool_null_column_inherits_env() -> None:
    # An existing row that never set the column reads NULL ⇒ env still governs.
    on = _env_all_on()
    for resolver, name in _BOOL_RESOLVERS:
        row = AppSettings(id=1)  # every #74 column defaults NULL
        assert getattr(row, name) is None, name
        assert resolver(row, on) is True, name


def test_bool_row_true_wins_over_env_false() -> None:
    off = _env_all_off()
    for resolver, name in _BOOL_RESOLVERS:
        row = AppSettings(id=1, **{name: True})
        assert resolver(row, off) is True, name


def test_bool_row_false_wins_over_env_true() -> None:
    # A UI disable must win over an env enable — the whole point of the arc.
    on = _env_all_on()
    for resolver, name in _BOOL_RESOLVERS:
        row = AppSettings(id=1, **{name: False})
        assert resolver(row, on) is False, name


# ------------------------------------------------------------ string resolvers


def test_str_none_or_blank_row_inherits_env() -> None:
    env = _settings(source_authority_domains="ex.example,gov.example")
    assert (
        resolve_effective_source_authority_domains(None, env) == "ex.example,gov.example"
    )
    blank = AppSettings(id=1, source_authority_domains="   ")
    assert (
        resolve_effective_source_authority_domains(blank, env) == "ex.example,gov.example"
    )


def test_str_non_blank_row_wins() -> None:
    env = _settings(source_authority_domains="env.example")
    row = AppSettings(id=1, source_authority_domains="row.example")
    assert resolve_effective_source_authority_domains(row, env) == "row.example"


def test_web_search_base_url_precedence() -> None:
    env = _settings(web_search_base_url="http://env.lan:8888")
    assert resolve_effective_web_search_base_url(None, env) == "http://env.lan:8888"
    row = AppSettings(id=1, web_search_base_url="http://row.lan:8888")
    assert resolve_effective_web_search_base_url(row, env) == "http://row.lan:8888"


def test_web_search_api_key_precedence_and_secrecy() -> None:
    env = _settings(web_search_api_key="k-env")
    assert resolve_effective_web_search_api_key(None, env) == "k-env"
    row = AppSettings(id=1, web_search_api_key="k-row-super-secret")
    assert resolve_effective_web_search_api_key(row, env) == "k-row-super-secret"
    # The secret must never surface through the row's repr/str (no custom __repr__).
    assert "k-row-super-secret" not in repr(row)
    assert "k-row-super-secret" not in str(row)


# ----------------------------------------------------- EffectiveWebResearch build


def test_web_research_enable_row_wins_over_env_disable() -> None:
    # The precedence bug the consult caught: env=false + row=true must ENABLE.
    env = _settings(
        voxint_web_research=False,
        web_search_base_url="http://env.lan:8888",
        web_search_api_key="k-env",
    )
    row = AppSettings(
        id=1,
        voxint_web_research=True,
        web_search_base_url="http://row.lan:8888",
        web_search_api_key="k-row",
    )
    effective = resolve_effective_web_research(row, env)
    assert effective.enabled is True
    assert effective.base_url == "http://row.lan:8888"
    assert effective.api_key == "k-row"


def test_effective_web_research_repr_hides_api_key() -> None:
    # The VO is threaded worker->agent->tools; its auto-repr must NOT expose the
    # credential (field(repr=False)) — a stray %r/log/assertion diff can't leak it.
    effective = resolve_effective_web_research(
        AppSettings(id=1, voxint_web_research=True, web_search_api_key="k-super-secret"),
        _settings(web_search_base_url=_BASE_URL),
    )
    assert effective.api_key == "k-super-secret"  # still resolves for real use
    assert "k-super-secret" not in repr(effective)
    assert "k-super-secret" not in str(effective)


def test_web_research_none_row_is_pure_env() -> None:
    env = _settings(
        voxint_web_research=True,
        web_search_base_url="http://env.lan:8888",
        web_search_api_key="k-env",
    )
    effective = resolve_effective_web_research(None, env)
    assert effective.enabled is True
    assert effective.base_url == "http://env.lan:8888"
    assert effective.api_key == "k-env"


# --------------------------------------------------------------- NULL-parity


def test_all_null_row_is_identical_to_no_row() -> None:
    # The load-bearing safety invariant: an app_settings row with every #74
    # column NULL resolves byte-for-byte the same as having no row at all.
    env = _settings(
        llm_enabled=True,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=False,
        enrichment_run_assets_enabled=True,
        enrichment_run_assets_autogenerate=False,
        voxint_web_research=True,
        enrichment_web_research_enabled=False,
        ytdlp_enabled=True,
        source_authority_domains="a.example,b.example",
        web_search_base_url="http://env.lan:8888",
        web_search_api_key="k-env",
    )
    null_row = AppSettings(id=1)
    for resolver, _name in _BOOL_RESOLVERS:
        assert resolver(null_row, env) == resolver(None, env)
    assert resolve_effective_source_authority_domains(
        null_row, env
    ) == resolve_effective_source_authority_domains(None, env)
    assert resolve_effective_web_research(null_row, env) == resolve_effective_web_research(
        None, env
    )


# ------------------------------------------------------ validate_effective_flags


def _flags(**overrides: object) -> EffectiveFlags:
    base: dict[str, object] = {
        "llm_enabled": False,
        "enrichment_names_enabled": False,
        "enrichment_names_llm_enabled": False,
        "enrichment_run_assets_enabled": False,
        "enrichment_run_assets_autogenerate": False,
        "voxint_web_research": False,
        "enrichment_web_research_enabled": False,
        "web_search_base_url": "",
    }
    base.update(overrides)
    return EffectiveFlags(**base)  # type: ignore[arg-type]


def test_valid_all_off_has_no_errors() -> None:
    assert validate_effective_flags(_flags()) == []


def test_valid_fully_enabled_stack_has_no_errors() -> None:
    assert (
        validate_effective_flags(
            _flags(
                llm_enabled=True,
                enrichment_names_enabled=True,
                enrichment_names_llm_enabled=True,
                enrichment_run_assets_enabled=True,
                enrichment_run_assets_autogenerate=True,
                voxint_web_research=True,
                enrichment_web_research_enabled=True,
                web_search_base_url=_BASE_URL,
            )
        )
        == []
    )


def test_names_llm_requires_llm() -> None:
    errors = validate_effective_flags(
        _flags(enrichment_names_enabled=True, enrichment_names_llm_enabled=True)
    )
    assert errors == [
        "enrichment_names_llm_enabled requires llm_enabled=true — the "
        "LLM name pass reuses the configured enhancement endpoint"
    ]


def test_names_llm_requires_names() -> None:
    errors = validate_effective_flags(
        _flags(llm_enabled=True, enrichment_names_llm_enabled=True)
    )
    assert errors == [
        "enrichment_names_llm_enabled requires enrichment_names_enabled=true"
        " — the LLM pass is additive to the offline name producer"
    ]


def test_web_research_producer_requires_retrieval() -> None:
    errors = validate_effective_flags(
        _flags(llm_enabled=True, enrichment_web_research_enabled=True)
    )
    assert errors == [
        "enrichment_web_research_enabled requires voxint_web_research=true"
        " — the producer's only egress is the controlled retrieval tools"
    ]


def test_web_research_producer_requires_llm() -> None:
    errors = validate_effective_flags(
        _flags(
            voxint_web_research=True,
            web_search_base_url=_BASE_URL,
            enrichment_web_research_enabled=True,
        )
    )
    assert errors == [
        "enrichment_web_research_enabled requires llm_enabled=true — the"
        " producer reuses the configured enhancement endpoint"
    ]


def test_run_assets_requires_llm() -> None:
    errors = validate_effective_flags(_flags(enrichment_run_assets_enabled=True))
    assert errors == [
        "enrichment_run_assets_enabled requires llm_enabled=true — the"
        " asset generators reuse the configured enhancement endpoint"
    ]


def test_run_assets_autogenerate_requires_run_assets() -> None:
    errors = validate_effective_flags(
        _flags(llm_enabled=True, enrichment_run_assets_autogenerate=True)
    )
    assert errors == [
        "enrichment_run_assets_autogenerate requires"
        " enrichment_run_assets_enabled=true — the post-finalize step"
        " only enqueues the feature it rides on"
    ]


def test_web_research_requires_valid_base_url() -> None:
    errors = validate_effective_flags(
        _flags(voxint_web_research=True, web_search_base_url="")
    )
    assert errors == [
        "voxint_web_research=true requires web_search_base_url — the"
        " searxng provider has no default endpoint"
    ]


def test_feature_flag_state_tristate() -> None:
    # The settings form (issue #62) renders the RAW column tri-state, not the
    # resolved effective value: NULL / no row => "inherit", True => "on", False
    # => "off". This is what keeps an untouched "use installation setting" save
    # writing NULL instead of pinning the env default onto the row.
    assert feature_flag_state(None, "ytdlp_enabled") == "inherit"
    assert feature_flag_state(AppSettings(id=1), "ytdlp_enabled") == "inherit"
    assert feature_flag_state(AppSettings(id=1, ytdlp_enabled=True), "ytdlp_enabled") == "on"
    assert feature_flag_state(AppSettings(id=1, ytdlp_enabled=False), "ytdlp_enabled") == "off"
