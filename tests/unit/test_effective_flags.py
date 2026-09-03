"""Tri-state feature-flag resolvers + the effective-flag validator (issue #74).

These pin the keystone contract of the #47 settings arc: every in-UI-editable
flag resolves DB-row-wins-over-env, an all-NULL row is byte-for-byte identical to
"no row" (the load-bearing NULL-parity invariant), the web-research provider
config threads through one frozen value object, and the five cross-flag
invariants live in exactly one validator shared with ``config.py``.
"""

from voxint.app_settings import (
    EffectiveFlags,
    build_bundled_llm_client,
    byo_llm_configured,
    effective_web_search_key_source,
    feature_flag_state,
    llm_bundled_active,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_enrichment_names_llm_enabled,
    resolve_effective_enrichment_run_assets_autogenerate,
    resolve_effective_enrichment_run_assets_enabled,
    resolve_effective_enrichment_web_research_enabled,
    resolve_effective_llm_bundled_enabled,
    resolve_effective_source_authority_domains,
    resolve_effective_voxint_web_research,
    resolve_effective_web_research,
    resolve_effective_web_search_api_key,
    resolve_effective_web_search_base_url,
    resolve_effective_ytdlp_enabled,
    str_flag_form_field,
    validate_effective_flags,
)
from voxint.config import DEFAULT_LLM_BASE_URL, Settings
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
    (resolve_effective_llm_bundled_enabled, "llm_bundled_enabled"),
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
        llm_bundled_enabled=True,
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
        llm_bundled_enabled=False,
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


def test_str_flag_form_field_renders_raw_column(monkeypatch) -> None:
    # The string form helper renders the ROW override, or "" when NULL/blank (so an
    # untouched save stays inheriting), with the env default as the placeholder —
    # the string counterpart to feature_flag_state (issue #76).
    monkeypatch.setenv("WEB_SEARCH_BASE_URL", "http://env.example")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert str_flag_form_field(None, settings, "web_search_base_url") == ("", "http://env.example")
    row_blank = AppSettings(id=1, web_search_base_url="")
    assert str_flag_form_field(row_blank, settings, "web_search_base_url") == (
        "",
        "http://env.example",
    )
    row = AppSettings(id=1, web_search_base_url="http://row.lan")
    assert str_flag_form_field(row, settings, "web_search_base_url") == (
        "http://row.lan",
        "http://env.example",
    )


# ---------------------------------------------------- bundled local LLM (#67)


def test_llm_bundled_active_needs_flag_and_url_and_model() -> None:
    # The routing predicate is AND(effective flag, a configured bundled URL, a
    # configured bundled model): the compose-injected URL+model are what make the
    # bundle exist, so the flag is inert without them (a fresh install with the
    # toggle on but no compose.llm.yaml). Both endpoint constants are required so
    # the client never POSTs "model": "" to a single-model llama-server (#67).
    model = "qwen3-4b-instruct-2507"
    url = "http://voxint-llm:8080/v1"
    no_url = _settings(llm_bundled_enabled=True, llm_bundled_base_url="", llm_bundled_model=model)
    assert llm_bundled_active(None, no_url) is False
    no_model = _settings(llm_bundled_enabled=True, llm_bundled_base_url=url, llm_bundled_model="")
    assert llm_bundled_active(None, no_model) is False
    on_url = _settings(llm_bundled_enabled=True, llm_bundled_base_url=url, llm_bundled_model=model)
    assert llm_bundled_active(None, on_url) is True
    off = _settings(llm_bundled_enabled=False, llm_bundled_base_url=url, llm_bundled_model=model)
    assert llm_bundled_active(None, off) is False


def test_llm_bundled_active_row_wins_over_env() -> None:
    # Tri-state: a UI enable (row True) activates the bundle even when the env
    # default is off, provided the bundled URL+model exist; a UI disable wins over
    # env on.
    url = "http://voxint-llm:8080/v1"
    model = "qwen3-4b-instruct-2507"
    eoff = _settings(llm_bundled_enabled=False, llm_bundled_base_url=url, llm_bundled_model=model)
    assert llm_bundled_active(AppSettings(id=1, llm_bundled_enabled=True), eoff) is True
    env_on = _settings(llm_bundled_enabled=True, llm_bundled_base_url=url, llm_bundled_model=model)
    assert llm_bundled_active(AppSettings(id=1, llm_bundled_enabled=False), env_on) is False


def test_byo_llm_configured_default_install_is_unconfigured() -> None:
    # The untouched install default (OpenAI public base URL, no key) is the "no BYO"
    # sentinel: a bundled install where the operator never pointed the BYO slot
    # anywhere has no endpoint to run topics on, so the topics gate must still fire.
    default = _settings(llm_enabled=True, llm_base_url=DEFAULT_LLM_BASE_URL, llm_api_key="")
    assert byo_llm_configured(None, default) is False


def test_byo_llm_configured_needs_enabled_and_endpoint() -> None:
    lan = "http://byo.lan:8100/v1"
    # LLM disabled ⇒ never configured, whatever the URL.
    off = _settings(llm_enabled=False, llm_base_url=lan)
    assert byo_llm_configured(None, off) is False
    # A blank model ⇒ not usable.
    no_model = _settings(llm_enabled=True, llm_base_url=lan, llm_model="")
    assert byo_llm_configured(None, no_model) is False


def test_byo_llm_configured_distinct_lan_endpoint_keyless() -> None:
    # The scoped-bundle-plus-separate-BYO case: a bundle serves enhancement while a
    # distinct, keyless LAN endpoint is configured for the BYO-only jobs. A
    # non-default base URL alone signals a deliberate endpoint (LAN models are
    # keyless), so topics can run there.
    s = _settings(
        llm_enabled=True,
        llm_base_url="http://byo.lan:8100/v1",
        llm_model="qwen3-27b",
        llm_api_key="",
        llm_bundled_enabled=True,
        llm_bundled_base_url="http://bundle.lan:8100/v1",
        llm_bundled_model="qwen3-27b",
    )
    assert byo_llm_configured(None, s) is True


def test_byo_llm_configured_key_at_default_url_counts() -> None:
    # A stored key at the default OpenAI URL is a real, deliberate BYO endpoint.
    keyed = _settings(llm_enabled=True, llm_base_url=DEFAULT_LLM_BASE_URL, llm_api_key="sk-real")
    assert byo_llm_configured(None, keyed) is True


def test_byo_llm_configured_false_when_byo_equals_bundle() -> None:
    # If the BYO endpoint IS the bundled endpoint, it is not a distinct place to run
    # topics — the bundle is the only endpoint, so the gate must still fire.
    same = _settings(
        llm_enabled=True,
        llm_base_url="http://bundle.lan:8100/v1",
        llm_model="m",
        llm_bundled_enabled=True,
        llm_bundled_base_url="http://bundle.lan:8100/v1",
        llm_bundled_model="m",
    )
    assert byo_llm_configured(None, same) is False


def test_byo_llm_configured_row_override_wins() -> None:
    # A UI-pinned BYO endpoint (row override) makes it configured even when the env
    # default is the untouched OpenAI placeholder.
    env_default = _settings(llm_enabled=True, llm_base_url=DEFAULT_LLM_BASE_URL, llm_api_key="")
    row = AppSettings(id=1, llm_enabled=True, llm_base_url="http://row.lan:8100/v1", llm_model="m")
    assert byo_llm_configured(row, env_default) is True


def test_build_bundled_llm_client_keyless_greedy() -> None:
    # The bundled endpoint is product-owned and local: NO api key, and the pinned
    # greedy SamplingProfile (byte-identical default) — never a leaked BYO key or
    # a Qwen-specific sampler on an arbitrary endpoint.
    settings = _settings(
        llm_bundled_base_url="http://voxint-llm:8080/v1",
        llm_bundled_model="qwen3-4b-instruct-2507",
    )
    client = build_bundled_llm_client(settings)
    try:
        assert client._model == "qwen3-4b-instruct-2507"
        assert client._api_key == ""
        assert client._sampling.as_payload() == {"temperature": 0}
    finally:
        client.close()


def test_effective_web_search_key_source(monkeypatch) -> None:
    # Honest UI status — never the value. Stored (row) wins, else environment, else
    # none; mirrors resolve_effective_web_search_api_key's precedence (issue #76).
    monkeypatch.delenv("WEB_SEARCH_API_KEY", raising=False)
    settings_no_env = Settings(_env_file=None)  # type: ignore[call-arg]
    assert effective_web_search_key_source(None, settings_no_env) == "none"
    assert (
        effective_web_search_key_source(
            AppSettings(id=1, web_search_api_key="k-row"), settings_no_env
        )
        == "stored"
    )
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "k-env")
    settings_env = Settings(_env_file=None)  # type: ignore[call-arg]
    assert effective_web_search_key_source(None, settings_env) == "environment"
    # A stored row key still wins the "stored" label over an env key.
    assert (
        effective_web_search_key_source(
            AppSettings(id=1, web_search_api_key="k-row"), settings_env
        )
        == "stored"
    )


# ---------------------------------------- _FEATURE_DEPS / validator drift guard


def test_feature_deps_match_validator_invariants() -> None:
    """Every Features-section dependency in ``_FEATURE_DEPS`` must also be
    enforced by ``validate_effective_flags`` (issue #406 drift guard).

    For each flag→dep pair, build a valid-except-for-this-dep combination
    (flag on, dep off) and assert the validator rejects it. A new invariant
    added to the validator without updating ``_FEATURE_DEPS`` (or vice versa)
    breaks this test.
    """
    from voxint.api.routers.settings import _DEP_LABELS, _FEATURE_DEPS

    # Build a fully-enabled baseline that passes validation.
    valid_base = _flags(
        llm_enabled=True,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=True,
        enrichment_run_assets_enabled=True,
        enrichment_run_assets_autogenerate=True,
        voxint_web_research=True,
        enrichment_web_research_enabled=True,
        web_search_base_url="http://searx.lan:8888",
    )
    assert validate_effective_flags(valid_base) == []

    for flag, deps in _FEATURE_DEPS.items():
        if not hasattr(valid_base, flag):
            continue
        for dep in deps:
            if not hasattr(valid_base, dep):
                continue
            # Flag on, dep off — validator must reject this.
            base_vals = {
                f.name: getattr(valid_base, f.name)
                for f in valid_base.__dataclass_fields__.values()
            }
            base_vals[flag] = True
            base_vals[dep] = False
            combo = _flags(**base_vals)
            errors = validate_effective_flags(combo)
            assert errors, (
                f"_FEATURE_DEPS says {flag} requires {dep}, but "
                f"validate_effective_flags accepts {flag}=True with {dep}=False"
            )

    # Every dep in _FEATURE_DEPS has a label in _DEP_LABELS.
    all_deps = {d for deps in _FEATURE_DEPS.values() for d in deps}
    missing_labels = all_deps - set(_DEP_LABELS)
    assert not missing_labels, f"_DEP_LABELS missing: {missing_labels}"
