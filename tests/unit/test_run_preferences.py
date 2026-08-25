"""Per-run preference resolution + application (slice 2).

``resolve_run_preferences`` layers the ``app_settings`` row over env defaults;
``apply_run_preferences`` swaps only the preference-derived fields onto the
process-cached base context. Both are pure — no DB, no network — so they are
unit-tested here; the live "settings edit takes effect with no restart" path is
covered end to end in tests/integration/test_run_preferences_live.py.
"""

import logging
from pathlib import Path

import httpx
import pytest

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.app_settings import resolve_effective_llm_api_key
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.domain_packs.base import DomainPack, dedup_order_preserving
from voxint.pipeline.stages.context import (
    StageContext,
    apply_run_preferences,
    resolve_run_preferences,
)
from voxint.pipeline.stages.transcribe import INITIAL_PROMPT_MAX_CHARS, _initial_prompt


def make_settings(
    *,
    llm_enabled: bool = False,
    llm_api_key: str = "",
    llm_base_url: str = "https://env.example/v1",
    llm_model: str = "env-model",
) -> Settings:
    return Settings(
        _env_file=None,
        llm_enabled=llm_enabled,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )


def make_base_ctx(
    *, vocabulary: tuple[str, ...] = (), enhancement_context: str = ""
) -> StageContext:
    # apply_run_preferences derives vocab + enhancement_context from the run's PACK
    # (issue #11), so the base pack carries the same fields the assertions expect.
    pack = DomainPack(
        name="test",
        vocabulary=vocabulary,
        prompt_fragments=(
            {"enhancement_context": enhancement_context} if enhancement_context else {}
        ),
    )
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=Path("/data/media"),
        domain_pack=pack,
        enhancement_context=enhancement_context,
        vocabulary=vocabulary,
    )


# ----------------------------------------------------------- resolve_run_preferences


def test_resolve_none_row_reproduces_env() -> None:
    settings = make_settings(
        llm_enabled=True, llm_base_url="https://env.example/v1", llm_model="env-model"
    )
    prefs = resolve_run_preferences(None, settings)
    assert prefs.llm_enabled is True
    assert prefs.llm_base_url == "https://env.example/v1"
    assert prefs.llm_model == "env-model"
    assert prefs.vocabulary == ()


def test_resolve_row_overrides_env() -> None:
    settings = make_settings(llm_enabled=False)
    row = AppSettings(
        id=1,
        llm_enabled=True,
        llm_base_url="https://row.example/v1",
        llm_model="row-model",
        vocabulary=["Alpha", "Beta"],
    )
    prefs = resolve_run_preferences(row, settings)
    assert prefs.llm_enabled is True
    assert prefs.llm_base_url == "https://row.example/v1"
    assert prefs.llm_model == "row-model"
    assert prefs.vocabulary == ("Alpha", "Beta")


def test_resolve_null_row_llm_fields_fall_back_to_env() -> None:
    settings = make_settings(llm_base_url="https://env.example/v1", llm_model="env-model")
    # NULL base_url/model in the row mean "use the env default".
    row = AppSettings(id=1, llm_enabled=True, llm_base_url=None, llm_model=None, vocabulary=[])
    prefs = resolve_run_preferences(row, settings)
    assert prefs.llm_base_url == "https://env.example/v1"
    assert prefs.llm_model == "env-model"


def test_resolve_dedups_and_strips_vocabulary() -> None:
    row = AppSettings(id=1, vocabulary=["Foo", " Foo ", "", "  ", "Bar", "Foo"])
    prefs = resolve_run_preferences(row, make_settings())
    assert prefs.vocabulary == ("Foo", "Bar")


# ------------------------------------------------------------- apply_run_preferences


def test_apply_unions_pack_and_user_vocab_order_preserving() -> None:
    base = make_base_ctx(vocabulary=("Pack", "Shared"))
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["Shared", "User"]), make_settings()
    )
    ctx = apply_run_preferences(base, make_settings(), prefs, base.domain_pack, llm_api_key="")
    assert ctx.vocabulary == ("Pack", "Shared", "User")


def test_apply_v1_snapshot_live_unions_glossary_default() -> None:
    """A pre-#153 (unversioned/v1) run keeps the live-union path byte-identical:
    a glossary edited AFTER submit still reaches the run. This is the invariant the
    version-2 freeze must not disturb (config_resolution_version defaults to 1)."""
    base = make_base_ctx(vocabulary=("Pack",))
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["Edited-Later"]), make_settings()
    )
    ctx = apply_run_preferences(
        base, make_settings(), prefs, base.domain_pack, llm_api_key=""
    )
    assert ctx.vocabulary == ("Pack", "Edited-Later")


def test_apply_v2_snapshot_uses_frozen_vocab_without_live_union() -> None:
    """A version-2 snapshot (issue #153) froze the effective vocabulary at submit,
    so the worker must NOT re-union the live glossary — a later settings edit can
    never leak into a deterministically-frozen run."""
    base = make_base_ctx(vocabulary=("Frozen-A", "Frozen-B"))
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["Edited-Later"]), make_settings()
    )
    ctx = apply_run_preferences(
        base,
        make_settings(),
        prefs,
        base.domain_pack,
        llm_api_key="",
        config_resolution_version=2,
    )
    assert ctx.vocabulary == ("Frozen-A", "Frozen-B")
    # The frozen vocab is what renders into the enhancement context, too.
    assert "Edited-Later" not in ctx.enhancement_context


def test_apply_unrecognized_version_falls_through_to_live_union() -> None:
    """An unknown future version is treated as the live-union path, never silently
    reinterpreted as a freeze it may not be (only version 2 is a freeze)."""
    base = make_base_ctx(vocabulary=("Pack",))
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["User"]), make_settings()
    )
    ctx = apply_run_preferences(
        base,
        make_settings(),
        prefs,
        base.domain_pack,
        llm_api_key="",
        config_resolution_version=99,
    )
    assert ctx.vocabulary == ("Pack", "User")


def test_v2_freeze_is_byte_identical_to_v1_live_union_global_baseline() -> None:
    """Numerics contract (issue #153): the version-2 frozen effective vocabulary is
    byte-identical to what the v1 live-union path computes for the global-baseline
    (default pack + operator glossary) case, so migrating a stock install to the
    freeze changes no run's ASR prompt. Pins the algebraic identity
    D(pack + app) == D(pack + D(app)) == D(frozen) on deliberately messy input
    (duplicate + whitespace variants a naive union would diverge on). The two sides
    stay equal only because the submit-time freeze and this live path both run the
    one shared dedup_order_preserving."""
    pack_vocab = ("Alpha", " Alpha ", "Beta")
    # The raw operator glossary as stored, BEFORE resolve_run_preferences dedups it.
    raw_app_vocab = [" Beta ", "Gamma", "", "Gamma ", "  "]

    # What the submit-time freeze bakes into the snapshot for the global baseline
    # (ingest.service._run_domain_pack_snapshot global-baseline branch).
    frozen_effective = list(dedup_order_preserving((*pack_vocab, *raw_app_vocab)))
    assert frozen_effective == ["Alpha", "Beta", "Gamma"]

    # The v1 live path a pre-#153 (unversioned) run runs in the worker: prefs is
    # dedup(app) because resolve_run_preferences pre-dedups the glossary, and apply
    # unions it onto the raw pack vocabulary.
    base = make_base_ctx(vocabulary=pack_vocab)
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=raw_app_vocab), make_settings()
    )
    v1 = apply_run_preferences(
        base, make_settings(), prefs, base.domain_pack, llm_api_key=""
    )

    # The v2 path: the freeze already resolved the effective list, so the decoded
    # pack carries frozen_effective and apply must NOT re-union the live glossary.
    frozen_base = make_base_ctx(vocabulary=tuple(frozen_effective))
    v2 = apply_run_preferences(
        frozen_base,
        make_settings(),
        prefs,
        frozen_base.domain_pack,
        llm_api_key="",
        config_resolution_version=2,
    )

    assert tuple(frozen_effective) == v1.vocabulary == v2.vocabulary


def test_apply_renders_vocab_into_enhancement_context() -> None:
    base = make_base_ctx(vocabulary=(), enhancement_context="PACK-FRAGMENT")
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["Alpha", "Beta"]), make_settings()
    )
    ctx = apply_run_preferences(base, make_settings(), prefs, base.domain_pack, llm_api_key="")
    assert ctx.enhancement_context.startswith("PACK-FRAGMENT")
    assert "Alpha, Beta" in ctx.enhancement_context


def test_apply_empty_vocab_leaves_enhancement_context_untouched() -> None:
    base = make_base_ctx(vocabulary=(), enhancement_context="PACK-FRAGMENT")
    prefs = resolve_run_preferences(None, make_settings())
    ctx = apply_run_preferences(base, make_settings(), prefs, base.domain_pack, llm_api_key="")
    assert ctx.enhancement_context == "PACK-FRAGMENT"


def test_apply_enables_llm_when_prefs_enabled_and_key_present() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    ctx = apply_run_preferences(
        base, make_settings(), prefs, base.domain_pack, llm_api_key="sk-test"
    )
    assert ctx.llm is not None


def test_apply_disables_llm_without_key_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    with caplog.at_level(logging.WARNING):
        ctx = apply_run_preferences(base, make_settings(), prefs, base.domain_pack, llm_api_key="")
    assert ctx.llm is None
    assert any("no API key is configured" in r.message for r in caplog.records)


def test_apply_no_row_enabled_without_key_disables_llm() -> None:
    # No app_settings row + env LLM enabled but no key → honest no-op (llm=None),
    # not an unusable client. (The one intentional refinement over the pre-wizard
    # path, which used to construct an enabled-but-keyless client.) The effective
    # key the worker would resolve from (no row, empty env) is "".
    base = make_base_ctx()
    settings = make_settings(llm_enabled=True, llm_api_key="")
    prefs = resolve_run_preferences(None, settings)
    key = resolve_effective_llm_api_key(None, settings)
    ctx = apply_run_preferences(base, settings, prefs, base.domain_pack, llm_api_key=key)
    assert ctx.llm is None


def test_apply_disables_llm_when_budget_exceeds_lease(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Deferred finding 2, runtime fail-closed guard: even with an enabled row and a
    # key present, a run budget that no longer fits the enhance_match lease must NOT
    # build a client — otherwise the recovery sweep could reclaim the stage
    # mid-flight. The env-time validator can't catch this because the wizard enables
    # the LLM with the env flag off, so this per-run check is the backstop.
    base = make_base_ctx()
    settings = Settings(
        _env_file=None,
        llm_api_key="sk-test",
        llm_enabled=False,  # keep the env-time validator quiet at construction
        llm_run_budget_seconds=999999.0,
        stage_lease_seconds=21600,
    )
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), settings)
    with caplog.at_level(logging.WARNING):
        ctx = apply_run_preferences(base, settings, prefs, base.domain_pack, llm_api_key="sk-test")
    assert ctx.llm is None
    assert any("lease" in r.message for r in caplog.records)


def test_apply_disables_llm_when_key_is_whitespace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A whitespace-only key must read as absent so enhancement degrades to llm=None
    # rather than building a client with an unusable key. The stripping now lives in
    # resolve_effective_llm_api_key (the single precedence source); this exercises the
    # resolver → apply path end to end so the guarantee cannot silently regress.
    base = make_base_ctx()
    settings = make_settings(llm_api_key="   ")
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), settings)
    key = resolve_effective_llm_api_key(AppSettings(id=1, llm_api_key="   "), settings)
    assert key == ""
    with caplog.at_level(logging.WARNING):
        ctx = apply_run_preferences(base, settings, prefs, base.domain_pack, llm_api_key=key)
    assert ctx.llm is None
    assert any("no API key is configured" in r.message for r in caplog.records)


def test_apply_disables_llm_when_client_construction_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A malformed base_url raises while httpx builds the client. Since
    # apply_run_preferences runs before execute_run's failure handling in
    # run_pipeline, propagating would strand the run QUEUED for the recovery sweep to
    # re-publish forever (poison loop). The guard must swallow it → llm=None.
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise httpx.InvalidURL("bad base url")

    monkeypatch.setattr("voxint.pipeline.stages.context.HttpLLMClient", _boom)
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    with caplog.at_level(logging.WARNING):
        ctx = apply_run_preferences(
            base, make_settings(), prefs, base.domain_pack, llm_api_key="sk-test"
        )
    assert ctx.llm is None
    assert any("could not be built" in r.message for r in caplog.records)


def test_apply_disables_llm_when_prefs_disabled() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=False), make_settings())
    ctx = apply_run_preferences(
        base, make_settings(), prefs, base.domain_pack, llm_api_key="sk-test"
    )
    assert ctx.llm is None


def test_apply_preserves_transport_clients() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(None, make_settings())
    ctx = apply_run_preferences(base, make_settings(), prefs, base.domain_pack, llm_api_key="")
    assert ctx.asr is base.asr
    assert ctx.diarizer is base.diarizer
    assert ctx.embedder is base.embedder


# -------------------------------------------------------- bundled local LLM (#67)


def _bundled_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        llm_enabled=False,  # keep the env-time validator quiet at construction
        llm_bundled_base_url="http://voxint-llm:8080/v1",
        llm_bundled_model="qwen3-4b-instruct-2507",
        **overrides,  # type: ignore[arg-type]
    )


def test_apply_bundled_builds_keyless_client_and_flags_it() -> None:
    # The scoped bundle (#67) needs NO api key: enhancement builds a client on the
    # keyless bundled endpoint and marks the context bundled so enhance_match drops
    # its name_hints. The bundled model — not the env BYO model — is used.
    base = make_base_ctx()
    settings = _bundled_settings()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), settings)
    ctx = apply_run_preferences(
        base, settings, prefs, base.domain_pack, llm_api_key="", bundled=True
    )
    assert isinstance(ctx.llm, HttpLLMClient)
    assert ctx.llm_bundled is True
    assert ctx.llm._model == "qwen3-4b-instruct-2507"
    assert ctx.llm._api_key == ""
    ctx.llm.close()


def test_apply_byo_path_is_not_flagged_bundled() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    ctx = apply_run_preferences(
        base, make_settings(), prefs, base.domain_pack, llm_api_key="sk-test"
    )
    assert isinstance(ctx.llm, HttpLLMClient)
    assert ctx.llm_bundled is False
    ctx.llm.close()


def test_apply_bundled_still_gated_by_llm_enabled() -> None:
    # Bundled-on but LLM enhancement off ⇒ nothing builds. llm_enabled is the master
    # gate (the toggle's help copy tells the operator to enable enhancement).
    base = make_base_ctx()
    settings = _bundled_settings()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=False), settings)
    ctx = apply_run_preferences(
        base, settings, prefs, base.domain_pack, llm_api_key="", bundled=True
    )
    assert ctx.llm is None
    assert ctx.llm_bundled is False


# --------------------------------------------------------- transcribe initial_prompt


def test_initial_prompt_empty_is_none() -> None:
    assert _initial_prompt(()) is None


def test_initial_prompt_joins_terms() -> None:
    assert _initial_prompt(("Foo", "Bar")) == "Foo, Bar"


def test_initial_prompt_skips_oversized_term_keeps_others() -> None:
    huge = "x" * (INITIAL_PROMPT_MAX_CHARS + 10)
    # The oversized first term is skipped, not fatal — later terms still make it in.
    assert _initial_prompt((huge, "Foo", "Bar")) == "Foo, Bar"


def test_initial_prompt_truncates_whole_terms_within_cap() -> None:
    vocab = tuple(f"term{i:04d}" for i in range(400))  # ~4000 chars unbounded
    prompt = _initial_prompt(vocab)
    assert prompt is not None
    assert len(prompt) <= INITIAL_PROMPT_MAX_CHARS
    kept = prompt.split(", ")
    assert 0 < len(kept) < len(vocab)  # some terms dropped at the boundary
    assert all(term in vocab for term in kept)  # never a partial term
