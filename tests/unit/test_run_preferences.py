"""Per-run preference resolution + application (slice 2).

``resolve_run_preferences`` layers the ``app_settings`` row over env defaults;
``apply_run_preferences`` swaps only the preference-derived fields onto the
process-cached base context. Both are pure — no DB, no network — so they are
unit-tested here; the live "settings edit takes effect with no restart" path is
covered end to end in tests/integration/test_run_preferences_live.py.
"""

import logging
from pathlib import Path

import pytest

from tests.fakes import FakeASR, FakeDiarizer, FakeEmbedder, FakeLLM
from voxint.config import Settings
from voxint.db.models import AppSettings
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
    return StageContext(
        asr=FakeASR(),
        diarizer=FakeDiarizer(),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        media_root=Path("/data/media"),
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
    ctx = apply_run_preferences(base, make_settings(), prefs)
    assert ctx.vocabulary == ("Pack", "Shared", "User")


def test_apply_renders_vocab_into_enhancement_context() -> None:
    base = make_base_ctx(vocabulary=(), enhancement_context="PACK-FRAGMENT")
    prefs = resolve_run_preferences(
        AppSettings(id=1, vocabulary=["Alpha", "Beta"]), make_settings()
    )
    ctx = apply_run_preferences(base, make_settings(), prefs)
    assert ctx.enhancement_context.startswith("PACK-FRAGMENT")
    assert "Alpha, Beta" in ctx.enhancement_context


def test_apply_empty_vocab_leaves_enhancement_context_untouched() -> None:
    base = make_base_ctx(vocabulary=(), enhancement_context="PACK-FRAGMENT")
    prefs = resolve_run_preferences(None, make_settings())
    ctx = apply_run_preferences(base, make_settings(), prefs)
    assert ctx.enhancement_context == "PACK-FRAGMENT"


def test_apply_enables_llm_when_prefs_enabled_and_key_present() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    ctx = apply_run_preferences(base, make_settings(llm_api_key="sk-test"), prefs)
    assert ctx.llm is not None


def test_apply_disables_llm_without_key_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=True), make_settings())
    with caplog.at_level(logging.WARNING):
        ctx = apply_run_preferences(base, make_settings(llm_api_key=""), prefs)
    assert ctx.llm is None
    assert any("LLM_API_KEY is unset" in r.message for r in caplog.records)


def test_apply_no_row_enabled_without_key_disables_llm() -> None:
    # No app_settings row + env LLM enabled but no key → honest no-op (llm=None),
    # not an unusable client. (The one intentional refinement over the pre-wizard
    # path, which used to construct an enabled-but-keyless client.)
    base = make_base_ctx()
    settings = make_settings(llm_enabled=True, llm_api_key="")
    prefs = resolve_run_preferences(None, settings)
    ctx = apply_run_preferences(base, settings, prefs)
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
        ctx = apply_run_preferences(base, settings, prefs)
    assert ctx.llm is None
    assert any("lease" in r.message for r in caplog.records)


def test_apply_disables_llm_when_prefs_disabled() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(AppSettings(id=1, llm_enabled=False), make_settings())
    ctx = apply_run_preferences(base, make_settings(llm_api_key="sk-test"), prefs)
    assert ctx.llm is None


def test_apply_preserves_transport_clients() -> None:
    base = make_base_ctx()
    prefs = resolve_run_preferences(None, make_settings())
    ctx = apply_run_preferences(base, make_settings(), prefs)
    assert ctx.asr is base.asr
    assert ctx.diarizer is base.diarizer
    assert ctx.embedder is base.embedder


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
