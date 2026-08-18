"""Repository for the singleton ``app_settings`` row — the first-run wizard's store.

Split from ``config.Settings`` (env-only, frozen at process start): infra config
and secrets stay in the environment; the user-facing preferences the wizard writes
live here in the DB. Exactly one row (``id = 1``) ever exists. Callers own the
transaction — every function takes a live ``Session`` and never commits.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.config import Settings
from voxint.db.models import AppSettings, PipelineRun

if TYPE_CHECKING:
    from voxint.clients.llm import HttpLLMClient

SINGLETON_ID = 1


def resolve_effective_llm_api_key(row: AppSettings | None, settings: Settings) -> str:
    """Effective LLM API key: a non-blank row value WINS, else env ``LLM_API_KEY``.

    Returns the canonical stripped key (surrounding whitespace trimmed); ``""``
    means "no key configured anywhere". This is the SINGLE source of the
    DB-row-wins-over-env precedence (issue #10) — every LLM client construction
    (transcript enhancement, the enrichment producers, ``voxint doctor``) resolves
    the key through here, so a UI-stored key reaches them all and none can drift
    onto an env-only path. The result is a credential: never log, render, or export
    it (see the :class:`~voxint.db.models.AppSettings` docstring).
    """
    stored = (row.llm_api_key or "").strip() if row is not None else ""
    return stored or settings.llm_api_key.strip()


def resolve_effective_llm_endpoint(
    row: AppSettings | None, settings: Settings
) -> tuple[str, str]:
    """Effective ``(base_url, model)``: a non-blank row value wins, else the env
    default. Non-secret — the shared source for the wizard, the per-run preference
    snapshot (:func:`voxint.pipeline.stages.context.resolve_run_preferences`), and
    the enrichment paths, so they can never disagree on the effective endpoint.
    """
    base_url = (
        row.llm_base_url if row is not None and row.llm_base_url else settings.llm_base_url
    )
    model = row.llm_model if row is not None and row.llm_model else settings.llm_model
    return base_url, model


def llm_endpoint_form_fields(
    row: AppSettings | None, settings: Settings
) -> tuple[str, str, str, str]:
    """``(base_value, base_default, model_value, model_default)`` for the LLM forms.

    ``*_value`` is the ROW override, or ``""`` when it is ``NULL`` — so the input
    renders BLANK and the operator sees they are inheriting the installation
    setting, rather than an env-sourced value silently prefilled into the field.
    ``*_default`` is the env default, shown as the placeholder.

    This deliberately does NOT collapse ``NULL`` into the effective value the way
    :func:`resolve_effective_llm_endpoint` does for reads: the form must
    distinguish "pinned row override" from "inheriting env" so that saving an
    untouched form leaves the column ``NULL`` (issue #46) instead of pinning the
    env value onto the row. The tri-state "revert to installation setting"
    semantics mirror the ``remove_llm_api_key`` checkbox for the key.
    """
    base_value = row.llm_base_url if row is not None and row.llm_base_url else ""
    model_value = row.llm_model if row is not None and row.llm_model else ""
    return base_value, settings.llm_base_url, model_value, settings.llm_model


def str_flag_form_field(
    row: AppSettings | None, settings: Settings, name: str
) -> tuple[str, str]:
    """``(value, default)`` for a tri-state STRING settings field (issue #76).

    ``value`` is the ROW override, or ``""`` when the column is ``NULL``/blank — so
    the input renders BLANK and the operator sees they are inheriting the
    installation setting; ``default`` is the env default, shown as the placeholder.
    The string counterpart to :func:`feature_flag_state` (and the generic form of
    :func:`llm_endpoint_form_fields`): it renders the RAW column, NOT the resolved
    effective value, so saving an untouched field leaves the column ``NULL``
    (keeps inheriting env) instead of pinning the env default onto the row. Serves
    the web-search endpoint and the authority-domains editor. Not for secrets — a
    key is never rendered back (see :func:`effective_web_search_key_source`).
    """
    stored: str | None = getattr(row, name) if row is not None else None
    value = stored if stored else ""
    return value, getattr(settings, name)


def resolve_effective_llm_enabled(row: AppSettings | None, settings: Settings) -> bool:
    """Effective LLM enablement: the ROW value wins, else env ``LLM_ENABLED``.

    The SINGLE source of the DB-row-wins-over-env precedence for enablement, mirroring
    :func:`resolve_effective_llm_api_key` for the key. Every capability gate that turns
    LLM work on or off — transcript enhancement
    (:func:`voxint.pipeline.stages.context.resolve_run_preferences`), the enrichment
    producers (run-assets / web-research / LLM names), and ``voxint doctor`` — resolves
    enablement through here, so a UI toggle applies system-wide with no restart: an
    operator who enables LLM only in the UI (env ``LLM_ENABLED=false``) gets enrichment
    jobs too, and one who disables it in the UI (env ``LLM_ENABLED=true``) stops them.
    """
    if row is not None:
        return bool(row.llm_enabled)
    return settings.llm_enabled


def _effective_key_source(row: AppSettings | None, settings: Settings, name: str) -> str:
    """Where the effective value of credential column ``name`` comes from, for honest
    UI copy — never its value. ``"stored"`` iff the ROW value is non-blank; else
    ``"environment"`` when the env default is set; else ``"none"``. Shared by the
    per-credential public helpers so their precedence matches ``_resolve_str_flag``
    (row-wins-over-env) exactly and the status shown can never disagree with the
    value actually used.
    """
    if row is not None and (getattr(row, name) or "").strip():
        return "stored"
    if getattr(settings, name).strip():
        return "environment"
    return "none"


def effective_llm_key_source(row: AppSettings | None, settings: Settings) -> str:
    """Source of the effective LLM API key, for honest UI copy (never its value).
    Mirrors :func:`resolve_effective_llm_api_key`'s precedence."""
    return _effective_key_source(row, settings, "llm_api_key")


def effective_web_search_key_source(row: AppSettings | None, settings: Settings) -> str:
    """Source of the effective web-search API key, for honest UI copy (never its
    value) — issue #76. Mirrors :func:`resolve_effective_web_search_api_key`."""
    return _effective_key_source(row, settings, "web_search_api_key")


# ---------------------------------------------------------------------------
# Tri-state feature-flag resolvers (issue #74)
#
# Each in-UI-editable flag has a nullable column on ``AppSettings``: NULL means
# "inherit the env default" (``config.Settings``), a non-NULL value overrides it
# (the ``llm_base_url`` nullable pattern). Every runtime gate resolves through the
# matching ``resolve_effective_<flag>`` so a UI toggle applies with no restart and
# no read-site can drift onto a bare ``settings.<flag>`` env read. The column name
# mirrors the config field name, which the private helpers rely on.
# ---------------------------------------------------------------------------


def _resolve_bool_flag(row: AppSettings | None, settings: Settings, name: str) -> bool:
    """Tri-state boolean: the ROW column wins when non-NULL, else the env default.

    ``NULL`` (column unset) inherits ``settings.<name>``; ``True``/``False`` on the
    row overrides it. Shared by the per-flag public resolvers so the precedence is
    defined exactly once.
    """
    if row is not None:
        value = getattr(row, name)
        if value is not None:
            return bool(value)
    return bool(getattr(settings, name))


def _resolve_str_flag(row: AppSettings | None, settings: Settings, name: str) -> str:
    """Tri-state string/secret: a non-blank ROW value wins, else the env default.

    ``NULL`` or a blank row value inherits ``settings.<name>`` (the
    ``resolve_effective_llm_endpoint``/``resolve_effective_llm_api_key`` precedent),
    so clearing the field in the UI reverts to the installation setting rather than
    pinning an empty override over a valid env value.
    """
    if row is not None:
        value: str | None = getattr(row, name)
        if value is not None and value.strip():
            # Return the stripped row value (the ``resolve_effective_llm_api_key``
            # precedent): a hand-entered override with surrounding whitespace must
            # not be sent verbatim as a header / break the provider URL. The env
            # branch is untouched, so all-NULL parity is unchanged.
            return value.strip()
    env_value: str = getattr(settings, name)
    return env_value


def resolve_effective_enrichment_names_enabled(
    row: AppSettings | None, settings: Settings
) -> bool:
    return _resolve_bool_flag(row, settings, "enrichment_names_enabled")


def resolve_effective_enrichment_names_llm_enabled(
    row: AppSettings | None, settings: Settings
) -> bool:
    return _resolve_bool_flag(row, settings, "enrichment_names_llm_enabled")


def resolve_effective_enrichment_run_assets_enabled(
    row: AppSettings | None, settings: Settings
) -> bool:
    return _resolve_bool_flag(row, settings, "enrichment_run_assets_enabled")


def resolve_effective_enrichment_run_assets_autogenerate(
    row: AppSettings | None, settings: Settings
) -> bool:
    return _resolve_bool_flag(row, settings, "enrichment_run_assets_autogenerate")


def resolve_effective_voxint_web_research(row: AppSettings | None, settings: Settings) -> bool:
    return _resolve_bool_flag(row, settings, "voxint_web_research")


def resolve_effective_enrichment_web_research_enabled(
    row: AppSettings | None, settings: Settings
) -> bool:
    return _resolve_bool_flag(row, settings, "enrichment_web_research_enabled")


def resolve_effective_ytdlp_enabled(row: AppSettings | None, settings: Settings) -> bool:
    return _resolve_bool_flag(row, settings, "ytdlp_enabled")


def resolve_effective_llm_bundled_enabled(row: AppSettings | None, settings: Settings) -> bool:
    """Effective enablement of the optional bundled local LLM (issue #67).

    Tri-state like the other feature flags: the ROW column wins when non-NULL, else
    env ``LLM_BUNDLED_ENABLED``. This is only the operator *intent* — whether the
    bundle actually powers a job also requires a configured bundled base URL; use
    :func:`llm_bundled_active` at the routing sites, never this alone.
    """
    return _resolve_bool_flag(row, settings, "llm_bundled_enabled")


def llm_bundled_active(row: AppSettings | None, settings: Settings) -> bool:
    """Whether the bundled local LLM is the active endpoint for the two in-scope
    jobs (transcript enhancement + run-asset summary/entities) — issue #67.

    The SINGLE predicate the routing sites resolve through, so enhancement
    (``pipeline.stages.context``) and run-assets (``enrichment.asset_jobs``) can
    never drift apart. Active iff the operator enabled the bundle AND both bundled
    endpoint constants are compose-injected (an empty base URL or model ⇒ no usable
    bundled endpoint exists, so the flag is inert and the BYO path governs — never
    build a client that would POST ``"model": ""``). Names + research NEVER consult
    this — they are structurally BYO-only (#66: Qwen fails those jobs).
    """
    return (
        resolve_effective_llm_bundled_enabled(row, settings)
        and bool(settings.llm_bundled_base_url)
        and bool(settings.llm_bundled_model)
    )


def build_bundled_llm_client(settings: Settings) -> "HttpLLMClient":
    """Construct the keyless client for the bundled local endpoint (issue #67).

    Shared by the two in-scope routing sites so the endpoint/model/sampling are
    resolved in exactly one place. The bundled endpoint takes NO API key (it is a
    product-owned local service), and sends the measured, pinned greedy
    :class:`SamplingProfile` default. Callers must gate on
    :func:`llm_bundled_active` first.
    """
    from voxint.clients.llm import HttpLLMClient, SamplingProfile

    return HttpLLMClient(
        settings.llm_bundled_base_url,
        settings.llm_bundled_model,
        "",
        settings.llm_timeout_seconds,
        sampling=SamplingProfile(),
    )


def resolve_effective_source_authority_domains(
    row: AppSettings | None, settings: Settings
) -> str:
    return _resolve_str_flag(row, settings, "source_authority_domains")


def resolve_effective_web_search_base_url(row: AppSettings | None, settings: Settings) -> str:
    return _resolve_str_flag(row, settings, "web_search_base_url")


def resolve_effective_web_search_api_key(row: AppSettings | None, settings: Settings) -> str:
    """Effective web-search provider credential — a non-blank ROW value WINS, else
    env ``WEB_SEARCH_API_KEY``. Like :func:`resolve_effective_llm_api_key` this is a
    credential: never log, render, or export it (see the ``AppSettings`` docstring).
    """
    return _resolve_str_flag(row, settings, "web_search_api_key")


def feature_flag_state(row: AppSettings | None, name: str) -> str:
    """Raw tri-state of a nullable boolean flag column, for the settings form.

    Returns ``"inherit"`` when the column is ``NULL`` (or no row exists) — the
    operator is inheriting the env default — else ``"on"``/``"off"`` for a stored
    ``True``/``False`` override. This is the boolean counterpart to
    :func:`llm_endpoint_form_fields`: the form renders the RAW column, not the
    resolved effective value, so saving an untouched "use installation setting"
    choice writes ``NULL`` (keeps inheriting env) instead of pinning the current
    env default onto the row. The three states map exactly onto the tri-state
    radio the ``Features`` section renders (issue #62).
    """
    if row is None:
        return "inherit"
    value = getattr(row, name)
    if value is None:
        return "inherit"
    return "on" if value else "off"


@dataclass(frozen=True)
class EffectiveWebResearch:
    """The resolved web-research provider config for one job execution (issue #74).

    Built once at the row-owning boundary (the research worker / the CLI) from
    ``(row, settings)`` and threaded down into :mod:`voxint.research.fetch` and
    :mod:`voxint.research.search`, which have no ``Session`` in scope. This makes the
    deep retrieval gate honor the *effective* (row-over-env) value — an env-only
    read there would silently veto a UI enable (env false + row true). ``api_key``
    is a credential: never log/render it (only for the provider call + redaction).
    """

    enabled: bool
    base_url: str
    # A credential (like ``llm_api_key``): kept OUT of the auto-generated repr so a
    # stray ``%r``/f-string on the whole VO, a pytest assertion diff, or a
    # traceback-locals dump cannot leak it (this VO is threaded worker→agent→tools).
    api_key: str = field(repr=False)


def resolve_effective_web_research(
    row: AppSettings | None, settings: Settings
) -> EffectiveWebResearch:
    """Resolve the whole web-research provider config in one shot (issue #74)."""
    return EffectiveWebResearch(
        enabled=resolve_effective_voxint_web_research(row, settings),
        base_url=resolve_effective_web_search_base_url(row, settings),
        api_key=resolve_effective_web_search_api_key(row, settings),
    )


@dataclass(frozen=True)
class EffectiveFlags:
    """POST-resolve (effective) feature-flag values for :func:`validate_effective_flags`.

    A plain value object so the cross-flag invariant check is pure and importable
    without depending on ``config.Settings`` (which would cycle: ``app_settings``
    already imports ``Settings``). Both the ``config.py`` env-time validator and
    every runtime settings form build one and call the validator, so the five
    invariants live in exactly one place.
    """

    llm_enabled: bool
    enrichment_names_enabled: bool
    enrichment_names_llm_enabled: bool
    enrichment_run_assets_enabled: bool
    enrichment_run_assets_autogenerate: bool
    voxint_web_research: bool
    enrichment_web_research_enabled: bool
    web_search_base_url: str


def validate_web_search_base_url(base: str) -> str | None:
    """Return an operator-facing error message if ``base`` is not a usable searxng
    endpoint, else ``None``. Extracted so the ``config.py`` env validator and
    :func:`validate_effective_flags` share one definition of "valid base URL" and
    can never drift. The base URL must be absolute http(s), whitespace/backslash-
    free, credential-free (provider auth goes in ``web_search_api_key`` so it can be
    redacted uniformly), and a bare endpoint (no query/fragment).
    """
    if not base.strip():
        return (
            "voxint_web_research=true requires web_search_base_url — the"
            " searxng provider has no default endpoint"
        )
    if base != base.strip() or any(c.isspace() for c in base) or "\\" in base:
        return "web_search_base_url must not contain whitespace or backslashes"
    try:
        parts = urlsplit(base)
        # .port parses lazily; touch it so ":abc"/out-of-range fails here instead
        # of as an opaque provider_error on first search.
        _ = parts.port
    except ValueError:
        return "web_search_base_url is malformed"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "web_search_base_url must be an absolute http(s) URL"
    if parts.username is not None or parts.password is not None:
        return "web_search_base_url must not embed credentials — use web_search_api_key"
    if parts.query or parts.fragment:
        return "web_search_base_url must be a bare endpoint (no query/fragment)"
    return None


def validate_effective_flags(effective: EffectiveFlags) -> list[str]:
    """The five cross-flag invariants over POST-resolve values (issue #74).

    Returns the operator-facing error messages for every violated invariant (empty
    list ⇒ valid), in a stable order. This is the SINGLE source of the invariant
    logic: ``config.py``'s ``@model_validator`` adapts env values into an
    :class:`EffectiveFlags` and raises the first message (preserving boot-strict
    behavior + the exact messages), and every runtime settings form validates the
    effective (row-over-env) combination the same way — so a DB override can never
    create a combination the env-time check would have rejected. The invariants:

    * ``enrichment_names_llm_enabled`` ⇒ ``llm_enabled`` + ``enrichment_names_enabled``
    * ``enrichment_web_research_enabled`` ⇒ ``voxint_web_research`` + ``llm_enabled``
    * ``enrichment_run_assets_autogenerate`` ⇒ ``enrichment_run_assets_enabled`` ⇒ ``llm_enabled``
    * ``voxint_web_research`` ⇒ a valid ``web_search_base_url``
    """
    errors: list[str] = []
    if effective.enrichment_names_llm_enabled and not effective.llm_enabled:
        errors.append(
            "enrichment_names_llm_enabled requires llm_enabled=true — the "
            "LLM name pass reuses the configured enhancement endpoint"
        )
    if effective.enrichment_names_llm_enabled and not effective.enrichment_names_enabled:
        errors.append(
            "enrichment_names_llm_enabled requires enrichment_names_enabled=true"
            " — the LLM pass is additive to the offline name producer"
        )
    if effective.enrichment_web_research_enabled and not effective.voxint_web_research:
        errors.append(
            "enrichment_web_research_enabled requires voxint_web_research=true"
            " — the producer's only egress is the controlled retrieval tools"
        )
    if effective.enrichment_web_research_enabled and not effective.llm_enabled:
        errors.append(
            "enrichment_web_research_enabled requires llm_enabled=true — the"
            " producer reuses the configured enhancement endpoint"
        )
    if effective.enrichment_run_assets_enabled and not effective.llm_enabled:
        errors.append(
            "enrichment_run_assets_enabled requires llm_enabled=true — the"
            " asset generators reuse the configured enhancement endpoint"
        )
    if effective.enrichment_run_assets_autogenerate and not effective.enrichment_run_assets_enabled:
        errors.append(
            "enrichment_run_assets_autogenerate requires"
            " enrichment_run_assets_enabled=true — the post-finalize step"
            " only enqueues the feature it rides on"
        )
    if effective.voxint_web_research:
        url_error = validate_web_search_base_url(effective.web_search_base_url)
        if url_error is not None:
            errors.append(url_error)
    return errors


def get_app_settings(session: Session) -> AppSettings | None:
    """Return the singleton row, or ``None`` when the wizard has never saved."""
    return session.get(AppSettings, SINGLETON_ID)


def is_onboarded(session: Session) -> bool:
    """True once the wizard's finish step has committed ``onboarding_complete``.

    A missing row means "not onboarded"; the first-run gate treats it as such.
    """
    row = session.get(AppSettings, SINGLETON_ID)
    return bool(row and row.onboarding_complete)


def get_or_create(session: Session, *, llm_enabled_default: bool) -> AppSettings:
    """Return the singleton row, inserting a defaulted one if absent.

    The insert is wrapped in a SAVEPOINT so the UNIQUE(id) race between two
    first-time writers rolls back only the losing insert — not the caller's outer
    transaction — letting us re-read and adopt the winner's row (mirrors
    ``ingest.service._get_or_create_media``).

    ``llm_enabled_default`` seeds a NEWLY-created row's ``llm_enabled`` and is
    keyword-only and REQUIRED so a caller can never silently default it to False.
    Pass the current env ``Settings.llm_enabled``: :func:`resolve_run_preferences`
    takes ``llm_enabled`` HARD from the row once one exists, so a row first created
    for an unrelated reason (saving media folders, finishing onboarding) must not
    flip an env-enabled LLM off. The wizard's LLM step later overwrites this field
    explicitly from the operator's choice. An existing row is returned unchanged —
    the default only ever applies at first insert, and the concurrent-writer winner
    fixes the initial value (safe: all processes share one environment).
    """
    row = session.get(AppSettings, SINGLETON_ID)
    if row is not None:
        return row
    row = AppSettings(id=SINGLETON_ID, llm_enabled=llm_enabled_default)
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        adopted = session.get(AppSettings, SINGLETON_ID)
        if adopted is None:
            # The IntegrityError was not the expected singleton race — re-raise
            # it rather than returning None (an `assert` here is stripped under
            # `python -O`, which would violate the `-> AppSettings` contract).
            raise
        return adopted
    return row


def ready_tutorial_run_id(session: Session) -> uuid.UUID | None:
    """The configured tutorial run id, but only if its run still exists.

    ``app_settings.tutorial_run_id`` is a FK with ``ON DELETE SET NULL``, so a
    dangling reference is impossible — but the row may simply never have been
    seeded (``NULL``) when ``voxint tutorial seed`` has not run. Returns the id iff
    a tutorial run is configured AND present, so callers (the launch redirect, the
    Settings page, the banner resolver, the complete/replay routes) share ONE
    "is the tutorial actually available?" answer instead of each re-deriving it.
    """
    row = session.get(AppSettings, SINGLETON_ID)
    if row is None or row.tutorial_run_id is None:
        return None
    if session.get(PipelineRun, row.tutorial_run_id) is None:
        return None
    return row.tutorial_run_id


def mark_tutorial_complete(session: Session) -> bool:
    """Stamp ``tutorial_completed_at`` (idempotent). Caller commits.

    Returns ``False`` when there is no available tutorial run to complete (the
    route maps that to a 409 — a stray Settings token must not "complete" an
    unseeded tutorial). The stamp is written only when currently ``NULL`` so a
    refresh or double-submit preserves the original completion time rather than
    rewriting it to "now" on every repost.
    """
    if ready_tutorial_run_id(session) is None:
        return False
    row = session.get(AppSettings, SINGLETON_ID)
    assert row is not None  # ready_tutorial_run_id returned non-None ⇒ row exists
    if row.tutorial_completed_at is None:
        row.tutorial_completed_at = datetime.now(tz=UTC)
        session.flush()
    return True


def clear_tutorial_completion(session: Session) -> bool:
    """Clear ``tutorial_completed_at`` so the walkthrough can be replayed. Caller
    commits.

    Returns ``False`` when no tutorial run is available (→ 409). Replay is
    deliberately NON-destructive: it only clears the completion stamp and lets the
    operator walk the banners again. It does NOT reset the run's prior speaker
    rulings — the seeded run's children have no ``ON DELETE CASCADE`` and its
    decisions are append-only, so a true reset would be disproportionate surgery
    for a local teaching tool. The Settings copy states that prior rulings remain.
    """
    if ready_tutorial_run_id(session) is None:
        return False
    row = session.get(AppSettings, SINGLETON_ID)
    assert row is not None  # ready_tutorial_run_id returned non-None ⇒ row exists
    row.tutorial_completed_at = None
    session.flush()
    return True


def complete_onboarding(session: Session, *, llm_enabled_default: bool) -> AppSettings:
    """Mark the wizard finished (idempotent, get-or-create). Caller commits.

    ``llm_enabled_default`` is forwarded to :func:`get_or_create` for the same
    reason — finishing onboarding must not be the write that flips an env-enabled
    LLM off when it is also the first write to create the row.
    """
    row = get_or_create(session, llm_enabled_default=llm_enabled_default)
    row.onboarding_complete = True
    session.flush()
    return row
