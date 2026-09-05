"""Settings area: first-run setup wizard (issue #3) and the settings surface.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151). Two
routers: ``setup_router`` carries the wizard and is registered on the app
directly, so the onboarding gate exempts it structurally (an un-onboarded
operator must be able to reach the page the gate redirects them to; auth still
applies). ``router`` carries /settings and the guided-tutorial lifecycle behind
the router-level onboarding gate.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.datastructures import FormData
from starlette.responses import RedirectResponse

from voxint.api import settings_view
from voxint.api.auth import AuthContext
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, CSRF_USERS, mint_csrf_token
from voxint.api.host_metrics import HostMetricsSnapshot, collect_host_metrics_or_empty
from voxint.api.languages import LANGUAGE_NAMES
from voxint.api.resource_status import (
    ResourceSnapshot,
    build_resource_strip,
    collect_resource_status_or_empty,
    vram_percent,
)
from voxint.api.routers import deps
from voxint.api.routers.deps import (
    AdminDep,
    OperatorDep,
    SessionDep,
    _require_admin,
    _require_csrf,
    require_onboarded,
    require_users_enabled,
    templates,
    viewer_write_guard,
)
from voxint.api.service_identity import collect_service_identity
from voxint.api.setup_wizard import (
    STEP_ORDER,
    ScanResult,
    SetupValidationError,
    WizardStep,
    list_media_subdirs,
    next_step,
    normalize_llm_api_key,
    normalize_llm_base_url,
    normalize_llm_model,
    normalize_vocabulary,
    normalize_web_search_api_key,
    parse_step,
    scan_media_folders,
    validate_llm_enable,
)
from voxint.app_settings import (
    EffectiveFlags,
    clear_tutorial_completion,
    complete_onboarding,
    effective_llm_key_source,
    effective_web_search_key_source,
    feature_flag_state,
    get_app_settings,
    get_or_create,
    llm_bundled_active,
    llm_endpoint_form_fields,
    mark_tutorial_complete,
    ready_tutorial_run_id,
    resolve_effective_enrichment_names_enabled,
    resolve_effective_enrichment_names_llm_enabled,
    resolve_effective_enrichment_run_assets_autogenerate,
    resolve_effective_enrichment_run_assets_enabled,
    resolve_effective_enrichment_web_research_enabled,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_semantic_index_autogenerate,
    resolve_effective_semantic_index_enabled,
    resolve_effective_synthdetect_autogenerate,
    resolve_effective_synthdetect_enabled,
    resolve_effective_translation_autogenerate,
    resolve_effective_translation_target_language,
    resolve_effective_voxint_web_research,
    resolve_effective_watch_folder_enabled,
    resolve_effective_web_search_api_key,
    resolve_effective_web_search_base_url,
    semantic_index_flags_ok,
    str_flag_form_field,
    synthdetect_flags_ok,
    translation_flags_ok,
    validate_effective_flags,
    validate_web_search_base_url,
)
from voxint.config import Settings, llm_budget_fits_stage_lease
from voxint.db.models import AppSettings
from voxint.diagnostics import LLM_NOT_CONFIGURED_DETAIL, check_state, run_diagnostics
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import (
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    OperatorCorrectionError,
    normalize_operator_corrections,
)
from voxint.domain_packs.registry import available_domain_packs, default_domain_pack
from voxint.embeddings.onnx_embedder import minilm_artifacts_available
from voxint.enrichment.translation_jobs import translation_gates_open
from voxint.enrichment.triage import validate_authority_domains
from voxint.ingest import submit_media_item_if_new
from voxint.media.registration import (
    PACK_DEFAULT_SENTINEL,
    folder_pack_map,
    register_folder,
    registered_folder_paths,
    set_folder_pack,
    unregister_folder,
)
from voxint.plugins import PluginRegistry
from voxint.tutorial.seed import seed_tutorial_run

logger = logging.getLogger(__name__)

setup_router = APIRouter(dependencies=[Depends(viewer_write_guard)])
router = APIRouter(dependencies=[Depends(require_onboarded), Depends(_require_admin)])

# Bounded, non-secret operator guidance for a failed UI-triggered tutorial seed
# (issue #75). At most two messages: a storage failure vs. broken/missing bundled
# data. The real exception is logged server-side; the operator never sees a path
# or traceback.
_TUTORIAL_SEED_STORAGE_ERROR = (
    "Could not set up the tutorial: the media folder is not writable. "
    "Check that the app can write to its media directory, then try again."
)
_TUTORIAL_SEED_ASSET_ERROR = (
    "Could not set up the tutorial: its bundled sample data is missing or "
    "unreadable. This is likely a broken installation — reinstall or reach out "
    "for help."
)


def _try_seed_tutorial(
    session: Session, settings: Settings
) -> tuple[uuid.UUID | None, str | None]:
    """Seed the tutorial, mapping known environment/asset failures to bounded copy.

    Returns ``(run_id, None)`` on success or ``(None, message)`` on a classified
    failure. Only filesystem-storage and bundled-resource failures are caught —
    programmer/DB/builder defects (SQLAlchemy errors, ``TutorialSeedError`` from a
    fixture-invariant bug) propagate so they are never masked as tutorial copy
    (issue #75): a builder bug is not fixed by "reinstalling", so it must surface
    loudly, not be dressed up as missing data. The real exception is logged
    server-side; the message is path-free and traceback-free. The CALLER must
    ``session.rollback()`` before re-rendering — the request session commits on any
    successful response.

    Classification is by exception *type*, not origin — a v1 approximation. In the
    rare cross case (e.g. a ``PermissionError`` reading bundled resources) the
    operator gets the storage message; acceptable for a single-operator install and
    the true exception is always in the server log.
    """
    try:
        run_id = seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )
    # FileNotFoundError MUST precede the OSError clause below — it is an OSError
    # subclass, and a missing bundled asset is a data problem, not a storage one.
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        logger.exception("Tutorial seed failed: bundled sample data missing/unreadable")
        return None, _TUTORIAL_SEED_ASSET_ERROR
    except OSError:
        logger.exception("Tutorial seed failed: media folder not writable")
        return None, _TUTORIAL_SEED_STORAGE_ERROR
    return run_id, None


def _setup_context(
    request: Request,
    session: Session,
    step: WizardStep,
    folder_path: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Template context for a setup-wizard step.

    Read-only: it uses ``get_app_settings`` (never ``get_or_create``) so rendering a
    GET can't create the app_settings row. Fields prefill from the saved row layered
    over env defaults, matching how a run would resolve them; a POST that fails
    validation passes ``error=`` plus the raw submitted text via ``overrides`` so the
    operator's in-progress input survives the re-render.

    ``folder_path`` overrides the MEDIA-step browse location (the folder panel); it
    defaults to ``?path=`` from the query string. A non-HTMX folder mutation that
    fails re-renders the full page and passes the submitted ``path`` here so the
    browser stays put instead of snapping back to the root.
    """
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    media_folders = registered_folder_paths(session)
    vocabulary = list(row.vocabulary) if row and row.vocabulary else []
    _base_value, _base_default, _model_value, _model_default = llm_endpoint_form_fields(
        row, settings
    )
    context: dict[str, Any] = {
        "request": request,
        "step": step,
        "steps": STEP_ORDER,
        "step_index": STEP_ORDER.index(step),
        "next_step": next_step(step),
        "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
        "media_root": str(settings.media_root),
        # The finish-step summary counts registered folders; the media step itself
        # now renders them through the folder panel (issue #63), not a textarea.
        "media_folders": media_folders,
        "vocabulary": vocabulary,
        "vocabulary_text": "\n".join(vocabulary),
        # LLM step: the row's enablement over env, its (non-secret) endpoint OVERRIDE
        # (blank when inheriting env; env default shown as placeholder — issue #46),
        # and whether an EFFECTIVE key (UI-stored row value winning over env) is
        # present and where it comes from — never the key value itself.
        "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
        "llm_base_url": _base_value,
        "llm_base_url_default": _base_default,
        "llm_model": _model_value,
        "llm_model_default": _model_default,
        "llm_key_present": bool(resolve_effective_llm_api_key(row, settings)),
        "llm_key_source": effective_llm_key_source(row, settings),
        "llm_budget_ok": llm_budget_fits_stage_lease(settings),
        # The finish step offers "Finish setup & start tutorial" (seeds if needed,
        # issue #75) alongside a plain "Finish setup"; this flag only selects the
        # already-seeded vs. seed-on-finish copy, never the redirect.
        "tutorial_available": ready_tutorial_run_id(session) is not None,
        "active_nav": "setup",
        "error": None,
        # Bounded, non-secret message when a seed-on-finish fails (issue #75).
        "tutorial_error": None,
    }
    # The folder browser (issue #63) is built only for the MEDIA step, so the
    # other five wizard screens never pay for a MEDIA_ROOT walk or a domain-pack
    # load. The untrusted ?path= is revalidated inside list_media_subdirs.
    if step is WizardStep.MEDIA:
        context.update(
            _folder_panel_context(
                session,
                settings,
                action_prefix="/setup/folders",
                csrf=context["csrf_setup"],
                path=folder_path if folder_path is not None else (
                    request.query_params.get("path") or "."
                ),
            )
        )
    context.update(overrides)
    return context


# Issue #77: when the LLM settings form turns LLM enhancement OFF, any feature that
# ``validate_effective_flags`` requires ``llm_enabled=true`` for would be stranded on
# — a combination the boot validator (config.py) rejects on restart. The LLM form
# refuses such a disable (writes nothing) rather than auto-disabling the dependent
# (#62's "never flip an unrelated setting"). The rejection copy is directed at THIS
# page and names the blocker, unlike ``_FEATURE_INVARIANT_COPY`` whose "turn it on in
# the LLM section below, or turn <feature> off" wording is written for the Features
# form and points the wrong way here. Keyed on the exact validator messages for the
# three llm-dependency invariants → the blocking feature's operator label; a drift
# test locks the keys against ``validate_effective_flags``.
_LLM_DEPENDENCY_LABELS: dict[str, str] = {
    "enrichment_names_llm_enabled requires llm_enabled=true — the "
    "LLM name pass reuses the configured enhancement endpoint": "the LLM name pass",
    "enrichment_run_assets_enabled requires llm_enabled=true — the"
    " asset generators reuse the configured enhancement endpoint": "run assets",
    "enrichment_web_research_enabled requires llm_enabled=true — the"
    " producer reuses the configured enhancement endpoint": "web-research enrichment",
}


def _join_operator_labels(labels: list[str]) -> str:
    """Grammatical, order-preserving join of one to three labels for operator copy."""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _llm_disable_strand_error(row: AppSettings | None, settings: Settings) -> str | None:
    """Refuse to turn LLM enhancement OFF while a feature that needs it is on (#77).

    ``validate_effective_flags`` stays the single source of the invariant decision:
    the current effective flag combination is validated at its present LLM state and
    again with ``llm_enabled`` forced ``False``, and we act ONLY on the *delta* — the
    violations disabling LLM would newly introduce. So an unrelated pre-existing
    violation (e.g. a malformed web-search endpoint the fail-closed enable path left
    behind, or a dependent already stranded) never blocks an LLM save and never
    mislabels the cause; only the LLM-dependency invariants can flip here because the
    other flags are held at their current effective values. Returns an LLM-page plain
    message naming the blocker(s), or ``None`` when disabling is safe.
    """
    others: dict[str, object] = {
        "enrichment_names_enabled": resolve_effective_enrichment_names_enabled(row, settings),
        "enrichment_names_llm_enabled": resolve_effective_enrichment_names_llm_enabled(
            row, settings
        ),
        "enrichment_run_assets_enabled": resolve_effective_enrichment_run_assets_enabled(
            row, settings
        ),
        "enrichment_run_assets_autogenerate": resolve_effective_enrichment_run_assets_autogenerate(
            row, settings
        ),
        "voxint_web_research": resolve_effective_voxint_web_research(row, settings),
        "enrichment_web_research_enabled": resolve_effective_enrichment_web_research_enabled(
            row, settings
        ),
        "web_search_base_url": resolve_effective_web_search_base_url(row, settings),
    }
    before = set(
        validate_effective_flags(
            EffectiveFlags(llm_enabled=resolve_effective_llm_enabled(row, settings), **others)  # type: ignore[arg-type]
        )
    )
    new_violations = [
        message
        for message in validate_effective_flags(
            EffectiveFlags(llm_enabled=False, **others)  # type: ignore[arg-type]
        )
        if message not in before
    ]
    if not new_violations:
        return None
    labels = [
        _LLM_DEPENDENCY_LABELS[message]
        for message in new_violations
        if message in _LLM_DEPENDENCY_LABELS
    ]
    if not labels:
        # Defensive: a newly-introduced violation with no label (unreachable — only
        # the three llm-dependency invariants change when llm_enabled flips). Still
        # refuse, without naming a feature we can't identify.
        return (
            "Another feature still needs LLM enhancement. Turn it off in the Features"
            " or Sources & research section before turning LLM enhancement off."
        )
    needs = "it needs" if len(labels) == 1 else "they need"
    return (
        f"Turn off {_join_operator_labels(labels)} before turning LLM enhancement off"
        f" — {needs} the LLM. You can turn features off in the Features and Sources &"
        " research sections."
    )


# ---- Media folders + per-folder domain packs (issue #63) --------------------
# The folder browser and the Settings "Media folders" section register folders
# through the shared media_folders write service (issue #153): it serializes every
# mutation on a Postgres advisory lock, refuses overlapping (nested) registrations,
# and enforces the folder cap.


def _apply_folder_mutation(
    session: Session,
    settings: Settings,
    *,
    action: str,
    folder: str,
    pack: str | None,
) -> str | None:
    """Dispatch a folder-panel mutation to the shared registration service.

    ``action`` is the consolidated route's verb (one POST per mount instead of
    three) — an unknown value is a client error (422), never a silent no-op.
    """
    if action == "add":
        return register_folder(session, settings, folder)
    if action == "remove":
        return unregister_folder(session, settings, folder)
    if action == "pack":
        return set_folder_pack(session, settings, folder, pack)
    raise HTTPException(status_code=422, detail=f"unknown folder action {action!r}")


def _folder_missing(root: Path, folder: str) -> bool:
    """True if a registered ``folder`` no longer resolves to a directory in the root.

    ``root`` is already ``resolve()``-d. Mirrors the browse/add containment posture:
    a path that escapes the root (symlink swap), fails to resolve (embedded NUL), or
    is not an existing directory reads as missing — never as a present directory
    sitting outside MEDIA_ROOT.
    """
    try:
        resolved = (root / folder).resolve()
        return not (resolved.is_relative_to(root) and resolved.is_dir())
    except (ValueError, OSError):
        return True


def _folder_panel_context(
    session: Session,
    settings: Settings,
    *,
    action_prefix: str,
    csrf: str,
    path: str,
) -> dict[str, Any]:
    """Template context for the shared folder-panel fragment (issue #63).

    ``action_prefix`` is ``/setup/folders`` or ``/settings/folders`` and ``csrf``
    the matching token, so one fragment serves both the wizard (CSRF_SETUP) and the
    Settings page (CSRF_SETTINGS). ``path`` is the untrusted current browse
    location, revalidated by ``list_media_subdirs``. ``available_domain_packs`` is
    loaded once; a registry-wide ``DomainPackError`` disables the pack selects with
    an honest message rather than crashing the page, and a stored pack no longer
    available is flagged so the select can show it as "(unavailable)" instead of a
    false "Default".
    """
    registered = registered_folder_paths(session)
    mapping = folder_pack_map(session)
    listing = list_media_subdirs(settings.media_root, path, set(registered))
    packs_available = True
    pack_names: list[str] = []
    try:
        pack_names = sorted(available_domain_packs(settings))
    except DomainPackError:
        packs_available = False
    # Full-page navigation base for the no-JS fallback (htmx enhances these into
    # #folder-panel swaps). Exactly two mounts, so deriving the prefix here keeps
    # the call sites clean.
    is_setup = action_prefix == "/setup/folders"
    if is_setup:
        nav_prefix, nav_suffix, nav_base = "/setup?step=media&path=", "", "/setup"
    elif settings.console_settings_enabled:
        nav_prefix, nav_suffix, nav_base = "/settings/media?path=", "#folders", "/settings/media"
    else:
        nav_prefix, nav_suffix, nav_base = "/settings?path=", "#folders", "/settings"
    root = settings.media_root.resolve()
    folder_rows: list[dict[str, Any]] = []
    for folder in registered:
        current_pack = mapping.get(folder, "")
        folder_rows.append(
            {
                "rel": folder,
                "pack": current_pack,
                # A stored pack the current registry doesn't offer (absent from a
                # healthy registry, or a registry that can't be listed at all) is
                # shown explicitly as its own selected "(unavailable)" option — never
                # a false "Default".
                "pack_unavailable": bool(current_pack) and current_pack not in pack_names,
                # A registered folder that has since vanished on disk is flagged
                # (still removable) — honest state, not a silent drop. Resolve +
                # contain (not a bare ``is_dir``) to match the browse/add posture: a
                # symlink or path that now escapes the root reads as missing rather
                # than as a present directory outside MEDIA_ROOT.
                "missing": _folder_missing(root, folder),
            }
        )
    return {
        "folders_browse_action": f"{action_prefix}/browse",
        "folders_mutate_action": action_prefix,
        "folders_nav_prefix": nav_prefix,
        "folders_nav_suffix": nav_suffix,
        "folders_nav_base": nav_base,
        "folders_wizard": is_setup,
        "folders_csrf": csrf,
        "folder_listing": listing,
        "folder_rows": folder_rows,
        "folder_pack_names": pack_names,
        "folder_packs_available": packs_available,
        # The "Default" <option> submits this explicit sentinel (see the constant) so
        # a disabled/absent select is never misread as "clear the mapping".
        "folder_pack_default_value": PACK_DEFAULT_SENTINEL,
        "folder_error": None,
    }


def _persist_llm_settings(
    session: Session,
    settings: Settings,
    *,
    enabled: bool,
    raw_base_url: str,
    raw_model: str,
    raw_key: str,
    remove_key: bool,
) -> str | None:
    """Apply the LLM settings from a form submission as ONE deliberate mutation.

    Shared by ``POST /setup/llm`` and ``POST /settings/llm``. API sessions commit on
    every successful response (including error re-renders), so this computes a
    *candidate* state, validates it, then performs a single mutation whose outcome is
    fully defined for every path:

    * **Pure format errors** (malformed URL/model, or the contradictory
      remove+replacement combination) raise :class:`SetupValidationError` *before*
      ``get_or_create`` — nothing is created or mutated, a prior valid config stays
      intact. The caller re-renders with the fixed message.
    * **Stranded-dependent disable** (issue #77 — turning LLM off while a feature that
      needs it is effectively on) returns a plain message and writes NOTHING (no
      ``get_or_create``), so LLM stays on rather than the form auto-disabling an
      unrelated feature (#62). See :func:`_llm_disable_strand_error`.
    * **Validation failure** (enable requested but no effective key / budget doesn't
      fit) still persists the valid non-secret overrides and the valid candidate key
      (a good key the operator typed is not thrown away) but forces
      ``llm_enabled=False`` and returns the fixed message.
    * **Success** persists candidate key + overrides + the requested ``llm_enabled``
      and returns ``None``.

    The candidate key is: NULL when ``remove_key`` (revert to env), the new value on a
    non-blank submission, else the existing row value (blank password = no change —
    it is never prefilled). The key is a credential: it is never rendered, and the
    returned message is a fixed string that never interpolates it.
    """
    base_url = normalize_llm_base_url(raw_base_url)
    model = normalize_llm_model(raw_model)
    # Tri-state "revert to installation setting" (issue #46): a blank field already
    # normalizes to None, but a submission that merely echoes the env default (the
    # forms render blank with that default as placeholder, yet an operator may still
    # type it) must ALSO store NULL so the row keeps inheriting env — otherwise
    # saving any LLM change silently pins the env value onto the row and a later
    # LLM_BASE_URL/LLM_MODEL change stops applying with no cue.
    if base_url is not None and base_url == settings.llm_base_url:
        base_url = None
    if model is not None and model == settings.llm_model:
        model = None
    new_key = normalize_llm_api_key(raw_key)
    if remove_key and new_key is not None:
        # Contradictory: the operator both typed a replacement and asked to remove.
        # Reject as a format error so neither intent is silently applied.
        raise SetupValidationError(
            "Choose either a new LLM API key or “remove saved key”, not both."
        )
    # Issue #77: refuse a deliberate disable that would strand a feature depending on
    # LLM, BEFORE any mutation — write nothing (no get_or_create), keep LLM on rather
    # than auto-disabling the dependent (#62). Scope is deliberate-disable-only: the
    # fail-closed *enable* path below (requested-on but no usable key/budget) forces
    # llm_enabled=False and so can ITSELF leave a dependent stranded, yet it is left
    # unguarded on purpose — it already returns the operator's real problem (the
    # key/budget error), keeps the valid key they typed (#46), and fixing that key
    # re-enables LLM and un-strands the dependent. Guarding it would have to either
    # drop that key or bury the key error under a strand message. Known residual (#77).
    if not enabled:
        strand_error = _llm_disable_strand_error(get_app_settings(session), settings)
        if strand_error is not None:
            return strand_error
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    if remove_key:
        candidate_key: str | None = None
    elif new_key is not None:
        candidate_key = new_key
    else:
        candidate_key = row.llm_api_key
    # Effective key from the CANDIDATE (row-wins-over-env), matching how a run/job
    # will resolve it post-save, so the enable guard reflects the saved state.
    effective_key = (candidate_key or "").strip() or settings.llm_api_key.strip()
    error: str | None = None
    if enabled:
        # Issue #67: a keyless enable is legitimate when the bundled local model is
        # the active endpoint — resolve it from the just-created row (the operator
        # may have turned the bundle on in Features first) so the keyless audience
        # can actually flip the master LLM switch. BYO-only jobs stay key-gated.
        bundled_active = llm_bundled_active(row, settings)
        try:
            validate_llm_enable(effective_key, settings, bundled_active=bundled_active)
        except SetupValidationError as exc:
            error = str(exc)
    # Single deliberate mutation. On a validation failure we fail closed
    # (llm_enabled=False) but still keep the valid overrides + candidate key.
    row.llm_base_url = base_url
    row.llm_model = model
    row.llm_api_key = candidate_key
    row.llm_enabled = enabled and error is None
    return error


# The live-read feature flags the Settings "Features" section exposes as tri-state
# runtime toggles (issue #62). Each entry is (column/config name, operator label,
# help text). Order is display order. LLM enablement lives in its own section, and
# the web-research provider toggles are the external-sources child (#76), so
# neither appears here. The names match the ``AppSettings`` columns / ``Settings``
# fields exactly, so the resolvers and the persist path key off them directly.
_FEATURE_FLAG_META: tuple[tuple[str, str, str], ...] = (
    (
        "enrichment_names_enabled",
        "Speaker name suggestions",
        "Scan finalized transcripts for likely speaker names. Runs fully offline —"
        " no LLM required.",
    ),
    (
        "enrichment_names_llm_enabled",
        "LLM name pass",
        "Additionally ask the enhancement LLM to propose names. Requires LLM"
        " enhancement and speaker name suggestions to be on.",
    ),
    (
        "enrichment_run_assets_enabled",
        "Run assets (summary, topics, entities)",
        "Generate a summary, topic list, and grounded entity mentions for each run."
        " Requires LLM enhancement.",
    ),
    (
        "enrichment_run_assets_autogenerate",
        "Auto-generate run assets",
        "Start run-asset generation automatically when a run is finalized. Requires"
        " run assets to be on.",
    ),
    (
        "llm_bundled_enabled",
        "Use the bundled local model",
        "Route transcript enhancement and run-asset summaries + entities to the"
        " bundled local model, so they work with no external API key. It powers"
        " ONLY those — topics, web research, and LLM name suggestions still need a"
        " BYO endpoint and key. Requires LLM enhancement to be on, and the bundled"
        " model service to be running (compose.llm.yaml).",
    ),
    (
        "ytdlp_enabled",
        "Download media from a URL",
        "Allow submitting media by URL, fetched with yt-dlp. Independent of the LLM"
        " features.",
    ),
)
_FEATURE_FLAG_NAMES: tuple[str, ...] = tuple(name for name, _, _ in _FEATURE_FLAG_META)

# Structured dependency map for the Features section UI (issue #406). Each key
# is a flag that requires ALL listed prerequisites to be effectively on before
# it can be toggled. Flags not listed here are standalone. Kept separate from
# ``_FEATURE_FLAG_META`` to preserve the three-tuple arity and the plugin
# ``FeatureFlag`` seam (#138). ``llm_enabled`` is a cross-section dependency
# (AI tab), resolved from the row/settings namespace, not from this section.
#
# ``llm_bundled_enabled`` is deliberately absent: the bundled model is routing
# intent, not an invariant — the #67 keyless-adoption flow requires enabling
# the bundle FIRST (with LLM off) so a keyless operator can then flip LLM on.
_FEATURE_DEPS: dict[str, tuple[str, ...]] = {
    "enrichment_names_llm_enabled": ("llm_enabled", "enrichment_names_enabled"),
    "enrichment_run_assets_enabled": ("llm_enabled",),
    "enrichment_run_assets_autogenerate": ("enrichment_run_assets_enabled",),
}

_DEP_LABELS: dict[str, str] = {
    "llm_enabled": "LLM transcript enhancement (AI tab)",
    "enrichment_names_enabled": "speaker name suggestions",
    "enrichment_run_assets_enabled": "run assets",
}


def _effective_feature_flag_meta(
    registry: PluginRegistry,
) -> tuple[tuple[str, str, str], ...]:
    """Core Features-section flags plus any a plugin's settings section contributes.

    The Features section render loop iterates this, so a converted plugin's
    tri-state flags render alongside the core ones (issue #138, rule 3). Returns
    the core ``_FEATURE_FLAG_META`` object *unchanged* when no plugin contributes a
    flag — the #138 dormant path, byte-identical to before.

    ⚠ This is the merge MECHANISM only. A plugin flag is fully coupled to its
    ``AppSettings`` column + ``Settings`` field + ``resolve_effective_*`` helper:
    the Features render loop reads ``feature_flag_state(row, name)`` and
    ``getattr(settings, name)``, and the POST persists the column — none of which
    exist until the conversion that introduces the plugin (#141). That conversion
    must, in the same change, add the column + Settings field, the resolver, the
    ``settings_features`` POST persistence + allowlist entry, and its candidate
    invariant validation. Until then the registry is empty, so no plugin flag ever
    reaches this merge, renders, or is submitted.
    """
    plugin_flags = tuple(
        (flag.name, flag.label, flag.help_text)
        for section in registry.settings_sections()
        for flag in section.feature_flags
    )
    if not plugin_flags:
        return _FEATURE_FLAG_META
    return _FEATURE_FLAG_META + plugin_flags
_FEATURE_FLAG_CHOICES: tuple[str, ...] = ("on", "off", "inherit")

# Operator-plain copy for the invariant violations the settings sections surface
# (Features #62 + Sources & research #76). validate_effective_flags is the SINGLE
# source of WHICH combinations are invalid, but its messages name the flag
# identifiers (enrichment_run_assets_enabled, web_search_base_url, …) — the exact
# jargon this arc exists to keep out of a non-technical operator's way. So the
# settings boundaries translate the reachable messages to plain language, while the
# config boot validator keeps the identifier-bearing strings (a .env editor wants
# the variable name). Keyed on the exact shared message; an un-mapped message falls
# through to the original. A drift test locks every reachable key so a reworded
# invariant can never silently fall back to jargon here.
_FEATURE_INVARIANT_COPY: dict[str, str] = {
    "enrichment_names_llm_enabled requires llm_enabled=true — the "
    "LLM name pass reuses the configured enhancement endpoint": (
        "The LLM name pass needs LLM transcript enhancement turned on. Turn it on"
        " in the LLM section below, or turn the LLM name pass off."
    ),
    "enrichment_names_llm_enabled requires enrichment_names_enabled=true"
    " — the LLM pass is additive to the offline name producer": (
        "The LLM name pass needs speaker name suggestions turned on — it adds to"
        " the offline name finder."
    ),
    "enrichment_run_assets_enabled requires llm_enabled=true — the"
    " asset generators reuse the configured enhancement endpoint": (
        "Run assets need LLM transcript enhancement turned on. Turn it on in the"
        " LLM section below, or turn run assets off."
    ),
    "enrichment_run_assets_autogenerate requires"
    " enrichment_run_assets_enabled=true — the post-finalize step"
    " only enqueues the feature it rides on": (
        "Auto-generating run assets needs run assets turned on."
    ),
    # Sources & research section (issue #76). The producer + base-URL invariants.
    "enrichment_web_research_enabled requires voxint_web_research=true"
    " — the producer's only egress is the controlled retrieval tools": (
        "Web-research enrichment needs Web research turned on above — the"
        " producer's only way out to the network is through it."
    ),
    "enrichment_web_research_enabled requires llm_enabled=true — the"
    " producer reuses the configured enhancement endpoint": (
        "Web-research enrichment needs LLM transcript enhancement turned on. Turn"
        " it on in the LLM section, or turn web-research enrichment off."
    ),
    "voxint_web_research=true requires web_search_base_url — the"
    " searxng provider has no default endpoint": (
        "Web research needs a search provider endpoint. Enter one below, or turn"
        " Web research off."
    ),
    "web_search_base_url must not contain whitespace or backslashes": (
        "The search provider endpoint can't contain spaces or backslashes — check"
        " for a stray character."
    ),
    "web_search_base_url is malformed": (
        "The search provider endpoint isn't a valid web address — check for a typo"
        " (for example an invalid port)."
    ),
    "web_search_base_url must be an absolute http(s) URL": (
        "The search provider endpoint must be a full web address starting with"
        " http:// or https://."
    ),
    "web_search_base_url must not embed credentials — use web_search_api_key": (
        "Don't put a username or password in the endpoint — use the API key field"
        " below instead."
    ),
    "web_search_base_url must be a bare endpoint (no query/fragment)": (
        "Enter just the endpoint address — no “?query” or “#fragment” at the end."
    ),
}

_SEMANTIC_INVARIANT_COPY = (
    "Turn semantic search on before auto-indexing new runs — the automatic"
    " step only runs the feature it rides on."
)
_TRANSLATION_INVARIANT_COPY = (
    "Pick a preferred language before turning auto-translate on — the"
    " automatic step needs to know which language to translate into."
)
_SYNTHDETECT_INVARIANT_COPY = (
    "Turn AI-content detection on before auto-detecting new runs"
    " — the automatic step only runs the feature it rides on."
)


# Issue #61: plain-language remediation for the wizard SERVICES step's readiness
# checks (``voxint doctor`` surfaced in the browser). Keyed by a coarse CATEGORY, not
# the raw diagnostics check name, because the three model services carry distinct
# names (transcription / diarization / speaker embedding) that share one fix. Shown
# only when a check is not ``ready``. The model copy is warm-up-aware: a reachable
# service whose model is still loading reads as ``failed`` (the pipeline genuinely
# can't run yet), so the text names "starting" so it isn't misread as a crash.
_DOCTOR_REMEDIATION: dict[str, str] = {
    "database": (
        "Start Postgres, then re-check. Voxint keeps every run and its transcript"
        " here — nothing can be submitted until it's up."
    ),
    "redis": (
        "Start Redis, then re-check. It's the task queue — submissions wait here to"
        " be picked up, so the pipeline can't run without it."
    ),
    "models": (
        "Start the model services — the GPU overlay (compose.gpu.yaml) or, with no"
        " NVIDIA GPU, the CPU overlay (compose.cpu.yaml); see the README quickstart."
        " If one just started it may still be loading its model — re-check in a"
        " moment."
    ),
    "llm": (
        "LLM transcript enhancement is on but the endpoint didn't answer (or rejected"
        " the check) — check the LLM section in Settings. Transcription and diarization"
        " still run; enhancement is simply skipped until the endpoint is reachable."
    ),
    # The bundled lane fails differently from the BYO one: the fix is to start
    # the bundled model service (or turn the bundle off), never to edit the BYO
    # endpoint settings — steering the operator there would be a wrong map.
    "llm bundled": (
        "The bundled AI model is turned on but isn't answering. Start its service"
        " (the compose LLM overlay, or the native launcher's model services), or"
        " turn the bundled model off in Settings. Transcription and diarization"
        " still run; enhancement is simply skipped until it's reachable."
    ),
    # Fallback for any diagnostics check not explicitly categorized below — a neutral
    # "look at this dependency" rather than wrongly steering the operator at the model
    # services. No current check lands here (see _doctor_category), but a future one
    # gets honest generic copy instead of an empty string or a misleading fix.
    "other": (
        "This dependency isn't ready — check its configuration in Settings or the"
        " README, then re-check."
    ),
}


def _doctor_category(name: str) -> str:
    """Map a diagnostics :class:`~voxint.diagnostics.CheckResult` name to a
    ``_DOCTOR_REMEDIATION`` category. Total by construction — an unrecognized name (a
    future check) falls through to the neutral ``other`` copy, never a KeyError and
    never the wrong (model-services) remediation."""
    if name == "postgres":
        return "database"
    if name == "redis":
        return "redis"
    if name in {"transcription", "diarization", "speaker embedding"}:
        return "models"
    if name == "llm endpoint":
        return "llm"
    if name == "llm bundled":
        return "llm bundled"
    return "other"


def _doctor_checks(request: Request, session: Session) -> list[dict[str, Any]]:
    """Run the wizard SERVICES readiness checks (issue #61) and shape them for the
    template. Self-contains every dependency failure (DB / redis / model / LLM), so
    it never raises into the request — the caller can render a 200 regardless."""
    settings: Settings = request.app.state.settings
    engine = cast(Engine, session.get_bind())
    with httpx.Client(timeout=httpx.Timeout(settings.health_probe_timeout_seconds)) as client:
        results = run_diagnostics(
            settings, engine, http_client=client, include_hf_token=False
        )
    return [
        {
            "name": r.name,
            "state": check_state(r),
            "detail": r.detail,
            "remediation": _DOCTOR_REMEDIATION[_doctor_category(r.name)],
        }
        for r in results
    ]


# ---- R6 Status page: component list + hardware gauges (issue #215) --------

_COMPONENT_LABELS: dict[str, str] = {
    "postgres": "Database",
    "redis": "Task queue",
    "transcription": "Transcriber",
    "diarization": "Voice separation",
    "speaker embedding": "Voice identity",
    # #316: the one "Local AI model" row painted a healthy bundled-only install
    # as rejected (it only ever probed the BYO endpoint). The two AI lanes are
    # independent capabilities with independent health, so they get one row each.
    "llm bundled": "Bundled AI model",
    "llm endpoint": "Your own AI endpoint",
}

_COMPONENT_ORDER = (
    "__api__",
    "transcription",
    "diarization",
    "speaker embedding",
    "postgres",
    "redis",
    "llm bundled",
    "llm endpoint",
)


def _build_components(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map doctor checks to the R6 component list with friendly names.

    Adds a synthetic "Console & API" entry (the app is serving this page, so it
    is always running). Sorts to the stable display order.

    The two AI lanes (#316) key their off-states on check PRESENCE, never on the
    env flags: ``check_llm``/``check_llm_bundled`` return no result when a lane
    is effectively off (row-over-env resolved inside ``run_diagnostics``), so a
    present, non-ready check always warns. Re-reading ``settings.llm_enabled``
    here would consult env only and paint the standard UI-enabled path
    (env off, row on) as "off" while its deliberately configured endpoint is
    rejecting — exactly the false calm #316 removes.
    """
    by_name: dict[str, dict[str, Any]] = {c["name"]: c for c in checks}
    rows: list[dict[str, Any]] = []
    for key in _COMPONENT_ORDER:
        if key == "__api__":
            rows.append({
                "label": "Console & API",
                "dot": "ok",
                "state_text": "running",
                "action_url": None,
                "action_label": None,
                "action_style": None,
            })
            continue
        check = by_name.get(key)
        label = _COMPONENT_LABELS.get(key, key)
        action_url: str | None = None
        action_label: str | None = None
        action_style: str | None = None
        if check is None:
            if key == "llm bundled":
                # The bundle is not active for runs (not installed, not enabled,
                # or the master LLM switch is off): an honest "off" row, not a
                # warning — many installs never add it.
                dot = "off"
                state_text = "off"
            elif key == "llm endpoint":
                # LLM work is effectively disabled; the row stays visible so the
                # operator can discover the capability.
                dot = "off"
                state_text = "off -- used for polish & profiles"
                action_url = "/settings#llm"
                action_label = "Turn on"
                action_style = "primary"
            else:
                continue
        else:
            state = check["state"]
            if (
                key == "llm endpoint"
                and state == "ready"
                and check["detail"] == LLM_NOT_CONFIGURED_DETAIL
            ):
                # #316: enabled, but no deliberate BYO endpoint (the untouched
                # install default). Not probed, not a warning — the bundled row
                # above reports the lane that is actually in use.
                dot = "off"
                state_text = LLM_NOT_CONFIGURED_DETAIL
                action_url = "/settings#llm"
                action_label = "Set up"
                action_style = "primary"
            elif state == "ready":
                dot = "ok"
                state_text = f"running · {check['detail']}" if check["detail"] else "running"
            else:
                dot = "warn"
                state_text = check["detail"] or state
        rows.append({
            "label": label,
            "dot": dot,
            "state_text": state_text,
            "action_url": action_url,
            "action_label": action_label,
            "action_style": action_style,
        })
    return rows


def _build_gauges(
    snapshot: ResourceSnapshot,
    host: HostMetricsSnapshot,
) -> list[dict[str, Any]]:
    """Build the hardware gauge rows from resource and host snapshots."""

    gauges: list[dict[str, Any]] = []
    if host.cpu_percent is not None:
        gauges.append({
            "label": "Processor",
            "value": f"{host.cpu_percent}%",
            "percent": host.cpu_percent,
        })
    if (
        host.memory_used_bytes is not None
        and host.memory_total_bytes is not None
        and host.memory_total_bytes > 0
    ):
        used_gb = host.memory_used_bytes / (1024**3)
        total_gb = host.memory_total_bytes / (1024**3)
        pct = round(100 * host.memory_used_bytes / host.memory_total_bytes)
        gauges.append({
            "label": "Memory",
            "value": f"{used_gb:.1f} / {total_gb:.0f} GB",
            "percent": pct,
        })
    for gpu in snapshot.gpus:
        if gpu.utilization_percent is not None:
            gauges.append({
                "label": "Graphics card",
                "value": f"{gpu.utilization_percent}%",
                "percent": gpu.utilization_percent,
            })
        vram_pct = vram_percent(gpu.vram_used_bytes, gpu.vram_total_bytes)
        if vram_pct is not None:
            assert gpu.vram_used_bytes is not None
            assert gpu.vram_total_bytes is not None
            used_gb = gpu.vram_used_bytes / (1024**3)
            total_gb = gpu.vram_total_bytes / (1024**3)
            gauges.append({
                "label": "Graphics memory",
                "value": f"{used_gb:.1f} / {total_gb:.0f} GB",
                "percent": vram_pct,
            })
    if (
        host.disk_used_bytes is not None
        and host.disk_total_bytes is not None
        and host.disk_total_bytes > 0
    ):
        used_gb = host.disk_used_bytes / (1024**3)
        total_gb = host.disk_total_bytes / (1024**3)
        pct = round(100 * host.disk_used_bytes / host.disk_total_bytes)
        gauges.append({
            "label": "Disk (media)",
            "value": f"{used_gb:.0f} / {total_gb:.0f} GB",
            "percent": pct,
        })
    return gauges


def _install_summary(settings: Settings, snapshot: ResourceSnapshot) -> str:
    """One-line install summary for the status banner."""
    import voxint

    kind = settings_view.install_kind(settings)
    parts = [f"{kind} install" if kind != "unknown" else "Install type unknown"]
    if settings.compute_tier == "cpu":
        parts.append("GPU acceleration off")
    elif snapshot.gpus:
        gpu = snapshot.gpus[0]
        if gpu.vram_total_bytes is not None:
            total_gb = round(gpu.vram_total_bytes / (1024**3))
            parts.append(f"GPU acceleration on ({total_gb} GB)")
        else:
            parts.append("GPU acceleration on")
    parts.append(f"version {voxint.__version__}")
    return " · ".join(parts)


def _minimal_services_context(request: Request, settings: Settings) -> dict[str, Any]:
    """DB-free context for the SERVICES step when the ``app_settings`` read itself
    fails — a down or unmigrated Postgres makes ``_setup_context``'s row + tutorial
    queries raise. Showing that (a failed ``postgres`` readiness row) is the whole
    point of this step, so it must still render rather than 500. Carries only the keys
    the SERVICES branch + the shared nav/layout touch; the other steps' DB-derived
    fields are absent (the Jinja env uses lenient ``Undefined`` and those fields are
    referenced only inside other steps' branches). The caller attaches the checks."""
    return {
        "request": request,
        "step": WizardStep.SERVICES,
        "steps": STEP_ORDER,
        "step_index": STEP_ORDER.index(WizardStep.SERVICES),
        "next_step": next_step(WizardStep.SERVICES),
        "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
        "media_root": str(settings.media_root),
        "active_nav": "setup",
        "error": None,
    }


def _services_step_response(request: Request, session: Session) -> Response:
    """Render the wizard SERVICES step with the full doctor readiness checks (#61).

    The checks run first — they self-contain every failure, so this GET is a 200 even
    with every dependency down — then the page context is built, falling back to a
    DB-free context when the ``app_settings`` read raises (Postgres down/unmigrated) so
    the operator still sees the failed-postgres row instead of a 500."""
    settings: Settings = request.app.state.settings
    checks = _doctor_checks(request, session)
    try:
        context = _setup_context(request, session, WizardStep.SERVICES)
    except SQLAlchemyError:
        # The poisoned session must be rolled back so the trailing commit in
        # _get_session is a harmless no-op rather than re-raising.
        session.rollback()
        context = _minimal_services_context(request, settings)
    context["doctor_checks"] = checks
    return templates.TemplateResponse(request, "settings/setup.html", context)


def _reconcile_switches(
    flag_names: Sequence[str],
    form_data: FormData,
    settings: Settings,
    row: AppSettings | None,
) -> dict[str, str]:
    """Convert switch checkbox submissions to tri-state strings for persisters.

    Each flag rendered as an enabled switch emits a hidden ``_rendered`` marker.
    Flags NOT in ``_rendered`` (disabled due to unmet dependency) are preserved
    at their stored raw state so the save never flips a disabled flag.

    **Save idempotency**: when a flag is stored as ``inherit`` (NULL) and the
    submitted effective value equals the environment default, the reconciled
    value stays ``"inherit"`` — clicking Save without touching anything never
    converts an inherited flag into a pinned override.
    """
    rendered = set(form_data.getlist("_rendered"))
    result: dict[str, str] = {}
    for name in flag_names:
        if name not in rendered:
            result[name] = feature_flag_state(row, name)
            continue
        desired_on = form_data.get(name) == "on"
        stored = feature_flag_state(row, name)
        env_default = bool(getattr(settings, name))
        if stored == "inherit" and desired_on == env_default:
            result[name] = "inherit"
        else:
            result[name] = "on" if desired_on else "off"
    return result


def _handle_reset_flag(
    session: Session,
    settings: Settings,
    request: Request,
    flag_name: str,
    tab: str,
) -> Response:
    """Reset a single tri-state flag to inherit (NULL) and redirect.

    After setting the column to NULL the resulting effective combination is
    validated through the same invariant web the persist paths use.  If the
    reset would NEWLY introduce a violation the transaction is rolled back and
    the page re-renders with a section notice (matching the tab POST save
    error pattern).  The before/after delta discipline (cf.
    ``_llm_disable_strand_error``) ensures a pre-existing unrelated violation
    never blocks or mislabels the cause (#404).
    """
    if flag_name in _RESETTABLE_FLAGS:
        row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
        errors = _reset_invariant_errors(row, settings, flag_name)
        if errors:
            session.rollback()
            section_key = _RESET_ERROR_SECTION.get(flag_name, "features_errors")
            overrides: dict[str, Any] = {
                section_key: [f"Reset not applied — {e}" for e in errors],
            }
            return templates.TemplateResponse(
                request,
                _settings_page_template(request),
                _settings_context(request, session, **overrides),
            )
        session.commit()

    anchor = f"sw-{flag_name}"
    return RedirectResponse(
        _settings_redirect(request, anchor, tab),
        status_code=303,
    )


def _effective_invariant_messages(
    row: AppSettings | None, settings: Settings
) -> list[str]:
    """Operator-plain invariant violations for the row's current effective state.

    Collects translated messages from all four invariant domains (feature flags,
    semantic index, translation, synthdetect) so the reset path can compute a
    delta without duplicating validator wiring.
    """
    errors = [
        _FEATURE_INVARIANT_COPY.get(message, message)
        for message in validate_effective_flags(
            EffectiveFlags(
                llm_enabled=resolve_effective_llm_enabled(row, settings),
                enrichment_names_enabled=resolve_effective_enrichment_names_enabled(
                    row, settings
                ),
                enrichment_names_llm_enabled=resolve_effective_enrichment_names_llm_enabled(
                    row, settings
                ),
                enrichment_run_assets_enabled=resolve_effective_enrichment_run_assets_enabled(
                    row, settings
                ),
                enrichment_run_assets_autogenerate=resolve_effective_enrichment_run_assets_autogenerate(
                    row, settings
                ),
                voxint_web_research=resolve_effective_voxint_web_research(row, settings),
                enrichment_web_research_enabled=resolve_effective_enrichment_web_research_enabled(
                    row, settings
                ),
                web_search_base_url=resolve_effective_web_search_base_url(row, settings),
            )
        )
    ]
    if (
        semantic_index_flags_ok(
            enabled=resolve_effective_semantic_index_enabled(row, settings),
            autogenerate=resolve_effective_semantic_index_autogenerate(row, settings),
        )
        is not None
    ):
        errors.append(_SEMANTIC_INVARIANT_COPY)
    if (
        translation_flags_ok(
            autogenerate=resolve_effective_translation_autogenerate(row, settings),
            target_language=resolve_effective_translation_target_language(row, settings),
        )
        is not None
    ):
        errors.append(_TRANSLATION_INVARIANT_COPY)
    if (
        synthdetect_flags_ok(
            enabled=resolve_effective_synthdetect_enabled(row, settings),
            autogenerate=resolve_effective_synthdetect_autogenerate(row, settings),
        )
        is not None
    ):
        errors.append(_SYNTHDETECT_INVARIANT_COPY)
    return errors


def _reset_invariant_errors(
    row: AppSettings, settings: Settings, flag_name: str
) -> list[str]:
    """Violations the reset would NEWLY introduce (delta discipline, #404).

    Snapshots the current invariant surface, applies the tentative NULL, then
    returns only the messages that were absent before.  A pre-existing unrelated
    violation never blocks an unrelated reset and never mislabels the cause.
    The caller is responsible for rollback on a non-empty return.
    """
    before = set(_effective_invariant_messages(row, settings))
    setattr(row, flag_name, None)
    return [m for m in _effective_invariant_messages(row, settings) if m not in before]


def _persist_feature_flags(
    session: Session, settings: Settings, *, submitted: dict[str, str]
) -> list[str]:
    """Apply the Features-section tri-state toggles as ONE deliberate mutation (#62).

    ``submitted`` maps each flag name to ``"on"``/``"off"``/``"inherit"``. The
    candidate column value is ``True``/``False``/``None`` respectively (``None`` =
    inherit the env default — the tri-state that never permanently pins an
    override). Returns the list of operator-plain error messages (empty ⇒ success);
    following the ``_persist_llm_settings`` contract, it computes the candidate
    effective combination and validates it through the SINGLE shared
    :func:`validate_effective_flags` BEFORE touching the row, so an
    invariant-violating submission (e.g. the LLM name pass without LLM enhancement)
    writes NOTHING — not even a get_or_create (the API session commits on the 200
    error re-render). A valid submission then performs the single mutation and the
    caller commits.

    Choice-membership validation is retained as defense in depth, but callers
    now pass through ``_reconcile_switches`` first, which normalises checkbox
    encoding to ``"on"``/``"off"``/``"inherit"`` — so the reject branch is
    unreachable under normal operation.

    The flags NOT edited here (``llm_enabled`` and the web-research provider trio)
    are resolved at their CURRENT effective value so a dependency invariant fires
    against the real system state — enabling ``run_assets`` while LLM is off is
    rejected — and this section never flips an unrelated setting.
    """
    row = get_app_settings(session)
    candidates: dict[str, bool | None] = {}
    for name in _FEATURE_FLAG_NAMES:
        choice = submitted.get(name, "inherit")
        if choice not in _FEATURE_FLAG_CHOICES:
            return ["Unrecognized feature setting — choose On, Off, or Use installation setting."]
        candidates[name] = None if choice == "inherit" else (choice == "on")

    def _effective(name: str) -> bool:
        candidate = candidates[name]
        return bool(getattr(settings, name)) if candidate is None else candidate

    errors = validate_effective_flags(
        EffectiveFlags(
            # Not edited in this section — resolved at the current effective value.
            llm_enabled=resolve_effective_llm_enabled(row, settings),
            voxint_web_research=resolve_effective_voxint_web_research(row, settings),
            enrichment_web_research_enabled=resolve_effective_enrichment_web_research_enabled(
                row, settings
            ),
            web_search_base_url=resolve_effective_web_search_base_url(row, settings),
            # Edited here — the candidate over env default.
            enrichment_names_enabled=_effective("enrichment_names_enabled"),
            enrichment_names_llm_enabled=_effective("enrichment_names_llm_enabled"),
            enrichment_run_assets_enabled=_effective("enrichment_run_assets_enabled"),
            enrichment_run_assets_autogenerate=_effective(
                "enrichment_run_assets_autogenerate"
            ),
        )
    )
    if errors:
        # Translate every violated invariant to operator-plain copy (all of them,
        # so a two-fault submission fixes in one pass). Nothing is written.
        return [_FEATURE_INVARIANT_COPY.get(message, message) for message in errors]
    # Valid → one mutation. get_or_create only now, so a rejected save writes nothing.
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    for name, value in candidates.items():
        setattr(row, name, value)
    return []


# The two live-read flags the Settings "Semantic search" section exposes as
# tri-state toggles (issue #121). Kept out of _FEATURE_FLAG_META on purpose: the
# semantic pair depends on nothing else and validates through its OWN self-contained
# invariant (semantic_index_flags_ok), never the EffectiveFlags web, matching the
# app_settings design note that these flags stay out of EffectiveFlags.
_SEMANTIC_FLAG_NAMES: tuple[str, ...] = (
    "semantic_index_enabled",
    "semantic_index_autogenerate",
)

_RESETTABLE_FLAGS: frozenset[str] = frozenset(
    _FEATURE_FLAG_NAMES
    + _SEMANTIC_FLAG_NAMES
    + (
        "translation_autogenerate",
        "watch_folder_enabled",
        "voxint_web_research",
        "enrichment_web_research_enabled",
        "synthdetect_enabled",
        "synthdetect_autogenerate",
    )
)

_RESET_ERROR_SECTION: dict[str, str] = {
    **{name: "features_errors" for name in _FEATURE_FLAG_NAMES},
    **{name: "semantic_index_errors" for name in _SEMANTIC_FLAG_NAMES},
    "translation_autogenerate": "translation_errors",
    "synthdetect_enabled": "synthdetect_errors",
    "synthdetect_autogenerate": "synthdetect_errors",
    "voxint_web_research": "web_research_errors",
    "enrichment_web_research_enabled": "web_research_errors",
}


def _persist_semantic_index(
    session: Session,
    settings: Settings,
    *,
    submitted: dict[str, str],
) -> list[str]:
    """Apply the Semantic search section's two tri-state toggles atomically (#121).

    Mirrors :func:`_persist_feature_flags`' candidate → validate → one-mutation
    shape, but validates the semantic pair's OWN self-contained invariant
    (:func:`semantic_index_flags_ok`: autogenerate rides on the feature, so it
    requires the feature enabled) rather than the EffectiveFlags web. The effective
    combination (candidate over env default) is checked BEFORE any get_or_create, so
    a rejected save writes NOTHING (the API session commits on the 200 error
    re-render). An unexpected choice is rejected, never coerced, for the same reason
    the Features section rejects one: silently mapping it would drop an override or
    disable a feature. Returns operator-plain error messages (empty ⇒ success); the
    caller commits on success.
    """
    candidates: dict[str, bool | None] = {}
    for name in _SEMANTIC_FLAG_NAMES:
        choice = submitted.get(name, "inherit")
        if choice not in _FEATURE_FLAG_CHOICES:
            return [
                "Unrecognized semantic-search setting — choose On, Off, or Use"
                " installation setting."
            ]
        candidates[name] = None if choice == "inherit" else (choice == "on")

    def _effective(name: str) -> bool:
        candidate = candidates[name]
        return bool(getattr(settings, name)) if candidate is None else candidate

    # semantic_index_flags_ok is the single source of WHICH combination is invalid;
    # its message names the flag identifiers (for a .env editor), so the operator
    # boundary translates the one reachable violation to plain language.
    if (
        semantic_index_flags_ok(
            enabled=_effective("semantic_index_enabled"),
            autogenerate=_effective("semantic_index_autogenerate"),
        )
        is not None
    ):
        return [_SEMANTIC_INVARIANT_COPY]
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    for name, value in candidates.items():
        setattr(row, name, value)
    return []


def _persist_translation(
    session: Session,
    settings: Settings,
    *,
    submitted: dict[str, str],
) -> list[str]:
    """Apply the Translation section atomically (#133).

    Two controls save together: the preferred-target-language select (a string
    override — "inherit" writes NULL, a code writes the override) and the
    autogenerate tri-state. The effective combination (candidates over env
    defaults) is validated through the shared :func:`translation_flags_ok`
    BEFORE any get_or_create, so a rejected save writes NOTHING. Returns
    operator-plain error messages (empty ⇒ success); the caller commits.
    """
    target_choice = submitted.get("translation_target_language", "inherit").strip()
    if target_choice != "inherit" and target_choice.lower() not in LANGUAGE_NAMES:
        return [
            "Unrecognized language choice — pick a language from the list or"
            " Use installation setting."
        ]
    auto_choice = submitted.get("translation_autogenerate", "inherit")
    if auto_choice not in _FEATURE_FLAG_CHOICES:
        return [
            "Unrecognized auto-translate setting — choose On, Off, or Use"
            " installation setting."
        ]
    target_candidate = None if target_choice == "inherit" else target_choice.lower()
    auto_candidate = None if auto_choice == "inherit" else (auto_choice == "on")

    effective_target = (
        target_candidate
        if target_candidate is not None
        else (
            settings.translation_target_language.strip()
            if settings.translation_target_language is not None
            and settings.translation_target_language.strip()
            else None
        )
    )
    effective_auto = (
        bool(settings.translation_autogenerate) if auto_candidate is None else auto_candidate
    )
    if (
        translation_flags_ok(autogenerate=effective_auto, target_language=effective_target)
        is not None
    ):
        return [_TRANSLATION_INVARIANT_COPY]
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.translation_target_language = target_candidate
    row.translation_autogenerate = auto_candidate
    return []


def _persist_web_research(
    session: Session,
    settings: Settings,
    *,
    submitted_master: str,
    submitted_producer: str,
    raw_base_url: str,
    raw_key: str,
    remove_key: bool,
    raw_domains: str,
) -> list[str]:
    """Apply the Sources & research section as ONE deliberate mutation (issue #76).

    The web-research capability's five operator controls — the master toggle
    (``voxint_web_research``) + producer toggle (``enrichment_web_research_enabled``)
    as tri-state on/off/inherit, the provider ``web_search_base_url`` (string
    override) + ``web_search_api_key`` (secret), and the ``source_authority_domains``
    editor — jointly configure one capability, so they save atomically: the whole
    form is a candidate that is validated against EVERY error source before the row
    is touched, and ANY violation (out-of-set choice, cross-flag invariant, a
    malformed endpoint, a malformed domain, or the contradictory key remove+replace)
    writes NOTHING (no ``get_or_create`` — the API session commits on the 200 error
    re-render, so the absence of a write is the guarantee). Returns the list of
    operator-plain error messages (empty ⇒ success); the caller commits on success.

    The flags NOT edited here (``llm_enabled`` + the names/run-assets flags) are
    resolved at their CURRENT effective value so a dependency invariant fires against
    real system state (enabling the producer while LLM is off is rejected) and this
    section never flips an unrelated setting. The API key is a credential: it never
    leaves this function and no returned message interpolates it.
    """
    row = get_app_settings(session)
    errors: list[str] = []

    # Tri-state toggles → True / False / None (inherit). An unrecognized choice (a
    # stale client / hand-crafted POST — never the shipped radios) is rejected, not
    # coerced (coercing would silently disable or drop an override), mirroring
    # _persist_feature_flags.
    if (
        submitted_master not in _FEATURE_FLAG_CHOICES
        or submitted_producer not in _FEATURE_FLAG_CHOICES
    ):
        return ["Unrecognized setting — choose On, Off, or Use installation setting."]
    cand_master = None if submitted_master == "inherit" else submitted_master == "on"
    cand_producer = None if submitted_producer == "inherit" else submitted_producer == "on"

    # String override candidates (base URL + domains): blank → NULL (inherit); a
    # value that merely echoes the env default → NULL too, so saving never silently
    # pins the env value onto the row (issue #46 tri-state precedent). Stored
    # verbatim-stripped; the runtime parser/validator re-reads the effective value.
    stripped_base = raw_base_url.strip()
    if not stripped_base:
        cand_base: str | None = None
    elif stripped_base == settings.web_search_base_url:
        cand_base = None
    else:
        cand_base = stripped_base
    stripped_domains = raw_domains.strip()
    if not stripped_domains:
        cand_domains: str | None = None
    elif stripped_domains == settings.source_authority_domains.strip():
        cand_domains = None
    else:
        cand_domains = stripped_domains

    # Secret candidate (the _persist_llm_settings precedent): blank = keep stored
    # (never prefilled), remove = NULL (revert to env), typed = the new value. The
    # contradictory remove+replacement combination is rejected.
    typed_key = bool(raw_key.strip())
    try:
        new_key = normalize_web_search_api_key(raw_key)
    except SetupValidationError as exc:
        new_key = None
        errors.append(str(exc))
    # Detect the contradiction from the RAW submission, so remove + a *malformed*
    # replacement still reports it in the same pass (a normalize failure nulls
    # new_key, which would otherwise hide the contradiction until a second attempt).
    if remove_key and typed_key:
        errors.append(
            "Choose either a new web-search API key or “remove saved key”, not both."
        )
    if remove_key:
        cand_key: str | None = None
    elif new_key is not None:
        cand_key = new_key
    else:
        cand_key = row.web_search_api_key if row is not None else None

    # Domain-format errors (strict — rejects exactly what the runtime parser drops).
    # Validate only what would actually be STORED: a blank or env-echoed submission
    # collapses to NULL (inherit env, permissive at runtime), so there is nothing to
    # reject — an operator who leaves the field alone is never shown a domain error.
    if cand_domains is not None:
        errors.extend(validate_authority_domains(cand_domains))

    # Cross-flag invariants over the effective (candidate-over-env) combination. The
    # effective base URL is validated HERE by validate_effective_flags when the
    # master toggle is effectively on (its voxint_web_research ⇒ valid-base-url rule).
    effective_master = bool(settings.voxint_web_research) if cand_master is None else cand_master
    effective_producer = (
        bool(settings.enrichment_web_research_enabled) if cand_producer is None else cand_producer
    )
    effective_base = cand_base if cand_base is not None else settings.web_search_base_url
    errors.extend(
        validate_effective_flags(
            EffectiveFlags(
                # Not edited in this section — resolved at the current effective value
                # so a dependency (producer ⇒ llm) fires against real state.
                llm_enabled=resolve_effective_llm_enabled(row, settings),
                enrichment_names_enabled=resolve_effective_enrichment_names_enabled(row, settings),
                enrichment_names_llm_enabled=resolve_effective_enrichment_names_llm_enabled(
                    row, settings
                ),
                enrichment_run_assets_enabled=resolve_effective_enrichment_run_assets_enabled(
                    row, settings
                ),
                enrichment_run_assets_autogenerate=(
                    resolve_effective_enrichment_run_assets_autogenerate(row, settings)
                ),
                # Edited here — the candidate over env default.
                voxint_web_research=effective_master,
                enrichment_web_research_enabled=effective_producer,
                web_search_base_url=effective_base,
            )
        )
    )
    # Standalone base-URL validation covers the complementary case only: an explicit
    # override saved WHILE the master toggle is off (validate_effective_flags checks
    # the base URL only when master is on, so this catches a latent-malformed
    # override that would break the instant master flips on). The two branches are
    # mutually exclusive, so no message is ever produced twice.
    if not effective_master and cand_base is not None:
        url_error = validate_web_search_base_url(cand_base)
        if url_error is not None:
            errors.append(url_error)

    if errors:
        # Translate identifier-bearing invariant messages to operator-plain copy;
        # nothing is written.
        return [_FEATURE_INVARIANT_COPY.get(message, message) for message in errors]

    # Valid → one mutation. get_or_create only now, so a rejected save writes nothing.
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.voxint_web_research = cand_master
    row.enrichment_web_research_enabled = cand_producer
    row.web_search_base_url = cand_base
    row.web_search_api_key = cand_key
    row.source_authority_domains = cand_domains
    return []


# ---- First-run setup wizard (issue #3) -------------------------------------
# Every wizard route is registered on `app`, NOT `protected`, so the onboarding
# gate exempts it: an un-onboarded operator must be able to reach the page the
# gate redirects them to. Auth still applies (OperatorDep) — only /healthz is
# unauthenticated. Each POST verifies CSRF_SETUP before any write. The exact
# paths below are enumerated in the route-inventory test's exempt allowlist
# (deliberately NOT a blanket /setup prefix, so an accidental ungated route
# still fails that guard).

@setup_router.get("/setup", include_in_schema=False)
def setup(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    step = parse_step(request.query_params.get("step"))
    if step is WizardStep.SERVICES:
        # The services step surfaces the full `voxint doctor` readiness checks
        # (issue #61) — a few-second network op we don't pay on other steps.
        # include_hf_token=False drops the Hugging Face check: the default install
        # runs on vendored weights, so it's noise AND a live huggingface.co call
        # this step has no reason to make. _services_step_response renders a 200
        # even when Postgres itself is down (that's exactly what it must show).
        return _services_step_response(request, session)
    context = _setup_context(request, session, step)
    return templates.TemplateResponse(request, "settings/setup.html", context)

def _setup_redirect(step: WizardStep) -> RedirectResponse:
    return RedirectResponse(f"/setup?step={step.value}", status_code=303)

# Folder registration on the media step is the browser panel (issue #63):
# POST /setup/folders, not a bulk textarea. The old POST /setup/media route
# was removed with the textarea.

def _scan_response(request: Request, session: Session, result: ScanResult) -> Response:
    """htmx → the scan preview/result fragment; a plain POST → back to the step.

    The scan feature is htmx-driven (a fragment swapped into the media step);
    without htmx it degrades to a redirect and the operator submits media
    individually — the scan is an optional convenience, never the only path.
    """
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "settings/setup_scan.html",
            {
                "request": request,
                "result": result,
                "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
            },
        )
    return _setup_redirect(WizardStep.MEDIA)

@setup_router.post("/setup/scan", include_in_schema=False)
def setup_scan(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings
    folders = registered_folder_paths(session)
    result = scan_media_folders(session, settings.media_root, folders, settings)
    return _scan_response(request, session, result)

@setup_router.post("/setup/scan/confirm", include_in_schema=False)
def setup_scan_confirm(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings
    folders = registered_folder_paths(session)
    # Re-scan fresh rather than trusting any client-supplied path list: the
    # preview is advisory and the filesystem may have changed since it rendered.
    result = scan_media_folders(session, settings.media_root, folders, settings)
    # submit_media_item_if_new is race-safe and returns None for an already-known
    # path, so a double-clicked confirm or a concurrent one cannot duplicate runs.
    # A freeze-time domain-pack collision (issue #84) / unresolvable pack (issue
    # #11) aborts the batch with a plain-language 422 rather than a raw 500 — no
    # run is committed (the commit is a single call below), so nothing is stranded.
    try:
        submissions = [
            sub
            for path in result.candidates
            if (sub := submit_media_item_if_new(session, path)) is not None
        ]
    except DomainPackError as exc:
        raise HTTPException(
            status_code=422, detail=deps._submit_domain_pack_detail(exc)
        ) from exc
    # Commit the whole batch ONCE (commit-before-publish); if the commit fails,
    # nothing is published and no partial state escapes.
    session.commit()
    # Publish each committed run. On the FIRST failure the broker is down, so
    # stop retrying (each apply_async pays a connect timeout and stalls the
    # request) and leave the rest deferred — the recovery sweep owns every
    # durable QUEUED row regardless.
    published = 0
    batch_limit = settings.watch_folder_batch_size
    broker_down = False
    for i, sub in enumerate(submissions):
        if broker_down or i >= batch_limit:
            break
        if sub.publish():
            published += 1
        else:
            broker_down = True
    confirmed = ScanResult(
        candidates=[],
        inspected=result.inspected,
        hit_entry_cap=result.hit_entry_cap,
        hit_file_cap=result.hit_file_cap,
        root_missing=result.root_missing,
    )
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "settings/setup_scan.html",
            {
                "request": request,
                "result": confirmed,
                "queued": len(submissions),
                "published": published,
                "broker_down": broker_down,
                "csrf_setup": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP),
            },
        )
    return _setup_redirect(WizardStep.MEDIA)

# ---- Folder browser + per-folder domain packs (issue #63) --------------
# Two routes per mount: a read-only browse GET (no CSRF — authenticated,
# bounded, never creates the row) and one mutate POST carrying an action verb.
# An HX request gets the panel fragment back; a plain request degrades to a
# full-page redirect on success, or a full-page re-render carrying the error
# inline on failure (mirrors _roster_response) — never a silent reload that
# discards the message and looks like the mutation succeeded.

def _folder_panel_response(
    request: Request,
    session: Session,
    settings: Settings,
    *,
    action_prefix: str,
    csrf_action: str,
    path: str,
    error: str | None,
    redirect_url: str,
    error_page: Callable[[str], Response],
) -> Response:
    if not request.headers.get("HX-Request"):
        if error is not None:
            # No-JS failure: re-render the whole page with the message inline
            # (and the browse position preserved), not a 303 that drops it.
            return error_page(error)
        return RedirectResponse(redirect_url, status_code=303)
    csrf = mint_csrf_token(request.app.state.csrf_secret, csrf_action)
    context = _folder_panel_context(
        session, settings, action_prefix=action_prefix, csrf=csrf, path=path
    )
    context["folder_error"] = error
    response = templates.TemplateResponse(
        request, "settings/folder_panel.html", {"request": request, **context}
    )
    # Authenticated fragment: keep it out of any shared/browser cache.
    response.headers["Cache-Control"] = "no-store"
    return response

@setup_router.get("/setup/folders/browse", include_in_schema=False)
def setup_folders_browse(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    path: Annotated[str, Query(max_length=4096)] = ".",
) -> Response:
    settings: Settings = request.app.state.settings
    csrf = mint_csrf_token(request.app.state.csrf_secret, CSRF_SETUP)
    context = _folder_panel_context(
        session, settings, action_prefix="/setup/folders", csrf=csrf, path=path
    )
    response = templates.TemplateResponse(
        request, "settings/folder_panel.html", {"request": request, **context}
    )
    response.headers["Cache-Control"] = "no-store"
    return response

@setup_router.post("/setup/folders", include_in_schema=False)
def setup_folders(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    action: Annotated[str, Form(max_length=16)] = "",
    folder: Annotated[str, Form(max_length=4096)] = "",
    pack: Annotated[str | None, Form(max_length=200)] = None,
    path: Annotated[str, Form(max_length=4096)] = ".",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings
    error = _apply_folder_mutation(
        session, settings, action=action, folder=folder, pack=pack
    )
    if error is None:
        session.commit()
    else:
        # A 200 re-render would otherwise commit the get_or_create insert /
        # the FOR UPDATE lock's transaction — roll back so a failed mutation
        # writes nothing (mirrors the wizard's other error re-renders).
        session.rollback()
    redirect = "/setup?" + urlencode({"step": "media", "path": path})

    def _error_page(message: str) -> Response:
        return templates.TemplateResponse(
            request,
            "settings/setup.html",
            _setup_context(
                request, session, WizardStep.MEDIA, folder_path=path,
                folder_error=message,
            ),
        )

    return _folder_panel_response(
        request,
        session,
        settings,
        action_prefix="/setup/folders",
        csrf_action=CSRF_SETUP,
        path=path,
        error=error,
        redirect_url=redirect,
        error_page=_error_page,
    )

@setup_router.post("/setup/vocabulary", include_in_schema=False)
def setup_vocabulary(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    vocabulary: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings
    try:
        terms = normalize_vocabulary(vocabulary)
    except SetupValidationError as exc:
        return templates.TemplateResponse(
            request,
            "settings/setup.html",
            _setup_context(
                request,
                session,
                WizardStep.VOCABULARY,
                error=str(exc),
                vocabulary_text=vocabulary,
            ),
        )
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.vocabulary = terms
    return _setup_redirect(WizardStep.LLM)

@setup_router.post("/setup/llm", include_in_schema=False)
def setup_llm(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    enabled: Annotated[bool, Form()] = False,
    llm_base_url: Annotated[str, Form()] = "",
    llm_model: Annotated[str, Form()] = "",
    llm_api_key: Annotated[str, Form()] = "",
    remove_llm_api_key: Annotated[bool, Form()] = False,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings

    def _rerender(error: str) -> Response:
        # The key field is a password, never prefilled — so the submitted key is
        # never echoed. Only the non-secret overrides survive the re-render.
        # Echo the operator's SUBMITTED endpoint text verbatim (blank stays
        # blank, with the env default as the placeholder) so their in-progress
        # input survives — never fall back to settings.llm_base_url here, which
        # would put the env default in the input `value` and falsely show an
        # inheriting field as pinned (issue #46). `llm_enabled` is NOT
        # overridden: _setup_context reads the persisted row, so a validation
        # failure that fail-closes shows the checkbox OFF — the honest state —
        # rather than echoing the submitted intent as if it stuck (matching
        # /settings/llm).
        return templates.TemplateResponse(
            request,
            "settings/setup.html",
            _setup_context(
                request,
                session,
                WizardStep.LLM,
                error=error,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
            ),
        )

    # Candidate-state → validate → ONE mutation (see _persist_llm_settings). A
    # pure format error raises and changes nothing; a validation failure persists
    # the valid overrides + candidate key, forces llm_enabled=False, and returns
    # the message to re-render — both re-render the LLM step, fail closed.
    try:
        error = _persist_llm_settings(
            session,
            settings,
            enabled=enabled,
            raw_base_url=llm_base_url,
            raw_model=llm_model,
            raw_key=llm_api_key,
            remove_key=remove_llm_api_key,
        )
    except SetupValidationError as exc:
        return _rerender(str(exc))
    if error is not None:
        return _rerender(error)
    return _setup_redirect(WizardStep.SERVICES)

@setup_router.post("/setup/finish", include_in_schema=False)
def setup_finish(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
    start_tutorial: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETUP, csrf_token)
    settings: Settings = request.app.state.settings
    # ``start_tutorial`` is the operator's explicit intent (the primary Finish
    # button), NOT the seed mechanism: it both seeds-if-needed and drives the
    # redirect, so a plain "Finish setup" never launches a tutorial and its
    # label never lies (issue #75). Seed BEFORE completing onboarding so a
    # storage/asset failure aborts the whole request with nothing committed.
    # Exact "1" (the button's value) — a crafted start_tutorial=0/false is not
    # an intent to start.
    wants_tutorial = start_tutorial == "1"
    seeded_run_id: uuid.UUID | None = None
    if wants_tutorial:
        seeded_run_id, error = _try_seed_tutorial(session, settings)
        if error is not None:
            session.rollback()
            return templates.TemplateResponse(
                request,
                "settings/setup.html",
                _setup_context(
                    request, session, WizardStep.FINISH, tutorial_error=error
                ),
            )
    complete_onboarding(session, llm_enabled_default=settings.llm_enabled)
    # Commit explicitly before the redirect so the request that follows cannot
    # observe stale onboarding state (the gate re-reads per request).
    session.commit()
    # Launch the guided tutorial only when the operator asked for it and only
    # AFTER onboarding commits: a pre-onboarding link to /runs/{id}?tutorial=run
    # would hit the protected gate and bounce back to /setup. Redirect off the
    # seeder's own returned id (idempotent — the existing run when already
    # seeded) rather than a re-read, so a successful seed can never silently
    # fall through to /review.
    if wants_tutorial and seeded_run_id is not None:
        return RedirectResponse(
            f"/runs/{seeded_run_id}?tutorial=run", status_code=303
        )
    return RedirectResponse("/review", status_code=303)


# ---- Settings + guided-tutorial lifecycle (issue #3, slice 6) --------------
# The persistent, re-runnable entry point: re-open the setup wizard, and
# start / replay / complete the guided tutorial. All @protected (an
# un-onboarded operator is bounced to /setup by the gate). The two POSTs verify
# CSRF_SETTINGS and 409 when no tutorial run is available, so a stray token can
# never "complete" or "replay" an unseeded tutorial.

def _settings_context(
    request: Request,
    session: Session,
    folder_path: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Shared context for the settings page (GET render + POST re-render).

    Carries the effective LLM state (issue #10) — enablement over env, the
    effective endpoint, and whether an effective key is present and where it
    comes from — never the key value — plus the Features-section tri-state flag
    rows (issue #62). ``overrides`` lets a POST re-render carry a section
    ``*_error`` and (for Features) ``features_submitted``, the operator's
    submitted radio selections, so a rejected save re-renders their choices.

    ``folder_path`` overrides the folder-panel browse location (defaults to
    ``?path=`` from the query string) so a non-HTMX folder mutation that fails
    can re-render the full page at the submitted path instead of the root.
    """
    settings: Settings = request.app.state.settings
    tutorial_run = ready_tutorial_run_id(session)
    row = get_app_settings(session)
    base_value, base_default, model_value, model_default = llm_endpoint_form_fields(
        row, settings
    )
    # Features section (issue #62): one tri-state row per live-read flag. On an
    # invariant-rejected save, render the operator's submitted choices back
    # (``features_submitted``); otherwise render the stored raw tri-state. The
    # effective meta (issue #138) appends any plugin-contributed flags to the
    # core set; empty registry => the core set unchanged.
    #
    # Dependency nesting (issue #406): compute which flags should be disabled
    # based on their prerequisites' effective state. Two passes: first build
    # states and resolve to effective booleans, then derive disabled/reason.
    features_submitted: dict[str, str] | None = overrides.pop("features_submitted", None)
    flag_meta = _effective_feature_flag_meta(request.app.state.plugins)

    # Pass 1: build tri-state and effective-boolean maps.
    flag_states: dict[str, str] = {}
    flag_effective: dict[str, bool] = {}
    for name, _, _ in flag_meta:
        state = (
            features_submitted.get(name, "inherit")
            if features_submitted is not None
            else feature_flag_state(row, name)
        )
        env_default = bool(getattr(settings, name))
        flag_states[name] = state
        flag_effective[name] = state == "on" or (state == "inherit" and env_default)

    # Seed cross-section dependencies into the effective namespace.
    flag_effective["llm_enabled"] = resolve_effective_llm_enabled(row, settings)

    # Pass 2: derive disabled state and build the template dicts. Flags are
    # processed in display order (parent before child), so propagating
    # effective=False for disabled flags gives transitive disablement: if
    # run_assets is disabled (LLM off), autogenerate sees it as unavailable.
    feature_flags: list[dict[str, object]] = []
    for name, label, help_text in flag_meta:
        deps = _FEATURE_DEPS.get(name, ())
        missing = [d for d in deps if not flag_effective.get(d, False)]
        disabled = bool(missing)
        if disabled:
            parts = [_DEP_LABELS[d] for d in missing]
            reason = f"Needs {_join_operator_labels(parts)} to be on."
            flag_effective[name] = False
        else:
            reason = ""
        effective = flag_effective[name]
        feature_flags.append({
            "name": name,
            "label": label,
            "help": help_text,
            "state": flag_states[name],
            "env_default": bool(getattr(settings, name)),
            "effective": effective,
            "disabled": disabled,
            "disabled_reason": reason,
            "dependent": any(d in flag_states for d in deps),
        })
    # Semantic search section (issue #121): two tri-state rows (the feature +
    # its autogenerate rider). On an invariant-rejected save, render the
    # operator's submitted choices back (``semantic_index_submitted``); otherwise
    # the stored raw tri-state.
    semantic_submitted: dict[str, str] | None = overrides.pop(
        "semantic_index_submitted", None
    )
    # The tri-state the toggle renders (submitted choice on an invariant-
    # rejected re-render, else the stored raw state). The weights-missing
    # notice gates on the EFFECTIVE enablement derived from it, not the raw
    # state: "inherit" with an installation default of Off is effectively
    # off, so warning that an on-but-weightless search cannot answer would be
    # untrue.
    semantic_enabled_state = (
        semantic_submitted.get("semantic_index_enabled", "inherit")
        if semantic_submitted is not None
        else feature_flag_state(row, "semantic_index_enabled")
    )
    # Translation section (issue #133): the preferred-language select (a
    # string override — "inherit" or a language code) + the autogenerate
    # tri-state. On an invariant-rejected save, render the operator's
    # submitted choices back.
    translation_submitted: dict[str, str] | None = overrides.pop(
        "translation_submitted", None
    )
    stored_translation_target = (
        row.translation_target_language.strip()
        if row is not None
        and row.translation_target_language is not None
        and row.translation_target_language.strip()
        else "inherit"
    )
    # Sources & research section (issue #76). On an invariant/format-rejected
    # save, render the operator's submitted choices back (``web_research_submitted``
    # — the four non-secret fields only, never the key); otherwise the stored raw
    # tri-state / override values.
    wr_submitted: dict[str, str] | None = overrides.pop("web_research_submitted", None)
    # Synthdetect section (#145): two tri-state toggles (enabled + autogenerate).
    synthdetect_submitted: dict[str, str] | None = overrides.pop(
        "synthdetect_submitted", None
    )
    wr_base_value, wr_base_default = str_flag_form_field(row, settings, "web_search_base_url")
    wr_domains_value, wr_domains_default = str_flag_form_field(
        row, settings, "source_authority_domains"
    )
    context: dict[str, Any] = {
        "request": request,
        "tutorial_available": tutorial_run is not None,
        "tutorial_run_id": tutorial_run,
        "tutorial_completed_at": row.tutorial_completed_at if row else None,
        "llm_enabled": bool(row.llm_enabled) if row is not None else settings.llm_enabled,
        # Endpoint OVERRIDE (blank when inheriting env), env default as
        # placeholder — the tri-state render that keeps an untouched save from
        # pinning the env value onto the row (issue #46).
        "llm_base_url": base_value,
        "llm_base_url_default": base_default,
        "llm_model": model_value,
        "llm_model_default": model_default,
        "llm_key_present": bool(resolve_effective_llm_api_key(row, settings)),
        "llm_key_source": effective_llm_key_source(row, settings),
        "llm_budget_ok": llm_budget_fits_stage_lease(settings),
        # Completion celebration after POST /settings/tutorial/complete —
        # shown ONLY when the tutorial is genuinely completed, so a spoofed
        # or bookmarked ?tutorial=done on an unseeded/incomplete tutorial
        # does not falsely claim completion.
        "tutorial_done": (
            request.query_params.get("tutorial") == "done"
            and row is not None
            and row.tutorial_completed_at is not None
        ),
        "csrf_settings": mint_csrf_token(request.app.state.csrf_secret, CSRF_SETTINGS),
        "active_nav": "settings",
        "llm_error": None,
        # Bounded, non-secret message when a UI-triggered tutorial seed fails
        # (issue #75); None on a normal render. Overridable via ``overrides``.
        "tutorial_error": None,
        "feature_flags": feature_flags,
        "features_errors": [],
        # Semantic search (issue #121): the two tri-state toggles (raw stored
        # tri-state, or submitted choice on re-render), their env defaults for the
        # "inherit" label, and whether the embedding weights are actually
        # installed — an honest note when the feature is on but weights are absent,
        # since enabling it then cannot answer a query.
        "semantic_index_enabled_state": semantic_enabled_state,
        "semantic_index_enabled_env_default": bool(settings.semantic_index_enabled),
        # Effective enablement (candidate/stored state resolved over the env
        # default) — the honest gate for the weights-missing notice below.
        "semantic_index_effective_enabled": (
            semantic_enabled_state == "on"
            or (
                semantic_enabled_state == "inherit"
                and bool(settings.semantic_index_enabled)
            )
        ),
        "semantic_index_autogenerate_state": (
            semantic_submitted.get("semantic_index_autogenerate", "inherit")
            if semantic_submitted is not None
            else feature_flag_state(row, "semantic_index_autogenerate")
        ),
        "semantic_index_autogenerate_env_default": bool(
            settings.semantic_index_autogenerate
        ),
        "semantic_index_weights_available": minilm_artifacts_available(),
        "semantic_index_errors": [],
        # Transcript translation (issue #133): the preferred-language select
        # state ("inherit" or a code), the env default (a code or None) for
        # the inherit label, the autogenerate tri-state, whether the LLM
        # path is effectively open (an honest note when auto-translate is on
        # but the LLM is off), and the sorted code/name options.
        "translation_target_state": (
            translation_submitted.get("translation_target_language", "inherit")
            if translation_submitted is not None
            else stored_translation_target
        ),
        "translation_target_env_default": (
            settings.translation_target_language.strip()
            if settings.translation_target_language is not None
            and settings.translation_target_language.strip()
            else None
        ),
        "translation_autogenerate_state": (
            translation_submitted.get("translation_autogenerate", "inherit")
            if translation_submitted is not None
            else feature_flag_state(row, "translation_autogenerate")
        ),
        "translation_autogenerate_env_default": bool(settings.translation_autogenerate),
        "translation_llm_open": translation_gates_open(settings, row),
        "translation_language_options": sorted(
            LANGUAGE_NAMES.items(), key=lambda item: item[1]
        ),
        "translation_errors": [],
        # Synthdetect (#145): two tri-state toggles (enabled + autogenerate).
        # Submitted choice on re-render, else the stored raw state.
        "synthdetect_enabled_state": (
            synthdetect_submitted.get("synthdetect_enabled", "inherit")
            if synthdetect_submitted is not None
            else feature_flag_state(row, "synthdetect_enabled")
        ),
        "synthdetect_autogenerate_state": (
            synthdetect_submitted.get("synthdetect_autogenerate", "inherit")
            if synthdetect_submitted is not None
            else feature_flag_state(row, "synthdetect_autogenerate")
        ),
        "synthdetect_enabled_env_default": bool(settings.synthdetect_enabled),
        "synthdetect_autogenerate_env_default": bool(
            settings.synthdetect_autogenerate
        ),
        "synthdetect_errors": [],
        # Sources & research (issue #76): the two web-research toggles (raw
        # tri-state, or submitted choice on re-render), the endpoint override +
        # env placeholder, the credential status (present + source, never the
        # value), and the authority-domains override + env placeholder.
        "web_research_master_state": (
            wr_submitted.get("voxint_web_research", "inherit")
            if wr_submitted is not None
            else feature_flag_state(row, "voxint_web_research")
        ),
        "web_research_master_env_default": bool(settings.voxint_web_research),
        "web_research_producer_state": (
            wr_submitted.get("enrichment_web_research_enabled", "inherit")
            if wr_submitted is not None
            else feature_flag_state(row, "enrichment_web_research_enabled")
        ),
        "web_research_producer_env_default": bool(settings.enrichment_web_research_enabled),
        "web_search_base_url": (
            wr_submitted.get("web_search_base_url", "")
            if wr_submitted is not None
            else wr_base_value
        ),
        "web_search_base_url_default": wr_base_default,
        "web_search_key_present": bool(resolve_effective_web_search_api_key(row, settings)),
        "web_search_key_source": effective_web_search_key_source(row, settings),
        "source_authority_domains": (
            wr_submitted.get("source_authority_domains", "")
            if wr_submitted is not None
            else wr_domains_value
        ),
        "source_authority_domains_default": wr_domains_default,
        "web_research_errors": [],
    }
    # Media folders + per-folder domain packs (issue #63). Always present on the
    # Settings page; the untrusted ?path= is revalidated in list_media_subdirs.
    context.update(
        _folder_panel_context(
            session,
            settings,
            action_prefix="/settings/folders",
            csrf=context["csrf_settings"],
            path=folder_path if folder_path is not None else (
                request.query_params.get("path") or "."
            ),
        )
    )
    # Watch-folder ingest (issue #60): the tri-state toggle beside the folder
    # panel, its env default (for the "inherit" label), the EFFECTIVE gate (drives
    # the on/off wording), and the latest sweep summary for the status line.
    context["watch_folder_state"] = feature_flag_state(row, "watch_folder_enabled")
    context["watch_folder_env_default"] = bool(settings.watch_folder_enabled)
    context["watch_folder_effective"] = resolve_effective_watch_folder_enabled(row, settings)
    last_sweep = row.watch_folder_last_sweep if row is not None else None
    context["watch_folder_last_sweep"] = last_sweep
    # Parse the stored ISO timestamp into a datetime so the template can format it
    # (the JSON column holds a string); None if never run or malformed.
    checked_at: datetime | None = None
    if isinstance(last_sweep, dict) and last_sweep.get("completed_at"):
        try:
            checked_at = datetime.fromisoformat(str(last_sweep["completed_at"]))
        except ValueError:
            checked_at = None
    context["watch_folder_checked_at"] = checked_at
    context["watch_folder_error"] = None
    # Corrections authoring (issue #84): the operator's current rules (for the
    # no-JS read-only fallback) plus the JSON props the corrections-editor
    # island hydrates from — the rules, the CSRF-guarded save action, and the
    # #80 bounds so the client can hint before the server (authoritative) gate.
    stored_corrections = (
        list(row.corrections) if row is not None and row.corrections else []
    )
    context["corrections"] = stored_corrections
    context["corrections_props"] = {
        "rules": stored_corrections,
        "action": "/settings/corrections",
        "csrfToken": context["csrf_settings"],
        "limits": {
            "maxRules": MAX_RULES_PER_PACK,
            "maxMatchChars": MAX_MATCH_CHARS,
            "maxReplacementChars": MAX_REPLACEMENT_CHARS,
        },
    }
    # Initialized like every sibling error (llm_error, watch_folder_error) so the
    # template never leans on Jinja's lenient Undefined; overridden on a rejected save.
    context["corrections_error"] = None
    # Glossary (issue #123): the operator's expected proper nouns, edited from the
    # console over the same app_settings.vocabulary the setup wizard writes. The
    # list drives the term count; the text is the textarea's replace-all body. Fed
    # to whisper as an initial_prompt hint, applied LIVE (not frozen), so a saved
    # change reaches the next run that starts.
    stored_vocabulary = list(row.vocabulary) if row is not None and row.vocabulary else []
    context["vocabulary"] = stored_vocabulary
    context["vocabulary_text"] = "\n".join(stored_vocabulary)
    # Initialized like corrections_error; overridden with the submitted text on a
    # rejected save so the operator does not lose their edit.
    context["glossary_error"] = None
    # Pipeline models panel (issue: configurable pipeline models). Live,
    # read-only identity of the three model services, probed concurrently on
    # each render (no cache: an operator who just changed .env reloads to
    # confirm). Best-effort; an unreachable service renders "unavailable",
    # never breaking the page. Only the flag-off flat page renders this panel;
    # the activated hub moved it to /settings/hardware (P6b, #161), so skip the
    # probe when the hub is active rather than probing three services on every
    # hub render and POST re-render for a result the hub never shows.
    context["pipeline_models"] = (
        () if settings.console_settings_enabled else collect_service_identity(settings)
    )
    # Plugin settings sections (issue #138): each active plugin's section,
    # ordered by (order, section_id). Console 2.0 renders these on Plugins;
    # the legacy flat page retains its original section loop.
    context["plugin_settings_sections"] = request.app.state.plugins.settings_sections()
    context["plugins"] = settings_view.build_plugins_view(
        request.app.state.plugins, row, settings
    )
    # Benchmark section: most recent runs for the settings page.
    try:
        from voxint.db.models import BenchmarkRun

        recent_runs = (
            session.execute(
                select(BenchmarkRun)
                .order_by(BenchmarkRun.created_at.desc())
                .limit(5)
            )
            .scalars()
            .all()
        )
        context["benchmark_runs"] = recent_runs
    except SQLAlchemyError:
        logger.debug("benchmark query failed; degrading to empty list", exc_info=True)
        session.rollback()
        context["benchmark_runs"] = []
    context.update(overrides)
    return context

def _settings_page_template(request: Request) -> str:
    """Select the legacy page or the Console 2.0 tab owning this request.

    Existing POST handlers use this helper for validation-error renders. Keeping
    the ownership map here lets those routes and their form actions stay stable
    while errors return to the tab containing the submitted form.
    """
    settings: Settings = request.app.state.settings
    if not settings.console_settings_enabled:
        return "settings/settings.html"
    path = request.url.path
    if path in {
        "/settings/folders", "/settings/watch-folder",
        "/settings/web-research", "/settings/media",
    }:
        return "settings/media.html"
    if path in {
        "/settings/llm",
        "/settings/translation",
        "/settings/corrections",
        "/settings/glossary",
        "/settings/semantic",
        "/settings/ai",
    }:
        return "settings/ai.html"
    # Plugin-contributed POST paths are intentionally not known to core. Any
    # remaining request rendered through this helper belongs to Plugins.
    if path != "/settings" and path not in {
        "/settings/features",
        "/settings/tutorial/seed",
        "/settings/tutorial/complete",
        "/settings/tutorial/replay",
    }:
        return "settings/plugins.html"
    return "settings/hub.html"


def _settings_redirect(request: Request, anchor: str, tab: str) -> str:
    """Success-redirect URL for a settings POST, tab-aware when Console 2.0 is on."""
    settings: Settings = request.app.state.settings
    if settings.console_settings_enabled:
        return f"/settings/{tab}#{anchor}" if tab else f"/settings#{anchor}"
    return f"/settings#{anchor}"


@router.get("/settings", name="settings_page")
def settings_page(request: Request, operator: OperatorDep, session: SessionDep) -> Response:
    return templates.TemplateResponse(
        request, _settings_page_template(request), _settings_context(request, session)
    )


@router.get("/settings/media", name="settings_media")
def settings_media_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    return templates.TemplateResponse(
        request, "settings/media.html", _settings_context(request, session)
    )


@router.get("/settings/ai", name="settings_ai")
def settings_ai_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    return templates.TemplateResponse(
        request, "settings/ai.html", _settings_context(request, session)
    )


def _sub_page_context(request: Request, **page: Any) -> dict[str, Any]:
    """Base context for a read-only settings sub-page (nav + page keys).

    Sub-pages are GET-only and stateless, so they need only the shell nav marker
    (``active_nav``) plus their own view; the shell's sidebar keys come from the
    template context processor.
    """
    return {"request": request, "active_nav": "settings", **page}


def _app_settings_or_none(session: Session) -> AppSettings | None:
    """Read the singleton settings row, or ``None`` if the DB is unreadable.

    The plugin pages render from the in-memory registry and only touch the row to
    resolve a plugin's enablement gate (which already accepts ``None``), so a DB
    hiccup degrades to ``None`` rather than 500ing the page."""
    try:
        return get_app_settings(session)
    except SQLAlchemyError:
        session.rollback()
        return None


@router.get("/settings/status", name="settings_status")
def settings_status_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    settings: Settings = request.app.state.settings
    snapshot = collect_resource_status_or_empty(settings)
    host = collect_host_metrics_or_empty(settings.media_root)
    gauges = _build_gauges(snapshot, host)
    strip = build_resource_strip(snapshot)
    warnings = list(strip.warnings)
    # The hardware gauges poll this route every 15s via htmx. Answer the poll
    # with just the gauge fragment (doctor checks are expensive network probes
    # and must not re-run on every tick). A boosted navigation wants the whole
    # document, so exclude it.
    if request.headers.get("HX-Request") and not request.headers.get("HX-Boosted"):
        return templates.TemplateResponse(
            request,
            "settings/_status_gauges.html",
            {
                "request": request,
                "gauges": gauges,
                "gauge_note": None,
                "warnings": warnings,
            },
        )
    checks = _doctor_checks(request, session)
    components = _build_components(checks)
    overall_ok = all(c["dot"] != "warn" for c in components)
    context = _sub_page_context(
        request,
        overall_ok=overall_ok,
        install_summary=_install_summary(settings, snapshot),
        components=components,
        gauges=gauges,
        gauge_note=None,
        warnings=warnings,
    )
    return templates.TemplateResponse(request, "settings/status.html", context)


@router.get("/settings/hardware", name="settings_hardware")
def settings_hardware_page(request: Request, operator: OperatorDep) -> Response:
    settings: Settings = request.app.state.settings
    view = settings_view.build_hardware_view(
        settings, tuple(collect_service_identity(settings))
    )
    # pipeline_models feeds the reused settings/_models.html panel (same live
    # service identity the flat page's Pipeline models section rendered).
    context = _sub_page_context(request, hardware=view, pipeline_models=view.services)
    return templates.TemplateResponse(request, "settings/hardware.html", context)


@router.get("/settings/database", name="settings_database")
def settings_database_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    settings: Settings = request.app.state.settings
    view = settings_view.build_database_view(session, settings)
    context = _sub_page_context(request, database=view)
    return templates.TemplateResponse(request, "settings/database.html", context)


@router.get("/settings/plugins", name="settings_plugins")
def settings_plugins_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    settings: Settings = request.app.state.settings
    registry: PluginRegistry = request.app.state.plugins
    row = _app_settings_or_none(session)
    view = settings_view.build_plugins_view(registry, row, settings)
    # Contributed settings sections are mutable and require the same context as
    # the core tabs (CSRF plus plugin-owned state). Preserve the registry page's
    # fail-soft behavior when the database is unavailable; forms cannot safely
    # render then, but the in-memory installed/disabled inventory still can.
    try:
        context = _settings_context(request, session)
    except SQLAlchemyError:
        session.rollback()
        context = _sub_page_context(
            request, plugin_settings_sections=(), plugins=view
        )
    else:
        context["plugins"] = view
    return templates.TemplateResponse(request, "settings/plugins.html", context)


@router.get("/settings/plugins/{plugin_id}", name="settings_plugin_detail")
def settings_plugin_detail_page(
    plugin_id: str, request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    settings: Settings = request.app.state.settings
    registry: PluginRegistry = request.app.state.plugins
    row = _app_settings_or_none(session)
    view = settings_view.build_plugin_detail(registry, plugin_id, row, settings)
    if view is None:
        raise HTTPException(status_code=404, detail="Unknown plugin")
    # A plugin's contributed section owns a CSRF-guarded form and reads the same
    # per-section context it gets on the hub, so the detail page renders it under
    # the full _settings_context (csrf_settings + section state), not a thin one.
    context = _settings_context(request, session)
    context["plugin"] = view
    return templates.TemplateResponse(request, "settings/plugin_detail.html", context)

@router.post("/settings/llm")
def settings_llm(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    enabled: Annotated[bool, Form()] = False,
    llm_base_url: Annotated[str, Form()] = "",
    llm_model: Annotated[str, Form()] = "",
    llm_api_key: Annotated[str, Form()] = "",
    remove_llm_api_key: Annotated[bool, Form()] = False,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings

    def _rerender(error: str) -> Response:
        # Password field, never prefilled: the submitted key is never echoed.
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, llm_error=error),
        )

    # Same candidate → validate → ONE mutation contract as /setup/llm.
    try:
        error = _persist_llm_settings(
            session,
            settings,
            enabled=enabled,
            raw_base_url=llm_base_url,
            raw_model=llm_model,
            raw_key=llm_api_key,
            remove_key=remove_llm_api_key,
        )
    except SetupValidationError as exc:
        return _rerender(str(exc))
    if error is not None:
        return _rerender(error)
    session.commit()
    return RedirectResponse(_settings_redirect(request, "llm", "ai"), status_code=303)

@router.post("/settings/features")
async def settings_features(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    # Plugin seam (#138): reconcile list must cover every rendered flag (#141).
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    reconciled = _reconcile_switches(_FEATURE_FLAG_NAMES, form, settings, row)
    errors = _persist_feature_flags(session, settings, submitted=reconciled)
    if errors:
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request, session, features_errors=errors, features_submitted=reconciled
            ),
        )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "features", ""), status_code=303)

@router.post("/settings/semantic")
async def settings_semantic(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Save the Semantic search section's two tri-state toggles (issue #121)."""
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    reconciled = _reconcile_switches(_SEMANTIC_FLAG_NAMES, form, settings, row)
    errors = _persist_semantic_index(session, settings, submitted=reconciled)
    if errors:
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request,
                session,
                semantic_index_errors=errors,
                semantic_index_submitted=reconciled,
            ),
        )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "semantic-search", "ai"), status_code=303)

@router.post("/settings/translation")
async def settings_translation(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Save the Translation section (#133): the preferred-language select
    (string override) + the auto-translate tri-state.
    """
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    translation_auto = _reconcile_switches(
        ("translation_autogenerate",), form, settings, row
    )
    submitted = {
        "translation_target_language": str(
            form.get("translation_target_language", "inherit")
        ),
        "translation_autogenerate": translation_auto.get(
            "translation_autogenerate", "inherit"
        ),
    }
    errors = _persist_translation(session, settings, submitted=submitted)
    if errors:
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request,
                session,
                translation_errors=errors,
                translation_submitted=submitted,
            ),
        )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "translation", "ai"), status_code=303)

@router.post("/settings/corrections")
def settings_corrections(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    rules: Annotated[str, Form()],
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Replace the operator's console-authored correction rules (issue #84).

    The corrections-editor island owns add/edit/remove/reorder client-side and
    submits the FULL ordered list as a JSON string in the ``rules`` form field
    (replace-all semantics, like vocabulary). The whole set is validated through
    the SAME #80 gate a pack gets — bounds, NUL/control-char rejection, unique
    ids, boundary-aware idempotence — plus a union check against the current
    default pack, so a rule the console accepts is one the frozen-per-run
    pipeline can apply. On any violation NOTHING is written: the island (Accept:
    application/json) gets a 422 with the plain-language message and the
    offending row; a JS-off submit re-renders the page with the message.
    """
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    wants_json = "application/json" in (request.headers.get("accept") or "")
    try:
        raw_items = json.loads(rules)
    except json.JSONDecodeError:
        message = "The corrections payload was not valid JSON."
        if wants_json:
            return JSONResponse(
                {"ok": False, "error": message, "row": None}, status_code=422
            )
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, corrections_error=message),
            status_code=422,
        )
    # Candidate → validate (field + cross-rule + default-pack union) → ONE
    # mutation. get_or_create only after validation, so a rejected save (which
    # the API session would commit on a 200/422 render) writes nothing.
    # The union check needs the default pack; if the registry is unreadable
    # (bad DOMAIN_PACKS_DIR / DOMAIN_PACK_PATH) degrade like the folder panel
    # (_set_folder_pack) — refuse with guidance, never 500. Skipping the union
    # silently would be worse (it would let a colliding rule save).
    try:
        pack_corrections = (
            default_domain_pack(settings).to_mapping().get("corrections")
        )
    except DomainPackError:
        message = (
            "Domain packs can't be loaded right now, so corrections can't be "
            "validated — check your domain-pack configuration and try again."
        )
        if wants_json:
            return JSONResponse(
                {"ok": False, "error": message, "row": None}, status_code=422
            )
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, corrections_error=message),
            status_code=422,
        )
    try:
        normalized = normalize_operator_corrections(
            raw_items, pack_corrections=pack_corrections
        )
    except OperatorCorrectionError as exc:
        if wants_json:
            return JSONResponse(
                {"ok": False, "error": exc.message, "row": exc.row},
                status_code=422,
            )
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, corrections_error=exc.message),
            status_code=422,
        )
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.corrections = normalized
    session.commit()
    if wants_json:
        return JSONResponse({"ok": True, "corrections": normalized})
    return RedirectResponse(_settings_redirect(request, "corrections", "ai"), status_code=303)

@router.post("/settings/glossary")
def settings_glossary(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    vocabulary: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Replace the operator's glossary (issue #123) from the console.

    Edits the same ``app_settings.vocabulary`` the setup wizard writes, through
    the SAME ``normalize_vocabulary`` gate (one term per line, deduped, 500-term
    / 120-char bounds), replace-all. On a bounds violation NOTHING is written and
    the page re-renders with the plain-language message AND the operator's
    submitted text (so a rejected save never eats their edit); get_or_create runs
    only after validation, so the rejected render's commit persists no change.
    """
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    try:
        terms = normalize_vocabulary(vocabulary)
    except SetupValidationError as exc:
        # Keep the term count honest with the textarea on a rejected save: show
        # the operator's own submitted lines, not the last stored list (the
        # partial reads only ``vocabulary|length`` for the count). So "501
        # terms." sits beside an "at most 500" message, not a stale "1 term.".
        submitted = [line for line in vocabulary.splitlines() if line.strip()]
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request,
                session,
                glossary_error=str(exc),
                vocabulary_text=vocabulary,
                vocabulary=submitted,
            ),
            status_code=422,
        )
    row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row.vocabulary = terms
    session.commit()
    return RedirectResponse(_settings_redirect(request, "glossary", "ai"), status_code=303)

@router.post("/settings/watch-folder")
async def settings_watch_folder(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Set the watch-folder ingest runtime override (issue #60)."""
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    reconciled = _reconcile_switches(("watch_folder_enabled",), form, settings, row)
    wf_choice = reconciled.get("watch_folder_enabled", "inherit")
    wf_row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    wf_row.watch_folder_enabled = (
        None if wf_choice == "inherit" else (wf_choice == "on")
    )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "watch-folder", "media"), status_code=303)

@router.post("/settings/web-research")
async def settings_web_research(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)
    wr_master = _reconcile_switches(
        ("voxint_web_research",), form, settings, row
    )
    wr_producer = _reconcile_switches(
        ("enrichment_web_research_enabled",), form, settings, row
    )
    errors = _persist_web_research(
        session,
        settings,
        submitted_master=wr_master.get("voxint_web_research", "inherit"),
        submitted_producer=wr_producer.get(
            "enrichment_web_research_enabled", "inherit"
        ),
        raw_base_url=str(form.get("web_search_base_url", "")),
        raw_key=str(form.get("web_search_api_key", "")),
        remove_key=form.get("remove_web_search_api_key") == "true",
        raw_domains=str(form.get("source_authority_domains", "")),
    )
    if errors:
        submitted = {
            "voxint_web_research": wr_master.get("voxint_web_research", "inherit"),
            "enrichment_web_research_enabled": wr_producer.get(
                "enrichment_web_research_enabled", "inherit"
            ),
            "web_search_base_url": str(form.get("web_search_base_url", "")),
            "source_authority_domains": str(form.get("source_authority_domains", "")),
        }
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request, session, web_research_errors=errors, web_research_submitted=submitted
            ),
        )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "sources", "media"), status_code=303)

# ---- Tab-level POST endpoints (#379) ------------------------------------
# One POST per settings tab, dispatching to the existing section persisters.
# These replace the per-section Save buttons with one Save per tab.  The old
# section endpoints remain as compatibility wrappers.


@router.post("/settings", name="settings_general_save")
async def settings_general_save(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Save the General tab: feature flags (switch encoding, issue #379)."""
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)

    reset_flag = form.get("reset_flag")
    if reset_flag and isinstance(reset_flag, str):
        return _handle_reset_flag(session, settings, request, reset_flag, tab="")

    reconciled = _reconcile_switches(
        _FEATURE_FLAG_NAMES, form, settings, row
    )
    errors = _persist_feature_flags(session, settings, submitted=reconciled)
    if errors:
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(
                request, session,
                features_errors=errors, features_submitted=reconciled,
            ),
        )
    session.commit()
    return RedirectResponse(_settings_redirect(request, "features", ""), status_code=303)


@router.post("/settings/ai", name="settings_ai_save")
async def settings_ai_save(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Save the AI tab: LLM + semantic + translation + glossary (#379).

    Corrections stays a separate island form.  Sections are processed in
    sequence; the first validation failure stops, rolls back any prior
    mutations in the session, and re-renders with the error.  Earlier
    sections' unsaved edits re-render from the DB (atomic rollback), not
    from the submitted form data.
    """
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)

    reset_flag = form.get("reset_flag")
    if reset_flag and isinstance(reset_flag, str):
        return _handle_reset_flag(session, settings, request, reset_flag, tab="ai")

    # ---- LLM section ----
    try:
        llm_error = _persist_llm_settings(
            session,
            settings,
            enabled=form.get("enabled") == "true",
            raw_base_url=str(form.get("llm_base_url", "")),
            raw_model=str(form.get("llm_model", "")),
            raw_key=str(form.get("llm_api_key", "")),
            remove_key=form.get("remove_llm_api_key") == "true",
        )
    except SetupValidationError as exc:
        llm_error = str(exc)
    if llm_error is not None:
        session.rollback()
        return templates.TemplateResponse(
            request, _settings_page_template(request),
            _settings_context(request, session, llm_error=llm_error),
        )

    # ---- Semantic search section ----
    semantic_reconciled = _reconcile_switches(
        _SEMANTIC_FLAG_NAMES, form, settings, row
    )
    semantic_errors = _persist_semantic_index(
        session, settings, submitted=semantic_reconciled
    )
    if semantic_errors:
        session.rollback()
        return templates.TemplateResponse(
            request, _settings_page_template(request),
            _settings_context(
                request, session,
                semantic_index_errors=semantic_errors,
                semantic_index_submitted=semantic_reconciled,
            ),
        )

    # ---- Translation section ----
    translation_auto = _reconcile_switches(
        ("translation_autogenerate",), form, settings, row
    )
    translation_reconciled = {
        "translation_target_language": str(
            form.get("translation_target_language", "inherit")
        ),
        "translation_autogenerate": translation_auto.get(
            "translation_autogenerate", "inherit"
        ),
    }
    translation_errors = _persist_translation(
        session, settings, submitted=translation_reconciled
    )
    if translation_errors:
        session.rollback()
        return templates.TemplateResponse(
            request, _settings_page_template(request),
            _settings_context(
                request, session,
                translation_errors=translation_errors,
                translation_submitted=translation_reconciled,
            ),
        )

    # ---- Glossary section ----
    vocabulary_text = str(form.get("vocabulary", ""))
    try:
        terms = normalize_vocabulary(vocabulary_text)
    except SetupValidationError as exc:
        session.rollback()
        submitted_terms = [
            line for line in vocabulary_text.splitlines() if line.strip()
        ]
        return templates.TemplateResponse(
            request, _settings_page_template(request),
            _settings_context(
                request, session,
                glossary_error=str(exc),
                vocabulary_text=vocabulary_text,
                vocabulary=submitted_terms,
            ),
            status_code=422,
        )
    row_mut = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    row_mut.vocabulary = terms

    session.commit()
    return RedirectResponse(
        _settings_redirect(request, "llm", "ai"), status_code=303
    )


@router.post("/settings/media", name="settings_media_save")
async def settings_media_save(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Save the Media tab: watch-folder + sources/research (#379).

    Folder browse/add/remove/pack commands remain separate action forms.
    """
    form = await request.form()
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    row = get_app_settings(session)

    reset_flag = form.get("reset_flag")
    if reset_flag and isinstance(reset_flag, str):
        return _handle_reset_flag(session, settings, request, reset_flag, tab="media")

    # ---- Watch-folder section ----
    wf_reconciled = _reconcile_switches(
        ("watch_folder_enabled",), form, settings, row
    )
    wf_choice = wf_reconciled.get("watch_folder_enabled", "inherit")
    wf_row = get_or_create(session, llm_enabled_default=settings.llm_enabled)
    wf_row.watch_folder_enabled = (
        None if wf_choice == "inherit" else (wf_choice == "on")
    )

    # ---- Sources & research section ----
    wr_master = _reconcile_switches(
        ("voxint_web_research",), form, settings, row
    )
    wr_producer = _reconcile_switches(
        ("enrichment_web_research_enabled",), form, settings, row
    )
    wr_errors = _persist_web_research(
        session,
        settings,
        submitted_master=wr_master.get("voxint_web_research", "inherit"),
        submitted_producer=wr_producer.get(
            "enrichment_web_research_enabled", "inherit"
        ),
        raw_base_url=str(form.get("web_search_base_url", "")),
        raw_key=str(form.get("web_search_api_key", "")),
        remove_key=form.get("remove_web_search_api_key") == "true",
        raw_domains=str(form.get("source_authority_domains", "")),
    )
    if wr_errors:
        session.rollback()
        wr_submitted = {
            "voxint_web_research": wr_master.get(
                "voxint_web_research", "inherit"
            ),
            "enrichment_web_research_enabled": wr_producer.get(
                "enrichment_web_research_enabled", "inherit"
            ),
            "web_search_base_url": str(form.get("web_search_base_url", "")),
            "source_authority_domains": str(
                form.get("source_authority_domains", "")
            ),
        }
        return templates.TemplateResponse(
            request, _settings_page_template(request),
            _settings_context(
                request, session,
                web_research_errors=wr_errors,
                web_research_submitted=wr_submitted,
            ),
        )
    session.commit()
    return RedirectResponse(
        _settings_redirect(request, "folders", "media"), status_code=303
    )


# ---- Settings → Media folders + domain packs (issue #63) ---------------
# The same folder browser as the wizard, mounted on the protected router with
# CSRF_SETTINGS. Shares _folder_panel_response / _folder_panel_context.

@router.get("/settings/folders/browse")
def settings_folders_browse(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    path: Annotated[str, Query(max_length=4096)] = ".",
) -> Response:
    settings: Settings = request.app.state.settings
    csrf = mint_csrf_token(request.app.state.csrf_secret, CSRF_SETTINGS)
    context = _folder_panel_context(
        session, settings, action_prefix="/settings/folders", csrf=csrf, path=path
    )
    response = templates.TemplateResponse(
        request, "settings/folder_panel.html", {"request": request, **context}
    )
    response.headers["Cache-Control"] = "no-store"
    return response

@router.post("/settings/folders")
def settings_folders(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    action: Annotated[str, Form(max_length=16)] = "",
    folder: Annotated[str, Form(max_length=4096)] = "",
    pack: Annotated[str | None, Form(max_length=200)] = None,
    path: Annotated[str, Form(max_length=4096)] = ".",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    settings: Settings = request.app.state.settings
    error = _apply_folder_mutation(
        session, settings, action=action, folder=folder, pack=pack
    )
    if error is None:
        session.commit()
    else:
        session.rollback()
    base = "/settings/media?" if settings.console_settings_enabled else "/settings?"
    redirect = base + urlencode({"path": path}) + "#folders"

    def _error_page(message: str) -> Response:
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, folder_path=path, folder_error=message),
        )

    return _folder_panel_response(
        request,
        session,
        settings,
        action_prefix="/settings/folders",
        csrf_action=CSRF_SETTINGS,
        path=path,
        error=error,
        redirect_url=redirect,
        error_page=_error_page,
    )

@router.post("/settings/tutorial/seed")
def tutorial_seed(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    # Idempotent seed from the UI so a non-technical operator never needs the
    # CLI (issue #75). On a classified storage/asset failure, roll back the
    # flushed partial rows (the request session commits on any 200) and
    # re-render the page with bounded, non-secret guidance — writing nothing.
    settings: Settings = request.app.state.settings
    run_id, error = _try_seed_tutorial(session, settings)
    if error is not None:
        session.rollback()
        return templates.TemplateResponse(
            request,
            _settings_page_template(request),
            _settings_context(request, session, tutorial_error=error),
        )
    session.commit()
    return RedirectResponse(f"/runs/{run_id}?tutorial=run", status_code=303)

@router.post("/settings/tutorial/complete")
def tutorial_complete(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    # Explicit, idempotent completion (mark_tutorial_complete stamps only when
    # currently NULL, so a refresh/repost preserves the original time); 409 when
    # there is no available tutorial run to complete.
    if not mark_tutorial_complete(session):
        raise HTTPException(status_code=409, detail="no tutorial run to complete")
    session.commit()
    return RedirectResponse("/settings?tutorial=done", status_code=303)

@router.post("/settings/tutorial/replay")
def tutorial_replay(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> RedirectResponse:
    _require_csrf(request, CSRF_SETTINGS, csrf_token)
    run_id = ready_tutorial_run_id(session)
    if run_id is None:
        raise HTTPException(status_code=409, detail="no tutorial run to replay")
    # Non-destructive replay: clear the completion stamp and re-enter the
    # walkthrough. Prior speaker rulings on the run are intentionally preserved
    # (see clear_tutorial_completion / settings.html copy).
    clear_tutorial_completion(session)
    session.commit()
    return RedirectResponse(f"/runs/{run_id}?tutorial=run", status_code=303)


# ---- User management (issue #362) ----------------------------------------
# Admin-only sub-page at /settings/users, dark-shipped behind
# console_users_enabled. All routes sit on the existing `router` (already
# admin-gated + onboarding-gated at router level) with an additional
# per-route require_users_enabled gate so they 404 cleanly when the flag
# is off.

def _users_context(
    request: Request,
    session: Session,
    admin: AuthContext,
    **overrides: Any,
) -> dict[str, Any]:
    from voxint.users import list_users

    return {
        "request": request,
        "active_nav": "settings",
        "users": list_users(session),
        "csrf_users": mint_csrf_token(request.app.state.csrf_secret, CSRF_USERS),
        "current_admin": admin.username,
        "users_error": None,
        "users_success": None,
        **overrides,
    }


def _users_redirect(message: str | None = None) -> RedirectResponse:
    url = "/settings/users"
    if message:
        url += "?ok=" + message
    return RedirectResponse(url, status_code=303)


@router.get(
    "/settings/users",
    name="settings_users",
    dependencies=[Depends(require_users_enabled)],
)
def settings_users_page(
    request: Request,
    admin: AdminDep,
    session: SessionDep,
) -> Response:
    ok_message = request.query_params.get("ok")
    return templates.TemplateResponse(
        request,
        "settings/users.html",
        _users_context(
            request,
            session,
            admin,
            users_success=ok_message,
        ),
    )


@router.post(
    "/settings/users/create",
    dependencies=[Depends(require_users_enabled)],
)
def settings_users_create(
    request: Request,
    admin: AdminDep,
    session: SessionDep,
    username: Annotated[str, Form(max_length=64)] = "",
    password: Annotated[str, Form()] = "",
    password_confirm: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = "reviewer",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    from sqlalchemy.exc import IntegrityError

    from voxint.db.models import UserRole
    from voxint.users import create_user

    _require_csrf(request, CSRF_USERS, csrf_token)
    if role not in ("admin", "reviewer", "viewer"):
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(
                request, session, admin,
                users_error="Choose Admin, Reviewer, or Viewer.",
            ),
        )
    if password != password_confirm:
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(
                request, session, admin,
                users_error="Passwords do not match.",
            ),
        )
    try:
        create_user(
            session, username=username, password=password, role=UserRole(role),
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(
                request, session, admin,
                users_error=str(exc),
            ),
        )
    except IntegrityError:
        session.rollback()
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(
                request, session, admin,
                users_error=f"User {username!r} already exists.",
            ),
        )
    session.commit()
    return _users_redirect(f"Created user {username}.")


@router.post(
    "/settings/users/{user_id}/role",
    dependencies=[Depends(require_users_enabled)],
)
def settings_users_role(
    user_id: uuid.UUID,
    request: Request,
    admin: AdminDep,
    session: SessionDep,
    role: Annotated[str, Form()] = "reviewer",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    from voxint.db.models import User, UserRole
    from voxint.users import set_role

    _require_csrf(request, CSRF_USERS, csrf_token)
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    if user.username == admin.username:
        raise HTTPException(status_code=403, detail="cannot change your own role")
    if role not in ("admin", "reviewer", "viewer"):
        raise HTTPException(status_code=422, detail="invalid role")
    try:
        set_role(session, user, UserRole(role))
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(request, session, admin, users_error=str(exc)),
        )
    session.commit()
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "settings/_users_row.html",
            _users_context(request, session, admin, u=user),
        )
    return _users_redirect(f"Changed {user.username} role to {role}.")


@router.post(
    "/settings/users/{user_id}/toggle",
    dependencies=[Depends(require_users_enabled)],
)
def settings_users_toggle(
    user_id: uuid.UUID,
    request: Request,
    admin: AdminDep,
    session: SessionDep,
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    from voxint.db.models import User
    from voxint.users import set_disabled

    _require_csrf(request, CSRF_USERS, csrf_token)
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    if user.username == admin.username:
        raise HTTPException(status_code=403, detail="cannot disable yourself")
    currently_disabled = user.disabled_at is not None
    try:
        set_disabled(session, user, disabled=not currently_disabled)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(request, session, admin, users_error=str(exc)),
        )
    session.commit()
    verb = "Enabled" if currently_disabled else "Disabled"
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "settings/_users_row.html",
            _users_context(request, session, admin, u=user),
        )
    return _users_redirect(f"{verb} user {user.username}.")


@router.post(
    "/settings/users/{user_id}/reset-password",
    dependencies=[Depends(require_users_enabled)],
)
def settings_users_reset_password(
    user_id: uuid.UUID,
    request: Request,
    admin: AdminDep,
    session: SessionDep,
    new_password: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    from voxint.db.models import User
    from voxint.users import reset_password

    _require_csrf(request, CSRF_USERS, csrf_token)
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    if user.username == admin.username:
        raise HTTPException(
            status_code=403,
            detail="use the CLI to change your own password",
        )
    try:
        reset_password(session, user, new_password)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "settings/users.html",
            _users_context(request, session, admin, users_error=str(exc)),
        )
    session.commit()
    return _users_redirect(f"Reset password for {user.username}.")
