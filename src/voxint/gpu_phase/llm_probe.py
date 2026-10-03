"""Probe the effective language-model endpoints without exposing configuration."""

from contextlib import ExitStack

import httpx
from sqlalchemy.orm import Session

from voxint.app_settings import (
    byo_llm_configured,
    get_app_settings,
    llm_bundled_active,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
from voxint.config import Settings
from voxint.diagnostics import check_llm, check_llm_bundled


def probe_llm(
    settings: Settings, session: Session, client: httpx.Client | None = None
) -> bool | None:
    """None means no configured LLM work; any failed check closes the lane."""
    try:
        row = get_app_settings(session)
        if not resolve_effective_llm_enabled(row, settings):
            return None
        bundled = llm_bundled_active(row, settings)
        byo = byo_llm_configured(row, settings)
        if not bundled and not byo:
            return None
        with ExitStack() as stack:
            if client is None:
                client = stack.enter_context(httpx.Client(timeout=3.0))
            base_url, _model = resolve_effective_llm_endpoint(row, settings)
            results = [
                check_llm_bundled(
                    active=bundled, base_url=settings.llm_bundled_base_url, client=client
                ),
                check_llm(
                    enabled=byo,
                    configured=byo,
                    base_url=base_url,
                    api_key=resolve_effective_llm_api_key(row, settings),
                    client=client,
                ),
            ]
            return all(result.ok for result in results if result is not None)
    except Exception:
        return False
