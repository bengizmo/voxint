"""Probe the effective language-model endpoints without exposing configuration."""

from contextlib import ExitStack

import httpx
from sqlalchemy.orm import Session

from voxint.app_settings import (
    get_app_settings,
    llm_bundled_active,
    resolve_effective_llm_api_key,
    resolve_effective_llm_enabled,
    resolve_effective_llm_endpoint,
)
from voxint.config import Settings, llm_endpoint_explicitly_set
from voxint.diagnostics import CheckResult, check_llm, check_llm_bundled


def probe_llm(
    settings: Settings, session: Session, client: httpx.Client | None = None
) -> bool | None:
    """True when every endpoint post-lane work would call answers; None when none would.

    The targets mirror the execution routing: the bundled endpoint when it is
    active (enhancement, run assets, translation), and the BYO endpoint whenever a
    client would be built for it (topics and research always use it; without the
    bundle, so does everything else). A BYO endpoint that is the bundled URL is
    probed once. Any failure, including an exception, closes the lane.
    """
    try:
        row = get_app_settings(session)
        if not resolve_effective_llm_enabled(row, settings):
            return None
        bundled = llm_bundled_active(row, settings)
        base_url, model = resolve_effective_llm_endpoint(row, settings)
        api_key = resolve_effective_llm_api_key(row, settings)
        byo = (
            bool(base_url)
            and bool(model)
            and (bool(api_key) or llm_endpoint_explicitly_set(base_url))
            and not (bundled and base_url == settings.llm_bundled_base_url)
        )
        if not bundled and not byo:
            return None
        with ExitStack() as stack:
            if client is None:
                client = stack.enter_context(httpx.Client(timeout=3.0))
            results: list[CheckResult | None] = []
            if bundled:
                results.append(
                    check_llm_bundled(
                        active=True, base_url=settings.llm_bundled_base_url, client=client
                    )
                )
            if byo:
                results.append(
                    check_llm(
                        enabled=True,
                        configured=True,
                        base_url=base_url,
                        api_key=api_key,
                        client=client,
                    )
                )
            # An endpoint that should have been checked and was not counts as closed.
            return bool(results) and all(r is not None and r.ok for r in results)
    except Exception:
        return False
