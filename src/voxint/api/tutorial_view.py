"""The guided-tutorial banner context shared by the run and editor pages.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151),
then remapped in #158 to target the media editor instead of the retired
review pages.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from voxint.api.csrf import CSRF_CLAIM, CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import ready_tutorial_run_id
from voxint.db.models import PipelineRun
from voxint.tutorial.steps import (
    PAGE_ROUTE_NAME,
    STEP_COPY,
    STEP_PAGE,
    WALKTHROUGH_TOTAL,
    TutorialPage,
    TutorialStep,
    parse_tutorial_step,
    walkthrough_number,
)

logger = logging.getLogger(__name__)


def _tutorial_media_id(session: Session, run_id: uuid.UUID) -> uuid.UUID | None:
    """Look up the media_item_id for a tutorial run (editor path resolution)."""
    run = session.get(PipelineRun, run_id)
    return run.media_item_id if run is not None else None


def _step_path(
    request: Request,
    step: TutorialStep,
    session: Session | None = None,
    tutorial_run_id: uuid.UUID | None = None,
    **path_params: object,
) -> str:
    """The URL path of the page a step renders on, resolved from the route table.

    Goes through ``STEP_PAGE`` + ``PAGE_ROUTE_NAME`` (issue #152) so the banner's
    continue-links follow a page wherever a later phase moves it — remap the step
    in ``voxint.tutorial.steps`` and every derived link updates with it.

    For editor-bound steps (``TutorialPage.EDITOR``), the route requires a
    ``media_id`` path parameter. When ``session`` and ``tutorial_run_id`` are
    provided, the media_id is resolved from the run's ``media_item_id``.
    """
    page = STEP_PAGE[step]
    name = PAGE_ROUTE_NAME[page]
    params = {k: str(v) for k, v in path_params.items()}

    if (
        page == TutorialPage.EDITOR
        and "media_id" not in params
        and session is not None
        and tutorial_run_id is not None
    ):
        media_id = _tutorial_media_id(session, tutorial_run_id)
        if media_id is not None:
            params["media_id"] = str(media_id)

    return str(request.app.url_path_for(name, **params))


def _tutorial_banner(
    request: Request,
    session: Session,
    *,
    page: TutorialPage,
    run_id: uuid.UUID | None = None,
    token: uuid.UUID | None = None,
) -> dict[str, Any] | None:
    """Resolve the guided-tutorial banner context for a page, or ``None``.

    Renders nothing unless ALL hold: the ``?tutorial=`` value parses to a step
    (an absent/unknown value is a quiet no-banner, never a 422); that step's bound
    page (:data:`STEP_PAGE`) is THIS page; a tutorial run is configured AND still
    present (``ready_tutorial_run_id``); and — for the run-scoped pages — the
    route's ``run_id`` is exactly that tutorial run. So a ``?tutorial=`` spoofed
    onto any other run, or onto the wrong page, shows nothing. Read-only: it never
    creates the ``app_settings`` row and never mutates.

    The returned dict is a flat bag the banner partial reads; each step populates
    only the action fields it needs (a next-link, a claim form, or the export +
    finish controls).
    """
    step = parse_tutorial_step(request.query_params.get("tutorial"))
    if step is None or STEP_PAGE[step] is not page:
        return None
    tutorial_run_id = ready_tutorial_run_id(session)
    if tutorial_run_id is None:
        return None
    if (
        page in (TutorialPage.RUN_DETAIL, TutorialPage.EDITOR)
        and run_id != tutorial_run_id
    ):
        return None

    copy = STEP_COPY[step]
    secret = request.app.state.csrf_secret
    tutorial_media_id = _tutorial_media_id(session, tutorial_run_id)
    banner: dict[str, Any] = {
        "step": step.value,
        "n": walkthrough_number(step),
        "total": WALKTHROUGH_TOTAL,
        "title": copy.title,
        "body": copy.body,
        "next_href": None,
        "next_label": None,
        "claim_run_id": None,
        "claim_media_id": str(tutorial_media_id) if tutorial_media_id else None,
        "claim_label": "Claim the tutorial run →",
        "csrf_claim": None,
        "export_href": None,
        "csrf_settings": None,
    }
    if step is TutorialStep.RUN:
        editor_path = _step_path(
            request, TutorialStep.REVIEW,
            session=session, tutorial_run_id=tutorial_run_id,
        )
        banner["next_href"] = f"{editor_path}?run={tutorial_run_id}&tutorial=review"
        banner["next_label"] = "Open the editor →"
    elif step is TutorialStep.REVIEW:
        banner["next_href"] = (
            f"{request.url.path}?run={tutorial_run_id}"
            f"&token={token}&tutorial=adjudicate"
            if token is not None
            else None
        )
        banner["next_label"] = "I've claimed — continue →" if token else None
        if token is None:
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.ADJUDICATE:
        banner["next_href"] = (
            f"{request.url.path}?run={tutorial_run_id}"
            f"&token={token}&tutorial=check_words"
            if token is not None
            else None
        )
        banner["next_label"] = "Continue to checking the words →" if token else None
        if token is None:
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.CHECK_WORDS:
        banner["next_href"] = (
            f"{request.url.path}?run={tutorial_run_id}"
            f"&token={token}&tutorial=export"
            if token is not None
            else None
        )
        banner["next_label"] = "I've checked the words →" if token else None
        if token is None:
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.EXPORT:
        banner["export_href"] = f"/review/{tutorial_run_id}/export.txt"
        banner["csrf_settings"] = mint_csrf_token(secret, CSRF_SETTINGS)
    return banner
