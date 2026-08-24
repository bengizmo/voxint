"""The guided-tutorial banner context shared by the run and review pages.

Moved verbatim from ``api/app.py`` in the P0b router decomposition (#151):
the queue, run, and review surfaces all render it, so it lives in a neutral
module none of them own.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from voxint.api.csrf import CSRF_CLAIM, CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import ready_tutorial_run_id
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


def _step_path(request: Request, step: TutorialStep, **path_params: object) -> str:
    """The URL path of the page a step renders on, resolved from the route table.

    Goes through ``STEP_PAGE`` + ``PAGE_ROUTE_NAME`` (issue #152) so the banner's
    continue-links follow a page wherever a later phase moves it — remap the step
    in ``voxint.tutorial.steps`` and every derived link updates with it.
    """
    name = PAGE_ROUTE_NAME[STEP_PAGE[step]]
    # request.app is typed Any; str() keeps the declared return honest (URLPath
    # is a str subclass).
    return str(
        request.app.url_path_for(name, **{k: str(v) for k, v in path_params.items()})
    )

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
    finish controls). The adjudicate→check_words→export next-links carry the
    verified claim ``token`` so the workbench and transcript stepper stay writable;
    the export link never does.
    """
    step = parse_tutorial_step(request.query_params.get("tutorial"))
    if step is None or STEP_PAGE[step] is not page:
        return None
    tutorial_run_id = ready_tutorial_run_id(session)
    if tutorial_run_id is None:
        return None
    # Run-scoped pages must be showing THE tutorial run; the queue page carries no
    # run_id and only needs the tutorial run to exist (checked above).
    if (
        page in (TutorialPage.RUN_DETAIL, TutorialPage.WORKBENCH, TutorialPage.TRANSCRIPT)
        and run_id != tutorial_run_id
    ):
        return None

    copy = STEP_COPY[step]
    secret = request.app.state.csrf_secret
    banner: dict[str, Any] = {
        "step": step.value,
        "n": walkthrough_number(step),
        "total": WALKTHROUGH_TOTAL,
        "title": copy.title,
        "body": copy.body,
        "next_href": None,
        "next_label": None,
        "claim_run_id": None,
        "claim_label": "Claim the tutorial run →",
        "csrf_claim": None,
        "export_href": None,
        "csrf_settings": None,
    }
    if step is TutorialStep.RUN:
        banner["next_href"] = f"{_step_path(request, TutorialStep.REVIEW)}?tutorial=review"
        banner["next_label"] = "Open the review console →"
    elif step is TutorialStep.REVIEW:
        banner["claim_run_id"] = tutorial_run_id
        banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.ADJUDICATE:
        if token is not None:
            # Step 1 → Step 2: hand off to the transcript stepper (issue #117 Phase
            # B), carrying the live claim token so verify/edit stay enabled there.
            transcript = _step_path(
                request, TutorialStep.CHECK_WORDS, run_id=tutorial_run_id
            )
            banner["next_href"] = f"{transcript}?token={token}&tutorial=check_words"
            banner["next_label"] = "Continue to checking the words →"
        else:
            # No live claim on this tab — offer to (re)claim and continue rather
            # than a dead next-link that would land on a read-only workbench.
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.CHECK_WORDS:
        if token is not None:
            # Step 2 → export: stay on the transcript page (both steps share it),
            # keeping the claim token so a returning tab is still writable.
            transcript = _step_path(request, TutorialStep.EXPORT, run_id=tutorial_run_id)
            banner["next_href"] = f"{transcript}?token={token}&tutorial=export"
            banner["next_label"] = "I've checked the words →"
        else:
            # A stale/absent token here means the workbench claim is gone; recover
            # by re-claiming (which re-enters the walkthrough on run identity)
            # rather than offering a dead read-only next-link.
            banner["claim_run_id"] = tutorial_run_id
            banner["csrf_claim"] = mint_csrf_token(secret, CSRF_CLAIM)
    elif step is TutorialStep.EXPORT:
        # Plaintext export opens in a new tab; the claim token is deliberately NOT
        # placed in its URL. Finishing is an explicit CSRF-guarded POST.
        banner["export_href"] = f"/review/{tutorial_run_id}/export.txt"
        banner["csrf_settings"] = mint_csrf_token(secret, CSRF_SETTINGS)
    return banner


