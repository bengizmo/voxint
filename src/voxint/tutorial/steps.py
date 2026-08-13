"""The guided-tutorial step model — the server-rendered ``?tutorial=<step>`` mode.

Slice 6 teaches Voxint by walking a first-run operator through ONE full
adjudication on the pre-seeded three-speaker sample (see :mod:`voxint.tutorial.seed`).
Rather than client-side coach-marks (brittle under htmx fragment swaps), each step
is a server-rendered banner injected above the body of an EXISTING console page.

This module is the pure-domain half: the step enum, its order, the step→page
binding, and a lenient parser. It imports no FastAPI/DB — the API layer
(``voxint.api.app``) resolves whether a request actually shows the real tutorial
run and builds the per-step next-links/tokens; here we only say which step maps to
which page and how it is worded.

A banner renders only when BOTH the ``?tutorial=<step>`` value parses AND the page
the request hit is the step's bound page. ``ADJUDICATE`` and ``EXPORT`` share the
workbench page (``/review/{id}``), so a page identity — not the run id alone —
disambiguates them; the API layer additionally requires the route's run id to be
the configured ``tutorial_run_id`` before rendering anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TutorialStep(StrEnum):
    """The guided-tutorial steps; the value is the ``?tutorial=`` query token."""

    RUN = "run"
    REVIEW = "review"
    ADJUDICATE = "adjudicate"
    EXPORT = "export"
    DONE = "done"


# The four numbered walkthrough steps, in order ("step N of 4"). DONE is terminal
# (a completion celebration on the Settings page), not a numbered walkthrough step,
# so it is deliberately excluded from the count.
WALKTHROUGH_STEPS: tuple[TutorialStep, ...] = (
    TutorialStep.RUN,
    TutorialStep.REVIEW,
    TutorialStep.ADJUDICATE,
    TutorialStep.EXPORT,
)


class TutorialPage(StrEnum):
    """Stable identifiers for the console pages a banner can render on.

    Each GET handler passes its own page id to the banner resolver; a step renders
    only on its bound page (:data:`STEP_PAGE`), so a ``?tutorial=adjudicate`` value
    on the run-detail page — or an ``?tutorial=export`` on the queue — shows nothing.
    """

    RUN_DETAIL = "run_detail"
    REVIEW_QUEUE = "review_queue"
    WORKBENCH = "workbench"
    SETTINGS = "settings"


# Which page each step is allowed to render on. ADJUDICATE and EXPORT both bind to
# the workbench (the same ``/review/{id}`` page, two banners); the API layer keys
# the run identity check off ``tutorial_run_id`` so neither renders on another run.
#
# DONE binds to SETTINGS, but the Settings page renders its terminal completion
# celebration INLINE (it needs a "submit your own media" link the walkthrough
# banner has no place for), NOT through ``_tutorial_banner``. DONE's entry here is
# still load-bearing: it makes ``?tutorial=done`` on a run/review/workbench page a
# clean page-mismatch (no banner) instead of a ``KeyError`` on this dict.
STEP_PAGE: dict[TutorialStep, TutorialPage] = {
    TutorialStep.RUN: TutorialPage.RUN_DETAIL,
    TutorialStep.REVIEW: TutorialPage.REVIEW_QUEUE,
    TutorialStep.ADJUDICATE: TutorialPage.WORKBENCH,
    TutorialStep.EXPORT: TutorialPage.WORKBENCH,
    TutorialStep.DONE: TutorialPage.SETTINGS,
}


@dataclass(frozen=True)
class StepCopy:
    """Clean-room banner wording for one step (no real brands/PII)."""

    title: str
    body: str


# Banner copy per step. Clean-room by construction: no real people, brands, or
# private hostnames — only the tutorial's own reserved sample voices, which the
# committed fixtures already name. The ADJUDICATE body points the operator at the
# evidence the labels fragment already renders (the grounded-vs-heard contrast)
# rather than re-explaining it here.
STEP_COPY: dict[TutorialStep, StepCopy] = {
    TutorialStep.RUN: StepCopy(
        title="Your tutorial run",
        body=(
            "This three-speaker sample has already been transcribed and split by "
            "voice, so you can see the whole result at once. Look over the stage "
            "ledger and open the transcript below, then continue to the review "
            "console to attribute each voice to a speaker."
        ),
    ),
    TutorialStep.REVIEW: StepCopy(
        title="Claim the tutorial run",
        body=(
            "Reviewing a run “claims” it, so only you can rule on its "
            "voices while you work. Claim the sample below to start attributing "
            "its three speakers."
        ),
    ),
    TutorialStep.ADJUDICATE: StepCopy(
        title="Attribute the three voices",
        body=(
            "Each voice below shows its evidence. One has a grounded machine match "
            "you can accept; one shows a heard name that is only a guess — you "
            "decide whether to trust it; one has no name at all. Assign an existing "
            "speaker, enroll a new one, or mark a voice excluded or unknown. Your "
            "rulings update the list below as you go."
        ),
    ),
    TutorialStep.EXPORT: StepCopy(
        title="Export the attributed transcript",
        body=(
            "Open the speaker-labelled transcript to see the finished result — "
            "that is the whole loop: submit, review, attribute, export. Finish the "
            "tutorial when you are ready."
        ),
    ),
    TutorialStep.DONE: StepCopy(
        title="You finished the tutorial \U0001f389",
        body=(
            "You have completed one full run end to end. When you are ready, submit "
            "your own audio or video and do the same with real speakers."
        ),
    ),
}


def parse_tutorial_step(raw: str | None) -> TutorialStep | None:
    """Map a ``?tutorial=`` value to a :class:`TutorialStep`, or ``None``.

    Lenient on purpose: an absent, blank, or unknown value yields ``None`` (no
    banner) rather than a 422 — the tutorial is a navigational overlay, and a
    stray/typo'd query param must never break the underlying page.
    """
    if not raw:
        return None
    try:
        return TutorialStep(raw)
    except ValueError:
        return None


def walkthrough_number(step: TutorialStep) -> int | None:
    """1-based position of ``step`` among the four numbered steps, else ``None``."""
    if step in WALKTHROUGH_STEPS:
        return WALKTHROUGH_STEPS.index(step) + 1
    return None


WALKTHROUGH_TOTAL = len(WALKTHROUGH_STEPS)
