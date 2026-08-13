"""The guided-tutorial step model (pure domain, no DB/FastAPI).

Covers the lenient parser (unknown/blank → no banner, never a 422), the
step→page binding that keeps ADJUDICATE and EXPORT on the workbench while
disqualifying the wrong page, and the 1-of-4 walkthrough numbering that excludes
the terminal DONE step.
"""

import pytest

from voxint.tutorial.steps import (
    STEP_COPY,
    STEP_PAGE,
    WALKTHROUGH_STEPS,
    WALKTHROUGH_TOTAL,
    TutorialPage,
    TutorialStep,
    parse_tutorial_step,
    walkthrough_number,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("run", TutorialStep.RUN),
        ("review", TutorialStep.REVIEW),
        ("adjudicate", TutorialStep.ADJUDICATE),
        ("export", TutorialStep.EXPORT),
        ("done", TutorialStep.DONE),
    ],
)
def test_parse_valid_steps(raw: str, expected: TutorialStep) -> None:
    assert parse_tutorial_step(raw) is expected


@pytest.mark.parametrize("raw", [None, "", "bogus", "RUN", "adjudicate ", "0"])
def test_parse_absent_or_unknown_is_none(raw: str | None) -> None:
    # A stray/typo'd/absent value is a quiet no-banner, never an exception — the
    # tutorial is a navigational overlay that must not break the underlying page.
    assert parse_tutorial_step(raw) is None


def test_step_page_binds_every_step() -> None:
    # Every step must have a page (the resolver indexes STEP_PAGE unconditionally).
    assert set(STEP_PAGE) == set(TutorialStep)


def test_adjudicate_and_export_share_the_workbench() -> None:
    assert STEP_PAGE[TutorialStep.ADJUDICATE] is TutorialPage.WORKBENCH
    assert STEP_PAGE[TutorialStep.EXPORT] is TutorialPage.WORKBENCH


def test_run_review_done_pages() -> None:
    assert STEP_PAGE[TutorialStep.RUN] is TutorialPage.RUN_DETAIL
    assert STEP_PAGE[TutorialStep.REVIEW] is TutorialPage.REVIEW_QUEUE
    assert STEP_PAGE[TutorialStep.DONE] is TutorialPage.SETTINGS


def test_walkthrough_numbering() -> None:
    assert walkthrough_number(TutorialStep.RUN) == 1
    assert walkthrough_number(TutorialStep.REVIEW) == 2
    assert walkthrough_number(TutorialStep.ADJUDICATE) == 3
    assert walkthrough_number(TutorialStep.EXPORT) == 4
    # DONE is terminal — not one of the four numbered steps.
    assert walkthrough_number(TutorialStep.DONE) is None


def test_walkthrough_total_and_order() -> None:
    assert WALKTHROUGH_TOTAL == 4
    assert WALKTHROUGH_STEPS == (
        TutorialStep.RUN,
        TutorialStep.REVIEW,
        TutorialStep.ADJUDICATE,
        TutorialStep.EXPORT,
    )
    assert TutorialStep.DONE not in WALKTHROUGH_STEPS


def test_every_step_has_nonempty_copy() -> None:
    for step in TutorialStep:
        copy = STEP_COPY[step]
        assert copy.title.strip()
        assert copy.body.strip()
