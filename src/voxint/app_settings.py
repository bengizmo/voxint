"""Repository for the singleton ``app_settings`` row — the first-run wizard's store.

Split from ``config.Settings`` (env-only, frozen at process start): infra config
and secrets stay in the environment; the user-facing preferences the wizard writes
live here in the DB. Exactly one row (``id = 1``) ever exists. Callers own the
transaction — every function takes a live ``Session`` and never commits.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import AppSettings, PipelineRun

SINGLETON_ID = 1


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
