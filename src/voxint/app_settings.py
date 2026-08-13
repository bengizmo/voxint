"""Repository for the singleton ``app_settings`` row — the first-run wizard's store.

Split from ``config.Settings`` (env-only, frozen at process start): infra config
and secrets stay in the environment; the user-facing preferences the wizard writes
live here in the DB. Exactly one row (``id = 1``) ever exists. Callers own the
transaction — every function takes a live ``Session`` and never commits.
"""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from voxint.db.models import AppSettings

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
