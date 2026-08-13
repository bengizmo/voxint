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


def get_or_create(session: Session) -> AppSettings:
    """Return the singleton row, inserting a defaulted one if absent.

    The insert is wrapped in a SAVEPOINT so the UNIQUE(id) race between two
    first-time writers rolls back only the losing insert — not the caller's outer
    transaction — letting us re-read and adopt the winner's row (mirrors
    ``ingest.service._get_or_create_media``).
    """
    row = session.get(AppSettings, SINGLETON_ID)
    if row is not None:
        return row
    row = AppSettings(id=SINGLETON_ID)
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        adopted = session.get(AppSettings, SINGLETON_ID)
        assert adopted is not None  # the winner committed the singleton
        return adopted
    return row


def complete_onboarding(session: Session) -> AppSettings:
    """Mark the wizard finished (idempotent, get-or-create). Caller commits."""
    row = get_or_create(session)
    row.onboarding_complete = True
    session.flush()
    return row
