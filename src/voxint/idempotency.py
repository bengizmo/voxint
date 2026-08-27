"""Savepoint-based insert-or-adopt-or-conflict skeleton.

Every write path that guards a UNIQUE idempotency key follows the same dance:
look up by key, attempt a savepoint insert, and on IntegrityError re-look up
to distinguish a concurrent-winner adopt from a genuine constraint violation.
This module extracts that skeleton so new write paths get it right by
construction instead of copying ~30 lines each time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

T = TypeVar("T")


def savepoint_adopt_or_conflict(
    session: Session,
    *,
    lookup: Callable[[], T | None],
    adopt_or_conflict: Callable[[T], T],
    persist: Callable[[], T],
) -> T:
    """Insert-or-adopt with savepoint safety.

    1. ``lookup()`` -- if a row already keys this nonce, hand it to
       ``adopt_or_conflict`` which returns it (identical replay) or raises
       the caller's chosen exception (payload mismatch).
    2. ``persist()`` inside ``begin_nested()`` -- build the row, add it to
       the session, flush, and return it.  Any post-insert work that belongs
       inside the savepoint (candidate insertion, supersession) goes here.
    3. On ``IntegrityError`` from the savepoint: re-run ``lookup()`` to
       distinguish a concurrent-winner race (adopt/conflict) from a real
       constraint violation (re-raise).

    Callers needing advisory locks should acquire them and call ``lookup()``
    themselves before invoking this function; the helper's own ``lookup()``
    then acts as the post-lock re-check.
    """
    existing = lookup()
    if existing is not None:
        return adopt_or_conflict(existing)
    try:
        with session.begin_nested():
            row = persist()
    except IntegrityError:
        existing = lookup()
        if existing is None:
            raise
        return adopt_or_conflict(existing)
    return row
