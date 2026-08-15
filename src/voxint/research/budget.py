"""Per-invocation retrieval budget + mandatory fetch attribution (issue #39).

The budget is the tool-side half of the issue #40 contract: quotas are enforced
IN the tools, so the future research loop can hand one :class:`ResearchBudget`
to every tool call and rely on a structured ``budget_exhausted`` outcome once a
quota is spent — the tools never fetch past it, whatever the caller does.
Consumption is atomic (a lock covers the check-and-decrement), so concurrent
tool calls cannot both spend the last unit.

:class:`Attribution` is a mandatory argument on every retrieval operation — a
caller physically cannot fetch without stating which feature is asking and why.
Both fields are bounded machine identifiers, not free text: they are emitted
into every audit log line, and an unbounded string would be a log-injection and
secret-leak surface.
"""

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# Lowercase machine identifiers only — these land verbatim in audit log lines.
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Attribution:
    """Who is fetching and why — stamped on every outbound request's log line."""

    feature: str
    reason: str

    def __post_init__(self) -> None:
        for field_name, value in (("feature", self.feature), ("reason", self.reason)):
            if not _IDENTIFIER.match(value):
                raise ValueError(
                    f"attribution {field_name} must be a bounded lowercase"
                    " identifier ([a-z0-9._-], max 64 chars)"
                )


class ResearchBudget:
    """Atomic per-invocation quota over searches, URL reads, and wall clock.

    ``deadline_seconds`` starts the wall clock at construction time (monotonic).
    A ``try_consume_*`` call only succeeds while its counter has headroom AND
    the deadline (when set) has not passed; it decrements atomically under a
    lock. ``remaining_seconds`` feeds per-request timeout derivation in the
    tools — DNS resolution is the one step a deadline cannot hard-interrupt
    (documented in docs/architecture.md), so the guarantee is "no NEW work
    starts past the deadline and every bounded step gets only the remaining
    time", not a hard interrupt.
    """

    def __init__(
        self,
        *,
        max_searches: int,
        max_reads: int,
        deadline_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_searches < 0 or max_reads < 0:
            raise ValueError("budget maxima must be >= 0")
        if deadline_seconds is not None and deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive when set")
        self._lock = threading.Lock()
        self._clock = clock
        self._searches_left = max_searches
        self._reads_left = max_reads
        self._deadline = (
            None if deadline_seconds is None else clock() + deadline_seconds
        )

    def _expired_locked(self) -> bool:
        return self._deadline is not None and self._clock() >= self._deadline

    @property
    def expired(self) -> bool:
        with self._lock:
            return self._expired_locked()

    @property
    def searches_left(self) -> int:
        with self._lock:
            return self._searches_left

    @property
    def reads_left(self) -> int:
        with self._lock:
            return self._reads_left

    def remaining_seconds(self) -> float | None:
        """Time left before the deadline (``None`` = no deadline; floor 0.0)."""
        with self._lock:
            if self._deadline is None:
                return None
            return max(0.0, self._deadline - self._clock())

    def try_consume_search(self) -> bool:
        with self._lock:
            if self._expired_locked() or self._searches_left <= 0:
                return False
            self._searches_left -= 1
            return True

    def try_consume_read(self) -> bool:
        with self._lock:
            if self._expired_locked() or self._reads_left <= 0:
                return False
            self._reads_left -= 1
            return True
