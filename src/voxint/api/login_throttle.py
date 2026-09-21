"""Process-local login limits for a single-instance, single-event-loop server."""

import math
import time
from collections import defaultdict, deque
from collections.abc import Callable


class LoginThrottle:
    """Sliding failure windows; in-flight reservations prevent parallel overshoot.

    Call start/finish on the event loop, without awaiting between checking and
    reserving. Rejected requests never extend the window. No account lookup is
    involved, so nonexistent and existing usernames receive identical treatment.
    """

    WINDOW = 300
    ACCOUNT_LIMIT = 5
    SOURCE_LIMIT = 20

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._pending: dict[tuple[str, str], int] = defaultdict(int)

    def start(self, account: str, source: str) -> int:
        """Reserve an attempt, or return a positive Retry-After in seconds."""
        now = self._clock()
        # Also discard abandoned keys, including random nonexistent usernames.
        for key, failures in list(self._failures.items()):
            while failures and failures[0] <= now - self.WINDOW:
                failures.popleft()
            if not failures:
                del self._failures[key]
        keys = (("account", account), ("source", source))
        retry = 0
        for key, limit in zip(keys, (self.ACCOUNT_LIMIT, self.SOURCE_LIMIT), strict=True):
            failures = self._failures.get(key, deque())
            if len(failures) >= limit:
                retry = max(retry, math.ceil(failures[0] + self.WINDOW - now))
            elif len(failures) + self._pending.get(key, 0) >= limit:
                retry = max(retry, 1)
        if retry:
            return retry
        for key in keys:
            self._pending[key] += 1
        return 0

    def finish(self, account: str, source: str, *, success: bool | None) -> float:
        """Release reservation; None means an error, not a credential failure.

        Failures three and onward delay by 1, 2, 4, then at most 8 seconds.
        Success clears account failures only; source failures still expire normally.
        """
        keys = (("account", account), ("source", source))
        for key in keys:
            self._pending[key] -= 1
            if not self._pending[key]:
                del self._pending[key]
        if success is None:
            return 0
        if success:
            self._failures.pop(keys[0], None)
            return 0
        now = self._clock()
        count = 0
        for key in keys:
            failures = self._failures[key]
            while failures and failures[0] <= now - self.WINDOW:
                failures.popleft()
            failures.append(now)
            count = max(count, len(failures))
        return float(2 ** min(count - 3, 3)) if count >= 3 else 0
