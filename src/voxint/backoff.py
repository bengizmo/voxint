"""Shared bounded retry delay."""


def backoff_seconds(attempts: int, base: float, cap: float) -> float:
    """Exponential in completed attempts, capped; jitter is the caller's."""
    exp = min(max(attempts - 1, 0), 30)
    return float(min(base * 2**exp, cap))
