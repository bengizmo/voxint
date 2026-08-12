"""Space-tagged embedding vectors — the harness's cross-space guardrail.

Embedding models emit vectors in *different spaces*; two spaces can share a
dimensionality, so a dims check is not an isolation check. Every vector that
enters harness scoring carries its ``embedding_space`` tag, and every cosine
entry point rejects a space mismatch before touching numpy. Raw ndarray dot
products stay private to this module.
"""

from dataclasses import dataclass, field

import numpy as np


class SpaceMismatchError(ValueError):
    """Two vectors from different embedding spaces were about to be compared."""


class InvalidVectorError(ValueError):
    """A vector failed structural validation (shape, finiteness, norm, space)."""


@dataclass(frozen=True)
class TaggedVector:
    """A finite, non-zero 1-D embedding vector bound to its embedding space.

    Construction validates; a ``TaggedVector`` that exists is safe to score.
    ``unit`` is the L2-normalized ndarray (cached, never mutated).
    """

    space: str
    values: tuple[float, ...]
    _unit: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space.strip():
            raise InvalidVectorError("embedding space must be a non-empty string")
        try:
            arr = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidVectorError(f"not a numeric vector: {exc}") from exc
        if arr.ndim != 1 or arr.size == 0:
            raise InvalidVectorError("expected a non-empty 1-D vector")
        if not np.all(np.isfinite(arr)):
            raise InvalidVectorError("vector has non-finite components")
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm == 0.0:
            raise InvalidVectorError("zero-norm vector")
        object.__setattr__(self, "_unit", arr / norm)

    @property
    def dims(self) -> int:
        return len(self.values)

    @property
    def unit(self) -> np.ndarray:
        return self._unit


def tagged_vector(space: str, values: object) -> TaggedVector:
    """Build a :class:`TaggedVector` from untrusted input (e.g. parsed JSON).

    Rejects strings/bytes/mappings/booleans-in-lists up front so a JSON payload
    like ``"[1,2]"`` or ``{"0": 1.0}`` cannot masquerade as a vector.
    """
    if values is None or isinstance(values, (str, bytes, dict)):
        raise InvalidVectorError("embedding must be a list of numbers")
    if isinstance(values, (list, tuple)):
        cleaned: list[float] = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidVectorError("embedding must contain only numbers")
            cleaned.append(float(v))
        return TaggedVector(space=space, values=tuple(cleaned))
    raise InvalidVectorError("embedding must be a list of numbers")


def cosine(a: TaggedVector, b: TaggedVector) -> float:
    """Cosine similarity of two same-space tagged vectors.

    Raises :class:`SpaceMismatchError` on differing spaces and
    :class:`InvalidVectorError` on differing dimensionality (same space implies
    same model implies same dims; a mismatch means corrupt input).
    """
    if a.space != b.space:
        raise SpaceMismatchError(
            f"cross-space cosine refused: {a.space!r} vs {b.space!r}"
        )
    if a.dims != b.dims:
        raise InvalidVectorError(
            f"dims mismatch within space {a.space!r}: {a.dims} vs {b.dims}"
        )
    return float(np.dot(a.unit, b.unit))
