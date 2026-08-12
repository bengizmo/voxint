"""Tagged-vector validation + the cross-space cosine guardrail."""

import numpy as np
import pytest

from voxint.harness.vectors import (
    InvalidVectorError,
    SpaceMismatchError,
    TaggedVector,
    cosine,
    tagged_vector,
)


def test_valid_vector_is_unit_normalized() -> None:
    vec = TaggedVector(space="space-a", values=(3.0, 4.0))
    assert vec.dims == 2
    assert np.allclose(vec.unit, [0.6, 0.8])


@pytest.mark.parametrize("space", ["", "   "])
def test_blank_space_rejected(space: str) -> None:
    with pytest.raises(InvalidVectorError, match="non-empty string"):
        TaggedVector(space=space, values=(1.0,))


def test_empty_vector_rejected() -> None:
    with pytest.raises(InvalidVectorError, match="non-empty 1-D"):
        TaggedVector(space="s", values=())


def test_non_finite_rejected() -> None:
    with pytest.raises(InvalidVectorError, match="non-finite"):
        TaggedVector(space="s", values=(1.0, float("nan")))
    with pytest.raises(InvalidVectorError, match="non-finite"):
        TaggedVector(space="s", values=(float("inf"), 0.0))


def test_zero_norm_rejected() -> None:
    with pytest.raises(InvalidVectorError, match="zero-norm"):
        TaggedVector(space="s", values=(0.0, 0.0))


@pytest.mark.parametrize(
    "values", [None, "1,2,3", b"\x00", {"0": 1.0}, [[1.0, 2.0]], [1.0, "x"], [True, 1.0]]
)
def test_tagged_vector_rejects_non_numeric_payloads(values: object) -> None:
    with pytest.raises(InvalidVectorError):
        tagged_vector("s", values)


def test_tagged_vector_accepts_list_of_numbers() -> None:
    vec = tagged_vector("s", [1, 2.5, 3])
    assert vec.values == (1.0, 2.5, 3.0)


def test_cosine_same_space() -> None:
    a = TaggedVector(space="s", values=(1.0, 0.0))
    b = TaggedVector(space="s", values=(0.0, 1.0))
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_is_scale_invariant() -> None:
    a = TaggedVector(space="s", values=(1.0, 2.0, 3.0))
    b = TaggedVector(space="s", values=(10.0, 20.0, 30.0))
    assert cosine(a, b) == pytest.approx(1.0)


def test_cross_space_cosine_refused_even_at_equal_dims() -> None:
    """The invariant: equal dimensionality is NOT a license to compare."""
    a = TaggedVector(space="model-a", values=(1.0, 0.0, 0.0))
    b = TaggedVector(space="model-b", values=(1.0, 0.0, 0.0))
    with pytest.raises(SpaceMismatchError, match="cross-space"):
        cosine(a, b)


def test_same_space_dims_mismatch_is_an_error_not_a_skip() -> None:
    a = TaggedVector(space="s", values=(1.0, 0.0))
    b = TaggedVector(space="s", values=(1.0, 0.0, 0.0))
    with pytest.raises(InvalidVectorError, match="dims mismatch"):
        cosine(a, b)
