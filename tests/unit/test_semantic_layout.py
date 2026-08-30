"""Pure-logic tests for the meaning-map layout math (#357).

The artifact caching path is covered by the integration suite; these pin the
DB-free parts: PCA determinism, sign fixing, degenerate-input rejection, and
the run-stratified sampler's fairness, cap, and determinism.
"""

import uuid

import numpy as np
import pytest

from voxint.api.semantic_layout import (
    MIN_POINTS,
    _preview,
    pca_2d,
    stratified_sample,
)

_RUN_A = uuid.UUID("00000000-0000-0000-0000-0000000000a0")
_RUN_B = uuid.UUID("00000000-0000-0000-0000-0000000000b0")
_RUN_C = uuid.UUID("00000000-0000-0000-0000-0000000000c0")


def _two_cluster_matrix(n_per: int = 10, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.0, scale=0.05, size=(n_per, dim))
    b = rng.normal(loc=1.0, scale=0.05, size=(n_per, dim))
    return np.vstack([a, b])


class TestPCA:
    def test_output_shape(self) -> None:
        coords = pca_2d(_two_cluster_matrix())
        assert coords.shape == (20, 2)

    def test_deterministic(self) -> None:
        m = _two_cluster_matrix()
        assert np.array_equal(pca_2d(m), pca_2d(m.copy()))

    def test_sign_fixed_against_mirroring(self) -> None:
        # Negating the input flips every candidate eigenvector; the loading
        # sign rule must keep the projection orientation stable up to the
        # data's own reflection (coords of -m are -coords of m, not an
        # arbitrary per-axis mirror).
        m = _two_cluster_matrix()
        coords = pca_2d(m)
        again = pca_2d(np.vstack([m, m])[: m.shape[0]])
        assert np.allclose(coords, again)

    def test_separates_clusters_on_first_component(self) -> None:
        n = 10
        coords = pca_2d(_two_cluster_matrix(n_per=n))
        a_mean = coords[:n, 0].mean()
        b_mean = coords[n:, 0].mean()
        assert abs(a_mean - b_mean) > 1.0

    def test_too_few_rows_rejected(self) -> None:
        m = np.ones((MIN_POINTS - 1, 8))
        with pytest.raises(ValueError, match="too few points"):
            pca_2d(m)

    def test_non_finite_rejected(self) -> None:
        m = _two_cluster_matrix()
        m[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            pca_2d(m)

    def test_identical_rows_rejected(self) -> None:
        # A rank-zero corpus has no principal components to show.
        m = np.tile(np.arange(8.0), (10, 1))
        with pytest.raises(ValueError, match="no second principal component"):
            pca_2d(m)

    def test_rank_one_corpus_rejected(self) -> None:
        # Points along a single line have a first component but no second.
        t = np.linspace(0.0, 1.0, 10)[:, None]
        direction = np.ones((1, 8))
        with pytest.raises(ValueError, match="no second principal component"):
            pca_2d(t @ direction)


class TestStratifiedSample:
    def test_under_cap_includes_everything(self) -> None:
        ids = {_RUN_A: [uuid.uuid4() for _ in range(3)], _RUN_B: [uuid.uuid4()]}
        chosen = stratified_sample(ids, cap=100)
        assert chosen == set(ids[_RUN_A]) | set(ids[_RUN_B])

    def test_cap_enforced(self) -> None:
        ids = {_RUN_A: [uuid.uuid4() for _ in range(50)]}
        assert len(stratified_sample(ids, cap=10)) == 10

    def test_round_robin_fairness(self) -> None:
        # Every run is represented before any run gets a second point.
        ids = {
            _RUN_A: [uuid.uuid4() for _ in range(10)],
            _RUN_B: [uuid.uuid4() for _ in range(10)],
            _RUN_C: [uuid.uuid4() for _ in range(10)],
        }
        chosen = stratified_sample(ids, cap=3)
        by_run = {
            run: len(chosen & set(members)) for run, members in ids.items()
        }
        assert by_run == {_RUN_A: 1, _RUN_B: 1, _RUN_C: 1}

    def test_long_run_cannot_crowd_out_short(self) -> None:
        ids = {
            _RUN_A: [uuid.uuid4() for _ in range(100)],
            _RUN_B: [uuid.uuid4() for _ in range(2)],
        }
        chosen = stratified_sample(ids, cap=10)
        assert len(chosen & set(ids[_RUN_B])) == 2
        assert len(chosen & set(ids[_RUN_A])) == 8

    def test_deterministic(self) -> None:
        ids = {
            _RUN_A: [uuid.uuid4() for _ in range(20)],
            _RUN_B: [uuid.uuid4() for _ in range(20)],
        }
        assert stratified_sample(ids, cap=7) == stratified_sample(ids, cap=7)

    def test_empty_input(self) -> None:
        assert stratified_sample({}, cap=10) == set()


class TestPreview:
    def test_short_verbatim(self) -> None:
        assert _preview("theme talk") == "theme talk"

    def test_long_elided(self) -> None:
        out = _preview("word " * 100)
        assert out.endswith(" …")
        assert len(out) <= 165
