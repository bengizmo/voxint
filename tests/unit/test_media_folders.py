"""Pure folder-layer helpers (issue #153)."""

from __future__ import annotations

from voxint.media.folders import (
    deepest_ancestor,
    is_ancestor,
    nested_pairs,
    overlapping_registration,
)


class TestDeepestAncestor:
    def test_deepest_wins(self) -> None:
        assert deepest_ancestor("audio/pods/ep.wav", ["audio", "audio/pods"]) == "audio/pods"

    def test_component_boundary_not_string_prefix(self) -> None:
        # 'audio/pod' must not match a file under 'audio/podcasts'.
        assert deepest_ancestor("audio/podcasts/ep.wav", ["audio/pod"]) is None

    def test_root_folder_matches_everything(self) -> None:
        assert deepest_ancestor("anything/deep/x.wav", ["."]) == "."

    def test_root_is_shallowest(self) -> None:
        assert deepest_ancestor("audio/x.wav", [".", "audio"]) == "audio"

    def test_no_match(self) -> None:
        assert deepest_ancestor("incoming/uuid/source", ["audio", "podcasts"]) is None

    def test_exact_folder_matches(self) -> None:
        assert deepest_ancestor("audio", ["audio"]) == "audio"


class TestOverlap:
    def test_is_ancestor_proper_only(self) -> None:
        assert is_ancestor("audio", "audio/pods")
        assert not is_ancestor("audio", "audio")  # equal is not a proper ancestor
        assert not is_ancestor("audio/pods", "audio")

    def test_root_is_ancestor_of_all(self) -> None:
        assert is_ancestor(".", "audio")

    def test_overlapping_registration_both_directions(self) -> None:
        assert overlapping_registration("audio/pods", ["audio"]) == "audio"
        assert overlapping_registration("audio", ["audio/pods"]) == "audio/pods"

    def test_overlapping_ignores_exact_duplicate(self) -> None:
        # An exact duplicate is the UNIQUE(path) constraint's job, not overlap's.
        assert overlapping_registration("audio", ["audio"]) is None

    def test_no_overlap_for_siblings(self) -> None:
        assert overlapping_registration("audio", ["podcasts", "video"]) is None

    def test_nested_pairs(self) -> None:
        assert nested_pairs(["audio", "audio/pods", "podcasts"]) == [("audio", "audio/pods")]
