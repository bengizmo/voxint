"""Pure folder-layer helpers and the migration preflight report (issue #153)."""

from __future__ import annotations

from voxint.media.folders import (
    build_preflight_report,
    deepest_ancestor,
    folder_pack_name,
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


class TestFolderPackName:
    def test_deepest_mapped_ancestor(self) -> None:
        assert folder_pack_name("audio/pods/ep.wav", {"audio": "p1"}) == "p1"

    def test_unmapped_is_none(self) -> None:
        assert folder_pack_name("podcasts/ep.wav", {"audio": "p1"}) is None

    def test_empty_mapping(self) -> None:
        assert folder_pack_name("audio/x.wav", {}) is None


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


class TestPreflightReport:
    def test_clean_flat_registration_is_ok(self) -> None:
        report = build_preflight_report(
            folders=["audio", "podcasts"],
            folder_domain_packs={"audio": "p1"},
            source_paths=["audio/a.wav", "podcasts/b.wav", "incoming/x/source"],
        )
        assert report.ok
        assert report.nested == []
        assert report.pack_divergences == []

    def test_nested_shadow_is_flagged_and_blocks(self) -> None:
        # 'audio/pods' has no pack; a file there resolves to 'p1' today but to the
        # deeper packless folder after the cutover.
        report = build_preflight_report(
            folders=["audio", "audio/pods"],
            folder_domain_packs={"audio": "p1"},
            source_paths=["audio/pods/ep.wav"],
        )
        assert not report.ok
        assert report.nested == [("audio", "audio/pods")]
        assert len(report.pack_divergences) == 1
        div = report.pack_divergences[0]
        assert div.source_path == "audio/pods/ep.wav"
        assert div.old_pack == "p1"
        assert div.new_pack is None

    def test_nested_with_own_pack_preserves_resolution(self) -> None:
        # The child carries its own pack, so effective resolution is unchanged even
        # though the folders nest. Nesting alone still blocks (it is refused going
        # forward), but there is no pack divergence.
        report = build_preflight_report(
            folders=["audio", "audio/pods"],
            folder_domain_packs={"audio": "p1", "audio/pods": "p1"},
            source_paths=["audio/pods/ep.wav"],
        )
        assert report.pack_divergences == []
        assert report.nested == [("audio", "audio/pods")]
        assert not report.ok  # nesting itself is a blocking ambiguity

    def test_orphan_pack_key_flagged(self) -> None:
        report = build_preflight_report(
            folders=["audio"],
            folder_domain_packs={"audio": "p1", "video": "p2"},
            source_paths=[],
        )
        assert report.orphan_pack_keys == ["video"]
        assert not report.ok

    def test_missing_dir_is_non_blocking(self) -> None:
        report = build_preflight_report(
            folders=["audio", "gone"],
            folder_domain_packs={},
            source_paths=["audio/a.wav"],
            missing_dirs=["gone"],
        )
        assert report.missing_dirs == ["gone"]
        assert report.ok  # a vanished directory does not block the cutover
