import json
from pathlib import Path

import pytest
import yaml

from voxint.config import Settings
from voxint.domain_packs.base import DomainPack, DomainPackError, load_default
from voxint.domain_packs.corrections import (
    MAX_CORRECTIONS_MANIFEST_BYTES,
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    CorrectionRule,
    find_first,
    iter_matches,
    validate_corrections,
)
from voxint.domain_packs.registry import (
    available_domain_packs,
    default_domain_pack,
    resolve_domain_pack_by_name,
    resolve_folder_pack_name,
    resolve_run_domain_pack,
)


def _write_pack(root: Path, name: str, **fields: object) -> Path:
    pack_dir = root / name
    pack_dir.mkdir(parents=True)
    manifest = {"name": name, **fields}
    (pack_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return pack_dir


def test_generic_pack_loads() -> None:
    pack = load_default()
    assert pack.name == "generic"
    assert pack.vocabulary == ()
    assert "enhancement_context" in pack.prompt_fragments


# --- serialization round-trip (the per-run snapshot contract) ----------------


def test_to_mapping_from_mapping_round_trips() -> None:
    pack = DomainPack(
        name="interviews",
        description="Investigative interviews",
        vocabulary=("subpoena", "deposition"),
        name_seeds=("Jane Doe",),
        prompt_fragments={"enhancement_context": "Formal register.", "summary_context": "Neutral."},
    )
    restored = DomainPack.from_mapping(pack.to_mapping())
    assert restored == pack


def test_to_mapping_is_json_safe_types() -> None:
    pack = load_default()
    data = pack.to_mapping()
    assert isinstance(data["vocabulary"], list)
    assert isinstance(data["name_seeds"], list)
    assert isinstance(data["prompt_fragments"], dict)


def test_to_mapping_does_not_alias_live_dict() -> None:
    pack = DomainPack(name="p", prompt_fragments={"k": "v"})
    data = pack.to_mapping()
    data["prompt_fragments"]["k"] = "mutated"
    assert pack.prompt_fragments["k"] == "v"


@pytest.mark.parametrize(
    "bad",
    [
        {},  # no name
        {"name": ""},  # blank name
        {"name": "  "},  # whitespace-only name
        {"name": 3},  # non-string name
        {"name": "p", "description": 7},  # non-string description
        {"name": "p", "vocabulary": "notalist"},  # vocabulary not a list
        {"name": "p", "vocabulary": [1, 2]},  # vocabulary entries not strings
        {"name": "p", "name_seeds": {"a": "b"}},  # name_seeds not a list
        {"name": "p", "prompt_fragments": ["x"]},  # fragments not a mapping
        {"name": "p", "prompt_fragments": {"k": 5}},  # fragment value not a string
    ],
)
def test_from_mapping_rejects_corrupt_snapshots(bad: dict[str, object]) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping(bad)


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.load(tmp_path / "nope")


def test_load_malformed_yaml_raises(tmp_path: Path) -> None:
    pack_dir = tmp_path / "broken"
    pack_dir.mkdir()
    (pack_dir / "manifest.yaml").write_text("name: [unclosed")
    with pytest.raises(DomainPackError):
        DomainPack.load(pack_dir)


# --- registry resolution -----------------------------------------------------


def test_default_pack_is_generic_when_unconfigured() -> None:
    settings = Settings(_env_file=None)
    assert default_domain_pack(settings).name == "generic"


def test_default_pack_honors_domain_pack_path(tmp_path: Path) -> None:
    _write_pack(tmp_path, "custom", vocabulary=["widget"])
    settings = Settings(_env_file=None, domain_pack_path=tmp_path / "custom")
    pack = default_domain_pack(settings)
    assert pack.name == "custom"
    assert pack.vocabulary == ("widget",)


def test_available_packs_includes_generic_by_default() -> None:
    settings = Settings(_env_file=None)
    packs = available_domain_packs(settings)
    assert set(packs) == {"generic"}


def test_available_packs_scans_packs_dir(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["cold open"])
    _write_pack(tmp_path, "interview")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    packs = available_domain_packs(settings)
    assert set(packs) == {"generic", "podcast", "interview"}
    assert packs["podcast"].vocabulary == ("cold open",)


def test_available_packs_missing_dir_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path / "absent")
    with pytest.raises(DomainPackError):
        available_domain_packs(settings)


def test_available_packs_duplicate_name_conflict_raises(tmp_path: Path) -> None:
    _write_pack(tmp_path, "dup", vocabulary=["a"])
    # A second pack folder whose manifest also claims name "dup" but differs.
    other = tmp_path / "dup2"
    other.mkdir()
    (other / "manifest.yaml").write_text(yaml.safe_dump({"name": "dup", "vocabulary": ["b"]}))
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        available_domain_packs(settings)


def test_available_packs_same_pack_two_sources_is_idempotent(tmp_path: Path) -> None:
    # DOMAIN_PACK_PATH points at a pack that ALSO sits under DOMAIN_PACKS_DIR.
    _write_pack(tmp_path, "shared", vocabulary=["x"])
    settings = Settings(
        _env_file=None,
        domain_pack_path=tmp_path / "shared",
        domain_packs_dir=tmp_path,
    )
    packs = available_domain_packs(settings)
    assert packs["shared"].vocabulary == ("x",)


def test_resolve_by_name_returns_pack(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    assert resolve_domain_pack_by_name("podcast", settings).name == "podcast"


def test_resolve_by_name_unknown_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        resolve_domain_pack_by_name("nonesuch", settings)


# --- folder → pack name resolution (pure) ------------------------------------


def test_folder_match_returns_mapped_name() -> None:
    mapping = {"podcasts": "podcast", "interviews": "interview"}
    assert resolve_folder_pack_name("podcasts/ep1.wav", mapping) == "podcast"
    assert resolve_folder_pack_name("interviews/subdir/a.wav", mapping) == "interview"


def test_folder_unmapped_returns_none() -> None:
    assert resolve_folder_pack_name("misc/x.wav", {"podcasts": "podcast"}) is None


def test_folder_empty_mapping_returns_none() -> None:
    assert resolve_folder_pack_name("podcasts/ep1.wav", {}) is None


def test_folder_longest_ancestor_wins() -> None:
    mapping = {"audio": "general", "audio/interviews": "interview"}
    assert resolve_folder_pack_name("audio/interviews/a.wav", mapping) == "interview"
    assert resolve_folder_pack_name("audio/other/a.wav", mapping) == "general"


def test_folder_matches_on_components_not_string_prefix() -> None:
    # "pod" must NOT match a file under "podcasts" — component-wise ancestry only.
    mapping = {"pod": "wrong"}
    assert resolve_folder_pack_name("podcasts/ep1.wav", mapping) is None


def test_folder_key_equal_to_parent_dir_matches() -> None:
    assert resolve_folder_pack_name("podcasts/ep1.wav", {"podcasts": "podcast"}) == "podcast"


# --- resolve_run_domain_pack (snapshot selection precedence) -----------------


def test_run_pack_explicit_name_wins(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast")
    _write_pack(tmp_path, "interview")
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
        explicit_name="interview",
    )
    assert snap["name"] == "interview"


def test_run_pack_folder_mapping_used(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["cold open"])
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
    )
    assert snap["name"] == "podcast"
    assert snap["vocabulary"] == ["cold open"]


def test_run_pack_unmapped_folder_uses_default() -> None:
    settings = Settings(_env_file=None)
    snap = resolve_run_domain_pack(
        "misc/x.wav", settings=settings, folder_domain_packs={"podcasts": "podcast"}
    )
    assert snap["name"] == "generic"


def test_run_pack_none_source_uses_default() -> None:
    settings = Settings(_env_file=None)
    snap = resolve_run_domain_pack(
        None, settings=settings, folder_domain_packs={"podcasts": "podcast"}
    )
    assert snap["name"] == "generic"


def test_run_pack_mapped_name_unresolvable_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    with pytest.raises(DomainPackError):
        resolve_run_domain_pack(
            "podcasts/ep1.wav",
            settings=settings,
            folder_domain_packs={"podcasts": "ghost"},
        )


def test_run_pack_snapshot_round_trips_to_pack(tmp_path: Path) -> None:
    _write_pack(tmp_path, "podcast", vocabulary=["a"], name_seeds=["Jane"])
    settings = Settings(_env_file=None, domain_packs_dir=tmp_path)
    snap = resolve_run_domain_pack(
        "podcasts/ep1.wav",
        settings=settings,
        folder_domain_packs={"podcasts": "podcast"},
    )
    restored = DomainPack.from_mapping(snap)
    assert restored.name == "podcast"
    assert restored.name_seeds == ("Jane",)


# --- corrections (issue #80) -------------------------------------------------
#
# The corrections: field declares deterministic literal-substitution rules,
# frozen per run. #80 owns the rule type, all validation, and the single-rule
# matcher; the multi-rule apply engine is #81.


def test_generic_pack_has_no_corrections() -> None:
    # generic must stay a byte-preserving no-op (declares no corrections).
    assert load_default().corrections == ()


def test_corrections_round_trip_with_defaults() -> None:
    pack = DomainPack(
        name="newsroom",
        corrections=(
            CorrectionRule("zoning-board", "zoom board", "Zoning Board"),
            CorrectionRule("cdbg", "C D B G", "CDBG", case_sensitive=False, whole_word=False),
        ),
    )
    restored = DomainPack.from_mapping(pack.to_mapping())
    assert restored == pack
    # Defaults resolved on the first rule (constructed without the flags).
    assert restored.corrections[0].case_sensitive is True
    assert restored.corrections[0].whole_word is True


def test_corrections_defaults_applied_on_read() -> None:
    pack = DomainPack.from_mapping(
        {"name": "p", "corrections": [{"id": "a", "match": "x", "replace": "y"}]}
    )
    assert pack.corrections[0].case_sensitive is True
    assert pack.corrections[0].whole_word is True


def test_corrections_snapshot_emits_explicit_bools() -> None:
    pack = DomainPack(name="p", corrections=(CorrectionRule("a", "x", "y"),))
    snap_rule = pack.to_mapping()["corrections"][0]
    assert snap_rule == {
        "id": "a",
        "match": "x",
        "replace": "y",
        "case_sensitive": True,
        "whole_word": True,
    }


def test_corrections_snapshot_is_json_safe_list() -> None:
    pack = DomainPack(name="p", corrections=(CorrectionRule("a", "x", "y"),))
    data = pack.to_mapping()
    assert isinstance(data["corrections"], list)
    assert isinstance(data["corrections"][0], dict)


def test_pack_without_corrections_key_round_trips() -> None:
    # A legacy snapshot (pre-#80) has no corrections key and must restore to ().
    pack = DomainPack.from_mapping({"name": "p", "vocabulary": ["a"]})
    assert pack.corrections == ()


@pytest.mark.parametrize(
    "bad_corrections",
    [
        "notalist",  # corrections not a list
        [["not", "a", "mapping"]],  # a rule entry is not a mapping
        [{"id": "a", "match": "x", "replace": "y", "extra": 1}],  # unknown key
        [{"match": "x", "replace": "y"}],  # missing id
        [{"id": "a", "replace": "y"}],  # missing match
        [{"id": "a", "match": "x"}],  # missing replace
        [{"id": 5, "match": "x", "replace": "y"}],  # non-string id
        [{"id": "a", "match": 5, "replace": "y"}],  # non-string match
        [{"id": "a", "match": "x", "replace": 5}],  # non-string replace
        [{"id": "  ", "match": "x", "replace": "y"}],  # whitespace-only id
        [{"id": "a", "match": "", "replace": "y"}],  # empty match
        [{"id": "a", "match": "x", "replace": ""}],  # empty replace
        [{"id": "a\x00b", "match": "x", "replace": "y"}],  # NUL in id
        [{"id": "a", "match": "x\x00", "replace": "y"}],  # NUL in match
        [{"id": "a", "match": "x", "replace": "y\x00"}],  # NUL in replace
        [{"id": "a", "match": "x", "replace": "y", "case_sensitive": 1}],  # int flag
        [{"id": "a", "match": "x", "replace": "y", "whole_word": "yes"}],  # str flag
        [  # duplicate id
            {"id": "dup", "match": "x", "replace": "y"},
            {"id": "dup", "match": "p", "replace": "q"},
        ],
    ],
)
def test_from_mapping_rejects_corrupt_corrections(bad_corrections: object) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": bad_corrections})


def test_corrections_reject_too_many_rules() -> None:
    rules = [
        {"id": f"r{i}", "match": f"m{i}", "replace": f"v{i}"}
        for i in range(MAX_RULES_PER_PACK + 1)
    ]
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": rules})


def test_validate_corrections_count_bound_holds_for_direct_callers() -> None:
    # parse_corrections bounds the count early, so validate_corrections' own count
    # check only fires for a direct caller (e.g. the #81 engine) — keep it robust.
    rules = tuple(
        CorrectionRule(f"r{i}", f"m{i}", f"v{i}") for i in range(MAX_RULES_PER_PACK + 1)
    )
    with pytest.raises(DomainPackError):
        validate_corrections(rules)


def test_corrections_reject_overlong_match() -> None:
    rule = {"id": "a", "match": "x" * (MAX_MATCH_CHARS + 1), "replace": "y"}
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": [rule]})


def test_corrections_reject_overlong_replacement() -> None:
    rule = {"id": "a", "match": "x", "replace": "y" * (MAX_REPLACEMENT_CHARS + 1)}
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": [rule]})


def test_corrections_reject_oversize_manifest() -> None:
    # Many rules that each pass the per-field bounds, no cross-replacement match,
    # whose canonical JSON exceeds the byte cap. ~200-char unique payloads * 256
    # rules ≈ 60 KB; widen match/replace to clear 128 KiB while staying ≤ the
    # per-field limits.
    rules = [
        {
            "id": f"r{i}",
            "match": f"m{i}-" + "q" * 240,
            "replace": f"v{i}-" + "w" * 240,
        }
        for i in range(MAX_RULES_PER_PACK)
    ]
    encoded_len = len(
        json.dumps(
            [{**r, "case_sensitive": True, "whole_word": True} for r in rules],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert encoded_len > MAX_CORRECTIONS_MANIFEST_BYTES  # guard the fixture itself
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": rules})


# --- idempotence (boundary-aware, R-own-flags) -------------------------------


def test_corrections_reject_chain_not_idempotent() -> None:
    # [a→b, b→c]: applying twice would cascade a→b→c. Rejected at load.
    rules = [
        {"id": "a", "match": "a", "replace": "b"},
        {"id": "b", "match": "b", "replace": "c"},
    ]
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": rules})


def test_corrections_reject_self_chain_when_not_whole_word() -> None:
    # aa→aaa with whole_word=False: "aa" re-fires inside "aaa". Rejected.
    rule = {"id": "s", "match": "aa", "replace": "aaa", "whole_word": False}
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": [rule]})


def test_corrections_accept_self_chain_when_whole_word() -> None:
    # aa→aaa with whole_word=True (the default): "aa" is not a whole-word match
    # inside "aaa", so it never re-fires — genuinely idempotent, accepted.
    pack = DomainPack.from_mapping(
        {"name": "p", "corrections": [{"id": "s", "match": "aa", "replace": "aaa"}]}
    )
    assert pack.corrections[0].match == "aa"


def test_corrections_accept_whole_word_substring_replacement() -> None:
    # cat→category: "cat" is a substring of "category" but not a whole-word match,
    # so it cannot re-fire — accepted under the default whole_word=True.
    pack = DomainPack.from_mapping(
        {"name": "p", "corrections": [{"id": "c", "match": "cat", "replace": "category"}]}
    )
    assert len(pack.corrections) == 1


def test_corrections_reject_case_insensitive_chain() -> None:
    # A case-insensitive rule whose match appears in another case inside another
    # rule's replacement re-fires (case-folded) and is rejected. The lowercase
    # "ab" is a standalone word in the replacement, so the default whole-word
    # firer matches it under IGNORECASE.
    rules = [
        {"id": "ab", "match": "AB", "replace": "Z", "case_sensitive": False},
        {"id": "other", "match": "q", "replace": "say ab now"},
    ]
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": rules})


def test_corrections_accept_real_disjoint_set() -> None:
    rules = [
        {"id": "zoning-board", "match": "zoom board", "replace": "Zoning Board"},
        {"id": "cdbg", "match": "C D B G", "replace": "CDBG"},
    ]
    pack = DomainPack.from_mapping({"name": "p", "corrections": rules})
    assert [r.id for r in pack.corrections] == ["zoning-board", "cdbg"]


# --- single-rule matcher ------------------------------------------------------


def test_matcher_possessive_is_intra_word() -> None:
    rule = CorrectionRule("i", "it", "IT")
    assert find_first(rule, "it's here") is None  # apostrophe joins the word
    assert find_first(rule, "it is") == (0, 2)  # standalone fires


def test_matcher_hyphen_is_intra_word() -> None:
    assert find_first(CorrectionRule("c", "co", "CO"), "co-op") is None


def test_matcher_nfd_combining_mark_not_split() -> None:
    # "Zoë" decomposed = Z o e U+0308: "Zoe" must not match (combining mark after).
    decomposed = "Zoë"
    assert find_first(CorrectionRule("z", "Zoe", "ZOE"), decomposed) is None
    assert find_first(CorrectionRule("z", "Zoe", "ZOE"), "Zoe said") == (0, 3)


def test_matcher_regex_metachars_are_literal() -> None:
    rule = CorrectionRule("c", "C.D.B.G.", "CDBG")
    assert find_first(rule, "the C.D.B.G. grant") == (4, 12)
    assert find_first(rule, "CXDXBXGX here") is None  # '.' is not a wildcard


def test_matcher_case_insensitive_finds_uppercase() -> None:
    rule = CorrectionRule("s", "selectboard", "Selectboard", case_sensitive=False)
    assert find_first(rule, "the SELECTBOARD met") == (4, 15)


def test_matcher_collision_skips_substring_and_finds_standalone() -> None:
    rule = CorrectionRule("c", "cat", "CAT")
    assert find_first(rule, "catalog") is None  # substring, not whole word
    assert find_first(rule, "the cat sat") == (4, 7)
    # The boundary-invalid first occurrence (in "catalog") is skipped; the
    # standalone "cat" is found.
    assert list(iter_matches(rule, "catalog cat")) == [(8, 11)]


# --- matcher/guard hardening (code review) -----------------------------------


def test_matcher_boundary_invalid_does_not_hide_overlapping_valid() -> None:
    # A boundary-invalid candidate must not consume a later OVERLAPPING valid one.
    # "ha ha" in "aha ha ha": (1,6) is invalid (before='a'); (4,9) is valid.
    rule = CorrectionRule("h", "ha ha", "X")
    assert find_first(rule, "aha ha ha") == (4, 9)
    # Self-overlapping doubled word: "a a" in "xa a a ".
    assert find_first(CorrectionRule("a", "a a", "X"), "xa a a ") == (3, 6)


def test_corrections_guard_rejects_hidden_refire() -> None:
    # The overlap fix tightens the idempotence guard: a firer whose match is only
    # reachable via an overlapping occurrence inside a replacement is now rejected.
    rules = [
        {"id": "f", "match": "ha ha", "replace": "X"},
        {"id": "t", "match": "q", "replace": "aha ha ha"},
    ]
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": rules})


def test_matcher_ignorecase_folds_are_one_to_one() -> None:
    # The offset-stable design relies on re.IGNORECASE being length-preserving.
    # Kelvin sign (U+212A) folds 1:1 to 'k'; 'ß' must NOT match "ss".
    assert find_first(
        CorrectionRule("k", "k", "K", case_sensitive=False), "\u212a"
    ) == (0, 1)
    assert (
        find_first(CorrectionRule("s", "ss", "S", case_sensitive=False), "ß")
        is None
    )


def test_matcher_non_bmp_char_is_a_boundary() -> None:
    # A non-BMP emoji is a boundary character (not alphanumeric/intra-word/mark).
    assert find_first(CorrectionRule("c", "cat", "C"), "\U0001f642cat") == (1, 4)


@pytest.mark.parametrize(
    "bad_corrections",
    [
        [{"id": "a", "match": "x​", "replace": "y"}],  # zero-width space (Cf)
        [{"id": "a", "match": "x", "replace": "y﻿"}],  # BOM (Cf) in replace
        [{"id": "a", "match": "zoom board\n", "replace": "Y"}],  # trailing newline (Cc)
        [{"id": "a", "match": "   ", "replace": "y"}],  # whitespace-only match
        [{"id": "a", "match": "x", "replace": "  "}],  # whitespace-only replace
        [{"id": " a ", "match": "x", "replace": "y"}],  # id with surrounding space
        [{"id": "a", "match": "x", "replace": "y", 1: "q", "zzz": "w"}],  # mixed keys
    ],
)
def test_from_mapping_rejects_hardened_corrections(bad_corrections: object) -> None:
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping({"name": "p", "corrections": bad_corrections})


def test_corrections_reject_lone_surrogate() -> None:
    # A lone surrogate (e.g. from a tampered JSON snapshot) must raise
    # DomainPackError, never a raw UnicodeEncodeError that escapes the degrade path.
    surrogate = json.loads('"\\ud800"')
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping(
            {"name": "p", "corrections": [{"id": "a", "match": surrogate, "replace": "y"}]}
        )


def test_corrections_allow_combining_mark_and_curly_apostrophe() -> None:
    # Combining marks (Mn) and the curly apostrophe (Po) are NOT non-printing and
    # must round-trip (they are load-bearing for the boundary predicate).
    pack = DomainPack.from_mapping(
        {
            "name": "p",
            "corrections": [
                {"id": "z", "match": "Zoe", "replace": "Zoë"},
                {"id": "i", "match": "it's", "replace": "IT\u2019S"},
            ],
        }
    )
    assert pack.corrections[0].replace == "Zoë"


@pytest.mark.parametrize("bad", [["not", "a", "mapping"], "a string", 5, 3.5])
def test_from_mapping_rejects_non_mapping(bad: object) -> None:
    # The single validation point rejects a tampered non-mapping snapshot as
    # DomainPackError, not a raw AttributeError.
    with pytest.raises(DomainPackError):
        DomainPack.from_mapping(bad)  # type: ignore[arg-type]


# --- union_pack_name_seeds (issue #104) ----------------------------------------


def test_union_name_seeds_appends_after_pack_seeds() -> None:
    from voxint.domain_packs.base import union_pack_name_seeds

    snapshot = {"name": "p", "name_seeds": ["Alice", "Bob"]}
    merged = union_pack_name_seeds(snapshot, ["Carol", "Bob", "Dan"])
    assert merged["name_seeds"] == ["Alice", "Bob", "Carol", "Dan"]
    # The input snapshot is never mutated.
    assert snapshot["name_seeds"] == ["Alice", "Bob"]
    assert merged is not snapshot


def test_union_name_seeds_empty_inputs() -> None:
    from voxint.domain_packs.base import union_pack_name_seeds

    assert union_pack_name_seeds({"name": "p"}, [])["name_seeds"] == []
    assert union_pack_name_seeds({"name": "p"}, ["A"])["name_seeds"] == ["A"]


def test_union_name_seeds_exact_string_dedupe_only() -> None:
    from voxint.domain_packs.base import union_pack_name_seeds

    merged = union_pack_name_seeds({"name": "p", "name_seeds": ["alice"]}, ["Alice"])
    # Dedupe is exact-string: case variants are distinct entries.
    assert merged["name_seeds"] == ["alice", "Alice"]


def test_union_name_seeds_result_round_trips_from_mapping() -> None:
    from voxint.domain_packs.base import union_pack_name_seeds

    pack = load_default()
    merged = union_pack_name_seeds(pack.to_mapping(), ["Jane Doe"])
    restored = DomainPack.from_mapping(merged)
    assert "Jane Doe" in restored.name_seeds


def test_union_name_seeds_rejects_tampered_pack_seeds() -> None:
    from voxint.domain_packs.base import union_pack_name_seeds

    with pytest.raises(DomainPackError):
        union_pack_name_seeds({"name": "p", "name_seeds": [1, 2]}, ["A"])
