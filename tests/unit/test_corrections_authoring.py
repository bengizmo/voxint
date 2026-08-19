"""Console corrections-authoring seam (issue #84): validation, id auto-fill, union.

These are pure-function tests over the authoring helpers added to
``domain_packs.corrections``. The point of #84 is that authoring routes through
the SAME #80 gate a pack does — so every reject path here is the #80 validator
firing, only re-dressed with an operator-facing message and (for field faults) a
row index. The one genuinely new behavior is blank-id auto-generation.
"""

import pytest

from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import (
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    OperatorCorrectionError,
    normalize_operator_corrections,
    union_pack_corrections,
)


def test_normalize_applies_defaults_and_canonicalizes() -> None:
    result = normalize_operator_corrections(
        [{"id": "zb", "match": "zoom board", "replace": "Zoning Board"}]
    )
    assert result == [
        {
            "id": "zb",
            "match": "zoom board",
            "replace": "Zoning Board",
            # Both flags default True (the #80 conservative posture).
            "case_sensitive": True,
            "whole_word": True,
        }
    ]


def test_normalize_preserves_explicit_flags() -> None:
    result = normalize_operator_corrections(
        [
            {
                "id": "x",
                "match": "teh",
                "replace": "the",
                "case_sensitive": False,
                "whole_word": False,
            }
        ]
    )
    assert result[0]["case_sensitive"] is False
    assert result[0]["whole_word"] is False


def test_blank_id_is_generated_from_match() -> None:
    # Case-sensitive by default, so lowercase "zoning board" does not re-fire on the
    # capitalized replacement — the set stays idempotent.
    result = normalize_operator_corrections(
        [{"match": "zoning board", "replace": "Zoning Board"}]
    )
    assert result[0]["id"] == "zoning-board"


def test_missing_id_key_is_generated() -> None:
    # No "id" key at all (not just blank) is still auto-filled.
    result = normalize_operator_corrections([{"match": "CDBG", "replace": "C.D.B.G."}])
    assert result[0]["id"] == "cdbg"


def test_generated_ids_are_unique_for_identical_matches() -> None:
    result = normalize_operator_corrections(
        [
            {"match": "board", "replace": "Board"},
            {"match": "board", "replace": "the Board"},
        ]
    )
    ids = [rule["id"] for rule in result]
    assert ids == ["board", "board-2"]


def test_generated_id_avoids_collision_with_explicit_id() -> None:
    result = normalize_operator_corrections(
        [
            {"id": "board", "match": "planning", "replace": "Planning"},
            {"match": "board", "replace": "Board"},
        ]
    )
    assert result[1]["id"] == "board-2"


def test_generated_id_falls_back_when_match_has_no_slug_chars() -> None:
    result = normalize_operator_corrections([{"match": "!!!", "replace": "-> arrow"}])
    assert result[0]["id"] == "rule-1"


def test_non_mapping_list_entry_rejected_with_row() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections([{"id": "ok", "match": "a", "replace": "b"}, "nope"])
    assert excinfo.value.row == 1
    assert "must be a mapping" in excinfo.value.message


def test_missing_match_is_auto_id_then_rejected_for_missing_match() -> None:
    # id auto-fill runs first: a rule with neither id nor match gets a fallback id,
    # then fails the required-match field check (row set).
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections([{"replace": "x"}])
    assert excinfo.value.row == 0
    assert "match" in excinfo.value.message


def test_non_list_input_raises_whole_list_error() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections({"match": "a", "replace": "b"})
    assert excinfo.value.row is None
    assert "list of rules" in excinfo.value.message


def test_empty_field_raises_with_row_and_plain_message() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [
                {"id": "ok", "match": "a", "replace": "b"},
                {"id": "bad", "match": "  ", "replace": "x"},
            ]
        )
    err = excinfo.value
    assert err.row == 1
    # Operator-facing: the pack framing is stripped, the substance preserved.
    assert "domain pack" not in err.message
    assert err.message.startswith("Rule 2 ")
    assert "non-empty" in err.message


def test_nul_control_char_rejected_with_row() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections([{"id": "z", "match": "a\x00b", "replace": "c"}])
    assert excinfo.value.row == 0
    assert "non-printing" in excinfo.value.message


def test_match_too_long_rejected_with_row() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [{"id": "z", "match": "a" * (MAX_MATCH_CHARS + 1), "replace": "b"}]
        )
    # A length fault is a cross-rule (validate_corrections) check: row is None but
    # the message still names the offending rule so the operator can find it.
    assert "domain pack" not in excinfo.value.message
    assert str(MAX_MATCH_CHARS) in excinfo.value.message


def test_replacement_too_long_rejected() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [{"id": "z", "match": "a", "replace": "b" * (MAX_REPLACEMENT_CHARS + 1)}]
        )
    assert str(MAX_REPLACEMENT_CHARS) in excinfo.value.message


def test_too_many_rules_rejected() -> None:
    many = [
        {"id": f"r{i}", "match": f"m{i}", "replace": f"v{i}"}
        for i in range(MAX_RULES_PER_PACK + 1)
    ]
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(many)
    assert str(MAX_RULES_PER_PACK) in excinfo.value.message


def test_duplicate_explicit_id_rejected() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [
                {"id": "dup", "match": "a", "replace": "b"},
                {"id": "dup", "match": "c", "replace": "d"},
            ]
        )
    assert excinfo.value.row is None
    assert "duplicate id" in excinfo.value.message
    assert "domain pack" not in excinfo.value.message


def test_non_idempotent_rule_rejected() -> None:
    # "dog" fires as a whole word inside its own replacement "dog food".
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections([{"id": "d", "match": "dog", "replace": "dog food"}])
    assert "idempotent" in excinfo.value.message
    assert "domain pack" not in excinfo.value.message


def test_unknown_key_rejected_with_row() -> None:
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [{"id": "z", "match": "a", "replace": "b", "regex": True}]
        )
    assert excinfo.value.row == 0
    assert "unknown keys" in excinfo.value.message


def test_pack_union_duplicate_id_rejected_at_author_time() -> None:
    pack = [{"id": "shared", "match": "x", "replace": "y"}]
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [{"id": "shared", "match": "a", "replace": "b"}], pack_corrections=pack
        )
    assert excinfo.value.row is None
    assert "duplicate id" in excinfo.value.message


def test_pack_union_idempotence_rejected_at_author_time() -> None:
    # Pack rule replaces to "Zoning Board"; the operator rule "Board" would re-fire
    # on that replacement, breaking idempotence of the UNION.
    pack = [{"id": "zb", "match": "zoom board", "replace": "Zoning Board"}]
    with pytest.raises(OperatorCorrectionError) as excinfo:
        normalize_operator_corrections(
            [{"id": "b", "match": "Board", "replace": "Committee"}],
            pack_corrections=pack,
        )
    assert "idempotent" in excinfo.value.message


def test_pack_union_clean_passes() -> None:
    pack = [{"id": "zb", "match": "zoom board", "replace": "Zoning Board"}]
    result = normalize_operator_corrections(
        [{"id": "teh", "match": "teh", "replace": "the"}], pack_corrections=pack
    )
    # The operator list is returned (NOT the pack rules) — storage holds only the
    # operator's own rules; the union is re-formed per run at freeze time.
    assert [rule["id"] for rule in result] == ["teh"]


def test_empty_pack_corrections_is_noop() -> None:
    result = normalize_operator_corrections(
        [{"id": "teh", "match": "teh", "replace": "the"}], pack_corrections=[]
    )
    assert result[0]["id"] == "teh"


# --- union_pack_corrections --------------------------------------------------


def test_union_appends_operator_after_pack() -> None:
    pack_mapping = {
        "name": "generic",
        "corrections": [{"id": "p", "match": "x", "replace": "y"}],
    }
    operator = [
        {"id": "o", "match": "teh", "replace": "the", "case_sensitive": True, "whole_word": True}
    ]
    merged = union_pack_corrections(pack_mapping, operator)
    assert [rule["id"] for rule in merged["corrections"]] == ["p", "o"]
    # The pack's other fields survive.
    assert merged["name"] == "generic"


def test_union_does_not_mutate_input() -> None:
    pack_mapping = {"name": "generic", "corrections": [{"id": "p", "match": "x", "replace": "y"}]}
    original = [dict(rule) for rule in pack_mapping["corrections"]]
    union_pack_corrections(pack_mapping, [{"id": "o", "match": "a", "replace": "b"}])
    assert pack_mapping["corrections"] == original


def test_union_no_operator_rules_returns_fresh_mapping() -> None:
    pack_mapping = {"name": "generic", "corrections": [{"id": "p", "match": "x", "replace": "y"}]}
    merged = union_pack_corrections(pack_mapping, [])
    assert merged is not pack_mapping
    assert [rule["id"] for rule in merged["corrections"]] == ["p"]


def test_union_duplicate_id_raises_domain_pack_error() -> None:
    pack_mapping = {"name": "generic", "corrections": [{"id": "dup", "match": "x", "replace": "y"}]}
    with pytest.raises(DomainPackError):
        union_pack_corrections(pack_mapping, [{"id": "dup", "match": "a", "replace": "b"}])


def test_union_idempotence_across_pack_and_operator_raises() -> None:
    pack_mapping = {
        "name": "generic",
        "corrections": [{"id": "zb", "match": "zoom board", "replace": "Zoning Board"}],
    }
    operator = [{"id": "b", "match": "Board", "replace": "Committee"}]
    with pytest.raises(DomainPackError):
        union_pack_corrections(pack_mapping, operator)


def test_union_pack_without_corrections_key() -> None:
    merged = union_pack_corrections(
        {"name": "generic"}, [{"id": "o", "match": "a", "replace": "b"}]
    )
    assert [rule["id"] for rule in merged["corrections"]] == ["o"]
