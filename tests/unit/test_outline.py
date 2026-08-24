"""Pure navigable-outline resolution (issue #87).

Exercises ``voxint.enrichment.outline.resolve_outline`` with no database: feed
in-memory mention/summary/topics payloads plus a ``{segment_index:
start_seconds}`` map and assert the group/dedup/order/drop table and the honest
empty/stale states. ``build_outline`` (the session read) is covered by the
integration suite.
"""

from __future__ import annotations

from typing import Any

from voxint.enrichment.outline import (
    Outline,
    OutlineContext,
    resolve_outline,
)


def _mentions(*mentions: dict[str, Any], **diagnostics: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"mentions": list(mentions)}
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _mention(surface: str, kind: str | None, *occ: tuple[int, str, int]) -> dict[str, Any]:
    return {
        "surface": surface,
        "kind": kind,
        "occurrences": [
            {
                "segment_index": index,
                "quote": quote,
                "start_char": start_char,
                "end_char": start_char + len(quote),
            }
            for index, quote, start_char in occ
        ],
    }


STARTS = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0}


def test_absent_asset_is_unavailable() -> None:
    out = resolve_outline(None, None, None, STARTS, asset_stale=False, gated=False)
    assert out.available is False
    assert out.gated is False
    assert out.asset_stale is False
    assert out.mentions == ()


def test_gated_flag_propagates_when_absent() -> None:
    out = resolve_outline(None, None, None, STARTS, asset_stale=False, gated=True)
    assert out.available is False
    assert out.gated is True


def test_empty_mentions_is_available_but_empty() -> None:
    out = resolve_outline(_mentions(), None, None, STARTS, asset_stale=False, gated=False)
    assert out.available is True
    assert out.mentions == ()


def test_asset_stale_flag_propagates() -> None:
    payload = _mentions(_mention("Acme", "organization", (1, "Acme", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=True, gated=False)
    assert out.asset_stale is True
    assert out.available is True
    # Stale does not suppress the target: the timestamp is still grounded truth.
    assert out.mentions[0].occurrences[0].start_seconds == 10.0


def test_grounded_occurrence_resolves_to_start_seconds() -> None:
    payload = _mentions(_mention("Jane Doe", "person", (2, "Jane Doe", 5)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions) == 1
    mention = out.mentions[0]
    assert mention.surface == "Jane Doe"
    assert mention.kind == "person"
    assert mention.occurrences[0].segment_index == 2
    assert mention.occurrences[0].start_seconds == 20.0
    assert mention.occurrences[0].quote == "Jane Doe"


def test_unresolved_segment_index_is_dropped_and_counted() -> None:
    # segment 9 is not in STARTS -> the occurrence is dropped, not seeked to null.
    payload = _mentions(_mention("Ghost", "person", (9, "Ghost", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert out.mentions == ()
    assert out.diagnostics.dropped_unresolved == 1


def test_mention_with_some_unresolved_keeps_the_resolvable_ones() -> None:
    payload = _mentions(_mention("Acme", "organization", (1, "Acme", 0), (9, "Acme", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions) == 1
    assert [o.segment_index for o in out.mentions[0].occurrences] == [1]
    assert out.diagnostics.dropped_unresolved == 1


def test_kind_null_is_preserved_never_invented() -> None:
    payload = _mentions(_mention("Some Thing", None, (0, "Some Thing", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert out.mentions[0].kind is None


def test_generator_diagnostics_are_carried() -> None:
    payload = _mentions(
        _mention("Acme", "organization", (0, "Acme", 0)),
        dropped_unlocatable=3,
        dropped_out_of_run=2,
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert out.diagnostics.dropped_unlocatable == 3
    assert out.diagnostics.dropped_out_of_run == 2
    assert out.diagnostics.dropped_unresolved == 0


def test_person_and_org_same_surface_are_distinct_groups() -> None:
    payload = _mentions(
        _mention("Acme", "person", (0, "Acme", 0)),
        _mention("Acme", "organization", (1, "Acme", 0)),
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions) == 2
    kinds = {m.kind for m in out.mentions}
    assert kinds == {"person", "organization"}


def test_duplicate_occurrences_are_deduped() -> None:
    # Same (segment_index, start_char) twice -> one occurrence.
    payload = _mentions(_mention("Acme", "organization", (1, "Acme", 0), (1, "Acme", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions[0].occurrences) == 1


def test_same_segment_different_start_char_are_both_kept() -> None:
    payload = _mentions(_mention("Acme", "organization", (1, "Acme", 0), (1, "Acme", 12)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions[0].occurrences) == 2


def test_groups_ordered_by_first_occurrence_time() -> None:
    payload = _mentions(
        _mention("Later", "person", (3, "Later", 0)),
        _mention("Earlier", "person", (0, "Earlier", 0)),
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert [m.surface for m in out.mentions] == ["Earlier", "Later"]


def test_occurrences_ordered_by_time_within_a_mention() -> None:
    payload = _mentions(
        _mention("Acme", "organization", (3, "Acme", 0), (0, "Acme", 0), (1, "Acme", 0))
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert [o.start_seconds for o in out.mentions[0].occurrences] == [0.0, 10.0, 30.0]


def test_duplicate_group_keys_merge_occurrences() -> None:
    payload = _mentions(
        _mention("Acme", "organization", (0, "Acme", 0)),
        _mention("Acme", "organization", (2, "Acme", 0)),
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert len(out.mentions) == 1
    assert [o.segment_index for o in out.mentions[0].occurrences] == [0, 2]


def test_context_summary_and_topic_labels_only() -> None:
    summary = {"summary": "A short recap."}
    topics = {"topics": [{"label": "Onboarding", "confidence": 0.9}, {"label": "Billing"}]}
    out = resolve_outline(None, summary, topics, STARTS, asset_stale=False, gated=False)
    assert out.context == OutlineContext(summary="A short recap.", topics=("Onboarding", "Billing"))


def test_blank_summary_becomes_none() -> None:
    out = resolve_outline(None, {"summary": "   "}, None, STARTS, asset_stale=False, gated=False)
    assert out.context.summary is None


def test_bool_segment_index_is_not_treated_as_int() -> None:
    payload = _mentions(
        {
            "surface": "Acme",
            "kind": "organization",
            "occurrences": [
                {"segment_index": True, "quote": "Acme", "start_char": 0, "end_char": 4}
            ],
        }
    )
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert out.mentions == ()
    assert out.diagnostics.dropped_unresolved == 0  # skipped, not counted as a segment


def test_to_props_shape() -> None:
    payload = _mentions(
        _mention("Acme", "organization", (1, "Acme signed", 0)),
        dropped_unlocatable=1,
    )
    summary = {"summary": "Recap."}
    topics = {"topics": [{"label": "Billing"}]}
    out = resolve_outline(payload, summary, topics, STARTS, asset_stale=True, gated=False)
    props = out.to_props()
    assert props == {
        "available": True,
        "gated": False,
        "assetStale": True,
        "mentions": [
            {
                "surface": "Acme",
                "kind": "organization",
                "occurrences": [{"segmentIndex": 1, "startSeconds": 10.0, "quote": "Acme signed"}],
            }
        ],
        "context": {"summary": "Recap.", "topics": ["Billing"]},
        "diagnostics": {"droppedUnlocatable": 1, "droppedOutOfRun": 0, "droppedUnresolved": 0},
    }


def test_malformed_mentions_payload_is_total() -> None:
    # Garbage in the mentions list must not raise; it is skipped defensively.
    payload: dict[str, Any] = {"mentions": ["nope", 5, {"surface": ""}, {"no_surface": 1}]}
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert isinstance(out, Outline)
    assert out.available is True
    assert out.mentions == ()


def test_non_mapping_mentions_payload_is_unavailable_never_raises() -> None:
    # A top-level payload that is not a JSON object (list/str) must read as "no
    # usable asset", never an AttributeError that 500s the review transcript.
    out = resolve_outline(["not", "a", "dict"], None, None, STARTS, asset_stale=False, gated=True)  # type: ignore[arg-type]
    assert out.available is False
    assert out.gated is True
    assert out.mentions == ()


def test_non_mapping_context_payloads_are_ignored_never_raise() -> None:
    # Bad summary/topics payloads are read as absent context, not a crash, and the
    # context path runs even when there is no mentions asset (available=False).
    out = resolve_outline(None, ["bad"], "also-bad", STARTS, asset_stale=False, gated=False)  # type: ignore[arg-type]
    assert out.available is False
    assert out.context == OutlineContext(summary=None, topics=())


def test_whitespace_only_surface_is_skipped() -> None:
    payload = _mentions(_mention("   ", "person", (0, "   ", 0)))
    out = resolve_outline(payload, None, None, STARTS, asset_stale=False, gated=False)
    assert out.mentions == ()
