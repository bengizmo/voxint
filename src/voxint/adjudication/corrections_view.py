"""Read-time deterministic-correction provenance for the review console (#83).

Pure, DB-free helpers that turn what #82 already persists — the per-segment
``correction_trace`` envelope + ``corrector_version``, and the run's one frozen
``pipeline_runs.domain_pack`` snapshot — into (a) per-segment provenance the island
can display and (b) a run-level declared-rule reconciliation ("declared but never
fired"). No migration: everything is reconstructed from immutable evidence.

Numerics doctrine: reuse :func:`trace_has_entries` as the canonical "did a rule
materially fire" predicate; NEVER re-diff effective text. Reconciliation replays
the corrector over the immutable ``raw_text`` with the SAME growth ceiling the
pipeline used (:func:`enhanced_size_ceiling`), so the reconstructed fire-set is
byte-faithful to the raw pass in ``enhance_match.run``.

Reconciliation truth table (#83 Step 0a — the design contract these helpers pin):

  Per-segment provenance (the "corrected by domain pack" marker):
    trace == [] / absent ............... None (segment not materially corrected)
    envelope, version == CURRENT ....... shown; each entry resolved against the
                                         snapshot (pack/match/replace), an id absent
                                         from the snapshot stays VISIBLE as unresolved
    envelope, version != CURRENT ....... unavailable(version_mismatch) — never replay
                                         with mismatched semantics
    envelope, snapshot missing/corrupt . shown, entries unresolved (id/from/to/span
                                         come from the trace itself; pack name absent)
    split-child line ................... None (provenance is PARENT-scoped; a child
                                         slice must never claim parent-coordinate spans)

  Per-run reconciliation (the "declared but never fired" panel), per DECLARED rule,
  aggregated across every segment's immutable raw_text with precedence
  applied > growth_rejected > no_raw_match:
    fired on >=1 segment's raw ......... applied (with the count of segments)
    would-fire but that segment's raw
      transformation is growth-rejected . growth_rejected (raw pass; exact)
    matched no segment's raw ........... no_raw_match

  Out of v1 (honest, documented gaps — design report §6/§12-F5): LLM-enforcement-pass
  growth rejection (its deciding input is not persisted) and ``cross_segment``
  (a term ASR-split across a pause) — steer such terms to pack ``vocabulary``.

Reconciliation reflects the CURRENT corrector's semantics; with only
``CORRECTOR_VERSION == 1`` in the field today it is exact. A future engine bump must
version-dispatch replay (or report honestly) rather than reinterpret old runs.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from voxint.adjudication.splits import trace_has_entries
from voxint.clients.llm import enhanced_size_ceiling
from voxint.domain_packs.base import DomainPackError
from voxint.domain_packs.corrections import CorrectionRule, parse_corrections
from voxint.domain_packs.corrector import CORRECTOR_VERSION, apply_corrections

ReconStatus = Literal["applied", "no_raw_match", "growth_rejected"]


@dataclass(frozen=True)
class RuleDisplay:
    """The operator-facing identity of one declared correction rule."""

    id: str
    pack: str
    match: str
    replace: str


@dataclass(frozen=True)
class DeclaredRuleIndex:
    """A run's one frozen pack resolved for read-time provenance.

    ``by_id`` resolves a fired/declared rule id to its display identity; ``rules``
    keeps manifest order for the reconciliation replay (order is the corrector's
    final tie-break, so it must match what the pipeline used).
    """

    pack: str
    rules: tuple[CorrectionRule, ...]
    by_id: Mapping[str, RuleDisplay]


def build_declared_rule_index(
    snapshot: Mapping[str, Any] | None,
) -> DeclaredRuleIndex | None:
    """Resolve the run's ``pipeline_runs.domain_pack`` snapshot into a rule index.

    Reads the snapshot dict DIRECTLY (never ``domain_pack_from_snapshot``, which
    degrades a NULL/corrupt snapshot to the *current default pack* — that would
    fabricate declarations a run never had). Returns ``None`` when the snapshot is
    absent or unreadable, so the caller shows an honest "provenance unavailable"
    state instead of borrowing another pack's rules.
    """
    if not isinstance(snapshot, Mapping):
        return None
    name = snapshot.get("name")
    if not isinstance(name, str):
        return None
    try:
        rules = parse_corrections(snapshot.get("corrections"))
    except DomainPackError:
        return None
    by_id = {
        rule.id: RuleDisplay(
            id=rule.id, pack=name, match=rule.match, replace=rule.replace
        )
        for rule in rules
    }
    return DeclaredRuleIndex(pack=name, rules=rules, by_id=by_id)


def resolve_segment_provenance(
    trace: Mapping[str, Any] | Sequence[Any] | None,
    corrector_version: int | None,
    index: DeclaredRuleIndex | None,
) -> dict[str, Any] | None:
    """Per-segment display provenance from a persisted ``correction_trace``.

    ``None`` when no rule materially fired (``trace_has_entries`` is false — the
    canonical predicate, never a text diff). Otherwise a JSON-ready object:

    - ``{"status": "unavailable", "reason": "version_mismatch", ...}`` when the row
      was written by a different corrector version than this console reads;
    - ``{"status": "shown", "version", "inputBase", "entries": [...]}`` otherwise,
      each entry carrying the trace's own ``id/from/to/span`` plus the resolved
      ``pack/match/replace`` (``resolved: false`` and ``pack: null`` when the id is
      not in the snapshot — the entry stays visible, never silently dropped).
    """
    if not trace_has_entries(trace):
        return None
    assert isinstance(trace, Mapping)  # trace_has_entries guarantees the envelope
    version = trace.get("version")
    if version != CORRECTOR_VERSION or corrector_version != CORRECTOR_VERSION:
        return {
            "status": "unavailable",
            "reason": "version_mismatch",
            "recordedVersion": version if isinstance(version, int) else corrector_version,
        }
    entries: list[dict[str, Any]] = []
    for raw_entry in trace.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            continue
        rule_id = raw_entry.get("id")
        display = (
            index.by_id.get(rule_id)
            if index is not None and isinstance(rule_id, str)
            else None
        )
        entries.append(
            {
                "id": rule_id,
                "from": raw_entry.get("from"),
                "to": raw_entry.get("to"),
                "span": raw_entry.get("span"),
                "pack": display.pack if display is not None else None,
                "match": display.match if display is not None else None,
                "replace": display.replace if display is not None else None,
                "resolved": display is not None,
            }
        )
    return {
        "status": "shown",
        "version": version,
        "inputBase": trace.get("input_base"),
        "entries": entries,
    }


def _raw_fired_ids(raw_text: str, rules: Sequence[CorrectionRule]) -> tuple[set[str], bool]:
    """Which declared rule ids fire on one segment's immutable raw text.

    Replays the corrector with the SAME growth ceiling ``enhance_match`` used, so
    the fire-set and growth decision are byte-faithful to the pipeline's raw pass.
    On a growth-rejected segment the capped pass returns an empty trace, so re-run
    UNCAPPED to recover the ids that *would* have fired — those are the segment's
    ``growth_rejected`` candidates. Returns ``(fired_ids, growth_rejected)``.
    """
    capped = apply_corrections(
        raw_text, rules, max_output_chars=enhanced_size_ceiling(raw_text)
    )
    if not capped.growth_rejected:
        return {entry.id for entry in capped.trace}, False
    uncapped = apply_corrections(raw_text, rules)
    return {entry.id for entry in uncapped.trace}, True


def run_reconciliation(
    index: DeclaredRuleIndex | None,
    raw_texts: Iterable[str],
) -> list[dict[str, Any]]:
    """Per-declared-rule reconciliation for the run-level "declared but never fired"
    panel, aggregated over every segment's immutable ``raw_text``.

    ``[]`` when the snapshot is unavailable or declares no corrections (the panel
    simply does not render). Otherwise one entry per declared rule with its status
    (precedence ``applied`` > ``growth_rejected`` > ``no_raw_match``) and, for an
    applied rule, ``appliedCount`` — how many segments it fired on. Computed once
    per run by the caller; not per segment per request.
    """
    if index is None or not index.rules:
        return []
    applied_counts: dict[str, int] = {rule.id: 0 for rule in index.rules}
    growth_rejected: set[str] = set()
    for raw_text in raw_texts:
        fired, was_growth_rejected = _raw_fired_ids(raw_text, index.rules)
        for rule_id in fired:
            if was_growth_rejected:
                growth_rejected.add(rule_id)
            else:
                applied_counts[rule_id] += 1
    out: list[dict[str, Any]] = []
    for rule in index.rules:
        count = applied_counts[rule.id]
        if count > 0:
            status: ReconStatus = "applied"
        elif rule.id in growth_rejected:
            status = "growth_rejected"
        else:
            status = "no_raw_match"
        out.append(
            {
                "id": rule.id,
                "pack": index.pack,
                "match": rule.match,
                "replace": rule.replace,
                "status": status,
                "appliedCount": count,
            }
        )
    return out
