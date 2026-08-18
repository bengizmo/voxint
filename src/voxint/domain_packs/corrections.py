"""Operator-defined literal transcript corrections (issue #80).

A pack's ``corrections:`` list declares deterministic, offline **literal**
substitution rules — "zoom board" → "Zoning Board", "C D B G" → "CDBG" — that a
downstream engine (issue #81) applies to segment text during enhancement. This
module owns the *rule type*, all strict load-time validation, and the pure
single-rule **matcher** that both the idempotence validation here and the #81
apply engine consume. It performs **no text mutation** and touches no pipeline
stage.

Naming: :class:`CorrectionRule` (not "Correction") stays textually distinct from
the codebase's *manual* per-segment review edits, which are already called
corrections (``SegmentReviewState.corrected_text``, ``TranscriptText.CORRECTED``,
precedence ``corrected → enhanced → raw``, issue #58). Those are review-time
operator edits; these are deterministic literal rules declared in a pack manifest
and frozen per run.

The matcher deliberately avoids Python's bare ``\\b``: an explicit boundary
predicate treats apostrophes, hyphens, and combining marks as *intra-word*, so an
``it → IT`` rule never fires inside ``it's`` and a whole-word rule never splits an
NFD grapheme (``Zoë`` = ``Z o e U+0308``). Matching runs against the *original*
text via ``re.escape`` (metacharacters inert) with optional ``re.IGNORECASE``;
the ``replace`` string is an exact literal whose casing is authored, never
inherited from the matched surface (that substitution is #81's concern).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from voxint.domain_packs.base import DomainPackError

# Bounds are rejected, never truncated: a manifest that exceeds one is a
# configuration error the operator must see and fix.
MAX_RULES_PER_PACK = 256
MAX_MATCH_CHARS = 256
MAX_REPLACEMENT_CHARS = 512
MAX_CORRECTIONS_MANIFEST_BYTES = 131072

_ALLOWED_KEYS = frozenset({"id", "match", "replace", "case_sensitive", "whole_word"})

# Unicode general categories rejected in id/match/replace: Cc control (incl. NUL,
# CR, LF, TAB), Cf format (BOM, zero-width space/joiner, bidi overrides, soft
# hyphen), and Cs surrogate. All are either invisible in a transcript (an
# invisible-character injection surface) or, for lone surrogates, crash the
# canonical UTF-8 serialization used by the byte bound — so they are rejected at
# parse time as a clean DomainPackError rather than escaping as a raw
# UnicodeEncodeError past the single validation point. Combining marks (M*) and
# ordinary punctuation (the curly apostrophe) are NOT in these categories.
_NON_PRINTING_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})

# Characters that JOIN a word for whole-word boundary purposes, beyond the
# alphanumerics and combining marks handled in :func:`_is_word_char`: the ASCII
# and typographic apostrophes and the ASCII hyphen. Other Unicode dashes/hyphens
# are intentionally left as boundaries for v1 (a later issue can broaden this).
_INTRA_WORD = frozenset({"'", "’", "-"})  # noqa: RUF001 — the curly apostrophe is intentional


@dataclass(frozen=True)
class CorrectionRule:
    """One frozen literal-substitution rule from a pack's ``corrections:`` list.

    ``match`` is a literal phrase (never a regex); ``replace`` is the exact
    canonical form the operator authored. ``case_sensitive`` and ``whole_word``
    both default ``True`` — the conservative posture that best fits recurring
    domain terms and avoids surprising a non-technical operator.
    """

    id: str
    match: str
    replace: str
    case_sensitive: bool = True
    whole_word: bool = True

    def to_mapping(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for the per-run snapshot.

        The resolved booleans are always emitted explicitly (never omitted when
        default) so the frozen ``pipeline_runs.domain_pack`` snapshot is fully
        self-describing and round-trips deterministically.
        """
        return {
            "id": self.id,
            "match": self.match,
            "replace": self.replace,
            "case_sensitive": self.case_sensitive,
            "whole_word": self.whole_word,
        }


# --- parsing (strict) --------------------------------------------------------


def parse_corrections(value: Any) -> tuple[CorrectionRule, ...]:
    """Reconstruct and validate a pack's ``corrections`` field, strictly.

    Shared by :meth:`DomainPack.from_mapping` for both manifest load and the
    per-run snapshot restore. ``None`` or an empty list yields ``()`` (the honest
    no-op default). Any malformed rule, bound violation, duplicate id, or
    non-idempotent rule set raises :class:`DomainPackError`; the only caller that
    tolerates that (degrading to the default pack) is ``domain_pack_from_snapshot``
    on out-of-band-tampered snapshots.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise DomainPackError(
            f"domain pack 'corrections' must be a list, got {type(value).__name__}"
        )
    # Bound the rule count before parsing every entry, so a pathological manifest
    # with a huge list fails fast rather than materializing all of it first.
    if len(value) > MAX_RULES_PER_PACK:
        raise DomainPackError(
            f"domain pack declares {len(value)} corrections; the maximum is "
            f"{MAX_RULES_PER_PACK}"
        )
    rules = tuple(_rule_from_mapping(item, index) for index, item in enumerate(value))
    validate_corrections(rules)
    return rules


def _rule_from_mapping(value: Any, index: int) -> CorrectionRule:
    """Parse one ``corrections[index]`` entry (field-level validation)."""
    if not isinstance(value, Mapping):
        raise DomainPackError(
            f"domain pack corrections[{index}] must be a mapping, "
            f"got {type(value).__name__}"
        )
    # YAML allows non-string mapping keys (e.g. `1:` or `off:`), so map to str
    # before sorting — sorting a mixed-type set raises TypeError, which would
    # escape the single DomainPackError validation channel.
    unknown = set(value) - _ALLOWED_KEYS
    if unknown:
        raise DomainPackError(
            f"domain pack corrections[{index}] has unknown keys: "
            f"{sorted(map(str, unknown))}"
        )
    rid = _req_str(value, "id", index)
    if not rid.strip():
        raise DomainPackError(f"domain pack corrections[{index}].id must be non-empty")
    if rid != rid.strip():
        # Reject surrounding whitespace so " a " and "a" cannot be distinct ids
        # that look identical in diagnostics and the (#82) applied-rule trace.
        raise DomainPackError(
            f"domain pack corrections[{index}].id must not have leading or "
            "trailing whitespace"
        )
    match = _req_str(value, "match", index)
    replace = _req_str(value, "replace", index)
    for field_name, text in (("match", match), ("replace", replace)):
        # Whitespace-only would be a silent dead rule (a YAML `>` block scalar
        # yields a trailing newline) — the "never a silent skip" contract forbids
        # it, so reject loudly.
        if not text.strip():
            raise DomainPackError(
                f"domain pack corrections[{index}].{field_name} must be non-empty "
                "and not only whitespace"
            )
    for field_name, text in (("id", rid), ("match", match), ("replace", replace)):
        _reject_non_printing(field_name, text, index)
    return CorrectionRule(
        id=rid,
        match=match,
        replace=replace,
        case_sensitive=_opt_bool(value, "case_sensitive", index),
        whole_word=_opt_bool(value, "whole_word", index),
    )


def _req_str(mapping: Mapping[str, Any], key: str, index: int) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise DomainPackError(
            f"domain pack corrections[{index}].{key} is required and must be a string"
        )
    return value


def _opt_bool(mapping: Mapping[str, Any], key: str, index: int) -> bool:
    if key not in mapping:
        return True
    value = mapping[key]
    # `isinstance(True, int)` is True, so guard on bool explicitly to reject a
    # YAML integer (`1`) or string (`"yes"`) masquerading as a flag.
    if not isinstance(value, bool):
        raise DomainPackError(
            f"domain pack corrections[{index}].{key} must be a boolean"
        )
    return value


def _reject_non_printing(field_name: str, text: str, index: int) -> None:
    """Reject control/format/surrogate characters (see ``_NON_PRINTING_CATEGORIES``)."""
    for ch in text:
        if unicodedata.category(ch) in _NON_PRINTING_CATEGORIES:
            raise DomainPackError(
                f"domain pack corrections[{index}].{field_name} contains a "
                f"non-printing character (U+{ord(ch):04X})"
            )


# --- validation (cross-rule) -------------------------------------------------


def validate_corrections(rules: Sequence[CorrectionRule]) -> None:
    """Enforce the cross-rule invariants: bounds, unique ids, idempotence.

    Ordered bounds → uniqueness → manifest-bytes → idempotence so error messages
    are deterministic. Assumes each rule already passed field-level parsing.
    """
    if len(rules) > MAX_RULES_PER_PACK:
        raise DomainPackError(
            f"domain pack declares {len(rules)} corrections; the maximum is "
            f"{MAX_RULES_PER_PACK}"
        )
    for index, rule in enumerate(rules):
        if len(rule.match) > MAX_MATCH_CHARS:
            raise DomainPackError(
                f"domain pack corrections[{index}].match is {len(rule.match)} "
                f"characters; the maximum is {MAX_MATCH_CHARS}"
            )
        if len(rule.replace) > MAX_REPLACEMENT_CHARS:
            raise DomainPackError(
                f"domain pack corrections[{index}].replace is {len(rule.replace)} "
                f"characters; the maximum is {MAX_REPLACEMENT_CHARS}"
            )
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise DomainPackError(
                f"domain pack corrections has a duplicate id {rule.id!r}; "
                "each correction id must be unique"
            )
        seen.add(rule.id)
    # Measure a canonical serialization of the resolved rules, not raw YAML bytes:
    # `from_mapping` is the single validation point shared by manifest load and
    # snapshot restore, and a snapshot has no source file, so a stable json.dumps
    # is the only measure consistent across both. It caps rule payload, not YAML
    # comments/whitespace.
    encoded = json.dumps(
        [rule.to_mapping() for rule in rules],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_CORRECTIONS_MANIFEST_BYTES:
        raise DomainPackError(
            f"domain pack corrections serialize to {len(encoded)} bytes; the "
            f"maximum is {MAX_CORRECTIONS_MANIFEST_BYTES}"
        )
    _reject_replacement_contains_match(rules)


def _reject_replacement_contains_match(rules: Sequence[CorrectionRule]) -> None:
    """Reject rule sets that re-fire within a single replacement string.

    A single non-cascading pass does not give ``correct(correct(t)) == correct(t)``
    across rule chains: ``[a→b, b→c]`` yields ``correct("a") == "b"`` but
    ``correct(correct("a")) == "c"``. This guard rejects any set where some rule
    ``r`` would **actually re-fire** on some rule ``s``'s replacement (including
    ``r is s``), evaluated with **``r``'s own** ``case_sensitive``/``whole_word``
    — because ``r`` is the potential re-firer and the engine decides ``r``'s
    firing by ``r``'s flags; ``s``'s flags only governed how ``s`` matched the
    original text. This is boundary-aware: ``aa→aaa`` is idempotent (accepted)
    when ``whole_word`` is true (``aa`` is not a whole-word match inside ``aaa``)
    and non-idempotent (rejected) when it is false. O(n²) over ≤256 rules is
    trivial.

    Scope: the guarantee is the *intra-replacement* re-fire property — no rule's
    match occurs within any single replacement string. It does **not** cover a
    second-pass match that spans a replacement and adjacent original text
    (``[ab→x, xc→y]`` passes here yet ``correct(correct("abc")) != correct("abc")``)
    — those cross-boundary compositions are outside v1's load-time guard and are
    the province of the #81 apply engine's runtime faithfulness corpus.
    """
    for firer in rules:
        for target in rules:
            if find_first(firer, target.replace) is not None:
                raise DomainPackError(
                    f"domain pack corrections are not idempotent: rule "
                    f"{firer.id!r} (match {firer.match!r}) would re-fire on the "
                    f"replacement of rule {target.id!r} (replace "
                    f"{target.replace!r}); split the rules, or change a "
                    "replacement or its flags, so no replacement contains "
                    "another rule's match"
                )


# --- single-rule matcher (reused by the #81 apply engine) --------------------


def _is_word_char(ch: str) -> bool:
    """Whether ``ch`` JOINS a word for whole-word boundary purposes.

    Alphanumerics, the intra-word punctuation (the ASCII and typographic
    apostrophes and the ASCII hyphen), and any
    Unicode combining mark (category ``M*``) — so a decomposed grapheme like
    ``Zoë`` (``Z o e U+0308``) is never split between its base letter and its
    combining mark. This explicit predicate replaces Python's bare ``\\b``, which
    treats apostrophes, hyphens, and combining marks as boundaries.
    """
    return ch.isalnum() or ch in _INTRA_WORD or unicodedata.category(ch).startswith("M")


def _boundary_ok(text: str, start: int, end: int, whole_word: bool) -> bool:
    """Whether ``text[start:end]`` sits on word boundaries (trivially true if not whole-word)."""
    if not whole_word:
        return True
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before and _is_word_char(before)) and not (after and _is_word_char(after))


def compile_match(rule: CorrectionRule) -> re.Pattern[str]:
    """Compile ``rule.match`` as a literal (``re.escape``), honoring case sensitivity."""
    flags = 0 if rule.case_sensitive else re.IGNORECASE
    return re.compile(re.escape(rule.match), flags)


def first_match_from(
    rule: CorrectionRule, text: str, start: int = 0
) -> tuple[int, int] | None:
    """First boundary-valid ``[begin, end)`` match span of ``rule`` at or after ``start``.

    The single scan primitive that both :func:`iter_matches` (per-rule iteration)
    and the #81 apply engine (cursor-relative rediscovery across rules) share, so
    boundary/case/resume semantics live in exactly one place. A rejected
    (boundary-invalid) candidate resumes the scan **one code point past its
    start**, not at its end, so it can never hide a later *overlapping* candidate
    that IS boundary-valid — e.g. ``"ha ha"`` against ``"aha ha ha"`` still finds
    ``(4, 9)`` even though the candidate at offset 1 fails its left boundary.

    Honors ``case_sensitive`` (via ``re.IGNORECASE`` — for matching only) and the
    explicit whole-word predicate. ``re.escape`` makes ``match`` a literal, so
    regex metacharacters match themselves. Matching is against the original text —
    never a casefolded copy, which would shift the reported offsets.
    """
    pattern = compile_match(rule)
    pos = start
    while (m := pattern.search(text, pos)) is not None:
        if _boundary_ok(text, m.start(), m.end(), rule.whole_word):
            return (m.start(), m.end())
        pos = m.start() + 1
    return None


def iter_matches(rule: CorrectionRule, text: str) -> Iterator[tuple[int, int]]:
    """Yield half-open ``[start, end)`` spans in ORIGINAL ``text`` where ``rule`` matches.

    Accepted matches are non-overlapping — the scan resumes at each accepted end —
    mirroring the single left-to-right pass of the #81 engine. Boundary/case/resume
    semantics come from :func:`first_match_from`.
    """
    pos = 0
    while (span := first_match_from(rule, text, pos)) is not None:
        yield span
        pos = span[1]


def find_first(rule: CorrectionRule, text: str) -> tuple[int, int] | None:
    """The first boundary-valid match span of ``rule`` in ``text``, or ``None``."""
    return first_match_from(rule, text, 0)
