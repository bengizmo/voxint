"""Deterministic multi-rule transcript corrector (issue #81).

The pure apply engine for a pack's ``corrections:`` rules. It consumes the frozen
:class:`~voxint.domain_packs.corrections.CorrectionRule` type and single-rule
matcher from #80 and turns ``(segment_text, rules)`` into corrected text plus a
structured applied-rule trace. It is **pure** — no DB, no ORM, no I/O, no
pipeline — mirroring :mod:`voxint.enrichment.triage`'s import hygiene, and it
performs no persistence: composing this into ``enhance_match`` and persisting the
trace/version is #82's job.

**Apply semantics (frozen — see docs/reports/nonllm-transcript-correction-design-2026-08-18.md
§6/§7/§8):**

- **Per-segment, single, NON-CASCADING pass.** Matching runs against the ORIGINAL
  segment text; a replacement's own characters are never re-examined.
- **Overlap resolution: leftmost-longest**, with manifest order (the ``rules``
  sequence order) only as the final tie-break.
- **Cursor-relative rediscovery.** The engine re-searches each rule from the live
  cursor rather than materializing every per-rule match up front. #80's
  :func:`~voxint.domain_packs.corrections.iter_matches` yields per-rule
  *non-overlapping* spans (it resumes past an accepted match), so an eager
  materialization would hide an occurrence that only becomes eligible once another
  rule consumes an earlier overlapping region — e.g. ``xababa`` with ``xab→X`` and
  ``aba→A`` must yield ``XA`` (``B`` at ``[0,3)`` then ``A`` at ``[3,6)``), yet a
  single ``iter_matches(aba)`` only ever surfaces ``[1,4)``. Re-searching from the
  cursor via :func:`_first_match_from` fixes this while reusing #80's
  :func:`~voxint.domain_packs.corrections.compile_match` and boundary predicate
  ``_boundary_ok`` **unchanged** (no forked boundary/case logic).
- **Exact-literal replace.** ``rule.replace`` is inserted verbatim; case is never
  inherited from the matched surface (a ``preserve_case`` flag is a future issue).
- **Trace.** Every applied rule emits an :class:`AppliedCorrection` whose ``span``
  addresses the FINAL corrected string (offsets shift as earlier replacements
  apply). A no-op (empty rule set, or no rule matched) returns the input unchanged
  with an empty trace.
- **Growth rejection.** When ``max_output_chars`` is set and the transformation
  would exceed it, the WHOLE segment transformation is rejected atomically
  (input returned unchanged, ``growth_rejected=True``, empty trace) — never
  truncated, never partially applied. The engine owns this *mechanism*; the policy
  value is supplied by the #82 composition layer. A pure no-op never trips the
  bound: an already-oversized input that no rule touches is returned as-is with
  ``growth_rejected=False`` (the engine only rejects a transformation it would
  itself introduce).

Idempotence is **conditional**, guaranteed only for a rule set that passed #80's
load-time :func:`~voxint.domain_packs.corrections.validate_corrections` (which
rejects any replacement containing another rule's match). That guard covers
*intra-replacement* re-firing only; a second-pass match that spans a replacement
and adjacent original text (``[ab→x, xc→y]`` on ``abc``) is outside it and is not
promised to be idempotent — see the #80 module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from voxint.domain_packs.corrections import (
    CorrectionRule,
    first_match_from,
)

# Bumped on any change to the apply rules, trace shape, or default behavior — a
# bump means already-persisted traces were produced by a different engine, so #82
# stores it beside each segment's trace. Pinned in tests/contracts/.
CORRECTOR_VERSION = 1


@dataclass(frozen=True)
class AppliedCorrection:
    """One applied rule in a correction pass.

    ``span`` is a half-open ``[start, end)`` range in the **final corrected
    string** — ``result.text[start:end] == to_text`` — so #82 (and the console,
    #83) can address the replacement in the text it actually persists, not the
    original. ``from_text`` is the exact original surface that matched
    (``from`` is a Python keyword; :meth:`to_mapping` emits the JSON key ``from``).
    """

    id: str
    from_text: str
    to_text: str
    span: tuple[int, int]

    def to_mapping(self) -> dict[str, Any]:
        """Serialize to the JSON-safe ``{id, from, to, span}`` shape #82 persists."""
        return {
            "id": self.id,
            "from": self.from_text,
            "to": self.to_text,
            "span": [self.span[0], self.span[1]],
        }


@dataclass(frozen=True)
class CorrectionResult:
    """The outcome of applying a rule set to one segment.

    ``text`` is the corrected string (identical to the input on a no-op or a
    growth rejection); ``trace`` lists the applied rules in output order; and
    ``growth_rejected`` is ``True`` only when a would-be transformation was
    rejected whole because its projected **output length** exceeded
    ``max_output_chars`` (in which case ``text`` is the untouched input and
    ``trace`` is empty). The bound is on final size, not net growth: a rule set
    that net-shrinks a segment but still lands above the cap is rejected too.
    """

    text: str
    trace: tuple[AppliedCorrection, ...]
    growth_rejected: bool


def _validate_max_output_chars(max_output_chars: int | None) -> None:
    """Reject a nonsensical growth bound loudly (``None`` or a positive int only)."""
    if max_output_chars is None:
        return
    # bool is an int subclass; a stray True/False is a caller bug, not a limit.
    if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int):
        raise ValueError(
            f"max_output_chars must be a positive int or None, got "
            f"{type(max_output_chars).__name__}"
        )
    if max_output_chars <= 0:
        raise ValueError(
            f"max_output_chars must be positive, got {max_output_chars}"
        )


def apply_corrections(
    text: str,
    rules: Sequence[CorrectionRule],
    *,
    max_output_chars: int | None = None,
) -> CorrectionResult:
    """Apply ``rules`` to one segment ``text`` (leftmost-longest, non-cascading).

    ``rules`` is an **ordered** sequence: manifest order is the final tie-break for
    two matches with the same start and length. Returns a :class:`CorrectionResult`
    whose ``trace`` spans address ``result.text``. See the module docstring for the
    frozen semantics; ``max_output_chars`` triggers atomic growth rejection.

    The caller is expected to pass a rule set that already passed #80's
    :func:`~voxint.domain_packs.corrections.validate_corrections` (the normal path
    through :meth:`DomainPack.from_mapping`); the engine does not re-validate — but
    it fails loud rather than misbehaving on the two shapes validation would have
    caught: a one-shot iterator (materialized to a tuple so every cursor step sees
    the full set) and a zero-width match (``ValueError``, never a silent hang).
    """
    _validate_max_output_chars(max_output_chars)
    # Materialize once: the cursor loop re-enumerates `rules` every step, so a
    # generator would be exhausted after the first winner and silently drop the
    # rest. This also snapshots the sequence against mutation mid-apply.
    rules = tuple(rules)

    # 1-2-3-4: rediscover each rule's earliest eligible match from the live cursor,
    # pick the leftmost-longest-earliest-rule winner, advance, repeat.
    winners: list[tuple[int, int, CorrectionRule]] = []
    pos = 0
    text_len = len(text)
    while pos < text_len:
        best: tuple[int, int, int] | None = None  # (start, -end, manifest_index)
        best_rule: CorrectionRule | None = None
        best_span: tuple[int, int] | None = None
        for index, rule in enumerate(rules):
            span = first_match_from(rule, text, pos)
            if span is None:
                continue
            start, end = span
            # leftmost start, then longest (max end == min -end), then earliest rule.
            key = (start, -end, index)
            if best is None or key < best:
                best = key
                best_rule = rule
                best_span = span
        if best_span is None or best_rule is None:
            break
        start, end = best_span
        if end <= start:
            # #80 rejects empty/whitespace matches, so this only fires when a caller
            # bypasses validation with a zero-width-matching rule. Fail loud: a
            # zero-width winner would never advance the cursor (infinite loop).
            raise ValueError(
                f"correction rule {best_rule.id!r} produced a zero-width match at "
                f"offset {start}; rules must pass validate_corrections before apply"
            )
        winners.append((start, end, best_rule))
        pos = end

    if not winners:
        # Pure no-op: never a growth rejection, even if `text` already exceeds the
        # limit — the engine only rejects a transformation it would introduce.
        return CorrectionResult(text=text, trace=(), growth_rejected=False)

    # 5: preflight the projected length before allocating a doomed output.
    projected = text_len + sum(
        len(rule.replace) - (end - start) for start, end, rule in winners
    )
    if max_output_chars is not None and projected > max_output_chars:
        return CorrectionResult(text=text, trace=(), growth_rejected=True)

    # 6-7: build the final text from ORIGINAL slices + literal replacements, and
    # record each applied rule's span in the FINAL string via a running offset.
    parts: list[str] = []
    trace: list[AppliedCorrection] = []
    src_cursor = 0
    out_len = 0
    for start, end, rule in winners:
        gap = text[src_cursor:start]
        parts.append(gap)
        out_len += len(gap)
        final_start = out_len
        parts.append(rule.replace)
        out_len += len(rule.replace)
        trace.append(
            AppliedCorrection(
                id=rule.id,
                from_text=text[start:end],
                to_text=rule.replace,
                span=(final_start, out_len),
            )
        )
        src_cursor = end
    parts.append(text[src_cursor:])
    corrected = "".join(parts)

    return CorrectionResult(text=corrected, trace=tuple(trace), growth_rejected=False)
