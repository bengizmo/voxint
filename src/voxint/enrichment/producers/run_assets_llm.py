"""LLM generators for run-level assets (#41): summary, topics, entity mentions.

Each generator is one strict-JSON ``chat_json`` call over the run's source
snapshot (diarization-labeled transcript + #36 metadata + operator notes),
followed by fail-closed parsing. There is deliberately no tool loop here —
unlike #40's web researcher, the model sees everything it may use in the one
prompt, so a single request with one shot at valid output is the whole
protocol; anything malformed raises and the job records an honest failure.

Grounding: entity-mention offsets are never trusted from the model. The
producer locates each quoted span in the referenced segment itself (exact
match first, then case-insensitive, both requiring non-letter boundaries so
"Ann" cannot anchor inside "Joanne" — the #38 lesson) and records the found
offsets. Unlocatable or out-of-run spans are dropped fail-soft with counts in
the payload's ``diagnostics`` — a hallucinated span must never masquerade as
transcript evidence, but one bad span should not void the survivors.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from voxint.clients.llm import ChatMessage, HttpLLMClient
from voxint.config import Settings
from voxint.db.models import RunAssetKind
from voxint.enrichment.run_assets import (
    ENTITY_KINDS,
    MAX_MENTIONS,
    MAX_OCCURRENCES_PER_MENTION,
    MAX_SUMMARY_CHARS,
    MAX_SURFACE_CHARS,
    MAX_TOPIC_DESCRIPTION_CHARS,
    MAX_TOPIC_LABEL_CHARS,
    MAX_TOPICS,
    RunAssetSource,
    quote_matches_surface,
)

PRODUCER_NAME = "run_assets.llm"
# v2: transcript lines now carry the attributed speaker name (resolved through
# the shared display_name) instead of the raw diarization label, and the
# entity-mention instruction warns not to treat a speaker prefix as a mention.
# Both are provenance-only — recorded on new assets, never folded into
# source_content_hash — so bumping them does not force regeneration.
PRODUCER_VERSION = "2"
PROMPT_VERSION = 2
CONFIG_SCHEMA_VERSION = 1
PAYLOAD_SCHEMA_VERSION = 1

# Marker inserted where head+tail truncation removed the middle of a long
# transcript. Recorded via the config snapshot's ``truncated`` flag too — the
# hash always covers the FULL source, so staleness stays honest either way.
_TRUNCATION_MARKER = "\n[... transcript truncated for length ...]\n"


class RunAssetProducerError(Exception):
    """The model's reply could not be turned into a valid asset payload."""


def render_source(source: RunAssetSource, *, max_chars: int) -> tuple[str, bool]:
    """The prompt's source document; head+tail truncated over ``max_chars``.

    Refusing long runs outright would make the feature useless on exactly the
    runs that most need a summary, so the middle goes, the marker says so,
    and the config snapshot records ``truncated=true``.
    """
    lines = []
    if source.metadata is not None:
        lines.append("Source metadata:")
        lines.append(json.dumps(dict(source.metadata), sort_keys=True, ensure_ascii=False))
    if source.operator_notes:
        lines.append("Operator notes:")
        lines.append(source.operator_notes)
    lines.append("Transcript:")
    for segment in source.segments:
        # ``speaker`` is the attributed name (resolved + sanitized in
        # load_source); render it verbatim so the prompt and the staleness hash
        # describe the identical string.
        lines.append(f"[{segment.segment_index}] {segment.speaker}: {segment.text}")
    document = "\n".join(lines)
    if len(document) <= max_chars:
        return document, False
    keep = max_chars - len(_TRUNCATION_MARKER)
    if keep <= 0:  # settings floor prevents this; guard against future drift
        return document[:max_chars], True
    head = keep * 2 // 3
    tail = keep - head
    return document[:head] + _TRUNCATION_MARKER + document[-tail:], True


_SYSTEM = (
    "You are an analyst producing structured metadata about one transcribed"
    " recording. Reply with a single JSON object and nothing else — no prose,"
    " no markdown fences. Base every statement only on the provided document;"
    " never invent facts."
)

_INSTRUCTIONS: dict[str, str] = {
    RunAssetKind.SUMMARY.value: (
        "Write a short neutral abstract of the recording (3-8 sentences,"
        f" at most {MAX_SUMMARY_CHARS} characters). Cover who speaks, what is"
        " discussed, and any conclusions reached. Reply exactly as:"
        ' {"summary": "..."}'
    ),
    RunAssetKind.TOPICS.value: (
        f"List 1-{MAX_TOPICS} topics that describe what the recording is"
        " about, most central first. Labels are short noun phrases (at most"
        f" {MAX_TOPIC_LABEL_CHARS} characters), no duplicates. Reply exactly"
        ' as: {"topics": [{"label": "...", "description": "one sentence or'
        ' null", "confidence": 0.0-1.0 or null}]}'
    ),
    RunAssetKind.ENTITY_MENTIONS.value: (
        "List people, organizations, and products explicitly mentioned in the"
        f" transcript (at most {MAX_MENTIONS}). The speaker name before the"
        " colon on each line labels who is talking — it is NOT part of the"
        " transcript text, so do not report a speaker as a mention unless the"
        " name also appears inside a segment's own text. For each, give the"
        " surface form and every place it occurs: the segment index (the number"
        " in brackets) and the EXACT verbatim quote of the mention as it appears"
        " in that segment's text (at most"
        f" {MAX_OCCURRENCES_PER_MENTION} occurrences per entity). Reply"
        ' exactly as: {"mentions": [{"surface": "...", "kind": "person" |'
        ' "organization" | "product" | null, "occurrences": [{"segment_index":'
        ' 0, "quote": "..."}]}]}. An empty list is a valid answer.'
    ),
}


def build_messages(
    kind: RunAssetKind, source: RunAssetSource, *, max_chars: int
) -> tuple[Sequence[ChatMessage], bool]:
    document, truncated = render_source(source, max_chars=max_chars)
    return (
        (
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=f"{_INSTRUCTIONS[kind.value]}\n\n{document}"),
        ),
        truncated,
    )


def _parse_summary(body: Mapping[str, Any]) -> dict[str, Any]:
    summary = body.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RunAssetProducerError("reply has no usable summary string")
    if len(summary) > MAX_SUMMARY_CHARS:
        raise RunAssetProducerError(
            f"summary is {len(summary)} chars against a {MAX_SUMMARY_CHARS} bound"
        )
    return {"summary": summary.strip()}


def _parse_topics(body: Mapping[str, Any]) -> dict[str, Any]:
    raw = body.get("topics")
    if not isinstance(raw, list):
        raise RunAssetProducerError("reply has no topics array")
    topics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise RunAssetProducerError("topic entry is not an object")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise RunAssetProducerError("topic entry has no usable label")
        label = " ".join(label.split())
        if len(label) > MAX_TOPIC_LABEL_CHARS:
            raise RunAssetProducerError(f"topic label over {MAX_TOPIC_LABEL_CHARS} chars")
        folded = label.casefold()
        if folded in seen:
            continue  # keep the first spelling, drop the duplicate
        seen.add(folded)
        description = item.get("description")
        if not (isinstance(description, str) and description.strip()):
            description = None
        else:
            description = " ".join(description.split())[:MAX_TOPIC_DESCRIPTION_CHARS]
        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            confidence = None
        else:
            confidence = float(confidence)
        topics.append(
            {
                "label": label,
                "description": description,
                "confidence": confidence,
                # Reserved for #11 domain-pack vocabularies (payload v1: null).
                "vocabulary": None,
                "term_id": None,
            }
        )
        if len(topics) == MAX_TOPICS:
            break
    if not topics:
        # An empty topic list is a refusal to answer, not an asset — recording
        # it would authoritatively claim the run is about nothing.
        raise RunAssetProducerError("reply produced no usable topics")
    return {"topics": topics}


# An alphanumeric for boundary purposes (regex \w minus underscore) — a quote
# can anchor inside neither a longer word NOR a longer number ("1" in "2019").
# Mirrors run_assets.span_boundaries_ok, which the writer re-checks.
_ALNUM = r"[^\W_]"


def _locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Offsets of ``quote`` in ``text`` — exact first, then case-insensitive.

    Both passes require non-alphanumeric boundaries when the quote itself
    starts or ends with an alphanumeric. First match wins (deterministic);
    offsets always index the ORIGINAL text, so the recorded span reproduces
    the transcript verbatim modulo case only in the fallback — which is why
    the recorded quote is re-sliced from the text, never taken from the model.
    """
    prefix = rf"(?<!{_ALNUM})" if quote and quote[0].isalnum() else ""
    suffix = rf"(?!{_ALNUM})" if quote and quote[-1].isalnum() else ""
    for flags in (0, re.IGNORECASE):
        match = re.search(prefix + re.escape(quote) + suffix, text, flags | re.UNICODE)
        if match is not None:
            return match.start(), match.end()
    return None


def _parse_mentions(body: Mapping[str, Any], source: RunAssetSource) -> dict[str, Any]:
    raw = body.get("mentions")
    if not isinstance(raw, list):
        raise RunAssetProducerError("reply has no mentions array")
    segment_text = {segment.segment_index: segment.text for segment in source.segments}
    mentions: list[dict[str, Any]] = []
    dropped_unlocatable = 0
    dropped_out_of_run = 0
    for item in raw:
        if not isinstance(item, Mapping):
            raise RunAssetProducerError("mention entry is not an object")
        surface = item.get("surface")
        if not isinstance(surface, str) or not surface.strip():
            raise RunAssetProducerError("mention entry has no usable surface")
        surface = " ".join(surface.split())[:MAX_SURFACE_CHARS]
        kind = item.get("kind")
        if kind not in ENTITY_KINDS:
            kind = None
        raw_occurrences = item.get("occurrences")
        if not isinstance(raw_occurrences, list) or not raw_occurrences:
            # A mention asserted without a single occurrence is a protocol
            # violation, not a droppable span — silently omitting it would
            # let a reply full of such mentions become an authoritative
            # "no entities" asset with zero diagnostics.
            raise RunAssetProducerError("mention entry has no occurrences")
        occurrences: list[dict[str, Any]] = []
        for occurrence in raw_occurrences:
            if not isinstance(occurrence, Mapping):
                raise RunAssetProducerError("occurrence entry is not an object")
            index = occurrence.get("segment_index")
            quote = occurrence.get("quote")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not isinstance(quote, str)
                or not quote.strip()
            ):
                # Type-level message only — the occurrence body is model
                # output and must not ride into persisted error text.
                raise RunAssetProducerError(
                    "occurrence entry has wrong types (segment_index must be"
                    " an integer, quote a non-empty string)"
                )
            text = segment_text.get(index)
            if text is None:
                dropped_out_of_run += 1
                continue
            located = _locate_quote(quote.strip(), text)
            if located is None:
                dropped_unlocatable += 1
                continue
            if not quote_matches_surface(surface, text[located[0] : located[1]]):
                # A genuinely-located quote that is unrelated to the surface
                # ("Mallory" hung on "the") is not grounding for this entity.
                dropped_unlocatable += 1
                continue
            start, end = located
            occurrences.append(
                {
                    "segment_index": index,
                    # Re-sliced from the transcript, never the model's string —
                    # the persisted quote must reproduce the segment verbatim.
                    "quote": text[start:end],
                    "start_char": start,
                    "end_char": end,
                }
            )
            if len(occurrences) == MAX_OCCURRENCES_PER_MENTION:
                break
        if occurrences:
            mentions.append({"surface": surface, "kind": kind, "occurrences": occurrences})
        if len(mentions) == MAX_MENTIONS:
            break
    if raw and not mentions:
        # The model offered mentions and none survived grounding — that is a
        # failed generation, not an authoritative "no entities" (#40 lesson).
        raise RunAssetProducerError(
            f"no mention survived grounding ({dropped_unlocatable} unlocatable,"
            f" {dropped_out_of_run} out of run)"
        )
    return {
        "mentions": mentions,
        "diagnostics": {
            "dropped_unlocatable": dropped_unlocatable,
            "dropped_out_of_run": dropped_out_of_run,
        },
    }


def generate_payload(
    llm: HttpLLMClient | Any,
    kind: RunAssetKind,
    source: RunAssetSource,
    *,
    settings: Settings,
) -> tuple[dict[str, Any], bool]:
    """One generation: prompt, call, parse. Returns (payload, truncated).

    ``llm`` needs only ``chat_json`` (injection seam for tests / the CLI's
    inline mode). Raises :class:`voxint.clients.llm.LLMError` for transport or
    envelope failures and :class:`RunAssetProducerError` for replies that
    cannot become a valid payload — both mean the job fails; nothing partial
    is ever persisted.
    """
    messages, truncated = build_messages(
        kind, source, max_chars=settings.run_assets_max_input_chars
    )
    body = llm.chat_json(messages)
    if kind is RunAssetKind.SUMMARY:
        return _parse_summary(body), truncated
    if kind is RunAssetKind.TOPICS:
        return _parse_topics(body), truncated
    return _parse_mentions(body, source), truncated


def config_snapshot(settings: Settings, *, truncated: bool) -> dict[str, Any]:
    """The bounded execution-shape record stored on the asset."""
    return {
        "producer_version": PRODUCER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "max_input_chars": settings.run_assets_max_input_chars,
        "truncated": truncated,
    }
