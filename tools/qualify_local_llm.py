#!/usr/bin/env python
"""Qualify a bundled CPU-only local LLM against Voxint's real production paths (#66).

This maintainer tool drives Voxint's **unmodified** production code — the real
``HttpLLMClient``, the real run-asset producer, and the real research tool loop
(with the loop's deterministic offline web seams) — against the frozen corpus in
``tests/fixtures/llm_qual/``. It scores every fixture against the six committed
gates (see ``manifest.json``) mechanically, per-repetition, and emits the gate
table plus per-job wall-clock and a token/throughput summary.

The gate CONTRACT is frozen in the manifest before any model output is seen; this
tool only measures. A gate failure is a recorded result, never a silent skip.

The LLM endpoint (base_url/model/key) is host-local and passed by flag or env; it
never appears in the committed corpus. Run examples::

    uv run python tools/qualify_local_llm.py --profile unguarded --reps 3
    uv run python tools/qualify_local_llm.py --jobs run_assets --fixtures summary_near_max
    uv run python tools/qualify_local_llm.py --probe            # one-shot smoke

The ``--profile`` flag is a LABEL recorded in the results: the actual guarding
(server-side ``--n-predict`` cap, ``--json-schema '{}'``) is applied by launching
the llama.cpp container with those flags, then running this tool with the matching
label. The tool reports ``max_completion_tokens`` per job so the guarded
``--n-predict`` cap can be derived from the largest legitimate output.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from voxint.app_settings import resolve_effective_web_research
from voxint.clients.base import EnhancementRequestSegment
from voxint.clients.llm import HttpLLMClient, LLMError, SamplingProfile
from voxint.config import Settings
from voxint.db.models import RunAssetKind
from voxint.enrichment.producers.name_patterns import normalize_name
from voxint.enrichment.producers.run_assets_llm import RunAssetProducerError, generate_payload
from voxint.enrichment.run_assets import RunAssetSource, SegmentSource
from voxint.research.agent import (
    ResearchAgentError,
    ResearchCancelled,
    ResearchSeed,
    RosterMatch,
    run_research_loop,
)
from voxint.research.fetch import ClientFactory
from voxint.research.search import SearchResult

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm_qual"
_LETTER = r"[^\W\d_]"
KIND_BY_NAME = {
    "summary": RunAssetKind.SUMMARY,
    "topics": RunAssetKind.TOPICS,
    "entity_mentions": RunAssetKind.ENTITY_MENTIONS,
}

# Named sampling profiles for the qualification sweep (#67). ``greedy`` is the
# reproducible default the #66 verdict measured (temperature=0); ``qwen`` is the
# profile Qwen's model card recommends. The chosen profile is recorded in the
# results so the pinned #67 serving profile is measured, not assumed.
SAMPLING_PROFILES = {
    "greedy": SamplingProfile(),
    "qwen": SamplingProfile(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),
}


# --------------------------------------------------------------------------- #
# Endpoint config + raw-response capture
# --------------------------------------------------------------------------- #
class EndpointConfig:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        sampling: SamplingProfile | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.sampling = sampling or SamplingProfile()


class Capture:
    """Records every chat-completions response so gates can read the RAW
    ``message.content`` (reply-size bound, raw-protocol adherence) and the
    server's own usage/timings — independent of the production parser."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def hook(self, response: httpx.Response) -> None:
        response.read()
        rec: dict[str, Any] = {"status": response.status_code, "http_bytes": len(response.content)}
        try:
            data = response.json()
            rec["content"] = data["choices"][0]["message"]["content"]
            rec["usage"] = data.get("usage")
            rec["timings"] = data.get("timings")
        except (ValueError, KeyError, IndexError, TypeError):
            rec["content"] = None
        self.calls.append(rec)

    def reset(self) -> None:
        self.calls.clear()

    def content_len(self) -> int | None:
        if not self.calls:
            return None
        content = self.calls[-1].get("content")
        return len(content) if isinstance(content, str) else None

    def completion_tokens(self) -> int:
        total = 0
        for call in self.calls:
            usage = call.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                total += usage["completion_tokens"]
        return total


def build_client(cfg: EndpointConfig, cap: Capture) -> tuple[HttpLLMClient, httpx.Client]:
    raw = httpx.Client(
        base_url=cfg.base_url,
        timeout=httpx.Timeout(connect=10.0, read=cfg.timeout, write=cfg.timeout, pool=cfg.timeout),
        event_hooks={"response": [cap.hook]},
    )
    client = HttpLLMClient(
        cfg.base_url, cfg.model, cfg.api_key, cfg.timeout, client=raw, sampling=cfg.sampling
    )
    return client, raw


def build_settings(cfg: EndpointConfig) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_base_url=cfg.base_url,
        llm_model=cfg.model,
        llm_api_key=cfg.api_key,
        llm_timeout_seconds=cfg.timeout,
        voxint_web_research=True,
        web_search_base_url="http://searx.lan:8888",
    )


# --------------------------------------------------------------------------- #
# Faithfulness scoring helpers (mechanical)
# --------------------------------------------------------------------------- #
def _lex_norm(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    depunct = re.sub(r"[^\w\s]", " ", folded, flags=re.UNICODE)
    return re.sub(r"\s+", " ", depunct).strip()


def _ws_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _prot_norm(text: str) -> str:
    """Normalize a protected-token check to tolerate authorized hyphenation.

    An entity/number that keeps its words but gains or loses a hyphen ("ninety
    two" <-> "ninety-two") is an authorized punctuation edit, not a corruption;
    the protected-token guard is about the WORDS surviving, so hyphens collapse
    to spaces (case preserved, so a real word change still fails).
    """
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip()


def _authorized_edit_forms(source: str, edits: list[dict[str, str]]) -> set[str]:
    """All lex-normalized outputs reachable by applying any SUBSET of edits.

    Models the faithfulness gate as an authorized-edit SUBSET: a faithful
    under-correction (only some of the sanctioned homophone/grammar fixes
    applied, including none) must pass; any UNauthorized change has no reachable
    form and fails. Enumerated over 2**len(edits) subsets (edits are few).
    """
    forms: set[str] = set()
    for mask in range(1 << len(edits)):
        candidate = source
        for i, edit in enumerate(edits):
            if mask & (1 << i):
                candidate = candidate.replace(edit["from"], edit["to"])
        forms.add(_lex_norm(candidate))
    return forms


def score_segment(out: str, source: str, gold: dict[str, Any]) -> list[str]:
    """Return a list of failure reasons ([] == pass) for one enhancement segment."""
    fails: list[str] = []
    mode = gold["mode"]
    if mode == "byte_identical":
        if out != source:
            fails.append("not byte-identical")
    elif mode == "lexical_identical":
        if _lex_norm(out) != _lex_norm(source):
            fails.append("lexical tokens changed")
    elif mode == "allowed_variants":
        allowed = {_ws_norm(source), *(_ws_norm(v) for v in gold.get("allowed_variants", []))}
        if _ws_norm(out) not in allowed:
            fails.append("not the source nor an allowed variant")
    elif mode == "authorized_edits":
        if _lex_norm(out) not in _authorized_edit_forms(source, gold.get("edits", [])):
            fails.append("not the source nor an authorized-edit subset")
    else:  # pragma: no cover - manifest is frozen
        fails.append(f"unknown gold mode {mode!r}")
    for token in gold.get("protected_tokens", []):
        if _prot_norm(token) not in _prot_norm(out):
            fails.append(f"protected token {token!r} lost")
    return fails


# --------------------------------------------------------------------------- #
# Name-hint verbatim location (mirrors names_llm._locate)
# --------------------------------------------------------------------------- #
def _locate(name: str, segments: list[dict[str, Any]], label: str | None) -> bool:
    needle = unicodedata.normalize("NFKC", name).casefold()
    pattern = re.compile(rf"(?<!{_LETTER}){re.escape(needle)}(?!{_LETTER})", re.UNICODE)
    for seg in segments:
        if label is not None and seg.get("label") != label:
            continue
        haystack = unicodedata.normalize("NFKC", seg["text"]).casefold()
        if pattern.search(haystack):
            return True
    return False


def _name_eq(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def survivors_from_hints(
    name_hints: Sequence[Any], segments: list[dict[str, Any]]
) -> list[tuple[str, str, str]]:
    survivors: list[tuple[str, str, str]] = []
    for hint in name_hints:
        norm = normalize_name(hint.name)
        if norm is None:
            continue
        label = hint.diarization_label if hint.kind == "self" else None
        if _locate(norm, segments, label):
            survivors.append((hint.diarization_label, norm, hint.kind))
    return survivors


# --------------------------------------------------------------------------- #
# Research offline seams (copied from tests/unit/test_research_agent.py)
# --------------------------------------------------------------------------- #
class FakeProvider:
    name = "fake"
    dropped_last = 0

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        return list(self._results)[:max_results]


def resolver_map(table: dict[str, list[str]]) -> Callable[..., Any]:
    def resolve(host: str, *args: object, **kwargs: object) -> list[Any]:
        if host not in table:
            raise socket.gaierror("no such host")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0)) for addr in table[host]]

    return resolve


def page_factory(body: str, fetched_hosts: list[str]) -> ClientFactory:
    def handler(request: httpx.Request) -> httpx.Response:
        fetched_hosts.append(request.headers["host"])
        return httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=iter([body.encode()]),
        )

    def factory(timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
            trust_env=False,
            timeout=timeout,
        )

    return factory


# --------------------------------------------------------------------------- #
# Per-job runners: one repetition -> a per-rep record (raw failure reasons)
# --------------------------------------------------------------------------- #
def run_enhancement(
    fx: dict[str, Any], client: HttpLLMClient, cap: Capture, is_names: bool
) -> dict[str, Any]:
    cap.reset()
    segments = tuple(
        EnhancementRequestSegment(
            segment_index=s["index"], text=s["text"], diarization_label=s.get("label")
        )
        for s in fx["segments"]
    )
    gold = fx["gold"]
    rec: dict[str, Any] = {"gates": {}}
    start = time.perf_counter()
    try:
        result = client.enhance_segments(
            segments,
            fx.get("context", ""),
            name_attribution_context=fx.get("name_attribution_context", ""),
        )
    except LLMError as exc:
        rec["elapsed"] = time.perf_counter() - start
        rec["gates"]["structural_validity"] = [f"LLMError: {exc}"]
        rec["error"] = str(exc)
        return rec
    rec["elapsed"] = time.perf_counter() - start
    rec["completion_tokens"] = cap.completion_tokens()
    rec["gates"]["structural_validity"] = []

    # Faithfulness (per-index, against each segment's OWN source) + reply-size bound.
    faith: list[str] = []
    for seg in fx["segments"]:
        idx = seg["index"]
        gseg = gold["segments"][str(idx)]
        out = result.enhanced.get(idx)
        if out is None:
            faith.append(f"segment {idx} missing")
            continue
        faith.extend(f"segment {idx}: {r}" for r in score_segment(out, seg["text"], gseg))
    content_len = cap.content_len()
    cap_bound = gold.get("reply_size_max_chars")
    if content_len is not None and cap_bound is not None and content_len > cap_bound:
        faith.append(f"raw reply {content_len} chars over {cap_bound} bound")
    rec["gates"]["faithfulness"] = faith

    survivors = survivors_from_hints(result.name_hints, fx["segments"])
    rec["survivors"] = survivors
    if is_names:
        rec["gates"]["semantic_usefulness"] = score_names(survivors, gold)
        rec["gates"]["grounding"] = []  # location IS the grounding; no ungrounded hint persists
    else:
        # enhancement fixtures forbid any surviving name hint
        rec["gates"]["semantic_usefulness"] = (
            [] if not survivors else [f"unexpected name hints: {survivors}"]
        )
    return rec


def score_names(survivors: list[tuple[str, str, str]], gold: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    matched_idx: set[int] = set()
    for exp in gold.get("expected_name_hints", []):
        want = normalize_name(exp["name"]) or exp["name"]
        found = False
        for i, (s_label, s_name, s_kind) in enumerate(survivors):
            if s_kind != exp["kind"] or not _name_eq(s_name, want):
                continue
            if exp["kind"] == "self" and s_label != exp["label"]:
                continue
            found = True
            matched_idx.add(i)
            break
        if not found:
            fails.append(f"missing expected {exp['kind']} hint {exp['name']!r}")
    for i, surv in enumerate(survivors):
        if i not in matched_idx:
            fails.append(f"false-positive hint {surv}")
    return fails


def build_source(fx: dict[str, Any]) -> RunAssetSource:
    src = fx["source"]
    if "synthesize" in src:
        segments = synthesize_segments(src["synthesize"])
    else:
        segments = [
            SegmentSource(segment_index=s["segment_index"], speaker=s["speaker"], text=s["text"])
            for s in src["segments"]
        ]
    return RunAssetSource(
        pipeline_run_id=uuid.UUID(int=0),
        segments=tuple(segments),
        metadata=src.get("metadata"),
        operator_notes=src.get("operator_notes"),
    )


def synthesize_segments(spec: dict[str, Any]) -> list[SegmentSource]:
    speakers = spec["speakers"]
    pool = spec["sentence_pool"]
    target = spec["target_text_chars"]
    segments = [SegmentSource(segment_index=0, speaker=speakers[0], text=spec["head_anchor"])]
    total = len(spec["head_anchor"])
    i = 1
    while total < target:
        sentence = pool[(i - 1) % len(pool)]
        segments.append(
            SegmentSource(segment_index=i, speaker=speakers[i % len(speakers)], text=sentence)
        )
        total += len(sentence)
        i += 1
    segments.append(
        SegmentSource(
            segment_index=i, speaker=speakers[i % len(speakers)], text=spec["tail_anchor"]
        )
    )
    return segments


def run_run_asset(
    fx: dict[str, Any], client: HttpLLMClient, cap: Capture, settings: Settings
) -> dict[str, Any]:
    cap.reset()
    gold = fx["gold"]
    kind = KIND_BY_NAME[fx["kind"]]
    source = build_source(fx)
    rec: dict[str, Any] = {"gates": {}}
    start = time.perf_counter()
    try:
        payload, truncated = generate_payload(client, kind, source, settings=settings)
    except (LLMError, RunAssetProducerError) as exc:
        rec["elapsed"] = time.perf_counter() - start
        if gold.get("expect_success", True):
            rec["gates"]["structural_validity"] = [f"{type(exc).__name__}: {exc}"]
            rec["error"] = str(exc)
        else:
            rec["gates"]["bounded_failure"] = []  # a caught, bounded failure is the expected shape
        return rec
    rec["elapsed"] = time.perf_counter() - start
    rec["completion_tokens"] = cap.completion_tokens()
    rec["truncated"] = truncated
    rec["gates"]["structural_validity"] = []
    rec["gates"]["semantic_usefulness"] = score_run_asset_semantics(fx["kind"], payload, gold)
    rec["gates"]["grounding"] = score_run_asset_grounding(fx["kind"], payload, source)
    threshold = gold.get("latency_threshold_seconds")
    if threshold is not None:
        rec["gates"]["latency"] = (
            [] if rec["elapsed"] <= threshold else [f"{rec['elapsed']:.1f}s over {threshold}s"]
        )
    return rec


def score_run_asset_semantics(
    kind: str, payload: dict[str, Any], gold: dict[str, Any]
) -> list[str]:
    fails: list[str] = []
    if kind == "summary":
        text = payload["summary"].casefold()
        for fact in gold.get("required_facts", []):
            if fact.casefold() not in text:
                fails.append(f"required fact {fact!r} absent from summary")
        for bad in gold.get("forbidden_hallucinations", []):
            if bad.casefold() in text:
                fails.append(f"forbidden {bad!r} present in summary")
    elif kind == "topics":
        labels = [t["label"] for t in payload["topics"]]
        descs = [t.get("description") or "" for t in payload["topics"]]
        blob = " ".join(labels + descs).casefold()
        for req in gold.get("required_topic_labels", []):
            if not any(req.casefold() in label.casefold() for label in labels):
                fails.append(f"no topic label contains {req!r}")
        for bad in gold.get("forbidden_hallucinations", []):
            if bad.casefold() in blob:
                fails.append(f"forbidden {bad!r} present in topics")
    else:  # entity_mentions
        surfaces = [m["surface"].casefold() for m in payload["mentions"]]
        for exp in gold.get("expected_entities", []):
            want = exp["surface"].casefold()
            if not any(want in surf or surf in want for surf in surfaces):
                fails.append(f"expected entity {exp['surface']!r} not recalled")
        for bad in gold.get("forbidden_entities", []):
            if bad["surface"] == "any":
                if surfaces:
                    fails.append(f"entities present where none exist: {surfaces}")
                continue
            want = bad["surface"].casefold()
            if any(want == surf for surf in surfaces):
                fails.append(f"forbidden/decoy entity {bad['surface']!r} reported as mention")
    return fails


def score_run_asset_grounding(
    kind: str, payload: dict[str, Any], source: RunAssetSource
) -> list[str]:
    if kind != "entity_mentions":
        return []  # summary/topics are gated semantically, not server-grounded
    fails: list[str] = []
    text_by_index = {seg.segment_index: seg.text for seg in source.segments}
    for mention in payload["mentions"]:
        for occ in mention["occurrences"]:
            seg_text = text_by_index.get(occ["segment_index"], "")
            if seg_text[occ["start_char"] : occ["end_char"]] != occ["quote"]:
                fails.append(f"quote for {mention['surface']!r} not re-sliced from text")
    return fails


def run_research(
    fx: dict[str, Any], client: HttpLLMClient, cap: Capture, settings: Settings
) -> dict[str, Any]:
    cap.reset()
    gold = fx["gold"]
    seed_spec = fx["seed"]
    seed = ResearchSeed(
        display_name=seed_spec["display_name"],
        candidate_names=tuple(seed_spec.get("candidate_names", [])),
        context_lines=tuple(seed_spec.get("context_lines", [])),
        seed_urls=tuple(seed_spec.get("seed_urls", [])),
        operator_note=seed_spec.get("operator_note"),
    )
    provider = FakeProvider([SearchResult(**r) for r in fx["search_results"]])
    fetched: list[str] = []
    factory = page_factory(fx["page"], fetched)
    resolver = resolver_map(fx["resolver"])
    roster = [
        RosterMatch(
            speaker_id=uuid.uuid4(),
            display_name=r["display_name"],
            is_target=r.get("is_target", False),
        )
        for r in fx.get("roster", [])
    ]

    def roster_lookup(_query: str) -> list[RosterMatch]:
        return roster

    rec: dict[str, Any] = {"gates": {}}
    start = time.perf_counter()
    try:
        conclusion = run_research_loop(
            llm=client,
            settings=settings,
            effective_web=resolve_effective_web_research(None, settings),
            seed=seed,
            roster_lookup=roster_lookup,
            search_provider=provider,
            read_client_factory=factory,
            read_resolver=resolver,
        )
    except (ResearchAgentError, ResearchCancelled) as exc:
        rec["elapsed"] = time.perf_counter() - start
        rec["fetched_hosts"] = fetched
        rec["completion_tokens"] = cap.completion_tokens()
        rec["llm_calls"] = len(cap.calls)
        if gold.get("expect_bounded_failure"):
            rec["gates"]["bounded_failure"] = []
        else:
            rec["gates"]["structural_validity"] = [f"{type(exc).__name__}: {exc}"]
            rec["error"] = str(exc)
        # An injected fetch is a hard fail even inside a raise.
        bad = [h for h in gold.get("forbidden_fetch_hosts", []) if h in fetched]
        rec["gates"]["bounded_failure"] = rec["gates"].get("bounded_failure", []) + (
            [f"fetched forbidden host(s) {bad}"] if bad else []
        )
        return rec
    rec["elapsed"] = time.perf_counter() - start
    rec["fetched_hosts"] = fetched
    rec["completion_tokens"] = cap.completion_tokens()
    rec["llm_calls"] = len(cap.calls)
    rec["found"] = conclusion.found
    rec["n_claims"] = len(conclusion.claims)
    rec["dropped_claims"] = conclusion.dropped_claims
    rec["gates"]["structural_validity"] = []

    semantic: list[str] = []
    if gold.get("expected_found") is not None and conclusion.found != gold["expected_found"]:
        semantic.append(f"found={conclusion.found}, expected {gold['expected_found']}")
    if len(conclusion.claims) < gold.get("expected_min_grounded_claims", 0):
        semantic.append(f"{len(conclusion.claims)} grounded claims < min")
    for sub in gold.get("required_claim_substrings", []):
        if not any(sub.casefold() in c.value.casefold() for c in conclusion.claims):
            semantic.append(f"no grounded claim contains {sub!r}")
    if gold.get("must_investigate") and (conclusion.searches_used + conclusion.reads_used) < 1:
        semantic.append("concluded without any successful investigation")
    rec["gates"]["semantic_usefulness"] = semantic

    bounded = [
        f"fetched forbidden host {h}"
        for h in gold.get("forbidden_fetch_hosts", [])
        if h in fetched
    ]
    rec["gates"]["bounded_failure"] = bounded
    rec["gates"]["grounding"] = []  # every returned claim is server-grounded by construction
    return rec


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_fixtures(job: str) -> list[dict[str, Any]]:
    files = sorted((FIXTURES_DIR / job).glob("*.json"))
    return [json.loads(f.read_text()) for f in files]


def run_fixture_reps(
    job: str,
    fx: dict[str, Any],
    client: HttpLLMClient,
    cap: Capture,
    settings: Settings,
    reps: int,
) -> dict[str, Any]:
    per_rep: list[dict[str, Any]] = []
    for _ in range(reps):
        if job == "enhancement":
            per_rep.append(run_enhancement(fx, client, cap, is_names=False))
        elif job == "names":
            per_rep.append(run_enhancement(fx, client, cap, is_names=True))
        elif job == "run_assets":
            per_rep.append(run_run_asset(fx, client, cap, settings))
        else:
            per_rep.append(run_research(fx, client, cap, settings))
    return summarize_fixture(fx, per_rep)


def summarize_fixture(fx: dict[str, Any], per_rep: list[dict[str, Any]]) -> dict[str, Any]:
    # A gate PASSES only if it passed (empty failure list) in EVERY rep it was scored.
    gate_names: set[str] = set()
    for rep in per_rep:
        gate_names.update(rep["gates"])
    gates: dict[str, dict[str, Any]] = {}
    for gate in gate_names:
        reasons: list[str] = []
        scored = 0
        for rep in per_rep:
            if gate in rep["gates"]:
                scored += 1
                reasons.extend(rep["gates"][gate])
        gates[gate] = {
            "pass": not reasons,
            "scored_reps": scored,
            "reasons": sorted(set(reasons))[:8],
        }
    elapsed = [rep["elapsed"] for rep in per_rep if "elapsed" in rep]
    tokens = [rep.get("completion_tokens", 0) for rep in per_rep]
    return {
        "id": fx["id"],
        "pass": all(g["pass"] for g in gates.values()),
        "gates": gates,
        "latency": {
            "p50": round(median(elapsed), 2) if elapsed else None,
            "worst": round(max(elapsed), 2) if elapsed else None,
        },
        "max_completion_tokens": max(tokens) if tokens else 0,
        "reps": per_rep,
    }


def probe(client: HttpLLMClient, cap: Capture) -> None:
    cap.reset()
    seg = (EnhancementRequestSegment(segment_index=0, text="hello world", diarization_label=None),)
    start = time.perf_counter()
    result = client.enhance_segments(seg, "")
    elapsed = time.perf_counter() - start
    print(f"probe ok: {elapsed:.2f}s round-trip, enhanced={result.enhanced!r}")
    print(f"  raw content_len={cap.content_len()}, completion_tokens={cap.completion_tokens()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18099/v1")
    parser.add_argument("--model", default="granite-4.0-h-tiny")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file", default=str(Path.home() / "voxint-qual" / ".api_key"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--profile", choices=["unguarded", "guarded"], default="unguarded")
    parser.add_argument(
        "--sampling",
        choices=sorted(SAMPLING_PROFILES),
        default="greedy",
        help="sampling profile sent on every request (greedy=temp0, qwen=card-recommended)",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--jobs", default="enhancement,names,run_assets,research")
    parser.add_argument("--fixtures", default="", help="comma-separated fixture ids to filter")
    parser.add_argument("--out", default="")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key
    if api_key is None and Path(args.api_key_file).is_file():
        api_key = Path(args.api_key_file).read_text().strip()
    cfg = EndpointConfig(
        args.base_url, args.model, api_key or "", args.timeout, SAMPLING_PROFILES[args.sampling]
    )
    cap = Capture()
    client, raw = build_client(cfg, cap)
    settings = build_settings(cfg)

    try:
        if args.probe:
            probe(client, cap)
            return
        only = {f for f in args.fixtures.split(",") if f}
        results: dict[str, Any] = {
            "profile": args.profile,
            "sampling": args.sampling,
            "sampling_params": SAMPLING_PROFILES[args.sampling].as_payload(),
            "reps": args.reps,
            "model": args.model,
            "base_url": args.base_url,
            "jobs": {},
        }
        for job in [j for j in args.jobs.split(",") if j]:
            job_out: dict[str, Any] = {"fixtures": {}}
            for fx in load_fixtures(job):
                if only and fx["id"] not in only:
                    continue
                summary = run_fixture_reps(job, fx, client, cap, settings, args.reps)
                job_out["fixtures"][fx["id"]] = summary
                verdict = "PASS" if summary["pass"] else "FAIL"
                failed_gates = [g for g, v in summary["gates"].items() if not v["pass"]]
                print(
                    f"[{job}/{fx['id']}] {verdict}  "
                    f"p50={summary['latency']['p50']}s worst={summary['latency']['worst']}s "
                    f"maxtok={summary['max_completion_tokens']}"
                    + (f"  FAILED: {failed_gates}" if failed_gates else "")
                )
                for gate in failed_gates:
                    for reason in summary["gates"][gate]["reasons"]:
                        print(f"      - {gate}: {reason}")
            job_out["max_completion_tokens"] = max(
                (f["max_completion_tokens"] for f in job_out["fixtures"].values()), default=0
            )
            results["jobs"][job] = job_out

        out_path = Path(args.out) if args.out else (
            Path("/tmp") / f"voxint-qual-{args.profile}-{args.sampling}.json"
        )
        out_path.write_text(json.dumps(results, indent=2, default=str))
        total = sum(len(j["fixtures"]) for j in results["jobs"].values())
        passed = sum(
            1 for j in results["jobs"].values() for f in j["fixtures"].values() if f["pass"]
        )
        print(f"\n=== {passed}/{total} fixtures PASS ({args.profile}) -> {out_path} ===")
    finally:
        raw.close()


if __name__ == "__main__":
    main()
