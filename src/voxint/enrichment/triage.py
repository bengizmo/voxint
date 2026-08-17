"""Read-time draft triage: an explainable, multi-signal review-priority score (#42).

Pure and deterministic — no DB, no ORM, no I/O — so it unit-tests like
``producers/name_scoring.py``. The caller (``api/app.py``) loads candidates,
evidence, and the per-label cosine facts, then asks this module to fuse them
into a :class:`TriageScore` used **only** to order drafts and populate the
``unresolved`` bucket. Never stored, never exported, never auto-accepts, and
never compared across surfaces or producers-as-truth.

Doctrine (mirrors the name scorer): weights are module constants; the score is
**explainable, not calibrated**; every value is capped below 1.0 so nothing ever
reads as certain; components are a flat ``{name: float}`` map surfaced verbatim
to the operator. Changing any weight, cap, or component key changes ordering and
the audit trail, so **bump :data:`TRIAGE_VERSION`**.

Signals fuse **within a single surface/field only** (name candidates ranked
among themselves; profile candidates among themselves). This respects the
documented rule that per-producer ``score``/``score_components`` are *not*
comparable across producers: NAME name-match runs through per-producer adapters
(never a raw cross-producer ``score``), and priorities from different fields are
never placed on one scale.

Score families and their (fixed) component keys:

* **NAME** — ``name_match`` (per-producer adapter), ``voice_support`` (a grounded
  cosine that names the *same* identity), ``voice_conflict`` (a grounded cosine
  that names a *different* identity — visible, lightly demoting, never a boost),
  ``cross_source_agreement`` (distinct producers proposing the same name),
  ``peer_producer_count``.
* **PROFILE** (bio/affiliation/link) — ``independent_domains`` (distinct
  registrable domains citing the value, saturating), ``source_authority``
  (fraction of those domains on the operator allowlist), ``corroborated`` (a
  visible flag: ≥2 distinct domains agree — profile corroboration *is* the
  independent-domain count, so it is shown but not separately weighted, to avoid
  double-counting), ``distinct_domains_count``. BIO carries no cross-producer
  agreement term (exact equality never occurs; semantic similarity would be
  opaque inference).
"""

import ipaddress
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

TRIAGE_VERSION = 1

# Nothing ever reads as certain: the top achievable priority.
PRIORITY_CAP = 0.95

# Per-family weights. Each vector sums to 1.0 and every component is in [0, 1],
# so a raw weighted sum is in [0, 1] and, times PRIORITY_CAP, never exceeds it.
# A component that does not apply contributes 0.0 — weights are NEVER
# renormalized, because rewarding missing evidence would invert the signal.
NAME_W = {"name_match": 0.60, "voice_support": 0.25, "cross_source_agreement": 0.15}
PROFILE_W = {"independent_domains": 0.50, "source_authority": 0.50}

# A grounded cosine that names a different roster identity than the candidate's
# name subtracts this from the priority — a soft demotion, never negative.
VOICE_CONFLICT_PENALTY = 0.10

# cross_source_agreement steps: 1 producer -> 0.0, 2 -> 0.5, 3+ -> 1.0.
AGREEMENT_STEP = 0.5

# independent_domains saturates here: 3 distinct registrable domains -> 1.0.
INDEPENDENT_DOMAINS_TARGET = 3

# The LLM name producer emits a flat, uncalibrated marker score; its name-match
# contribution is a fixed modest constant rather than that raw 0.5 (which is not
# comparable to the offline producer's reliability).
LLM_NAME_MATCH = 0.5

# Producer logical keys (mirror the producer modules; kept as literals so this
# module imports nothing DB-bound). Offline exposes a rich, documented
# `base` reliability component; the LLM producer a flat marker.
_OFFLINE_PRODUCER = "names.offline"
_LLM_PRODUCER = "names.llm"

# ClaimField.NAME value (mirrored as a literal to avoid importing the ORM);
# every other field routes to the profile family.
_NAME = "name"

# EvidenceKind.URL value.
_URL_KIND = "url"

# Bounded, reviewable set of common multi-label public suffixes. Deliberately
# NOT a full Public Suffix List (that is a data-bearing dependency needing
# refresh — rejected under the project's anti-bloat stance). A registrable
# domain under a rare multi-level ccTLD absent here collapses to its last two
# labels, which can only nudge ordering — it never corrupts stored data.
MULTI_PART_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk",
        "co.jp", "or.jp", "ne.jp", "go.jp",
        "com.au", "net.au", "org.au", "gov.au", "edu.au",
        "co.nz", "org.nz", "govt.nz",
        "co.za", "org.za",
        "co.in", "net.in", "org.in", "gov.in",
        "com.br", "net.br", "org.br", "gov.br",
        "com.cn", "net.cn", "org.cn", "gov.cn",
        "com.mx", "com.sg", "com.hk", "com.tr", "com.tw",
    }
)

# Stable, per-family component-key sets (contract-pinned). A family always emits
# exactly these keys, present or zero, so the operator UI and ordering are
# schema-stable across drafts.
NAME_COMPONENT_KEYS = frozenset(
    {
        "name_match",
        "voice_support",
        "voice_conflict",
        "cross_source_agreement",
        "peer_producer_count",
    }
)
PROFILE_COMPONENT_KEYS = frozenset(
    {"independent_domains", "source_authority", "corroborated", "distinct_domains_count"}
)


@dataclass(frozen=True)
class TriageScore:
    """A review-priority in [0, PRIORITY_CAP] plus the explainable breakdown."""

    priority: float
    components: dict[str, float]


@dataclass(frozen=True)
class EvidenceRef:
    """The primitive view of one evidence row the scorer needs."""

    kind: str
    url: str | None


@dataclass(frozen=True)
class VoiceSignal:
    """The persisted cosine facts for a candidate's (run, label), if any.

    Only ``confidence`` (transformed cosine) and ``grounded`` survive to
    ``speaker_assignments`` — margin/vote-agreement are not columned.
    ``matches_value`` is whether the assignment's roster speaker name matches the
    candidate's proposed name; a grounded mismatch is a *conflict*, not support.
    """

    matches_value: bool
    grounded: bool
    confidence: float | None


@dataclass(frozen=True)
class TriageInputs:
    """Everything the pure scorer reads for one candidate."""

    field: str
    producer: str
    producer_score: float | None
    producer_components: Mapping[str, float]
    evidence: tuple[EvidenceRef, ...]
    voice: VoiceSignal | None
    peer_producer_count: int
    authority_domains: frozenset[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _registrable_from_labels(labels: Sequence[str]) -> str:
    if len(labels) <= 2:
        return ".".join(labels)
    last_two = ".".join(labels[-2:])
    if last_two in MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return last_two


def registrable_domain(url: str) -> str | None:
    """The registrable domain of ``url`` (bounded best-effort; see module doc).

    Collapses subdomains (``blog.example.com`` -> ``example.com``), keeps the
    third label under a known multi-part suffix (``foo.co.uk`` -> ``foo.co.uk``),
    and returns an IP literal unchanged. ``None`` when there is no usable host.
    """
    try:
        host = urlsplit(url).hostname  # already lowercased; IPv6 brackets stripped
    except ValueError:
        return None  # malformed authority (e.g. an unclosed IPv6 bracket)
    if not host:
        return None
    host = host.strip(".")
    if not host:
        return None
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    if any(not label for label in labels):
        return None  # empty label ("example..com") — not a real domain
    return _registrable_from_labels(labels)


def normalize_authority_domain(raw: str) -> str | None:
    """Normalize one operator-allowlist entry to a registrable domain.

    Accepts a bare domain only. Rejects entries carrying a scheme, path, port,
    credentials, wildcard, or otherwise malformed labels — the allowlist counts
    *domains*, never URLs. ``www.example.com`` and ``example.com`` both reduce to
    ``example.com`` (so an allowlisted registrable domain covers its subdomains).

    No IDN/punycode conversion is done (stdlib-only, anti-bloat): an
    internationalized domain must be entered in the same form the evidence hosts
    are served in — in practice **punycode** (``xn--…``), since fetched-page
    hostnames arrive already encoded. Entering the Unicode form would be accepted
    but never match. Documented in ``.env.example``.
    """
    candidate = raw.strip().lower().rstrip(".")
    if not candidate:
        return None
    if any(ch in candidate for ch in "/@*?#: \t") or ".." in candidate or candidate.startswith("."):
        return None
    labels = candidate.split(".")
    if len(labels) < 2:
        return None
    for label in labels:
        if not label or not all(ch.isalnum() or ch == "-" for ch in label):
            return None
    return _registrable_from_labels(labels)


def parse_authority_domains(raw: str) -> frozenset[str]:
    """Parse a comma/whitespace-separated allowlist string into registrable
    domains, dropping malformed entries. Empty input -> empty set (the score
    then reads 0.0 for every candidate: feature-neutral until configured)."""
    tokens = raw.replace(",", " ").split()
    return frozenset(d for token in tokens if (d := normalize_authority_domain(token)))


def _agreement(peer_producer_count: int) -> float:
    return _clamp01(max(0, peer_producer_count - 1) * AGREEMENT_STEP)


def _distinct_domains(evidence: Sequence[EvidenceRef]) -> set[str]:
    domains: set[str] = set()
    for ref in evidence:
        if ref.kind == _URL_KIND and ref.url:
            domain = registrable_domain(ref.url)
            if domain is not None:
                domains.add(domain)
    return domains


def _name_match(producer: str, score: float | None, components: Mapping[str, float]) -> float:
    """Per-producer name-match strength in [0, 1] — never a raw cross-producer
    ``score`` (those are not comparable). Offline uses its documented ``base``
    reliability; the LLM producer a fixed modest constant; unknown producers
    contribute nothing."""
    if producer == _OFFLINE_PRODUCER:
        base = components.get("base")
        return _clamp01(base if base is not None else (score or 0.0))
    if producer == _LLM_PRODUCER:
        return LLM_NAME_MATCH
    return 0.0


def _voice_terms(voice: VoiceSignal | None) -> tuple[float, float]:
    """(support, conflict). Support only for a grounded cosine naming the same
    identity; conflict only for a grounded cosine naming a different one."""
    if voice is None or not voice.grounded:
        return 0.0, 0.0
    if voice.matches_value:
        return _clamp01(voice.confidence or 0.0), 0.0
    return 0.0, 1.0


def score_name(inp: TriageInputs) -> TriageScore:
    name_match = _name_match(inp.producer, inp.producer_score, inp.producer_components)
    support, conflict = _voice_terms(inp.voice)
    agreement = _agreement(inp.peer_producer_count)
    weighted = (
        NAME_W["name_match"] * name_match
        + NAME_W["voice_support"] * support
        + NAME_W["cross_source_agreement"] * agreement
    )
    priority = min(PRIORITY_CAP, PRIORITY_CAP * weighted)
    priority = max(0.0, priority - VOICE_CONFLICT_PENALTY * conflict)
    return TriageScore(
        priority=round(priority, 4),
        components={
            "name_match": round(name_match, 4),
            "voice_support": round(support, 4),
            "voice_conflict": conflict,
            "cross_source_agreement": round(agreement, 4),
            "peer_producer_count": float(inp.peer_producer_count),
        },
    )


def score_profile(inp: TriageInputs) -> TriageScore:
    domains = _distinct_domains(inp.evidence)
    count = len(domains)
    independent = _clamp01(count / INDEPENDENT_DOMAINS_TARGET)
    authority = (len(domains & inp.authority_domains) / count) if count else 0.0
    weighted = (
        PROFILE_W["independent_domains"] * independent
        + PROFILE_W["source_authority"] * authority
    )
    priority = min(PRIORITY_CAP, PRIORITY_CAP * weighted)
    return TriageScore(
        priority=round(priority, 4),
        components={
            "independent_domains": round(independent, 4),
            "source_authority": round(authority, 4),
            # Corroboration for a profile value IS multiple independent domains
            # agreeing on it: shown to the operator, weighted via
            # independent_domains (not a second term). BIO included — the flag is
            # honest even where a weighted agreement term would not be.
            "corroborated": 1.0 if count >= 2 else 0.0,
            "distinct_domains_count": float(count),
        },
    )


def score(inp: TriageInputs) -> TriageScore:
    """Dispatch on claim field: NAME -> name family, else profile family."""
    if inp.field == _NAME:
        return score_name(inp)
    return score_profile(inp)
