"""Pure unit tests for the draft-triage scorer (#42) — no DB, no I/O.

Mirrors ``test_name_scoring.py``: exercises each component's mapping, the
per-family fusion and caps, the registrable-domain heuristic, the authority
allowlist parser, and the stable per-family component-key sets.
"""

import pytest

from voxint.enrichment import triage
from voxint.enrichment.triage import (
    NAME_COMPONENT_KEYS,
    PRIORITY_CAP,
    PROFILE_COMPONENT_KEYS,
    EvidenceRef,
    TriageInputs,
    VoiceSignal,
    normalize_authority_domain,
    parse_authority_domains,
    registrable_domain,
)


def _url(u: str) -> EvidenceRef:
    return EvidenceRef(kind="url", url=u)


def name_inputs(**over: object) -> TriageInputs:
    base: dict[str, object] = {
        "field": "name",
        "producer": "names.offline",
        "producer_score": 0.9,
        "producer_components": {"base": 0.8},
        "evidence": (),
        "voice": None,
        "peer_producer_count": 1,
        "authority_domains": frozenset(),
    }
    base.update(over)
    return TriageInputs(**base)  # type: ignore[arg-type]


def profile_inputs(**over: object) -> TriageInputs:
    base: dict[str, object] = {
        "field": "affiliation",
        "producer": "web_researcher",
        "producer_score": 0.5,
        "producer_components": {"web": 1.0},
        "evidence": (),
        "voice": None,
        "peer_producer_count": 1,
        "authority_domains": frozenset(),
    }
    base.update(over)
    return TriageInputs(**base)  # type: ignore[arg-type]


# --- registrable_domain ---------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/x", "example.com"),
        ("https://blog.example.com/x", "example.com"),  # subdomain collapse
        ("https://a.b.example.com", "example.com"),
        ("https://foo.co.uk/x", "foo.co.uk"),  # multi-part suffix keeps 3 labels
        ("https://sub.foo.co.uk/x", "foo.co.uk"),
        ("https://EXAMPLE.COM/x", "example.com"),  # lowercased
        ("https://example.com./x", "example.com"),  # trailing dot
        ("https://93.184.216.34/x", "93.184.216.34"),  # IPv4 literal
        ("https://[2606:2800:220:1:248:1893:25c8:1946]/x", "2606:2800:220:1:248:1893:25c8:1946"),
        ("https://localhost/x", "localhost"),
        ("not a url", None),
        ("mailto:a@b.com", None),
        ("", None),
    ],
)
def test_registrable_domain(url: str, expected: str | None) -> None:
    assert registrable_domain(url) == expected


# --- authority allowlist parsing ------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("example.com", "example.com"),
        ("WWW.Example.COM", "example.com"),  # www + case collapse
        ("sub.foo.co.uk", "foo.co.uk"),
        ("example.com.", "example.com"),  # trailing dot
        ("https://example.com", None),  # scheme rejected
        ("example.com/path", None),  # path rejected
        ("example.com:8080", None),  # port rejected
        ("*.example.com", None),  # wildcard rejected
        ("user@example.com", None),  # credentials rejected
        ("localhost", None),  # single label rejected
        ("", None),
        ("a..b.com", None),  # empty label rejected
    ],
)
def test_normalize_authority_domain(raw: str, expected: str | None) -> None:
    assert normalize_authority_domain(raw) == expected


def test_parse_authority_domains_splits_and_drops_bad() -> None:
    parsed = parse_authority_domains("example.com, foo.org  bad/entry\nblog.example.com")
    # blog.example.com collapses to example.com; bad/entry dropped.
    assert parsed == frozenset({"example.com", "foo.org"})


def test_parse_authority_domains_empty() -> None:
    assert parse_authority_domains("   ") == frozenset()


# --- NAME family ----------------------------------------------------------


def test_name_match_offline_uses_base_not_capped_score() -> None:
    s = triage.score(name_inputs(producer_score=0.95, producer_components={"base": 0.6}))
    assert s.components["name_match"] == 0.6


def test_name_match_offline_falls_back_to_score_without_base() -> None:
    s = triage.score(name_inputs(producer_components={}, producer_score=0.7))
    assert s.components["name_match"] == 0.7


def test_name_match_llm_is_fixed_constant() -> None:
    inp = name_inputs(producer="names.llm", producer_score=0.5, producer_components={"llm": 1.0})
    assert triage.score(inp).components["name_match"] == triage.LLM_NAME_MATCH


def test_name_match_unknown_producer_is_zero() -> None:
    s = triage.score(name_inputs(producer="mystery", producer_components={}))
    assert s.components["name_match"] == 0.0


def test_voice_support_only_when_grounded_and_matching() -> None:
    voice = VoiceSignal(matches_value=True, grounded=True, confidence=0.88)
    s = triage.score(name_inputs(voice=voice))
    assert s.components["voice_support"] == 0.88
    assert s.components["voice_conflict"] == 0.0


def test_ungrounded_voice_is_neutral() -> None:
    voice = VoiceSignal(matches_value=True, grounded=False, confidence=0.88)
    s = triage.score(name_inputs(voice=voice))
    assert s.components["voice_support"] == 0.0
    assert s.components["voice_conflict"] == 0.0


def test_grounded_voice_naming_other_identity_is_conflict_and_demotes() -> None:
    match = triage.score(name_inputs(voice=None))
    conflict_voice = VoiceSignal(matches_value=False, grounded=True, confidence=0.9)
    conflict = triage.score(name_inputs(voice=conflict_voice))
    assert conflict.components["voice_conflict"] == 1.0
    assert conflict.components["voice_support"] == 0.0
    assert conflict.priority == round(max(0.0, match.priority - triage.VOICE_CONFLICT_PENALTY), 4)


def test_cross_source_agreement_steps() -> None:
    def agreement(n: int) -> float:
        return triage.score(name_inputs(peer_producer_count=n)).components["cross_source_agreement"]

    assert agreement(1) == 0.0
    assert agreement(2) == 0.5
    assert agreement(5) == 1.0


def test_name_priority_never_exceeds_cap() -> None:
    voice = VoiceSignal(matches_value=True, grounded=True, confidence=1.0)
    s = triage.score(
        name_inputs(producer_components={"base": 1.0}, voice=voice, peer_producer_count=9)
    )
    assert s.priority == PRIORITY_CAP


def test_name_component_keys_are_stable() -> None:
    for inp in (name_inputs(), name_inputs(voice=None, producer="names.llm")):
        assert frozenset(triage.score(inp).components) == NAME_COMPONENT_KEYS


# --- PROFILE family -------------------------------------------------------


def test_independent_domains_saturates() -> None:
    ev = (_url("https://a.com/1"), _url("https://b.org/2"), _url("https://c.net/3"), _url("https://d.io/4"))
    s = triage.score(profile_inputs(evidence=ev))
    assert s.components["independent_domains"] == 1.0
    assert s.components["distinct_domains_count"] == 4.0
    assert s.components["corroborated"] == 1.0


def test_independent_domains_dedups_subdomains_to_one() -> None:
    ev = (_url("https://a.example.com/1"), _url("https://b.example.com/2"))
    s = triage.score(profile_inputs(evidence=ev))
    assert s.components["distinct_domains_count"] == 1.0
    assert s.components["corroborated"] == 0.0  # one registrable domain


def test_source_authority_is_fraction_of_distinct_domains() -> None:
    ev = (_url("https://a.com/1"), _url("https://b.org/2"))
    s = triage.score(profile_inputs(evidence=ev, authority_domains=frozenset({"a.com"})))
    assert s.components["source_authority"] == 0.5


def test_source_authority_zero_without_domains() -> None:
    s = triage.score(profile_inputs(evidence=()))
    assert s.components["source_authority"] == 0.0
    assert s.components["independent_domains"] == 0.0
    assert s.priority == 0.0


def test_profile_priority_never_exceeds_cap() -> None:
    ev = (_url("https://a.com/1"), _url("https://b.org/2"), _url("https://c.net/3"))
    s = triage.score(
        profile_inputs(evidence=ev, authority_domains=frozenset({"a.com", "b.org", "c.net"}))
    )
    assert s.priority == PRIORITY_CAP


def test_profile_component_keys_are_stable_incl_bio() -> None:
    for field in ("bio", "affiliation", "link"):
        inp = profile_inputs(field=field, evidence=(_url("https://a.com/1"),))
        assert frozenset(triage.score(inp).components) == PROFILE_COMPONENT_KEYS


def test_non_url_evidence_ignored_for_domains() -> None:
    ev = (EvidenceRef(kind="transcript_segment", url=None), _url("https://a.com/1"))
    s = triage.score(profile_inputs(evidence=ev))
    assert s.components["distinct_domains_count"] == 1.0


def test_scoring_is_deterministic() -> None:
    inp = name_inputs(voice=VoiceSignal(matches_value=True, grounded=True, confidence=0.7))
    assert triage.score(inp) == triage.score(inp)
