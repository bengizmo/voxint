"""Contract: draft-triage invariants stay pinned and documented (issue #42).

Triage is read-time review ordering, not model inference — no parity gates —
but its scoring IS a versioned, explainable contract. This pins the version, the
per-family component-key sets a template/operator relies on, and the operator
authority-allowlist knob's documentation + feature-neutral default.
"""

import re

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings
from voxint.enrichment.triage import (
    NAME_COMPONENT_KEYS,
    PROFILE_COMPONENT_KEYS,
    TRIAGE_VERSION,
)


def test_triage_version_pinned() -> None:
    # Bumping this is deliberate — it changes ordering and the audit trail.
    assert TRIAGE_VERSION == 1


def test_name_component_keys_pinned() -> None:
    assert frozenset(
        {
            "name_match",
            "voice_support",
            "voice_conflict",
            "cross_source_agreement",
            "peer_producer_count",
        }
    ) == NAME_COMPONENT_KEYS


def test_profile_component_keys_pinned() -> None:
    assert frozenset(
        {
            "independent_domains",
            "source_authority",
            "corroborated",
            "distinct_domains_count",
        }
    ) == PROFILE_COMPONENT_KEYS


def test_sources_per_claim_fits_evidence_rows() -> None:
    # Each grounded source becomes one evidence row; the per-claim source cap
    # must fit under the writer's evidence-row bound or sources truncate silently.
    from voxint.enrichment.drafts import MAX_EVIDENCE_ROWS
    from voxint.research.agent import MAX_SOURCES_PER_CLAIM

    assert MAX_SOURCES_PER_CLAIM <= MAX_EVIDENCE_ROWS


def test_source_authority_domains_documented_and_neutral_default() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert re.search(r"^#?\s*SOURCE_AUTHORITY_DOMAINS=", env_example, re.MULTILINE)
    # Empty default: source_authority reads 0.0 for every draft until configured.
    assert Settings(_env_file=None).source_authority_domains == ""
