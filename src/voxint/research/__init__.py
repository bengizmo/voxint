"""Controlled web retrieval (issue #39): pluggable search + hardened URL reader.

OFF by default (``voxint_web_research``), independent of the LLM capability by
design. Two operations behind narrow interfaces — :func:`web_search` and
:func:`read_url` — both budget-enforcing (structured ``budget_exhausted``
outcomes a future research loop, issue #40, can rely on) and both attributable
(a mandatory :class:`Attribution` names the requesting feature on every
outbound request). ``read_url`` extends the ingest SSRF doctrine
(``media.netcheck``) with per-redirect-hop revalidation and vetted-address
connection pinning; see docs/architecture.md "Web research egress".
"""

from voxint.research.budget import Attribution, ResearchBudget
from voxint.research.fetch import FetchOutcome, read_url
from voxint.research.search import (
    SearchOutcome,
    SearchProvider,
    SearchResult,
    build_web_search_provider,
    web_search,
)

__all__ = [
    "Attribution",
    "FetchOutcome",
    "ResearchBudget",
    "SearchOutcome",
    "SearchProvider",
    "SearchResult",
    "build_web_search_provider",
    "read_url",
    "web_search",
]
