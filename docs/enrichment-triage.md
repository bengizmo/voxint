# Draft triage: review-priority scoring

Once the name producers (offline + LLM) and the web-research agent generate
volume, the operator needs to know **which enrichment drafts to look at first**.
Voxint fuses several signals into a single, explainable **review priority** used
to order drafts and populate the `unresolved` bucket.

**What it is not.** The priority is an attention-ordering heuristic, *not* a
probability and *never* an accept/reject decision. Nothing is auto-accepted.
There is no score floor that hides weak drafts (that would be an uncalibrated
implicit reject). Auto-accept thresholds are deliberately out of scope — they
require a deployment's own adjudicated history to calibrate, which fresh installs
do not have. Priorities are compared **only within a single surface** (name
suggestions among themselves; profile drafts among themselves), never across
surfaces or treated as ground truth.

The score is computed **at read time** (like the derived review state in
`enrichment/queries.py`) — it is never stored, so it always reflects the current
producers, decisions, and configuration. The logic lives in the pure module
`src/voxint/enrichment/triage.py` and is versioned by `TRIAGE_VERSION`: changing
any weight, cap, or component key changes ordering and the audit trail, so the
version must be bumped (contract-pinned in `tests/contracts/test_triage_config.py`).

Every component is a number in `[0, 1]`; the final priority is capped at
`PRIORITY_CAP = 0.95`, so nothing ever reads as certain. Components that do not
apply contribute `0.0` — weights are **never renormalized**, because rewarding
missing evidence would invert the signal.

## Name suggestions (run / run_label)

| Component | Weight | Meaning |
|---|---|---|
| `name_match` | 0.60 | Per-producer name-match strength via an **adapter**, never a raw cross-producer score. The offline producer contributes its documented `base` reliability; the LLM producer a fixed modest constant (`LLM_NAME_MATCH`); an unknown producer contributes 0. This respects the rule that producer-local scores are not comparable across producers. |
| `voice_support` | 0.25 | A **grounded** cosine speaker assignment for this label whose roster identity name matches the proposed name — the persisted `speaker_assignments.confidence` (a transformed cosine, not a calibrated probability). A cosine that names a *different* identity is **not** support. |
| `cross_source_agreement` | 0.15 | Distinct producers proposing the same normalized name for this label: 1 producer → 0.0, 2 → 0.5, 3+ → 1.0. Computed over **active** (proposed or accepted) candidates only — a rejected or superseded claim never corroborates. |

`voice_conflict` is a visible companion component (0 or 1): a grounded cosine
naming a different identity. It lightly demotes the priority
(`VOICE_CONFLICT_PENALTY`) and is surfaced to the operator, never a boost.

Because the workbench collapses duplicate `(label, value)` claims to one
representative, a decision on one producer's claim could otherwise hide another
producer's still-open proposal. Triage surfaces an `unresolved_peers` count so
those hidden open proposals stay visible.

## Profile drafts (bio / affiliation / link)

| Component | Weight | Meaning |
|---|---|---|
| `independent_domains` | 0.50 | Count of **distinct registrable domains** citing the value (not raw URLs — multiple pages of one site count once), saturating at `INDEPENDENT_DOMAINS_TARGET = 3`. |
| `source_authority` | 0.50 | Fraction of those distinct domains that appear on the operator's authority allowlist (`SOURCE_AUTHORITY_DOMAINS`). 0.0 when the draft has no domains or the allowlist is empty. |

`corroborated` is a visible companion flag (0 or 1): ≥2 distinct domains agree on
the value. For a profile value, corroboration **is** the independent-domain count,
so it is shown but not separately weighted (weighting both would double-count the
same domains). BIO carries no cross-producer agreement term at all — exact bio
equality across sources effectively never occurs, and semantic similarity would
be opaque inference the operator cannot check.

### The authority allowlist

`SOURCE_AUTHORITY_DOMAINS` is a comma/space-separated list of **bare registrable
domains** you count as authoritative (e.g. `gov.uk, who.int, ieee.org`). A
subdomain matches its registrable domain (`news.example.org` → `example.org`).
Entries carrying a scheme, path, port, credentials, or wildcard are rejected.
There is deliberately **no built-in list** — what counts as authoritative differs
by deployment. Empty (the default) leaves `source_authority` at 0.0 for every
draft, and ordering falls to the other signals. It is env-only today (edit and
restart); it moves to the settings UI when that console lands.

## The registrable-domain heuristic

`registrable_domain()` is a **bounded, stdlib-only** best-effort: it collapses
subdomains to the last two labels, keeping three labels under a small reviewable
set of common multi-part public suffixes (`MULTI_PART_SUFFIXES`, e.g. `co.uk`,
`com.au`). It is deliberately **not** a full Public Suffix List — that is a
data-bearing dependency needing refresh, rejected here for the project's
audience. A registrable domain under a rare multi-level ccTLD absent from the set
collapses to its last two labels, which can only nudge ordering; it never
corrupts stored data. Bump `TRIAGE_VERSION` if the suffix set changes.

## Relationship to other numerics

This is review ordering, not model inference, so it has **no parity gates** (see
[`quality-gates.md`](quality-gates.md)). It does require determinism, stable
tie-breaking, a version, and regression fixtures showing the full breakdown —
enforced by `tests/unit/test_triage.py`, `tests/integration/test_triage_ordering.py`,
and `tests/contracts/test_triage_config.py`.
