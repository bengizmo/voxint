# Quality gates

How Voxint decides what to trust: which enhancement output is kept, which
diarized voices earn a speaker proposal, and what "grounded" means. All
thresholds below are `Settings` fields (see `.env.example`) — the values here
are the defaults, chosen conservatively for titanet-large embeddings and meant
to be calibrated against locally adjudicated same-/different-speaker pairs
before being loosened.

## LLM enhancement — best-effort by design

Enhancement is optional garnish over an immutable transcript (`raw_text` is
forever); its failure semantics reflect that:

| Bound | Setting | Default |
|---|---|---|
| Per-attempt timeout | `LLM_TIMEOUT_SECONDS` | 90 s |
| Attempts per batch | `LLM_ATTEMPTS_PER_BATCH` | 2 |
| Batch size | `LLM_BATCH_MAX_SEGMENTS` / `LLM_BATCH_MAX_CHARS` | 32 / 12 000 |
| Per-run wall-clock budget | `LLM_RUN_BUDGET_SECONDS` | 4 h |
| Circuit breaker | `LLM_CONSECUTIVE_FAILURE_LIMIT` | 3 batches |

- Segments travel in contiguous, ID-keyed batches; a reply must return exactly
  the requested segment-index set or the whole batch is rejected — partial or
  misaligned output is never trusted.
- A failed batch (transport, HTTP error, malformed reply) retries once, then
  its segments keep `enhanced_text = NULL`. Three consecutive failed batches
  open the circuit for the rest of the run; the budget bounds total LLM time
  inside the `enhance_match` stage lease.
- **An unreachable LLM never fails a run.** Speaker matching still executes;
  only matching/persistence invariant violations fail the stage.

## Turn eligibility (matching input)

A `diarization_turns` row feeds matching only if it carries an embedding
(skipped windows keep their auditable `too_short` / `low_snr` reason and are
ignored) and its overlap ratio (`overlap_seconds` / duration) is at most
`MATCH_MAX_OVERLAP_RATIO` (0.20) — heavily overlapped speech contaminates
pooled embeddings. Eligible turns are weighted by usable non-overlap seconds,
capped at `MATCH_TURN_WEIGHT_CAP_SECONDS` (10 s) so one long monologue can't
drown the rest of the evidence.

## Cosine proposal and grounding gates

Per label: a duration-weighted unit centroid over eligible turns is compared
against each roster speaker's enrollment centroid (mean of unit vectors —
never max-over-enrollments), strictly within one `embedding_space`. The top
speaker is proposed only when **all** proposal gates pass; `grounded` is
claimed only when the stricter grounding set also passes:

| Gate | Proposal | Grounded |
|---|---|---|
| Eligible turns | ≥ 2 | ≥ 3 |
| Usable seconds | ≥ 6 | ≥ 10 |
| Raw cosine (top-1) | ≥ 0.60 | ≥ 0.70 |
| Top-1 vs top-2 margin | ≥ 0.05 | ≥ 0.08 |
| Vote agreement (duration-weighted) | ≥ 0.60 | ≥ 0.67 |

Vote agreement is the fraction of eligible-turn weight whose individually
nearest roster speaker is the proposed one — a per-turn consistency check that
catches labels whose centroid only *averages* into a speaker.

An unmatched or ineligible label produces **no row**: absence of evidence is
not a low-confidence proposal. P4 never creates `speakers` rows for unknown
voices — that is the adjudication UI's job (P5).

## Confidence is not probability

`speaker_assignments.confidence` stores `(cosine + 1) / 2`, clamped to
[0, 1] — a *transformed similarity*, not a calibrated probability. Margin and
vote agreement act as separate decision gates rather than being blended into
an opaque score. `llm_hint` rows store `confidence = NULL`: model-reported
confidence is not calibrated and is not recorded.

## Named ≠ grounded

An LLM name hint (`method = 'llm_hint'`, `proposed_name`) is review-side
evidence only: never a `speaker_id`, never `grounded`, even when the heard
name matches a roster speaker's `display_name` exactly. Enforcement is
layered: DB CHECK constraints (grounded ⇒ cosine + concrete speaker;
method-shape checks keep cosine and hint rows disjoint) plus a single typed
writer — `voxint.speakers.matching.replace_run_proposals` — through which
every proposal insert passes. One hint per label: an explicit
self-introduction ("I'm Jane…") beats being named by someone else; within a
kind, the earliest heard wins.

## Adjudication precedence (P5)

Attribution is resolved at read time, everywhere, by one rule
(`adjudication/resolver.py`):

1. **Effective human decision** — the newest `adjudication_decisions` row per
   (run, label), ordered `created_at DESC, id DESC`. Corrections are new
   appends; nothing is edited (the table's trigger forbids it).
2. **Grounded cosine proposal** — machine identity stands only at grounding
   strength.
3. Otherwise the label is **unresolved** and the run sits in the review queue.

`exclude` suppresses speaker attribution, never transcript text. `unknown` is
a terminal human answer — the label leaves the queue without an identity.
`llm_hint` names are displayed as evidence and never resolve attribution.

Speaker **enrollment** builds its centroid from exactly the matching-side
eligibility rules and duration-capped weighting (imported from
`speakers/matching.py`); a label with no eligible embedded turns cannot be
enrolled — rule it `unknown` or assign it to an existing speaker instead.

### Accepted risk: enrollment provenance is code-enforced

`speaker_embeddings.source_*` provenance consistency (decision is an `assign`
for the same run/label/speaker) is guaranteed by the single writer
(`adjudication/enrollment.py`) plus the unique constraint on the source
decision id — not by cross-table DB triggers. A constraint trigger would be
the only way to make it structurally airtight and is deliberately omitted as
disproportionate for a single-writer path; revisit if a second writer ever
appears.

## Offline measurement

The quality gates above act at pipeline time. The *measurement* layer — name
accuracy against ground truth, acoustic agreement verdicts, two-voter fusion,
regression gate metrics — is the offline harness, documented in
[harness.md](harness.md) and exposed as `voxint score …` (file-based, DB-free).
