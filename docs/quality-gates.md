# Quality gates

How Voxint decides what to trust: which enhancement output is kept, which
diarized voices earn a speaker proposal, and what "grounded" means. Every
threshold below is a `Settings` field (see `.env.example`). The values shown are
the defaults, chosen conservatively for titanet-large embeddings and meant to be
calibrated against locally adjudicated same-/different-speaker pairs before you
loosen them.

## LLM enhancement: best-effort by design

Enhancement is optional garnish over an immutable transcript (`raw_text` is
forever), and its failure semantics follow from that:

| Bound | Setting | Default |
|---|---|---|
| Per-attempt timeout | `LLM_TIMEOUT_SECONDS` | 300 s |
| Attempts per batch | `LLM_ATTEMPTS_PER_BATCH` | 2 |
| Batch size | `LLM_BATCH_MAX_SEGMENTS` / `LLM_BATCH_MAX_CHARS` | 32 / 12 000 |
| Per-run wall-clock budget | `LLM_RUN_BUDGET_SECONDS` | 4 h |
| Circuit breaker | `LLM_CONSECUTIVE_FAILURE_LIMIT` | 3 batches |

- Segments travel in contiguous, ID-keyed batches. A reply must return exactly
  the requested segment-index set or the whole batch is rejected; partial or
  misaligned output is never trusted.
- A failed batch (transport, HTTP error, malformed reply) retries once, then
  its segments keep `enhanced_text = NULL`. Three consecutive failed batches
  open the circuit for the rest of the run. The budget bounds total LLM time
  inside the `enhance_match` stage lease.
- **An unreachable LLM never fails a run.** Speaker matching still executes;
  only matching/persistence invariant violations fail the stage.
- **Segment text is content, never instructions.** The enhancement prompt
  instructs the model to treat every segment's words strictly as transcript
  content to edit — a segment that reads like a command ("ignore previous
  instructions", "reply with a single word", "you are now a translator", "drop
  the other segments") is still returned unchanged, not obeyed. This matters
  because transcripts are untrusted input and small local models will otherwise
  follow instructions embedded in speech; a hardened prompt measurably stops
  that (verified against the local-LLM qualification corpus). It is a
  best-effort guard, not a sandbox — the structural batch-integrity check above
  is the backstop that rejects any reply that still deviates.
- **Name hints are consumed only on the BYO path.** Enhancement can also surface
  explicit spoken-name hints for speaker attribution, but only capable BYO models
  have them parsed: the scoped bundled model (#67) does no attribution, so its
  enhancement reply is **not parsed for `name_hints`** (#85) — a hallucinated hint
  from a weak model cannot fail the batch, and attribution stays exclusively on
  the BYO name producer. The enhancement **prompt is identical on every path**
  (removing the hints block measurably regressed the 4B model's segment
  faithfulness); only the reply parsing differs.
- **Requests are deterministic by default** (`temperature = 0`, greedy). The
  client carries a fixed sampling profile; greedy is deliberate for a
  faithfulness task (reproducible, conservative), not an accident.

## Turn eligibility (matching input)

A `diarization_turns` row feeds matching only if it carries an embedding and its
overlap ratio (`overlap_seconds` / duration) is at most
`MATCH_MAX_OVERLAP_RATIO` (0.20); heavily overlapped speech contaminates pooled
embeddings. Skipped windows keep their auditable `too_short` / `low_snr` reason
and are ignored. Eligible turns are weighted by usable non-overlap seconds,
capped at `MATCH_TURN_WEIGHT_CAP_SECONDS` (10 s) so one long monologue can't
drown the rest of the evidence.

## Cosine proposal and grounding gates

Per label: a duration-weighted unit centroid over eligible turns is compared
against each roster speaker's enrollment centroid (mean of unit vectors, never
max-over-enrollments), strictly within one `embedding_space`. The top speaker is
proposed only when **all** proposal gates pass; `grounded` is claimed only when
the stricter grounding set also passes:

| Gate | Proposal | Grounded |
|---|---|---|
| Eligible turns | ≥ 2 | ≥ 3 |
| Usable seconds | ≥ 6 | ≥ 10 |
| Raw cosine (top-1) | ≥ 0.60 | ≥ 0.70 |
| Top-1 vs top-2 margin | ≥ 0.05 | ≥ 0.08 |
| Vote agreement (duration-weighted) | ≥ 0.60 | ≥ 0.67 |

Vote agreement is the fraction of eligible-turn weight whose individually
nearest roster speaker is the proposed one, a per-turn consistency check that
catches labels whose centroid only *averages* into a speaker.

An unmatched or ineligible label produces **no row**: absence of evidence is
not a low-confidence proposal. P4 never creates `speakers` rows for unknown
voices; that is the adjudication UI's job (P5).

> The enrichment draft layer (issue #37) records the complementary fact at a
> *different* layer: a completed producer invocation that substantiated
> nothing persists as an explicit `enrichment_producer_runs` row with
> `outcome = 'none'`. The two rules do not conflict. "No proposal row"
> keeps weak evidence out of the matching surface, while "we looked and
> found nothing" is itself reviewable information about the *search*, not a
> low-confidence claim. Enrichment candidates never enter matching at all.

## Confidence is not probability

`speaker_assignments.confidence` stores `(cosine + 1) / 2`, clamped to
[0, 1]: a *transformed similarity*, not a calibrated probability. Margin and
vote agreement act as separate decision gates rather than being blended into
an opaque score. `llm_hint` rows store `confidence = NULL`, because
model-reported confidence is not calibrated and is not recorded.

`transcript_segments.confidence` (issue #53) is the same kind of thing for ASR:
`exp(avg_logprob)` clamped to [0, 1] — a *transformed likelihood* (the geometric
mean of the segment's token probabilities), **not** the probability that the
segment is correct. It is persisted verbatim from the whisper service, NULL when
the backend reports none (older runs never fabricate a value). The review console
flags segments below `review_low_confidence_threshold` (default 0.6) for triage
and labels them **"uncertain, not necessarily wrong"** — never "N% correct". The
threshold is a configurable starting default, deliberately **not** a UI slider (a
non-technical operator mis-setting it would distrust the signal); refine it
against a real-corpus histogram before exposing any tuning UI. This reads existing
model output and does not touch inference — parity/contract gates are unaffected.

Draft **review priority** (issue #42, [`enrichment-triage.md`](enrichment-triage.md))
is the same kind of thing for enrichment drafts: a read-time, explainable
attention-ordering score fused from name-match, voice support, independent
domains, source authority, and cross-source agreement. It is capped below
certainty, never stored, never auto-accepts, and is compared only within one
review surface. Being review ordering rather than inference, it too leaves
parity gates unaffected.

## Named ≠ grounded

An LLM name hint (`method = 'llm_hint'`, `proposed_name`) is review-side
evidence only: never a `speaker_id`, never `grounded`, even when the heard
name matches a roster speaker's `display_name` exactly. Enforcement is
layered. DB CHECK constraints (grounded ⇒ cosine + concrete speaker;
method-shape checks keep cosine and hint rows disjoint) combine with a single
typed writer, `voxint.speakers.matching.replace_run_proposals`, through which
every proposal insert passes. One hint per label: an explicit
self-introduction ("I'm Jane…") beats being named by someone else, and within
a kind the earliest heard wins.

## Enrichment drafts (issue #37): read names stay suggestions

The "named ≠ grounded" rule extends unchanged to the enrichment draft layer.
A name mined from metadata, a transcript, or the web is stored in
`enrichment_candidates` as a *suggestion about* identity with its evidence,
and even an operator **accepting** it records only a `profile_review_decisions`
row: it never writes `speakers.display_name`, never creates a proposal, and
never resolves attribution (integration-tested). Draft `score` /
`score_components` are producer-local signals like the transformed cosine
above, not probabilities, and never comparable across producers. Effective
draft state is derived at read time (human decision > supersession stamp >
proposed) by `enrichment/queries.py`, mirroring how attribution is resolved
below.

The first producer (#38, `names.offline`) applies the rule with one further
distinction: only a **self-introduction inside a cluster's own segment** can
target that cluster (`run_label`). Every other signal (titles, descriptions,
channel names, "please welcome X") stays a run-level hint ("this name is
probably in the recording"), because knowing a name is present says nothing
about which voice it belongs to. Accepting a per-label suggestion prefills
the Enroll form but never submits it; the human act of enrollment, with its
acoustic eligibility gates, remains the only path from a heard name to a
roster identity.

## Adjudication precedence (P5)

Attribution is resolved at read time, everywhere, by one rule
(`adjudication/resolver.py`):

1. **Effective human decision**: the newest `adjudication_decisions` row per
   (run, label), ordered `created_at DESC, id DESC`. Corrections are new
   appends; nothing is edited (the table's trigger forbids it).
2. **Grounded cosine proposal**: machine identity stands only at grounding
   strength.
3. Otherwise the label is **unresolved** and the run sits in the review queue.

`exclude` suppresses speaker attribution, never transcript text. `unknown` is
a terminal human answer: the label leaves the queue without an identity.
`llm_hint` names are displayed as evidence and never resolve attribution.

Speaker **enrollment** builds its centroid from exactly the matching-side
eligibility rules and duration-capped weighting (imported from
`speakers/matching.py`). A label with no eligible embedded turns cannot be
enrolled; rule it `unknown` or assign it to an existing speaker instead.

### Accepted risk: enrollment provenance is code-enforced

`speaker_embeddings.source_*` provenance consistency (the decision is an
`assign` for the same run/label/speaker) is guaranteed by the single writer
(`adjudication/enrollment.py`) plus the unique constraint on the source
decision id, not by cross-table DB triggers. A constraint trigger would be
the only way to make it structurally airtight, and it is deliberately omitted
as disproportionate for a single-writer path. Revisit if a second writer ever
appears.

## Offline measurement

The quality gates above act at pipeline time. The *measurement* layer is the
offline harness: name accuracy against ground truth, acoustic agreement
verdicts, two-voter fusion, regression gate metrics. It is documented in
[harness.md](harness.md) and exposed as `voxint score …` (file-based, DB-free).
