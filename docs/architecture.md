# Voxint architecture

## Pipeline shape

```
input ──▶ acquire ──▶ prepare ──▶ transcribe ──▶ diarize_embed ──▶ enhance_match ──▶ finalize
          (yt-dlp     (ffmpeg      (ASR)          (diarization +     (LLM enhance +
           URL         16 kHz,                     speaker            speaker matching)
           download;   gates)                      embeddings)
           no-op when
           local)
```

Six coarse stages run as self-contained (idempotent) stage functions, driven by
a small engine (`voxint.pipeline.engine`). The engine owns everything the stages
should not care about: state transitions, per-stage transactions, attempt
bookkeeping, and crash recovery. Stage bodies own the science. Celery tasks are
thin wrappers around the engine, with no orchestration logic hiding in task code.

**ACQUIRE** (`STAGE_ORDER[0]`) is the universal first stage: a successful no-op for
local or uploaded media (`source_url IS NULL`), and a yt-dlp download for URL runs
(`voxint fetch` / `POST /fetch`). Making it the first stage, rather than a special
submit-time step, keeps the "every fresh run starts at `STAGE_ORDER[0]`" invariant
intact, so legacy `queued`/`current_stage=NULL` rows route safely into the no-op and
`submit()` keeps its signature. Download mechanics and the SSRF model are below.

## State machine

Run state lives in Postgres (`pipeline_runs.status` + `current_stage`), guarded by
compare-and-swap on an explicit `revision` column: every transition is an
`UPDATE … WHERE id = :id AND revision = :held` that also increments `revision`.
A worker holding a stale snapshot gets `StaleRevisionError` and must re-read.
Lost updates are structurally impossible.

```
queued ──▶ running ──▶ completed
  ▲          │ ▲  │
  │          │ │  └──▶ awaiting_adjudication ──▶ running   (human pause = DB state)
  │          │ └──── running (stage advance)
  │          ▼
  └──── failed          (requeue is explicit: failed ──▶ queued, keeping current_stage)
```

`completed` and `cancelled` are terminal. A requeued run **retries the stage it was
interrupted in**: earlier stages are not re-run and nothing is skipped.

Validation covers the full `(status, stage)` tuple, not just status membership: a
run cannot start at the wrong stage, advance backwards or by more than one stage,
complete mid-pipeline, or requeue at an unrelated stage.

### Stage claims and recovery

CAS alone decides whose *database* write survives; it cannot stop two workers
from both invoking a GPU call. So before executing a stage body, a worker
**claims** the stage by committing a `running` row in `stage_runs` carrying its
worker id and a lease; the `(pipeline_run_id, stage, attempt)` unique constraint
arbitrates ties. A worker that finds an unexpired claim yields without executing.

Workers that die mid-stage leave the run `running` with a claim whose lease
eventually expires. `recover_interrupted_runs` sweeps **only expired claims** (a
healthy worker three hours into a transcription is never robbed), marks the
interrupted attempt `failed`, and requeues the run through the same transition
rules. Stage bodies remain at-least-once for non-transactional effects
(filesystem, GPU services) and must be idempotent.

## Data model (alembic revisions 0001–0010)

| Table | Role |
|---|---|
| `app_settings` | single-row instance configuration set by the first-run setup wizard: onboarding-complete flag, registered media folders, custom vocabulary, LLM-enhancement toggle/endpoint, and guided-tutorial state (revision 0006) |
| `media_items` | media identity, one row per source file. `source_path` (UNIQUE) is already present for local/uploaded media, pre-assigned and materialized by ACQUIRE for URL runs; a nullable, non-unique `source_url` records URL provenance (revision 0005) |
| `media_source_metadata` | **write-once** acquisition context, 0-or-1 per media item (revision 0009): normalized extractor fields (title, uploader/channel, description, upload date, source-claimed duration, tags, canonical URL, extractor name/version) plus a bounded, allowlisted, schema-versioned `raw` JSONB subset and `acquired_at`. Context, not identity: nothing here feeds attribution, and a MediaItem is per-acquisition, so a snapshot can never rewrite the context a past adjudication was made against |
| `pipeline_runs` | execution state + CAS revision, plus the reviewer claim (token, holder, expiry) and the operator's free-text `operator_notes` (revision 0009: human input, kept structurally apart from scraped metadata, edited last-write-wins outside the CAS) |
| `stage_runs` | per-stage attempt ledger **and execution claim** (worker id, lease, status, timing, error, metrics) |
| `audio_artifacts` | derived files (preprocessed audio, chunks, exports); `reclaimed_at`/`reclaimed_bytes` record GC reclamation of the preprocessed-audio intermediate (issue #15) — the row survives as an audit stamp after its file is unlinked |
| `audio_chunks` | chunk boundaries for long-file processing |
| `transcript_segments` | raw ASR text (immutable) + `enhanced_text` beside it + `suspect` soft-tag; two GIN expression indexes (`english` tsvectors over each text variant separately, revision 0008) back the `/runs` transcript search. Both variants stay searchable; enhancement never shadows raw |
| `diarization_turns` | run-scoped observation ledger, one row per turn: interval, label, overlap, and the window's embedding outcome (vector + space, or an auditable `skip_reason`) |
| `speakers` | the grown speaker roster, with a curation lifecycle: `merged_into_id`/`merged_at` (merge tombstones) and `deleted_at` (reversible archive) (revision 0007) |
| `speaker_embeddings` | `vector(192)` + `embedding_space` tag; enrollment rows carry provenance (source run, label, and a unique link to the human decision that created them) |
| `speaker_assignments` | **machine proposals** (method, confidence, grounded flag; `llm_hint` rows carry `proposed_name`, method-shape CHECKs keep the two shapes disjoint) |
| `adjudication_decisions` | **immutable human ledger** (insert-only, idempotency key) |
| `enrichment_producer_runs` | one row per **completed** enrichment-producer invocation (revision 0010): producer key + version, XOR target scope (speaker \| run \| run+label), declared `covered_fields`, monotonic per-scope `generation` (allocated under an advisory lock; supersession compares generations, never wall clock), derived `outcome` (`'none'` = "we looked and found nothing", reviewable information), bounded schema-versioned `config` snapshot, idempotency key |
| `enrichment_candidates` | **immutable machine-derived claims** (revision 0010): suggestions *about* identity, never identity. Claim field/value, producer-local score + components. No stored review state; effective state is derived (decision > supersession stamp > proposed). A trigger blocks DELETE and every UPDATE except the write-once `superseded_by_producer_run_id` stamp |
| `enrichment_candidate_evidence` | 1:many field-level provenance per claim (revision 0010): a `media_source_metadata` column/`raw.` key, a transcript segment (+ timestamp), or a fetched URL, so one claim can cite several sources together; append-only (trigger) |
| `profile_review_decisions` | **append-only human trail for enrichment claims** (revision 0010), deliberately separate from `adjudication_decisions`: accepting a bio is a different act from ruling on who spoke. UNIQUE per candidate (terminal accept/reject); corrections arrive as fresh candidates from a producer re-run |
| `run_enrichment_assets` | **immutable run-level assets** (revision 0012, issue #41): one successful summary/topics/entity-mentions generation per row, keyed (run, kind, generation) with a monotonic per-kind generation under an advisory lock. Whole documents, not per-field claims: no review lifecycle, and regenerate supersedes (write-once stamp, trigger-enforced like `enrichment_candidates`). Carries producer + version, model, schema-versioned payload and config snapshot, and a `source_content_hash` over the canonical generation inputs (transcript text + #36 metadata + operator notes + each segment's **attributed** speaker, resolved through the shared `display_name`), so re-adjudicating, renaming, or merging a speaker marks the run's assets stale, staleness is recomputable, and a prompt/model upgrade never masquerades as a source change |
| `run_asset_jobs` | mutable orchestration for one asset-generation attempt (revision 0012): queued → running (guarded claim) → succeeded \| failed \| cancelled, one active job per (run, kind) via a partial unique index, deadline-aware cancel. Failed/cancelled jobs record NO asset and consume no generation; the three kinds fail independently by construction |

Three invariants worth naming:

- **Raw is forever.** Enhancement writes `enhanced_text`; it never touches `raw_text`.
- **Named ≠ grounded.** An LLM-proposed name is not grounded until it has
  embedding-level evidence or a human ruling; a CHECK constraint enforces that only
  a cosine proposal with a concrete speaker can claim `grounded`, and machine
  proposals are never merged into the human ledger. The ledger itself is
  append-only at the database level (a trigger rejects UPDATE/DELETE) and writes
  go through one idempotent-replay operation.
- **One embedding space at a time.** Cosine similarity is only meaningful within a
  single `embedding_space`; all vector SQL lives in one module and always filters
  by space.
- **Drafts are suggestions about identity, not identity.** The `enrichment_*`
  tables hold machine-derived claims as reviewable drafts; nothing there (accepted
  or not) feeds attribution, mutates `speakers.display_name`/`notes`, or writes the
  adjudication ledger. Writes go through the single sanctioned writers in
  `voxint.enrichment` (`drafts.py` for producers, `review.py` for the human trail);
  each new producer run atomically supersedes only its own older still-proposed
  claims within the fields it covered.

### Enrichment producers (issue #38: `names.offline`)

Producers live under `voxint/enrichment/producers/` in three layers: pure
pattern extraction (`name_patterns.py`: a bounded, versioned regex inventory
over metadata and transcript text, capitalization-independent with explicit
false-positive guards), pure per-target aggregation/scoring
(`name_scoring.py`: max-reliability base plus small corroboration/diversity/
seed bonuses capped at 0.95; no frequency reward; run_label claims are built
from self-introductions only, so a title mention can never create or inflate
a cluster-identity claim), and DB orchestration (`names.py`).

`names.offline` always invokes at **run scope**. A run-scope invocation may
emit both run-level and run_label-level candidates, and supersession keys on
the invocation scope, so every rerun cleanly retires the prior generation.
Its idempotency key is an **input signature** (producer/pattern/scoring
versions + domain-pack seeds + exact metadata/segment content): identical
inputs short-circuit to the stored row before fresh timestamps are minted,
changed inputs mint a new key and a superseding generation, and
`outcome='none'` is recorded only after a successful scan (read failures
raise). Invocation is operator-triggered only, via `voxint enrich names` or the
workbench's claim-gated button, never a pipeline stage.

The roster page (`/speakers`) can rename, merge, archive/restore speakers and
remove bad enrollment embeddings, all through `speakers.roster`, and none of it
ever writes the decision ledger:

- A speaker is **active** while `merged_into_id` and `deleted_at` are both NULL.
  The one active predicate (`roster.active_speaker_clause`) governs matching
  centroids, the workbench assign dropdown, and the decide route. Merged and
  archived speakers stop attracting proposals and decisions.
- **Merge B→A** repoints B's `speaker_embeddings` and `speaker_assignments` to A
  and keeps B as a tombstone (`merged_into_id = A`), so historical ledger FKs
  stay valid. Readers canonicalize through the merge map at read time: an old
  `assign(B)` decision *renders* as A while the ledger row keeps B forever.
  Writes collapse chains to depth 1; readers still follow chains defensively and
  fail loudly on a cycle.
- **Archive** is reversible (`deleted_at`) and deletes the speaker's cosine
  assignments: stale machine grounding must not outlive the operator's verdict.
  Embeddings and human decisions are preserved; restore does not resurrect the
  purged proposals (matching re-proposes on future runs).
- **Removing an embedding** hard-deletes the derived centroid (the minting
  decision and the raw `diarization_turns` vectors survive) and deletes all of
  that speaker's cosine assignments, because assignments carry no centroid
  lineage, so narrower invalidation would be a guess.
- **Names stay globally unique** across every lifecycle state. Re-creating an
  archived name is refused with restore guidance; enrollment replay validates
  against durable provenance (run, label, operator), never the mutable
  `display_name`, so a rename can never break a replayed enrollment POST.

## URL ingestion & SSRF (the ACQUIRE stage)

`voxint fetch <url>` / `POST /fetch` register a URL as a `MediaItem.source_url`
and queue a run; the pipeline's first stage, **ACQUIRE**, downloads it with
yt-dlp on the worker (a no-op when `source_url IS NULL`, meaning local/uploaded
media). URL ingestion is an authenticated **admin egress** capability
(`ytdlp_enabled`, on by default), **not a sandbox**, and is documented as such.

**Two SSRF gates, one policy.** A submitted URL is checked at two independent
points that share a single per-address rule (`media.netcheck.ip_is_public`).
That rule is stricter than the stdlib `is_global`: it rejects IPv6 **site-local**
(`fec0::/10`) and unwraps **IPv4-in-IPv6 embeddings** (deprecated `::a.b.c.d`, RFC
6052 NAT64 `64:ff9b::/96`, IPv4-mapped/6to4/Teredo) to judge the embedded IPv4, so
`[64:ff9b::127.0.0.1]` is refused, which `is_global` alone would pass. The two
gates:

1. **String gate (submit time).** `ingest.validate_ingest_url` requires an
   absolute http/https URL with a plain host, no embedded credentials, no
   whitespace/control chars, under a length ceiling, and (for an IP *literal*)
   a public address. It deliberately does **not** resolve DNS: a name that looks
   public now can rebind before the worker fetches it.
2. **Resolved-host gate (download time).** `media.netcheck.assert_host_resolves_public`
   re-resolves the host (A + AAAA) in the worker immediately before the download
   and rejects it (via the same `ip_is_public`) if *any* resolved address is
   non-public, closing the rebind-after-submit window for DNS *names*. It
   fail-closes on an unresolvable/empty/unparseable result. On refusal the run
   parks FAILED @ acquire for a manual Requeue, with a host-only (URL-free) error.

**yt-dlp lockdown** (`media.ytdlp`, verified against yt-dlp 2026.07.04): the argv
runs with `--no-config`, `--no-plugin-dirs` (no local/remote plugin loading),
`--no-exec` (no post-processor command), `--no-playlist --max-downloads 1`, a
size cap, and hard wall-clock timeouts; `file://` URLs are refused by yt-dlp's
own default (we never pass `--enable-file-urls`). `--proxy` is passed **always**:
the configured `ytdlp_proxy` when set, otherwise an empty value that means
"explicit direct connection", so an ambient `HTTP(S)_PROXY` in the worker env can
never silently reroute egress. `--cookies` is passed only when `ytdlp_cookies_file`
is set. Both are treated as credentials, scrubbed verbatim from any surfaced error.

**Source metadata capture** (issue #36) rides the SAME invocation:
`--write-info-json --clean-info-json --no-write-playlist-metafiles` with a typed
`infojson:` output pinning `source.info.json`, never a second `--dump-json`
call, which would double bot-block exposure and could describe different
upstream state than the downloaded bytes. The stage sanitizes the info-JSON
through a strict allowlist (`media.source_metadata`; secret-bearing keys like
`formats`/`http_headers`/`cookies` are never copied), publishes the sanitized
snapshot as a hash-addressed sidecar (`source.<sha256>.metadata.v1.json`,
linked BEFORE the media file so a crash between publish and DB commit replays
to a repaired row without re-downloading), and inserts the write-once
`media_source_metadata` row. The raw info-JSON never leaves the attempt temp
dir. Capture is **best-effort**: a missing/malformed/oversized info-JSON logs a
warning and never fails an otherwise-valid acquisition. Surfaced on the run
detail page, the runs browser (title), and `GET /runs/{id}/export.json`, a
versioned object envelope (run + source_metadata + operator_notes + the same
segment objects as the pinned bare-array `/review/{id}/export.json`, which
stays frozen).

**Residual: needs network policy, not a userland check.** yt-dlp re-resolves the
host *independently* when it connects, and its generic extractor follows HTTP
redirects and constructs URLs. So a host that rebinds between our re-resolution
and yt-dlp's fetch, an HTTP redirect to a private address, or an
extractor-constructed private URL is **beyond** these gates. Closing that requires
running the worker where it has **no route to RFC1918 / link-local / the cloud
metadata endpoint** (egress firewall or a dedicated egress). The resolved-host
gate raises the bar and closes the literal / rebind-at-check-time holes; it is not
a substitute for egress control.

**CSRF.** Four mutation forms (`POST /submit`, `/fetch`, `/runs/{id}/requeue`,
and `POST /review/{id}/claim`) carry a stateless, action-bound HMAC token
(`api.csrf`, keyed by `csrf_secret`, independent of the Basic-auth password); a
missing/mis-signed token is refused before any state change. `/claim` needs its
own because claiming is what *mints* the run's claim token: it has no unguessable
token of its own yet. The remaining review-workbench mutations (release, decision,
enroll) are instead gated by that per-run claim token.

## Web research egress (issue #39)

`voxint.research` is the second outbound-fetch capability, and the THIRD
consumer of the single egress policy in `media.netcheck` (`ip_is_public`, the
shared string gate `parse_http_url`, and the fail-closed resolver core
`resolve_public_addresses`), one module to audit for every path that leaves
the box. It is **off by default** (`VOXINT_WEB_RESEARCH=false`) and
deliberately **independent of the LLM capability**: configuring an LLM never
implies egress, enabling retrieval never requires an LLM, and a config
validator coupling the two is contract-tested absent. When off, both
operations return structured `disabled` outcomes before any DNS or socket
work.

Two operations, built as a library for the future research loop (issue #40)
plus a feature-gated CLI (`voxint research search|read`):

- **`web_search`.** One bounded query to a pluggable provider
  (`SearchProvider` protocol; SearxNG built in, its base URL being
  operator-configured egress in the same trust class as `LLM_BASE_URL`).
  Everything the provider RETURNS is untrusted: result URLs pass the shared
  string gate before being surfaced (refused ones are dropped and counted),
  titles/snippets are sanitized and capped.
- **`read_url`.** A hardened single-page fetcher that CLOSES, for its own
  path, the redirect/rebinding residual documented above for yt-dlp (which it
  can't close because yt-dlp owns its connections; this fetcher owns its own).
  Every hop (the submitted URL and each redirect target, 301/302/303/307/308
  only, `Location` resolved against the *logical* URL) is re-gated and
  re-resolved fail-closed, and the connection is **pinned** to a vetted
  address: the request URL's host is rewritten to the vetted IP while the
  `Host` header and TLS SNI carry the canonical (IDNA-encoded once) hostname,
  on a FRESH client per attempt so no keepalive connection can cross host
  identities. The checked address *is* the connected address. Responses must
  be identity-encoded (compressed responses are refused, removing the
  decompression-bomb class rather than bounding it), the streamed byte count
  is authoritative over `Content-Length`, and only `text/html`,
  `application/xhtml+xml`, and `text/plain` are readable. Extraction is
  stdlib-only (`html.parser`; no C parser on attacker bytes) and strips
  invisible-instruction characters (Unicode tag block, zero-width, bidi,
  C0/C1); retrieved content is data, never instructions.

Both operations take a mandatory bounded-identifier `Attribution` and an
atomic per-invocation `ResearchBudget` (quotas enforced IN the tools; a spent
budget yields a structured `budget_exhausted` outcome the #40 loop concludes
from; quota is charged only after validation and the concurrency slot, so a
refusal that performed no network work never burns budget). Every outbound
request logs one attribution line (feature, reason, host, verdict, bytes,
duration), and no ERROR detail or log line ever carries a URL, query string,
or redirect `Location` (`media.redaction` throughout). The one deliberate
exception: `FetchOutcome.final_url` on a **successful** read is provenance,
the fragment-free logical URL actually read, which evidence records (#40's
`UrlEvidence`) require and which may carry a query; consumers store it
deliberately and never echo it into shared logs (the CLI prints it
query-stripped). **Timing caveat:** the total wall clock bounds every HTTP
operation via remaining-time propagation, but blocking DNS resolution cannot
be hard-interrupted. DNS is the one non-hard-bounded step.

## Web-research speaker enrichment (issue #40)

The `web_researcher` producer is the consumer #39 was built for: an
**operator-initiated, per-speaker research job** that drives a bounded LLM
tool loop (`voxint.research.agent`) and quarantines everything it finds in
the #37 draft layer for field-by-field human review. Gated by
`ENRICHMENT_WEB_RESEARCH_ENABLED=false`, which **requires both**
`VOXINT_WEB_RESEARCH` and `LLM_ENABLED` at startup (fail-closed validator)
and is re-checked in the worker so queued jobs cannot outlive a capability
shutdown.

- **The orchestrator owns everything.** The loop is a hand-rolled
  strict-JSON action protocol over the plain `/chat/completions` transport
  (`HttpLLMClient.chat_json`), with no provider function-calling and no agent
  framework, so budgets, the allowed-tool set, and evidence rules live in
  auditable application code, never in a prompt. Each round the model either
  requests up to `RESEARCH_MAX_ACTIONS_PER_ROUND` actions from exactly three
  tools (`web_search`, `read_url`, read-only `query_existing_speakers`) or
  concludes; anything outside the closed schema gets one repair attempt,
  then the job **fails**, never a silent `found=false`.
- **Budgets are hard.** Retrieval quotas and the wall clock ride #39's
  `ResearchBudget` (exhaustion is a structured tool result); rounds are
  counted by the loop, and after the last round the model gets exactly one
  tools-disabled conclude request. All budgets snapshot onto the job row:
  the preview the operator approved is the contract.
- **Retrieved pages are hostile data.** Page text reaches the model only as
  a JSON-encoded, untrusted-marked tool result capped at a 4k-char excerpt;
  `read_url` accepts only URLs from this job's own search results or the
  operator-stored seed URLs (copied exactly), so an injected page cannot
  steer fetches; and the server-side conclusion gate (the actual security
  boundary) drops any claim whose `source` is not a server-issued id of a
  page actually read, whose snippet does not locate verbatim
  (NFKC + casefold + whitespace-collapsed) in that page's kept text, or
  whose value is generic ("the host", "Speaker 2"). Surviving claims become
  speaker-scoped bio/affiliation/link candidates with `UrlEvidence`.
- **Jobs are durable, honest state.** `research_jobs` holds queued → running
  (guarded claim; a duplicate Celery delivery no-ops) → succeeded | failed |
  cancelled, plus progress counters the console polls (htmx, 3 s while
  active) and a cooperative `cancel_requested` flag the loop re-reads
  between rounds. A confident `found=false` records an authoritative
  `outcome='none'` producer run; transport/LLM/contract failures and
  cancellation record **nothing**: a failure must never retire prior
  drafts. No automatic retries and no recovery sweep, deliberately: hidden
  re-execution of a non-deterministic web loop is worse than a visible
  stall the operator can cancel and restart.
- **Idempotency is per job, not per input.** The producer-run key is
  `web_researcher:speaker:{speaker_id}:{job_id}`, one durable execution.
  Web research is non-deterministic, so an input-derived key would wrongly
  suppress deliberate re-research; an intentional rerun is a new job minting
  a new generation that supersedes still-proposed prior claims.

## Provider seams

ASR, diarizer, embedder, and LLM sit behind typed protocols
(`voxint.clients.base`). The GPU services speak versioned HTTP
(`/v1/transcribe`, `/v1/diarize`, `/v1/embed`, `/healthz`) and share a
`MEDIA_ROOT` volume with the workers, no multipart uploads. The LLM stage
targets any OpenAI-compatible endpoint and is optional (`LLM_ENABLED=false`
by default); enhancement is **best-effort**: bounded ID-keyed batches, one
retry, a circuit breaker, and a wall-clock budget inside the stage lease, with
failures degrading to NULL `enhanced_text` rather than failing the run (see
`docs/quality-gates.md`). Speaker matching always runs and its invariant
violations DO fail the stage. Test fakes satisfy the same protocols, which is
how the end-to-end contract tests run without a GPU.

Domain-specific vocabulary and prompts are their own seam: a **domain pack**
(`voxint.domain_packs`, selected via `DOMAIN_PACK_PATH`) supplies ASR
vocabulary hints, name seeds, and LLM prompt fragments; a neutral
meeting/podcast pack ships as the default.

## Review console (P5)

Adjudication is **post-hoc**: runs complete normally and the console works a
queue over COMPLETED runs. A run needs review while any diarization label has
neither an effective human decision nor a *grounded* cosine proposal.
(`AWAITING_ADJUDICATION` stays in the state machine, reserved for a future
flow that genuinely blocks downstream processing; nothing enters it today.)

- **One resolver** (`adjudication/resolver.py`) settles attribution at read
  time for the workbench, the queue, and the transcript export alike:
  effective human decision (newest ledger row per label; corrections are
  appends) beats grounded cosine beats nothing. `llm_hint` names render as
  evidence, never as identity; `exclude` suppresses attribution, never text.
- **Runs search** (`GET /runs`, revision 0008): transcript full-text (`q=`,
  `websearch_to_tsquery` over per-segment `english` tsvectors of raw AND
  enhanced text separately, so the search document is one segment), a speaker
  facet answered by a SQL mirror of the resolver
  (`speaker_attributed_exists`: effective decision or grounded cosine, merge
  tombstones expanded via `roster.alias_ids`; archived speakers stay
  offered, marked), source-substring and UTC date facets. Everything
  AND-composes with status/review and the `(created_at, id)` keyset cursor;
  results stay newest-first, no relevance ranking; matching runs get one
  escaped `ts_headline` snippet (first matching segment).
- **Reviewer slot**: claim columns on `pipeline_runs`, guarded by the same CAS
  `revision` as pipeline transitions. The claim token is an opaque per-claim
  secret required on every mutation; a re-claim rotates it, so a stale tab
  gets 409 instead of acting on a slot someone else holds. Claims expire on a
  TTL, so an abandoned tab never dams the queue.
- **Decisions** POST through the existing idempotent ledger append. Each
  rendered form carries a fresh server-issued nonce as the idempotency key:
  htmx retries are harmless replays, new submissions are new (superseding)
  rulings.
- **Enrollment** turns an unmatched voice into a roster identity atomically:
  a `speakers` row, one duration-weighted centroid in `speaker_embeddings`
  (same eligibility rules and centroid math as matching, imported from
  `speakers/matching.py` so they cannot drift), and the `assign` ruling.
  Raw per-turn vectors stay in `diarization_turns`; the centroid is
  re-derivable. Provenance columns plus a unique constraint on the source
  decision make duplicate enrollment structurally impossible.
- **Auth**: single-operator HTTP Basic (constant-time compare) on every route
  but `/healthz`, fragments and media included; operator identity comes only
  from credentials. Startup refuses to bind off-loopback with the default
  password.
- **Media**: audio streams through a gate that requires the file to be
  DB-referenced, to resolve inside `MEDIA_ROOT` (symlink escapes rejected),
  and to carry a decodable audio stream per ffprobe (bounded subprocess,
  cached per path/size/mtime). Single-range HTTP semantics: 206/416,
  open-ended and suffix forms; multipart ranges are ignored per RFC.

## Frontend islands (issue #48)

Jinja owns every page; interactive regions are React **islands** mounted into
server-rendered markup. This extends the review console — it is not a new
subsystem and adds no page routing.

- **Vite, not Astro (settled decision).** Plain Vite v6 multi-entry + React 19
  + Tailwind v3, `vite build` only. Astro's value is its own page/SSR
  rendering, content collections, and `.astro` format — all unused here, since
  Jinja renders every page and mounting into a foreign template engine still
  means hand-writing `createRoot(el).render(...)` against `data-*` points. Vite's
  multi-entry + `manifest.json` output is the first-class workflow for compiling
  independent TS/TSX entries into content-hashed bundles that some *other* system
  embeds; it has no server-runtime concept, so nothing can reach for a Node
  adapter that would violate "no Node at operator runtime". It also keeps the
  npm supply-chain surface (every `npm audit` line) smaller. **Trade-off (honest):**
  if voxint ever wants genuinely server-rendered `.astro` pages we'd migrate then
  — cheap, because the React components are framework-agnostic beyond their mount
  call — and we accept hand-writing the ~20-line manifest→`<script>`/`<link>`
  lookup on the Python side as the price of not carrying an unused meta-framework.
- **Progressive enhancement is the contract.** The server HTML is the fallback,
  fully usable with JS disabled or the asset route unbuilt; islands are additive
  and replace only their region's *visual* role once hydrated. An island that
  fails to hydrate degrades one region, never the page — the server markup inside
  its mount div stays visible. The transcript page demonstrates this: a native
  `<audio>` plus the segment list, active segment highlighted on `timeupdate`,
  over the same `{% for ln in lines %}` loop the JS-off page renders.
- **Auth-aware asset route, never a `StaticFiles` mount.** Bundles serve through
  `GET /static/app/{path}` carrying the operator auth dependency on every byte —
  a mount would bypass it, and "everything but `/healthz` authenticates" is
  absolute. The route resolves+contains the untrusted path (traversal/symlink
  escapes 404 before any filesystem read) and sets `immutable` caching only for
  Vite-hashed filenames. A contract test pins the absence of any `StaticFiles`
  import/instantiation.
- **Build-stage boundary.** The Dockerfile's `node:22-slim` stage builds the
  bundles and is then discarded; only `dist/` is COPYed into the Python image
  before `uv sync --no-editable`, so the wheel packages the static tree. No Node
  binary ships — empirically verifiable via `docker history` /
  `command -v node` in the runtime image.
- **Mount convention #49–#59 extend, not reinvent.** `base.html` pulls one
  shared module (`main.ts`) that scans for `[data-island]` nodes and
  dynamically imports only the bundles present; a page carries an island by
  emitting `<div data-island="name" data-props='{...}'>` with a server-rendered
  fallback inside. Adding an island never edits `base.html`. Islands read props
  via `readProps()` and call voxint's own routes through the shared
  `api-client.ts` `apiFetch`, whose `ApiError` mirrors FastAPI's `{detail}`
  shape — the seam #54/#55 consume for capability-aware responses.
- **Per-turn playback + fail-closed seek gating (issues #49/#55).** Two islands
  add "play this turn"/"preview this speaker" seeking. `transcript-player`
  (transcript.html) is fully in-React: per-line ▶ buttons and click-to-seek call
  the shared `lib/playback.ts` `playTurn`, which seeks + plays + stops at the
  segment end via a rate-aware guard (a one-shot `timeupdate` check plus a
  `setTimeout` fallback, so a coarse timeupdate can't overshoot into the next
  voice) and holds exactly one cancellable active turn. `workbench-player`
  (run.html) is the harder case: the per-turn buttons are **server-rendered
  inside `#labels`**, which every adjudication ruling replaces via
  `hx-swap="innerHTML"`. So the island mounts **outside `#labels`** (wrapping the
  `<audio>`, which survives swaps) and drives those buttons with **document-level
  event delegation**: one delegated `click` listener scoped to the current
  `#labels`, plus an `htmx:afterSwap` listener filtered to `#labels` swaps that
  re-runs an "enable pass". Buttons render `disabled` + `type="button"`
  server-side (honest JS-off default, never submitting a form); the island
  removes `disabled` only when seeking is safe. Both listeners are installed in a
  single StrictMode-safe effect with symmetric cleanup.
- **The fail-closed capability contract (issue #55).** `api/playback.py`'s
  `playback_capability()` is the seek predicate: `seek_enabled` is true only when
  the media is actually servable, the duration is finite and positive, every
  transcript interval is well-formed, and no interval runs past `duration +
  0.05s` (a fixed tolerance, absorbing float noise without scaling on long
  files). It accumulates **every** applicable reason with plain-language messages
  the islands show in a visible banner — never a bare tooltip. Media servability
  reuses `resolve_servable_media()`, the **single seam** `GET /media` itself
  calls, so capability can never advertise seeking while `/media` would 404/410.
  "Preview this speaker" seeks a clean `DiarizationTurn` (longest non-overlap,
  fallback longest) — never the longest transcript segment, which carries only a
  dominant-overlap label and can contain other voices.

## Worker orchestration (P3)

One Celery task, `voxint.run_pipeline`, drives a run through all stages via
the engine (task-per-stage would open an unclaimed window between handoffs
that recovery misreads as a crash; the engine already resumes an interrupted
run at its current stage). Failure handling is two-lane:

- **Transient** (`retryable` service errors: `saturated`, `model_unavailable`,
  transport failures): the failed attempt stays in the `stage_runs` ledger,
  the run is CAS-requeued at the same stage (against the exact revision that
  failure produced, so a stale callback can never requeue a newer failure),
  and the task retries itself with exponential backoff. The attempt budget
  (`STAGE_MAX_ATTEMPTS`) counts transient *service* failures from the
  persisted ledger (restarts and broker loss never reset it); lease-expiry
  interruptions don't eat it, and the sweep applies the same ceiling to
  crash loops separately.
- **Deterministic** (`inference_failed`, protocol violations, bad media): the
  run stays FAILED for the failure lane; `voxint requeue` is the explicit
  human override.

A beat task (`voxint.recovery_sweep`, every `RECOVERY_SWEEP_SECONDS`)
requeues runs whose stage lease expired and re-enqueues QUEUED runs whose
task evaporated with the broker (`QUEUED_RUN_STALE_SECONDS` grace so pending
retry countdowns aren't stepped on). Duplicate enqueues are safe by design:
claims and CAS arbitrate.

A second, **opt-in** beat task (`voxint.gc_sweep`, issue #15) reclaims the
large normalized-audio intermediate for old terminal runs when
`MEDIA_RETENTION_ENABLED` — it unlinks `artifacts/{run_id}/normalized.wav` and
stamps the `audio_artifacts` row (`reclaimed_at`/`reclaimed_bytes`; the row is
kept as an audit record). File reclamation only: source media, transcript,
diarization, and the decision ledger are never touched, so a reclaimed run
stays re-processable from source. Rows are claimed oldest-first with `FOR
UPDATE ... SKIP LOCKED`, so overlapping sweeps neither double-count nor clobber
a byte measurement. See operations.md for tuning.

Timeout ordering that must hold: HTTP client timeout
(`GPU_HTTP_TIMEOUT_SECONDS`) **<** stage lease (`STAGE_LEASE_SECONDS`; two
stages carry their own longer lease: `DIARIZE_EMBED_LEASE_SECONDS` because it
makes one diarization call plus several sequential embedding batches, and
`ACQUIRE_LEASE_SECONDS` (3 h) because a URL download runs under a wall-clock
`ACQUIRE_TIMEOUT_SECONDS` (2 h) that must itself sit below the lease by a
cleanup margin, validated at startup) **<** Redis visibility timeout
(`CELERY_VISIBILITY_TIMEOUT_SECONDS`, which covers a whole run and is sized to
the six-stage worst case, 48 h).
