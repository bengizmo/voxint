# Voxint architecture

## Pipeline shape

```
media file ──▶ prepare ──▶ transcribe ──▶ diarize_embed ──▶ enhance_match ──▶ finalize
              (ffmpeg      (ASR)          (diarization +     (LLM enhance +
               16 kHz,                     speaker            speaker matching)
               gates)                      embeddings)
```

Five coarse stages, executed as pure functions driven by a small engine
(`voxint.pipeline.engine`). The engine owns everything the stages should not care
about: state transitions, per-stage transactions, attempt bookkeeping, and crash
recovery. Stage bodies own the science. Celery tasks are thin wrappers around the
engine — there is no orchestration logic hiding in task code.

## State machine

Run state lives in Postgres (`pipeline_runs.status` + `current_stage`), guarded by
compare-and-swap on an explicit `revision` column: every transition is an
`UPDATE … WHERE id = :id AND revision = :held` that also increments `revision`.
A worker holding a stale snapshot gets `StaleRevisionError` and must re-read —
lost updates are structurally impossible.

```
queued ──▶ running ──▶ completed
  ▲          │ ▲  │
  │          │ │  └──▶ awaiting_adjudication ──▶ running   (human pause = DB state)
  │          │ └──── running (stage advance)
  │          ▼
  └──── failed          (requeue is explicit: failed ──▶ queued, keeping current_stage)
```

`completed` and `cancelled` are terminal. A requeued run **retries the stage it was
interrupted in** — earlier stages are not re-run and nothing is skipped.

Validation covers the full `(status, stage)` tuple, not just status membership: a
run cannot start at the wrong stage, advance backwards or by more than one stage,
complete mid-pipeline, or requeue at an unrelated stage.

### Stage claims and recovery

CAS alone decides whose *database* write survives — it cannot stop two workers
from both invoking a GPU call. So before executing a stage body, a worker
**claims** the stage by committing a `running` row in `stage_runs` carrying its
worker id and a lease; the `(pipeline_run_id, stage, attempt)` unique constraint
arbitrates ties. A worker that finds an unexpired claim yields without executing.

Workers that die mid-stage leave the run `running` with a claim whose lease
eventually expires; `recover_interrupted_runs` sweeps **only expired claims** —
a healthy worker three hours into a transcription is never robbed — marks the
interrupted attempt `failed`, and requeues the run through the same transition
rules. Stage bodies remain at-least-once for non-transactional effects
(filesystem, GPU services) and must be idempotent.

## Data model (alembic revisions 0001–0004)

| Table | Role |
|---|---|
| `media_items` | media identity — one row per source file |
| `pipeline_runs` | execution state + CAS revision, plus the reviewer claim (token, holder, expiry) |
| `stage_runs` | per-stage attempt ledger **and execution claim** (worker id, lease, status, timing, error, metrics) |
| `audio_artifacts` | derived files (preprocessed audio, chunks, exports) |
| `audio_chunks` | chunk boundaries for long-file processing |
| `transcript_segments` | raw ASR text (immutable) + `enhanced_text` beside it + `suspect` soft-tag |
| `diarization_turns` | run-scoped observation ledger: one row per turn — interval, label, overlap, and the window's embedding outcome (vector + space, or an auditable `skip_reason`) |
| `speakers` | the grown speaker roster |
| `speaker_embeddings` | `vector(192)` + `embedding_space` tag; enrollment rows carry provenance (source run, label, and a unique link to the human decision that created them) |
| `speaker_assignments` | **machine proposals** (method, confidence, grounded flag; `llm_hint` rows carry `proposed_name`, method-shape CHECKs keep the two shapes disjoint) |
| `adjudication_decisions` | **immutable human ledger** (insert-only, idempotency key) |

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

## URL ingestion & SSRF (the ACQUIRE stage)

`voxint fetch <url>` / `POST /fetch` register a URL as a `MediaItem.source_url`
and queue a run; the pipeline's first stage, **ACQUIRE**, downloads it with
yt-dlp on the worker (a no-op when `source_url IS NULL` — local/uploaded media).
URL ingestion is an authenticated **admin egress** capability (`ytdlp_enabled`,
on by default), **not a sandbox**, and is documented as such.

**Two SSRF gates, one policy.** A submitted URL is checked at two independent
points that share a single per-address rule (`media.netcheck.ip_is_public`).
That rule is stricter than the stdlib `is_global`: it rejects IPv6 **site-local**
(`fec0::/10`) and unwraps **IPv4-in-IPv6 embeddings** (deprecated `::a.b.c.d`, RFC
6052 NAT64 `64:ff9b::/96`, IPv4-mapped/6to4/Teredo) to judge the embedded IPv4 —
so `[64:ff9b::127.0.0.1]` is refused, which `is_global` alone would pass. The two
gates:

1. **String gate (submit time)** — `ingest.validate_ingest_url` requires an
   absolute http/https URL with a plain host, no embedded credentials, no
   whitespace/control chars, under a length ceiling, and — for an IP *literal* —
   a public address. It deliberately does **not** resolve DNS: a name that looks
   public now can rebind before the worker fetches it.
2. **Resolved-host gate (download time)** — `media.netcheck.assert_host_resolves_public`
   re-resolves the host (A + AAAA) in the worker immediately before the download
   and rejects it (via the same `ip_is_public`) if *any* resolved address is
   non-public — closing the rebind-after-submit window for DNS *names*. It
   fail-closes on an unresolvable/empty/unparseable result. On refusal the run
   parks FAILED @ acquire for a manual Requeue, with a host-only (URL-free) error.

**yt-dlp lockdown** (`media.ytdlp`, verified against yt-dlp 2026.07.04): the argv
runs with `--no-config`, `--no-plugin-dirs` (no local/remote plugin loading),
`--no-exec` (no post-processor command), `--no-playlist --max-downloads 1`, a
size cap, and hard wall-clock timeouts; `file://` URLs are refused by yt-dlp's
own default (we never pass `--enable-file-urls`). An optional `ytdlp_proxy` /
`ytdlp_cookies_file` is wired to `--proxy` / `--cookies` **only when set**, and
both are treated as credentials — scrubbed verbatim from any surfaced error.

**Residual — needs network policy, not a userland check.** yt-dlp re-resolves the
host *independently* when it connects, and its generic extractor follows HTTP
redirects and constructs URLs. So a host that rebinds between our re-resolution
and yt-dlp's fetch, an HTTP redirect to a private address, or an
extractor-constructed private URL is **beyond** these gates. Closing that requires
running the worker where it has **no route to RFC1918 / link-local / the cloud
metadata endpoint** (egress firewall or a dedicated egress). The resolved-host
gate raises the bar and closes the literal / rebind-at-check-time holes; it is not
a substitute for egress control.

**CSRF.** The three mutation forms — `POST /submit`, `/fetch`,
`/runs/{id}/requeue` — carry a stateless, action-bound HMAC token (`api.csrf`,
keyed by `csrf_secret`, independent of the Basic-auth password); a
missing/mis-signed token is refused before any state change. The review-workbench
mutations are instead gated by their unguessable per-run claim token.

## Provider seams

ASR, diarizer, embedder, and LLM sit behind typed protocols
(`voxint.clients.base`). The GPU services speak versioned HTTP
(`/v1/transcribe`, `/v1/diarize`, `/v1/embed`, `/healthz`) and share a
`MEDIA_ROOT` volume with the workers — no multipart uploads. The LLM stage
targets any OpenAI-compatible endpoint and is optional (`LLM_ENABLED=false`
by default); enhancement is **best-effort** — bounded ID-keyed batches, one
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
flow that genuinely blocks downstream processing — nothing enters it today.)

- **One resolver** (`adjudication/resolver.py`) settles attribution at read
  time for the workbench, the queue, and the transcript export alike:
  effective human decision (newest ledger row per label — corrections are
  appends) beats grounded cosine beats nothing. `llm_hint` names render as
  evidence, never as identity; `exclude` suppresses attribution, never text.
- **Reviewer slot**: claim columns on `pipeline_runs`, guarded by the same CAS
  `revision` as pipeline transitions. The claim token is an opaque per-claim
  secret required on every mutation; a re-claim rotates it, so a stale tab
  gets 409 instead of acting on a slot someone else holds. Claims expire on a
  TTL — an abandoned tab never dams the queue.
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
  but `/healthz` — fragments and media included; operator identity comes only
  from credentials. Startup refuses to bind off-loopback with the default
  password.
- **Media**: audio streams through a gate that requires the file to be
  DB-referenced, to resolve inside `MEDIA_ROOT` (symlink escapes rejected),
  and to carry a decodable audio stream per ffprobe (bounded subprocess,
  cached per path/size/mtime). Single-range HTTP semantics: 206/416,
  open-ended and suffix forms; multipart ranges are ignored per RFC.

## Worker orchestration (P3)

One Celery task, `voxint.run_pipeline`, drives a run through all stages via
the engine (task-per-stage would open an unclaimed window between handoffs
that recovery misreads as a crash; the engine already resumes an interrupted
run at its current stage). Failure handling is two-lane:

- **Transient** (`retryable` service errors: `saturated`, `model_unavailable`,
  transport failures) — the failed attempt stays in the `stage_runs` ledger,
  the run is CAS-requeued at the same stage — against the exact revision that
  failure produced, so a stale callback can never requeue a newer failure —
  and the task retries itself with exponential backoff. The attempt budget
  (`STAGE_MAX_ATTEMPTS`) counts transient *service* failures from the
  persisted ledger (restarts and broker loss never reset it); lease-expiry
  interruptions don't eat it, and the sweep applies the same ceiling to
  crash loops separately.
- **Deterministic** (`inference_failed`, protocol violations, bad media) —
  the run stays FAILED for the failure lane; `voxint requeue` is the explicit
  human override.

A beat task (`voxint.recovery_sweep`, every `RECOVERY_SWEEP_SECONDS`)
requeues runs whose stage lease expired and re-enqueues QUEUED runs whose
task evaporated with the broker (`QUEUED_RUN_STALE_SECONDS` grace so pending
retry countdowns aren't stepped on). Duplicate enqueues are safe by design —
claims and CAS arbitrate.

Timeout ordering that must hold: HTTP client timeout
(`GPU_HTTP_TIMEOUT_SECONDS`) **<** stage lease (`STAGE_LEASE_SECONDS`; the
diarize_embed stage gets its own longer `DIARIZE_EMBED_LEASE_SECONDS` because
it makes one diarization call plus several sequential embedding batches)
**<** Redis visibility timeout (`CELERY_VISIBILITY_TIMEOUT_SECONDS`, which
covers a whole run).
