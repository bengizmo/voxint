# Plan — Run notifications / webhooks (issue #12)

**Created:** 2026-08-16 · **Issue:** #12 · **Branch:** `feat/run-notifications-webhooks`

## Goal

Notify an operator's configured endpoint when a run reaches a **notifiable
transition** (`awaiting_adjudication`, `completed`, `failed`) via a single
**signed webhook POST**, delivered **at-least-once** without ever holding
pipeline correctness hostage to remote latency or failure. Opt-in, off by
default. Email, in-console feed, multiple endpoints, templates, and proxy
support are explicitly **out of scope** (anti-bloat; single-operator audience).

## Design of record (transactional outbox)

Reviewed by codex (planner role) 2026-08-16 — see "Review notes". Verdict:
"Keep the transactional outbox and defer email/feed, but tighten event
semantics and delivery claiming."

### Why an outbox (not inline POST / post-commit Celery / run scan)
- **Inline POST** holds pipeline correctness hostage to remote latency/failure.
- **Post-commit Celery publish** keeps a commit→broker-loss window (the exact
  gap `_publish_or_defer` already fights elsewhere).
- **Periodic scan of `pipeline_runs`** cannot reconstruct repeated
  `FAILED`/`AWAITING_ADJUDICATION` arrivals without becoming an implicit,
  less-precise outbox.
- The outbox insert shares the transition's transaction → atomic at-least-once.

### Phase 1 — Event contract
- Notifiable transitions: `awaiting_adjudication`, `completed`, `parked failed`
  (a FAILED that has *settled*, i.e. not immediately requeued — see FAILED
  semantics below).
- Immutable payload (versioned): `schema_version`, `event`, `run_id`,
  `transition_revision`, `occurred_at`, `delivery_id`. **Minimal by default —
  omit `run.error`** (leak risk); add only if a real need appears.
- Receiver contract: at-least-once; deduplicate on `delivery_id`; verify
  `X-Voxint-Signature` over `timestamp + "." + body` within a clock-skew window.

### Phase 2 — Persistence + atomic emission
- **Migration `0015`** (down_revision `0014`): `notification_deliveries`
  - `id` (uuid pk), `run_id` (fk → pipeline_runs, ondelete CASCADE),
    `event` (text/enum), `transition_revision` (int), `payload` (jsonb),
    `status` (`pending|in_flight|delivered|dead|suppressed`),
    `attempts` (int, default 0), `next_attempt_at` (timestamptz),
    `lease_expires_at` (timestamptz null), `delivered_at` (timestamptz null),
    `last_error` (text null, capped+redacted), `created_at` (timestamptz).
  - `UNIQUE(run_id, transition_revision)` — the occurrence key. Each distinct
    arrival is one row; delivery retries reuse it. **Not** `(run_id, event)`.
  - Partial index on due work: `(status, next_attempt_at)` WHERE
    `status IN ('pending','in_flight')`.
- **Emission** in `cas_update_run`, immediately after `rowcount == 1`, only when
  the target status is notifiable AND `settings.notify_enabled`. Delegates to a
  persistence-only helper `voxint.notify.record_transition(session, snapshot,
  settings)` — **no HTTP/Celery imports in `transitions.py`**. Insert uses
  `ON CONFLICT (run_id, transition_revision) DO NOTHING` for concurrency safety.
  Rolls back atomically with the transition.
  - `cas_update_run` currently takes no `settings`; thread it through (optional
    param, default None → no emission) so non-worker callers (tests) are
    unaffected. Confirm all 6 engine call sites + ingest + tasks.

### Phase 3 — Safe delivery sweep
- Opt-in beat entry `notify-sweep` in `build_beat_schedule`, gated on
  `notify_enabled`; task **re-checks the flag at runtime** (gc_sweep precedent).
- **Claim → commit → deliver → record** (never hold a tx across network I/O):
  1. In a short tx: `SELECT ... FOR UPDATE SKIP LOCKED` oldest due rows
     (`status='pending'` and `next_attempt_at<=now`, plus `in_flight` rows whose
     `lease_expires_at<now` — lease reclaim), bounded by `notify_batch_limit`;
     set `status='in_flight'`, `lease_expires_at=now+lease`; commit.
  2. **FAILED suppression**: before POSTing a `failed` event, re-read the run;
     if it has advanced past `transition_revision` (e.g. requeued), mark the row
     `suppressed` and skip. A short `notify_initial_delay_seconds` on FAILED rows
     lets a synchronous requeue settle first.
  3. POST outside any tx: deterministic JSON body, signed headers
     (`X-Voxint-Delivery`, `X-Voxint-Timestamp`, `X-Voxint-Signature` =
     HMAC-SHA256(secret, `ts + "." + body`)). Bounded timeout, **no redirects**,
     `trust_env=False`, **address-pinned transport** reused from
     `research/fetch.py` (revalidate host is public on every attempt — DNS
     rebinding). 2xx → `delivered`; else retry.
  4. Retry: capped exponential backoff + jitter; `attempts >= notify_max_attempts`
     → `dead`.

### Phase 4 — Operational bounds
- Log `delivery_id, run_id, event, attempt, outcome` — never URL/query/secret/
  payload. Redact secret+URL from any error text (`redact(extra_secrets=...)`).
- Bounded cleanup of old `delivered`/`suppressed` rows (fixed retention); keep
  `dead` rows until operator action.
- Docs: `docs/operations.md` (setup + at-least-once + receiver verification +
  a signature-check snippet), `.env.example` entries, CHANGELOG `[Unreleased]`.
- **Contract tests**: config default/validator coverage; if `.env.example`
  pin-parity contract covers new keys, extend it.

## Config (pydantic Settings)
- `notify_enabled: bool = False`
- `notify_webhook_url: str = ""` (validated public http/https, no creds; via
  `netcheck.parse_http_url`)
- `notify_webhook_secret: str = ""` (credential — redacted; min strength when
  enabled, mirror `csrf_secret` validator)
- `notify_sweep_seconds: PositiveSeconds` (default e.g. 30)
- `notify_max_attempts: int = Field(default=8, ge=1)`
- `notify_batch_limit: int = Field(default=50, ge=1)`
- `notify_lease_seconds: int = Field(default=60, gt=0)`
- `notify_initial_delay_seconds: int = Field(default=10, ge=0)` (FAILED settle)
- `notify_backoff_base_seconds` / cap — mirror existing backoff idioms.
- `@model_validator`: enabled ⇒ URL present+public and secret strong; else
  `SettingsError` (sanitized, no secret in message).

## Correctness traps (from review — must be covered by tests)
1. Emission inside a tx that later rolls back → both vanish (integration test).
2. Stale-CAS (`rowcount!=1`) → no emission.
3. Disabled → no emission; pending rows survive a later disable.
4. Concurrent sweeps: SKIP LOCKED + lease → no double-claim; crash mid-delivery
   → lease reclaim redelivers (duplicate → receiver dedup on `delivery_id`).
5. Deterministic body ↔ signature (sign exact transmitted bytes; JSONB ≠ bytes).
6. Replay: timestamp in signature + skew window (documented receiver-side).
7. SSRF/DNS-rebind: address-pinned transport, no redirects, revalidate per attempt.
8. Unbounded growth: cleanup of delivered/suppressed.

## Sequencing / checkpoints
- **A (foundation, self-contained):** config settings + validator + migration
  0015 + ORM model + config unit tests. Commit.
- **B:** emission in `cas_update_run` + `notify.record_transition` + integration
  tests (atomicity, stale-CAS, disabled, idempotency). Commit.
- **C:** delivery sweep (claim/lease, signing, address-pinned transport, FAILED
  suppression, backoff/dead) + tests. Commit.
- **D:** cleanup task, docs, `.env.example`, CHANGELOG, contract tests. Commit.
- Multi-model `/code-review` on the full diff before FF-merge to `main`.

## Review notes
- **codex (planner), 2026-08-16** — full design review. Applied: emit inside
  `cas_update_run` post-rowcount (rejected call-site + ORM-listener + post-commit
  Celery); gate emission on `notify_enabled`; key on `(run_id,
  transition_revision)` not `(run_id, event)`; FAILED-suppression at claim time +
  initial delay ("FAILED is not necessarily terminal"); claim-lease `in_flight`
  state (SKIP LOCKED insufficient once the claim tx commits); sign
  `timestamp+body` with delivery-id + timestamp headers; `trust_env=False`,
  reuse pinned-address transport, no redirects; minimal payload (omit
  `run.error`); bounded cleanup; renamed "terminal" → "notifiable" transitions.
  Deferred per review: email, feed, proxy, multiple endpoints, templates,
  redirect-following, mgmt UI.
