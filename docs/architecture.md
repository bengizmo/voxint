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

## Data model (alembic revisions 0001–0002)

| Table | Role |
|---|---|
| `media_items` | media identity — one row per source file |
| `pipeline_runs` | execution state + CAS revision |
| `stage_runs` | per-stage attempt ledger **and execution claim** (worker id, lease, status, timing, error, metrics) |
| `audio_artifacts` | derived files (preprocessed audio, chunks, exports) |
| `audio_chunks` | chunk boundaries for long-file processing |
| `transcript_segments` | raw ASR text (immutable) + `enhanced_text` beside it + `suspect` soft-tag |
| `diarization_turns` | run-scoped observation ledger: one row per turn — interval, label, overlap, and the window's embedding outcome (vector + space, or an auditable `skip_reason`) |
| `speakers` | the grown speaker roster |
| `speaker_embeddings` | `vector(192)` + `embedding_space` tag |
| `speaker_assignments` | **machine proposals** (method, confidence, grounded flag) |
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

## Provider seams

ASR, diarizer, embedder, and LLM sit behind typed protocols
(`voxint.clients.base`). The GPU services speak versioned HTTP
(`/v1/transcribe`, `/v1/diarize`, `/v1/embed`, `/healthz`) and share a
`MEDIA_ROOT` volume with the workers — no multipart uploads. The LLM stage
targets any OpenAI-compatible endpoint and is optional (`LLM_ENABLED=false`
by default). Test fakes satisfy the same protocols, which is how the
end-to-end contract tests run without a GPU.

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
(`GPU_HTTP_TIMEOUT_SECONDS`) **<** stage lease (`STAGE_LEASE_SECONDS`, which
covers a whole stage — diarize_embed makes several sequential calls) **<**
Redis visibility timeout (`CELERY_VISIBILITY_TIMEOUT_SECONDS`, which covers a
whole run).
