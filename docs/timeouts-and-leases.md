# Timeouts, leases, and compute tiers

Voxint's pipeline distinguishes a **slow healthy run** from a **hung run** with
a chain of nested time budgets. Slower compute tiers (CPU, and to a lesser
degree ROCm) stretch inference wall-time by an order of magnitude, so the chain
is a first-class design surface: get it wrong and a healthy 4-hour CPU
transcription is indistinguishable from a dead worker — recovery reclaims the
stage, a second worker re-executes it, and the run pays twice.

## The inequality chain

Every budget must fit strictly inside the next one out:

```
per-call HTTP timeout            gpu_http_timeout_seconds
  + persistence margin           GPU_CALL_PERSISTENCE_MARGIN_SECONDS (600 s)
< stage lease                    stage_lease_seconds / diarize_embed_lease_seconds
< sum of all six stage leases    (acquire + prepare + transcribe + diarize_embed
                                  + enhance_match + finalize)
≤ Celery visibility horizon      celery_visibility_timeout_seconds
```

with the recovery sweep (`recovery_sweep_seconds`, 5 min) running much more
often than any lease so expiry is noticed promptly.

Why each link exists:

- **HTTP timeout < stage lease.** A stage that calls a model service must
  survive the call *and still persist its results* inside its lease. If the
  timeout could reach the lease, a slow-but-successful call would return to a
  stage that no longer owns the run: recovery has reclaimed it, a second worker
  is executing the same stage, and both race to commit — duplicate execution.
  The 600 s margin covers result persistence + the stage commit.
- **diarize_embed's dedicated lease.** That stage makes one diarization call
  *plus N sequential embedding batches* under a single lease. The static
  validator can only enforce the single-call floor; sizing the lease for the
  batch sum is operational: budget roughly
  `diarization_call + ceil(windows / 512) × embed_call` for your longest media,
  and raise `DIARIZE_EMBED_LEASE_SECONDS` if you routinely process very long
  recordings on a slow tier.
- **Sum of leases ≤ visibility horizon.** `run_pipeline` is one acks-late
  Celery task that advances through all six stages. If Redis redelivered it
  while a worker legitimately held a late stage, a second worker would start
  the same run from its current stage. The horizon must therefore outlast every
  lease held back to back (enforced by `_celery_visibility_covers_all_leases`).
- **acquire is download-bound, not inference-bound.** Its chain
  (`acquire_timeout_seconds` + kill/hash/publish tail < `acquire_lease_seconds`)
  is validated separately and is *not* scaled by compute tier — network speed
  doesn't change with the inference backend.

All four links are enforced at startup by `Settings` validators
(`src/voxint/config.py`); a violating combination refuses to boot rather than
open a duplicate-execution window in production.

## Compute-tier profiles

`COMPUTE_TIER` selects a named **timing profile** (default `gpu`):

| Tier | Meaning | Timing |
|------|---------|--------|
| `gpu` | CUDA services (`compose.gpu.yaml`) | Baseline defaults (4 h call / 6 h lease / 12 h diarize_embed / 48 h visibility) |
| `rocm` | AMD-accelerated services | GPU-class timing (same defaults) |
| `metal` | Bare-metal Apple Silicon services (`compose.metal.yaml` + native launcher) | GPU-class timing (same defaults). v1 runs whisper on CPU under these budgets; a metal-specific factor is a post-measurement decision. |
| `cpu` | CPU-only services (`compose.cpu.yaml`) | Baseline × `CPU_TIER_TIMEOUT_FACTOR` (4×): 16 h call / 24 h lease / 48 h diarize_embed / 192 h visibility |

Design decision — **per-tier static profile, not per-request scaling**. The
alternative (scaling each request's timeout by media duration) was rejected:
clients are process-cached with one static timeout, per-request scaling still
needs a static lease to fit inside (leases are claimed before media duration is
known), and it turns every timeout bug into a per-request heisenbug. A static
profile keeps the whole chain inspectable at startup.

Rules:

- The profile scales only `gpu_http_timeout_seconds`, `stage_lease_seconds`,
  `diarize_embed_lease_seconds`, and `celery_visibility_timeout_seconds`
  (`TIER_SCALED_TIMING_FIELDS`), and only when the field is at its default.
- **An explicitly-set env value always wins** — the profile never overrides an
  operator decision. If your explicit value breaks the chain against the other
  (scaled) values, startup fails with the exact inequality named; set the
  related fields explicitly too.
- The factor is deliberately coarse. It does not try to predict your CPU; it
  keeps the *ratios* of the chain intact while giving slow tiers real headroom.
  Operators with unusually slow (or fast) hardware should set explicit values
  for all four fields.

## What "too slow" looks like operationally

If a stage genuinely exceeds its lease on a slow tier, the run is reclaimed
and retried (`stage_max_attempts` budget), and the log line from
`apply_run_preferences` / recovery names the lease that expired. The fix is
configuration (raise the lease / pick the right tier), not code. The CPU tier's
release gate includes a duplicate-execution test with an artificially slow
service precisely to keep this failure mode loud in CI rather than silent in
production.
