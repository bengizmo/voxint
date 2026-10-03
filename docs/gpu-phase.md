# Shared GPU phases

GPU phase coordination is disabled by default. With `GPU_PHASE_ENABLED=true`,
Celery beat schedules `voxint.gpu_phase_tick` on the `gpu_phase` queue at
`GPU_PHASE_TICK_SECONDS` intervals. Each message expires after one interval.
Run beat and a worker consuming that queue; a worker without `-Q` consumes all
three declared queues (`celery`, `post`, `gpu_phase`). A dedicated phase worker
keeps ticks from waiting behind long audio jobs.

Configure `GPU_LEASE_ACQUIRE_URL`, `GPU_LEASE_RELEASE_URL`, and
`GPU_LEASE_STATUS_URL` for an external host broker. Voxint does not provide the
broker. `GPU_LEASE_TOKEN`, when nonempty, is sent as a bearer authorization
header. Service control must be available through the existing service control
configuration; otherwise the phase stays in `llm` with an error recorded.

The broker's acquire endpoint accepts a POST body with `holder: "voxint"`,
`ttl_seconds`, and `lease_id` (null for a new lease). TTL is four tick intervals
or 120 seconds, whichever is greater. A current lease ID renews the lease:
there is no separate heartbeat endpoint. A 200 response supplies `lease_id`
and optional ISO8601 `expires_at`; absent or null expiry means explicit release
only. A 202 response with `state: "pending"` requests another attempt. A 409
response supplies `holder` and optional integer `retry_after_seconds`. Renewal
404 or 409 means the lease has been lost.

The release endpoint accepts POST `{"lease_id": "..."}`; 200 and 404 both mean
success. GET status returns `state` (`free` or `held`), `holder`, `lease_id`, and
optional `expires_at`. Every hook call has a ten-second timeout. Failed calls
produce bounded diagnostics that exclude response bodies and exception text.

The normal phase sequence is:

```text
llm → draining_post → acquiring → starting_services → audio
    → draining → stopping_services → releasing → llm
```

Only `llm` opens post processing and LLM jobs; only `audio` opens GPU work.
Draining waits for work already in flight. Acquisition and service readiness
each have a ten-minute window. Active audio phases renew on every tick.
Services must be confirmed stopped before the lease is released. Failed
teardown enters `error`, with retries starting at 30 seconds and capped at
15 minutes. An operator release request skips that retry delay.

Each tick holds a PostgreSQL transaction advisory lock and performs at most
one phase transition. Another tick returns immediately if the lock is held.
In `llm`, service inspection and broker status detect leaked services or a
lease still held by Voxint, even when no audio work is queued. Startup repeats
idempotent service starts when inspection shows they are needed. A restart in
`releasing` checks services again before returning the lease.

Opening a lane commits its state before publishing queued work. Pipeline
publication uses `RECOVERY_PUBLISH_BATCH_SIZE`; reopening the post lane also
publishes at most 100 jobs each for run assets, translations, and research,
oldest first. Broker publication failures leave jobs queued for the recovery
sweep. A teardown caused by a failure preserves its error and retry delay when
it returns to `llm`.
