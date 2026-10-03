# GPU sharing

Run Voxint's model services and another GPU service, such as a local LLM
server, on one GPU that cannot hold both at once. Voxint takes turns: it
borrows the GPU for audio work, then hands it back. This page covers how the
turns work, the HTTP contract for the small "broker" you run on the host to
stop and start the other service, the settings, and what to do when a
hand-over fails.

GPU sharing is opt-in and off by default. Leave it off when the GPU has room
for both, or when Voxint has the GPU to itself.

## Who it is for

A single host with one GPU, running Voxint's transcription, voice separation
and voice identity services (whisper, pyannote, titanet) next to a language
model that needs most of the same GPU memory. The language model also serves
Voxint's own LLM work (transcript enhancement, run assets, translation,
speaker research) through `LLM_BASE_URL`, so it has to be up for those steps
and down for the audio steps.

Supported on the Docker `gpu` and `rocm` tiers. Not available on the `cpu`
tier (no GPU to share) or the `metal` tier.

## How it behaves

Voxint splits each recording's pipeline into two lanes:

| Lane | Stages | Needs |
|---|---|---|
| GPU lane | `acquire`, `prepare`, `transcribe`, `diarize_embed` | the three model services |
| Post lane | `enhance_match`, `finalize`, plus run assets, translations and speaker research | the language model |

A row in the `gpu_phase` table records which lane owns the GPU. A periodic
task, `voxint.gpu_phase_tick` on the Celery queue `gpu_phase`, runs every
`GPU_PHASE_TICK_SECONDS` and moves the phase at most one step per tick:

| Phase | Who has the GPU | GPU lane | Post lane | Leaves when |
|---|---|---|---|---|
| `llm` | the other service | waits | runs | recordings are waiting and the phase has lasted `GPU_PHASE_MIN_DWELL_SECONDS` (and any backoff has passed), or the operator runs `audio-now` |
| `draining_post` | the other service | waits | waits | post-lane runs and LLM jobs already in progress finish |
| `acquiring` | being handed over | waits | waits | the broker grants the lease |
| `starting_services` | Voxint | waits | waits | whisper, pyannote and titanet all report healthy |
| `audio` | Voxint | runs | waits | the GPU lane is empty and min dwell has passed, or `GPU_PHASE_MAX_AUDIO_SECONDS` is reached, or the operator runs `release` |
| `draining` | Voxint | waits | waits | GPU-lane runs already in progress finish |
| `stopping_services` | Voxint | waits | waits | the three model services are stopped |
| `releasing` | being handed back | waits | waits | the broker confirms the release (`200` or `404`) and the other service is running again |
| `error` | neither (the hand-back failed) | waits | waits | a retry of the hand-back succeeds |

"Waits" means queued work stays queued. Nothing is lost; a run picks up where
it stopped once its lane opens. Every phase other than `llm` and `audio` pauses
both lanes, and so does `error`.

A failed step (the broker refuses, a service does not come up in time) returns
to `llm` and backs off before the next try: 30 seconds, doubling each time, up
to 15 minutes. The phase only lands in `error` when Voxint could not give the
GPU back, because then neither side can run.

## The broker contract

Voxint never stops the other service itself. It calls three HTTP endpoints you
provide, the "broker", which runs on the host and knows how to stop and start
the other service. Set their URLs in `GPU_LEASE_ACQUIRE_URL`,
`GPU_LEASE_RELEASE_URL` and `GPU_LEASE_STATUS_URL`.

Every call has a 10 second timeout. When `GPU_LEASE_TOKEN` is set, every call
carries `Authorization: Bearer <token>`. Bodies are JSON.

### Acquire

`POST` to `GPU_LEASE_ACQUIRE_URL`:

```json
{"holder": "voxint", "ttl_seconds": 600, "lease_id": null}
```

| Field | Type | Meaning |
|---|---|---|
| `holder` | `str` | Always `"voxint"`. |
| `ttl_seconds` | `int` | How long the lease should last without a renewal: `max(10 * GPU_PHASE_TICK_SECONDS, 600)`. |
| `lease_id` | `str \| null` | `null` for a new lease. While Voxint holds the GPU it sends the same request each tick with its current `lease_id`; that is the renewal. There is no separate heartbeat endpoint. |

Responses:

| Status | Body | Meaning |
|---|---|---|
| `200` | `{"lease_id": "b7c1", "expires_at": "2026-10-02T18:40:00Z"}` | Granted (or renewed). The other service is stopped and its GPU memory is free. `expires_at` may be `null`, in which case the broker does not expire leases and relies on an explicit release. |
| `202` | `{"state": "pending"}` | The broker is still stopping the other service. Voxint asks again on the next tick. |
| `409` | `{"holder": "someone-else", "retry_after_seconds": 300}` | Refused for now. `retry_after_seconds` is optional; Voxint waits at least that long before the next try. |

On a renewal (the request carried a `lease_id`), `404` or `409` means the lease
is lost: the broker no longer considers Voxint the holder.

### Release

`POST` to `GPU_LEASE_RELEASE_URL`:

```json
{"lease_id": "b7c1"}
```

| Status | Body | Meaning |
|---|---|---|
| `200` | any | Released, and the other service is running again. Voxint opens the LLM lane. |
| `202` | `{"state": "pending"}` | The broker is still restoring the other service. Voxint stays in `releasing` and asks again on the next tick. |
| `404` | any | No such lease; the GPU is already free. Voxint treats this as released and opens the LLM lane. |
| `500` | any, for example `{"error": "the language model did not come back"}` | Restoring the other service failed. The phase goes to `error`. |

Voxint opens the LLM lane only after a `200` or `404`. Any other answer, or no
answer, puts the phase in `error`.

### Status

`GET` `GPU_LEASE_STATUS_URL` returns who holds the GPU:

```json
{"state": "held", "holder": "voxint", "lease_id": "b7c1", "expires_at": "2026-10-02T18:40:00Z"}
```

| Field | Type | Meaning |
|---|---|---|
| `state` | `"free" \| "held"` | Whether a lease is active. |
| `holder` | `str \| null` | The holder's name, or `null` when free. |
| `lease_id` | `str \| null` | The active lease, or `null`. |
| `expires_at` | ISO 8601 `str \| null` | When the lease expires without a renewal, or `null`. |

Voxint uses it to reconcile after a restart, for example to find a lease it
still holds from before the restart.

## Enable it

1. Write and start a broker on the host (see [Writing a broker](#writing-a-broker)).
   Check it answers before you go further:

   ```bash
   curl -s http://127.0.0.1:9100/gpu/status
   ```

   The port and paths here are examples; use your broker's.

2. Set the settings in `.env`. The worker calls the broker from inside a
   container, where `127.0.0.1` is the container itself. The overlay maps
   `host.docker.internal` to the host (Docker's `host-gateway`, which works on
   Linux too), so point the URLs there. On Linux that name resolves to the
   Docker bridge address (often `172.17.0.1`), so the broker must listen on
   that address as well as `127.0.0.1`, not on loopback alone:

   ```bash
   GPU_PHASE_ENABLED=true
   GPU_LEASE_ACQUIRE_URL=http://host.docker.internal:9100/gpu/acquire
   GPU_LEASE_RELEASE_URL=http://host.docker.internal:9100/gpu/release
   GPU_LEASE_STATUS_URL=http://host.docker.internal:9100/gpu/status
   # GPU_LEASE_TOKEN=change-me
   ```

3. Bring the stack up with the `compose.gpu-phase.yaml` overlay after your
   compute tier:

   ```bash
   docker compose -f compose.yaml -f compose.gpu.yaml -f compose.gpu-phase.yaml up -d
   ```

   On ROCm, use `compose.rocm.yaml` in place of `compose.gpu.yaml`. Keep any
   other overlays your deployment uses.

4. Check it took:

   ```bash
   docker compose exec api voxint gpu-phase status
   docker compose exec api voxint doctor
   ```

The overlay adds one service, a `gpu-phase` worker. It is the only consumer
of the `gpu_phase` queue, and it starts and stops the whisper, pyannote and
titanet containers. For that it mounts the Docker socket, which gives that
container control over every container on the host. The same trade-off as the
[service controls overlay](service-controls.md#security) applies: accept it
only on a host where you trust everyone who can reach the stack.

The overlay leaves the regular worker's command alone, so a `--concurrency`
setting or other worker overrides keep working. The regular worker never runs
phase ticks. Without the overlay nothing consumes the ticks, so the phase never
changes: with `GPU_PHASE_ENABLED=true` and no `gpu-phase` service, recordings
wait in the queue indefinitely.

With the feature on, the phase starts in `llm` and the model services stay
stopped until there is audio work.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GPU_PHASE_ENABLED` | `false` | Turns GPU sharing on. When on, the three lease URLs are required. |
| `GPU_LEASE_ACQUIRE_URL` | empty | Broker acquire endpoint (`http` or `https`). |
| `GPU_LEASE_RELEASE_URL` | empty | Broker release endpoint. |
| `GPU_LEASE_STATUS_URL` | empty | Broker status endpoint. |
| `GPU_LEASE_TOKEN` | empty | Optional bearer token sent to the broker. Keep it private. |
| `GPU_PHASE_TICK_SECONDS` | `30` | How often the phase task runs. Minimum `5`. Also sets the lease TTL (`max(10 * tick, 600)` seconds). |
| `GPU_PHASE_MIN_DWELL_SECONDS` | `600` | Minimum time in `llm` before switching to audio for waiting recordings, and minimum time in `audio` before switching back on an empty queue. Stops the GPU from flapping. |
| `GPU_PHASE_MAX_AUDIO_SECONDS` | `7200` | Longest audio phase. After this the GPU goes back to the language model even with recordings waiting. Must be at least `GPU_PHASE_MIN_DWELL_SECONDS`. |

The dwell and maximum defaults are starting points. Tune them to how long your
recordings take and how long the other service takes to stop and start.

## What you see, and what to do

| Where | What it shows |
|---|---|
| Runs page | A banner while recordings wait for the GPU ("Waiting for the GPU. 3 recordings are queued; the GPU is serving the language model until the next audio window."), short notes while the GPU switches, and an alert in `error` or when the phase task has stopped running. The progress strip and the page summary carry the same note. |
| Settings > Status | A **GPU sharing** row with the phase in plain words. GPU sharing manages the three model services in every phase, so they have no start, stop or restart buttons, and an old form for one is refused. A model service GPU sharing stopped shows as off with the reason. |
| `voxint doctor` | A `gpu sharing` line. A model service GPU sharing stopped prints as `[off ]` and does not change the exit code. While Voxint holds the GPU, an `llm endpoint` that does not answer also prints as `[off ]`. |
| `voxint gpu-phase status` | Phase, since when, lease expiry, last error, failure count, retry time, pending operator request, and how many runs are waiting for or running on each lane. |

A stopped service is only called expected when the connection could not be
opened at all (refused, or the connect itself timed out) and the phase task is
running: the
stored phase is present and was written within the last
`max(3 * GPU_PHASE_TICK_SECONDS, 120)` seconds. A service that answers with an
error (HTTP 401, a 5xx, a malformed reply), one that accepts the connection
and then stalls, or a bad URL, is always reported as a failure. So is a stopped service while the phase task is not running or no
phase is recorded, because then nothing stopped it on purpose. The last error
is shown with control characters removed and cut to 300 characters.

Commands (run them where the CLI can reach the database, for example
`docker compose exec api ...`):

```bash
voxint gpu-phase status      # where things stand
voxint gpu-phase audio-now   # switch to audio work now, skipping dwell and backoff
voxint gpu-phase release     # hand the GPU back now, or retry after an error
```

`audio-now` and `release` record a request; the phase task acts on it over the
next few ticks. Recording a request does not count as a run of the phase task,
so it never hides the "has not run since" warning, and with no phase recorded
yet the request waits for the task's first run. With GPU sharing off, they exit with an error and change
nothing; `status` says GPU sharing is off.

How a request plays out depends on the phase:

- `release` during `audio` starts the normal hand-back: no new GPU work starts,
  and runs in progress finish first.
- `release` during the hand-back (`draining`) stops the model services at once,
  even while audio runs are still in progress. Those runs fail their current
  stage and retry it in the next audio window.
- `release` while already in `llm` holds off the next switch to audio for
  `GPU_PHASE_MIN_DWELL_SECONDS`.
- `release` in `error` retries the hand-back now instead of waiting for the
  backoff.
- `audio-now` made while the GPU is being handed back, or while Voxint recovers
  from an error, is kept and acted on at the next tick in `llm`.

A failed or timed-out acquire also goes through `releasing` before `llm`, so the
broker is told to release any lease it may have granted before language-model
work resumes.

| Problem | What you see | What to do |
|---|---|---|
| Recordings stay queued | Banner "Waiting for the GPU", phase `llm` | Expected until min dwell passes. Run `voxint gpu-phase audio-now` to start now. If `status` shows failures and a retry time, check the last error and the broker. |
| The broker refuses or is down | Phase back in `llm` with a last error and a retry time | Fix the broker. Voxint retries on its own with backoff, or run `audio-now` to retry at once. |
| A model service does not come up | Phase goes back through `stopping_services` and `releasing` to `llm`, with a last error | Check the service's logs (`docker compose logs whisper`). |
| No phase recorded | `voxint gpu-phase status` shows phase `none`; doctor warns "no phase recorded yet". Both lanes wait. | The phase task writes the row on its next run. If it stays this way, check that the `gpu-phase` worker is running (`docker compose ps gpu-phase`). |
| The phase task is not running | Doctor and the Status page warn "the GPU sharing task has not run since ...". Phases do not change. | Check that the `gpu-phase` service from `compose.gpu-phase.yaml` is running (`docker compose ps gpu-phase`, `docker compose logs gpu-phase`). |
| The GPU could not be handed back | Phase `error`, alert on the Runs page, `gpu sharing` warning in doctor and on the Status page. LLM work is paused. | Fix the cause in the last error (often the broker), then run `voxint gpu-phase release` to retry. Voxint also retries on its own with backoff. |

## Turn it off safely

Turning GPU sharing off while Voxint holds the GPU leaves the lease held and the
other service stopped. Hand the GPU back first:

1. Run `voxint gpu-phase release` and wait until `voxint gpu-phase status`
   shows phase `llm`.
2. Set `GPU_PHASE_ENABLED=false` and bring the stack up without the overlay:

   ```bash
   docker compose -f compose.yaml -f compose.gpu.yaml up -d --remove-orphans
   ```

With the feature off, the model services run all the time. Stop the other
service, or move it to another GPU, before you do this, or the two will compete
for GPU memory.

## Limitations

- A run already in the GPU lane finishes its whole GPU segment, even past
  `GPU_PHASE_MAX_AUDIO_SECONDS`. The maximum stops new GPU work from starting;
  it does not cut a running stage short.
- If the lease is lost while a stage is running, the stage retries and can
  fail. Requeue it from the run's page once the GPU is back.
- The synthdetect plugin's own GPU service is not phase-aware. Leave it off on
  a shared GPU, or budget its memory next to the language model.
- The default dwell and maximum times are starting points, not tuned values.

## Writing a broker

The broker is yours: a small HTTP service on the host that owns the other
service's lifecycle. It can be a few dozen lines in any language. It needs to:

1. **Acquire.** On a new lease, stop the other service, wait until its GPU
   memory is actually free, then answer `200` with a new `lease_id`. While the
   stop is still in progress, answer `202 {"state": "pending"}`; Voxint asks
   again on the next tick. If someone else holds the GPU, answer `409`. On a
   renewal with the current `lease_id`, extend the expiry and answer `200` with
   the same `lease_id`; with an unknown or expired one, answer `404`.
2. **Release.** On the current `lease_id`, start the other service again. Answer
   `202 {"state": "pending"}` until it is healthy (for an LLM server, until it
   answers a request), and `200` only then: Voxint opens the LLM lane on the
   `200`, and opening it early would send work to a model that is not up yet.
   If the other service fails to come back, answer `500`. On an unknown
   `lease_id`, answer `404`.
3. **Status.** Report the current lease, or `"free"`.
4. **Expire leases Voxint stops renewing.** Voxint renews every tick, so a lease
   that passes its `expires_at` means Voxint stopped (crashed, or the host lost
   power). On expiry, try to give the GPU back to the other service. First check
   that the GPU memory was freed: if Voxint's model services are still loaded,
   starting the other service would run the GPU out of memory, so leave it
   stopped and tell the operator (a log line, a notification) instead.
5. **Keep one holder at a time,** and keep its lease in memory or on disk so a
   broker restart does not forget who holds the GPU.

A quick manual test, with your broker's address:

```bash
curl -s -X POST http://127.0.0.1:9100/gpu/acquire \
  -H 'Content-Type: application/json' \
  -d '{"holder": "voxint", "ttl_seconds": 600, "lease_id": null}'
curl -s http://127.0.0.1:9100/gpu/status
curl -s -X POST http://127.0.0.1:9100/gpu/release \
  -H 'Content-Type: application/json' -d '{"lease_id": "b7c1"}'
```

Related: [service-controls.md](service-controls.md) for the Docker socket
setup, [operations.md](operations.md) for day-to-day operation, and the
[docs index](README.md).
