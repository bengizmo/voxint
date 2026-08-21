# Changing pipeline models

*How to point transcription or diarization at a different model, when that is
worth doing, and what you give up by doing it.*

Voxint ships with two models chosen and measured for you: **whisper large-v2**
for transcription and **pyannote speaker-diarization-3.1** for speaker
separation. They are built into Voxint, run without any internet connection, and
are the only two models whose accuracy has actually been tested. For almost
everyone, the right choice is to leave them alone.

This guide is for the exception: you have a specific reason to run a different
model, you understand it has not been measured, and you want to do it safely.
Everything here is optional and advanced. If that is not you, you can close this
page.

> ⚠️ Changing a model is not free. The shipped models are validated, which means
> Voxint's accuracy numbers were measured against them. Any other model is an
> unvalidated mechanism: it will run, but no one has checked how good its results
> are, and some popular alternatives are known to be worse. Whisper v3 and turbo,
> for example, hallucinate text that was never spoken. Only change a model if you
> can live with unmeasured results.

---

## Validated versus unvalidated

Two words show up throughout Voxint, and they mean something exact here.

- **Validated** means the shipped default: its accuracy was measured and the
  results are what Voxint's documentation promises. There are exactly two, one
  per configurable stage.
- **Unvalidated** means anything else. It is a supported mechanism, so Voxint
  will load it and run it, but its accuracy is unknown. You are on your own for
  judging whether the output is good enough.

The **Settings** page tells you which one each service is running right now. Open
the console, click **Settings**, and find the **Pipeline models** panel. Each
service shows its model and whether it is the validated default or an unvalidated
override. That panel reads the live services every time you load it, so it is the
honest place to confirm a change took effect after you restart a service.

The third model, **speaker embedding** (TitaNet), is fixed. Voxint depends on it
for speaker identity, so it is not something you change, and the panel shows it
without a warning.

---

## How a change is applied

Model choices live in your `.env` file, the same file the installer wrote. You
edit one or two lines, then restart the single service that changed. Only that
service restarts; the rest of Voxint keeps running.

The restart command uses the same compose files you installed with. If you are
not sure which they are, they are listed in
[the operations guide](../operations.md) and in [setup](../setup.md) for your
hardware. For an NVIDIA install, restarting only the transcription service looks
like this:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d whisper
```

Compose recreates just the `whisper` container with the new settings and leaves
everything else running. Swap `compose.gpu.yaml` for `compose.cpu.yaml` or
`compose.rocm.yaml` to match your tier, and swap `whisper` for `pyannote` when
you change the diarization model.

> After the service comes back up, reload **Settings > Pipeline models** to
> confirm it reports the model you expected. A slow first start is normal (see
> below); if the service reads **Unavailable** for more than a few minutes,
> check its logs.

---

## Changing the transcription model (whisper)

To use a transcription model other than large-v2, set three keys in `.env`. All
three are required together, and the whisper service refuses to start (naming
exactly what is missing) if you set the model without the other two:

| Key | What to set it to |
|---|---|
| `WHISPER_MODEL` | The model you want, for example a Hugging Face repo id. |
| `WHISPER_ALLOW_DOWNLOAD` | `1`, to permit the one-time download. |
| `WHISPER_REVISION` | The model's full 40-character commit hash (lowercase). |

The revision has to be a full commit hash, not a branch name like `main` and not
a short hash. This is deliberate: it pins the exact weights you downloaded so a
later restart cannot quietly pull a different version.

The first time the service starts with a new model it downloads the weights,
which is slow and can be several gigabytes (roughly 3 GB for a large model). The
download goes into a separate cache that never overwrites the built-in large-v2,
so you can always get back to the validated default by removing these keys and
restarting. Later starts reuse the cache and are fast.

> A larger transcription model uses more video memory (VRAM), and it **replaces**
> the large-v2 model in that GPU slot rather than running alongside it. If a
> model is too big for your GPU, the service will fail to load it. There is no
> partial fallback: either it fits and runs, or it does not start.

---

## Changing the diarization model (pyannote)

To use a different speaker-diarization pipeline, set these keys:

| Key | What to set it to |
|---|---|
| `DIARIZER_MODEL_NAME` | The Hugging Face pipeline id to load. |
| `DIARIZER_REVISION` | Optional. A commit to pin the pipeline to; blank uses the repo's default. |
| `HF_TOKEN` | Only if the pipeline is gated on Hugging Face and needs an account. |

Pinning `DIARIZER_REVISION` is worth doing so the model recorded against each run
is reproducible instead of drifting with whatever the repo's default branch
points at today.

> One honest caveat about the pin. For the shipped pyannote version (3.1.1), the
> revision pins the download of the pipeline's configuration file, and Voxint
> records it as the *requested* revision, not a guarantee of the exact resolved
> build. It is a reproducibility aid, not a cryptographic proof. The vendored
> default does not use this key at all (its configuration is itself the pin), and
> setting it there is ignored with a warning.

A gated pipeline also needs a Hugging Face account token in `HF_TOKEN`. Voxint
never displays that token back to you, including in the Settings panel.

---

## When a model is slower

A bigger or slower model changes how long a stage takes, and Voxint has time
limits that decide when to give up on a stage it believes has hung. If you move
to a much slower model on slower hardware, a healthy run can bump into those
limits and be treated as dead, then re-run, doubling the work.

If you see that happen, the two budgets to raise are the per-call timeout
(`GPU_HTTP_TIMEOUT_SECONDS`) and the stage lease that must always be larger than
it. The rules for sizing them so they stay consistent are in
[Timeouts, leases, and compute tiers](../timeouts-and-leases.md); read that
before changing either, because setting them incorrectly causes exactly the
double-run problem it is meant to prevent.

---

## A note for Apple Silicon (Metal) installs

The native macOS path runs the model services directly on your Mac rather than in
containers, and it pins the shipped validated models on purpose. The
alternate-model keys above apply to the Docker installs (CPU, NVIDIA, AMD) only;
they are not read on the Metal tier today. If you run the native path and need a
different model, you are for now on the validated default. See
[the native macOS preview](../native-macos-preview.md) for what that path does
and does not cover.

---

## For developers

Voxint's transcription service can host more than one inference engine behind a
fail-closed registry (the built-in CT2 path, and others added over time). Adding
an engine is a code change in `services/whisper/`, not an `.env` setting, and it
carries the same rule as swapping a model: a new engine has to earn a measured
parity verdict before it is validated. The service contracts every model service
must satisfy are documented in [GPU service contracts](../gpu-contracts.md), and
the contribution process is in [CONTRIBUTING.md](../../CONTRIBUTING.md).
