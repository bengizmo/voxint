# ADR 0009: Voice cloning plugin defaults and consent model

> **Status:** Accepted 2026-09-29 (voice cloning epic). Records the defaults
> for the `voiceclone` plugin, and the two operator overrides that may relax
> them, before any code lands. The engine selection paragraph is filled in
> after the packaging and falsification gates report; everything else here is
> decided now. Engine recorded in section 7 on 2026-09-30 (issue #663).

## Context

Voxint is adding a second greenfield plugin: speech synthesis and voice
conversion in a speaker's voice (`voiceclone`). It follows the shape fixed by
[ADR 0006](0006-plugin-scope-native-vs-greenfield.md) and worked through in the
`synthdetect` plugin: one model-service container behind a compose overlay, a
job lane on the post queue, a standalone page, a template-only run-detail
panel, and CLI commands.

A generator changes the product's risk profile in a way a detector does not.
The audience is journalists, researchers and educators working with recordings
of real people. Two facts from the landscape research shape the defaults:

- Modern generators evade passive detectors. ASVspoof-trained AASIST detectors
  sit near chance against flow-matching TTS (VoxENES 2026), and Voxint's own
  `synthdetect` scores Chatterbox output at roughly 45% EER before fine-tuning
  and 26% after (issue #252). A shipped generator therefore needs a
  deterministic provenance signal, not only a classifier.
- Voice conversion preserves prosody, wording and context. Converting a
  source's timbre reduces the chance of voice-based identification; it does not
  make the person unidentifiable.

## Decision

### 1. Every output is watermarked, verified, and labelled

- By default the service applies the generator's watermark to every output.
  There is no per-request flag. The only way to turn it off is the
  instance-wide watermark override in section 8.
- The service re-detects the watermark on its own output before responding. A
  failed self-check is an error response, never a silent unmarked file.
- The worker re-verifies the watermark on the final file it publishes, at the
  pipeline's 16 kHz rate, and refuses to publish on failure.
- The two watermark checks are skipped only while the watermark override is on.
  Provenance, the metadata chunk, the badge and the generation log below apply
  to every output either way.
- Every published output carries provenance in a plugin table and in the WAV
  metadata chunk: engine inference-space id, weights sha, watermark state
  (verified, or disabled by override), consent basis (profile record, or
  override), input hashes, profile id, operator, timestamp. Assembled
  outputs (conversion and dubbing) also carry a per-line provenance row.
- Wherever an output plays or is listed, the console shows a "synthetic" badge.

### 2. Consent is recorded per voice profile, with two purposes

A voice profile is the unit of consent. It holds:

- the enrolled roster speaker it represents (`speakers.id`, a cross-run
  identity, so consent follows the person across recordings);
- a reference clip **copied** into plugin-owned storage and hashed. A profile
  never points into run audio that media reclaim can delete;
- a consent purpose: `clone` (new words may be synthesized in this voice) or
  `risk_reduction` (this person's turns may be converted into a synthetic
  voice). The two are separate records. Protecting a source does not require
  the source's consent to be cloned;
- an operator attestation (who, when, free-text note), and an optional stored
  consent clip. The clip is stored only; it is never verified against the
  voice, and the code must not grow a similarity check that pretends to.

Enrollment enforces that the chosen reference turns belong to the consented
speaker and pass duration, signal-to-noise and no-overlap gates. The plugin API
accepts only a profile id; the reference is resolved server-side. Nothing an
operator can reach passes an arbitrary reference path.

While the consent override in section 8 is on, it stands in for the
attestation step for both purposes. The profile itself still exists: the
reference clip is still copied, hashed and put through the same gates, and
the API still accepts only a profile id.

Consent enforcement is procedural, not cryptographic. The tool makes the
honest path the default and the dishonest path visible in an append-only
generation log. The operator is the accountable party.

### 3. Revocation blocks the future and keeps the past

Revoking a profile blocks new jobs immediately. Existing outputs are flagged
`consent_revoked` and offered for bulk delete. Nothing is deleted silently and
the generation log is never edited.

### 4. Conversion targets are synthetic voices only

Voice risk reduction converts into a vendored pack of synthetic target voices
that belong to no real person, shipped as a sha-pinned release asset with
provenance. A real third party's voice is never a conversion target.

### 5. The feature is called voice risk reduction, never anonymization

User-facing copy, route names and documentation say "voice risk reduction".
The console shows the measured re-identification score (maximum TitaNet cosine
of the converted output against the source speaker's enrollment and against
every roster centroid) beside the matching threshold, as a number, never as a
"safe" badge. The caveat is not dismissible:

> This reduces the chance that a voice is recognised. Wording, delivery,
> and what is said can still identify a person.

Converted turns replace the original audio. The original is never mixed
underneath for ambience continuity, because that leaks the source voice.

### 6. Every generation is an explicit operator action

There is no autogenerate. The plugin does not enqueue work from the
run-completed hook. Generation routes require the admin dependency in
multi-user mode and a per-action CSRF token.

### 7. Hardware lane and engine

Version 1 is CUDA only, matching `synthdetect`. Settings copy and `voxint
doctor` say so; ROCm, CPU and Apple Silicon are out of scope until an ONNX
route is measured.

One generation is in flight per service at a time, held by an advisory lock in
the worker, because the plugin's Celery tasks run on the post queue rather
than the serialized GPU lane.

The engine is chosen by two gates before the service is built: a packaging
gate (every dependency and weight sha-pinned into a reproducible image, no
network at build or run) and a falsification gate on Voxint-realistic audio
(similarity on 3 s phone-quality references measured with Voxint's own TitaNet
embedder, word error rate with faster-whisper large-v2, watermark survival
through the 16 kHz chain, peak VRAM measured on maintainer hardware). A
license allowlist
constant (engine, license, weights sha) is checked at boot.

**Engine selection (recorded 2026-09-30): Chatterbox** (Resemble AI, MIT),
pinned at commit `5de7a54a` with weights `ResembleAI/chatterbox@5bb1f6ee`.
CosyVoice 3 remains the recorded fallback.

- **Packaging gate: pass.** The engine builds into a fully offline CUDA image
  with every dependency and weight sha-pinned, and a no-cache rebuild
  reproduced the same dependency receipt. Two upstream behaviours that break
  offline use (a first-use model download and a cache-layout lookup) are handled
  without forking, and the watermark bypass seam that the section 8 override
  needs works per request without forking.
- **Falsification gate: pass.** 15 speakers from AMI and VoxConverse, 625
  generated clips on maintainer hardware (RTX 5090), measured with Voxint
  0.46.0's own whisper and TitaNet services:
  - Similarity to the speaker's held-out speech (TitaNet centroid cosine) is
    0.63 to 0.72 from a 10 s reference and 0.51 to 0.57 from 3 s, against a
    real-speech ceiling of 0.83 to 0.88. A 3 s phone-quality reference loses a
    further 0.17, while real 3 s phone audio loses 0.13 (AMI) to 0.17
    (VoxConverse) against the same held-out speech, so most of that drop is
    the channel. Enrollment guidance: prefer
    10 s of clean reference audio.
  - Word error rate is 0.4% for English synthesis and 0.2% for the
    multilingual model in English.
  - Voice conversion puts all 60 outputs below the 0.60 match floor against
    their source speaker (mean cosine 0.12 to 0.20) and adds 0.5 points of word
    error rate (the interval spans zero). Closeness to the synthetic target
    averages 0.65 to 0.81 and depends on the source recording: 4 of 15 source
    speakers convert to a target cosine below 0.60.
  - The watermark is detected on every output through 16 kHz resampling, Opus,
    telephone band-limiting and MP3, whether applied at the engine's native
    rate or at 16 kHz. Unmarked synthetic audio produced no false positives;
    4 of 60 real clips crossed the threshold after band-limiting, and the
    threshold-free margin between marked and unmarked audio is positive in
    every condition.
  - Peak reserved VRAM is about 5 GiB (multilingual model). Median real-time
    factor is 0.38 for synthesis and 0.04 for conversion.
- **Passive detection is informational.** The pinned `synthdetect` detector,
  run unchanged on a rented RTX 3090 because its CUDA 11.8 runtime has no
  kernels for the RTX 5090, scores 32% to 38% EER per generator arm on this
  corpus. On the 3-speaker VoxConverse slices, synthesis sits at or above
  chance (55% to 61%). This is consistent with issue #252. This confirms section 1: the watermark is the provenance
  signal for this engine, and a classifier is not.
- The multilingual weights' license must be confirmed from the model card
  before they are pinned for dubbing.

### 8. Two operator overrides, off by default

An operator who has their own grounds for consent, or their own reason to ship
unmarked audio, may take that responsibility explicitly. Two separate
instance-wide settings exist, and both are off on every install:

| Override | While on | Still enforced |
|---|---|---|
| Consent override | Blanket approval for both purposes. Any profile may be used for `clone` and `risk_reduction` without a per-profile attestation. | Profile enrollment gates, profile-id-only API, synthetic-only conversion targets, admin and CSRF gates, the generation log |
| Watermark override | The service does not embed the watermark, and neither check in section 1 runs. | Provenance row and WAV metadata chunk (watermark state `disabled`), the console "synthetic" badge, the generation log |

Rules for both:

- Only an admin can turn an override on, from the settings page or the CLI.
  Each has its own disclaimer. The console requires a checkbox and a typed
  confirmation; the CLI prints the disclaimer and requires an explicit
  `--accept` flag. There is no environment variable that turns either on
  silently.
- The disclaimer states that the operator takes full responsibility for how
  the feature is used and for every output it generates. The watermark
  disclaimer also states that unmarked output cannot be recognised by the
  watermark detector, and that passive detectors, Voxint's own `synthdetect`
  included, mostly fail on this generator (issue #252).
- Turning an override on or off appends a row to the generation log: who,
  when, which override, and the disclaimer version accepted. Turning one off
  takes effect for the next job. Outputs made while it was on keep the consent
  basis and watermark state recorded in their provenance.
- A disclaimer text change bumps its version and requires a fresh acceptance.

## Consequences

- The service is stateless and mounts media read-only; it returns audio bytes
  and the worker is the single writer under `artifacts/`. Assembled outputs are
  built by the worker from per-line responses.
- Detection of the shipped watermark lives inside the `synthdetect` service as a
  pre-classifier step. There is no import or HTTP dependency between the two
  plugins; the framework's import rule is unchanged.
- The `media/reclaim.py` eligibility query gains a second audio-consuming
  plugin predicate, so the existing single-plugin `NOT EXISTS` clause is
  generalized into a small shared helper.
- A profile keyed to a roster speaker means a speaker merge or delete must be
  handled explicitly by the plugin (profiles follow the canonical speaker on
  merge; a deleted speaker's profiles are revoked).
- The presumptive engine embeds its watermark unconditionally inside its own
  generate call, so the watermark override needs a pinned bypass seam in the
  service. The packaging gate identifies that seam; a contract test proves
  output is unmarked only while the override is on.
- With the watermark override on, the `synthdetect` watermark pre-check cannot
  flag those outputs. That trade is the operator's, and the disclaimer says so.
- Section 8 is the complete list of operator overrides. Anything else a later
  change wants to relax (synthetic-only targets, no autogenerate, the copy in
  section 5, a third override) requires a superseding ADR, not a settings knob.
