# ADR 0009: Voice cloning plugin defaults and consent model

> **Status:** Proposed (voice cloning epic). Records the non-negotiable
> defaults for the `voiceclone` plugin before any code lands. The engine
> selection paragraph is filled in after the packaging and falsification gates
> report; everything else here is decided now.

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

- The service applies the generator's watermark to every output unconditionally.
  There is no request flag, setting, or CLI switch to turn it off.
- The service re-detects the watermark on its own output before responding. A
  failed self-check is an error response, never a silent unmarked file.
- The worker re-verifies the watermark on the final file it publishes, at the
  pipeline's 16 kHz rate, and refuses to publish on failure.
- Every published output carries provenance in a plugin table and in the WAV
  metadata chunk: engine inference-space id, weights sha, watermark verified,
  input hashes, profile id, operator, timestamp. Assembled outputs (conversion
  and dubbing) also carry a per-line provenance row.
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
through the 16 kHz chain, peak VRAM on a 12 GB card). A license allowlist
constant (engine, license, weights sha) is checked at boot.

*Engine selection: to be recorded here after the gates report.*

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
- Anything in this record that a later change wants to relax (watermark
  enforcement, synthetic-only targets, no autogenerate, the copy in section 5)
  requires a superseding ADR, not a settings knob.
