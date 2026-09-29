# Voice cloning plugin epic (#672)

Status: draft

## Goal

Add a second greenfield plugin, `voiceclone`, that synthesizes speech in a
consented speaker's voice and converts a speaker's turns into a synthetic
voice, locally, with every output watermarked, labelled and provenance-tracked.
The epic is #672; its children are #662 to #670, with post-epic follow-ups in
#671. This plan records the scoping decisions and the acceptance criteria the
children inherit. Defaults are fixed in
[ADR 0009](../adr/0009-voice-cloning-defaults.md).

## Assumptions and constraints

- **Maintainer decisions (binding, 2026-09-29):** voice risk reduction ships
  before re-voicing; a voice profile attaches consent to an enrolled roster
  speaker, not to a run-local diarization label; the free-text synthesis box
  stays in the console behind the clone-consent gate; the epic and children
  are filed on GitHub at scoping time.
- The plugin follows `docs/plugins.md` and the `synthdetect` reference
  implementation without growing the framework's hook surface (ADR 0006).
- The model service mounts the media root read-only, takes media-relative
  paths, and returns audio bytes. The worker is the single writer under
  `artifacts/`.
- Version 1 is CUDA only. ROCm, CPU and Apple Silicon are follow-ups (#671).
- The engine is presumptively Chatterbox and is confirmed by two gates (#663)
  before the service is built. Two research reports disagreed on Seed-VC
  (GPL-3.0 and archived per primary sources) and CosyVoice 3 weights
  (Apache-2.0 per the model card); any candidate's license is re-verified from
  its own repository or model card before it enters the gate.
- No living spec declared: spec deltas are none.
- The main-chain Alembic head is `0066` at scoping time; verify before the
  V4 migration.

## Proposed approach

One epic, nine children, in this order:

| Child | Issue | What lands |
|---|---|---|
| V0 | #662 | ADR 0009 accepted: watermark, provenance, consent model, revocation, synthetic targets, risk-reduction naming, no autogenerate, CUDA lane |
| V1 | #663 | Packaging gate (reproducible sha-pinned image) and falsification gate on Voxint-realistic audio; engine paragraph appended to the ADR |
| V2 | #664 | Vendored synthetic target voice pack as a sha-pinned release asset with provenance |
| V3 | #665 | `services/voiceclone/` container, HTTP contract, compose overlay pair, contract tests, `docs/gpu-contracts.md` |
| V4 | #666 | Plugin backend, migration, job lane, voice library page, free-text synthesis, run-detail panel, CLI, tests |
| V5 | #667 | Voice risk reduction with the re-identification report and honest copy |
| V6 | #668 | Re-voicing of translations with per-line fit reporting |
| V7 | #669 | Watermark pre-check inside the synthdetect service (independent PR) |
| V8 | #670 | Lay-reader how-to, setup, operations, changelog, version bump, release wiring |

Design points that every child honours:

- **Consent.** A profile holds a copied and hashed reference clip in
  plugin-owned storage, never a pointer into reclaimable run audio. The plugin
  API accepts only a profile id. Enrollment enforces that the chosen turns
  belong to the consented speaker and pass duration, signal-to-noise and
  no-overlap gates. Clone consent and risk-reduction consent are separate
  records.
- **Watermark.** Applied unconditionally in the service, re-detected there
  before the response, re-verified by the worker at 16 kHz on the published
  file, refused on failure. Assembled outputs are re-verified after assembly.
- **Assembly.** Per-line or per-turn service calls; 10 to 30 ms raised-cosine
  crossfades; sample-accurate placement at each segment's start sample;
  original audio kept under non-speech and other speakers; converted turns
  replace, never mix; RMS-match per turn; resample once at the boundary and
  never lossy-encode before the watermark and re-identification checks;
  assemble to a temp path and publish only on the success CAS; one generation
  in flight per service.
- **Honesty.** "Voice risk reduction", never "anonymize". The measured maximum
  cosine against the source enrollment and every roster centroid is shown as a
  number beside the matching threshold, with a non-dismissible caveat.
  Re-voicing reports per-line fit status and passthrough turns. Cross-lingual
  similarity loss is shown from the V1 numbers.
- **Per-slice hygiene.** Each child ships its tests, route goldens,
  `.env.example` lines and CHANGELOG entry, per the `docs/plugins.md` ship
  order.

## Acceptance criteria

### Requirement: Every output is watermarked and labelled

The system SHALL refuse to publish any generated audio whose watermark cannot
be verified, and SHALL label every published output as synthetic.

#### Scenario: Service self-check fails

WHEN the service cannot detect its own watermark on an output
THEN it returns an error and no audio is returned

#### Scenario: Worker re-verification fails

WHEN the worker's 16 kHz re-verification fails on the file it is about to
publish
THEN the job is failed with the reason and nothing is written under
`artifacts/`

#### Scenario: Output plays in the console

WHEN a published output is listed or played on the plugin page or the
run-detail panel
THEN a synthetic badge is shown beside it

### Requirement: Consent gates every generation

The system SHALL generate only from a profile with an unrevoked consent record
whose purpose matches the job kind.

#### Scenario: Missing or mismatched consent

WHEN a synthesis job is requested for a profile with only a
`risk_reduction` record, or a conversion for a speaker with only a `clone`
record, or any job for a revoked profile
THEN the request is refused with the reason and no job row is created

#### Scenario: Reference turns belong to another speaker

WHEN a profile is enrolled with reference turns whose speaker is not the
consented roster speaker
THEN enrollment is refused

#### Scenario: Reference survives media reclaim

WHEN the source run's intermediates are reclaimed
THEN the profile's copied reference clip and its sha remain valid

### Requirement: Voice risk reduction is reported honestly

The system SHALL show the measured re-identification score and the caveat on
every conversion output.

#### Scenario: Conversion completes

WHEN a conversion job succeeds
THEN the panel shows the maximum cosine against the source enrollment, the
matching threshold, the skipped turn list, and the caveat text from ADR 0009

#### Scenario: Copy audit

WHEN the tree is searched for "anonymize" in user-facing copy, routes and docs
THEN there are no matches

### Requirement: Re-voicing fits lines without cutting words

The system SHALL report per-line fit and never truncate synthesized speech.

#### Scenario: Translated line overruns

WHEN a line cannot fit the source span at or below the speed-up ceiling
THEN it overruns into the trailing gap, never into the next turn, and the
line is reported as overrun

#### Scenario: Translation superseded

WHEN the pinned translation row is superseded
THEN the dub is shown as stale and a new job pins the new row

## Spec deltas

Spec deltas: none (no living spec declared)

## Affected files / components

| Area | Change |
|---|---|
| `docs/adr/0009-voice-cloning-defaults.md`, `docs/adr/README.md` | New ADR and index row (this change) |
| `services/voiceclone/` | New container (V3) |
| `compose.plugin-voiceclone.yaml`, `compose.plugin-voiceclone.build.yaml` | Overlay pair (V3) |
| `src/voxint/plugins/voiceclone/`, `src/voxint/plugins/discover.py` | Plugin package and registration (V4) |
| `alembic/versions/0067_voiceclone.py`, `src/voxint/db/models.py`, `src/voxint/config.py`, `src/voxint/app_settings.py`, `.env.example` | Schema and settings (V4) |
| `src/voxint/media/reclaim.py` | Shared audio-consuming plugin predicate (V4) |
| `tests/contracts/fixtures/*.json` | Route goldens per slice |
| `services/synthdetect/` | Watermark pre-check (V7) |
| `docs/how-to/`, `docs/setup.md`, `docs/gpu-contracts.md`, `docs/plugins.md`, `docs/operations.md`, `CHANGELOG.md`, `release.yml` | V8 |

## Implementation slices

The children above are the slices. V0 and V1 have no code. V3 and V4 land as
separate PRs. V5 then V6 each get the full multi-model review panel (they
touch a released artifact, a migration, and user-facing safety copy). V7 is
its own small PR. V8 closes the epic.

## Testing strategy

| Gate | What it covers |
|---|---|
| V1 gate report | Engine fitness on Voxint-realistic audio, watermark survival, VRAM, cross-lingual similarity |
| Contract | Service schema and provenance shas, plugin framework parity, settings parity, route goldens, migration up and down with rows |
| Unit | Gate, invariants, CAS claim, bounds, picker gates, fit arithmetic |
| Integration | Routes, CSRF, admin gate, consent refusal, revocation, GC exclusion, cancellation, serialization, re-identification report shape |
| Browser lane | Plugin page, panel, badge, non-dismissible caveat |
| Maintainer CUDA gate | Live end-to-end on the reporting host before tagging. The only CUDA card available during the current maintainer window is a 32 GB card shared with another project and borrowed in short exclusive windows through that project's controller; peak VRAM is reported as measured on that card, and smaller-card support is a follow-up (#671) |

## Rollout / risks / open questions

- **Packaging.** The presumptive engine pulls its watermark library from an
  unpinned git reference upstream. If it cannot be sha-pinned into a
  reproducible image, the epic pivots at V1 before any service work.
- **Overclaiming.** Voice risk reduction does not make a person
  unidentifiable. The copy, the score display and the name are the release
  gate for V5, not a polish item.
- **Wrong-person clone.** Mitigated by roster-speaker binding, copied and
  hashed references, speaker-match enforcement at enrollment, and a
  profile-id-only API.
- **Speaker merge and delete.** Profiles follow the canonical speaker on
  merge; a deleted speaker's profiles are revoked. Specified in V4.
- **Open:** where the watermark is embedded (native rate then resample vs at
  16 kHz) is decided by a V1 arm.
- **Hardware window.** The shared 32 GB card is on loan to the maintainer's
  workstation for a limited period; V1 and the V3 build must fit borrow
  windows measured in hours, not days. The service's published host port must
  not collide with the other project's audio ports (Voxint's model services
  already publish on a separate range there).

## Review notes

### Four-model consult (deepseek-v4-pro, grok-4.5, glm-5.2, kimi-k3; 2026-09-29)

Codex was omitted at the maintainer's request.

**Consensus, adopted:** Chatterbox as presumptive engine because of its
permissive license, first-party voice conversion and default watermark, not
its quality; replace a five-engine bakeoff with a packaging gate and a small
falsification gate; drop UTMOS, ECAPA and listener panels; service returns
bytes with the worker as single writer; copied reference clips instead of
pointers into reclaimable audio; profile-id-only API; two consent purposes;
watermark detection inside the synthdetect service as its own PR; "voice risk
reduction" naming with the score shown as a number; per-line provenance rows;
dub jobs pinned to the translation row id; one generation in flight per
service; assembly traps (speed-up ceiling, overrun as the default fallback,
crossfades, sample-accurate placement, replace never mix).

**Splits and resolutions:**

| Question | Positions | Decision |
|---|---|---|
| Risk reduction or re-voicing first | grok, glm, kimi: risk reduction first; deepseek: re-voicing first because risk reduction is the most dangerous to get wrong | Risk reduction first, with deepseek's mitigation: the honest copy and the re-identification report are its release gate |
| Free-text synthesis in the console | kimi, grok: preview only, CLI for text; deepseek, glm: a real capability | Maintainer kept the box, gated by clone consent, admin-only, capped, audited |
| Standalone plugin page | grok: run-scoped only; others: keep | Kept; ADR 0006 requires the plugin to own its surface; generation launches from the run panel |
| Neutral voice for unconsented speakers in re-voicing | grok: cut; glm: default for cross-lingual | Cut from v1; passthrough and report; follow-up in #671 |
| Profile identity scope (raised by kimi) | run-local label vs enrolled roster speaker | Maintainer chose the roster speaker |

**Rejected:** none. UTMOS and a blind listener panel were dropped as not
moving any decision for this audience.
