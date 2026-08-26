# Synthdetect M1 S5: organic corpus, degradation chains, production-windowing validation

## Context

Synthdetect Milestone 1 (issue #144) is the maintainer evaluation capability that
must exist before any detector ships: a reference corpus, a detector harness, and
reproduction of the upstream published numbers. S1 through S4 are landed: the pure
CI scaffolding (schema, seeded splits, host scorer, pinned runner), the eval
container and weights qualification, and both DF benchmark reproduction gates
(Gate-2 paired per-clip equivalence #184, Gate-1 full-cohort anchor #188). What
does not yet exist is any corpus of organic (real human) speech, any degradation
chains, or any validation of the production windowing path against the upstream
crop. That is S5, the next slice of the M1 arc. It is the first half of the road
to the shipped calibration policy: S5 builds the bona-fide domain side, S6 adds the
synthetic (spoof) side, and S7 runs the bakeoff, calibration, and holdout.

This plan was drafted, then pressure-tested by two independent models in parallel.
Their convergence reshaped it materially (see Review notes). The single biggest
correction: the pure layer must emit a materialization plan, never a final
manifest, because a valid manifest cannot exist until audio execution supplies the
canonical PCM sha256 and the sample-derived duration.

## Goal

Deliver S5 as a sequence of small, individually reviewable, CI-gated PRs, starting
with a single unified pure-logic PR (PR-1) that freezes the corpus-building and
windowing-comparison contracts before any audio bytes or GPU run exist, exactly as
every prior synthdetect slice landed its pure layer first.

## Assumptions and constraints

- **Eval-first discipline.** Pure, audio-free, CI-testable logic freezes and unit
  tests to >=85% before any audio or GPU step (a separate maintainer action). Audio
  is never committed; the corpus root is always a CLI argument.
- **Canonical identity.** The manifest `sha256` is the sha256 of the canonical PCM
  data payload only (`pcm-s16le-mono-16000-v1`: 16 kHz mono s16le). The runner does
  not resample; canonicalization happens once at acquisition via ffmpeg.
- **Bona-fide source (decided).** Two CC-BY-4.0 diarization corpora (a meeting-room
  set and a web-video set), staged on maintainer storage with RTTMs. The staged
  meeting-room audio is a speaker-mixed track, so `prepare` drops any turn region
  overlapping another speaker (>100 ms) and keeps only the clean single-speaker
  sub-spans. This addresses the split-leakage hazard that a "single-speaker" turn
  could otherwise carry crosstalk from a speaker assigned to a different split. Both
  domains are represented.
- **S5 verdict scope.** With a bona-fide-only corpus, the production-windowing
  verdict can validate false-positive stability but not separability or EER (no
  spoof side until S6). The verdict is explicitly limited to bona-fide robustness;
  the full separability gate defers to after S6.
- **Numerics doctrine.** Any runner windowing change is pre-registered in
  `docs/gpu-contracts.md` before the GPU run, with a dated verdict block after.
- **Workflow.** Work lands via `feat/144-...` branches, PR to GitHub `main`
  (branch-protected, strict; required checks lint-test + secrets-scan + coverage),
  then private-origin sync. PR bodies and public issues are clean-room: no internal
  host or tool names.

## What would invalidate this plan

- If the merged-segment production unit turns out to need audio-derived boundaries
  (it does not: RTTM times plus a pinned decimal-to-sample rule are enough).
- If RTTM speaker labels are not usable even recording-scoped (they are; we
  namespace conservatively, see PR-1).
- If the existing host calibration machinery cannot fit on a bona-fide-plus-spoof
  manifest split (it can; that is S7, out of scope here).

## The S5 PR arc

| PR | Lane | Content |
|---|---|---|
| **PR-1** | pure, CI | All S5 pure logic: RTTM parse, decimal-to-sample and same-speaker `<1s` merge planning, overlap/min-duration policy, materialization-plan schema, `finalize_manifest(measured_facts)`, degradation recipe registry + ffmpeg argv builders, hardened lineage invariants. Plan/dry-run CLI only. Corpus + degradation pre-registration in `gpu-contracts.md`. |
| **PR-2** | maintainer CPU audio, reviewed | The executing `prepare`/`degrade` verbs in a digest-pinned ffmpeg container: verify source + RTTM hashes, canonicalize once then slice by sample offset, encode-decode-canonical-PCM round trip, final PCM sha + sample-count duration, atomic output, toolchain receipt. Acceptance: two clean materializations must produce identical final PCM hashes. |
| **PR-3** | pure + maintainer | Corpus and calibration-cohort freeze: degraded children only from calibration-split parents; one pre-registered variant per parent (no derivative multiplicity inflating Platt); coverage requirements counted by independent speakers; license/attribution/modification receipts. |
| **PR-4** | pure runner change + pure metrics, CI | Per-window observability journal, paired windowing-comparison metrics, and the runner tail/width fixes (below). Windowing-comparison pre-registration in `gpu-contracts.md`. |
| **PR-5** | GPU maintainer | Run upstream vs production windowing on the frozen cohort, dated verdict (bona-fide FP robustness). |

The immediately actionable deliverable is **PR-1**. The rest is sketched so PR-1's
contracts are shaped correctly; each later PR gets its own plan when reached.

## PR-1 detailed design (the slice to implement next)

All pure, no ffmpeg execution, no audio reads, no GPU. Reuses existing code where it
exists.

### Affected files

- `tools/synthdetect_corpus.py` (primary): add the materialization-plan dataclasses
  (`IngestRecord`, `SessionSegmentPlan`, `MaterializationPlan`), the pure RTTM parser
  and cleaner, the `<1s` same-speaker merge planner, the decimal-to-sample interval
  rule, `finalize_manifest(plan, measured_facts)`, the ffmpeg argv builders and
  canonical chain serialization, and the hardened lineage invariants. Add `prepare`
  and `degrade` CLI subcommands that emit a `MaterializationPlan` JSON in
  dry-run/plan mode only (they refuse to run without the executor from PR-2). Reuses
  existing `validate_clip`, `load_manifest`, `assign_splits`, `_is_safe_id`,
  `_is_sha256`.
- `tools/synthdetect_sources.py` (pins-as-data): add the versioned degradation
  recipe registry (recipe id -> canonical ordered parameters, and the toolchain pin
  references). This mirrors the file's existing role (weights, benchmarks, windowing
  all live here as data). The argv builders and serialization stay in
  `synthdetect_corpus.py`; only the recipe vocabulary is data here.
- `tests/unit/test_synthdetect_prepare.py` (new): RTTM parse edge cases, overlap
  drop, min-duration floor, sample-interval conversion, merge planner, speaker
  namespace, golden `MaterializationPlan` JSON, `finalize_manifest` from measured
  facts.
- `tests/unit/test_synthdetect_degrade.py` (new): recipe-registry integrity,
  canonical chain serialization (order matters, one normal form), exact ffmpeg argv
  arrays, child-record derivation and lineage inheritance, hostile-input rejection
  (free-form filter strings, unknown recipe ids, traversal in paths).
- `tests/unit/test_synthdetect_manifest.py` (extend): the hardened lineage
  invariants (multi-node cycle rejection, degraded-child inheritance of
  label/speaker/language/split/license).
- `docs/gpu-contracts.md` (extend): an S5 pre-registration subsection recording the
  corpus protocol, strata vocabulary, overlap and min-duration floors, the pinned
  decimal-to-sample rule, and the frozen degradation recipe set.
- `CHANGELOG.md`: `[Unreleased]` entry.

### Pure components

1. **RTTM parser and cleaner.** Strict parse of `SPEAKER <rec> 1 <start> <dur> <NA>
   <NA> <label> ...`. Fail closed on malformed rows. Word-level RTTMs (sub-second
   segments) build turns by merging adjacent same-speaker words; the parser
   preserves the atomic word/turn spans for provenance.
2. **Sample-interval rule (pinned).** `start_sample = floor(start * 16000)`,
   `end_sample = ceil((start + dur) * 16000)`. Pinned in code and pre-registered so
   the corpus is byte-reproducible from the RTTM.
3. **Overlap and min-duration policy.** Given all turns in a recording, subtract
   other-speaker regions (>100 ms overlap) from each speaker's spans, yielding clean
   single-speaker sub-spans. Drop sub-spans below a pre-registered turn-clip floor.
   Signal how many turns/samples were dropped.
4. **Speaker namespace (conservative).** `speaker_id = f"{source}-{recording}-{label}"`.
   Recording-scoped, so a mislabeled or recording-local id can never cause
   cross-split leakage. Deliberate: it can split one real person across recordings,
   which is acceptable for a bona-fide FP corpus (no generalization claim in S5).
   Recorded as an open question for S6/S7 if a stronger speaker-disjoint claim is
   later needed.
5. **Same-speaker `<1s` merge planner.** The production windowing merges same-speaker
   turns separated by `< merge_gap_s`. This is an RTTM-level operation done before
   per-turn clipping, so `prepare` emits a `SessionSegmentPlan` (the merged speech
   segments, each retaining its constituent turn spans) as a distinct view. Per-turn
   clips feed strata/degradation/calibration; merged segments feed production-windowing
   validation. Segments >= model width (64600 samples) are the validation set.
6. **Materialization-plan schema.** `IngestRecord` (rel_path, source, RTTM provenance
   in the existing `acquire` field, sample interval, speaker_id, domain stratum, split
   from `assign_splits`) and `MaterializationPlan` (records + planned lineage +
   destinations). It carries no output hashes: those do not exist yet.
7. **`finalize_manifest(plan, measured_facts)`.** Consumes the executor's measured
   per-clip `pcm_sha256` and `sample_count` (duration = `bytes / (16000 * 2)`), builds
   v1 `ClipEntry` records, and runs the existing `load_manifest` validation. This is
   the only path from plan to manifest.
8. **Degradation recipe registry + argv builders.** A closed, versioned vocabulary
   (no free-form ffmpeg). Initial set, each with an explicit implementation:
   `mp3-cbr48-v1` (libmp3lame), `opus-voip-cbr16-f20-v1` (libopus, VBR off),
   `aac-lc-cbr48-v1`, `g711-mulaw-8k-v1` (pcm_mulaw), `amr-nb-122-v1`
   (libopencore_amrnb, pending encoder-availability check in the pinned build),
   `speed-atempo-0p90-v1` (tempo-preserving), and an additive-noise recipe using a
   pinned licensed noise PCM asset (preferred over seeded ffmpeg `anoise` for
   provenance; seeded ffmpeg is the documented fallback). Every lossy recipe is a real
   round trip (canonical PCM -> pinned encoder -> temporary bitstream -> pinned
   decoder/resampler -> canonical PCM), and only the final PCM payload sha is the clip
   identity. `-threads 1` throughout; raw input always framed with `-f s16le -ar 16000
   -ac 1` before `-i`. Chains are ordered and canonically serialized into the manifest
   `degradation` string (one normal form).
9. **Hardened lineage invariants.** Extend `load_manifest`: reject multi-node cycles
   (not just self-parent); a degraded child must inherit its parent's label,
   speaker_id, language, split, and license_spdx, and remain generator-null; the
   `degradation` string must name a recipe id in the registry.

### Runner windowing fixes (pre-registered in PR-1, implemented in PR-4)

Verified in `tools/synthdetect_infer.py:plan_windows`:

- **Tiny-tail window.** Production mode appends a final partial span even when it is
  one sample; that span is repeat-padded to 64600 and logit-mean-pooled with equal
  weight to a full window. Fix: drop a trailing partial window when a full window
  exists and the tail is below a pre-registered floor. Version the change.
- **64000 vs 64600 mismatch.** `production_window_s = 4.0` -> 64000 samples, but the
  model width is 64600, so every full production window is repeat-padded by 600
  samples, unlike the upstream 64600 crop. Recommend setting the production window to
  exactly the model width (4.0375 s / 64600 samples) so a full production window
  equals the upstream crop content, making the comparison clean. This is a
  `WindowingPolicy` data change in `synthdetect_sources.py` and is pre-registered.

These are defining (not changing) production scoring, which has never been shipped;
still pre-registered, unit-tested, and eventually GPU-determinism-checked.

## Testing strategy

- **PR-1 (CI gate):** unit tests to >=85% on every pure component. Golden
  `MaterializationPlan` JSON, exact ffmpeg argv arrays, stable clip/segment ids, RTTM
  edge cases, overlap-drop math, merge-planner boundaries, chain-serialization normal
  form, lineage-inheritance and cycle rejection, `finalize_manifest` from synthetic
  measured facts, hostile-path and unknown-recipe rejection. No ffmpeg, no audio, no
  GPU. `ruff` + `mypy` clean.
- **PR-2 acceptance (maintainer):** two clean materializations into separate roots
  must yield identical final PCM hashes; source and RTTM hashes verified before work;
  fail-closed on unexpected streams, sample rates, or plan drift.
- **PR-4 (CI):** pure windowing-comparison metrics tested with synthetic journals; the
  tail/width fixes unit-tested against the exact sample-count boundaries.
- **PR-5 (GPU maintainer):** paired upstream vs production run on the frozen cohort;
  pre-registered acceptance (bona-fide FP-rate stability, upper 95% CI of the
  production-minus-upstream FPR increase within the declared bound; Spearman and MAE
  reported as diagnostics, not primary criteria). Dated verdict block.

## Rollout, risks, open questions

- **Derivative multiplicity (PR-3):** N degraded children of one utterance are not N
  independent calibration observations. Resolve at PR-3 by one pre-registered variant
  per parent, or parent-group weighting in the Platt fit.
- **AMR-NB availability:** the pinned ffmpeg build may lack `libopencore_amrnb`;
  confirm at PR-2 or drop that recipe from the frozen set with a note.
- **Reproducibility scope:** exact byte regeneration is promised only on the pinned
  realization platform (container digest + codec library versions), never as universal
  ffmpeg reproducibility. State this in the pre-registration.
- **Speaker namespace (open):** recording-scoped speaker ids weaken a future
  speaker-disjoint generalization claim; revisit if S6/S7 needs one.

## Review notes

Two independent models reviewed the draft in parallel. They converged strongly.

- **Manifest-before-hash (both, accepted).** The pure layer cannot build a final
  manifest before audio exists. Adopted the plan -> materialize ->
  `finalize_manifest(measured_facts)` split as the core of PR-1.
- **Merged-segment production unit (both, accepted).** Per-turn clips lose the
  merge-gap information the production policy depends on. `prepare` now emits a
  `SessionSegmentPlan` merged view for windowing validation.
- **Full toolchain pinning and closed recipe vocabulary (both, accepted).** Pin by
  container digest + codec library versions, no free-form ffmpeg, real round trips,
  final PCM sha as identity, `-threads 1`, duration from sample count.
- **Statistical traps (both, accepted; deferred to PR-3):** degraded children only
  from calibration parents; one variant per parent to avoid inflating Platt.
- **Overlap leakage (both, accepted):** drop other-speaker overlap from single-speaker
  clips.
- **Tail-window and 64000/64600 padding (verified in code):** confirmed real in
  `plan_windows`; folded into PR-4 with pre-registration in PR-1.
- **Bona-fide-only verdict limit (accepted):** S5 validates FP stability, not
  separability; verdict scope stated explicitly.
- **Divergence, split prepare and degrade into two PRs vs one unified pure PR:**
  resolved to the unified pure PR-1. The pure logic still lands as separable commits
  (prepare, degrade, shared plan/realize schema) so review stays tractable.
- **Not adopted as PR-1 scope:** speaker-cluster bootstrap confidence intervals and
  per-domain non-inferiority tables are richer than a single-operator bona-fide FP
  check needs now; they belong to PR-5's verdict if warranted, not the pure contract.
