# Synthdetect M1 S5 PR-2a: the prepare executor (ffmpeg-free bona fide materialization)

> This plan was drafted, then pressure-tested by an independent model (codex,
> planner role). Its review reshaped the plan materially: what began as one PR-2
> splits into PR-2a (prepare, this plan) and PR-2b (degrade, its own later plan),
> because the degrade path carries several contract blockers that prepare does not.
> See Review notes for the full findings and their resolutions. The immediately
> actionable deliverable is PR-2a.

## Context

S5 builds the bona fide (real-speech) domain side of the synthdetect M1 corpus
(issue #144) and validates the production windowing path. It lands as a 5-PR arc.
PR-1 (the pure, audio-free layer) is merged on `main` (GitHub PR #192, merge
`629fee6`): it emits a `MaterializationPlan` from RTTMs, freezes the pinned
decimal-to-sample rule, the overlap-clean and same-speaker merge planners, the
`IngestRecord`/`DegradedRecord`/`MaterializationPlan` schema, `finalize_manifest`,
the closed `DEGRADATION_RECIPES` vocabulary, `build_recipe_argv`, and the hardened
lineage invariants. All of it is dry-run only: `prepare` and `degrade` emit plan
JSON and touch no audio.

The executor turns those plans into real audio. It reads and writes audio bytes, so
it is maintainer-hardware work against the staged organic corpora, not plain CI. It
splits in two:

- **PR-2a (this plan), the prepare executor.** Materializes bona fide turn and
  segment clips by slicing the staged organic recordings at the plan's integer
  sample offsets. The staged sources are already canonical PCM (measured below), so
  prepare needs no codec and no ffmpeg at all: it is pure, deterministic byte
  slicing of a validated payload. This is the low-risk half.
- **PR-2b (a later plan), the degrade executor.** Materializes degraded children by
  running the frozen `build_recipe_argv` ffmpeg round trips in a digest-pinned
  ffmpeg container. This is where every codec, container, AMR-NB, and combined
  manifest question lives, and where codex's contract blockers concentrate. It gets
  its own plan once PR-2a lands.

The prepare half has a partial template already in the tree.
`tools/synthdetect_df_import.py` (the S3 ASVspoof `emit` verb) already does the
verify-by-sha, source-format gate, `read_canonical_pcm` measurement (sha256 +
sample count), per-clip receipt, whole-corpus re-audit, and atomic publish shape.
PR-2a reuses those disciplines. It differs in two ways: it slices a validated
canonical payload by sample offset instead of transcoding per file, and its source
gate is WAV-aware (the df_import gate is FLAC-specific and must not be imported
verbatim).

## Measured source formats (B4 resolved by measurement, not assumption)

Both staged organic sources on maintainer storage were probed:

- AMI `*.Mix-Headset.wav`: `pcm_s16le`, 16000 Hz, mono, `s16`.
- VoxConverse dev and test `*.wav`: `pcm_s16le`, 16000 Hz, mono, `s16`.

Every staged recording is already in canonical format. The canonicalization
contract is therefore **decode-only, gate-and-slice, no resample and no downmix**: a
source that is not already 16 kHz mono s16 PCM fails the gate closed (the maintainer
pre-canonicalizes it as a separate, documented step). No resampling numerics enter
the corpus, so there is nothing codec-dependent to pin in PR-2a.

## Goal

Implement the executing `prepare` verb so a `MaterializationPlan` becomes a
validated v1 manifest plus materialized canonical-PCM WAV clips under a
caller-supplied corpus root, by deterministic byte slicing of the validated source
payload, with byte-for-byte determinism plus an independent correctness oracle as
the acceptance gate. No audio and no corpus root enter the repo; the root is always
a CLI argument. `degrade` execution is out of scope (PR-2b); `degrade` stays
dry-run.

## Assumptions and constraints

- **Eval-first is already satisfied.** The pure contract froze in PR-1 and is unit
  tested to >=85%. PR-2a adds only audio execution around it; it must not change any
  pure identity (clip ids, the sample rule, `finalize_manifest`'s validation, plan
  JSON). A change to any of those is a PR-1-level contract change and out of scope.
- **Canonical identity is the payload, not the file.** The manifest `sha256` is the
  sha256 of the canonical PCM data-chunk payload only. `read_canonical_pcm` computes
  exactly this from a WAV and is the single measurement path. A re-muxed header can
  never change identity.
- **Slice on samples, never seconds, and only the validated payload.** The plan's
  `acquire.start_sample`/`end_sample` are authoritative; `start_s`/`end_s` are
  documentary floats. The executor slices the decoded little-endian s16 payload of
  the whole recording, never raw bytes of the WAV file (the RIFF header must never be
  indexed as audio). Half-open `[start_sample, end_sample)`, two bytes per sample.
- **Decode-only canonicalization.** No resample, no downmix. Fail closed on any
  non-canonical source property.
- **Fail closed.** Source recording and RTTM hashes are verified against a pinned
  acquisition manifest before any work, on the exact bytes consumed (no
  verify-then-reopen race). Any plan drift, out-of-range interval, sample-count
  disagreement, odd or truncated payload, or non-canonical source aborts the run
  with no partial publish.
- **Clean-room.** No internal host or tool names, no corpus paths, in code, tests,
  docs, the PR body, or the public issue. The corpus root and staged-source paths
  are CLI arguments; the acquisition manifest is keyed by logical recording id and a
  relative path, never an absolute host path.
- **Workflow.** Land via `feat/144-s5-executor`, PR to GitHub `main`
  (branch-protected, strict; required checks lint-test + secrets-scan + coverage),
  then private-origin sync.

## What would invalidate this plan

- If a staged source is not already canonical PCM (measured false: both are). Then
  PR-2a would need pinned resample numerics, which would push it toward the PR-2b
  toolchain-pin discipline. Not the case today.
- If `read_canonical_pcm` cannot measure a freshly written slice WAV (it can: the
  slice is canonical by construction).
- If two clean runs do not agree byte-for-byte (impossible without a codec: the
  slice is a pure byte range of a fixed payload wrapped in a fixed WAV header).

## Design

### Where the code lives

Add the prepare executor to `tools/synthdetect_corpus.py` as new functions plus a
CLI flag, reusing `build_plan`, `finalize_manifest`, and the dataclasses directly,
and reusing `synthdetect_infer.read_canonical_pcm`. Lift the small pure helpers
worth sharing with df_import (a WAV-payload reader, a deterministic canonical-WAV
writer, `_tool_version` is not needed here, atomic-replace, receipt serialization)
into a shared spot in a clearly separated commit if that reads cleaner than
importing across tool modules; otherwise import them. Do not import df_import's
FLAC-specific `_ffprobe_source`.

### The executing `prepare` path

`prepare` gains an execution mode selected by a new `--corpus-root PATH` flag.
Absent it, `prepare` stays exactly PR-1's dry-run (prints plan JSON, touches no
audio); the executed-plan-equals-dry-run-plan equivalence is contract-tested so the
executor can never drift the frozen plan. When a root is given:

1. **Load and pin-verify inputs on the exact consumed bytes.** A pinned acquisition
   manifest (data in `synthdetect_sources.py`, keyed by `source` -> logical
   recording id -> `{rel_path, sha256, size}`) declares every source recording and
   RTTM. The executor opens each file once, streams it to compute sha256 and size,
   and parses/decodes from that same handle or an immutable content-addressed copy,
   so the bytes verified are the bytes used (the df_import open-once-fd discipline).
   Require exact keyset coverage: every plan recording and RTTM must be pinned, and
   every pin must be consumed. Fail closed on any sha or size mismatch, or coverage
   gap.
2. **Build the plan** with the existing `build_plan`, identical to dry-run.
3. **Gate and read each source recording once.** For each distinct
   `acquire.source_file`: a WAV-aware format gate (mono, 16 kHz, s16, uncompressed,
   whole-frame non-truncated data chunk, the same checks `read_canonical_pcm`
   makes), then read the whole decoded payload once as a little-endian s16 byte
   buffer. Keep it as bytes, not a native-endian NumPy array, so slicing preserves
   the exact bytes; release each recording's buffer before the next.
4. **Slice each clip by sample offset.** For every `IngestRecord` (turn and
   segment), take `payload[start_sample*2 : end_sample*2]`, assert the interval is
   in range (`0 <= start_sample < end_sample <= n_samples`; out-of-range is drift,
   fail closed), wrap the slice in a fresh deterministic canonical WAV (fixed 44-byte
   header, bitexact, no metadata), write it at `rel_path` under the staging root, and
   measure it with `read_canonical_pcm`. Assert `n_samples == end_sample -
   start_sample`. Because turn and segment clips slice the same fixed payload,
   overlapping sample ranges are bit-identical by construction (see acceptance).
5. **Finalize.** Collect `measured = {clip_id: (pcm_sha256, sample_count)}` and call
   `finalize_manifest(records, measured)` to build and validate the v1 manifest.
6. **Publish atomically to one root, with receipts.** Materialize into a temp
   staging tree on the same filesystem as the destination, then `os.replace` the
   whole tree into place (single immutable corpus artifact; no dual-root needed for
   prepare). Emit a clip receipt (`clip_receipt.jsonl`: clip_id, source recording id,
   sample interval, canonical sha) and a source receipt (per recording: the whole
   canonical payload sha and sample count), both sorted and host-path-free. Refuse to
   overwrite an existing populated root.

### Determinism and ordering (Q1, Q2)

- Identity is the payload hash, never the container. The slice WAV header is fixed
  and bitexact, so the file is also stable, which helps clean-tree comparison.
- Sort every emitted sequence explicitly (clip records, receipt rows, file
  enumeration), not just JSON object keys. `sort_keys` does not order arrays, RTTM
  input order, glob results, or sets.
- Preserve little-endian bytes directly; avoid a native-endian NumPy round trip that
  could reorder bytes on a big-endian host.
- Record the acquisition-manifest and toolchain identity as evidence, never a
  host-supplied assertion that could describe something not actually used; for PR-2a
  the only toolchain is Python + the pinned reader/writer, so the receipt records the
  code identity (module version) and the canonicalization id, not an ffmpeg build.

### Fail-closed matrix (each aborts with no partial publish)

- Acquisition-manifest sha or size mismatch, or keyset coverage gap.
- Source not mono / 16 kHz / s16 / uncompressed, or a truncated or odd-length data
  chunk.
- A sample interval out of range, or `start_sample >= end_sample`.
- `read_canonical_pcm` sample count disagreeing with the plan interval.
- `finalize_manifest` raising (orphan facts, missing measurement, lineage/cycle).
- Destination root already populated, or staging not on the destination filesystem.

## Testing strategy

PR-2a's audio path uses no codec, so far more of it is CI-testable than a
transcoding executor would be. A tiny synthetic canonical-PCM WAV (a few hundred
coordinate-coded samples, written directly, no ffmpeg) exercises the whole
read-slice-wrap-measure path in CI.

- **CI unit tests (no ffmpeg):**
  - Pin-verify helper: pass, and fail closed on a mismatched sha, a wrong size, and
    a coverage gap.
  - WAV-payload reader and deterministic WAV writer: round-trips, extra RIFF chunks,
    odd payload, truncation, a non-44-byte header on read.
  - Byte slicing against an independent oracle: a coordinate-coded synthetic source
    (each sample encodes its own index) so an off-by-one or a global shift is
    detectable, not just internally consistent. Every emitted clip payload equals an
    independently computed byte range; the exact clip set and counts match the plan.
  - Fail-closed matrix reachable without a codec (interval out of range, count
    mismatch, `finalize_manifest` propagation, populated-root refusal,
    staging-filesystem check).
  - Receipt serializers: stable, sorted, host-path-free.
  - CLI wiring: `--corpus-root` absent stays dry-run; present executes.
- **Contract tests in the same commit (numerics doctrine):**
  - The acquisition-manifest pins are data with a keyset/shape guard (mirrors the
    ASVspoof `OFFICIAL_ARCHIVE_SHA256` guard).
  - Executed-plan-equals-dry-run-plan equivalence, so the executor cannot drift the
    frozen plan JSON.
  - The canonicalization-id and receipt schema.
- **Maintainer acceptance (the PR-2a gate; run on maintainer hardware, recorded in a
  dated report, harness not committed):** strengthened past mere two-run equality
  (two identical runs can agree on consistently wrong audio):
  1. **Correctness oracle.** Independently slice the verified source payload and
     confirm every clip payload sha matches; the exact clip set and sample counts
     equal the plan.
  2. **Reproducibility.** Two clean roots produce identical clip PCM hashes, manifest
     bytes, normalized receipts, and whole-recording canonical payload hashes.
  3. **Overlap identity (exhaustive, coordinate-aware).** For every pair of clips on
     the same recording with a non-empty sample-interval intersection, the
     corresponding payload byte ranges are identical. (A merged segment equals its
     constituent turns only on the intersection, including the gaps the segment
     spans; it is not the concatenation of the turns.)
  4. **Whole-tree re-audit.** No missing, extra, duplicated, or path-escaping clip;
     manifest, receipts, and on-disk payload hashes all reconcile.
  Record per-clip counts, the source payload hashes, and the verdict in
  `docs/reports/synthdetect-s5-pr2a-prepare-YYYY-MM-DD.md` and a Verdict block in
  `docs/gpu-contracts.md`.

## Rollout, risks, open questions

- **Single immutable corpus artifact.** PR-2a publishes one root by atomic replace;
  no dual-root/rollback complexity is needed because prepare writes a fresh tree, not
  an in-place mutation. (Degrade's in-place-add problem is a PR-2b question.)
- **Long recordings.** AMI Mix-Headset recordings are up to ~1 hour; a whole decoded
  canonical payload is ~32 kB/s (~115 MB/hour). Acceptable read-once serially;
  release each before the next, or seek within the validated data chunk if a
  recording is pathological.
- **Acquisition manifest source.** Prefer a reviewed pin table keyed by logical
  recording id with sha256 + size; retain any upstream-provided archive hashes as
  extra evidence but still pin the actual staged bytes this executor consumes.
- **No ffmpeg in PR-2a.** The digest-pinned ffmpeg container, codec determinism,
  AMR-NB availability, codec delay/padding, the raw-s16le vs WAV framing, and the
  combined parent+child manifest are all PR-2b concerns and are captured in Review
  notes so PR-2b starts from them.

## Review notes

An independent model (codex, planner role) pressure-tested the original single-PR
PR-2 draft. Findings and resolutions:

- **Split PR-2 into PR-2a (prepare) and PR-2b (degrade) [Q7, accepted].** They are
  separate failure domains and degrade carries three contract blockers prepare does
  not. This plan is now PR-2a only; PR-2b gets its own plan.
- **B1, raw-s16le vs WAV framing in degrade [confirmed in code, deferred to PR-2b].**
  `build_recipe_argv` frames input and output as `-f s16le` (headerless raw), while
  `read_canonical_pcm` reads a WAV data chunk (verified: `synthdetect_corpus.py`
  `_CANONICAL_OUTPUT`/`_RAW_INPUT_FRAMING`; `synthdetect_infer.read_canonical_pcm`
  uses `wave.open`). Feeding a parent WAV to the raw-framed argv would decode its
  RIFF header as audio and reproduce a corrupt clip deterministically. PR-2b must
  extract the parent WAV data payload to raw s16le before the recipe and wrap the
  raw result in a canonical WAV before measuring. Not a PR-2a issue: prepare never
  touches `build_recipe_argv`.
- **B2, degraded child-only manifest cannot validate [confirmed in code, deferred to
  PR-2b].** `load_manifest` (`synthdetect_corpus.py:513`) rejects any clip whose
  `parent_clip_id` is not present in the same manifest. `finalize_manifest(children,
  measured)` therefore cannot validate a child-only manifest. PR-2b must finalize a
  combined parent-plus-child manifest (re-auditing parent measurements) or define and
  validate a different composition contract. Not a PR-2a issue.
- **B3, acceptance = repeatability != correctness [accepted, folded into PR-2a].**
  Two identical runs can agree on consistently wrong audio (wrong source binding,
  off-by-one, header-as-PCM, omitted clips). The PR-2a acceptance now adds an
  independent slice oracle, a coordinate-coded synthetic CI source, exact clip/file
  sets, manifest+receipt reconciliation, exhaustive coordinate-aware overlap checks,
  and a whole-tree re-audit.
- **B4, canonicalization contract [resolved by measurement].** Both staged sources
  are already 16 kHz mono s16 PCM, so PR-2a is decode-only, gate-and-slice, no
  resample; a WAV-aware gate replaces df_import's FLAC gate. Recorded above.
- **B5, degrade atomic publish underspecified [deferred to PR-2b].** Prepare renames
  a fresh tree (clean); degrade adds children into an existing parent root and cannot
  use the same mechanism. PR-2b must publish a full replacement tree or a separate
  immutable degradation artifact with a combined-manifest assembly step.
- **Verified-byte consumption [Q3, accepted].** Hash and consume the exact bytes
  (open-once-fd or a content-addressed copy), never verify-a-path-then-reopen. Pin by
  a reviewed acquisition manifest keyed by logical recording id with sha256 + size
  and exact keyset coverage.
- **Determinism and ordering [Q2, accepted].** Sort every sequence explicitly, not
  just object keys; preserve little-endian bytes without a native-endian NumPy round
  trip; record real toolchain/acquisition identity as evidence, not a bare assertion.
- **Ffmpeg-free prepare [refinement beyond the review].** Because the measured
  sources are already canonical, PR-2a needs no codec at all, which removes the
  container, AMR-NB, and codec-determinism surface from this PR entirely. This is a
  deliberate deviation from the pre-registration wording that called the executor a
  "digest-pinned ffmpeg step"; the ffmpeg-pinned step is now PR-2b (degrade), where
  lossy codecs actually run. The pre-registration should note this split when PR-2a
  lands.
- **Not adopted / kept simple:** no dual-root publication for prepare (one immutable
  artifact is easier to reason about, and prepare writes a fresh tree); receipts
  avoid unstable temporary paths and host detail.
- **AMR-NB host probe [evidence for PR-2b].** The maintainer host ffmpeg 6.1.1 is
  built without `libopencore_amrnb` (mp3/opus/aac/mulaw/atempo present; AMR absent).
  PR-2b's pinned container must supply `libopencore_amrnb` or drop `amr-nb-122-v1`
  from the frozen set with a note and a contract test.
