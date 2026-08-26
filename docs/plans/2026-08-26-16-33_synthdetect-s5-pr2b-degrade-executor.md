# Synthdetect M1 S5 PR-2b: the degrade executor

> This plan was drafted, then pressure-tested by an independent model (codex,
> planner role). Its review surfaced 14 findings, several critical. All accepted
> findings are folded into this revision. See Review notes for the full audit
> trail.

## Context

S5 builds the bona fide (real-speech) domain side of the synthdetect M1 corpus
(issue #144). It lands as a multi-PR arc. PR-1 (the pure, audio-free layer) and
PR-2a (the ffmpeg-free prepare executor) are both merged on main. PR-2b is the
second half of the executor arc: it materializes codec/telephony/speed-degraded
children of the already-materialized bona fide clips.

PR-2a materialized bona fide clips by deterministic byte slicing of pin-verified
source recordings (no codecs, no ffmpeg). PR-2b introduces the codec seam: it
runs the frozen `build_recipe_argv` ffmpeg round trips inside a digest-pinned
container, wrapping the raw-framed I/O in canonical WAV for measurement and
storage. The output is a separate immutable degrade root containing child WAVs
plus a combined parent-plus-child manifest. This is maintainer CPU-audio work
against real organic corpora, not plain CI.

## Goal

Implement the executing `degrade` verb in `tools/synthdetect_corpus.py` that:

1. Reads a parent bona fide corpus (the output of `prepare`)
2. Re-audits every parent clip (sha256, sample_count, duration_s)
3. Derives degraded children via `derive_degraded_record`
4. Executes the full ordered recipe chain per child via `build_recipe_argv` in a
   digest-pinned ffmpeg container
5. Wraps raw output in canonical WAV, measures identity
6. Publishes a separate immutable degrade root with a combined parent-plus-child
   manifest validated through `load_manifest`

## Assumptions and constraints

- **PR-2a is landed.** `materialize_prepare`, `read_canonical_wav_payload`,
  `write_canonical_wav`, `payload_sha_and_count`, `canonical_payload_from_bytes`,
  and `build_recipe_argv` are on main and stable.
- **AMR-NB is kept.** `amr-nb-122-v1` stays in `DEGRADATION_RECIPES`; the
  digest-pinned ffmpeg container must supply `libopencore_amrnb`.
- **Separate immutable artifact.** The degrade root is a new, empty directory;
  parent WAVs are not copied.
- **Combined manifest.** Contains both parent and child clip entries, validated
  through `load_manifest`.
- **Byte-for-byte determinism on the pinned platform only.** Two runs of the
  same recipe chain on the same parent input in the same container produce
  identical output bytes. Cross-platform determinism is not claimed.
- **Clean-room.** No internal hostnames, IPs, credentials, or corpus paths in
  code, tests, docs, or PR body.
- **One variant per parent (PR-3 freezes cohort policy).** The `--split` filter
  is preserved. No grandchildren (already-degraded parents skipped).
- **Docker available on host.** An implicit dependency for executor mode.

## What would invalidate this plan

- If no readily available Docker ffmpeg image supplies `libopencore_amrnb` and
  building a custom image is blocked. (Mitigated by Phase 0 prerequisite.)
- If `build_recipe_argv` produces incorrect ffmpeg commands when run against a
  real ffmpeg. (Caught by Phase 0 smoke tests.)
- If ffmpeg determinism breaks even with `-threads 1 -filter_threads 1` on the
  pinned platform. (Caught by Phase 3 acceptance.)
- If the combined manifest triggers an unanticipated `load_manifest` rejection.

## Design

### Blocker resolutions

**B1 (raw/WAV framing).** `build_recipe_argv` frames I/O as headerless raw
s16le (`_RAW_INPUT_FRAMING` / `_CANONICAL_OUTPUT`), while parent clips are WAV
files. The executor:

1. Reads the parent clip WAV via `read_canonical_wav_payload` (returns raw
   data-chunk payload, headerless s16le bytes).
2. Writes raw payload to a temp file (`input.raw`).
3. Builds argv via `build_recipe_argv` with `/work/`-prefixed paths (the
   container mount point).
4. Runs each command in the container. For **ordered recipe chains**, each
   recipe's output becomes the next recipe's input: recipe-0 reads `input.raw`,
   writes `stage-0.raw`; recipe-1 reads `stage-0.raw`, writes `stage-1.raw`,
   and so on. Only the final stage's output is the child clip.
5. Reads the final raw output, wraps in canonical WAV via `write_canonical_wav`.
6. Re-reads the written WAV via `read_canonical_wav_payload` to prove the
   round-trip (payload identity, not file-bytes identity).
7. Measures via `payload_sha_and_count`. The canonical-PCM payload sha is the
   child's manifest identity.

**B2 (combined manifest).** `finalize_manifest` expects `IngestRecord |
DegradedRecord`, but parent clips are `ClipEntry` objects from the manifest. A
new `_assemble_combined_manifest` function reproduces `finalize_manifest`'s
measurement-coverage rails without requiring IngestRecords:

1. Reads the parent `manifest.json`, hashes its exact bytes, and validates via
   `load_manifest`.
2. Re-audits every parent clip: reads WAV from parent root, computes
   `payload_sha_and_count`, verifies sha256 matches the manifest, verifies
   `sample_count` matches (i.e., `duration_s == sample_count /
   CANONICAL_SAMPLE_RATE` under the canonical duration rule). Records sample
   count in the receipt.
3. Derives children + executes + measures. Enforces exact child-record /
   measurement keyset equality in both directions (no orphan records, no orphan
   measurements).
4. Builds child clip dicts via `record.clip_dict(sha, count)` (duration derived
   from measured sample_count, not from any external source).
5. Combines the raw parent clip dicts + child clip dicts.
6. Validates through `load_manifest({"schema_version": 1, "clips":
   combined})`.

**B5 (publish).** The degrade root is a new, empty directory published by
atomic `os.replace` (same `mkdtemp` + `finally rmtree` pattern as
`materialize_prepare`). It contains:

- Child WAVs at the `DegradedRecord.rel_path` positions (e.g.,
  `ami/degraded/{clip_id}.wav`).
- `manifest.json`: the combined parent-plus-child manifest.
- `degrade_receipt.json`: parent manifest sha256, combined manifest sha256,
  container image digest, resolved platform image ID, toolchain versions, per-
  child measurements, per-parent re-audit confirmations, sorted deterministically.
- `clip_receipt.jsonl`: per-child clip receipts.

Parent WAVs are NOT in the degrade root. The combined manifest's parent clips
have `rel_path` values that reference the parent root. A new
`resolve_clip_path` helper resolves a clip's `rel_path` against an ordered list
of roots (degrade root first, parent root second), with exact-once resolution
(every clip resolves in exactly one root, no duplicates, no misses).
Contract-tested in PR-2b; scorer integration is PR-3.

### Container execution model

The executor runs on the host. For each ffmpeg command, it invokes `docker run`
with the pinned container image. A `_run_containerized_ffmpeg` function wraps
each call:

```
docker run --rm --network none
  --entrypoint ffmpeg
  -v <workdir>:/work:rw
  <image@sha256:digest>
  <argv-without-ffmpeg-prefix>
```

Key mechanics:

- **`--entrypoint ffmpeg`** explicitly, and pass `argv[1:]` (stripping the
  `"ffmpeg"` prefix from `build_recipe_argv`'s output). This avoids double-
  command bugs when the image has its own ENTRYPOINT.
- **`--network none`** for isolation.
- **Timeout**: bounded subprocess timeout (e.g., 300 s per command). A hanging
  codec process must not block the entire run.
- **Output validation**: after each command, verify the output file exists, is
  a regular file (not a symlink), and is non-empty.
- **Error capture**: capture bounded stderr (last 4 KiB), normalize any host
  paths before surfacing.
- For **lossy 2-pass recipes**, two separate `docker run` calls with the same
  mounted workdir; the intermediate file persists on the host between them.

### Codec delay and length policy

Lossy codecs may change decoded length vs the input (encoder delay, frame
padding, resampling). Policy:

- The child's length is the **actual output length**. The manifest records the
  measured `sample_count` and `duration_s`.
- Length change is **expected, not a bug** (especially `speed-atempo-0p90-v1`
  which stretches to ~111%).
- **Determinism, not length preservation**, is the contract.
- **Sanity bounds** (fail-closed): reject output whose sample count is < 50% or
  > 200% of the parent's (broad enough for speed transforms, narrow enough to
  catch catastrophic codec failures like truncation to zero or runaway padding).
  Per-recipe bounds can be tightened after characterization in Phase 0.

### Preflight validation (fail before any container launch)

Before executing any ffmpeg, the executor validates:

1. Parent manifest is schema_version 1, corpus_kind "synthesis".
2. No existing degraded entries in the parent manifest (prevents three-root
   artifacts from chaining degrade runs).
3. All parent clips are label "bona_fide" (spoof children would need generator
   provenance that `DegradedRecord` does not carry).
4. No child clip_id / rel_path collisions with parent clip_ids / rel_paths.
5. Parent root contains `manifest.json` and all referenced WAVs as regular
   files (no symlinks).
6. Container image string matches `<repo>@sha256:<64hex>` format.
7. Degrade root is empty or nonexistent.

### CLI changes

`cmd_degrade` gains execution-mode flags (same pattern as `cmd_prepare`):

- `--corpus-root PATH`: degrade output root (must be empty/nonexistent).
- `--parent-root PATH`: parent corpus root (must contain manifest.json).
- `--container-image IMAGE@sha256:DIGEST`: the pinned ffmpeg container.

Without `--corpus-root`, unchanged dry-run (prints plan JSON, touches no audio).
With it, executes.

### Turn vs segment selection (noted, deferred to PR-3)

The manifest does not distinguish turn clips from session segments after
finalization (no `kind` field). The pre-registration says turns are for
degradation and calibration, segments for windowing validation. PR-2b degrades
all root clips (no `parent_clip_id`) in the selected split, including segments.
PR-3 can add a stratum-pattern filter if segments must be excluded.

## Affected files

- `tools/synthdetect_corpus.py`: add `materialize_degrade`,
  `_run_containerized_ffmpeg`, `_degrade_one_clip` (with sequential chain
  execution), `_assemble_combined_manifest`, `resolve_clip_path`,
  `_write_degrade_artifacts`, `DegradeResult` dataclass; extend `cmd_degrade`
  with `--corpus-root` / `--parent-root` / `--container-image` flags.
- `tests/unit/test_synthdetect_degrade.py`: executor tests with fake Docker
  executable (not just mocked subprocess), combined manifest assembly, re-audit,
  chain execution, preflight rejection, collision detection, length sanity.
- `tests/contracts/`: contract test for multi-root resolution; container image
  pin format.
- `docs/gpu-contracts.md`: PR-2b Verdict block (after acceptance).
- `CHANGELOG.md`: entry under [Unreleased].

## Step-by-step implementation

### Phase 0: prerequisites (before any executor code)

0a. **Select or build the digest-pinned ffmpeg container image.** Candidates:
    `jrottenberg/ffmpeg:7-ubuntu` (full codec build, well-known),
    `linuxserver/ffmpeg`, or a repository-owned image built from pinned inputs.
    Pull, verify all 6 codec implementations are present (encoder + decoder +
    muxer + demuxer): libmp3lame, libopus, aac, pcm_mulaw, atempo filter,
    libopencore_amrnb. Pin by `image@sha256:<digest>`.

0b. **Smoke all 6 recipes.** Run a short synthetic s16le clip (1 s of sine)
    through each recipe's full round trip inside the container: encode +
    decode + re-read. Verify non-empty output for each, and that AMR-NB
    specifically works end-to-end (encode + mux + demux + decode).

0c. **Freeze entrypoint behavior.** Verify that `docker run --entrypoint ffmpeg
    <image> -version` works correctly (no doubled command). Record the image's
    native ENTRYPOINT for the receipt.

0d. **Collect full provenance.** Inside the container: `ffmpeg -version`,
    `ffmpeg -buildconf`, `ffmpeg -encoders | grep -E 'mp3lame|libopus|aac|
    pcm_mulaw|amrnb|atempo'`. Record os/arch, resolved platform image ID.

0e. **Characterize per-recipe length behavior.** Run each recipe on boundary
    inputs (1-sample, 1-frame, 1-second clips) and record the output length
    delta. Use this to set tighter per-recipe sanity bounds if warranted.

### Phase 1: implement executor (one commit, tests in same commit)

1. **Cut the branch**: `git checkout -b feat/144-s5-degrade` from synced main.
   Push to both remotes.

2. **`DegradeResult` dataclass**: mirrors `PrepareResult` but records parent
   manifest sha, combined manifest sha, container image, children count.

3. **`_run_containerized_ffmpeg(argv, *, workdir, container_image, timeout=300)`**:
   docker run wrapper. Strips the `"ffmpeg"` prefix from argv, prepends
   `docker run --rm --network none --entrypoint ffmpeg -v workdir:/work
   container_image`. Runs via `subprocess.run` with timeout, check=True,
   bounded stderr capture. Validates output file after each call.

4. **`_degrade_one_clip(parent_clip, child_record, recipe_chain, *, parent_root,
   staging, container_image)`**: per-clip executor with **ordered chain
   execution**.
   - Read parent WAV payload via `read_canonical_wav_payload(parent_root /
     parent_clip.rel_path)`.
   - Create per-clip workdir under staging temp.
   - Write raw payload to `workdir/input.raw`.
   - For each recipe in the chain (in order):
     - Build argv via `build_recipe_argv(recipe, in_path="/work/stage-{i-1}.raw"
       or "/work/input.raw", out_path="/work/stage-{i}.raw",
       intermediate_path="/work/intermediate-{i}.{fmt}")`.
     - Run each argv via `_run_containerized_ffmpeg`.
   - Read final `stage-{n}.raw`.
   - Wrap in canonical WAV via `write_canonical_wav` at child's `rel_path` in
     staging.
   - Re-read WAV via `read_canonical_wav_payload`, prove payload identity.
   - Measure via `payload_sha_and_count`.
   - **Sanity bound**: reject if output sample_count is < 50% or > 200% of
     parent's sample_count.
   - Return `(sha, count)`.

5. **`_assemble_combined_manifest(parent_clip_dicts, child_records, measured)`**:
   dedicated assembler reproducing `finalize_manifest`'s measurement-coverage
   rails.
   - Enforce exact child-record / measurement keyset equality (both directions).
   - Validate sha256 format and positive int sample_count for each child.
   - Derive child `duration_s` from measured `sample_count /
     CANONICAL_SAMPLE_RATE`.
   - Build child clip dicts via `record.clip_dict(sha, count)`.
   - Combine parent clip dicts + child clip dicts.
   - Validate through `load_manifest({"schema_version": 1, "clips":
     combined_dicts})`.
   - Return the validated `Manifest` + the exact combined clip dicts (for
     deterministic serialization).

6. **`resolve_clip_path(clip, *, roots: tuple[Path, ...])`**: resolve a clip's
   `rel_path` against an ordered list of roots. Returns the first root where the
   path exists as a regular file. Fails closed if the clip resolves in zero or
   multiple roots.

7. **`materialize_degrade(*, parent_root, corpus_root, container_image, recipe_ids,
   split_filter)`**: the orchestrator.
   - **Preflight**: validate parent manifest shape (v1, synthesis, no degraded
     entries, all bona_fide), degrade root empty, container image format, no
     clip_id / rel_path collisions.
   - **Parent re-audit**: for each parent clip, read WAV from parent root,
     compute `payload_sha_and_count`, verify sha256 + sample_count +
     `duration_s == count / 16000`.
   - **Derive children**: `derive_degraded_record` for each eligible parent
     (non-degraded, matching split).
   - **Execute**: `_degrade_one_clip` for each child, collect measurements.
   - **Assemble**: `_assemble_combined_manifest`.
   - **Write artifacts**: `_write_degrade_artifacts` (manifest, receipts).
   - **Atomic publish**: `os.replace`.
   - Return `DegradeResult`.

8. **`_write_degrade_artifacts(staging, manifest, clip_dicts, child_records,
   measured, parent_manifest_sha, container_image, toolchain_info)`**: write
   the manifest.json (sorted, deterministic), clip_receipt.jsonl, and
   degrade_receipt.json. Receipt includes: parent_manifest_sha256,
   combined_manifest_sha256, container_image, resolved_image_id, ffmpeg_version,
   ffmpeg_buildconf, canonicalization_id, per-child measurements, per-parent
   re-audit confirmations. All arrays sorted by clip_id.

9. **Extend `cmd_degrade`**: add `--corpus-root`, `--parent-root`,
   `--container-image` flags. Without `--corpus-root`, unchanged dry-run. With
   it, call `materialize_degrade`.

10. **Unit tests** in `tests/unit/test_synthdetect_degrade.py`:
    - `_degrade_one_clip` with a fake Docker executable (a shell script that
      copies input to output with a known transform, placed on PATH in the test).
    - Sequential chain execution: verify stage-0 feeds stage-1, argv order
      matches recipe order, reversing a chain changes the output identity.
    - Combined manifest assembly: parent + child validates through
      `load_manifest`; orphan records/measurements rejected; sha format + int
      sample_count enforced; duration derived from sample_count.
    - Re-audit failure: tampered parent (sha mismatch, duration mismatch).
    - Preflight rejection: v2 manifest, degraded entries, spoof labels,
      populated degrade root, non-digest container image.
    - Collision detection: duplicate child rel_path, collision with parent
      rel_path.
    - Length sanity: output < 50% or > 200% rejected.
    - `resolve_clip_path`: resolves in correct root, fails on zero/multiple.
    - Atomic publish cleanup on failure.
    - `--split` filter, grandchild skip.
    - CLI dry-run / execution-plan equivalence.

11. **Contract tests**:
    - `resolve_clip_path` exact-once resolution against two roots.
    - Container image pin format (`image@sha256:64hex`).

12. **Gates**: `uv run ruff check`, `uv run mypy`, focused tests + coverage >=
    85%, `~/.local/bin/gitleaks git . --log-opts="origin/main..HEAD"`, grep diff
    for internal identifiers.

### Phase 2: multi-model review

13. **Commit before review** (reviewers can stash-wipe uncommitted changes).

14. **Full multi-model review** (real audio numerics + new codec seam = high
    blast radius per review policy): codex + grok or deepseek + kimi if
    warranted. Apply fixes, record in commit message.

### Phase 3: maintainer acceptance (the PR-2b gate)

15. On the pinned container, with real organic audio:
    a. Materialize a small parent corpus via `prepare` (reuse the PR-2a
       acceptance recipe: one AMI recording, one VoxConverse recording).
    b. Run `degrade` twice on the same parent, into separate roots, for every
       recipe individually and for at least one multi-recipe chain.
    c. Verify byte-identical child PCM hashes across the two runs.
    d. Verify parent clip hashes were re-audited before work (in the receipt).
    e. Verify AMR-NB presence: the `amr-nb-122-v1` recipe ran successfully.
    f. Verify `resolve_clip_path` resolves every combined-manifest entry
       exactly once against the parent + degrade roots.
    g. Re-audit the complete published tree: every child WAV re-reads to its
       manifest sha256.
    h. Compare manifest bytes, normalized receipts, and exact clip sets between
       the two runs (not only child hashes).
    i. Write dated report
       `docs/reports/synthdetect-s5-pr2b-degrade-YYYY-MM-DD.md` + Verdict
       block in `docs/gpu-contracts.md`.

### Phase 4: land

16. Open PR to GitHub `main` via REST, wait for required checks green
    (lint-test + secrets-scan + coverage), self-merge.
17. Sync Forgejo: `git fetch origin && git push local origin/main:main`.
18. Delete branch on both remotes and locally.
19. Update CLAUDE.md memory `voxint-synthdetect.md`.

## Testing strategy

| Layer | What | How |
|---|---|---|
| Unit (mocked) | argv construction, WAV framing, chain sequencing, manifest assembly, re-audit, preflight, collisions, length sanity, split filter, grandchild skip | Fake Docker executable + synthetic WAVs |
| Contract | multi-root resolution, container pin format, raw-framing identity | Pure assertions, no Docker |
| Acceptance (real) | all 6 recipes + at least 1 chain, determinism, AMR-NB, full-tree audit, manifest-bytes equality, receipt equality | Real ffmpeg in pinned container, real organic audio |

## Risks and open questions

**Risks:**
- Container choice: if no well-known image ships all 6 codecs including
  AMR-NB, a repository-owned image must be built and maintained. Phase 0
  resolves this before any executor code.
- Docker on host: implicit dependency. If the host lacks Docker, executor mode
  fails at preflight.
- ffmpeg determinism edge cases beyond `-threads 1 -filter_threads 1` (e.g.,
  codec-internal RNG, non-deterministic memory allocation). The acceptance gate
  tests this on the pinned platform.

**Open questions:**
- **Turn vs segment selection** (deferred to PR-3): PR-2b degrades all root
  clips in the selected split. PR-3 may restrict to turn clips only via a
  stratum-pattern filter, once the manifest carries enough information.
- **Multi-recipe composition**: PR-2b degrades each parent through one chain
  per invocation. A manifest with children from multiple chains requires
  multiple degrade runs + a subsequent assembly. PR-3 territory.
- **Parallel clip execution**: serial for PR-2b. Parallel (multiprocessing)
  is a performance optimization for PR-3 if needed.

## Review notes

An independent model (codex, planner role) pressure-tested the draft. 14
findings; resolutions:

- **F1 (critical), multi-recipe chain execution [accepted, folded in].** The
  draft described single-recipe flow but `derive_degraded_record` supports
  ordered chains. The revised plan adds sequential chain execution with unique
  per-stage paths, each recipe consuming the previous stage's output.

- **F2 (critical), combined manifest not resolvable [accepted, folded in].**
  The current scorer assumes one corpus root. The revised plan adds
  `resolve_clip_path` with an ordered-roots contract (degrade root first,
  parent second) and contract-tests it in PR-2b. Scorer integration is PR-3,
  but the resolution contract is defined and tested now.

- **F3 (high), parent re-audit incomplete [accepted, folded in].** The draft
  checked only sha256. The revised plan also verifies sample_count and
  `duration_s == sample_count / 16000`.

- **F4 (high), parent input scope [accepted, folded in].** Preflight now
  rejects v2 manifests, manifests with existing degraded entries, and spoof
  labels (DegradedRecord does not carry generator provenance).

- **F5 (high), finalize_manifest bypass needs replacement rails [accepted,
  folded in].** `_assemble_combined_manifest` reproduces the exact
  measurement-coverage checks (keyset equality, sha format, positive int
  sample_count, duration derived from count) before calling `load_manifest`.

- **F6 (high), container ENTRYPOINT composition [accepted, folded in].**
  Explicit `--entrypoint ffmpeg` with `argv[1:]` to avoid doubled command.

- **F7 (high), container provenance underspecified [accepted, folded in].**
  Receipt now records resolved image ID, os/arch, ffmpeg -version, -buildconf.

- **F8 (high), codec availability as prerequisite [accepted, folded in].**
  Container selection and 6-recipe smoke testing promoted from open question to
  Phase 0 prerequisite.

- **F9 (medium), Docker bind mount hardening [accepted, scoped].** Added
  `--network none`, subprocess timeout, output file validation (regular, non-
  symlink, non-empty). uid/gid, SELinux, cap-drop deferred (single-operator
  tool on trusted hardware).

- **F10 (medium), codec delay sanity bounds [accepted, folded in].** Output
  length must be within 50%-200% of parent length. Broader than any real codec
  needs, narrow enough to catch catastrophic failures. Per-recipe tightening
  after Phase 0 characterization.

- **F11 (medium), path and collision safety [accepted, folded in].** Preflight
  verifies unique child clip_ids, unique child rel_paths, no collision with
  parent IDs or paths, regular-file status in parent root.

- **F12 (medium), parent artifact consistency [accepted, folded in].** Manifest
  bytes hashed before parsing; hash stored in receipt. Immutability is
  operational: must-be-empty check + atomic rename + manifest sha fingerprint.

- **F13 (medium), turn vs segment selection [noted, deferred to PR-3].** The
  manifest has no `kind` field to distinguish turns from segments. PR-2b
  degrades all root clips in the selected split. PR-3 can add a stratum-
  pattern filter.

- **F14 (low), immutability definition [accepted briefly].** Must-be-empty
  destination + atomic rename + content-addressed manifest sha = operational
  immutability for a single-operator tool. Formal immutability (read-only
  filesystem, content-addressed storage) is out of scope.

Codex's suggested validation phases (prerequisite, preflight, execution
mechanics, real-codec acceptance) map directly to the revised Phase 0-3
structure.
