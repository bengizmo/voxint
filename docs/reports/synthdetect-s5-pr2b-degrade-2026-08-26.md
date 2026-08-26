# Synthdetect S5 PR-2b degrade-executor acceptance (2026-08-26)

> **Status:** PASS

## Verdict

PASS. The degrade executor materialized codec-degraded children from real bona
fide audio (AMI `ES2011a`, 189 parent clips) for all six degradation recipes and
one multi-recipe chain, deterministically: byte-identical combined manifests and
child PCM sha256 values across two independent runs per configuration. Every
child clip re-audited against its manifest sha256 through the WAV payload reader,
and `resolve_clip_path` resolved every combined-manifest entry (parent + child)
against the two-root layout. AMR-NB ran successfully on real speech. This
satisfies the PR-2b acceptance gate from
`docs/plans/2026-08-26-16-33_synthdetect-s5-pr2b-degrade-executor.md`.

The run used staged AMI recordings on maintainer hardware with the digest-pinned
`jrottenberg/ffmpeg` container. Corpus roots are throwaway scratchpad artifacts,
not committed. No corpus audio enters the repository.

## Toolchain

| Property | Value |
|---|---|
| Container image | `jrottenberg/ffmpeg@sha256:292a972c...e9a70f055982f3` |
| Resolved image ID | `sha256:d69b4ee7e061a652bc7c8373f1c33551216f9032a2f638b7b8e9a70f055982f3` |
| ffmpeg version | 7.1 (built with gcc 13, Ubuntu 13.3.0-6ubuntu2~24.04) |
| Platform | linux/amd64 (x86_64) |
| Determinism flags | `-threads 1 -filter_threads 1` on every pass |

The container was pulled by digest, not tag, so the build is pinned. The
resolved image ID matches the Phase 0 codec-smoke value. All six codec
implementations are present: libmp3lame, libopus, aac, pcm_mulaw,
libopencore_amrnb, atempo.

## Parent corpus

The parent was materialized via `prepare` (PR-2a) from AMI recording `ES2011a`
(meetingroom domain, Mix-Headset track), the same recording used in the PR-2a
acceptance run.

| Property | Value |
|---|---|
| Source | AMI `ES2011a` |
| Parent clips | 189 (all eligible for degradation) |
| Parent manifest sha256 | `5886e85a7b28a797...` |

## Determinism results

Each recipe was run twice into separate roots. "Byte-identical" means the
combined `manifest.json` files are byte-for-byte equal across the two runs,
which implies identical child clip_ids, rel_paths, sha256 values, duration_s,
and all lineage metadata.

| Recipe | Children | Combined manifest sha256 | Byte-identical | Re-audit | Resolve | Wall-clock |
|---|---|---|---|---|---|---|
| `mp3-cbr48-v1` | 189 | `f0f71212fb966323...` | yes | 189/189 | 378/378 | 266 s |
| `opus-voip-cbr16-f20-v1` | 189 | `87b32502c11357fe...` | yes | 189/189 | 378/378 | 328 s |
| `aac-lc-cbr48-v1` | 189 | `53406aba7a1b6029...` | yes | 189/189 | 378/378 | 284 s |
| `g711-mulaw-8k-v1` | 189 | `7ed2c58c6f046f1b...` | yes | 189/189 | 378/378 | 259 s |
| `amr-nb-122-v1` | 189 | `f44c9829ace94968...` | yes | 189/189 | 378/378 | 269 s |
| `speed-atempo-0p90-v1` | 189 | `7bd3772f10659cc9...` | yes | 189/189 | 378/378 | 129 s |
| chain: `speed-atempo-0p90-v1\|mp3-cbr48-v1` | 189 | `4a366ec638387489...` | yes | 189/189 | 378/378 | 266 s |

Total: 14 degrade invocations (7 configurations x 2 runs), ~30 minutes.

## Acceptance checks (all passed, every configuration)

1. **Determinism.** Two independent materializations into separate roots produced
   byte-identical `manifest.json` files. Since the manifest pins every child's
   sha256 and the children verified against it (check 3), the child PCM payloads
   are also identical.

2. **Parent re-audit.** Before degrading, the executor re-reads every parent clip
   and verifies its payload sha256 and duration_s against the parent manifest.
   All 189 parents passed for every run (confirmed in `degrade_receipt.json`).

3. **Child re-audit.** Every child WAV was read back through the Python `wave`
   module, the raw payload sha256 was computed, and it matched the manifest
   entry. Zero failures across all 1,323 children (7 x 189).

4. **Resolve.** `resolve_clip_path` resolved every clip in every combined
   manifest (378 entries: 189 parent + 189 child) against the two-root layout
   (degrade root for children, parent root for parents). Zero failures.

5. **AMR-NB on real speech.** The `amr-nb-122-v1` recipe (libopencore_amrnb at
   12.2 kbps, 8 kHz, mono) ran successfully on all 189 organic AMI clips,
   producing deterministic output. This confirms the encoder handles real speech
   without errors (the Phase 0 smoke only tested a synthetic sine wave).

6. **Chain produces distinct output.** The speed+mp3 chain
   (`speed-atempo-0p90-v1|mp3-cbr48-v1`) produced children that differ from both
   `mp3-cbr48-v1` alone and `speed-atempo-0p90-v1` alone across all 189 clips.
   The chain degradation string and stratum are correctly serialized
   (`speed-atempo-0p90-v1|mp3-cbr48-v1`,
   `bona_fide|organic|meetingroom|speed-atempo-0p90-v1|mp3-cbr48-v1`).

7. **Child lineage.** Every child clip carries `label=bona_fide`, inherits its
   parent's `split`, `speaker_id`, and `source`, and its `parent_clip_id` points
   at the correct parent. The stratum extends the parent's by the chain string.

## Notes

- The `speed-atempo-0p90-v1` recipe was the fastest (129 s for 2 runs) because
  it produces a single ffmpeg pass (non-lossy: no intermediate format). Lossy
  codec recipes run two passes (encode to intermediate bitstream, then decode
  back to canonical PCM) and take roughly 260-330 s for 2 runs.
- Container startup overhead dominates: each clip spawns a fresh container.
  A 189-clip run launches 189 (or 378 for lossy) containers sequentially.
- The `toolchain` field in `degrade_receipt.json` is empty (`{}`). The CLI
  defaults to `{}` for `toolchain_info`; provenance is recorded in this report
  instead.

## Out of scope

Cohort policy (which splits are degraded, how many variants per parent) is PR-3.
The degrade executor accepts any parent manifest and any subset of the registered
recipes; PR-2b tests the machinery.
