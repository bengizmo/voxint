# Synthdetect S5 PR-2a prepare-executor acceptance (2026-08-26)

## Verdict

PASS. The ffmpeg-free prepare executor materialized real bona fide corpora from
both organic domains, deterministically and with byte-exact identity, and every
materialized clip re-audited through the scoring-path reader
(`synthdetect_infer.read_canonical_pcm`) against the finalized manifest. This
satisfies the PR-2a acceptance gate from
`docs/plans/2026-08-26-14-21_synthdetect-s5-pr2-executor.md`.

The run used staged AMI (meetingroom) and VoxConverse (webvideo) recordings on
maintainer hardware. The corpus roots and the acquisition manifests are throwaway
scratchpad artifacts, not committed. No corpus audio enters the repository.

## What was run

A harness built an acquisition manifest (per-file sha256 and byte size for each
source recording and RTTM), then called `materialize_prepare` twice into two
separate roots per case, then ran the acceptance checks below.

| Case | Domain | Recording | Turn clips | Segments | Total | Manifest sha256 |
|---|---|---|---|---|---|---|
| AMI | meetingroom | `ES2011a` | 139 | 50 | 189 | `5886e85a7b28a797...` |
| VoxConverse | webvideo | `abjxc` | 2 | 2 | 4 | `5bc0daf1d19a670e...` |

The AMI case split 189 clips into 93 calibration and 96 eval by speaker (holdout
drew none from this single recording, which is expected: splits are assigned per
speaker and one recording holds few speakers).

## Acceptance checks (all passed, both cases)

1. **Manifest validity.** The written `manifest.json` loads and validates through
   `load_manifest` (schema plus lineage integrity).
2. **Determinism.** Two independent materializations into separate roots produced
   identical `manifest_sha256`, byte-identical `manifest.json`, an identical clip
   file set, and byte-identical clip WAVs.
3. **Independent slice oracle.** Every clip payload equals the source recording's
   canonical payload sliced at the plan's integer sample interval
   (`source[start*2:end*2]`), computed independently of the executor. No off-by-one
   and no wrong-source binding.
4. **Overlap identity.** For every turn/segment pair on a recording whose sample
   intervals intersect, the corresponding payload byte ranges are identical (51
   pairs for AMI, 2 for VoxConverse). A merged session segment shares the exact
   bytes of its constituent turns on their overlap.
5. **Whole-tree re-audit.** The on-disk clip set equals the manifest clip set with
   no strays; each on-disk WAV read back through the scoring reader
   `synthdetect_infer.read_canonical_pcm` matched the manifest `sha256` and the
   sample count derived from `duration_s`. Only the three receipts
   (`manifest.json`, `clip_receipt.jsonl`, `prepare_receipt.json`) sit beside the
   clip tree.

Check 5 is the load-bearing one: it proves the numpy-free corpus writer and the
numpy scoring reader agree on clip identity for real audio, not only for the
synthetic coordinate-coded fixtures the CI tests use.

## Notes

- The AMI source recordings are already canonical PCM (16 kHz mono s16le), so
  materialization was decode-and-slice with no codec, matching the measured
  canonicalization contract in `docs/gpu-contracts.md`.
- The acquisition manifest keys recordings by logical id and resolves the staged
  filename through its `rel_path` (for AMI, `ES2011a` maps to
  `ES2011a.Mix-Headset.wav`), so the speaker-mixed headset track is bound to the
  RTTM without renaming.
- Receipts are host-path-free; the clip receipt carries the measured `n_samples`
  and the canonical `sha256` per clip, bound to the source recording and its sample
  interval.

## Out of scope

Degraded children (the ffmpeg codec round trips), the combined parent-plus-child
manifest, and the AMR-NB encoder decision are S5 PR-2b, unchanged by this run.
