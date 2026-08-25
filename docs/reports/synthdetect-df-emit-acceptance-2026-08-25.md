> **Status:** S3 acceptance evidence. First real-archive run of the ASVspoof
> 2021 DF emission verb (`tools/synthdetect_df_import.py emit`, issue #144,
> Milestone 1 Session S3), executed on maintainer hardware against the official
> public archives. The merged verb was unit-tested only on synthetic fixtures;
> both design and implementation reviews required an acceptance run on the real
> bytes before the DF corpus is trusted. This is a materialisation and
> byte-identity claim for the 53,392-clip subset, not a benchmark-reproduction
> claim: scoring the subset against the official scorer is the Gate-2 compare
> step, and the full 611k anchor is S4.

# synthdetect DF emission acceptance run (2026-08-25)

## What this run proves

The emission verb takes the operator's four official DF audio archives plus the
keys archive and produces two artifacts: an untouched native FLAC tree (the
Gate-1 view) and a canonical `pcm-s16le-mono-16000-v1` corpus with a
`schema_version: 2` `imported_benchmark` manifest (the Gate-2 view). Until this
run the verb had only ever seen the synthetic tar fixtures in
`tests/unit/test_synthdetect_df_emit.py`. This run drove it end to end on the
real 34.5 GB of public audio and confirmed that the frozen selection, the
extraction guards, the ffprobe gate, the ffmpeg transcode, and the atomic
publish all hold on the genuine data.

## Inputs

The four Zenodo eval parts (record 4835108) and the asvspoof.org keys archive,
all public and unauthenticated. Provenance was cross-checked before the run:
each part's local sha256 matches the value recorded at download, each part's
local md5 equals the md5 Zenodo publishes for that file, and each byte size
matches Zenodo exactly. The four part sha256 digests are now pinned in
`OFFICIAL_ARCHIVE_SHA256` (part00 was already pinned; part01 through part03 were
added on the strength of this cross-check), so the emitter refuses any archive
whose bytes do not match.

| Archive | sha256 (pinned) | Zenodo md5 cross-check |
|---|---|---|
| `ASVspoof2021_DF_eval_part00.tar.gz` | `99273ef0…` | match |
| `ASVspoof2021_DF_eval_part01.tar.gz` | `5c3c749c…` | match |
| `ASVspoof2021_DF_eval_part02.tar.gz` | `04f2fb70…` | match |
| `ASVspoof2021_DF_eval_part03.tar.gz` | `a45bdae8…` | match |
| `DF-keys-full.tar.gz` | `426f93e1…` | (asvspoof.org keys) |

## Result

`emit` exited 0 and reported `ok: true`. Both roots published atomically (no
staging directory survived).

| Property | Value |
|---|---|
| Native FLAC extracted | 611,829 (the full untouched eval tree) |
| Canonical WAVs transcoded | 53,392 |
| `cohort_hash` | `13c4607cf50e9d633226e1cfa85b1ea557bd44495e6e05a1c918533b888952d1` |
| Emitted `manifest_sha256` | `f8e54f92…` |
| Manifest `schema_version` | 2 (`imported_benchmark`) |
| ffmpeg / ffprobe | 6.1.1 |
| `canonicalization_id` | `pcm-s16le-mono-16000-v1` |

The `cohort_hash` equals the value pre-registered in `docs/gpu-contracts.md`
("S3 reproduction pre-registration"), and the 53,392 count equals the
round-half-up 10 % per-stratum draw over the 533,928 eval trials. The selection
is byte-for-byte the same cohort the audio-free `select` verb produced on the
same metadata, so the audio path did not perturb the frozen subset.

## Binding checks

Three clips (first, middle, and last in manifest order) were re-derived from the
published bytes and matched all the way through the identity chain:

- The canonical WAV read back through the runner's own `read_canonical_pcm`
  yields a PCM-payload sha256 equal to the clip's manifest `sha256`.
- The `clip_receipt.jsonl` row for the trial records that same canonical PCM
  sha256, and a `native_flac_sha256` equal to the recomputed sha256 of the
  actual native FLAC on disk.

So for each checked clip the path from a pinned archive byte to the scored
canonical sample is verified, which is the property that lets the Gate-2
comparison trust that a trial id points at its own audio. The
`selection_receipt.json` is internally consistent: its recorded
`clip_receipt_sha256` matches the sha256 of the emitted `clip_receipt.jsonl`
file, and it stamps the four archive sha256 digests, the keys sha256, the tool
versions, and the strata.

## Timing and cost

Extraction of the full native tree from the four verified 8 GB archives took a
few minutes. The serial transcode loop ran at roughly 528 clips per minute
(one ffprobe plus one ffmpeg per trial), so the 53,392 clips took about an hour
and forty minutes. The transcode loop, not the extraction, is the wall-clock
cost of an emission. This is acceptable for a one-time materialisation; the
emitted corpus is reused for every subsequent scoring run.

## Scope and clean-room note

The emitted native tree, canonical corpus, and receipts are run outputs and are
not committed to the repository. They record no hostnames or absolute operator
paths; a scan of the manifest and both receipts for machine identifiers came up
clean. The corpus is the input to the Gate-2 compare step (our container scores
it, the official scorer scores the same subset via the native path, and
`tools/synthdetect_eval.py compare` measures the drift), which is the next slice.
