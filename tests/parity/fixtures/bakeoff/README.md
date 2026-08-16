# Whisper Metal ASR bakeoff corpus (issue #33)

Fixed, deterministically selected corpus for the pre-registered whisper engine
bakeoff (`docs/gpu-contracts.md`, "Whisper Metal ASR engine (issue #33) —
pre-registered bakeoff gate"). Acquired and verified by
`tools/prepare_bakeoff_corpus.py`; sources pinned in
`tests/parity/bakeoff/corpus_sources.py`.

## What is (and is not) committed

| Artifact | Committed? | Why |
|---|---|---|
| `manifest.json` | **yes** | The corpus definition: per-file `dataset`, `upstream_id`, `sha256`, `duration_s`, `strata[]`, `license_spdx`, `transcript_sha256`, `ts_granularity`. |
| AMI word-aligned gold + frozen CT2 baseline | **yes** | AMI is CC-BY-4.0 — derived references are redistributable with attribution. |
| AMI / TED **audio** | **no** | Repo size; fetched + checksum-verified locally. |
| TED-LIUM 3 **transcripts** | **no** | CC-BY-NC-ND-3.0 — metrics only, never redistributed. |

Audio and TED transcripts are fetched to a work dir **outside** the repo
(default `~/.voxint-metal/bakeoff/`), never staged.

## Sources (see `corpus_sources.py` for the pins)

- **AMI IHM** (`CC-BY-4.0`, `ts_granularity: word`): individual `Headset-{channel}.wav`
  files + the official manual NXT annotations v1.6.2 (word-aligned references).
  Channel mapping comes from `meetings.xml` — it is **not** positional
  (agent A ≠ channel 0 in general). Only AMI supplies word-level gold, so only
  AMI files are eligible for the segment/word **boundary-drift** gate.
- **TED-LIUM 3** (`CC-BY-NC-ND-3.0`, `ts_granularity: segment`): the legacy
  Release-3 dev+test archives (19 full-talk `.sph`+`.stm` pairs, ~481 MB) at a
  pinned `kfajdsl/tedlium` commit — the official OpenSLR source is gone and the
  old HF endpoint 401s. STM is segment-level only and carries
  `ignore_time_segment_in_scoring` masks that **must** be honored in WER
  scoring (masked speech is excluded, not counted as insertions).
- **Synthetic** (`CC0`, `ts_granularity: none`): the existing espeak-ng fixtures
  for controlled silence / 30 s-window seam / hallucination-bait, referenced —
  not re-downloaded.

## Workflow

```
# maintainer, once, with network — writes a CANDIDATE manifest for review:
python tools/prepare_bakeoff_corpus.py generate

# normal / gate — re-fetch + verify every file against the committed manifest,
# fail closed on any mismatch:
python tools/prepare_bakeoff_corpus.py prepare
```

Checksums are **pre-committed**: `generate` records the sha256 of the exact
prepared 16 kHz mono PCM audio fed to Whisper (and a `transcript_sha256` over
the canonical scorer input — integer-microsecond times, not floats); `prepare`
only verifies. A mirror re-encode changes `sha256` and stops the run; it never
changes the logical `upstream_id`.

## Determinism

File selection is a seeded hash-rank (`SELECTION_VERSION` / `SELECTION_SEED` in
`corpus_sources.py`), not hand-picking — so the corpus is reproducible and the
pre-registration holds. Changing the seed or version is a visible, deliberate
event recorded in the manifest.
