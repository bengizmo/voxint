# Plan: Voxint Benchmark Feature

## Context

Voxint has no user-facing way to measure pipeline performance or transcription accuracy. The maintainer eval harness (`tools/eval_quality.py`) scores DER/WER/cpWER against AMI + VoxConverse ground truth, but it requires heavyweight dependencies (`pyannote.metrics`, `meeteval`) and never ships in the release. Users who deploy Voxint on new hardware have no standardized way to verify their setup produces good results or to compare performance across configuration changes.

The goal is a shipped benchmark feature: a small corpus of openly-licensed audio with reference transcripts bundled into the release, a CLI to run them through the pipeline, a lightweight scorer, and DB-backed results that can be compared across runs.

This is a smoke benchmark for regression detection and configuration comparison, not an absolute quality measurement. It is framed as such in all user-facing copy.

---

## Scope: what v1 measures (and does not)

**In scope:**
- **Transcription accuracy (WER)**: pooled micro-WER across benchmark clips, using a shipped lightweight normalizer + Levenshtein scorer
- **Pipeline throughput**: wall-clock time per stage and end-to-end, with defined timing semantics
- **Hallucination resistance**: word count emitted on silence/non-speech inputs (reported separately, not folded into WER)
- **Configuration provenance**: effective service identities and settings snapshot per run

**Not in scope (v1):**
- Diarization accuracy (DER/JER) -- requires `pyannote.metrics`, not shipped
- Speaker identification accuracy -- requires roster setup
- Auto-optimization -- future consumer of benchmark data
- Auto-run on install -- on-demand only (mention in doctor output and setup completion)

---

## Benchmark corpus (~3 MB, shipped in release builds)

All files CC-BY-4.0 or CC0. Shipped under `src/voxint/benchmark/assets/` following the `tutorial/resources.py` pattern.

### Files (12 total)

| # | File | Source | License | Duration | Purpose |
|---|---|---|---|---|---|
| 1-9 | `libri_01` through `libri_09` | LibriSpeech test-clean | CC-BY-4.0 | 4-7s each | WER baseline (single speaker, clean) |
| 10 | `short_clean_00.wav` | Bakeoff synthetic (espeak TTS) | CC0 | ~4s | Known-transcript TTS reference |
| 11 | `silence_00.wav` | Bakeoff synthetic | CC0 | ~5s | Hallucination resistance (expect empty) |
| 12 | `bait_00_applause.wav` | Bakeoff synthetic | CC0 | ~8s | Hallucination resistance (non-speech) |

**Selection rationale**: All 9 LibriSpeech clips are used (no hand-selection; deterministic). 3 synthetics cover the edge cases. Total ~3 MB.

**Not included**: AMI Mix-Headset excerpts (26-91 MB each, too large for release builds; and without DER scoring they produce no accuracy signal). New multi-speaker synthetic fixture (no scored metric consumes its RTTM in v1; the tutorial fixture already demonstrates diarization).

### Reference data

`src/voxint/benchmark/assets/manifest.json` with:
- `corpus_version`: integer, bumped on any corpus change
- `scorer_protocol`: version string for normalization + aggregation
- Per-file: `id`, `sha256`, `duration_s`, `num_speakers`, `category` (speech/silence/bait), `reference_transcript` (null for silence/bait), `source`, `license_spdx`

### Provenance

`src/voxint/benchmark/assets/provenance.json`: upstream URLs, speaker IDs, selection criteria, conversion commands, generator versions, source and derived hashes. Follows the tutorial provenance pattern.

`src/voxint/benchmark/assets/ATTRIBUTION.md`: human-readable CC-BY attribution for LibriSpeech.

---

## Lightweight WER scorer

`src/voxint/benchmark/scorer.py` (~80 lines)

**Normalizer**: Voxint-owned, minimal (lowercase, collapse whitespace, strip ASCII punctuation, strip leading/trailing whitespace). NOT the whisper EnglishTextNormalizer (which is test-only, requires `more-itertools` + `regex` from the parity extra, and is not a shipped dependency). The benchmark normalizer is simpler and documented as such.

**WER computation**: Standard Levenshtein edit distance on word tokens. Returns `(substitutions, insertions, deletions, reference_words)` per file. Pooled micro-WER as the primary aggregate: `sum(S+I+D) / sum(ref_words)` across all speech-category files. Silence/bait files are excluded from WER and reported as hallucination metrics (emitted word count, non-empty rate).

**Hypothesis extraction**: `raw_text` from `TranscriptSegment`, chronological order by `segment_index`. No enhanced text, no enrichment output, no corrections. This is the deterministic raw-output contract.

**Protocol fingerprint**: A hash of normalizer version + aggregation rules + hypothesis extraction contract, stored with each benchmark run for compatibility checking.

---

## Database schema

New alembic migration (next available revision after current head).

### `benchmark_runs` table

```
id              UUID PK
tag             Text | NULL (user label, max 60 chars)
status          Text CHECK (pending, running, completed, failed)
corpus_version  Integer NOT NULL
protocol_hash   Text NOT NULL (scorer/normalizer fingerprint)
voxint_version  Text NOT NULL
config_snapshot JSONB NOT NULL (effective settings + service health identities)
system_info     JSONB NOT NULL (CPU, GPU, RAM, OS)
summary         JSONB | NULL (populated on completion: pooled WER, total time, hallucination counts)
started_at      TimestampTZ NOT NULL
finished_at     TimestampTZ | NULL
created_at      TimestampTZ server_default now
```

### `benchmark_items` table

```
id                  UUID PK
benchmark_run_id    UUID FK -> benchmark_runs.id ON DELETE CASCADE
corpus_file_id      Text NOT NULL (manifest file id)
pipeline_run_id     UUID FK -> pipeline_runs.id ON DELETE SET NULL
status              Text CHECK (pending, submitted, completed, failed, skipped)
stage_timings       JSONB | NULL (per-stage wall-clock from successful attempts)
wer_counts          JSONB | NULL ({substitutions, insertions, deletions, reference_words})
hallucination_words Integer | NULL (for silence/bait category)
error               Text | NULL
started_at          TimestampTZ | NULL
finished_at         TimestampTZ | NULL
```

Index on `(benchmark_run_id, corpus_file_id)` UNIQUE. Index on `benchmark_runs(created_at DESC)` for list queries.

**Design**: Normalized items (not a JSONB blob) so each item has a real FK to its PipelineRun, partial failures are representable, and per-item queries are efficient.

---

## Runner (`src/voxint/benchmark/runner.py`)

### Execution protocol

1. **Preflight**: verify all model services are healthy (`voxint doctor` checks)
2. **Copy assets**: extract benchmark WAVs from package data into a reserved `benchmark/` subdirectory under MEDIA_ROOT
3. **Submit serially**: one file at a time through `submit_media_item`, poll until completed/failed before submitting the next. Serial submission produces consistent per-file timing unaffected by concurrency contention.
4. **Score**: extract raw_text from TranscriptSegments, run WER scorer against reference, record results per item
5. **Record**: update benchmark_items with timing and scores, compute and store summary on benchmark_run

### Timing semantics

- **Per-stage**: wall-clock of the successful StageRun attempt only (last successful `finished_at - started_at` for that stage). Retry count stored separately.
- **Per-file**: monotonic start-to-finish (from submission to pipeline COMPLETED), includes queue time
- **End-to-end**: monotonic wall-clock from first submission to last completion
- **Warm-up**: the first benchmark file's timing is included but flagged. No separate warm-up pass.

### Lifecycle

- **Timeout**: configurable per-file timeout (default 300s). Exceeded = item marked failed, benchmark continues.
- **Interruption (Ctrl+C)**: mark in-flight items as skipped, mark run as failed, commit partial results. No silent data loss.
- **No cross-restart resume**: a new `benchmark run` always starts fresh. Simplifies v1; resume is a future enhancement.
- **Cleanup**: benchmark WAVs in the reserved `benchmark/` subdirectory are left in place (cheap, reused by future runs). PipelineRuns from benchmark are normal rows but the reserved source path prefix `benchmark/` lets queries exclude them from the review queue.

### Benchmark isolation

- Source paths use the reserved prefix `benchmark/{corpus_file_id}.wav`
- The review queue filters out items under `benchmark/` (a one-line WHERE clause addition)
- Benchmark PipelineRuns are normal runs with all stages (including enrichment). This measures the real pipeline, not a synthetic subset.
- No domain pack, no project assignment for benchmark files (they use the default/global settings)

---

## CLI (`src/voxint/cli.py`)

```
voxint benchmark run [--quick] [--tag TAG] [--timeout SECONDS]
voxint benchmark list [--limit N]
voxint benchmark compare RUN_ID_1 RUN_ID_2
```

### `run`
- `--quick`: only LibriSpeech speech clips (9 files, skip synthetics). Faster, WER-only.
- `--tag TAG`: label for comparison (e.g. "gpu-batch16"). Max 60 chars.
- `--timeout SECONDS`: per-file timeout (default 300).
- Exit 0 on success, 1 on partial failure, 2 on config/setup error.
- Prints progress per file, summary table at end.

### `list`
- Tabular: tag, status, date, WER, total time, file count.
- `--limit N`: default 10.

### `compare`
- Side-by-side: per-file WER and timing, summary deltas.
- Warns (does not refuse) when corpus_version or protocol_hash differ.

---

## Console integration (minimal, in v1)

A "Benchmark" section in the Settings page:

- **Most recent run summary**: WER, total time, file count, tag, date. Shows "No benchmark runs yet. Run `voxint benchmark run` to get started." when empty.
- **"Run Benchmark" button**: triggers `POST /api/benchmark/run` which enqueues the benchmark via a Celery task. Progress shown via polling. Not a React island initially; server-rendered partial that refreshes.
- **Run history table**: last 5 runs with tag, WER, total time, date. Link to per-file details page.
- **Comparison**: select two runs to compare side-by-side (same data as CLI `compare`).

**Files:**
- `src/voxint/api/routers/benchmark.py` -- API routes (run, status, list, compare)
- `src/voxint/api/templates/partials/benchmark.html` -- server-rendered Settings section
- Modify `src/voxint/api/templates/settings.html` -- add Benchmark section

This is step 10 in the implementation order (after CLI is working and tested).

---

## Files to create/modify

**New:**
- `src/voxint/benchmark/__init__.py`
- `src/voxint/benchmark/assets/manifest.json`
- `src/voxint/benchmark/assets/provenance.json`
- `src/voxint/benchmark/assets/ATTRIBUTION.md`
- `src/voxint/benchmark/assets/*.wav` (12 files, ~3 MB)
- `src/voxint/benchmark/resources.py` (asset loader, follows tutorial pattern)
- `src/voxint/benchmark/scorer.py` (normalizer + WER)
- `src/voxint/benchmark/runner.py` (submit, poll, score, record)
- `alembic/versions/XXXX_benchmark_runs.py`

**Modified:**
- `src/voxint/cli.py` -- add `benchmark` subcommand group
- `src/voxint/db/models.py` -- add BenchmarkRun + BenchmarkItem models
- `src/voxint/adjudication/resolver.py` or the review queue query -- filter `benchmark/` prefix
- `src/voxint/api/templates/settings.html` -- add Benchmark section
- `src/voxint/api/routers/settings.py` -- expose benchmark data to Settings context

---

## Implementation steps

1. **Corpus assembly**: copy 9 LibriSpeech WAVs + 3 synthetics into `src/voxint/benchmark/assets/`. Write manifest.json, provenance.json, ATTRIBUTION.md. Verify sha256s.
2. **Scorer**: implement normalizer + WER with unit tests. Broad test vectors: empty, identical, all wrong, Unicode, punctuation-only, WER > 100%.
3. **DB schema**: BenchmarkRun + BenchmarkItem models, migration, CHECK constraints, indexes. Contract tests.
4. **Runner**: submit/poll/score/record with timeout, interruption handling, serial submission. Component tests with fakes for broker/session.
5. **CLI**: `run`, `list`, `compare` subcommands. Integration with runner.
6. **Review queue filter**: exclude `benchmark/` source paths from the adjudication queue.
7. **Release validation**: verify assets ship in built wheel, sdist, and Docker image. Clean-install import test.
8. **Live smoke**: `voxint benchmark run --quick` on a real stack.
9. **Console integration**: Benchmark section in Settings (API routes, template partial, run trigger).
10. **Documentation**: `docs/benchmark.md` (user-facing), CHANGELOG entry.

---

## Testing strategy

- **Scorer**: broad frozen test vectors (known S/I/D counts), empty-reference handling, normalizer edge cases, protocol fingerprint stability
- **DB**: migration up/down, CHECK constraint enforcement, JSONB shape validation, cascade delete
- **Runner**: component tests with fake session/publisher for timeout, interruption, partial failure, serial submission order, stage timing extraction, duplicate-run safety
- **Assets**: sha256 integrity on import, manifest schema validation, clean-wheel import test
- **Release**: inspect built wheel contents for benchmark assets, clean-venv install + import
- **Integration** (maintainer-only): full `benchmark run --quick` on live stack, verify DB rows and output

---

## Risks

1. **Corpus too small for meaningful WER**: 9 clips totaling ~50s. Mitigated by framing as regression detection, not absolute quality.
2. **Normalizer divergence from maintainer harness**: different normalization = different WER numbers. Acceptable: the benchmark is self-consistent; users compare runs against each other, not against external baselines.
3. **Benchmark runs accumulate**: each run creates PipelineRuns. Mitigated by the reserved path prefix (easy to query/delete).
4. **Model changes break comparisons**: a Whisper update changes WER. Not a bug: that is the point. `compare` warns on different voxint_version.

---

## Review notes (codex critique resolution)

| # | Codex finding | Severity | Resolution |
|---|---|---|---|
| 1 | Whisper normalizer not shipped | blocker | **Accept**: ship Voxint-owned minimal normalizer instead |
| 2 | DER claimed but not scored | blocker | **Accept**: narrow to WER + speed. No diarization accuracy in v1 |
| 3 | Runner lifecycle underspecified | blocker | **Accept**: defined serial submission, timeout, interruption, no resume |
| 4 | Benchmark pollutes operator data | blocker | **Accept**: reserved `benchmark/` path prefix, review queue filter |
| 5 | No protocol identity for comparisons | blocker | **Accept**: corpus_version + protocol_hash on every run |
| 6 | Timing protocol undefined | blocker | **Accept**: defined warm-up, serial, wall-clock, attempt reduction |
| 7 | Use pooled micro-WER | major | **Accept**: pool S/I/D counts, not macro-average |
| 8 | Silence/bait zero-denominator | major | **Accept**: separate hallucination metrics, exclude from WER pool |
| 9 | Hypothesis extraction undefined | major | **Accept**: raw_text, chronological, no enhancement |
| 10 | Corpus too small for "accuracy" | major | **Accept**: frame as smoke benchmark for regression detection |
| 11 | Selection bias risk | major | **Accept**: use ALL 9 LibriSpeech clips, no hand-selection |
| 12 | Weak multi-speaker fixture | major | **Accept**: drop new fixture; tutorial WAV already demonstrates diarization |
| 13 | AMI source confusion | major | **Accept**: drop AMI fixture entirely (too large, no DER scoring) |
| 14 | JSONB blob loses integrity | major | **Accept**: normalized benchmark_items table with PipelineRun FK |
| 15 | Missing DB constraints | major | **Accept**: CHECK, indexes, cascade, timezone semantics |
| 16 | Config snapshot incomplete | major | **Accept**: capture service health identities at run start |
| 17 | Stage timing ambiguous | major | **Accept**: successful attempt only, retry count separate |
| 18 | Quick mode definition | major | **Partially accept**: quick = speech files only (skip synthetics) |
| 19 | Orchestration testing gaps | major | **Accept**: component tests with fakes, not just one manual run |
| 20 | Scorer testing weak | major | **Accept**: broad frozen vectors, not just a few pairs |
| 21 | Release validation missing | major | **Accept**: wheel/sdist/image content checks added |
| 22 | Licensing incomplete | major | **Accept**: ATTRIBUTION.md + provenance.json following tutorial pattern |
| 23 | CLI output unspecified | moderate | **Accept**: defined exit codes, progress, warnings |
| 24 | Concurrency uncontrolled | moderate | **Accept**: serial submission is the default protocol |
| 25 | Auto-optimization premature | moderate | **Accept**: treat as future consumer, not v1 validation |
| 26 | CLI-only vs non-technical audience | moderate | **Accept**: label as technical preview |

**User decisions** (2026-08-28):
1. Benchmark is on-demand only (no auto-run on install). Mention in doctor output.
2. Minimal console integration in v1 (Benchmark section in Settings with run trigger + history).
