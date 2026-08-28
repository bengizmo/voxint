# Benchmark

Voxint ships a small corpus of openly licensed audio clips with known reference
transcripts. The benchmark runs them through the pipeline and scores the results,
giving you a repeatable way to measure transcription accuracy and throughput on
your hardware. Use it to compare configurations, track regressions after an
upgrade, or verify a fresh deployment.

The benchmark is a smoke test for regression detection, not an absolute quality
measurement. It scores word error rate (WER) on short single-speaker clips plus
hallucination resistance on silence and non-speech audio. Results are stored in
the database for cross-run comparison.

## Corpus

12 files, about 3 MB total:

| Files | Source | License | Purpose |
|---|---|---|---|
| `libri_01` through `libri_09` | LibriSpeech test-clean | CC-BY-4.0 | WER baseline (single speaker, clean speech) |
| `short_clean_00` | espeak-ng TTS | CC0-1.0 | Known-transcript reference |
| `silence_00` | Generated silence | CC0-1.0 | Hallucination resistance (expect empty output) |
| `bait_00_applause` | Synthesized non-speech | CC0-1.0 | Hallucination resistance (non-speech audio) |

The corpus, manifest, and attribution ship inside the `voxint` package under
`src/voxint/benchmark/assets/`.

## Running a benchmark

```bash
# Full corpus (12 files)
voxint benchmark run

# Speech files only, skip silence/bait (faster, WER-only)
voxint benchmark run --quick

# Tag a run for later comparison
voxint benchmark run --tag "gpu-batch16"

# Per-file timeout (default 300 seconds)
voxint benchmark run --timeout 600
```

The runner submits each file serially through the pipeline, polls until
completion, then scores against the reference transcript. Progress prints to
stderr; the run UUID prints to stdout on completion.

Exit codes: 0 on success, 1 on partial failure (some files failed), 2 on setup
error (missing services, bad config).

Ctrl+C during a run marks unfinished items as skipped and saves partial results.

## Viewing results

```bash
# List recent runs
voxint benchmark list
voxint benchmark list --limit 20

# Compare two runs side by side
voxint benchmark compare <RUN_ID_1> <RUN_ID_2>
```

The `compare` command warns (but does not refuse) when the two runs used
different corpus versions or scorer protocols.

Benchmark results also appear in the Settings page under the Benchmark section.

## What it measures

**Pooled micro-WER.** The primary metric. Word error rate pooled across all
speech-category files: total (substitutions + insertions + deletions) divided by
total reference words. Silence and bait files are excluded from the WER pool.

**Hallucination metrics.** For silence and non-speech files, the benchmark counts
how many words the pipeline emitted. Ideally zero. Reported separately from WER.

**Per-file timing.** Wall-clock time per pipeline stage and per file. The first
file's timing is included (no warm-up pass), but the serial submission protocol
makes per-file numbers consistent.

**Configuration snapshot.** Each run records the Voxint version, system info
(CPU, RAM, OS), and effective settings, so comparisons can show what changed.

## Scorer

The benchmark uses a shipped lightweight normalizer and standard Levenshtein WER
computation. The normalizer lowercases, strips ASCII punctuation, and collapses
whitespace. It is simpler than the whisper `EnglishTextNormalizer` (which is not
a shipped dependency) and documented as such. The scorer protocol is fingerprinted
so runs scored under different rules can be detected.

Hypothesis extraction uses `raw_text` from transcript segments in chronological
order, with no enhancement or correction applied.

## Benchmark isolation

Benchmark files are stored under a reserved `benchmark/` prefix in the media
root. Pipeline runs from benchmark submissions do not appear in the review queue.
They are normal pipeline runs in every other respect (all stages execute,
including enrichment).

## Database

Two tables store results:

- `benchmark_runs`: one row per run with tag, status, corpus version, protocol
  fingerprint, config snapshot, system info, and a summary JSONB (pooled WER,
  total time, hallucination counts).
- `benchmark_items`: one row per file per run, with a foreign key to the
  pipeline run, per-stage timings, WER counts, and hallucination word count.

Migration: `alembic/versions/0047_benchmark_runs.py`.
