# Eval-quality harness (diarization + ASR tripwire)

A maintainer-only instrument that scores the pipeline's diarization and
transcription against public ground truth. It is not the `voxint score` name
harness in [harness.md](harness.md): that one grades speaker-name accuracy on
your own runs, this one grades the structural numerics (who spoke when, and
what words) against annotated corpora. Different tool, different question.

The code is `tools/eval_quality.py` (issue #97). It never ships to users, never
runs in CI, and never publishes a single combined score.

## Why it exists

Voxint's numerics doctrine treats model output as contract: a change that could
move inference numerics needs measured evidence, not reasoning. The GPU-knob
work in #96 is exactly such a change. This harness produces that evidence for
the two stages a parity fixture cannot cover end to end, diarization and ASR, by
scoring against AMI and VoxConverse.

It is a tripwire, not a benchmark. The scoring subset is 14 recordings, which
cannot prove non-regression; it can only catch gross breakage. Every threshold
must be measured from a zero-change noise floor (run the harness K times with no
code change and read the spread), never guessed. A number here is a smoke alarm,
not a leaderboard.

Scope is English-only, on purpose: WER and cpWER use the frozen English Whisper
normalizer from the bakeoff stack as their tokenizer. Non-English corpora are
out of scope until that changes.

## The three subcommands

The workflow is `run` (produce hypotheses) then `score` (grade them) then
`report` (render the dated Markdown). Canonical invocation prefix:

```
uv run --isolated --extra parity --extra eval-quality tools/eval_quality.py <subcommand> ...
```

`score` and `report` are pure and need no worker or GPU (the pyannote/jiwer
imports are lazy). `run` needs a live pipeline (worker + model services + a
Postgres it can poll).

### `run` — drive the pipeline over a subset

Submits each recording in the subset through the running stack, polls the
database to completion, reads back the durable output, and writes a self-contained
score bundle (relabelled hypothesis RTTM/text plus a `manifest.json` and a
crash-safe journal). It is resumable and never mutates the app: submission goes
through the same `submit_media_item_if_new` seam a normal ingest uses, with a
unique staged media path per pass.

| Flag | Meaning |
|---|---|
| `--corpus {ami,voxconverse}` | required; which ground-truth corpus |
| `--subset PATH` | required; scoring-subset JSON (a bare array or `{items:[...]}` / `{files:[...]}`) |
| `--out-dir PATH` | required; bundle + journal output directory |
| `--pipeline-env PATH` | required; the static pipeline-environment JSON (see below) |
| `--corpus-root PATH` | ground-truth root (or `EVAL_CORPUS_ROOT`) |
| `--only IDS` | comma-separated recording ids to restrict to |
| `--database-url URL` | Postgres URL (or `DATABASE_URL`) |
| `--media-root PATH` | host path to stage audio into |
| `--media-subdir NAME` | staged subdirectory (default `eval`) |
| `--container-prefix NAME` | compose project prefix for the fingerprint probe (default `voxint`) |
| `--cuda-visible-devices VAL` | recorded in the fingerprint |
| `--interval SECS` | poll interval (default `10`) |
| `--timeout SECS` | per-recording timeout (default `3600`) |
| `--duration-tol SECS` | audio/annotation duration tolerance (default `2.0`) |
| `--batch-id ID` | provenance tag; a fresh id forces a fresh run (default: fresh uuid) |
| `--resume` | honor an existing `--out-dir` journal |
| `--retry-failed` | re-submit recordings that failed |

### `score` — grade a manifest

Manifest-driven and corpus-layout-agnostic, so no ground-truth path or hostname
is ever hardcoded. It reads a JSON manifest mapping each recording to its
hypothesis and reference files and emits one metrics JSON.

```
tools/eval_quality.py score --manifest <manifest.json> --out <metrics.json>
```

Diarization is scored with `pyannote.metrics` (optimal mapping, collar, overlap,
UEM crop) fed through pyannote's own accumulators, so the pooled DER/JER is a
true micro-average and not a mean of per-file rates. ASR reuses the frozen
bakeoff WER stack verbatim. For AMI it also emits cpWER (concatenated
minimum-permutation WER via meeteval), which needs a per-speaker reference role
in the cohort.

### `report` — render the dated baseline

A pure metrics-JSON to Markdown step. Pass every scored metrics JSON tagged by
corpus; repeat `--run` for K passes or for more corpora.

```
tools/eval_quality.py report --date 2026-08-20 --out docs/reports/eval-quality-baseline-2026-08-20.md \
    --run ami=<ami_pass1.json> --run ami=<ami_pass2.json> --run voxconverse=<vc_pass1.json>
```

Each corpus is scored separately (one metrics JSON per corpus, never a mixed
accumulator), so no single AMI+VoxConverse number is ever published. When a
corpus is passed K times, the report renders the zero-change noise band, the
maximum spread per metric across passes, both pooled and for the worst single
file. That band is a small-sample maximum, a lower bound on a heavy-tailed
distribution, and the report says so; thresholds use the worst-file spread.

## Cohort binding and the fail-closed fingerprint

Every metrics JSON is bound to a cohort: the subset identity, the frozen
references, and the `pipeline-env.json` are hashed into a `cohort_sha256`. The
`report` step refuses to combine passes that do not share one cohort, so you
cannot accidentally average runs from different code, weights, or hardware.

`pipeline-env.json` is a static, hand-authored provenance file (the app image
digest, model-weight shas, GPU identity, runtime versions, decode settings). It
records the true environment and must be authored honestly for the host it runs
on; it is not a knob to copy between machines.

The `run` driver additionally probes the live deploy before and after the batch
(image digests via `docker inspect`, GPU via `nvidia-smi`) and fails closed if
the probe is degraded, changes mid-batch, or disagrees with the static
`pipeline-env` identity. A passing run means the hypotheses were produced under
exactly the attested environment. The GPU leg is currently NVIDIA-only; attesting
a ROCm host is tracked in issue #119.

## Reading the report

- Read each corpus on its own. There is no grand total by design.
- A metric inside the zero-change noise band is noise, not a regression.
- The overlap-deletion floor note explains why DER cannot reach zero on
  overlapping speech; treat it as the corpus floor, not a bug.
- cpWER appears for AMI only, with a per-recording column. `unassigned_words`
  must be zero (every hypothesis word was assigned to a reference speaker).

## Not a CI or release gate

This is a manual maintainer tripwire and is deliberately kept out of CI and the
release checklist: it needs a live worker, a large ground-truth corpus, and GPU
time that GitHub runners do not have. Do not wire it into a workflow. The
automated gates live in [quality-gates.md](quality-gates.md) and
[release-process.md](release-process.md); the parity fixtures that this harness
complements live under `tests/parity/` and are described in
[testing.md](testing.md) and [gpu-contracts.md](gpu-contracts.md).

The harness's own tests do run in the normal suites:
`tests/unit/test_eval_run_*.py`, `tests/parity/test_eval_quality_*.py`, and the
pin-parity contract `tests/contracts/test_eval_quality_extra.py`.
