# Transcript semantic search

Voxint indexes finished transcripts so you can find a passage by meaning, not
only by exact words. This page is the operator and maintainer reference: what the
index is, how to build or refresh it, and the one requirement it has on a
native (no-Docker) install. For the internal design, see the embedding-spine
section of [architecture.md](architecture.md#transcript-semantic-search-the-embedding-spine-issue-121).

## What it is

Exact full-text search finds a passage only when you already know a word in it.
Semantic search embeds each finished transcript into vectors and ranks passages
by similarity, so a query like "who paid for the repairs" can surface a passage
that never uses those words. The embedding runs in-process on sha-pinned MiniLM
ONNX weights baked into the app image: no LLM, no network egress, no external
cost, and nothing leaves the machine.

This release ships the index (the "spine"). The ranked query and its console UI
land in a follow-up; until then the index builds and stays fresh in the
background, ready for that query.

## On by default

Two independent flags govern it, both default on. In this release they are set
through the environment; the resolver already reads them row-over-env (a stored
value overrides the environment default) for the settings toggle that arrives
with the query UI.

| Flag | Default | Effect |
|---|---|---|
| `SEMANTIC_INDEX_ENABLED` | `true` | Turns the feature on. Off means no index is built or served. |
| `SEMANTIC_INDEX_AUTOGENERATE` | `true` | Embeds each run as it completes, so search covers new recordings with no manual step. Best-effort: the completed run is never affected, but if the enqueue does not go through the run is left unindexed until you run `voxint embed backfill`. Requires `SEMANTIC_INDEX_ENABLED=true`. |

The defaults are set for the recommended Docker install, which always bakes the
weights. See the "Transcript semantic search" block in `.env.example` for the
environment form.

## Building the index by hand: `voxint embed backfill`

Autogenerate covers new runs. Use `voxint embed backfill` to index runs that
finished before the feature existed, to catch up after fetching the weights on a
native install, or to reindex after a bulk correction. It drives the same job
lane the worker uses, synchronously, so it needs no broker.

```bash
voxint embed backfill              # index every run whose index is missing or stale
voxint embed backfill --force      # reindex every completed run, stale or not
voxint embed backfill <run_id>     # (re)index a single run
```

A run is **stale** when its transcript changed since it was last indexed (a
whole-transcript content hash detects this). The default form indexes only
missing or stale runs, so re-running it is cheap.

Behavior worth knowing:

- It **refuses up front** (exit code 2) when the feature is disabled or the
  weights are absent, so you see the one fix instead of a failed job per run.
- A run with no resolvable transcript is a **per-run skip**, not a hard failure:
  a corpus sweep does not abort on one unindexable run.
- Exit code 1 means at least one job actually failed. Exit 0 means no attempted
  job failed; runs that were skipped (no resolvable transcript, or a job already
  active for the run) may still be unindexed.

## The weights requirement on native installs

The Docker app image bakes the MiniLM weights, so a Docker install has semantic
search working out of the box with no extra step. A native (no-Docker) install
never fetched that asset, so the weights may be absent. When they are:

- Autogenerate and the backfill CLI **skip honestly** rather than enqueuing jobs
  that cannot run. The finalize hook logs an actionable warning; the CLI exits 2
  with the fix.
- `scripts/native/voxint-native.sh doctor` reports a failing **Semantic search
  weights** check when the feature is on but the weights are missing.

To supply them, fetch `model.onnx` and `tokenizer.json` from the
`minilm-onnx-v1` release asset (their sha256s are pinned in
`src/voxint/embeddings/models/provenance.json`) and point the app at them:

```bash
export VOXINT_MINILM_ONNX_PATH=/path/to/model.onnx
export VOXINT_MINILM_TOKENIZER_PATH=/path/to/tokenizer.json
```

Then run `voxint embed backfill` to index the existing corpus.

## Correctness

The stored vectors are a numerics contract, like every other model output in
Voxint. The ONNX embedder is held to a measured equivalence gate
(`tests/parity/test_text_embedding.py`): each vector must match a reference
generated once in a throwaway sentence-transformers environment to within cosine
0.9999. The embedding space id (`minilm-multi-l12-onnx-fp32-mean-v1`) changes
only when the weights or the pooling change, and a change is a visible reindex,
never silent drift. A weights refresh publishes a new immutable asset release and
updates the provenance file, the Dockerfile sha ARGs, `release.yml`, and (when
the tokenizer hash changes) the `.gitleaks.toml` allowlist together.
