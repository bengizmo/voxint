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

## Searching by meaning

The console has two search modes, switched by the **Exact** / **Meaning** tab
strip beside the runs search box. **Exact** is the chronological `/runs` browse
that finds a run by the words it contains. **Meaning** is the ranked `/search`
page: it reads the embedding index and returns a finite top list of passages from
across every transcript, each with the run, the speaker, the time range, and a
link that opens the transcript scrolled to that passage.

A Meaning query fuses three signals over the index in one snapshot:

- a **vector** arm, the cosine nearest passages to the query's own embedding, so
  a paraphrase finds the passage even when the words differ;
- a **lexical** arm, a language-neutral full-text match, so a passage in any
  language stays findable by its own words; and
- an **exact-quote** arm: any `"quoted phrase"` in the query is matched
  literally and floated to the top, so a phrase you remember verbatim wins over a
  near paraphrase. A phrase written `-"like this"` is an exclusion and is not
  promoted.

The vector and lexical arms are combined with reciprocal rank fusion, and one
recording cannot flood the results because passages are capped per run. Meaning
search has no "older" pager: it is a ranked answer, not a chronological feed.

## On by default

Two independent flags govern it, both default on. Set them per instance from
**Settings > Semantic search** (On / Off / use the installation setting for
each), or as the installation default through the environment. A stored setting
overrides the environment default and applies to the next run or query with no
restart.

| Flag | Default | Effect |
|---|---|---|
| `SEMANTIC_INDEX_ENABLED` | `true` | Turns the feature on. Off means no index is built or served. |
| `SEMANTIC_INDEX_AUTOGENERATE` | `true` | Embeds each run as it completes, so search covers new recordings with no manual step. Best-effort: the completed run is never affected, and if the enqueue does not go through, a background recovery pass re-dispatches the stranded job on its own (you can also run `voxint embed backfill` to index it at once). Requires `SEMANTIC_INDEX_ENABLED=true`. |

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
  job failed. If an earlier run left an embedding job stranded in the queue,
  backfill adopts and runs that job instead of skipping it. It skips a run only
  when a job is actively running for it on a worker (let it finish, or cancel it
  from the run page) or when the run has no resolvable transcript, so those may
  still be unindexed.

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
