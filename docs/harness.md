# Scoring harness: `voxint score`

The harness (`src/voxint/harness/`) is the offline quality-measurement layer:
pure, DB-free cores plus file-based CLI adapters. Nothing in it touches
settings, the database, or the worker. `voxint score …` runs on any machine
against plain JSON/JSONL files (`pip install voxint` is all it needs; no
Docker stack).

All JSON documents (aliases, enrollment, thresholds) and all *output* records
carry `"schema_version": 1`; *input* JSONL streams (name-accuracy items,
agreement slots) are versioned by their command's contract rather than per
record. Unknown extra fields are ignored; missing/malformed required fields
(including non-finite numbers and negative durations) are an error reported
with file and line number, exit code 2. Reports are written atomically (temp
file + rename) with deterministic key ordering, so identical inputs produce
byte-identical outputs. A small synthetic dataset exercising all three
commands lives in [`examples/`](../examples/README.md).

## Why these scorers exist

Structural diarization metrics (DER/JER, permutation-optimal WER variants)
optimally relabel speakers before scoring, so they are blind to whether the
*name* shown to a user is the right person. The harness measures exactly that:

- **`score name-accuracy`** scores assigned display names against ground
  truth with a strict person-matcher (a bare first name is not proof of
  identity), and compares two runs with paired statistics.
- **`score agreement`** reports one embedding voter's conservative acoustic
  verdict: is this curated host's voice actually present in this item, judged
  by cosine against a *held-out* voiceprint, independent of any LLM or name
  surface.
- **`score ensemble`** fuses two voters' verdicts. Verdicts only: the
  ensemble layer cannot see vectors, so cross-embedding-space comparison is
  structurally impossible (see "The cross-space invariant" below).

## `voxint score name-accuracy`

```
voxint score name-accuracy items.jsonl [--baseline base.jsonl] \
    [--aliases aliases.json] [--target-accuracy 0.95] [--seed 0] [--out report.json]
```

### Input: items JSONL (one object per line)

```json
{"item_id": "ep-001",
 "slots": {
   "SPEAKER_00": {"assigned_name": "Dana Fox", "truth": "Dana Fox",
                  "confidence": 0.91, "duration": 412.5},
   "SPEAKER_01": {"assigned_name": null, "truth": "__ABSTAIN__"}}}
```

- `item_id`: unique non-empty string; a duplicate is an error.
- `slots`: non-empty object of slot label → fields.
- `assigned_name`: the display name the system produced, or `null`/a
  placeholder (`speaker_…`, `auto_…`, `unknown…`) for an abstention.
- `truth`: a real person name, `"__ABSTAIN__"` (no name should be assigned),
  or `"__NEITHER_DETERMINABLE__"`/`null` (unscoreable → excluded).
- `confidence` (optional): feeds the *descriptive* risk-coverage curve only.
- `duration` (optional): the slot's weight in the duration-weighted metrics;
  omitted slots weigh 1.0.

### Aliases JSON (optional)

```json
{"schema_version": 1,
 "aliases": {"Daniela Fox": ["Dana Fox", "D. Fox"]}}
```

Both names appearing under one canonical entry (the key counts as a member)
match at `alias` strength.

### Verdicts

Per slot: `TP` (correct person), `FP_WRONG` (a different real person),
`FP_OVERNAME` (named where truth is abstain), `FN` (missed a real person),
`TN` (correct abstain), `EXCLUDED` (unscoreable). Name matching is
Unicode-aware (NFKC + casefold) and strict: id equality > alias table > exact
string > surname + given(-initial); single-token containment never matches.

### Output report

One JSON object: counts and precision/recall/F1 (plain and duration-weighted),
a confusion matrix, slot accuracy with a 95% Wilson CI, `per_item` verdicts,
and, when any confidence was supplied, a descriptive `risk_coverage` curve
(never a gate input: confidence is not proven calibrated).

With `--baseline`, both files must cover identical `item_id`s and slot labels
and agree on every slot's `truth` (paired statistics are meaningless across
diverging ground truth, so a mismatch is an error); the report gains a `paired`
block: exact McNemar on discordant slot pairs plus
an item-clustered bootstrap CI on the mean per-slot delta (deterministic for a
given `--seed`).

## `voxint score agreement`

```
voxint score agreement --slots slots.jsonl --enrollment enrollment.json \
    --thresholds thresholds.json [--out verdicts.jsonl]
```

### Enrollment JSON: one embedding space per file

```json
{"schema_version": 1, "embedding_space": "acme-voice-v1", "dims": 192,
 "voiceprints": {
   "host-dana": {"embedding": [0.01, "…"], "enrollment_items": 5,
                  "held_out": true, "source_item_ids": ["ep-002", "ep-003"]}}}
```

- `embedding_space`: the model/space tag; every vector in this file and in
  the slots file is bound to it.
- `held_out`: attests the voiceprint was built only from *other* items. A
  false value abstains every use (`session_leakage_risk`).
- `source_item_ids`: the items the voiceprint was built from. Scoring an item
  in this list abstains (`session_leakage_risk`): a voiceprint must never
  judge the item it was built from.
- `enrollment_items` below the thresholds' `min_enrollment_items` abstains
  (`weak_enrollment`).

### Thresholds JSON

```json
{"schema_version": 1, "tau": 0.62, "margin": 0.08, "min_duration": 45.0,
 "min_segments": 6, "low_band": 0.35, "neg_min_total_duration": 300.0,
 "min_enrollment_items": 3}
```

Validated on load: `low_band <= tau`, cosines within [-1, 1], non-negative
floors. Choose values by impostor-trial calibration
(`voxint.harness.agreement.far_frr_at` + one-sided Wilson bounds).

### Slots JSONL

```json
{"item_id": "ep-001", "kind": "curated", "host_id": "host-dana",
 "embedding_space": "acme-voice-v1", "total_speech": 1810.0,
 "slots": {"SPEAKER_00": {"embedding": [0.02, "…"], "duration": 412.5,
                           "segments": 44}}}
```

- `kind`: `curated` (score `host_id`'s voiceprint) or `negative_control` (a
  no-host channel: score *all* usable voiceprints, expecting absence).
- `embedding_space` is required on every record and must equal the enrollment
  file's. The record proves its space rather than inheriting a tag, so
  vectors from a different (even equal-dimensional) model are rejected.
- Embeddings are validated (finite, non-zero, `dims`-length).

### Output verdicts JSONL

One object per item: `verdict` (`CONFIDENT_HOST_PRESENT`,
`NO_CURATED_HOST_DETECTED`, or `ABSTAIN` + `reason`), evidence
(`host_slot`, `top_cosine`, `runner_up_cosine`, `margin`, duration/segments),
a `contradiction` flag (curated host confidently absent on their own channel,
or present on a negative control: candidate channel-fact errors for human
review), and the `embedding_space`. Verdicts are **silver** evidence, never
gold truth; the bias is deliberately conservative (abstain on near-ties, short
slots, weak/leaking enrollment, low cosine).

## `voxint score ensemble`

```
voxint score ensemble titanet-verdicts.jsonl other-verdicts.jsonl [--out fused.jsonl]
```

Joins two agreement outputs on `item_id` (must cover identical items with
matching `kind`s, and the two files must be in *different* embedding spaces;
two runs of the same model are not independent voters). Records are validated
semantically before fusion: verdicts must fit the item kind, a
confident-present must carry its winning `host_slot`, `contradiction` must be
a literal boolean. Then it AND-gates them: both confident on the same slot →
`SILVER_HOST_PRESENT`; both confidently absent (negative controls) →
`SILVER_NO_HOST`; any contradiction, slot mismatch, or single-voter confidence
→ `FLAG_REVIEW`; otherwise `ABSTAIN`. Voter names in reasons are the files'
embedding-space tags.

## The cross-space invariant

Different embedding models emit vectors in different spaces, and two spaces
can share a dimensionality, so a dims check is not an isolation check. The
harness enforces isolation structurally:

1. Every vector is a `TaggedVector` carrying its `embedding_space`; every
   cosine entry point (`voxint.harness.vectors.cosine`) refuses mismatched
   spaces before touching numpy.
2. One agreement invocation handles exactly one space (the enrollment file
   defines it; a slots record scored against it inherits it).
3. Voter fusion (`voxint.harness.ensemble`) accepts only typed verdicts
   (no numpy import, no vector parameter), so cross-space comparison cannot be
   expressed at the ensemble layer.

This mirrors the pipeline-side invariant in `docs/architecture.md`
(`speaker matching filters by embedding_space`); the guardrail tests in
`tests/unit/test_harness_vectors.py` and `tests/unit/test_harness_ensemble.py`
pin it.

## Library-only pieces

- `voxint.harness.gate_metrics.assemble_gate_metrics`: paired
  baseline/candidate verdict records → release-gate counts (host regressions,
  correct-to-wrong swaps, over-naming introduction, audited-subset
  regressions, one-sided-95% Wilson upper bound on the item regression rate).
  A `paired_tally` with `n_items` 0 yields an upper bound of 1.0: "no
  information" must read as unprovable, not as safe.
- `voxint.harness.goldset_strata` provides deterministic (SHA-256-spread)
  priority-ordered stratified sampling plus provenance-gated auto-labeling:
  a channel fact auto-labels a host truth only when the host's voiceprint is
  groundable on that item; everything else routes to a human label queue.

These have no CLI yet; if one grows a public workflow, it gets a `score`
subcommand and a contract section here.

## Relation to the private ancestors

The cores are fresh implementations of scorers developed in the private
upstream project. Deliberate public-API divergences: identity ids are opaque strings
(not integers), name normalization is Unicode NFKC + casefold (not
ASCII-lowercase), goldset hashing is SHA-256 (not MD5), the surface
reconciliation layer (private storage schema) was not ported, and gate-metrics
key names are generic (`baseline`/`candidate`/`audit`).
