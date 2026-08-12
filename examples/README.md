# Examples — `voxint score` walkthrough

A tiny synthetic dataset exercising all three scoring-harness commands. The
harness is pure and DB-free, so everything here runs on any machine with the
package installed (`pip install voxint`) — no database, no GPU services, no
compose stack. Contracts: [`docs/harness.md`](../docs/harness.md).

All names, voices, and embeddings are fabricated. Embeddings are 8-dimensional
toy vectors (real speaker embeddings are 192-d+); the harness only requires
that a file's vectors agree with its declared `dims`.

Run everything from this directory:

```bash
cd examples
```

## 1. Name accuracy

Six items, nine scoreable slots, deliberately covering every verdict:
correct names (including an alias match — `"D. Fox"` vs truth
`"Daniela Fox"` via `aliases.json`), a wrong person, an over-name where the
truth is abstain, missed names, a correct abstain, and one unscoreable slot.

```bash
voxint score name-accuracy name-accuracy/items.jsonl \
    --aliases name-accuracy/aliases.json \
    --baseline name-accuracy/baseline.jsonl \
    --seed 0 --out report.json
```

Expect in `report.json`: `aggregate` with `tp: 4, fn: 2, fp_wrong: 1,
fp_overname: 1, tn: 1, excluded: 1`, slot `accuracy` ≈ 0.556 with a Wilson
CI, duration-weighted variants, a `risk_coverage` curve (some slots carry
`confidence`), and — because `--baseline` is given — a `paired` block:
McNemar over 2 discordant slots (both improvements: the baseline missed the
alias on ep-002 and named the wrong person on ep-006) plus a seeded
item-clustered bootstrap CI on the mean per-slot delta.

## 2. Acoustic agreement (one voter per embedding space)

Two enrollment files describe the *same* two curated hosts in two different
embedding spaces (`titanet-large-v1`, `ecapa-tdnn-v2`); each has a matching
slots file covering the same five items. Run the voter once per space:

```bash
voxint score agreement --slots agreement/slots-titanet.jsonl \
    --enrollment agreement/enrollment-titanet.json \
    --thresholds agreement/thresholds.json --out verdicts-titanet.jsonl

voxint score agreement --slots agreement/slots-ecapa.jsonl \
    --enrollment agreement/enrollment-ecapa.json \
    --thresholds agreement/thresholds.json --out verdicts-ecapa.jsonl
```

Per-item expectations (the dataset is built to hit each lane):

| item | kind | titanet | ecapa | why |
|---|---|---|---|---|
| ep-001 | curated (host-dana) | `CONFIDENT_HOST_PRESENT` | `CONFIDENT_HOST_PRESENT` | clean high-cosine host slot in both spaces |
| ep-002 | curated (host-lee) | `CONFIDENT_HOST_PRESENT` | `ABSTAIN` (`near_tie`) | in ecapa two slots score within the 0.08 margin — abstain rather than guess |
| seed-01 | curated (host-dana) | `ABSTAIN` (`session_leakage_risk`) | same | the item is in host-dana's `source_item_ids` — a voiceprint never judges the item it was built from |
| neg-001 | negative control | `NO_CURATED_HOST_DETECTED` | same | all cosines below `low_band` with adequate speech |
| neg-002 | negative control | `ABSTAIN` + `contradiction: true` | `NO_CURATED_HOST_DETECTED` | in titanet a slot clears the present-gates on a supposedly host-free channel — a candidate channel-fact error |

Swapping a slots file against the other space's enrollment file is rejected
(`embedding_space` mismatch) — try it; that refusal is the cross-space
invariant doing its job.

## 3. Ensemble fusion

Fuse the two voters (verdicts only — the ensemble layer cannot see vectors):

```bash
voxint score ensemble verdicts-titanet.jsonl verdicts-ecapa.jsonl --out fused.jsonl
```

Expect in `fused.jsonl`:

| item | fused verdict | why |
|---|---|---|
| ep-001 | `SILVER_HOST_PRESENT` | both voters confident on the same slot |
| ep-002 | `FLAG_REVIEW` (`single_voter_confident:…`) | one confident, one abstained |
| neg-001 | `SILVER_NO_HOST` | both confidently absent |
| neg-002 | `FLAG_REVIEW` (`candidate_host_present_on_neg_control`) | one voter flagged a contradiction |
| seed-01 | `ABSTAIN` | no confident signal from either voter |

Reports are deterministic: identical inputs (and `--seed`) produce
byte-identical outputs, so the expectations above are stable.
