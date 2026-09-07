# Speaker-attribution baseline (2026-09-06)

> Generated 2026-09-06. Pipeline git sha `9c80c838c16b9e0db7337b9ec4e434b5382203f6`, Voxint 0.35.0.

This is a maintainer baseline for the speaker-attribution pipeline. It measures the false accept rate (FAR), false reject rate (FRR), and auto-attribution coverage against corpus gold labels. These numbers are not a claim about accuracy on unseen recordings; they catch gross attribution breakage and establish the operating point the calibration phase will tighten.

Truth source is corpus gold (AMI global participant IDs), not post-proposal human adjudication. Session independence is enforced at protocol build time (A2): enrollment and test data must not share a base session.

## Results

| metric | value | 95% CI upper |
| --- | --- | --- |
| FAR | 0.00% | 100.00% |
| FRR | 0.00% | 18.43% |
| Auto-attribution coverage | 100.00% | |

## Trial counts

| category | count |
| --- | --- |
| Genuine trials | 17 |
| Impostor trials | 0 |
| Unscoreable | 78 |
| Auto correct | 17 |
| Auto wrong | 0 |
| Review | 0 |
| Abstain | 0 |
| Speaker clusters | 13 |

## Alignment attrition

| classification | count |
| --- | --- |
| genuine | 22 |
| impostor | 0 |
| mixed | 22 |
| no_gold_overlap | 0 |
| unscoreable_coverage | 11 |
| unscoreable_eligibility | 37 |
| unscoreable_margin | 3 |
| unscoreable_purity | 0 |

## Noise floor (2 zero-change runs)

FAR spread: 0.00%. FRR spread: 0.00%. Coverage spread: 0.00%.

## Limitations

AMI Mix-Headset recordings (multi-speaker far-field). Baseline-only status, not calibration certification. Effective sample counts may be small; a zero impostor count makes FAR undefined (the CI upper bound is not meaningful). Wilson CIs assume independent labels; with clustered speakers the true interval may be wider.
