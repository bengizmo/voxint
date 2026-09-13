> **Status:** NO_DECISION. The pre-registered certification protocol returned NO_DECISION on the confirm split. The safety requirement, zero `auto_wrong`, was met across all 311 scoreable trials. The procedural sample-size floor was not met on the confirm half. Defaults are unchanged.

# Speaker-attribution calibration (2026-09-13)

This maintainer report records the confidence-calibration result for issue #114.

## 1. Protocol summary

The calibration used 170 AMI Mix-Headset meetings: 44 enrollment meetings, 10 genuine-test meetings, and 116 open-set meetings. The split seed was `ami-attribution-split-v1`; assignment hashes the string `seed:meeting_id` with SHA-256. A base-session component graph kept the DEV and CONFIRM splits speaker-disjoint.

The candidate grid contained 32 points: eight grounded cosine thresholds from 0.66 through 0.80, crossed with four grounded margin thresholds from 0.05 through 0.12. Only grounded gates varied. Accept-tier gates and eligibility floors remained at the base configuration.

Committed evidence is under `media/.benchmark/ami/attribution/calibration/`. The paths below are relative to `media/.benchmark/ami/attribution/`.

| artifact | purpose |
| --- | --- |
| `calibration/selection.json` | DEV selection result |
| `calibration/certification.json` | CONFIRM certification result (`NO_DECISION`) |
| `calibration/metrics_all.json` | Pooled metrics, descriptive only |
| `calibration/base_gates.json` | Base gate configuration used |
| `calibration/environment.json` | Image digests, git sha, and GPU tier |
| `calibration/trials.json` | All 1,229 trials with split assignments |
| `calibration/roster/enrolled_speaker_map.json` | The 48 of 62 cross-session speakers enrolled |

| environment field | value |
| --- | --- |
| Git sha | `41dda4bfcc1351f6b71e276cbff1432dd2423c68` |
| Voxint | 0.35.0 |
| GPU | RTX 5090 32GB, maintainer hardware |
| Images | App, Whisper, pyannote, and TitaNet images built from the checkout; digests recorded in `calibration/environment.json` |

## 2. DEV selection

DEV returned `SELECTED` with `grounded_min_cosine=0.80` and `grounded_min_margin=0.12`, the strictest grid point. All 32 grid points were feasible and had zero `auto_wrong` over 55 impostor clusters, 4 genuine clusters, and 168 scoreable trials. The one-sided 95% Wilson FAR upper bound was 4.69%.

Every grid point produced the same trial classifications: 4 `auto_correct`, 0 `review`, and 164 `abstain`. The strictest point won only through the protocol's tie-breaking rule. This corpus does not empirically distinguish any threshold in the grid at this operating point.

## 3. CONFIRM certification

CONFIRM returned `NO_DECISION` for both recorded reasons:

- `insufficient_impostor_clusters`: 49 impostor clusters, below the floor of 50.
- `far_bound_exceeded`: the one-sided 95% Wilson FAR upper bound was 5.23%, above the 5.00% ceiling.

| metric | CONFIRM value |
| --- | --- |
| Genuine clusters | 9 |
| Impostor clusters | 49 |
| Genuine trials | 13 |
| Impostor trials | 130 |
| Scoreable trials | 143 |
| `auto_wrong` | 0 |
| FAR upper, one-sided 95% Wilson | 5.23% |

PRE used the base grounded gates, 0.70 cosine and 0.08 margin. POST used the selected candidate, 0.80 cosine and 0.12 margin. There were zero band changes. PRE and POST both produced 13 `auto_correct`, 0 `review`, 130 `abstain`, and 0 `auto_wrong` on the confirm trials.

## 4. Protocol design finding

The 50-cluster floor, `MIN_INDEPENDENT_CLUSTERS`, was imported from the recurrence viability check. It was not derived from the Wilson method used for certification.

With zero errors, the one-sided 95% Wilson upper bound is `z² / (n + z²)`, where `z=1.645`. The bound clears the 5% ceiling only at `n >= 52`. A result of 0/50 has a 5.13% upper bound, and 0/51 has a 5.04% upper bound, so both fail certification. The floor of 50 was never independently binding. The FAR ceiling was the effective power check, and the confirm half needed at least 52 impostor clusters.

The complete corpus yielded only 104 impostor clusters. A two-way split therefore cannot reliably place at least 52 in each half. Future revisions should derive the sample floor from the confidence method and FAR ceiling, then check split feasibility against that derived requirement.

## 5. Pooled evidence (descriptive supplement)

The pooled result was 0 `auto_wrong` over 104 impostor clusters. Its one-sided 95% Wilson FAR upper bound was 2.5%. These are descriptive results, not confirmatory.

| metric | pooled value |
| --- | --- |
| Scoreable trials | 311 |
| Genuine trials | 17 |
| Impostor trials | 294 |
| Unscoreable trials | 918 |
| Genuine clusters | 13 |
| Impostor clusters | 104 |
| `auto_wrong` | 0 |
| Auto-attribution coverage | 5.5% |
| FAR upper, one-sided 95% Wilson | 2.5% |

Selection and certification used different splits of the same data. Pooling includes DEV, which is in-sample for selection. The pooled result is supporting evidence for the safety of the existing defaults, but it cannot replace the confirmatory protocol.

The impostor clusters include both open-set meetings (speakers with no enrollment opportunity) and unenrollable cross-session speakers (14 of 62 failed the roster builder's purity/coverage/eligibility floors). Both types produce `impostor_open` trials, so the 104 impostor clusters are a mix of true open-set and enrollment attrition. This is the conservative direction: every grounded match for these speakers counts as `auto_wrong`.

## 6. Defaults decision

Defaults remain at `grounded_min_cosine=0.70` and `grounded_min_margin=0.08`.

Certification returned `NO_DECISION`, so there is no measured basis for tightening the gates. The corpus also produced identical classifications at all 32 grid points. Moving to 0.80 cosine and 0.12 margin would therefore be a judgment call with no measured benefit on these data.

## 7. Future work

1. Derive `MIN_INDEPENDENT_CLUSTERS` from the Wilson method and FAR ceiling instead of importing a fixed constant.
2. Evaluate a second corpus, such as VoxConverse or deployment-representative recordings, to add genuine and impostor clusters.
3. Consider a resampling-robust design, such as leave-one-meeting-out with per-fold selection, that reduces single-split variance while preserving train/test separation within each fold.
