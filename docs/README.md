# Voxint documentation

New to Voxint? Start here:

| Doc | Contents |
|---|---|
| [`setup.md`](setup.md) | **Install Voxint** on your operating system and hardware: prerequisites, guided vs. manual install, and every compute tier (CPU / NVIDIA / AMD / Apple) |
| [`onboarding.md`](onboarding.md) | First-run path once it's installed: guided installer, setup wizard, and the bundled guided tutorial |
| [`how-to/`](how-to/README.md) | **Day-to-day task guides** for non-technical operators: add media & manage runs, review & adjudicate, manage speakers & export, settings & troubleshooting, and (advanced) changing pipeline models |

Reference & internals:

| Doc | Contents |
|---|---|
| `architecture.md` | Pipeline stages, state machine, data model |
| `semantic-search.md` | Transcript semantic search: what the embedding index is, building/refreshing it with `voxint embed backfill`, and the weights requirement on native installs |
| `annotations.md` | Operator annotations: the anchor contract (kinds, coordinate mapping, hashing, staleness, refresh, API taxonomy) |
| `domain-packs.md` | Domain packs: manifest, resolution, the per-run frozen snapshot, and which fields shape which stages |
| `gpu-contracts.md` | Versioned HTTP contracts for the ASR / diarizer / embedder services |
| `gpu-smoke.md` | Build + real-inference smoke procedure for the GPU service images |
| `quality-gates.md` | Enhancement failure semantics, matching eligibility + grounding gates, confidence semantics |
| `enrichment-triage.md` | Draft triage: the read-time, explainable review-priority score, its components, and the source-authority allowlist |
| `interpreting-diarization.md` | Reading the output: segment labels vs the turn ledger, short-clip over-splitting |
| `timeouts-and-leases.md` | COMPUTE_TIER timing profiles: stage timeouts and worker leases per tier |
| `harness.md` | Offline scoring harness: `voxint score` file contracts, verdict vocabularies, the cross-space invariant, and feeding it from live runs (`voxint.harness_export` + the `export_match_evidence` driver) |
| `operations.md` | Deployment, migrations, pipeline operations, recovery, adjudication workflow |
| `native-macos-preview.md` | Technical preview: run the whole stack on macOS/arm64 without Docker, under `launchd` (`scripts/native/voxint-native.sh`) |
| `testing.md` | Test layers and how to run them; the manual browser-verification procedure for the review console; the offline eval-quality harness (DER/JER/WER/cpWER against AMI/VoxConverse) |
| `release-process.md` | Maintainers: how a release is cut (tag → GHCR images → PyPI → GitHub Release) |
| `security/` | Security audits calibrated to the single-operator threat model: the whole-repo audit, its threat model, and the standing findings plus their remediation status |
| `reports/` | Dated measurement reports (parity screens, bakeoff diagnostics, negative results): the evidence behind verdict blocks in `gpu-contracts.md` |
| [`../examples/`](../examples/README.md) | End-to-end `voxint score` walkthrough on a synthetic dataset |

Writing or editing docs? Follow the two-lane house style described in
[the Documentation section of `CONTRIBUTING.md`](../CONTRIBUTING.md#documentation).
Agents should load the `voxint-docs` skill first.
