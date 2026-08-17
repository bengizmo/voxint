# Voxint documentation

| Doc | Contents |
|---|---|
| `onboarding.md` | First-run path: guided installer, setup wizard, and the bundled guided tutorial |
| `architecture.md` | Pipeline stages, state machine, data model |
| `domain-packs.md` | Domain packs: manifest, resolution, the per-run frozen snapshot, and which fields shape which stages |
| `gpu-contracts.md` | Versioned HTTP contracts for the ASR / diarizer / embedder services |
| `gpu-smoke.md` | Build + real-inference smoke procedure for the GPU service images |
| `quality-gates.md` | Enhancement failure semantics, matching eligibility + grounding gates, confidence semantics |
| `enrichment-triage.md` | Draft triage: the read-time, explainable review-priority score, its components, and the source-authority allowlist |
| `interpreting-diarization.md` | Reading the output: segment labels vs the turn ledger, short-clip over-splitting |
| `timeouts-and-leases.md` | COMPUTE_TIER timing profiles: stage timeouts and worker leases per tier |
| `harness.md` | Offline scoring harness: `voxint score` file contracts, verdict vocabularies, the cross-space invariant |
| `operations.md` | Deployment, migrations, pipeline operations, recovery, adjudication workflow |
| `testing.md` | Test layers and how to run them; the manual browser-verification procedure for the review console |
| `release-process.md` | Maintainers: how a release is cut (tag → GHCR images → PyPI → GitHub Release) |
| `reports/` | Dated measurement reports (parity screens, bakeoff diagnostics, negative results) — the evidence behind verdict blocks in `gpu-contracts.md` |
| [`../examples/`](../examples/README.md) | End-to-end `voxint score` walkthrough on a synthetic dataset |
