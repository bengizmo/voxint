# Architecture decision records

Short, numbered records of the load-bearing decisions behind a significant
change. An ADR captures the context, the decision, and the consequences at the
time it was made, so a later reader can see why the code is shaped the way it is
without reconstructing the argument from a diff.

These are maintainer artifacts in the technical lane. Each opens with a
`> **Status:**` line and stays immutable once accepted: a decision that is later
reversed gets a new ADR that supersedes the old one, and the old one is marked
superseded rather than edited away.

## Index

| ADR | Decision |
|---|---|
| [`0001-media-identity-vs-location.md`](0001-media-identity-vs-location.md) | Media identity is `media_items.id` anchored to an immutable `source_path`; physical location moves to a future `current_path`. Includes the grep-verified byte-opener audit. |
| [`0002-project-membership-invariant.md`](0002-project-membership-invariant.md) | Media belongs to a project through exactly one `media_folders` row via an FK, never by path-prefix inference. |
| [`0003-editor-run-selection.md`](0003-editor-run-selection.md) | The `/media/{id}` editor opens the latest completed run by default, with an explicit `?run=` override and version chooser. |
| [`0004-claim-lifecycle.md`](0004-claim-lifecycle.md) | Review claims are `(media, run)`-scoped, acquired on first edit intent (never on GET), reused per operator, and renewed by heartbeat. |
| [`0005-speaker-profile-provenance.md`](0005-speaker-profile-provenance.md) | Speaker aggregation reads effective-resolution output, and `speaker_profiles` carries per-field provenance for manual vs accepted-enrichment values. |
| [`0006-plugin-scope-native-vs-greenfield.md`](0006-plugin-scope-native-vs-greenfield.md) | The existing optional features (translation, semantic search, LLM enrichment) stay native; the merged plugin framework ships dormant and is reserved for greenfield features that render on their own surface. |
| [`0007-media-operations-journal.md`](0007-media-operations-journal.md) | Byte-touching file operations (move, trash, restore, purge) are recorded in a durable journal with a state machine, CAS-based concurrency, and a reconciler that drives interrupted rows to a consistent terminal state. |
| [`0008-enrichment-persistence-simplification-scope.md`](0008-enrichment-persistence-simplification-scope.md) | The enrichment append-only persistence model is proportionate and stays; implementation scoped to narrow transaction-choreography extraction and two translation integrity gaps (idempotency key, immutability trigger). |

Records 0001-0005 back the Console 2.0 refactor (epic #149); their current-state
contracts are tested in [`tests/contracts/test_console2_characterization.py`](../../tests/contracts/test_console2_characterization.py).
Record 0006 fixes the scope of the plugin architecture (epic #136). Record 0007
covers the journaled media operations (P2c, issue #155). Record 0008 scopes the
enrichment simplification (refactoring plan Phase 2, finding H7). The full
phased plans live in the maintainer's internal notes.
