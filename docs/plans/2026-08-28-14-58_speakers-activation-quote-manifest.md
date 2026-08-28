# Plan: #159 Speakers Activation + #122 Quote/Clip Provenance Manifest

## Goal

Two sequential PRs:

1. **#159**: Activate the speakers overview and profile pages by flipping
   `console_speakers_enabled` to `True`. The code is feature-complete
   (aggregation, tiers, profiles, routes, templates, tests); this is the
   activation flip, following the P6b Settings pattern (PR #189).
2. **#122**: Add a JSON provenance manifest to the existing pull-quote and
   clip export paths: quote text, per-line speakers, timecodes, clip
   reference with digest, per-stage model identity from `StageRun.metrics`,
   and the input-media `sha256`. For journalists who need to defend a
   quote's chain of evidence.

Version bump to 0.29.0 after both land.

## Assumptions and constraints

- #159 code is feature-complete and dark-shipped. Activation is a flag flip
  plus documentation and test updates.
- #122 builds on the landed annotation layer (#86), clip service (#88), and
  the existing `model_provenance` / `model_identity` infrastructure.
- `MediaItem.sha256` already exists (nullable, with a backfill command).
  No on-the-fly hashing in GET handlers.
- Per-attempt model identity is already stamped in `StageRun.metrics`
  under the `model_identity` key, with a latest-completed-attempt selector
  in `api/model_provenance.py`. Reuse it; do not fall back to config.
- `ResolvedAnnotation.speakers` is already a tuple of ordered, deduped
  speaker names. The manifest preserves per-line attribution.
- `AudioArtifact.meta` stores `annotation_id`; `idempotency_key` is
  content-addressed from annotation_id + sample bounds. Clip lookup uses
  the same key derivation.
- Both features are numerics-neutral. No parity gates needed.
- The `feat/auto-enroll` branch may land concurrently, adding
  `Decision.AUTO_ENROLL`. Low risk: `TranscriptLine.speaker` resolves
  through the canonical pipeline regardless.

## Part 1: #159 Speakers overview activation

### Proposed approach

Replicate the P6b Settings activation pattern:

1. **Flip default**: `config.py:542`, `console_speakers_enabled` from
   `False` to `True`.
2. **Update .env.example**: line 87 currently says
   `# CONSOLE_SPEAKERS_ENABLED=false`. Update the comment to reflect the
   new default and note rollback via `CONSOLE_SPEAKERS_ENABLED=false`.
3. **Update config docstring**: the dark-ship language in lines 537-542
   should reflect activation.
4. **No nav changes**: the Speakers link is always rendered (no conditional
   wrapper in `base.html:1300-1304`), unlike Settings which needed a
   repoint. The overview handler branches skins internally.
5. **No redirect**: unlike Settings (`/resources` -> `/settings/status`),
   there is no legacy URL to redirect from.
6. **Tests**: add a no-override default-assertion test
   (`Settings(_env_file=None)` pattern from `test_settings_pages.py:113`),
   verify overview renders by default, verify profile routes are live,
   retain an explicit `speakers_enabled=False` rollback test.
7. **CHANGELOG**: entry under `[Unreleased] -> Changed`.

### Alternative considered

Adding the speakers flag to `_shell_template_context` and conditionally
rendering the nav link. Rejected: the `/speakers` route is live regardless
(renders legacy roster when off), so hiding the link would break navigation.

### Affected files

| File | Change |
|---|---|
| `src/voxint/config.py` | Flip default, update docstring |
| `.env.example` | Update speakers flag comment |
| `tests/integration/test_speakers_console.py` | Add default-assertion + rollback test |
| `CHANGELOG.md` | Activation entry |

### Testing strategy

- Default-assertion test: `Settings(_env_file=None)` with no override,
  assert `console_speakers_enabled is True`.
- Rollback test: explicit `CONSOLE_SPEAKERS_ENABLED=false` restores legacy
  roster rendering.
- Full suite run: existing 870+ lines of speaker tests cover both flag
  states.
- Route goldens: verify zero diff (URL exists in both states; content
  differs, but goldens track route structure, not body).

## Part 2: #122 Quote/clip provenance manifest

### Proposed approach

Add JSON manifest routes alongside existing `.md` export routes, with a
frontend download action.

**New routes:**

- `GET /review/{run_id}/annotations/{annotation_id}/export.json`: single-
  annotation manifest.
- `GET /review/{run_id}/annotations/export.json`: bulk manifest (all
  annotations, or filtered by `?tag=`). Run-level provenance appears once
  in the envelope, not repeated per quote.

**Manifest schema (single):**

```json
{
  "schema_version": 1,
  "kind": "quote_provenance",
  "exported_at": "2026-08-28T15:30:00Z",
  "quote": {
    "lines": [
      {
        "text": "The exact quoted text...",
        "speaker": "Speaker Name",
        "start_seconds": 12.3,
        "end_seconds": 45.6
      }
    ],
    "timing_precision": "word",
    "tags": ["key-quote"],
    "note": "Reporter's note",
    "annotation_id": "hex-uuid",
    "source_text_hash": "sha256:abc...",
    "annotation_updated_at": "2026-08-27T10:00:00Z"
  },
  "clip": {
    "id": "hex-uuid",
    "download_url": "/runs/{run_id}/clips/{clip_id}",
    "filename": "voxint-abcd1234-clip-ef567890.wav",
    "sha256": "abc123...",
    "sample_rate": 16000,
    "channels": 1,
    "start_sample": 196800,
    "end_sample": 729600
  },
  "source": {
    "media_id": "hex-uuid",
    "run_id": "hex-uuid",
    "title": "Interview recording.mp3",
    "media_sha256": "abc123..."
  },
  "pipeline_provenance": {
    "app_version": "0.29.0",
    "stages": {
      "transcribe": {
        "attempt": 1,
        "finished_at": "2026-08-25T09:55:00Z",
        "roles": {
          "asr": {
            "reachable": true,
            "model": "large-v2",
            "revision": null,
            "engine": "ct2-legacy"
          }
        }
      },
      "diarize_embed": {
        "attempt": 1,
        "finished_at": "2026-08-25T09:58:00Z",
        "roles": {
          "diarizer": {"reachable": true, "model": "speaker-diarization-3.1"},
          "embedder": {"reachable": true, "model": "titanet-large-v1"}
        }
      }
    },
    "observed_before_attempt": true
  }
}
```

Nullable fields: `clip` (null when no clip extracted), `source.media_sha256`
(null when backfill has not run), each role's `revision`/`engine` (null when
not recorded or unreachable). Missing hash is explicit null, never
on-the-fly computation.

**Bulk envelope:**

```json
{
  "schema_version": 1,
  "kind": "quote_provenance_bundle",
  "exported_at": "...",
  "source": { "media_id": "...", "run_id": "...", "title": "...", "media_sha256": "..." },
  "pipeline_provenance": { ... },
  "quotes": [ { "quote": {...}, "clip": {...} }, ... ]
}
```

Run-level facts (source, provenance) appear once at envelope level.

**Manifest builder**: new `src/voxint/export/manifest.py`. Pure function
taking primitives and frozen dataclasses, not ORM rows. `exported_at` is
injected by the caller. Uses the existing `model_provenance.stage_models()`
selector for per-stage identity.

**Clip lookup**: query the `audio_artifacts` table for the latest non-
reclaimed `audio_clip` row whose `meta->>'annotation_id'` matches and whose
`idempotency_key` matches the current annotation span's key derivation.
If no match (no clip extracted, or clip reclaimed), `clip` is null.

**Frontend**: add a "Download manifest" action to `AnnotationLayer.tsx`
alongside the existing "Copy quote" and "Extract clip" actions. Simple
`<a href="...export.json" download>` link.

### Alternatives considered

1. **Embed provenance as YAML frontmatter in .md export.** Rejected: mixes
   formats, the issue specifies JSON + human-readable as separate concerns.
2. **ZIP bundle (.md + .json + .wav).** Deferred: nice-to-have but out of
   scope. The three files are independently downloadable.
3. **Singular `speaker` field per quote.** Rejected (codex critique):
   annotations span multiple speakers. Per-line attribution preserves
   fidelity.
4. **Config-derived model versions.** Rejected (codex critique): the
   codebase already records per-attempt model identity in
   `StageRun.metrics` via `pipeline/model_identity.py`. Config would
   describe export-time state, not what produced the transcript.

### Affected files

| File | Change |
|---|---|
| `src/voxint/export/manifest.py` (NEW) | Schema constant, `build_quote_manifest()`, `build_quote_bundle()` |
| `src/voxint/api/routers/legacy_review.py` | Two new route handlers + clip lookup helper |
| `frontend/src/components/AnnotationLayer.tsx` | "Download manifest" action |
| `tests/unit/test_manifest.py` (NEW) | Schema, field types, nullable handling, multi-speaker, legacy provenance |
| `tests/integration/test_export_manifest.py` (NEW) | Route round-trips, tag filtering, stale 409, auth |
| `tests/contracts/test_console2_characterization.py` | Route golden updates (2 new routes) |
| `CHANGELOG.md` | Feature entry |

### Step-by-step implementation

#### PR 1: #159 Speakers activation

1. Flip `console_speakers_enabled` default to `True` in `config.py`.
2. Update config docstring to reflect activation.
3. Update `.env.example` comment.
4. Add default-assertion test (`Settings(_env_file=None)`).
5. Add explicit-false rollback test.
6. Add CHANGELOG entry.
7. Run full suite, lint, mypy.

#### PR 2: #122 Quote/clip provenance manifest + version bump

1. Read `api/model_provenance.py` `stage_models()` API to confirm the
   integration surface.
2. Write `src/voxint/export/manifest.py`: pure builder, typed schema
   constant, `exported_at` injected.
3. Write `tests/unit/test_manifest.py`: schema validation, multi-speaker
   lines, nullable clip, nullable media hash, legacy (no model_identity)
   runs, timing precision, UTC formatting, UUID hex representation.
4. Add single-annotation JSON route to `legacy_review.py`. Reuse existing
   annotation-loading helpers. Clip lookup via `meta->>'annotation_id'`
   query + idempotency key match.
5. Add bulk JSON route. Batch-load annotations, clips, and stage attempts
   in bounded queries under one transaction snapshot. Source and provenance
   at envelope level.
6. Update route goldens (`console2_route_characterization.json`,
   `console2_route_order.json`).
7. Write `tests/integration/test_export_manifest.py`: route round-trips
   with seeded data, tag filtering, stale 409, missing clip, missing hash,
   auth/onboarding gating.
8. Add "Download manifest" action to `AnnotationLayer.tsx`.
9. Atomic version bump to 0.29.0: `pyproject.toml` + `__init__.__version__`
   + 6 compose `VOXINT_IMAGE_TAG` defaults + `.env.example`.
10. CHANGELOG entry, lint, mypy, full suite.

### Testing strategy

- **Unit (manifest builder)**: correct schema version, all fields populated
  from primitives, nullable clip/hash, multi-speaker lines, legacy runs
  with no model_identity (renders explicit "not recorded" rather than
  falling back to config), timing precision propagation, UTC ISO format,
  UUID as hex strings, no NaN/Infinity values.
- **Integration (routes)**: valid JSON response, correct Content-Type
  (`application/json`), Content-Disposition with paired filename, tag
  filtering, 404 on missing annotation, 409 on stale annotation
  (consistent with `.md` behavior), auth gating, onboarding redirect.
- **Contract**: route golden updates (2 new routes in characterization +
  order fixtures). Manifest `schema_version` pinned.
- **Frontend**: download action present in annotation toolbar, href
  correct.
- **Bulk**: bounded query count verified (no N+1), canonical transcript
  ordering preserved, shared `exported_at` across all quotes.

## Rollout, risks, and open questions

1. **Media hash nullability**: `MediaItem.sha256` is nullable. When null,
   the manifest carries `"media_sha256": null`. The operational backfill
   command exists. Decision: export never fails on a missing hash; the null
   value honestly represents "not yet computed."
2. **Clip lifecycle edge cases**: re-anchored annotations produce new clips
   (different idempotency key); the manifest attaches the current clip. A
   reclaimed clip yields null. An annotation with multiple historical clips
   gets the latest non-reclaimed one.
3. **Manifest is not authentication**: the `observed_before_attempt` flag
   carries through from the model identity probe. The manifest documents
   what was observed, not what is cryptographically provable. This is
   honest and matches the existing UI copy.
4. **Activation performance (#159)**: aggregation queries were EXPLAIN-
   checked during development per issue requirements. Re-verify on
   the primary host's seeded dataset before merging.
5. **Version bump ownership**: bump lands in PR 2 to avoid exposing #159
   under the old version number.
6. **Auto-enrollment interaction**: if `feat/auto-enroll` lands first,
   auto-enrolled speakers appear on the overview immediately (by design).
   No manifest impact.

## Review notes (codex critique resolution)

Codex returned 16 findings (7 high, 9 medium). Resolution:

| # | Finding | Resolution |
|---|---|---|
| 1 | Config-derived model versions wrong; `model_provenance.py` exists | **Accepted.** Rewrote to use existing `stage_models()` selector. |
| 2 | `MediaItem.sha256` exists; no on-the-fly hashing | **Accepted.** Read-only, null when missing. |
| 3 | Clip-to-annotation lookup via JSON metadata is indirect | **Accepted.** Added explicit lookup strategy via `meta->>'annotation_id'` + idempotency key match. |
| 4 | Singular `speaker` field is lossy | **Accepted.** Changed to per-line `lines[]` array with speaker per line. |
| 5 | Conflates pipeline/review/export time | **Accepted.** Labeled `pipeline_provenance` separately from `quote.annotation_updated_at` and top-level `exported_at`. |
| 6 | `completed_at` not real; `DIARIZE_EMBED` is one stage | **Accepted.** Used actual `StageRun` field names (`finished_at`) and stage names (`transcribe`, `diarize_embed`). |
| 7 | No frontend work in affected files | **Accepted.** Added `AnnotationLayer.tsx` download action. |
| 8 | Relative URL not durable; missing clip digest | **Accepted.** Added `sha256`, `sample_rate`, `channels`, `start_sample`, `end_sample` to clip record. URL kept as convenience locator. |
| 9 | Schema contract too weak | **Accepted.** Added typed schema tests: UTC format, UUID hex, null vs omission, no NaN. |
| 10 | Bulk N+1 and repetition | **Accepted.** Run-level provenance at envelope level; batch queries. |
| 11 | Route goldens omitted | **Accepted.** Added to affected files list. |
| 12 | ORM rows in builder not pure | **Accepted.** Builder takes primitives and frozen dataclasses. |
| 13 | `.env.example` and docstrings still say dark-shipped | **Accepted.** Added to #159 steps. |
| 14 | Speaker tests override flag; no default-path test | **Accepted.** Added `Settings(_env_file=None)` test. |
| 15 | Activation performance not verified | **Accepted partially.** Added re-verify step; original development included EXPLAIN checks per issue requirements. |
| 16 | Version bump ownership ambiguous | **Accepted.** Bump in PR 2. |

**Deferred from codex suggestions:**

- ZIP bundle: out of scope, can be added later.
- Canonical manifest digest/signature: out of scope for v1. The manifest
  documents observations, not cryptographic proof. Threat model is noted.
- Clip hashing at creation time: clips are content-addressed by sample
  bounds already; sha256 can be computed at export from the served file
  (small WAV clips, bounded size).
