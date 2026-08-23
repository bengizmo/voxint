# Plan: persist and surface faster-whisper's detected language (issue #124)

Status: DRAFT for review. Codex second opinion folded in (see Review notes).
Target branch: `feat/124-detected-language` off `main` @ `81f149c`.

## Goal

Voxint should record which language each run was transcribed in, and let an
operator see and filter by it. Today the transcription service is told to use
English on every request (`TranscribeRequest.language` defaults to `"en"`, the
ASR client never overrides it, and `main.py` passes that straight into
faster-whisper), so faster-whisper never auto-detects and the response's
`language` field always echoes `"en"`. Persisting that value as-is would store a
constant, which is why #124 is not "persist an existing output" but "turn
detection on, then persist and surface it." Ben approved enabling auto-detection
(not an operator language-override setting, which was declined). The audience is
multilingual corpora (immigrant communities, journalists, bilingual
interviews), so correct per-run language is real value, and the honesty of what
we display matters.

## Assumptions and constraints

- Whisper is pinned to large-v2; model outputs are contract, governed by the
  numerics doctrine and the parity gates in `tests/parity/`.
- The service HTTP contract is versioned. `docs/gpu-contracts.md:29-30`:
  additive response fields are allowed within `v1`; renames, removals, and
  **semantic changes require `/v2`**. Line 236 documents `language (default
  "en")` with `null -> auto-detect` already supported by the service.
- The service is self-hosted; we cannot assume Voxint's pipeline is the only
  caller of a given deployment. Changing an omitted-field default silently
  changes behavior for any external caller. Adding a nullable response field
  does not.
- No new runtime dependency; no operator override setting this release.

## Proposed approach

Two changes carry the feature; the rest is plumbing and surfacing.

1. **Activate detection at the client, not the contract default.** Have
   `HttpASRClient.transcribe` send `"language": null` in its request body,
   using the already-documented `null -> auto-detect` feature. Leave
   `TranscribeRequest.language`'s v1 default at `"en"`. This fixes Voxint's
   production behavior without a semantic change to the v1 contract (which would
   require `/v2`) and without changing behavior for any other client that omits
   the field. **[Codex C1 — this replaces the draft's "change the default"
   lever.]**

2. **Persist the detected language (and its detection score) on the run**,
   stamped by the transcribe stage right where `initial_prompt` is stamped,
   after a successful decode, in the same transaction as the transcript write.
   Surface the language as a runs-browser column and search facet; surface the
   score (with honest, qualified copy) on the run detail page.

Adding `language_probability` to the response is additive within `v1` (allowed),
so it needs no version bump of the contract. It is included as nullable
provenance, described as **Whisper's language-detection score** — not a
calibrated confidence and not a code-switch signal. **[Codex C4 corrects the
draft's UX framing.]**

### Alternatives considered

- **Change the v1 request default `"en" -> null`** (original draft lever).
  Rejected: it is a semantic contract change that the contract doc assigns to
  `/v2`, and it changes behavior for any external caller that omits the field.
  The client-explicit-null path achieves the same production result cleanly.
- **A `/v2` endpoint with `null` as the default.** Contract-clean but
  unnecessary for #124; large surface for no added value here.
- **Backfill detected_language for legacy runs.** Rejected: reconstructing it
  from source metadata, filename, or transcript heuristics is not the same as
  Whisper's recorded output; it would fabricate provenance. Legacy rows stay
  NULL. **[Codex, confirmed.]**

## Probability semantics (define before coding) [Codex step-2/C4]

`language_probability` must be defined for every branch, so a forced request
never reports a detection score for a detection that did not happen:

- **Auto-detected, multilingual model** (the production path): the score from
  `detect_language` (VAD path) or `info.language_probability` (raw/legacy) — a
  finite float in `[0, 1]`.
- **Forced language** (caller sent a language): **`None`** — no detection
  occurred, so there is no honest score to report, even if faster-whisper
  exposes a sentinel/`1.0`.
- **Non-multilingual model fallback** (`is_multilingual` false -> forced `"en"`):
  **`None`** — same reason.
- **Language fallback** (`raw.language` falsy -> front layer substitutes
  `"en"`): the probability must **not** survive the fallback (it no longer
  describes the emitted language); force it to `None` when the language is
  substituted. **[Codex step-2.]**
- **No-speech / dummy detection**: whatever `detect_language` returns on the
  dummy-feature path is passed through as the auto path; document that short and
  silent clips produce low-confidence or arbitrary detections.

The VAD branch currently binds `_prob` only inside the auto-detect branch; the
plumbing must initialize the probability to `None` for the forced and
non-multilingual paths so `RawResult.language_probability` is always defined.

## Affected files / components

Service (numerics-sensitive; touch with the parity gate in mind):
- `services/whisper/app/schemas.py` — add `TranscribeResponse.language_probability:
  float | None = None` with a Pydantic `ge=0, le=1` finite constraint; leave
  `TranscribeRequest.language` default `"en"` unchanged.
- `services/whisper/app/backends/ct2.py` — `RawResult` gains
  `language_probability: float | None = None`; VAD path captures the computed
  `_prob` (was discarded) and sets `None` on the forced/non-multilingual
  branches; raw path sets `info.language_probability`.
- `services/whisper/app/backends/ct2_legacy.py` — pass
  `info.language_probability` into `assemble_transcription_output`.
- `services/whisper/app/transcription.py` — `assemble_transcription_output`
  gains a `language_probability` param -> `TranscriptionOutput` field (mind
  dataclass default-ordering); the front layer passes `raw.language_probability`
  and forces it to `None` wherever it substitutes `"en"` for a falsy language.
- `services/whisper/app/main.py` — build the response with
  `language_probability=result.language_probability`.

App / client:
- `src/voxint/clients/asr.py` — send `"language": null` in the request body;
  parse `body.get("language_probability")` with a dedicated validator (do **not**
  reuse `_parse_confidence` — its error text says "segment confidence"; either
  generalize it with a field-name arg or add `_parse_language_probability`).
  Absent key -> `None` (back-compat with older services); present-but-malformed
  -> loud `ProtocolError`. The `language` string is already parsed.
- `src/voxint/clients/base.py` — `TranscriptionResult` gains
  `language_probability: float | None = None`.

Pipeline / DB:
- `src/voxint/pipeline/stages/transcribe.py` (~line 107) — stamp
  `pipeline_run.detected_language = result.language` and
  `pipeline_run.detected_language_probability = result.language_probability`
  next to the `initial_prompt` stamp, after decode, same transaction.
- `src/voxint/db/models.py` (~line 392) — `PipelineRun` gains
  `detected_language: Mapped[str | None] = mapped_column(Text)` and
  `detected_language_probability: Mapped[float | None] = mapped_column(Float)`,
  next to `initial_prompt`, with a docstring mirroring #123's.
- `alembic/versions/0035_pipeline_run_detected_language.py` — copy `0034` as the
  template; add both columns, nullable, no default (NULL = legacy/pre-column or
  a run not yet transcribed). Add a CHECK constraint:
  `detected_language_probability IS NULL OR (detected_language_probability >= 0
  AND detected_language_probability <= 1)`. **[Codex step-5.]** No index on
  `detected_language` (low-cardinality, single-operator) — an explicit
  decision, not an omission.

Runs browser:
- `src/voxint/api/runs_query.py` — `SearchFilters` gains `language`;
  `parse_search_filters` parses it (blank/absent -> None); `runs_url` preserves
  it; `RunListItem` gains `language` (+ score if shown); `list_runs` selects
  `PipelineRun.detected_language`, adds a `WHERE detected_language == filter`
  predicate, and populates the row. Add `searchable_languages(session)` **here,
  in runs_query.py, not in the speaker roster module** **[Codex step-6]** —
  `SELECT DISTINCT detected_language WHERE detected_language IS NOT NULL`,
  ordered for display.
- `src/voxint/api/app.py` (`GET /runs`, ~line 3577) — add a `language` query
  param, thread it into `parse_search_filters`, pass `facet_languages` into the
  template context.
- `src/voxint/api/templates/runs.html` — add a Language `<select>` facet
  (modeled on Speaker), a Language column in the header + rows (NULL renders an
  em dash / honest unknown), and bump the snippet row `colspan` 5 -> 6.
- `src/voxint/api/templates/run_detail.html` — surface detected language and,
  when present, the detection score with a qualified label ("Whisper
  language-detection score"), where there is room for explanatory context.
- New: a dependency-free pinned code->name map for large-v2's supported
  languages so the UI shows "Spanish (es)" with the raw code as fallback for
  unknown values; facet ordered by display name. **[Codex step-7 — replaces the
  draft's code-only lean; better for the non-technical audience.]** Label the
  facet "Detected language" / "Whisper language", not "corpus language."

Docs / changelog:
- `docs/gpu-contracts.md` — document the additive `language_probability`
  response field; note that the v1 `language` request default stays `"en"` and
  auto-detection is opted into per-request with `null` (which Voxint's client
  now does). Document the probability's limitations (per-run single language,
  short/silent-clip uncertainty).
- `CHANGELOG.md` `[Unreleased]` — add the entry (house style).
- `README.md` — only if it enumerates the runs-browser facets (avoid doc
  churn). **[Codex step-9.]**
- **No version bump on this branch.** Per `docs/release-process.md`, feature
  branches update `[Unreleased]`; the atomic pyproject/`__init__`/six-compose/
  `.env.example` bump happens in the release session. **[Codex step-10 —
  resolves the draft's open version question.]**

## Step-by-step implementation

1. **Contract lever + client (C1).** Client sends `language: null`; parse the
   new response field. Add a route/contract test that captures the arguments
   reaching the transcriber and proves an omitted API `language` still resolves
   to the v1 default while Voxint's client sends `null`. Keep the existing
   explicit-null schema test.
2. **Probability semantics + service plumbing.** Implement the per-branch
   semantics above across `ct2`, `ct2_legacy`, `transcription` front layer,
   `main`, `schemas`. Update `tests/contracts/fixtures/whisper_transcribe.json`
   (add `language_probability` to the response; the request stays `"en"` as the
   contract example, plus a second fixture or test exercising `null`).
3. **Client/base threading + strict validation** of the probability.
4. **DB migration 0035 + model columns + CHECK constraint.**
5. **Transcribe-stage stamping** (both columns), after decode, same txn.
6. **Runs browser**: filter, facet helper, column, run-detail surface,
   code->name map.
7. **Docs + CHANGELOG.**
8. **Gates** (see below), including the new numerics gate.

## Testing strategy

CPU-lane (CI, must pass before land):
- `tests/contracts/test_schemas.py` — keep the explicit-null test; add the
  additive `language_probability` response field (finite `[0,1]`, rejects NaN /
  inf / <0 / >1 / non-number). Add a test proving the omitted-field default is
  still `"en"` (`{"path": "a.wav"}`). **[Codex step-1.]**
- `tests/contracts/test_client_compat.py` — the ASR client parses `language`
  and `language_probability`; back-compat when the response omits the field
  (older service); loud `ProtocolError` on malformed. Cover missing/null/int/
  float/bool/string/NaN/inf/negative/>1. **[Codex step-3.]**
- `tests/contracts/test_routes.py` — route response includes
  `language_probability`; captures transcriber args to prove the client sends
  `null`. **[Codex step-1/8.]**
- `tests/contracts/test_service_logic.py` — transcribe stage stamps both
  columns via a parametrized `FakeASR` (`tests/fakes.py:40`); rerun replaces the
  prior language; a failed rerun does not erase a committed value after
  rollback. **[Codex step-4.]**
- Unit coverage of the service internals: ct2 VAD probability capture, ct2 raw
  probability, ct2-legacy probability, front-layer propagation and the
  language-fallback-forces-None rule. **[Codex step-8.]**
- `tests/integration/test_migration_0035.py` — both columns, types,
  nullability, value round-trip, downgrade, ORM/reflected-schema equality, and
  the CHECK constraint rejecting an out-of-range probability. **[Codex step-5/8;
  this test was missing from the draft inventory.]**
- Runs browser (`tests/unit/test_runs_search.py`,
  `tests/integration/test_runs_search_api.py`): language facet filters
  correctly; NULL-language rows (queued/failed/legacy/no-speech) appear
  unfiltered, are excluded by a specific-language filter, render an em dash;
  language survives next-page cursor URLs and the archive toggle and composes
  with status/review/speaker/source/date/transcript filters; keyset pagination
  with an active language filter; `searchable_languages` distinctness + ordering;
  HTML assertions for the header, selected option, escaped labels, NULL render,
  probability copy, and snippet `colspan=6`. **[Codex step-6/8.]**
- Coverage >= 85% on changed files (not a substitute for the numerics gate).

Numerics gate (maintainer hardware; CI has no GPU) **[Codex C2/C3 — the draft's
biggest gap]**: with the client now sending `null`, production runs the ct2
shared-VAD `detect_language` branch and faster-whisper raw/legacy auto-detection
— paths the existing parity fixtures never exercise (they force `"en"`).
Preserve the frozen forced-`"en"` oracle unchanged and add a separate
`language=None` gate:
- **Tier 1 (reuses existing English references, cheaper):** run the existing
  English parity fixtures with `language=None` across ct2 and ct2-legacy, both
  VAD modes; assert (a) detected language is `"en"` and (b) the decode matches
  the frozen forced-`"en"` oracle within the existing transcript/timestamp
  tolerances. This is the measured evidence for the narrowed equivalence claim:
  *when auto-detection selects en, decoding normally matches forced-en* (not an
  unconditional byte-identity claim). **[Codex C3.]**
- **Tier 2 (new references, maintainer CUDA work, larger):** add non-English (>=
  2 languages), short/ambiguous, and silence/no-speech fixtures with fresh CUDA
  references; assert backend agreement on detected language and a defined
  probability tolerance. Stronger coverage of the new production path.

## Rollout / risks / open questions

Resolved by codex:
- Activation lever -> client sends `null`, v1 default preserved (C1).
- Legacy rows -> NULL, no backfill.
- Version bump -> not on this branch; CHANGELOG only.
- Probability -> include, nullable, strict validation, honest copy, run-detail
  surface (not a bare table badge).

Decisions (Ben, 2026-08-22):
1. **Parity gate scope: Tier 1 now, Tier 2 as a fast-follow issue.** Tier 1
   (reuse English references, prove en-equivalence with `language=None`) is a
   land gate for #124. File Tier 2 (non-English + ambiguous + silence CUDA
   references, maintainer hardware) as a separate follow-up issue; note it in
   the #124 close-out.
2. **Include `language_probability`.** Nullable, strict-validated, surfaced on
   run detail (not the main table) as "Whisper language-detection score." The
   full service-internals plumbing (steps 2-3) is in scope.
3. **Facet display: name + code via a pinned map.** Dependency-free
   code->name map for large-v2's languages; show "Spanish (es)"; raw code as
   fallback for unknown values; order by display name.

Still open (not this release):
- **Low-confidence guardrail** (fall back to a configured default below some
  detection score): NO for v1 — that is the override feature Ben declined;
  recorded so it is a decision, not an omission.

Residual risks:
- Auto-detect can misfire on short/noisy/musical/accented clips where forced-en
  "worked." No guardrail this release (see Q3); the run-detail score gives the
  operator a signal.
- The `detect_language` extra forward pass on the VAD BatchedInferencePipeline
  path adds a one-window encode per run (small) and must not perturb the decode
  (temperature 0 greedy, separate pass; validated by the Tier-1 gate).
- `detected_language` becomes a slightly narrow name if a future forced-language
  option arrives; document that this release always requests auto-detection, and
  revisit (persist detection mode / requested language) when overrides land.
  **[Codex step-4.]**

## Review notes (codex second opinion, zen clink planner)

Codex (GPT-5-family via clink, 197s, returned COMPLETE) inspected the real
files and flagged the draft as "directionally sound but not ready to execute."
Resolutions:

- **C1 (accepted, load-bearing):** do not change the v1 request default
  `"en" -> null`; that is a semantic contract change the doc assigns to `/v2`
  and would alter behavior for any external caller. Instead the client sends
  `null` explicitly. Verified against `docs/gpu-contracts.md:29-30,236`. The
  whole approach was rebuilt around this.
- **C2 (accepted):** existing parity fixtures force `"en"` and never exercise
  the auto-detect paths production will now use; added a separate `language=None`
  numerics gate (Tier 1 reusing en references + Tier 2 new fixtures), preserving
  the frozen oracle.
- **C3 (accepted):** narrowed the "byte-identical" claim to "when detection
  selects en, decode normally matches forced-en, subject to a measured gate."
- **C4 (accepted):** `language_probability` is a detection score, not calibrated
  confidence and not a code-switch signal; reworded all UX/copy and moved it to
  run detail with a qualified label.
- **Per-step accepts:** probability semantics defined for every branch (forced/
  non-multilingual/fallback -> None); dedicated probability validator (not the
  segment-confidence one); DB CHECK constraint; `searchable_languages` lives in
  `runs_query.py`, not the roster; code->name map for the non-technical
  audience; migration 0035 test added; rerun-replace + failed-rerun-rollback
  tests; NULL-row + cursor-preservation + filter-composition browser tests;
  version bump kept off this branch.
- **Deferred to Ben (open questions above):** parity Tier 2 scope, whether to
  include probability at all, low-confidence guardrail.
- **Not adopted:** none outright rejected; the `/v2` alternative was noted and
  set aside as unnecessary for #124.
