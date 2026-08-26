# Lever 5 first slice: relocate the pure export-match-evidence tests to the unit lane

_Plan written 2026-08-26 on the maintainer host. Codex (zen clink, planner role)
reviewed the draft; see Review notes at the end._

## Goal

Move the pure (database-free) tests out of
`tests/integration/test_export_match_evidence_driver.py` into a new
`tests/unit/` module, with every assertion preserved verbatim, leaving the six
`Session`-taking tests in the integration lane. This is the **first slice** of
"lever 5" from the dev-loop speedup thread: shift integration tests that are
really unit tests down to the fast lane. The primary deliverable is a **proven,
reviewed, repeatable pattern** (and a short written rubric for future slices),
not a large one-shot speedup. The raw wall-clock win from a single module is
marginal and will be reported honestly.

## Why this module, and only this module

A survey classified every non-migration integration module. All the
pure-logic-*looking* modules (`test_config_resolution_freeze`,
`test_enhance_match_corrections`, `test_adjudication_resolver`,
`test_attributed_intervals`, `test_segment_scope`, `test_app_settings_repository`,
and others) genuinely need a live `Session` (DB precedence walks, persistence
round-trips, SQL `CHECK`/`IntegrityError` behaviour, advisory-lock races) or the
FastAPI `TestClient`. They stay.

`test_export_match_evidence_driver.py` is the one module that splits on a clean
seam: 27 of its 33 tests exercise pure functions of the `tools/export_match_evidence.py`
driver (manifest parsing, deterministic serialization, atomic writes, git
helpers, and CLI error paths that fail closed before any DB access), and 6 take a
`Session`.

## Assumptions and constraints

- **Hard rule (lever 5):** never weaken an assertion to move a test. A moved
  test asserts identical behaviour at the unit level or it stays put. Every
  assertion in this plan moves byte-for-byte.
- The one deliberate, *additive* deviation from a pure copy: a small no-DB
  tripwire fixture on the three `main()` error-path tests (see Step 4). It adds a
  guard, changes no existing assertion, and makes the unit classification
  enforceable rather than incidental.
- `tools/export_match_evidence.py` imports `Session`, `build_engine`, and
  `session_scope` at module top but opens **no** DB connection at import time;
  the pure functions never connect. Importing the module in a unit test is safe.
  (This is a documented assumption in the rubric, backstopped by the tripwire.)
- Both `tests/unit/` and `tests/integration/` are packages (`__init__.py`
  present); pytest runs in default prepend import mode with `testpaths=["tests"]`.
  A same-basename file in `tests/unit/` is therefore a distinct module from the
  integration one, with no "import file mismatch" collision.
- `mypy` is configured against the `voxint` package only. It does **not**
  type-check `tools/` or `tests/`; it will not catch mistakes in these test
  files. Ruff (F401 unused, F821 undefined) plus pytest collection and execution
  are the real guards.
- The required CI `coverage` job runs the full suite (unit + integration +
  contracts) with `--cov -n 8`, 85% floor. Relocating tests between lanes is
  **coverage-neutral**: the same production code is still exercised.
- Numerics parity, contract goldens, and public contracts are untouched. What
  invalidates this plan: discovering that any of the 27 "pure" tests in fact
  depends on the integration `session_factory` autouse setup, or that
  `parse_manifest`/`_dumps`/etc. reach the DB. Both are contradicted by reading
  the code, and the DB-env-unset run in Step 8 proves it empirically.

## The exact split (verified by reading the file)

**Move to `tests/unit/test_export_match_evidence_driver.py` (27 tests + 1 helper):**

- Manifest parsing (17): `test_parse_name_accuracy_only`,
  `test_parse_agreement_curated_and_negative`,
  `test_parse_rejects_bad_schema_version`, `test_parse_requires_embedding_space`,
  `test_parse_requires_at_least_one_lane`, `test_parse_rejects_bad_truth_anchoring`,
  `test_parse_rejects_empty_and_duplicate_run_ids`, `test_parse_rejects_bad_uuid`,
  `test_parse_rejects_bad_kind`, `test_parse_curated_requires_host`,
  `test_parse_negative_control_rejects_host`, `test_parse_rejects_empty_agreement_runs`,
  `test_parse_rejects_duplicate_agreement_run`, `test_parse_rejects_non_object`,
  `test_parse_rejects_non_string_uuid`, `test_parse_rejects_non_list_run_ids`,
  `test_parse_rejects_non_string_truth_anchoring`
- Serialization (4): `test_dumps_matches_harness_serialization`,
  `test_dumps_rejects_non_finite`, `test_dump_jsonl_is_newline_terminated`,
  `test_write_atomic_creates_dirs_and_content`
- `write_artifacts` (1): `test_write_artifacts_writes_all`
- git helpers (2): `test_git_sha_none_outside_repo`, `test_git_sha_present_in_repo`
- CLI error paths (3): `test_main_bad_manifest_returns_2`,
  `test_main_missing_manifest_returns_2`, `test_main_refuses_dirty_tree`
- Helper: `_na_block`

**Stay in `tests/integration/test_export_match_evidence_driver.py` (6 tests):**

- `test_build_name_accuracy_only`, `test_build_agreement_emits_enrollment_and_slots`,
  `test_build_both_lanes_union_dedupes_snapshot`, `test_build_is_deterministic`,
  `test_build_propagates_export_error` (all take `session`)
- `test_main_end_to_end_and_score_round_trip` (takes `session`; round-trips
  through the real `voxint score` commands)
- Helpers that stay: the `session` fixture, `_agreement_manifest`, `_StubEngine`,
  `_passthrough_session_scope`.

**Unit-module imports (trimmed):**

```python
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from tools import export_match_evidence as drv
from tools.export_match_evidence import ManifestError, main, parse_manifest, write_artifacts
from voxint.harness.score_cli import _dumps as harness_dumps
from voxint.harness_export import TruthAnchoring

SPACE = "titanet-large-v1"
ANCHOR = TruthAnchoring.INDEPENDENT.value
```

**Imports/symbols to remove from the integration file** (now unused after the
move): `ManifestError`, `parse_manifest`, `write_artifacts` (from the
`tools.export_match_evidence` import list), `harness_dumps`
(`from voxint.harness.score_cli import _dumps`), `Any` (from `typing`), and the
module-level `ANCHOR = TruthAnchoring.INDEPENDENT.value` assignment.
**Keep** in the integration file: `Session`, `sessionmaker`, `contextlib`,
`json`, `uuid`, `Path`, `pytest`, `drv`, `main`, `build_artifacts`, the
`Manifest`/`NameAccuracyLane`/`AgreementLane`/`AgreementRun` dataclasses,
`SPACE`/`E0`/`_grounded_run`/`add_speaker`/`add_turn`/`make_run`/`run_matcher`
(from `tests.integration.test_harness_export`), `voxint.cli.main as voxint_main`,
`Settings`, `ExportError`, `TruthAnchoring`. Ruff F401 is the backstop for any
miss.

## Affected files / components

- `tests/unit/test_export_match_evidence_driver.py` — **new.** The 27 pure tests
  + `_na_block` + the no-DB tripwire fixture, trimmed imports, local
  `SPACE`/`ANCHOR`, and a scope docstring describing the pure-driver surface.
- `tests/integration/test_export_match_evidence_driver.py` — **shrinks to 6
  tests.** Remove the moved functions + `_na_block`, prune the now-unused
  imports and the `ANCHOR` assignment, and update its module docstring to say it
  now covers only `build_artifacts` and the end-to-end round trip.
- `docs/testing.md` — **add a short lever-5 relocation rubric** (the repeatable
  checklist). Documents the pattern so later slices are mechanical, and keeps
  docs in step with the behaviour change per repo policy.
- `CHANGELOG.md` — an `[Unreleased]` note under a suitable heading (internal /
  test-infra; no user-facing behaviour change).

## Step-by-step implementation

1. **Snapshot the baseline.** `git rev-parse --short HEAD` (expect `38d2d5f` or
   later). Record the original 33 test names:
   `uv run pytest tests/integration/test_export_match_evidence_driver.py --collect-only -q > /tmp/…/names_before.txt`.
2. **Create the unit module** with the trimmed import block above and a scope
   docstring. Paste the 27 pure test functions and `_na_block` **verbatim** —
   assertions unchanged.
3. **Add the no-DB tripwire fixture** (local to the unit module), applied only to
   the three `main()` error-path tests, e.g.:
   ```python
   @pytest.fixture()
   def no_db(monkeypatch: pytest.MonkeyPatch) -> None:
       def _boom(*_a: object, **_k: object) -> object:
           raise AssertionError("build_engine must not be reached on the error path")
       monkeypatch.setattr(drv, "build_engine", _boom)
   ```
   Give each of `test_main_bad_manifest_returns_2`,
   `test_main_missing_manifest_returns_2`, `test_main_refuses_dirty_tree` the
   `no_db` parameter. This asserts the DB is never constructed on these paths;
   it does not alter any existing assertion.
4. **Shrink the integration module:** delete the 27 functions + `_na_block`,
   prune the unused imports and the `ANCHOR` assignment, and update the module
   docstring.
5. **Set-equality proof of the move.** Collect names from both files:
   `--collect-only -q` on the new unit file and the shrunk integration file.
   Assert the two name sets are disjoint and their union equals
   `names_before.txt` (33 = 27 + 6). Eyeball a function-body diff of a couple of
   moved tests to confirm the paste was byte-identical.
6. **Collision check.** `uv run pytest tests/unit/test_export_match_evidence_driver.py tests/integration/test_export_match_evidence_driver.py --collect-only -q` in one invocation — proves the shared basename resolves to two distinct modules with no import-file-mismatch error.

## Testing strategy (CI-shaped, run each lane as CI runs it)

- **Ruff** (primary import guard): `uv run ruff check .` — F401 catches any
  leftover unused import in the integration file; F821 catches an undefined name
  in the unit file.
- **Unit lane with the DB env explicitly unset** (proves DB-independence — the
  load-bearing check): run `tests/unit tests/contracts` with
  `VOXINT_TEST_DATABASE_URL` and `DATABASE_URL` unset. The 27 relocated tests
  must pass with no database reachable. The three `main()` tests additionally
  exercise the tripwire.
- **Integration lane** with the disposable DB URL set
  (`VOXINT_TEST_DATABASE_URL=…voxint_test`): the 6 retained tests must pass
  against real Postgres.
- **Parity** serially (`VOXINT_PARITY_REQUIRED=1` semantics as in CI) — expected
  untouched, run to confirm no incidental breakage.
- **Full coverage invocation** as the required job runs it
  (full suite `--cov -n 8`), confirm the 85% floor still holds
  (coverage-neutral, so it must).
- **Type-check the two test files explicitly** as a courtesy
  (`uv run mypy tests/unit/test_export_match_evidence_driver.py tests/integration/test_export_match_evidence_driver.py`),
  acknowledging the required `mypy src` gate does not cover them.

## Measurement (report medians, treat module timing as diagnostic only)

The integration autouse session fixture builds a database per participating
xdist worker, so timing the single module under `-n 8` is misleading (worker
count and setup overhead shift). Report:

- **Full integration lane** and **unit+contracts lane** wall-clock over several
  runs each, medians, at fixed worker counts, before vs after.
- The **`lint-test` critical-path delta** (the fast required job) if measurable.
- Single-module timing only as a diagnostic footnote.

Be explicit in the PR/commit: a single slice moves ~27 of ~1707 integration
tests (~1.6%); the headline win is directional and the value is the proven,
documented pattern. Do not project cumulative savings from one slice.

## Rollout / risks / open questions

- **Shared basename (accepted, low risk).** Both dirs are packages and the repo
  already has cross-lane duplicate basenames; the Step 6 combined collect-only is
  the guard. A distinct basename (e.g. `test_export_match_evidence_manifest.py`)
  is marginally more robust against a future removal of `__init__.py` or an
  import-mode change, but is not needed today. **Open question for the reviewer:
  keep the shared basename, or rename for future-proofing?**
- **`SPACE` hardcoded locally** rather than imported from a canonical source. The
  integration module also hardcodes it (via `test_harness_export`). If a shared
  `titanet-large-v1` constant exists in `src/voxint`, importing it would be DRYer;
  leaning local for a self-contained unit test. Minor.
- **Fixture inheritance change (expected, desirable).** The moved tests stop
  inheriting integration's autouse DB setup and begin inheriting the unit lane's
  dotenv-isolation fixture. None of the 27 should depend on either; the
  DB-env-unset run proves it.
- **Deferred:** the survey also found pure-but-unit-uncovered functions
  (`winning_attribution`, `display_name`, `segment_speaker`,
  `parse_transcript_text`) exercised today only via DB paths. Filling those is
  *authoring new tests*, not relocating existing ones; out of scope for this
  slice to keep it a clean pattern-proving move.

## Review classification (per repo "Reviews" policy)

- **Code-review depth: multi-model.** This changes a test's *lane*, not a gate's
  *strength*; it touches test infrastructure but no production code, numerics,
  contracts, security, or migrations. It is more than a trivial rename (new file,
  a new fixture, import pruning across two files), so it clears the
  single-model bar but does not rise to the full high-risk panel. Multi-model
  review, no auto-fix beyond the panel's findings.
- **Browser acceptance lane: skip.** No island code, template, route, asset,
  header, or console-consumed backend response changes. Test-only.
- Record both classifications, the gates run, and each applied fix / deliberate
  skip in the commit message or PR per policy.

## Review notes (codex critique and resolutions)

Codex (zen clink, planner role) reviewed the draft. Resolutions:

- **Add a no-DB tripwire to the three `main()` error-path tests** — ACCEPTED.
  Folded into Step 3 as a local `no_db` fixture; their unit classification is now
  executable, no assertion changed.
- **Collection counts don't prove a byte-identical move** — ACCEPTED. Step 5 now
  does set-equality + disjointness over the recorded 33 names plus a spot body
  diff.
- **`mypy src` won't catch mistakes in these test files** (mypy scoped to the
  `voxint` package) — ACCEPTED as a correction. Ruff + collection/execution are
  the guards; the two files are mypy-checked explicitly as a courtesy, not via
  the `mypy src` gate.
- **Run CI-shaped lanes separately, unit lane with DB env unset, include parity +
  full coverage** — ACCEPTED. Rewrote the testing strategy accordingly; the
  DB-env-unset unit run is the load-bearing proof of DB-independence.
- **Measurement: full-lane medians at fixed worker counts, module timing
  diagnostic only** — ACCEPTED. The autouse per-worker DB makes single-module
  `-n 8` timing misleading; rewrote the measurement section.
- **`ANCHOR` also becomes unused in the integration file; update stale
  docstrings** — ACCEPTED. Both were missing from the draft's prune list and are
  now explicit (Step 4, affected-files).
- **Keep this candidate; a marginal win is not a reason to pick a less-clean
  (DB-dependent) target** — ACCEPTED. Honesty about the small win retained.
- **Shared basename is low-risk; a distinct basename is more future-proof but not
  required; add a combined collect-only check** — folded in as Step 6 and left as
  an explicit open question for the reviewer.
- Codex confirmed answers to the four posed questions: 27-in-one-slice is fine
  (cohesive module); no ordering/xdist dependency found; the honest framing is
  correct; pruning is safe once `ANCHOR` and the docstring are handled and the six
  retained tests are executed against Postgres.
