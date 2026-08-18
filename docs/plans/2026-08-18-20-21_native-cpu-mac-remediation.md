# Plan — Remediate the native/CPU/macOS audit findings (all 16), sequenced

> Durable home on execution: copy this to
> `docs/plans/2026-08-18-20-21_native-cpu-mac-remediation.md` (project rule — plans live in
> `docs/plans/`, kebab-case). Written in plan mode; this scratch copy is the working draft.
> Repo @ `d8e85e7` (`main == origin/main`, CI green, v0.17.0).

## Context

The 2026-08-18 audit (`docs/reports/native-cpu-mac-audit-2026-08-18.md`) reviewed Voxint's
three non-NVIDIA install paths (CPU-tier Docker, Apple-Silicon metal, docker-free native macOS
preview) for the single-operator, often-non-technical, self-hosted audience. It found **no
Critical/High** but 8 Medium footguns + 8 Low polish items: two onboarding trip-hazards
(`setup.md` omits a `uv` prereq and understates CPU memory → silent OOM), a native data-safety
inversion (plain `restore` has no safety backup while `--fresh` does), the native preview's
isolation from the onboarding funnel, and two `status`/`doctor` honesty gaps. This plan
remediates **all 16**, sequenced by value/risk, each phase independently shippable. A codex
planner review (folded in below) corrected four substantive flaws in the first draft.

## Goal

Close every audit finding with proportionate, verified fixes — docs first (zero-risk, highest
value), then the data-safety fix, then diagnostic/robustness fixes, then the one broad-blast
change (`pipefail`) in isolation — landing as a single `0.18.0` release.

## Assumptions & constraints

- **Audience lens governs severity:** single-operator, non-technical, self-hosted Mac/CPU;
  correctness + onboarding matter, enterprise scale/security do not.
- **Numerics doctrine untouched:** nothing here changes inference; no parity gate is affected.
- **Bash 3.2 compatibility is mandatory** (installer + native launcher; macOS ships 3.2.57,
  present on this box). Library seams: `VOXINT_NATIVE_LIB=1`, `VOXINT_INSTALL_LIB=1`.
- **Pin-parity is contract-tested:** `tests/contracts/test_service_logic.py` globs
  `compose*.yaml` and requires exactly one `VOXINT_IMAGE_TAG:-X.Y.Z` per image-bearing file —
  **six today** (`compose.yaml`, `.cpu`, `.gpu`, `.rocm`, `.llm`, `.ytdlp-egress`; `.metal`/build
  overlays carry none). This test — not the stale "all four" note in `CLAUDE.md` — is the source
  of truth for the atomic version bump. (Bonus: `CLAUDE.md`'s "all four compose" line is stale;
  fix it in the same bump commit.)
- **Invalidators:** if the `pipefail` pipeline inventory (Phase 3) surfaces a real latent bug, it
  becomes its own fix, not a reason to skip. If the Bash percent-encoder (Phase 2 #7) proves
  fragile under 3.2, fall back to rejecting reserved chars at validation (see open question O1).

## Proposed approach (recommended; rationale inline)

Four phases, each its own feature branch cut **from freshly-merged `main`** (branches all cut
from today's `main` will not all FF-merge sequentially — codex), FF-merged when green.
Docs ship independently of code so onboarding fixes land immediately at zero regression risk.
`pipefail` is isolated last because it is the broadest semantic change and wants a clean baseline.

**Alternatives considered & rejected:**
- *One big PR for everything* — rejected: mixes zero-risk docs with the riskiest shell change;
  hard to review and to bisect.
- *Add a native `voxint` CLI wrapper now* (to make #14's terminal recipes work natively) —
  deferred, not rejected: it's a real feature (env-aware entrypoint exporting the launcher's
  `DATABASE_URL`/`MEDIA_ROOT`), out of scope for a doc-accuracy remediation. Phase 0 instead makes
  the docs *honest* (label the recipes Docker-only, point native users to the browser). Wrapper is
  logged as a future enhancement (open question O2).

---

### Phase 0 — Docs-only batch (no behaviour change; ships first)

Findings: #1, #2, #4, #5, #8, #9, #13(doc part), #14, #15, #16. Branch `docs/native-cpu-mac-audit`.

- **`docs/setup.md`:**
  - #1 CPU memory: add a clause mirroring `operations.md:90-96` — "8 GB is the tight floor
    (services OOM-kill with an opaque exit below it); 16 GB is comfortable" (near :103).
  - #2 metal `uv`/Homebrew prereq above the metal command block (:142) — the script hard-fails at
    `voxint-metal.sh:544` without `uv`. **Also update `docs/operations.md` metal section
    (:157-162)** — the audit named both docs; Phase 0 must not fix only one (codex).
  - #3(doc)/M3: add `voxint-metal.sh status` (with the expected device line) to the metal block.
  - #4 metal-vs-native differentiation clause at :150 ("Most Mac users want **metal**; choose the
    native preview only if you can't/won't run Docker Desktop — it's a hands-on technical preview").
  - #8 Docker-Desktop-only caveat in the metal section (Colima/OrbStack/plain dockerd break the
    host-gateway→loopback mapping; from `compose.metal.yaml:13-19`).
  - #9 decision aid at §3 (:87). **Separate the two axes** (codex): a *deployment path* row
    (Docker vs docker-free native) and a *compute tier* row (CPU/NVIDIA/AMD/metal) — do **not**
    add "native" as a fifth compute tier (native is a core-stack deployment mode that delegates to
    the metal model tier; conflating them misleads).
  - **Page-level framing fix** (codex): the top of `setup.md` implies Docker is the *only* hard
    requirement and the guided-install paragraph says nothing beyond Docker is needed — false for
    metal (`uv`) and native. Adjust that framing so the prereqs are honest.
  - #15 CPU `pull` parenthetical — **include it** (goal is to dispose of every finding, so no
    "optional"): add "(`up -d` pulls the images on first run)" to the CPU block.
- **`docs/native-macos-preview.md`:**
  - #5 add a "Next: first-run walkthrough" block after the install/run section (~:58) linking
    `onboarding.md` + `how-to/README.md` — the native path currently dead-ends.
  - #16 wording: `doctor` reports *port reachability*, not "collisions" (:32) — reword; smoke gate
    also asserts `/setup` 200 (:257-259) — add. **Leave the CR/LF/"newline" wording as-is**
    (codex): `_reject_control_chars` rejects **only CR/LF**, not all control chars, so "newline" is
    the *accurate* word — do **not** change it to "control characters" (my draft nit was wrong).
- **`docs/how-to/add-media-and-manage-runs.md:136-140` + `docs/onboarding.md:114`:** #14 — these
  `docker compose exec api voxint …` recipes **do not** work on the native install (venv bin not on
  PATH; no launcher env exported). **Do not document a bare `voxint` as the native equivalent**
  (codex — it's wrong). Instead label the recipes **Docker-only** and point native users to the
  browser equivalents.
- **`docs/testing.md`:** #13 document `upgrade-db --rehearse` as a maintainer self-test (keep it
  out of the operator preview guide). *(The built-in `--help` line for it is a 1-line code touch —
  see Phase 2.)*
- **`CHANGELOG.md`** `[Unreleased]` (Docs section).
- **Gate:** `bash -n` n/a (no scripts); a **pinned relative-link + anchor check** (no markdown
  linter exists in the repo — codex; use a small repo command, e.g. a `python`/`grep` script that
  resolves every relative link + `#anchor` in the touched docs) + human read. No pytest needed.

### Phase 1 — Native data-safety: plain-restore safety backup (#3)

Branch `fix/native-plain-restore-safety-backup`. `scripts/native/voxint-native.sh`.

- Extract the safety-backup logic from `fresh_restore` (:1401-1421) into a shared helper
  `pre_restore_safety_backup <stamp-label>`: `pg_dump -Fc --exclude-extension=vector` →
  `<name>.partial`, abort-on-failure (nothing destroyed), `mv`, `chmod 600`, print
  `SAFETY_BACKUP <path>`. **Make the filename collision-safe** (codex): second-resolution stamps
  can collide and `mv` would overwrite — add a uniquifier (e.g. `.$$`/counter or `mktemp`-style).
  Fix `fresh_restore` to use the same collision-safe helper (same latent bug).
- Call it in `plain_restore` **immediately after `restore_preflight`, before any mutation**
  (codex — before the `pg_restore --clean`, and before any `CREATE EXTENSION`), naming
  `pre-restore-<stamp>.dump`.
- **Tests** (`tests/unit/test_native_launcher.py`, `run_lib` + stub bins; mirror
  `test_plain_restore_uses_safe_clean_flags` :1206):
  - `test_plain_restore_takes_safety_backup` — assert `pg_dump` runs **before** `pg_restore` using
    a **single shared event log** the stubs append to (separate per-bin logs can't prove ordering —
    codex); assert `SAFETY_BACKUP` printed and `chmod 600`.
  - `test_plain_restore_aborts_if_safety_backup_fails` — stub `pg_dump` non-zero; assert **no**
    `pg_restore`, **no** `CREATE EXTENSION`, **no** finalized dump, **no leaked `.partial`**.
- **Doc:** `native-macos-preview.md` backup/restore section — both restore forms now take a safety
  backup. `CHANGELOG` Fixed.
- **Acceptance (macOS):** extend the `voxint-native-e2e` `--with-restore` lane to also run **plain**
  `restore` (today it only exercises `--fresh` — codex) and assert `SAFETY_BACKUP` creation +
  0600 + restart success + data read-back.

### Phase 2 — Native honesty/robustness: #6, #7, #10, #12

Branch `fix/native-status-dsn-doctor`. (#12 moved here from Phase 3 to keep `pipefail` isolated.)

- **#6 `status` worker/beat liveness** (`cmd_status` :1106-1121). Add a best-effort helper that
  reads `launchctl print` output for `state = …` / `last exit code = …`. **Scope to worker/beat
  only** (drop the datastore expansion — they already have reachability signals; parsing them adds
  exposure for no value — codex). **`launchctl print` is explicitly non-API output** (codex): parse
  conservatively — states `running` / `restarting (last exit N)` / `state unknown`; a **stale
  non-zero last-exit on a currently-running job must not read unhealthy**, and a **missing/unparsable
  field must not fall back to the misleading `[supervised]`** — surface `state unknown` instead.
  - Tests: stub `launchctl` emitting unloaded, running-with-stale-exit, crash-loop, signal-kill,
    and missing-field blocks. **Plus a real-macOS fault-injection gate** (a genuinely crash-looping
    launch agent) in the E2E lane — Linux stub tests cannot validate the real output shape (codex).
- **#7 DSN reserved-char handling.** Two composers, both must change (codex):
  - `native_database_url` (:188-191, Bash): percent-encode **only the password** with a
    correct RFC-3986 encoder (byte-loop under `LC_ALL=C`; encode `%` first; cover space/`+`/`#`/`@`
    /`:`/`/` and all non-unreserved bytes — not just the 5 cited chars).
  - `tools/native_e2e_lifecycle.py` `NativeConfig.database_url` (:187-189, Python): use
    `urllib.parse.quote(self.db_password, safe="")`. `create_engine` at :528 then gets a valid URL.
  - Keep `validate_native_inputs` newline-reject as-is (the password is deliberately quote-safe for
    psql — comment :252-257). Tests: assert with **SQLAlchemy `make_url`** that host/port/db/**and
    the original password round-trip** (string-equality is insufficient — codex); add a
    **reserved-password `--no-models` smoke** so alembic + service startup actually consume the
    encoded URL. (See open question O1 for the encode-vs-reject fallback.)
- **#10 `doctor` foreign-postmaster detection** (`cmd_doctor`). When the managed cluster exists and
  Postgres is reachable, assert `SHOW data_directory == $NATIVE_PGDATA` (reuse the
  `restore_preflight` check :1288-1291). **Aggregate the failure through the existing
  `doctor_report FAIL` mechanism — do not call `fail`/exit** (which would suppress later
  diagnostics — codex). Soften the header "ports" wording. Tests: matching `data_directory`,
  foreign `data_directory`, and query/auth failure — three separate cases.
- **#12 `VOXINT_NATIVE_OLD_PG_BINDIR` validation.** Add `_reject_control_chars
  VOXINT_NATIVE_OLD_PG_BINDIR` to `validate_native_inputs` (consumed :1532-1538, bypasses the gate
  today). **No existing `validate_native_inputs` test to extend** (codex) — add a new parameterized
  validation test, **and add the var to `run_lib`'s ambient-env scrub list** (it isn't scrubbed
  today, so the test would otherwise leak the host value).
- **#13(code):** add a one-line `upgrade-db --rehearse (maintainer self-test)` to the built-in
  usage block (:2106-2108) so the accepted flag is discoverable, while keeping it out of the
  operator doc (codex).
- `CHANGELOG` Fixed/Changed. Acceptance: the reserved-password smoke + the crash-loop fault gate.

### Phase 3 — `pipefail` only (#11)

Branch `chore/native-pipefail`. Nothing else — keep the broadest change isolated (codex).

- **Inventory every pipeline first** and classify expected non-zero producers before flipping
  `set -eu` → `set -euo pipefail` (:52). Known SIGPIPE-sensitive spots: the `psql … | grep -q`
  probes in `ensure_database` and `doctor`, `rotate-logs`, and `upgrade-db --rehearse` (codex).
  Replace fragile probe pipelines with captured scalar results where practical.
- **Dedicated acceptance matrix** (codex): setup / up / doctor / status / rotate-logs / **plain and
  fresh restore** / **upgrade-db --rehearse** — the existing smoke + restore lanes don't cover
  rotate-logs or `--rehearse`, both of which contain pipelines.
- `CHANGELOG` Changed.

### Release (after the phases that ship together)

- **One `0.18.0` bump** when the code phases ship together (the `[Unreleased]` section already
  holds feature work, so a minor is right — codex). Atomic across: `pyproject.toml`,
  `src/voxint/__init__.py`, **all six** `compose*.yaml` `VOXINT_IMAGE_TAG` pins (per the contract
  glob), `.env.example`. **Run `uv lock`** so the editable `voxint` entry updates (CI uses
  `uv sync --frozen` — codex). Update the `v0.17.0` banner in
  `docs/how-to/settings-and-troubleshooting.md`. Fix the stale "all four compose" line in
  `CLAUDE.md`. Then follow `docs/release-process.md` (gates + Gate M metal lane).

## Affected files / components

- `docs/setup.md` — CPU memory, metal `uv`/status/Docker-Desktop/differentiation, decision aid,
  page framing, CPU pull note.
- `docs/operations.md` — metal `uv`/Homebrew prereq (#2, both-doc fix).
- `docs/native-macos-preview.md` — onboarding links (#5), wording nits (#16), restore-safety note.
- `docs/how-to/add-media-and-manage-runs.md`, `docs/onboarding.md` — #14 Docker-only labelling.
- `docs/testing.md` — `--rehearse` maintainer note.
- `scripts/native/voxint-native.sh` — shared collision-safe safety-backup helper + plain-restore
  call (#3); `cmd_status` liveness (#6); `native_database_url` encode (#7); `cmd_doctor`
  data_directory (#10); `validate_native_inputs` OLD_PG_BINDIR (#12); usage `--rehearse` (#13);
  `set -euo pipefail` + pipeline hardening (#11).
- `tools/native_e2e_lifecycle.py` — DSN encode parity (#7); plain-restore acceptance step.
- `tests/unit/test_native_launcher.py` — restore-safety, launchctl-state, DSN round-trip,
  validation, doctor cases; `run_lib` scrub-list addition.
- `.claude/skills/voxint-native-e2e/SKILL.md` — plain-restore + reserved-password + crash-loop
  acceptance rungs.
- Release: `pyproject.toml`, `src/voxint/__init__.py`, six `compose*.yaml`, `.env.example`,
  `uv.lock`, `docs/how-to/settings-and-troubleshooting.md`, `CLAUDE.md`, `CHANGELOG.md`.

## Testing strategy (gates per code phase)

- **Offline, every phase:** `uv run ruff check .`, `uv run mypy`, `bash -n scripts/native/*.sh`
  under Bash 3.2, and the **actual CI pytest command** — the full `tests/` tree with PostgreSQL +
  coverage, not just unit+contracts (a shippable phase must pass what CI runs — codex).
- **Reference named commands/behaviours, not fixed test counts** (the "126"/"125" figure drifts —
  codex).
- **macOS-only acceptance** (maintainer, opt-in, serial): Phase 1 → extended `--with-restore`
  covering **plain** restore; Phase 2 → reserved-password `--no-models` smoke + real crash-loop
  fault gate; Phase 3 → the dedicated `pipefail` matrix (incl. rotate-logs + `--rehearse`).
- Never weaken an assertion to pass; a red gate is information.

## Rollout / risks / open questions

- **Risk — `pipefail` blast radius (#11):** highest despite being "polish." Mitigated by the
  pre-flip inventory + dedicated acceptance matrix; any breakage is a real latent bug to fix.
- **Risk — `launchctl` parse portability (#6):** non-API output; kept best-effort with conservative
  unknown-state handling + a real-macOS gate. Never regresses `status` into a false "healthy."
- **Risk — Bash percent-encoder correctness (#7):** RFC-3986 encoding in Bash 3.2 is fiddly;
  covered by the `make_url` round-trip + reserved-password smoke. Fallback = reject (O1).
- **O1 (decide):** #7 — **percent-encode both composers** (recommended; preserves operator freedom)
  vs **reject reserved chars at validation** (simpler/Bash-safe, restricts passwords). Plan assumes
  encode.
- **O2 (decide):** #14 — ship the honest **Docker-only labelling** now (recommended) vs also build
  a native env-aware `voxint` CLI wrapper (larger, separate feature). Plan assumes labelling now,
  wrapper deferred.
- **O3 (confirm):** version cadence — **one `0.18.0`** when the code phases ship together
  (recommended, per codex) vs a bump per phase.

## Verification (end-to-end)

1. Per phase: offline gates green (ruff/mypy/`bash -n`/full CI pytest) on the phase branch; FF-merge
   only when green; re-cut the next branch from the merged `main`.
2. Phase 1: run the extended native `--with-restore` lane; confirm a `SAFETY_BACKUP` 0600 dump is
   written on a **plain** restore and the app reads back the pre-restore data.
3. Phase 2: run the reserved-password `--no-models` smoke (alembic + services start on the encoded
   DSN); trip a crash-looping worker and confirm `status` shows a non-healthy state (not
   `[supervised]`); point `doctor` at a foreign postmaster and confirm a `FAIL`.
4. Phase 3: run the `pipefail` acceptance matrix; all subcommands succeed and no pipeline regresses.
5. Release: `0.18.0` atomic bump green under the pin-parity contract test + `uv sync --frozen`;
   `docs/release-process.md` gates incl. Gate M metal lane.

## Review notes (codex planner critique — what was flagged & how resolved)

- **#14 bare `voxint` is invalid** (venv not on PATH, no launcher env) → **accepted**; Phase 0 now
  labels the recipes Docker-only + points to the browser; native CLI wrapper deferred (O2).
- **#6 `launchctl print` is non-API** → **accepted**; conservative state model, no false-healthy
  fallback, real-macOS crash-loop gate; scope narrowed to worker/beat (dropped datastores).
- **#7 second composer in `tools/native_e2e_lifecycle.py`** → **accepted** (verified :187-189/:528);
  both composers fixed, full RFC-3986 encode, `make_url` round-trip + reserved-password smoke.
- **#2 both docs** → **accepted**; `operations.md` metal prereq added alongside `setup.md`.
- **#12 test plan** (no existing test to extend; `run_lib` doesn't scrub the var) → **accepted**;
  new parameterized test + scrub-list addition; **#12 moved to Phase 2** to isolate `pipefail`.
- **#16 CR/LF wording** → **accepted, reverses my draft**: code rejects only CR/LF, so "newline" is
  accurate — do not broaden to "control characters."
- **#15 "optional"** conflicts with full-remediation goal → **accepted**; parenthetical included.
- **#13** → **accepted**; testing.md note **plus** a maintainer-labelled `--help` usage line.
- **Phase 0 framing/decision-aid conflation** → **accepted**; page-level Docker framing fixed;
  decision aid separates deployment-path from compute tier; native not listed as a 5th tier.
- **Phase 1 helper placement/collision/ordering-test/failure-assertions/plain-restore E2E** →
  **all accepted**; folded into Phase 1.
- **Phase 2 doctor must aggregate not exit; percent-encode completeness; make_url test** →
  **accepted**.
- **Phase 3 pipeline inventory + dedicated acceptance; isolate #12** → **accepted**.
- **Gates: use real CI command, not unit+contracts; name commands not counts** → **accepted**.
- **Versioning: six compose files not four (verified against the pin-parity contract glob); run
  `uv lock`; update the v0.17.0 banner; `CLAUDE.md` "four" is stale** → **accepted**; also logged
  as a bonus `CLAUDE.md` fix. Branch-from-merged-main mechanics → **accepted**.
- **Not deferred/ignored:** every codex point was folded in; the only judgment calls left to the
  user are O1–O3 above.
