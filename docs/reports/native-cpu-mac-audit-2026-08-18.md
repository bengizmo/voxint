# Audit: Non-NVIDIA install paths (CPU tier, Apple-Silicon metal, docker-free native) — features, behaviour & docs

> Date: 2026-08-18 · Repo @ `d8e85e7` (`main == origin/main`, CI green) · Version `0.17.0`
> Posture: **audit-and-report only** — no code or docs changed. This report ranks findings
> for maintainer triage; nothing here is applied yet.

## Audit lens (load-bearing)

Per project `CLAUDE.md` "Who it is for": Voxint serves **single-operator, self-hosted,
often non-technical** users (researchers, journalists, educators) running on **their own
Mac or a CPU-only box**. Correctness, numerics stability, and non-technical onboarding
matter; **enterprise-scale security and scalability are explicitly not concerns**. Findings
are severity-ranked by *real impact on that operator successfully choosing, installing,
and running Voxint* — not by abstract robustness.

## Scope — the three distinct non-NVIDIA paths

1. **CPU tier (Docker)** — `compose.yaml` + `compose.cpu.yaml`, `-cpu` images, `COMPUTE_TIER=cpu`,
   installed via `scripts/install.sh` tier `cpu`. Doc: `docs/setup.md` §3 CPU.
2. **Metal tier (Docker core + native model services)** — `compose.metal.yaml` +
   `scripts/metal/voxint-metal.sh`, installed via `scripts/install.sh` tier `metal`.
   Doc: `docs/setup.md` §3 metal.
3. **Native (no-Docker) macOS preview** — `scripts/native/voxint-native.sh` (brew Postgres 17
   + pgvector + Redis, launchd; delegates to the metal launcher for models), entry point
   `voxint-native.sh setup`. An explicit **technical preview**. Doc: `docs/native-macos-preview.md`.

Method: four parallel read-only sub-audits (one per path + one cross-path coherence pass)
against the scripts, compose overlays, `config.py`, and the unit/contract tests; a fifth pass
on the native script's subcommand/failure-mode surface. Every finding below was checked against
the cited file; the five highest-impact ones were re-verified directly by the author.

---

## Headline

**No Critical or High findings.** No path has a broken install command, a guaranteed
data-loss operation, or version drift; image tags, ports, the `COMPUTE_TIER` timing story,
the security hardening, `upgrade-db`, and PG skew-detection are all accurate and well-tested.
The native doc in particular is exceptionally accurate.

What the audit *did* surface is a cluster of **Medium footguns and coherence gaps**, all of
which bite this specific audience harder than they would a technical user:

- Two **onboarding trip-hazards** where `setup.md` omits a prerequisite/limit that the code
  enforces or that `operations.md` documents (metal `uv`; CPU 8 GB → OOM).
- One **data-safety inversion** in the native restore path (plain restore has no safety backup;
  `--fresh` does).
- The **native preview's isolation** from the onboarding funnel (under-differentiated from
  metal at the one decision point; dead-ends without linking the first-run walkthrough).
- Two **honesty gaps** in native `status`/`doctor` (a crash-looping worker reads as healthy;
  a foreign port-squatter reads as ours).

## Severity summary

| # | Severity | Path | Finding | Fix size |
|---|----------|------|---------|----------|
| 1 | Medium | CPU | `setup.md` presents 8 GB as adequate; it is the OOM floor (16 GB comfortable) | 1 line |
| 2 | Medium | Metal | `uv` prerequisite undocumented in `setup.md` — script hard-fails without it | 1 line |
| 3 | Medium | Native | Plain `restore` takes no safety backup, but the scarier `--fresh` does (inversion) | small code |
| 4 | Medium | Coherence | Metal vs native under-differentiated at the one decision point (`setup.md`) | 1–2 lines |
| 5 | Medium | Coherence | `native-macos-preview.md` dead-ends — no link to wizard / tutorial / how-to | 1 line |
| 6 | Medium | Native | `status` reports a crash-looping worker/beat as `[supervised]` (healthy) | small code |
| 7 | Medium | Native | Operator DB password with URL-reserved chars → malformed `DATABASE_URL` | small code |
| 8 | Medium | Metal | Docker-Desktop-only caveat absent where the tier is chosen | 1 line |
| 9 | Low | Coherence | No "which path is for me?" decision table in `setup.md` §3 | small doc |
| 10 | Low | Native | `doctor`/`status` can't detect a port collision / foreign squatter; wording overpromises | small code |
| 11 | Low | Native | `set -eu` without `pipefail` (no silent-loss path found; defensive only) | 1 line |
| 12 | Low | Native | `VOXINT_NATIVE_OLD_PG_BINDIR` bypasses `validate_native_inputs` | 1 line |
| 13 | Low | Native | `upgrade-db --rehearse` is a real flag, undocumented in usage/doc | 1 line |
| 14 | Low | Coherence | `docker compose exec` CLI forms in how-to/onboarding break for native users | 2 notes |
| 15 | Low | CPU | CPU section shows no `pull` while GPU section does (cosmetic asymmetry) | optional |
| 16 | Low | Native | `native-macos-preview.md` nits: `doctor` "port collision" wording, smoke-gate subset, newline vs control-char wording | wording |

---

## Medium findings

### 1 — CPU: `setup.md` presents 8 GB as adequate; it is the OOM floor *(verified)*
- **Where:** `docs/setup.md:34` ("raise the VM's memory limit to ≥ 8 GB") and `docs/setup.md:103`
  ("about 8 GB of memory free") vs `docs/operations.md:90-96`.
- **Reality:** `operations.md` documents 8 GB as the *bare floor* — whisper alone is ~4.8 GiB
  resident, the tier idles ~6 GiB *including* the core stack (Postgres/Redis/api/worker in the
  same VM), **"16 GB is comfortable"**, and crucially **"Under the floor the services are
  OOM-killed with an opaque exit, not a clear message."** A non-technical operator who reads only
  `setup.md`, sets the Docker Desktop VM to exactly 8 GB, and submits a long recording can hit
  silent OOM-kills they cannot diagnose — the exact failure mode this audience can't recover from.
- **Fix:** add one clause to the CPU section mirroring `operations.md`: "8 GB is the tight floor
  (services OOM-kill with an opaque exit below it); 16 GB is comfortable."

### 2 — Metal: `uv` prerequisite undocumented where the user reads it *(verified)*
- **Where:** `docs/setup.md:136-153` and `docs/operations.md:157-162` present `voxint-metal.sh setup`
  with no prerequisite; contradicted by `scripts/metal/voxint-metal.sh:544` (`require_tools` →
  `fail "uv is required (… brew install uv)"`), called first in `cmd_setup`.
- **Reality:** `uv` is not on stock macOS and is mentioned nowhere in `setup.md`/`operations.md`;
  the only hint is a parenthetical inside the installer's interactive menu (`install.sh:347`) a
  user won't see again. Following the docs, the operator hits an immediate hard-fail. Recoverable
  (the error text names the fix), but a first-run wall for the target audience.
- **Fix:** add "Requires `uv` (`brew install uv`) and Homebrew" above the metal command block in
  `setup.md` (and `operations.md`).

### 3 — Native: plain `restore` takes no safety backup, inverting the `--fresh` guarantee *(verified)*
- **Where:** `scripts/native/voxint-native.sh` `plain_restore` (1356-1387, no safety backup) vs
  `fresh_restore` (1401-1421, pre-drop 0600 safety dump with abort-on-failure). Doc:
  `docs/native-macos-preview.md:207-236`.
- **Reality:** Plain `restore <file>` runs `pg_restore --clean --if-exists --single-transaction
  --exit-on-error` against the live DB. The single transaction means a *failed* restore rolls back
  cleanly (original data intact) — that part is solid. But a **successful** restore of a *valid but
  wrong/older* voxint dump silently replaces current rows with no automatic backup and no
  confirmation. The identity gate only proves the dump *is* a voxint dump, not that it is the right
  one. Net inversion: `--fresh` (drop+rebuild) is recoverable via its safety dump; plain `restore`
  is not. This is the most reachable unrecoverable-data path for a preview operator.
- **Fix:** take the same `pre-restore-<stamp>.dump` 0600 safety backup in `plain_restore` that
  `fresh_restore` already does before the first `--clean`. (Small, symmetric with existing code.)

### 4 — Coherence: metal vs native under-differentiated at the one decision point *(verified)*
- **Where:** `docs/setup.md:136-152`. Native appears only as a trailing sentence ("There is also a
  docker-free native preview…"); the selection criterion lives only in the *target* doc
  (`native-macos-preview.md:6-7`).
- **Reality:** Both paths target Apple Silicon and both run models natively (metal keeps the core in
  Docker; native drops Docker entirely). A Mac user reading `setup.md` cannot tell which Apple path
  is theirs, or that native is more advanced/preview-only.
- **Fix:** one clause at `setup.md:150`, e.g. "Most Mac users want **metal**. Choose the native
  preview only if you can't or won't run Docker Desktop — it's a hands-on technical preview."

### 5 — Coherence: `native-macos-preview.md` dead-ends without routing to first-run docs *(verified links)*
- **Where:** `docs/native-macos-preview.md` (whole) — links only to `security/audit-2026-08-18.md`
  and `testing.md`; never to `onboarding.md` (wizard + guided tutorial) or `how-to/`.
- **Reality:** A user arriving via `setup.md`'s one-liner completes `voxint-native.sh up`, the
  browser opens, and the doc leaves them at "now what?" The Docker paths all funnel through
  `setup.md` §5 → `onboarding.md`; the native path does not.
- **Fix:** add a "Next: first-run walkthrough" line after the install/run block (~:58) linking
  `onboarding.md` and `how-to/README.md`.

### 6 — Native: `status` reports a crash-looping worker/beat as healthy *(verified via agent; code-cited)*
- **Where:** `scripts/native/voxint-native.sh` `cmd_status` (1107-1121). Only `api` gets a real
  liveness signal (`/healthz`); `worker`/`beat` print `[supervised]` purely from `launchctl print`
  exit status.
- **Reality:** `KeepAlive{SuccessfulExit=false}` (511-513) keeps a failing job bootstrapped and
  restarting; `launchctl print` still returns 0, so `status` shows `worker [supervised]` while it
  crash-loops (bad import, missing model URL, DB auth). `doctor` never inspects api/worker/beat
  supervision either. The operator gets no signal the worker is dead — submissions silently never
  progress.
- **Fix:** in `status`, surface each job's `state = …` / `last exit code = …` from `launchctl print`
  output rather than only its exit code.

### 7 — Native: operator DB password with URL-reserved chars → malformed `DATABASE_URL` *(verified via agent; code-cited)*
- **Where:** `native_database_url` (188-191, raw `printf` interpolation) vs `validate_native_inputs`
  (225-268, which only rejects CR/LF in the password).
- **Reality:** The auto-generated secret is 64-hex (safe), but an operator setting
  `VOXINT_NATIVE_DB_PASSWORD` to a value containing `@ : / # %` yields a DSN the app/alembic parse
  wrong → cryptic connection failure at `up` (after the migrate gate), not a clear error at input
  time. `validate_native_inputs` advertises covering the externally-influenced inputs but doesn't
  catch this.
- **Fix:** percent-encode the password in `native_database_url`, **or** add a reserved-char reject
  for `VOXINT_NATIVE_DB_PASSWORD` in `validate_native_inputs` with a clear message.

### 8 — Metal: Docker-Desktop-only caveat absent where the tier is chosen *(verified via agent)*
- **Where:** `docs/setup.md:136-153` (no mention); stated fully only in `compose.metal.yaml:13-19`
  (Colima/OrbStack/plain dockerd break the host-gateway→loopback mapping) and partially at
  `operations.md:171`.
- **Reality:** `setup.md`'s metal section never states Docker Desktop is *required* (not merely one
  option). Largely mitigated — `setup.md:34` funnels Mac users to Docker Desktop and
  `voxint-metal.sh doctor` catches a broken engine so submissions "fail cleanly" — but a user
  already on Colima gets a broken tier with no upfront warning.
- **Fix:** one sentence in the `setup.md` metal section: "This tier needs **Docker Desktop**
  specifically; Colima/OrbStack/plain dockerd can't route the containers to the native services."

---

## Low findings

- **9 — No "which path is for me?" decision aid.** `docs/setup.md:85-152` lists five tiers in
  prose with no at-a-glance selector, and native isn't in the tier list at all. Mitigated by the
  installer's hardware auto-suggestion (`onboarding.md:39-44`). *Fix:* add a small hardware→tier→doc
  decision table at `setup.md:87` with native as an explicit "no-Docker, preview" row.
- **10 — `doctor`/`status` can't detect a port collision / foreign squatter.**
  `voxint-native.sh` header (line 26) advertises "ports"; `cmd_doctor` (1863-1995) and `cmd_status`
  (1097-1103) only probe reachability — a foreign Postgres/Redis on the configured port reads as
  `listening`/`reachable`. Only `restore_preflight` (1288-1291) proves `data_directory` is ours.
  *Fix:* in `doctor`, when the managed cluster exists and Postgres is reachable, assert
  `SHOW data_directory == $NATIVE_PGDATA` and fail on mismatch; soften the "ports" wording.
- **11 — `set -eu` without `pipefail`** (`voxint-native.sh:52`). Audited the risky pipelines
  (`ensure_database` probes, `env_value_from_file`, the destructive drop/restore gates use
  scalar-capture + `|| fail`, not pipes) — **no silent-success path found**. *Fix (defensive only):*
  add `-o pipefail`.
- **12 — `VOXINT_NATIVE_OLD_PG_BINDIR` bypasses `validate_native_inputs`** (consumed at 1532-1538).
  Only ever used quoted as a path, never serialized into a plist/state.env, so the CR/LF-forgery
  threat model doesn't apply — but it's an operator input outside the gate the script says covers
  "the only externally-influenced inputs." *Fix:* add a CR/LF reject (documentation-of-intent).
- **13 — `upgrade-db --rehearse` undocumented.** A real, reachable flag (parsed at 1837; forces a
  same-major cycle as a mechanical proof) omitted from both the built-in usage block (2106-2108)
  and `native-macos-preview.md`. Consistently treated as a maintainer aid. *Fix:* add a one-line
  "(maintainer self-test)" note to the usage block/`testing.md`, or state it's intentionally
  internal. Leave out of the operator doc.
- **14 — `docker compose exec` CLI forms break for native users.**
  `docs/how-to/add-media-and-manage-runs.md:136-140` and `docs/onboarding.md:114` show
  `docker compose exec api voxint …`, which the no-Docker native install can't run. Each is gated as
  an optional "for people comfortable with the terminal" escape, but presented as universal.
  *Fix:* one-line note at each block: "(Native preview uses `voxint …` directly, not
  `docker compose exec`.)"
- **15 — CPU section shows no `pull`, GPU section does** (`setup.md:97-99` vs `112-115`). Not a
  defect — CPU services use `pull_policy: missing`, so `up -d` auto-pulls — but the asymmetry can
  read as a skipped step. *Fix (optional):* add "(`up -d` pulls the images on first run)" for
  symmetry, or drop the standalone `pull` from the GPU block.
- **16 — `native-macos-preview.md` wording nits.** (a) `:32-33` "doctor checks … port collisions"
  → it reports reachability, not collision (see #10); (b) `:257-259` smoke-gate description is a
  subset of `SKILL.md` (omits `/setup` 200); (c) `:94-98` "may [not] contain a newline" understates
  the `_reject_control_chars` gate, which rejects all control chars over more vars than the
  illustrative list. All conservative/illustrative, none wrong. *Fix:* minor rewording if touched.

---

## Verified accurate — no action (reassurance)

These were checked and hold; they are the load-bearing correctness claims and they pass.

- **No version drift.** `${VOXINT_IMAGE_TAG:-0.17.0}` compose defaults match `__init__.py` /
  `pyproject.toml`; `-cpu` image tags, ports (8080/8021/8022/8024), and CPU/metal install commands
  all match the scripts and `tests/unit/test_installer.py` / `test_metal_launcher.py`.
- **`COMPUTE_TIER=cpu` 4× scaling** is real and documented correctly:
  `CPU_TIER_TIMEOUT_FACTOR = 4.0` (`config.py:51`) over `TIER_SCALED_TIMING_FIELDS`; baselines
  4h/6h/12h/48h × 4 = 16h/24h/48h/192h exactly as `timeouts-and-leases.md` states. "Results
  identical" is backed by the titanet three-level parity gate.
- **Native doc (`native-macos-preview.md`) is exceptionally accurate.** Every subcommand, port,
  the PostgreSQL-17 references, secret hygiene (0600/0700), `validate_native_inputs`, skew
  detection, `upgrade-db`, and backup/restore claims match the script line-for-line. The
  "technical preview, not the non-technical release" framing is clear and names epic child #73 as
  the packaged path. Metal-launcher delegation for models is described correctly.
- **Destructive-operation safety is otherwise solid.** `fresh_restore` (pre-drop safety dump +
  DROP…WITH FORCE + prove-empty OID/table gates, every probe status-checked), `upgrade-db`
  (stack-down + disk-headroom + single-DB inventory gate + pgvector round-trip + dump-restorable
  proof before touching pgdata + atomic rename retaining the old cluster + rollback trap + `up`
  refusal on a half-done upgrade, one-major-forward only), and PG **skew detection** (offline
  major reads, Homebrew-suffix-safe, fires in `up` before any bootstrap, mirrored in `doctor`,
  fail-closed on damaged clusters) all verified. `down` is non-destructive. #3 above is the one
  gap in an otherwise strong data-safety story.
- **Secret handling is solid.** umask-077 mktemp + atomic mv + `chmod 600` for `state.env`,
  0700 dirs, 0600 plists/backups; `validate_native_inputs` enforces numeric/range ports,
  `[A-Za-z_][A-Za-z0-9_]*` identifiers, positive log sizes, CR/LF rejection on every serialized
  value, and runs both at dispatch and again after reading `state.env` (closes the hand-edited hole).
- **Partial-install behaviour is reasonable.** `cmd_setup` is idempotent/re-runnable (cluster init
  early-returns on existing `PG_VERSION`, `state.env` reuse preserves ports/secrets, metal-setup
  failure is recorded and returned so `setup && up` halts). No cleanup trap, but idempotency covers it.
- **Coherence basics hold.** No dead links — all in-scope relative file links and same-page anchors
  resolve (checked against disk). No factual contradictions between `setup.md` and `onboarding.md`
  (installer tier-suggestion, metal two-step handoff, 8 GB figure all consistent). Doc hygiene
  clean: all in-scope docs live in `docs/` subdirs, kebab-case; dated reports carry ISO suffixes.
- **Gate M / metal-lane** (`release-process.md:115-139`) matches `.github/workflows/metal-lane.yml`
  on `macos-15` and the present parity tests; the `#33` ct2-legacy verdict matches the code pins.

---

## Recommended remediation order (if/when you choose to act)

1. **Docs-only, zero-risk, high-leverage (batch as one small doc PR):** #1, #2, #4, #5, #8, #9, #14,
   #15, #16, #13. These close the two first-run trip-hazards and the native-preview isolation, and
   are all one-to-few-line edits with no behaviour change. This is the highest value-per-effort.
2. **Native data-safety (#3):** add the plain-`restore` safety backup — small, symmetric with
   existing `fresh_restore` code, closes the one reachable unrecoverable-data path. Gets a test.
3. **Native honesty/robustness (#6, #7, #10):** `status` liveness for worker/beat, DB-password
   reserved-char handling, `doctor` data_directory assertion. Small code + tests each.
4. **Defensive polish (#11, #12):** `pipefail`, `OLD_PG_BINDIR` validation. Lowest urgency.

Each code change follows the normal gate (multi-model review where non-trivial, tests,
CHANGELOG under `[Unreleased]`, docs updated in the same change). **No changes have been made;
awaiting your call on what to act on.**
