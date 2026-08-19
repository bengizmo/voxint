---
name: voxint-docs
description: >-
  Write and edit Voxint documentation in the project house style. Use whenever
  you author or revise any doc a person will read: the README, anything under
  docs/ (setup, onboarding, how-to guides, architecture, contracts,
  operations, reference), CONTRIBUTING, installer and first-run and error copy,
  or a dated report/plan. First pick the audience lane (lay-reader vs
  technical), then apply the hard rules that hold in both. Hard bans, no
  exceptions: no emdashes anywhere, and no LLM-isms (negative parallelism,
  rule-of-three padding, empty summary closers, over-signposting). Voxint prose
  is emoji-free; only a hazard or success glyph in a callout is allowed.
---

# Writing Voxint documentation

Voxint serves individuals and small teams running locally hosted audio
intelligence on their own hardware: non-technical researchers, journalists, and
educators alongside the developers who deploy and extend the stack. Those are
two different readers, and a doc that tries to serve both serves neither. This
skill gives you one house style with two audience lanes. Pick the lane, write to
that reader, and follow the hard rules that hold in every lane.

The audience mandate in `CLAUDE.md` is load-bearing here: correctness, numerics
stability, and non-technical onboarding are the priorities, and "no unnecessary
bloat" applies to prose as much as to code. A shorter doc that says the true
thing beats a longer one that pads.

## Step 1: pick the lane

Choose the reader before you write the first line. The lane sets vocabulary,
sentence length, and structure.

| Lane | Reader | Docs in this lane |
|---|---|---|
| **Lay-reader** | Grade-10 education. Does not know what a terminal, package manager, or environment variable is. Wants an outcome. | `README.md`, `docs/setup.md`, `docs/onboarding.md`, `docs/how-to/*`, the user-facing parts of `docs/native-macos-preview.md`, and all installer / first-run / wizard / error copy the operator sees. |
| **Technical** | Post-secondary computer-science background. Has a working dev environment. Wants the contract and the command. | `CONTRIBUTING.md`, `docs/architecture.md`, `docs/gpu-contracts.md`, `docs/gpu-smoke.md`, `docs/quality-gates.md`, `docs/enrichment-triage.md`, `docs/interpreting-diarization.md`, `docs/timeouts-and-leases.md`, `docs/harness.md`, `docs/domain-packs.md`, `docs/operations.md`, `docs/testing.md`, `docs/release-process.md`. |

There is a third register for internal working artifacts: `docs/reports/*` and
`docs/plans/*` are status-tagged and analytical, written for a maintainer
reviewing evidence or a plan. They open with a `> **Status:**` blockquote and
use numbered analytical sections. They are not user-facing; do not write them in
either audience voice. Anything that names internal machines, IPs, or hosts does
not belong in this public repo at all (see Placement below).

When a single doc genuinely spans both readers (the top-level `README.md` is the
case in point), keep the body in the lay-reader lane and push the technical
material into a clearly marked section near the end ("For developers").

## Step 2: hard rules (every lane)

These hold whether you are writing for an operator or a compiler engineer.

### No emdashes

Never use an emdash (`—`) or an en-dash used as one. Restructure instead. A
comma, a period, a colon, or parentheses almost always reads better.

- Instead of: `Voxint runs locally — nothing leaves your machine.`
- Write: `Voxint runs locally, so nothing leaves your machine.`
- Or: `Voxint runs locally. Nothing leaves your machine.`

The existing docs use spaced hyphens and restructured sentences for parenthetical
asides. Match that.

### No LLM-isms

These structural tics read as machine-written. Cut every one.

| Tic | Do not write | Write instead |
|---|---|---|
| Negative parallelism | "These are proposals, not verdicts, they are not conclusions but suggestions." | "These are proposals. You have the final say." |
| "Not just X but Y" | "This isn't just a transcriber, it's a full intelligence pipeline." | "It transcribes, separates speakers, and identifies voices." |
| Rule-of-three padding | "It is fast, powerful, and flexible." | Name the one property that matters and show it. |
| Grand opener | "In today's fast-paced world of audio analysis..." | Start with what the doc does for the reader. |
| Empty summary closer | "In conclusion, Voxint is a great tool for..." | End on the last useful instruction. Delete the wrap-up. |
| Over-signposting | "It's worth noting that", "It is important to understand that" | State the fact directly. |
| Hedge stacking | "This may possibly help in some cases perhaps." | Say what is true, or say you do not know. |

If a sentence would survive being deleted with no loss to the reader, delete it.

### Honest copy

State what is actually true. If a service being down means submissions fail, say
that; do not soften it. Do not claim behavior that is not there (the
`CLAUDE.md` example: do not say "downloading weights" when the weights are baked
into the image). A doc that oversells is a bug.

### Emoji policy

Voxint prose is emoji-free. Do not decorate headers or sprinkle emoji through
running text. Two glyphs are allowed, and only inside a callout line:

- `⚠️` (or the bare `⚠` already used in `CLAUDE.md`) for a genuine hazard the
  reader must not miss.
- `✅` for a success confirmation in a lay-reader walkthrough ("You should now
  see **Console is up** ✅").

That is the whole budget. When in doubt, use a `>` blockquote instead.

### Markdown

Use GitHub Flavored Markdown only. Syntax-highlight every fenced code block
(` ```bash `, ` ```python `, ` ```yaml `). Use tables for options and
configuration. Use relative links (`docs/setup.md`, not a full URL) and
cross-link heavily: existing docs point forward to the next step and back up to
their index, and yours should too.

## Step 3a: the technical lane

Write for a developer who is comfortable in a shell and wants to get to a working
state fast. Approachable and direct, with earned confidence. Keep setup terse:
assume `uv`, `docker`, and git are already installed and configured.

Be code-first. Minimize the throat-clearing before the first command. Structure
is a toolkit, not a mandatory template; use the parts that fit the doc:

- **Title and framing.** An H1, then one sentence on what the doc gives the
  reader and why it exists. For a landing-style doc you may open with a row of
  Shields.io badges (build status, version, coverage, license). Do not badge an
  ordinary reference page.
- **Quickstart.** The real install command followed by a minimal,
  copy-pasteable snippet that produces a result. Voxint is Python with `uv` and
  Docker Compose, so the commands are `uv sync --extra dev`, `uv run pytest ...`,
  `docker compose up -d ...`. Never write `npm install` or a bare `pip install`;
  the source style guidance that mentions them is generic, and this repo does not
  use them.
- **In-line API documentation.** Document parameters, types, defaults, and
  return values in a GFM table inside the doc. Do not link out to a wiki for the
  contract. `docs/gpu-contracts.md` is the reference for how detailed a contract
  table should be.
- **Governance footer** where it fits (a README, CONTRIBUTING): link
  `CONTRIBUTING.md`, `SECURITY.md`, and the license.

A parameter table looks like this:

| Field | Type | Default | Description |
|---|---|---|---|
| `compute_tier` | `str` | `cpu` | Selects the timing profile in `docs/timeouts-and-leases.md`. |
| `hf_token` | `str \| None` | `None` | Optional; restores the online model-download path. |

## Step 3b: the lay-reader lane

Write for someone who has never opened a terminal and does not want to. Be
empathetic, reassuring, and patient. Where the technical lane is terse, this lane
is thorough: complete sentences, and a gentle hand through anything that could
confuse. Explain what the software does for the reader and skip how it works
inside.

Never assume jargon. If a step truly needs a technical term, define it in plain
words the first time. Match the existing operator voice in
`docs/how-to/reviewing-and-adjudicating.md`: second person, warm, an italic
one-line subtitle under the H1, bold for the UI elements the reader clicks or
looks at, and `>` blockquotes for tips and cautions.

Structure for a user-facing guide:

- **Plain-English intro.** A relatable scenario, not an elevator pitch. Why would
  a normal person reach for this tool today?
- **Getting the software (releases funnel).** Do not mention cloning or source
  code. Send the reader to the GitHub Releases page and tell them exactly which
  file to download for their operating system ("Download the `.dmg` for Mac").
- **First-run walkthrough.** A numbered tour of what happens when they open the
  app the first time, with screenshot placeholders that point at the existing
  images directory: `![Welcome screen](images/welcome-screen.png)`.
- **Task-based how-tos.** Organize by what the reader wants to accomplish ("How
  to export a transcript", "How to rename a speaker"), not by feature list.
- **Developer handoff.** A short, polite section at the end for advanced readers:
  "Looking to build from source? See [CONTRIBUTING.md](../CONTRIBUTING.md)."

### Plain-English troubleshooting

Anticipate the friction a non-technical reader actually hits and walk them
through it click by click. The common ones:

- **macOS Gatekeeper** ("Voxint can't be opened because it is from an
  unidentified developer"): tell them to open **System Settings**, go to
  **Privacy & Security**, and click **Open Anyway**, in numbered steps.
- **Windows Defender SmartScreen** ("Windows protected your PC"): tell them to
  click **More info**, then **Run anyway**.

Give the exact button names in bold, and reassure them the step is safe and
expected.

## Step 4: placement and filenames

Put the doc where readers and the indexes expect it, and codify the naming.

- **Reference docs** go in `docs/` root with kebab-case names
  (`audio-pipeline-architecture.md`). No dates on reference docs.
- **Task guides** go in `docs/how-to/`. Add a row to both
  `docs/how-to/README.md` and `docs/README.md` in the same change.
- **Dated artifacts.** Reports go in `docs/reports/` as
  `slug-YYYY-MM-DD.md`; plans go in `docs/plans/` as
  `YYYY-MM-DD-HHMM_slug.md`.
- **Never** put a new doc flat in the repo root or bare in `docs/` when a
  subdirectory fits.
- **Public clean-room rule.** This is a public repo. Never commit internal
  hostnames, IPs, credentials, or machine names. In public releases and issues,
  say "the reporting host" or "maintainer hardware". Internal notes
  (session prompts, runbooks with machine names) live in the separate
  `internal/` repo, never here.
- **Keep docs current.** Update `docs/` in the same change that alters the
  behavior it documents. Stale docs are treated as bugs.

## Step 5: pre-publish checklist

Before you consider a doc done, scan it against this list:

- [ ] Lane chosen, and the voice matches it start to finish.
- [ ] No emdashes anywhere (search for `—`).
- [ ] No LLM-isms: reread against the table in Step 2.
- [ ] Emoji budget honored (none in prose; at most a `⚠️`/`✅` callout glyph).
- [ ] Copy is honest: every claim is actually true of the current build.
- [ ] Code blocks are language-tagged; options are in tables.
- [ ] Links are relative and resolve; the doc cross-links to its neighbors.
- [ ] Placed in the right directory; indexes (`docs/README.md`, `how-to/README.md`) updated.
- [ ] Nothing internal (hostnames, IPs, credentials) leaked into the public tree.
