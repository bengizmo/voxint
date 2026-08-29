# Domain packs

A **domain pack** supplies the domain-specific knowledge the pipeline consumes
(ASR vocabulary hints, speaker name seeds, and LLM prompt fragments) as a small
folder with a `manifest.yaml`. Point the pipeline at a pack for your subject
matter (a podcast, a beat you cover, a course) and transcription, name
attribution, and the run-level enrichment all get the same domain framing.

The bundled **`generic`** pack (neutral meeting / podcast, no specialized
vocabulary) is the zero-config default, so nothing here is required to run
Voxint. It earns its place only when your recordings share vocabulary, recurring
names, or a consistent style.

## Manifest

A pack is a folder containing `manifest.yaml`:

```yaml
name: newsroom            # required, unique across all resolvable packs
description: >
  Local-government reporting: council members, agencies, and beat jargon.
vocabulary:               # ASR/enhancement hints (names, jargon, acronyms)
  - Selectboard
  - Zoning Board of Appeals
  - CDBG
name_seeds:               # names the diarized speakers are likely to be
  - Maria Delgado
  - Councilor Okafor
prompt_fragments:         # advisory LLM framing; see "Fragment keys" below
  enhancement_context: >
    Municipal meeting audio. Preserve procedural wording; expand acronyms only
    when the speaker does.
  summary_context: >
    Summaries should foreground decisions, votes, and who is accountable.
  name_attribution_context: >
    Speakers address each other by title ("Councilor", "Chair"); use those.
corrections:                # deterministic literal fixes (see "Corrections" below)
  - id: zoning-board
    match: "zoom board"     # literal phrase (never a regex)
    replace: "Zoning Board" # exact canonical form you author
  - id: cdbg
    match: "C D B G"
    replace: "CDBG"
```

Only `name` is required (non-empty, unique). Every other field defaults to
empty. The manifest is parsed strictly: a wrong type (e.g. `vocabulary:` not a
list) is a loud configuration error, never a silent skip.

## Where packs live and how they resolve

Three sources are searched, in this precedence order when two packs claim the
same `name`:

1. the bundled **`generic`** pack, always available, zero config;
2. the configured **default pack**, `DOMAIN_PACK_PATH`, if set;
3. every direct child folder of **`DOMAIN_PACKS_DIR`** that contains a
   `manifest.yaml` (one child folder per pack, resolved by its manifest `name`).

Two *different* packs claiming the same `name` is a configuration error, raised
loudly (which pack did you mean?). The same pack reached by two sources (e.g.
`DOMAIN_PACK_PATH` also sits under `DOMAIN_PACKS_DIR`) is fine.

```bash
# One pack for everything:
DOMAIN_PACK_PATH=/data/voxint/packs/newsroom

# Or a library of named packs, selectable per run/folder:
DOMAIN_PACKS_DIR=/data/voxint/packs      # packs/newsroom, packs/podcast, packs/lectures, ...
```

These paths must resolve **inside the container**. The compose stack mounts
`MEDIA_ROOT` but does not mount an arbitrary `DOMAIN_PACK_PATH`. If the path
exists on the host but not in the container, submissions fail with "no domain
pack manifest at ...". Either volume-mount the packs directory in your compose
override, or place the packs under `MEDIA_ROOT` where the existing mount
reaches them.

## Selecting a pack for a run

When a run is created, its pack is resolved once with this precedence:

1. an **explicit pack name** supplied at submit time, else
2. the pack mapped to the **deepest watched folder** that contains the media
   (longest-ancestor wins, compared on path components, so `/audio/pod` never
   matches a file under `/audio/podcasts`), else
3. the **default pack** (`DOMAIN_PACK_PATH`, else `generic`).

Uploads and URL fetches have no watched-folder path, so they take the default
pack unless an explicit name is given. An explicit or mapped name that does not
resolve is a configuration error the operator sees. The pipeline never silently
substitutes `generic`, which would produce plausible-but-inconsistent output.

### Operator surface

The **default pack** (`DOMAIN_PACK_PATH`) and the **named-pack library**
(`DOMAIN_PACKS_DIR`) are the shipped operator controls: with a default pack set,
every run picks it up and the pack now shapes more of the pipeline than before
(see "What a pack shapes"), and each run **freezes** the pack it used (below).

Per-**folder** assignment (`{media_folder → pack_name}`) is editable in the
console (issue #63): the setup wizard's media step and **Settings → Media
folders** both host a folder browser where each registered folder gets a domain-
pack `<select>` (a "Default" leaves it unmapped). The options come from
`available_domain_packs` (the bundled `generic`, any `DOMAIN_PACK_PATH`, and each
named pack under `DOMAIN_PACKS_DIR`), so populate those to give the picker
choices. A per-**submission** pack override remains a backend capability
(`submit_media_item(..., domain_pack_name=...)`), not yet a console control.

## The frozen snapshot (why editing a manifest never rewrites history)

At submit time the resolved pack's **content** (name, description, vocabulary,
name seeds, prompt fragments, and corrections) is stamped **write-once** onto the run
(`pipeline_runs.domain_pack`, a JSON column added in migration 0017). Every stage
that consumes pack content reads *that snapshot*, not the live manifest on disk:

- the **pipeline worker** (transcription vocabulary + enhancement framing),
- the **offline name producer**, and
- the **run-level asset producers** (summary / topics / entity mentions).

Consequences that matter for correctness:

- Editing or deleting a pack's `manifest.yaml` **after** a run was submitted does
  **not** change that run's results. Transcription and the enrichment that reads
  it hours later always see the exact pack the audio was transcribed with.
- Two runs submitted under different packs differ naturally; that is expected,
  not drift.
- A run created before this feature (`NULL` snapshot) reproduces the prior
  global-pack behavior (the current default pack).
- A corrupt snapshot (only reachable by out-of-band database tampering, since the
  snapshot is Voxint's own validated round-trip) degrades to the default pack
  with a warning rather than wedging the run.

## What a pack shapes (fields and their single consumers)

Each piece of pack content has **one** documented consumer. Prompt fragments are
**never concatenated**: a key either has its single consumer or it is unused.

| Pack field | Consumer | Effect |
|---|---|---|
| `vocabulary` | ASR (Whisper `initial_prompt`) **and** the enhancement prompt | Biases transcription toward domain terms; also rendered as a "Domain vocabulary" line appended to `enhancement_context`. |
| `name_seeds` | Offline name producer | Boosts scoring for names the pack expects; hashed into the producer's idempotency signature (changing seeds re-runs it). |
| `prompt_fragments.enhancement_context` | Transcript-enhancement system prompt | ASR/enhancement framing (tone, what to preserve vs. fix). The vocabulary line is appended here. |
| `prompt_fragments.summary_context` | Run-asset LLM producer system prompt | Domain framing for the run summary, topics, and entity mentions. |
| `prompt_fragments.name_attribution_context` | Transcript-enhancement name-hint pass | A second labeled advisory block on the call that harvests speaker-name hints (e.g. anchoring a recurring host or a titled speaker). |
| `corrections` | Deterministic corrector (issue #81, inside `enhance_match`) | Literal substitutions applied to segment text and frozen per run. Empty in `generic`, a byte-preserving no-op. See "Corrections" below. |

Fragments are **advisory**: each is fenced so a pack can *guide* the model but
never override the strict reply schema, and an absent fragment leaves the prompt
**byte-for-byte unchanged**. Unknown keys in `prompt_fragments` are simply
carried in the snapshot and ignored (no consumer reads them).

## Corrections (deterministic literal substitutions)

A pack's `corrections:` list declares **deterministic, offline literal
substitutions** that fix recurring transcription mis-hears for your domain:
`zoom board → Zoning Board`, `C D B G → CDBG`. They run with **no model and no
prompt**, so they are reproducible, fast, and cannot translate, drop filler, or
obey an instruction hidden in the audio. They **complement** the optional LLM
enhancement rather than replacing it, and (like the rest of the pack) they are
frozen onto each run's snapshot, so editing the manifest never rewrites history.

> Distinct from the review console's **manual corrections**: those are per-segment
> edits an operator makes while reviewing one transcript (`corrected → enhanced →
> raw`). A pack's `corrections:` are declared rules applied automatically to every
> run under that pack.

Each rule:

```yaml
corrections:
  - id: zoning-board      # stable, operator-readable; unique within the pack
    match: "zoom board"   # the literal phrase to find (NOT a regex)
    replace: "Zoning Board"
    case_sensitive: true  # default true
    whole_word: true      # default true
```

Semantics (validated strictly at load, before a run is submitted, so a malformed
`corrections:` is a loud configuration error, never a silent skip):

- **Literal only.** `match` is a literal phrase; regex metacharacters (`.`, `*`,
  …) match themselves. There is no regex in v1.
- **`id`, `match`, `replace` are required, non-empty, and NUL-free.** `replace`
  is an **exact literal**: its casing is what you author and is **not** inherited
  from the matched text (`selectboard → Selectboard` turns `SELECTBOARD` into
  `Selectboard`, not `SELECTBOARD`).
- **`case_sensitive` and `whole_word` default `true`**, the conservative posture
  for domain terms. Whole-word uses an explicit boundary rule where **apostrophes,
  hyphens, and combining marks count as part of a word**, so an `it → IT` rule
  never fires inside `it's` and a rule never splits a combining-mark grapheme
  (`Zoë`).
- **No invisible characters.** `id`, `match`, and `replace` reject control,
  format, and surrogate code points, so a stray zero-width space or a block-scalar
  trailing newline is a loud error, never a silently dead rule.
- **Ids are unique**, and the set must be **idempotent**: no rule's `replace` may
  contain any rule's `match` in a way that would let that rule fire again
  (evaluated with that rule's own case/whole-word flags). So `zoom board → Zoning
  Board` is fine, but a chain like `a → b` together with `b → c` is rejected:
  split the rules or change a replacement/flag.
- **Bounds** (exceeding one rejects the pack, never truncates): at most **256
  rules**, `match` ≤ **256** code points, `replace` ≤ **512** code points, and the
  corrections payload ≤ **128 KiB**.

### Authoring corrections in the console (#84)

You do **not** have to hand-edit a `manifest.yaml` to add a correction. **Settings
→ Corrections** is a list editor for your own rules: add, edit, remove, and
reorder them, set *Match case* / *Whole word only*, and save. Leave the id blank
and one is generated from your `match`. Every rule is checked the **same way** a
pack's `corrections:` are (the bounds, invisible-character rejection, unique ids,
and idempotence above), with plain-language errors that point at the offending
row, so a bad rule is refused when you save it, not when a run fails.

These console rules live per **deployment** (not in a pack file), so they **survive
pack upgrades** and apply on top of **whichever** pack a run resolves to. At submit
time they are **unioned onto the selected pack's own corrections** and frozen into
that run's snapshot (pack rules first, then yours) exactly like a pack's rules,
so the corrector applies them and the review console shows their provenance (#83)
with no difference. Editing them affects your **next** run, never one already
submitted. If one of your rules would collide with the selected pack's own rules
(a duplicate id, or a replacement that would re-fire another rule), the save is
refused with the reason.

Editing a pack's `manifest.yaml` directly is still the way to ship corrections
*inside a shareable pack*; the console editor is for a single deployment's own
recurring fixes.

### How the rules apply (the corrector engine, #81)

A pure, versioned engine (`CORRECTOR_VERSION`) turns a segment string plus a rule
set into corrected text and a structured `{id, from, to, span}` trace. Its
behavior is frozen:

- **One left-to-right, non-cascading pass.** Matching runs against the original
  segment text; a replacement's own characters are never re-examined. So a
  `a → ba` rule and a `ba → Z` rule applied to `a` give `ba`, not `Z`.
- **Overlapping matches resolve leftmost-longest**, with the manifest order (the
  order rules appear in `corrections:`) as the final tie-break. When two rules
  could fire at the same spot, the one starting earliest wins; ties go to the
  longer match, then to the earlier-listed rule.
- **The trace addresses the corrected text.** Each applied rule records the
  original phrase it replaced and a `[start, end)` span **in the final corrected
  string** (offsets shift as earlier replacements grow or shrink the text), so the
  console (#83) can highlight exactly what changed. A pass where no rule fires
  returns the text unchanged with an empty trace.
- **Idempotence holds for any pack that loaded**: the load-time
  replacement-contains-match check above is what guarantees `correct(correct(t)) ==
  correct(t)` for the sets the engine actually receives. (One documented edge is
  outside that guard: a second-pass match that spans a replacement *and* adjacent
  original text, e.g. `ab → x` with `xc → y` on `abc`; the single pass is still
  deterministic.)
- **A transformation that would grow a segment past the enhancement size limit is
  rejected whole**: the segment falls back to its pre-correction text rather than
  applying a partial or truncated result.

The bundled `generic` pack declares no corrections, so the default pipeline stays
byte-preserving until you author or select a pack with them.

### How corrections compose with enhancement (the dual pass, #82)

Corrections run **inside** the `enhance_match` stage, composed with the optional
LLM enhancement through a **raw-gated dual pass**, designed so the operator's
canonical form wins on terms genuinely spoken, without ever letting the LLM's
output launder a term the recording never contained:

1. **Rules run on the raw ASR text first**, for every segment, fixing the
   authoritative set of rule ids that matched *in the evidence*.
2. **LLM enhancement runs as usual** (if enabled): punctuation, casing, filler.
3. **Only the rules that matched raw are re-enforced on the LLM output.** A rule
   with no raw basis can never fire, so a surface the LLM *invents* is never
   corrected into a domain term and mistaken for operator-authored. If the LLM is
   off or its batch failed, this step is a no-op and the raw-pass result stands.

A matched rule applies to **all** its occurrences in the LLM output (including any
the LLM introduced by rephrasing). The amplified term is still the operator's own
declared canonical form; those spans are marked LLM-coordinate, not evidence.

**What is persisted (provenance).** Enhanced text and a correction trail are
written **only when the final text materially differs from the raw text**: a
no-op segment stores nothing and reads back byte-identical to raw. When it does
differ, three things are written to the segment:

- `enhanced_text`: the final composed text (also what exports and search read);
- `correction_trace`: either `[]` (no material correction) or the envelope
  `{"version", "input_base", "entries"}`, where `input_base` is `"raw"` (LLM
  off/failed) or `"llm"` (enforced on the LLM output) and `entries` is the
  `{id, from, to, span}` list from the engine above (empty when only the LLM
  changed the text);
- `corrector_version`: the `CORRECTOR_VERSION` the engine stamped. **`NULL`**
  marks a legacy pre-#82 `enhanced_text` (shown as "enhanced (unversioned)") or a
  segment with no persisted enhanced output; it is **never recomputed at read
  time**.

**Atomic and idempotent.** On every (re-)enhance the three columns are reset
together first, so a stale trace can never outlive the `enhanced_text` it
described; re-running with the rule removed clears the correction wholesale.

**Splitting is disabled on a materially-corrected segment.** Word-boundary
splitting (#59) reads the **stored** trace: a non-empty `entries` list means a
rule fired, so the segment renders whole rather than deriving children at offsets
that no longer match the corrected surface. (A purely LLM-enhanced segment, with
empty `entries` but changed text, is likewise unsplittable, via the enhanced-text
check.) This is the authoritative stored signal, not a re-diff of text, so it also
catches a correction that alters only outer whitespace.

### Seeing corrections in the review console (#83)

Everything above is invisible unless the operator can see it, so the review
console surfaces the provenance the pipeline persisted, at **read-time** and with
**no new migration**: it is reconstructed from the per-segment `correction_trace` + the
run's frozen `domain_pack` snapshot each time the page loads.

- **"corrected by domain pack" marker.** A corrected segment shows a marker
  **distinct from the "edited" badge**: "edited" is *your* change; this marker is a
  change the domain pack made automatically. Expanding it lists the exact rule(s)
  that fired on that segment as `match → replace`, with the pack name and rule id. A
  rule id present in the trace but missing from the snapshot stays visible as
  "unresolved rule `<id>`", never silently dropped.
- **Raw text, one action away.** The immutable raw ASR text for the segment can be
  revealed to compare against the corrected text, copied, or used to **reset the
  edit box to raw**. Reset **populates the edit box only**: it does not save; you
  still press Save, so the unsaved-edit discard protection is never bypassed.
- **Operator edit supersedes provenance.** Once you save your own text for a
  segment, the "corrected by domain pack" marker clears: the trace's spans described
  the *pipeline's* enhanced text, not your edit, so showing them against your text
  would be misleading.
- **Splitting stays honest.** A materially-corrected segment is unsplittable (above);
  the console says so in plain language rather than offering a cut that would fail.
- **"Declared but never fired" reconciliation.** A run-level panel lists **every**
  rule the pack declared and whether it materially fired, reconstructed by replaying
  the corrector over each segment's immutable `raw_text`:
  - **`applied`**: fired on one or more segments' raw text (with the count).
  - **`no_raw_match`**: matched no segment's raw text. Usually the recording simply
    didn't contain the term, or the term was **split across a pause** so no single
    segment held it whole. For cross-segment terms, declare them as `vocabulary`
    (biased at transcription time) rather than as a correction.
  - **`growth_rejected`**: the rule would fire but its raw transformation overflowed
    the enhancement growth ceiling, so the corrector skipped it (a guard against a
    runaway substitution).

**Read precedence and v1 boundaries.** The console reads a segment's text as
`corrected → enhanced → raw`, and provenance is version-gated: a trace recorded by a
different `corrector_version` than the console reads is shown as **"unavailable"**
rather than replayed with mismatched semantics. Two cases are **honest, documented
v1 gaps** (see the design report §6/§12-F5): growth rejection during the
**LLM-enforcement** pass (its deciding input is not persisted) and exact
**cross-segment** detection are not exhaustively computed; steer such terms to pack
`vocabulary`. Corrections are **literal substitutions only**; regex is not supported
in v1.

## Vocabulary precedence: pack vs. custom vocabulary

Operators can also add custom vocabulary in the setup wizard (per deployment).
For a run, the **effective vocabulary is the pack's words first, then the
operator's custom words appended**, with duplicates removed on
first-occurrence-wins:

```text
effective_vocabulary = dedup_order_preserving(pack.vocabulary + custom_vocabulary)
```

So a pack term always appears before a custom term, and listing the same word in
both keeps only the pack's (earlier) position. The combined list biases Whisper's
`initial_prompt` and is rendered into the enhancement prompt's "Domain
vocabulary" line.

## See also

- [docs/operations.md](operations.md): the `.env` knobs (`DOMAIN_PACK_PATH`, `DOMAIN_PACKS_DIR`).
- [docs/architecture.md](architecture.md): the `pipeline_runs.domain_pack` snapshot column and
  the stages that read it.
- [docs/onboarding.md](onboarding.md): custom vocabulary in the setup wizard.
