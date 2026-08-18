# Domain packs

A **domain pack** supplies the domain-specific knowledge the pipeline consumes —
ASR vocabulary hints, speaker name seeds, and LLM prompt fragments — as a small
folder with a `manifest.yaml`. Point the pipeline at a pack for your subject
matter (a podcast, a beat you cover, a course) and transcription, name
attribution, and the run-level enrichment all get the same domain framing.

The bundled **`generic`** pack (neutral meeting / podcast, no specialized
vocabulary) is the zero-config default, so nothing here is required to run
Voxint — it earns its place only when your recordings share vocabulary, recurring
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
```

Only `name` is required (non-empty, unique). Every other field defaults to
empty. The manifest is parsed strictly: a wrong type (e.g. `vocabulary:` not a
list) is a loud configuration error, never a silent skip.

## Where packs live and how they resolve

Three sources are searched, in this precedence order when two packs claim the
same `name`:

1. the bundled **`generic`** pack — always available, zero config;
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

## Selecting a pack for a run

When a run is created, its pack is resolved once with this precedence:

1. an **explicit pack name** supplied at submit time, else
2. the pack mapped to the **deepest watched folder** that contains the media
   (longest-ancestor wins, compared on path components — `/audio/pod` never
   matches a file under `/audio/podcasts`), else
3. the **default pack** (`DOMAIN_PACK_PATH`, else `generic`).

Uploads and URL fetches have no watched-folder path, so they take the default
pack unless an explicit name is given. An explicit or mapped name that does not
resolve is a configuration error the operator sees — the pipeline never silently
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
`available_domain_packs` — the bundled `generic`, any `DOMAIN_PACK_PATH`, and each
named pack under `DOMAIN_PACKS_DIR` — so populate those to give the picker
choices. A per-**submission** pack override remains a backend capability
(`submit_media_item(..., domain_pack_name=...)`), not yet a console control.

## The frozen snapshot (why editing a manifest never rewrites history)

At submit time the resolved pack's **content** — name, description, vocabulary,
name seeds, and prompt fragments — is stamped **write-once** onto the run
(`pipeline_runs.domain_pack`, a JSON column added in migration 0017). Every stage
that consumes pack content reads *that snapshot*, not the live manifest on disk:

- the **pipeline worker** (transcription vocabulary + enhancement framing),
- the **offline name producer**, and
- the **run-level asset producers** (summary / topics / entity mentions).

Consequences that matter for correctness:

- Editing or deleting a pack's `manifest.yaml` **after** a run was submitted does
  **not** change that run's results — transcription and the enrichment that reads
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
**never concatenated** — a key either has its single consumer or it is unused.

| Pack field | Consumer | Effect |
|---|---|---|
| `vocabulary` | ASR (Whisper `initial_prompt`) **and** the enhancement prompt | Biases transcription toward domain terms; also rendered as a "Domain vocabulary" line appended to `enhancement_context`. |
| `name_seeds` | Offline name producer | Boosts scoring for names the pack expects; hashed into the producer's idempotency signature (changing seeds re-runs it). |
| `prompt_fragments.enhancement_context` | Transcript-enhancement system prompt | ASR/enhancement framing (tone, what to preserve vs. fix). The vocabulary line is appended here. |
| `prompt_fragments.summary_context` | Run-asset LLM producer system prompt | Domain framing for the run summary, topics, and entity mentions. |
| `prompt_fragments.name_attribution_context` | Transcript-enhancement name-hint pass | A second labeled advisory block on the call that harvests speaker-name hints (e.g. anchoring a recurring host or a titled speaker). |

Fragments are **advisory**: each is fenced so a pack can *guide* the model but
never override the strict reply schema, and an absent fragment leaves the prompt
**byte-for-byte unchanged**. Unknown keys in `prompt_fragments` are simply
carried in the snapshot and ignored (no consumer reads them).

## Vocabulary precedence: pack vs. custom vocabulary

Operators can also add custom vocabulary in the setup wizard (per deployment).
For a run, the **effective vocabulary is the pack's words first, then the
operator's custom words appended**, with duplicates removed on
first-occurrence-wins:

```
effective_vocabulary = dedup_order_preserving(pack.vocabulary + custom_vocabulary)
```

So a pack term always appears before a custom term, and listing the same word in
both keeps only the pack's (earlier) position. The combined list biases Whisper's
`initial_prompt` and is rendered into the enhancement prompt's "Domain
vocabulary" line.

## See also

- `docs/operations.md` — the `.env` knobs (`DOMAIN_PACK_PATH`, `DOMAIN_PACKS_DIR`).
- `docs/architecture.md` — the `pipeline_runs.domain_pack` snapshot column and
  the stages that read it.
- `docs/onboarding.md` — custom vocabulary in the setup wizard.
