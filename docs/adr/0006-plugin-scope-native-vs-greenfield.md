# ADR 0006: Plugin scope: existing features stay native, framework for greenfield only

> **Status:** Accepted (epic #136). The plugin framework (#137, #138) is merged
> and ships dormant with an empty registry. The three planned conversions of
> existing features (#139 translation, #140 semantic search, #141 LLM enrichment)
> are cancelled. This ADR records why the wired-in framework carries no builtins.

## Context

The plugin framework was built (#137) and wired into every core seam (#138) to
turn Voxint's optional features into self-contained, flag-gated plugins under
`src/voxint/plugins/`: maintainable on parallel branches, removable with a kill
switch, and a foundation for third-party plugins later. The plan was to convert
translation, semantic search, and LLM enrichment, then reuse that shape for new
features.

Converting the first feature, translation, exposed a boundary the plan had not
accounted for. The load-bearing rule is that core never imports a plugin package:
the one sanctioned import site is `plugins/discover.py`, grep-enforced by
`tests/contracts/test_plugin_framework.py`. Translation's render helpers are
called directly by the shared review-console pages: the run detail page, the
read-only transcript page and its interleaved view, the review stepper, and the
language-filtered exports. A render helper that a core page calls cannot move into
a plugin, and neither can the modules it imports.

Reviewing all three planned conversions against that rule:

- **Translation** is woven into the run detail, transcript, review, and export
  renders. Only a task wrapper, a post-completion hook, and two templates could
  move; the logic stays in core.
- **LLM enrichment** is woven more deeply still: run-asset state renders into the
  run detail page, research into the speaker roster, the name producer writes
  through the core review substrate, and its flags sit inside the effective-flags
  validation.
- **Semantic search** is the one feature that extracts cleanly. It lives behind a
  standalone `/search` page with a self-contained flag validator and no coupling
  into the shared renders. It is, however, a core capability of the console.

Two of the three would become thin wrappers over code that stays in core, and the
third is a core feature. The framework's v1 hook surface (template-only panel and
settings-section contributions, a verify-only CSRF action with no token minting)
is correct for its intended use but would have to grow to force-fit the woven
features.

## Decision

1. **The three existing optional features stay native.** Translation, semantic
   search, and LLM enrichment remain core code. Issues #139, #140, and #141 are
   cancelled.

2. **The plugin framework stays, dormant.** The #137 and #138 work remains merged
   with an empty `BUILTIN` registry, so behavior is byte-identical and nothing an
   operator sees changes. There is no revert; the wiring is correct and waiting
   for a feature that fits it.

3. **Boundary rule for what becomes a plugin.** A feature that renders into the
   shared review-console pages (run detail, transcript, review, speaker roster)
   belongs in core. A feature with its own standalone surface, or one that runs
   after pipeline completion through a post-finalize hook, its own job lane, and
   its own page, is a good plugin candidate.

4. **The framework's audience is greenfield features.** The first real plugin is
   synthetic-speech detection, which fits the rule: its own model service, a
   post-completion job, and its own review surface rather than an edit interleaved
   into the transcript.

## Consequences

- A fully wired plugin framework ships with an empty registry and no conversions.
  This ADR is the reason a later reader finds that state instead of a confusing
  half-finished diff.
- The v1 hook surface is not grown on speculation. Panel and settings-section
  hooks stay template-only, rendering against core-populated context, and
  `PluginRouteDeps` keeps its single verify-only CSRF action with no minting
  helper. The features that would have needed richer hooks are staying native.
- Placement gets a one-question test for the next optional feature: does a core
  page render this feature's state? If yes, it stays in core. That saves the next
  contributor from repeating the translation investigation.
- The `VOXINT_PLUGINS_DISABLED` kill switch and the `voxint doctor` plugin surface
  remain, now scoped to greenfield plugins rather than converted features.
- The plugin author guide (#142) is written when the first greenfield plugin
  lands, using it as the worked example.
