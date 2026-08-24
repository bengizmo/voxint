# ADR 0003: Editor run selection

> **Status:** Accepted (Console 2.0 P0a, issue #150). The `/media/{id}` editor and
> its backend endpoints land in a later phase (P3a/P3b).

## Context

The Console 2.0 editor opens on a media item (`/media/{id}`), but transcription
and review attach to a run (`pipeline_runs`), and one media item can have several
runs: an original, a requeue after a failure, a rerun with a different model or
speaker count. The old review flow was addressed by run, so the ambiguity never
surfaced. Addressing the editor by media forces a decision about which run the
editor shows, and how an operator reaches an older one.

The completeness guarantee also matters. The old two-step review flow walked the
operator through every unreviewed segment (a stepper) so nothing was skipped. The
editor replaces that flow. If the editor simply drops the operator into a free
scroll, the completeness guarantee regresses.

## Decision

1. **Default selection is the latest completed run.** Opening `/media/{id}` with
   no run specified selects the most recent run in a completed state. This is the
   run an operator almost always means.

2. **`?run=` overrides, backed by a version chooser.** An explicit `?run={id}`
   selects that run, and the editor surfaces a chooser listing the media item's
   runs (newest first, with status) so an older or alternate run is reachable
   without hand-editing the URL.

3. **Walk mode is the default entry for an unreviewed file.** When the selected
   run has unreviewed segments, the editor opens in verify-walk mode: the
   stepper's completeness guarantee survives inside the editor. A fully reviewed
   run opens in free scroll.

4. **Selection is a read.** Choosing or switching runs never mutates state and
   never acquires a claim (ADR 0004). It changes what the editor displays, nothing
   more.

## Consequences

- The common case (open the newest transcript) needs no run in the URL, and the
  uncommon case (inspect an older run) is one click in the chooser.
- The completeness guarantee is preserved by construction: an unreviewed file
  cannot be opened into a mode that lets the operator skip segments silently.
- Deep links to a specific run stay stable, because `?run=` names the run
  explicitly and the default only applies when the parameter is absent.
- Because selection is a pure read, back, forward, and refresh are safe, and the
  claim lifecycle (ADR 0004) can be reasoned about independently of navigation.
