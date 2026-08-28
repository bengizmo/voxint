# ADR 0004: Claim lifecycle

> **Status:** Accepted and partially implemented (Console 2.0 P0a, issue #150).
> PR #242 (issue #156) landed claim-token verification on the editor detail page
> and the `no-store` generalization to `/media/{uuid}/editor*` paths. The full
> claim lifecycle (heartbeat, release) lands in a later phase.

## Context

Review edits are guarded by a claim: the console holds a per-claim token that is
both a soft lock (one operator edits a run at a time) and the CSRF defense for
claim-gated mutations (the token rides in the URL as `?token=`, so a forged
cross-site POST cannot supply it). The editor keeps this model, but changes the
addressing from run to media, and it wants multi-tab use to be safe rather than a
self-inflicted lock conflict.

Two current behaviors constrain the design. The claim token appears in the URL,
so any token-bearing page must never be written to a browser or proxy cache, and
that `no-store` handling is currently keyed to the `/review` path prefix. And a
claim acquired on a GET would mean merely opening a page locks a run, which breaks
multi-tab browsing and back/forward navigation.

## Decision

1. **Claims are `(media, run)`-scoped.** A claim names the specific run being
   edited, not just the media item, so switching runs (ADR 0003) is a distinct
   claim rather than an ambiguous reuse.

2. **Never acquire a claim on GET.** Opening the editor, choosing a run, and
   scrolling acquire nothing. A claim is acquired on the first edit intent via a
   POST, so navigation and read-only viewing are free of locks.

3. **Same-operator claims are reused, not rotated.** A second tab or a repeated
   edit intent from the same operator reuses the existing claim rather than
   fighting it, so multi-tab editing does not deadlock the operator against
   themselves.

4. **A bounded heartbeat renews an active claim**, and a lost or expired claim
   surfaces a clear read-only state on a 409 rather than silently discarding an
   edit.

5. **`no-store` handling is generalized before any token-bearing `/media` page
   ships.** The cache-control that currently keys on `/review` moves to cover
   every token-bearing response, so the editor's token never leaks into a cache.

## Consequences

- Opening the editor is cheap and side-effect free, which is what makes the
  run-selection read (ADR 0003) and ordinary browser navigation safe.
- Multi-tab use by one operator works, because same-operator reuse means the tabs
  share a claim instead of racing for it.
- A stale claim fails visibly: the operator sees a read-only banner and a 409, not
  a silently dropped edit. Honest failure over a masked one.
- Undo for enroll and merge is modeled as compensating rulings on the append-only
  ledger, not row deletion, so the claim never needs to reach back into committed
  history. Generalized undo is deferred.
