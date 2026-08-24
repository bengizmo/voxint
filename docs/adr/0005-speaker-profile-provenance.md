# ADR 0005: Speaker profile provenance

> **Status:** Accepted (Console 2.0 P0a, issue #150). The aggregation module and
> `speaker_profiles` provenance land in a later phase (P4).

## Context

Console 2.0 adds speaker overview pages: a roster with per-speaker counts (files,
minutes) and a profile page. The counts must reflect what the operator actually
decided, and the profile must combine manual entry with accepted enrichment
without losing track of which value came from where.

The raw tables do not answer these questions directly. Speakers are merged, and
later rulings override earlier assignments; a naive join over assignments would
double-count a split range or count a superseded assignment. Separately, a profile
field (a name, a role, a note) can originate from an operator typing it or from
the operator accepting a research-enrichment candidate, and a later reader needs
to tell those apart.

## Decision

1. **Aggregate from effective-resolution output, not raw joins.** Per-speaker
   counts and rosters read the resolver's effective attribution: canonicalized
   through merges, with later rulings overriding earlier assignments and
   split-range overrides counted once. A speaker's `verified` state is the
   currently-effective human assignment, not the existence of any historical one.

2. **`speaker_profiles` carries per-field provenance.** Each profile field records
   whether its value is manual or an accepted enrichment candidate. Accepting a
   candidate materializes the value onto the profile without erasing the draft
   candidate history it came from.

3. **No duplication of existing state.** `speakers.notes` is not copied onto the
   profile; the profile references rather than duplicates data that already has a
   home, so the two cannot drift.

## Consequences

- Roster and profile numbers match the operator's decisions, including after
  merges and split reassignments, because they read the same effective resolution
  the review console acts on rather than a parallel raw count.
- A profile field's origin is always answerable: manual or accepted enrichment,
  with the candidate trail intact for audit.
- Aggregation over effective resolution is more expensive than a raw join, so the
  overview queries get EXPLAIN-checked indexes in the phase that ships them; the
  correctness of the numbers is not traded for the speed of a wrong count.
- The confidence display (qualitative tiers plus a verified badge, with numbers
  behind a reveal) reads from this same effective state, and its later upgrade
  path (issue #114) does not change the aggregation contract.
