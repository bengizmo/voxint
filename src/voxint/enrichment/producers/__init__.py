"""Enrichment producers: offline jobs that write draft claims via the #37 layer.

Each producer is a thin orchestration over pure extraction/scoring modules and
persists exclusively through :func:`voxint.enrichment.drafts.record_producer_run`.
Producers never write ``speakers``, ``speaker_assignments``, or
``adjudication_decisions`` — their output is suggestions for human review.
"""
