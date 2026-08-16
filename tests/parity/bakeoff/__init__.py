"""Whisper Metal ASR engine bakeoff (issue #33) — shared measurement code.

This package holds the pieces the bakeoff's baseline generator (``tools/``) and
its parity harness (``tests/parity/``) both depend on, so a single frozen
definition backs every number the pre-registered gate reports
(``docs/gpu-contracts.md``, "Whisper Metal ASR engine (issue #33) —
pre-registered bakeoff gate").

Currently exposes the frozen text normalizer (:mod:`.normalize`); the corpus
manifest, baseline references, and scoring harness land alongside it in later
slices.

Tools import it by putting the repo root on ``sys.path`` first::

    sys.path.insert(0, str(REPO))
    from tests.parity.bakeoff.normalize import normalize_text
"""
