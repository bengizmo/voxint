"""Offline scoring harness: DB-free quality-measurement cores + file-based CLIs.

Nothing in this package touches the database, Celery, or settings. The modules
are pure (dict/dataclass in, dataclass out); the ``voxint score`` CLI adapters
in :mod:`voxint.harness.score_cli` read JSONL/JSON files and write reports.
Contracts are documented in ``docs/harness.md``.
"""
