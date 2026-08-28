"""The one sanctioned core -> plugin import site (issue #137).

:data:`BUILTIN` lists every plugin class baked into the app image. It is the ONLY
place in ``voxint`` where core imports a plugin package; the import-direction
guard test asserts nothing else does. All three consult models rejected Python
entry points for discovery (a public repo has no untrusted-plugin threat model to
justify the indirection, and an explicit list is greppable and fails loud when a
class is missing), so registration is this hand-maintained tuple.

Empty until the conversions land: #139 appends the translation plugin, #140
semantic search, #141 LLM enrichment. Order here does not matter — the registry
sorts by ``manifest.id`` for a deterministic route/task order.
"""

from __future__ import annotations

from voxint.plugins.base import VoxintPlugin
from voxint.plugins.synthdetect import SynthdetectPlugin

BUILTIN: tuple[type[VoxintPlugin], ...] = (SynthdetectPlugin,)
