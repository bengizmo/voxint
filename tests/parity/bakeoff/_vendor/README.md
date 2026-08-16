# Vendored code (bakeoff)

Byte-identical copies of upstream releases, kept in-tree so the whisper Metal
ASR bakeoff (issue #33) has a **frozen denominator** that cannot move under a
dependency resolution. Everything here is sha-pinned in `provenance.json` and
excluded from `ruff` (its integrity check is the recorded digest, not lint).

## `openai_whisper_normalizers/`

The OpenAI Whisper text normalizers (`EnglishTextNormalizer` and friends),
vendored **verbatim** from `openai/whisper` at the commit recorded in
`provenance.json`. This is the community-standard normalizer for Whisper WER, so
it is the correct apples-to-apples denominator for a Whisper-vs-Whisper,
English-only bakeoff. Voxint code reaches it only through the wrapper
`tests/parity/bakeoff/normalize.py` (`normalize_text`, `NORMALIZER_VERSION`) —
never import the vendored package directly.

Why vendor instead of depending on `openai-whisper`: that package pulls Torch,
tiktoken, numba, and more — heavy runtime the normalizer itself does not need.
The only real runtime deps are `more-itertools` (on the number path) and `regex`
(imported by `basic.py`, though `EnglishTextNormalizer` uses stdlib `re`); both
are exact-pinned in the `parity` extra of the root `pyproject.toml`.

### Rules

- **Never hand-edit** a vendored file. It breaks the digest and voids the freeze.
- The `.py` files carry **no added license header** (that would make them
  non-verbatim); the upstream MIT license ships beside them as
  `LICENSE.openai-whisper` and is hashed too.
- Refreshing to a new upstream commit is a **normalizer change**: update
  `upstream_commit` + every digest in `provenance.json`, re-record the golden
  input→output vectors in `tests/contracts/test_bakeoff_normalizer.py`, and
  regenerate any affected baselines — together, in one deliberate change.
- Regeneration commands live in `provenance.json` (`regeneration`).

### What actually freezes the output

The API alone does not. The freeze is the union of: the vendored byte digests,
the exact-pinned `more-itertools` / `regex`, the recorded runtime fingerprint
(Python minor + `unicodedata` Unicode version — the likeliest exotic-character
drift source), and the golden vectors asserted by the contract test.
