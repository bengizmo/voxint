# synthdetect eval container

Maintainer evaluation runtime for synthetic-speech (audio deepfake) detection,
Milestone 1 of issue #144. This is not a shipped Voxint service. It packages the
pinned runtime that `tools/synthdetect_infer.py` runs inside to produce raw-score
journals, which the host scorer `tools/synthdetect_eval.py` reads on the CPU.

## What is here

- `Dockerfile.eval` builds the CUDA 11.8 / cuDNN8 image with the pinned reference
  runtime (torch cu118 plus fairseq at a frozen commit). The default detector
  candidate, `w2v2-aasist`, loads its wav2vec2-XLS-R front-end through fairseq, so
  the runtime is part of the inference space: a different runtime is a different
  space even with identical weights.
- `requirements.eval.txt` lists the non-torch pins for review and CI.
- `provenance.eval.json` records the runtime identity, the canonicalization id,
  the scoring polarity, and the CANDIDATE weight and commit pins. A contract test
  (`tests/contracts/test_synthdetect_container.py`) binds it to the Dockerfile,
  the requirements file, the registry, and the runner.

## What is deliberately absent

- **No weights.** The detector weights are CANDIDATE, license-gated, and live on
  maintainer storage. They are mounted read-only at run time under
  `SYNTHDETECT_WEIGHTS_DIR`, never baked, so the image redistributes nothing and
  never freezes a checkpoint the build did not verify.
- **No server.** The container runs a one-shot CLI, so it declares no port and no
  health check. The entrypoint is the inference runner; the maintainer passes its
  subcommand and arguments.

## Building and running (S2b, maintainer GPU)

The image is a specification until S2b builds and validates it on a GPU. Freeze
the fairseq commit, base-image digest, and weight shas together (see the
`s2b_freeze_checklist` in `provenance.eval.json`), then build with the frozen
commit:

```
docker build -f services/synthdetect/Dockerfile.eval \
  --build-arg FAIRSEQ_COMMIT=<frozen-commit> \
  -t voxint-synthdetect-eval:local services/synthdetect
```

The build refuses to proceed without `FAIRSEQ_COMMIT`, so an unpinned runtime
cannot be built by accident. Mount the repository and the weights, then run the
inference runner or the source-verification pass:

```
docker run --rm --gpus '"device=2"' \
  -v "$PWD":/repo -w /repo -v /path/to/weights:/weights:ro \
  voxint-synthdetect-eval:local \
  run --manifest <m.json> --corpus-root <dir> --out <journal.jsonl> --split eval
```

`verify-sources` computes a dated weight receipt (real sha256 and byte size per
file) and compares it to the registry pins; it never rewrites the registry, so
freezing a CANDIDATE pin stays a reviewed diff.

## Corpus audio format

The runner does not resample. Corpus clips are canonicalized once at acquisition
to `pcm-s16le-mono-16000-v1` (16 kHz mono signed-16-bit little-endian PCM, no
dither, normalization, or trim), and the manifest `sha256` is the digest of the
PCM `data` payload bytes only. The runner asserts that format, hashes the
payload, and fails closed on a mismatch, so corpus identity never depends on this
container's decoder. See `docs/gpu-contracts.md` for the full pre-registration.
