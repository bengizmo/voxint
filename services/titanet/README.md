# Voxint titanet service

Speaker embeddings. Contract:
[docs/gpu-contracts.md](../../docs/gpu-contracts.md) (`POST /v1/embed`,
`GET /healthz`, port **8021**).

## Model & embedding space

NVIDIA NeMo **TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`),
192-dim, exported to an ONNX graph and run on ONNX Runtime. The images carry no
NeMo and no torch. The graph is baked into the image, sha256-pinned from the
[`titanet-onnx-v1`](https://github.com/bengizmo/voxint/releases/tag/titanet-onnx-v1)
asset release (see
[`tests/parity/fixtures/onnx/provenance.json`](../../tests/parity/fixtures/onnx/provenance.json)).
To build from source, first place the graph at `models/titanet-large.onnx`
(download the release asset, or re-export with `tools/export_titanet_onnx.py`);
the build fails on a sha256 mismatch.

Embedding space id: **`titanet-large-v2`**. The id versions the model *and* the
preprocessing chain (noise reduction, then -16 LUFS, then peak 0.95, then L2).
Change either one and you get a new space id; vectors from different spaces must
never be compared. The ONNX engine holds that id on measured equivalence with
the original CUDA reference; see the parity verdict in
[docs/gpu-contracts.md](../../docs/gpu-contracts.md).

Windows under 1.0 s skip as `too_short`; windows below the SNR threshold skip as
`low_snr`. Skipped windows return `embedding: null` with the reason, and
positional alignment with the request is always preserved.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MEDIA_ROOT` | `/data/media` | Shared media volume (mount read-only) |
| `TITANET_SNR_THRESHOLD_DB` | `5.0` | Windows below this SNR skip as `low_snr` |
| `EMBED_ENGINE` | `onnx` in the images | `onnx`, or `nemo` for the maintainer-only NeMo engine (needs torch and NeMo installed) |
| `TITANET_ONNX_PATH` | `/app/models/titanet-large.onnx` | The exported graph |
| `TITANET_ORT_PROVIDERS` | `CUDAExecutionProvider` in the CUDA image, `CPUExecutionProvider` otherwise | ONNX Runtime providers; the service fails at startup if a requested provider is unavailable rather than silently running on another device |
| `MAX_PENDING_REQUESTS` | `8` | Admission bound; beyond it → retryable 503 |
| `PORT` | `8021` | Listen port |

## Image matrix

**CUDA** (`Dockerfile`): Python 3.11 · CUDA 12.8.1 cuDNN runtime ·
onnxruntime-gpu 1.28.0 (CUDA execution provider).

**CPU** (`Dockerfile.cpu`, `-cpu` tag): Python 3.11 · multi-arch
(amd64 + arm64) · onnxruntime 1.28.0.

Both: numpy 1.24.3 · librosa 0.10.1 · pyloudnorm 0.1.0 · noisereduce 2.0.1. These
preprocessing pins are part of the embedding space definition. Contract tests
keep them identical across both images and keep the onnxruntime pin equal to the
version the parity harness measured.

Replace `X.Y.Z` with a release version (the `VOXINT_IMAGE_TAG` default in
`compose.yaml` is the current one):

```bash
docker pull ghcr.io/bengizmo/voxint-titanet:X.Y.Z   # prebuilt release image
docker build -t voxint-titanet services/titanet     # ...or build from source
docker run --rm --gpus all -p 127.0.0.1:8021:8021 \
  -v /path/to/media:/data/media:ro voxint-titanet
curl -s localhost:8021/healthz
```
