# Voxint whisper service

faster-whisper ASR with hallucination soft-tagging. Contract:
[docs/gpu-contracts.md](../../docs/gpu-contracts.md) — `POST /v1/transcribe`,
`GET /healthz`, port **8022**.

## Model policy

Default **large-v2** at int8, baked into the image at build time. large-v3 and
large-v3-turbo trade quiet-audio robustness for speed and produce repetition
hallucinations on real-world recordings; the repetition detector soft-tags
those (`suspect: true`) rather than dropping text, but large-v2 produces far
fewer of them. `WHISPER_MODEL` overrides the pin at your own risk (a
non-baked model downloads at startup and needs network + writable cache).

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MEDIA_ROOT` | `/data/media` | Shared media volume (mount read-only) |
| `WHISPER_MODEL` | `large-v2` | faster-whisper model id (policy pin) |
| `WHISPER_DOWNLOAD_ROOT` | `/app/.cache/whisper` | Model cache; the image bakes large-v2 here so startup needs no network |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `int8` | CTranslate2 compute type |
| `BATCH_SIZE` | `16` | BatchedInferencePipeline batch size |
| `MAX_PENDING_REQUESTS` | `8` | Admission bound; beyond it → retryable 503 |
| `WHISPER_REPETITION_TAG` | `on` | Hallucination soft-tagging (off/0/false to disable) |
| `PORT` | `8022` | Listen port |

## Image matrix

Python 3.10 · CUDA 12.4.1 runtime (cuDNN 9, symlinked to cuDNN-8 names for
CTranslate2) · torch/torchaudio 2.1.1+cu121 · faster-whisper 1.2.1 ·
numpy 1.24.3. VRAM: ~1.5 GB (large-v2 int8) + batch overhead.

```bash
docker pull ghcr.io/bengizmo/voxint-whisper:0.1.0   # prebuilt release image
docker build -t voxint-whisper services/whisper     # …or build from source
docker run --rm --gpus all -p 127.0.0.1:8022:8022 \
  -v /path/to/media:/data/media:ro voxint-whisper
curl -s localhost:8022/healthz
```
