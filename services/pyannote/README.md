# Voxint pyannote service

Speaker diarization. Contract: [docs/gpu-contracts.md](../../docs/gpu-contracts.md) —
`POST /v1/diarize`, `GET /healthz`, port **8024**.

## Model policy

**pyannote/speaker-diarization-3.1** on **pyannote.audio 3.1.1** — a deliberate
pin, not a lag: the 4.x pipeline (community-1) silently rejects the
`clustering.threshold` / `min_cluster_size` hyperparameters this service tunes.

**Weights are HF-gated and not in the image.** Before first start:

1. Accept the conditions of **both** [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
2. Supply `HF_TOKEN` (a read token for that account). Weights download to
   `/app/models` on first startup — mount a volume there to persist them.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MEDIA_ROOT` | `/data/media` | Shared media volume (mount read-only) |
| `HF_TOKEN` | — | **Required** for the gated weights |
| `DIARIZER_MODEL_NAME` | `pyannote/speaker-diarization-3.1` | Pipeline id (policy pin) |
| `PYANNOTE_CLUSTERING_THRESHOLD` | `0.55` | Below the ~0.70 default — the default under-clusters quiet audio to 0 speakers |
| `PYANNOTE_CLUSTERING_MIN_SIZE` | `10` | Min cluster size |
| `PYANNOTE_MIN_DURATION_OFF` | `0.6` | Same-speaker gaps shorter than this merge in post-processing |
| `PYANNOTE_SEGMENTATION_BATCH_SIZE` | `8` | Verified good for 12 GB-class GPUs |
| `PYANNOTE_EMBEDDING_BATCH_SIZE` | `12` | Verified good for 12 GB-class GPUs |
| `PYANNOTE_SEGMENTATION_STEP` | `0.5` | Larger than the 0.1 default → sustained GPU load |
| `MAX_PENDING_REQUESTS` | `8` | Admission bound; beyond it → retryable 503 |
| `PORT` | `8024` | Listen port |

## Image matrix

Python 3.10 · CUDA 11.8 runtime (cuDNN 8) · torch/torchaudio 2.5.0+cu118 ·
pyannote.audio 3.1.1 · huggingface_hub 0.23.4 (pin — 0.26+ removes the
`use_auth_token=` kwarg pyannote 3.1.1 uses). VRAM: ~1–2 GB loaded.

```bash
docker pull ghcr.io/bengizmo/voxint-pyannote:0.3.0  # prebuilt release image
docker build -t voxint-pyannote services/pyannote   # …or build from source
docker run --rm --gpus all -p 127.0.0.1:8024:8024 \
  -e HF_TOKEN=hf_yourtoken \
  -v /path/to/media:/data/media:ro \
  -v pyannote-models:/app/models voxint-pyannote
curl -s localhost:8024/healthz
```
