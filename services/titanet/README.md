# Voxint titanet service

Speaker embeddings. Contract:
[docs/gpu-contracts.md](../../docs/gpu-contracts.md) (`POST /v1/embed`,
`GET /healthz`, port **8021**).

## Model & embedding space

NVIDIA NeMo **TitaNet-Large** (`nvidia/speakerverification_en_titanet_large`),
192-dim, weights baked into the image (ungated). Embedding space id:
**`titanet-large-v2`**. The id versions the model *and* the preprocessing chain
(noise reduction → −16 LUFS → peak 0.95 → L2). Change either one and you get a
new space id; vectors from different spaces must never be compared.

Windows under 1.0 s skip as `too_short`; windows below the SNR threshold skip as
`low_snr`. Skipped windows return `embedding: null` with the reason, and
positional alignment with the request is always preserved.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MEDIA_ROOT` | `/data/media` | Shared media volume (mount read-only) |
| `TITANET_SNR_THRESHOLD_DB` | `5.0` | Windows below this SNR skip as `low_snr` |
| `TITANET_WINDOW_CAP_SECONDS` | `30.0` | Windows longer than this are embedded as contiguous <= cap sub-windows whose unit vectors are mean-pooled and re-normalized; must be >= 1.0 |
| `MAX_PENDING_REQUESTS` | `8` | Admission bound; beyond it → retryable 503 |
| `PORT` | `8021` | Listen port |

## Image matrix

Python 3.10 · CUDA 11.8 devel (cuDNN 8; NeMo's youtokentome compiles at build) ·
torch/torchaudio 2.1.0+cu118 · nemo_toolkit[asr] 1.22.0 · huggingface-hub
0.23.5 (repinned after NeMo's resolver) · pyloudnorm 0.1.0 · noisereduce 2.0.1.
VRAM: ~1 GB loaded.

```bash
docker pull ghcr.io/bengizmo/voxint-titanet:0.4.0   # prebuilt release image
docker build -t voxint-titanet services/titanet     # …or build from source
docker run --rm --gpus all -p 127.0.0.1:8021:8021 \
  -v /path/to/media:/data/media:ro voxint-titanet
curl -s localhost:8021/healthz
```
