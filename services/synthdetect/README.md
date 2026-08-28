# Synthdetect Service

Standalone GPU service for synthetic speech detection. Scores audio intervals
using a fine-tuned w2v2-AASIST model (XLS-R 300M front end, AASIST back end)
and returns raw logits per interval. Calibration is applied downstream by the
plugin task, not here.

## Quick start

Stage weights (not in git), build, run:

```bash
mkdir -p weights
cp ~/synthdetect-weights/finetuned_aasist.pth weights/
cp ~/synthdetect-weights/xlsr2_300m.pt weights/

docker build -t voxint-synthdetect:dev .
docker run --gpus device=0 \
  -v /path/to/media:/data/media:ro \
  -p 8025:8025 \
  voxint-synthdetect:dev
```

Model loading takes 10-15 seconds. The healthcheck has a 120-second start
period.

## API

**GET /healthz** returns service status, inference space identity, and
windowing parameters. `model_loaded: true` means the model is ready.

**POST /v1/score** accepts a JSON body:

```json
{
  "path": "relative/to/media_root.wav",
  "intervals": [
    {"start_seconds": 0.0, "end_seconds": 10.5}
  ]
}
```

Returns one result per interval. Each result has either `raw_score` (higher
means more likely synthetic) or `skip_reason` (interval too short to score).

## Inference space

`w2v2-aasist-df-m2-s0e11`: wav2vec2 XLS-R + AASIST, initialized from the DF
checkpoint, fine-tuned on Piper + Chatterbox TTS (M2, seed 0, epoch 11).

Windowing: 64,600-sample (4.0375s) non-overlapping windows, repeat-pad for
short clips, mean-pool raw logits across windows. Intervals shorter than
8,000 samples (0.5s) are skipped.

## Weights

Not committed to git. Shas pinned in `provenance.json`, `scoring.py`, and
the Dockerfile ARG defaults. The reconstituted checkpoint combines XLS-R
base weights with the fine-tuned AASIST backend. See `provenance.json` for
full lineage.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `SYNTHDETECT_WEIGHTS_DIR` | `/app/weights` | Weights directory |
| `SYNTHDETECT_AASIST_CHECKPOINT` | `finetuned_aasist.pth` | AASIST checkpoint filename |
| `SYNTHDETECT_XLSR_CHECKPOINT` | `xlsr2_300m.pt` | XLS-R checkpoint filename |
| `SYNTHDETECT_DEVICE` | `cuda:0` | Torch device |
| `MEDIA_ROOT` | `/data/media` | Audio file root |
| `PORT` | `8025` | HTTP port |
