# GPU service smoke procedure

CPU-only contract tests (`uv run pytest tests/contracts`) prove the wire
schemas; this procedure proves the images actually load their models and
infer on a real GPU. Run it after any change to a `services/*/` Dockerfile,
requirements, or model-touching code. First verified end-to-end 2026-08-11 on
an RTX 3090.

## Setup

Any ~1 min 16 kHz mono WAV with at least two speakers works; an interview
clip is ideal (exercises diarization + per-speaker embeddings).

```bash
mkdir -p /tmp/voxint-smoke/media
ffmpeg -y -i your-interview.wav -t 60 -ac 1 -ar 16000 /tmp/voxint-smoke/media/smoke.wav

docker build -t voxint-whisper:dev  services/whisper
docker build -t voxint-pyannote:dev services/pyannote   # needs models/*.bin first (see services/pyannote/README.md)
docker build -t voxint-titanet:dev  services/titanet
```

## whisper

```bash
docker run -d --name smoke-whisper --gpus all -p 127.0.0.1:18022:8022 \
  -v /tmp/voxint-smoke/media:/data/media:ro voxint-whisper:dev
curl -s localhost:18022/healthz          # {"status":"ok",...,"model":"large-v2"}
curl -s -X POST localhost:18022/v1/transcribe -H 'Content-Type: application/json' \
  -d '{"path":"smoke.wav"}'
```

Expect: model loads from the baked cache in seconds (no multi-GB download in
the logs), a coherent `transcript`, `duration_seconds` ≈ 60, and well over
real-time throughput (reference: 60 s clip in ~1.7 s wall on a 3090).

## pyannote

No token needed. The diarization weights are vendored into the image.

```bash
docker run -d --name smoke-pyannote --gpus all -p 127.0.0.1:18024:8024 \
  -v /tmp/voxint-smoke/media:/data/media:ro voxint-pyannote:dev
curl -s localhost:18024/healthz
curl -s -X POST localhost:18024/v1/diarize -H 'Content-Type: application/json' \
  -d '{"path":"smoke.wav","min_speakers":1,"max_speakers":5}'
```

Expect: startup log line `Applied clustering hyperparameters` (confirms the
pyannote-3.1 stack accepted the overrides; its absence means the pinned
stack drifted), plausible `turns` for your clip, and `overlap_seconds` on
each turn.

## titanet

```bash
docker run -d --name smoke-titanet --gpus all -p 127.0.0.1:18021:8021 \
  -v /tmp/voxint-smoke/media:/data/media:ro voxint-titanet:dev
curl -s localhost:18021/healthz
# Feed two long windows from different speakers + one sub-second window:
curl -s -X POST localhost:18021/v1/embed -H 'Content-Type: application/json' \
  -d '{"path":"smoke.wav","windows":[{"start_seconds":11.2,"end_seconds":19.4},{"start_seconds":27.0,"end_seconds":40.0},{"start_seconds":19.4,"end_seconds":19.9}]}'
```

Expect, in request order: two entries with 192-dim embeddings whose L2 norm
is 1.0, and one `{"embedding":null,"skip_reason":"too_short","snr_db":null}`.
Cosine similarity between the two different-speaker vectors should be low
(reference: ~0.2); same-speaker windows should land noticeably higher.

## Error semantics (any service)

```bash
curl -s -X POST localhost:18022/v1/transcribe -H 'Content-Type: application/json' \
  -d '{"path":"/etc/passwd"}'      # 400 {"detail":{"code":"path_violation",...}}
curl -s -X POST localhost:18022/v1/transcribe -H 'Content-Type: application/json' \
  -d '{"path":"nope.wav"}'         # 404 {"detail":{"code":"file_not_found",...}}
```

Tear down with `docker rm -f smoke-whisper smoke-pyannote smoke-titanet`.

## Gotcha: docker build snapshots the context at build start

If you edit `app/` code while a build is already running, the image silently
contains the pre-edit files (this bit us on first smoke: a half-applied
async→sync refactor). When a smoke failure makes no sense against the source
you're reading, rebuild first; the pip layers are cached, so it's seconds.
