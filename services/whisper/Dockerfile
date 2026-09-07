# Voxint whisper service — faster-whisper large-v2 int8, model baked at build.
# Contract: docs/gpu-contracts.md (POST /v1/transcribe, GET /healthz).
#
# CUDA 12.8.1 base + torch 2.8.0/cu128 + CTranslate2 4.8.2: one image serves
# sm_86 (RTX 3060/3090) through sm_120 (RTX 5090). The CT2 wheel dlopens
# libcublas.so.12 (provided by the 12.8.1 runtime image) and carries sm_86
# PTX only, which the driver JIT-compiles for sm_120 — CUDA_CACHE_PATH keeps
# that JIT result across restarts. CT2 4.8.x links no cuDNN.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt

# Bake the model at build time so the image needs no network at runtime.
# large-v2 is a deliberate policy pin — see docs/gpu-contracts.md. The HF
# revision is sha-pinned so the bake cannot float with upstream main.
ARG WHISPER_HF_REVISION=f0fe81560cb8b68660e564f55dd99207059c092e
RUN python3.10 -c "from faster_whisper import WhisperModel; WhisperModel('Systran/faster-whisper-large-v2', device='cpu', download_root='/opt/whisper_models', revision='${WHISPER_HF_REVISION}')"

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV OMP_NUM_THREADS=4
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
# Persist the driver's PTX->SASS JIT cache so sm_120 pays the JIT cost only
# on the first run after a build, not on every container restart.
ENV CUDA_CACHE_PATH=/app/.cache/nv

RUN apt-get update && apt-get install -y \
    python3.10 libgomp1 libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r voxint && useradd -r -g voxint -s /bin/bash voxint

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/whisper_models /app/.cache/whisper
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# Pre-create the alternate-model cache mountpoint so a fresh named volume mounted
# there inherits voxint ownership; otherwise Docker creates it root-owned and the
# non-root service cannot write an alternate model's download (WHISPER_DOWNLOAD_ROOT).
RUN mkdir -p /app/.cache/huggingface /app/model-cache/whisper /app/.cache/nv && chown -R voxint:voxint /app
COPY --chown=voxint:voxint app/ /app/app/

USER voxint
ENV HOME=/app

ENV MEDIA_ROOT=/data/media
ENV WHISPER_MODEL=large-v2
ENV WHISPER_DOWNLOAD_ROOT=/app/.cache/whisper
# Weights are baked — the service must never phone home at startup (a hub
# revision check would stall/fail on air-gapped hosts). WHISPER_REVISION pins
# the load to the baked snapshot so offline resolution needs no ref lookup.
ARG WHISPER_HF_REVISION=f0fe81560cb8b68660e564f55dd99207059c092e
ENV WHISPER_REVISION=${WHISPER_HF_REVISION}
# Baked-snapshot reference for the startup resolver: an alternate model whose
# WHISPER_REVISION still equals this baked large-v2 SHA is a forgotten-revision
# mistake and fails closed (app/whisper_startup.py). Never operator-facing.
ENV WHISPER_BAKED_REVISION=${WHISPER_HF_REVISION}
ENV HF_HUB_OFFLINE=1
ENV DEVICE=cuda
ENV COMPUTE_TYPE=int8
ENV BATCH_SIZE=16
ENV PORT=8022

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

EXPOSE 8022

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
